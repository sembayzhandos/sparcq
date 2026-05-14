"""SPARCq — FastAPI server + background polling daemon."""

import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, HTTPException, Header, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from models import Base, Job, User
from slurm import SLURMCluster
from voltagepark import VoltageParkCluster

# ---------------------------------------------------------------------------
# Config + DB
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base (overlay wins for non-dict values)."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


with open("config.yaml") as _f:
    CONFIG = yaml.safe_load(_f)

# Overlay secrets from config.local.yaml (gitignored) if it exists
import os.path
if os.path.exists("config.local.yaml"):
    with open("config.local.yaml") as _f:
        _local = yaml.safe_load(_f) or {}
    _deep_merge(CONFIG, _local)

engine = create_engine("sqlite:///sparcq.db", connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


def _build_cluster(name: str, cfg: dict):
    ctype = cfg.get("type", "slurm").lower()
    if ctype == "voltagepark":
        return VoltageParkCluster(name, cfg)
    return SLURMCluster(name, cfg)


CLUSTERS: dict = {
    name: _build_cluster(name, cfg) for name, cfg in CONFIG["clusters"].items()
}

REQUIRE_APPROVAL: bool = CONFIG.get("require_admin_approval", True)

# ---------------------------------------------------------------------------
# Bootstrap: create admin if no users exist
# ---------------------------------------------------------------------------

def _bootstrap_admin():
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            key = secrets.token_urlsafe(32)
            admin = User(username="admin", api_key=key, is_admin=True)
            db.add(admin)
            db.commit()
            # Write key to file so it survives noisy terminal output
            with open("admin_key.txt", "w") as fh:
                fh.write(key + "\n")
            msg = (
                "\n" + "=" * 60 + "\n"
                + "  SPARCq first-run setup complete.\n"
                + f"  Admin API Key: {key}\n"
                + "  Key also saved to: admin_key.txt\n"
                + "=" * 60
            )
            import sys
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
    finally:
        db.close()

# ---------------------------------------------------------------------------
# FastAPI app — startup/shutdown via lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    _bootstrap_admin()
    _scheduler.start()
    yield
    _scheduler.shutdown(wait=False)


app = FastAPI(title="SPARCq", version="1.0", lifespan=_lifespan)
templates = Jinja2Templates(directory="templates")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_user(x_api_key: str = Header(...), db: Session = Depends(get_db)) -> User:
    user = db.query(User).filter(User.api_key == x_api_key).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user


def _require_admin(user: User = Depends(_get_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class JobIn(BaseModel):
    cluster: str
    script_content: str
    estimated_hours: float
    gpu_count: int = 1
    description: Optional[str] = None


class JobOut(BaseModel):
    id: int
    submitter: str
    cluster: str
    partition: str
    estimated_hours: float
    gpu_count: int
    description: Optional[str]
    script_content: Optional[str] = None
    status: str
    is_valid: bool
    validation_message: Optional[str]
    admin_approved: bool
    slurm_job_id: Optional[str]
    notes: Optional[str]
    submitted_at: datetime
    approved_at: Optional[datetime]
    dispatched_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def web_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------------------------
# Job endpoints
# ---------------------------------------------------------------------------

@app.post("/jobs", response_model=JobOut, status_code=201)
def submit_job(body: JobIn, user: User = Depends(_get_user), db: Session = Depends(get_db)):
    if body.cluster not in CLUSTERS:
        raise HTTPException(400, f"Unknown cluster '{body.cluster}'. Valid: {list(CLUSTERS)}")

    cluster = CLUSTERS[body.cluster]
    is_valid, msg = cluster.validate_script(body.script_content)

    if is_valid and not REQUIRE_APPROVAL:
        initial_status = "approved"
        admin_approved = True
    elif is_valid:
        initial_status = "validated"
        admin_approved = False
    else:
        initial_status = "invalid"
        admin_approved = False

    job = Job(
        submitter=user.username,
        cluster=body.cluster,
        partition=getattr(cluster, "gpu_partition", "instant-vm"),
        script_content=body.script_content,
        estimated_hours=body.estimated_hours,
        gpu_count=body.gpu_count,
        description=body.description,
        status=initial_status,
        is_valid=is_valid,
        validation_message=msg,
        admin_approved=admin_approved,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@app.get("/jobs", response_model=list[JobOut])
def list_jobs(user: User = Depends(_get_user), db: Session = Depends(get_db)):
    q = db.query(Job)
    if not user.is_admin:
        q = q.filter(Job.submitter == user.username)
    return q.order_by(Job.submitted_at.desc()).all()


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, user: User = Depends(_get_user), db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if not user.is_admin and job.submitter != user.username:
        raise HTTPException(403, "Access denied")
    return job


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: int, user: User = Depends(_get_user), db: Session = Depends(get_db)):
    """Fetch stdout/stderr + sacct accounting for a dispatched job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if not user.is_admin and job.submitter != user.username:
        raise HTTPException(403, "Access denied")
    if not job.slurm_job_id:
        raise HTTPException(400, "Job hasn't been dispatched yet — no logs available")

    cluster = CLUSTERS.get(job.cluster)
    if not cluster:
        raise HTTPException(500, f"Cluster '{job.cluster}' is not configured")
    if isinstance(cluster, VoltageParkCluster):
        raise HTTPException(400, "Log fetch is not implemented for Voltage Park yet")

    try:
        return cluster.fetch_logs(job.slurm_job_id, job.script_content)
    except Exception as exc:
        raise HTTPException(500, f"Failed to fetch logs: {exc}")


@app.delete("/jobs/{job_id}")
def withdraw_job(job_id: int, user: User = Depends(_get_user), db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if not user.is_admin and job.submitter != user.username:
        raise HTTPException(403, "Cannot withdraw another user's job")
    if job.status in ("running", "completed", "failed"):
        raise HTTPException(400, f"Cannot withdraw job with status '{job.status}'")
    if job.status == "withdrawn":
        raise HTTPException(400, "Job already withdrawn")
    job.status = "withdrawn"
    db.commit()
    return {"message": f"Job #{job_id} withdrawn"}


# ---------------------------------------------------------------------------
# Cluster status
# ---------------------------------------------------------------------------

@app.get("/clusters")
def cluster_status(user: User = Depends(_get_user)):
    result = {}
    for name, cluster in CLUSTERS.items():
        is_vp = isinstance(cluster, VoltageParkCluster)
        try:
            info = cluster.get_partition_info()
            idle = info.get("available", 0) if is_vp else info.get("idle", 0)
            result[name] = {
                "type": "voltagepark" if is_vp else "slurm",
                "partition": cluster.gpu_partition if not is_vp else "instant-vm",
                "host": "cloud-api.voltagepark.com" if is_vp else cluster.host,
                "node_states": info,
                "idle_nodes": idle,
                "error": None,
            }
        except Exception as exc:
            result[name] = {
                "type": "voltagepark" if is_vp else "slurm",
                "partition": "instant-vm" if is_vp else getattr(cluster, "gpu_partition", "?"),
                "host": "cloud-api.voltagepark.com" if is_vp else getattr(cluster, "host", "?"),
                "node_states": {},
                "idle_nodes": 0,
                "error": str(exc),
            }
    return result


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

@app.post("/admin/jobs/{job_id}/approve", response_model=JobOut)
def approve_job(job_id: int, admin: User = Depends(_require_admin), db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.is_valid:
        raise HTTPException(400, "Cannot approve an invalid job")
    if job.status not in ("validated", "pending"):
        raise HTTPException(400, f"Job is already '{job.status}'")
    job.admin_approved = True
    job.approved_at = datetime.utcnow()
    job.status = "approved"
    db.commit()
    db.refresh(job)
    return job


@app.post("/admin/jobs/{job_id}/reject", response_model=JobOut)
def reject_job(
    job_id: int,
    reason: str = "",
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_db),
):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status in ("running", "completed", "failed", "withdrawn"):
        raise HTTPException(400, f"Cannot reject job with status '{job.status}'")
    job.status = "rejected"
    job.notes = reason
    db.commit()
    db.refresh(job)
    return job


@app.post("/admin/users", status_code=201)
def create_user(
    username: str,
    is_admin: bool = False,
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_db),
):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(400, f"User '{username}' already exists")
    api_key = secrets.token_urlsafe(32)
    user = User(username=username, api_key=api_key, is_admin=is_admin)
    db.add(user)
    db.commit()
    return {"username": username, "api_key": api_key, "is_admin": is_admin}


@app.get("/admin/users")
def list_users(admin: User = Depends(_require_admin), db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"id": u.id, "username": u.username, "is_admin": u.is_admin, "created_at": u.created_at} for u in users]


@app.post("/admin/users/{username}/rotate-key")
def rotate_api_key(username: str, admin: User = Depends(_require_admin), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(404, "User not found")
    user.api_key = secrets.token_urlsafe(32)
    db.commit()
    return {"username": username, "api_key": user.api_key}


@app.get("/me")
def whoami(user: User = Depends(_get_user)):
    return {"username": user.username, "is_admin": user.is_admin}


# ---------------------------------------------------------------------------
# Background polling daemon
# ---------------------------------------------------------------------------

def _poll_and_dispatch():
    """Check each cluster for idle GPU nodes; dispatch approved jobs."""
    db = SessionLocal()
    try:
        for cluster_name, cluster in CLUSTERS.items():
            try:
                idle = cluster.get_idle_gpu_nodes()
                if idle == 0:
                    continue

                is_vp = isinstance(cluster, VoltageParkCluster)

                # VP: each job gets its own VM — dispatch all approved jobs up to idle capacity
                # SLURM: dispatch up to idle_nodes jobs (one per node)
                pending = (
                    db.query(Job)
                    .filter(Job.cluster == cluster_name, Job.status == "approved")
                    .order_by(Job.submitted_at)
                    .limit(idle)
                    .all()
                )

                for job in pending:
                    try:
                        remote_id = cluster.submit_job(job.script_content, job.id)
                        if remote_id:
                            job.slurm_job_id = remote_id
                            job.status = "submitted"
                            job.dispatched_at = datetime.utcnow()
                            label = "VM" if is_vp else "SLURM"
                            print(f"[dispatch] Job #{job.id} → {cluster_name} {label}:{remote_id}", flush=True)
                        else:
                            print(f"[dispatch] Job #{job.id}: no ID returned from {cluster_name}", flush=True)
                    except Exception as exc:
                        print(f"[dispatch] Failed to submit job #{job.id}: {exc}", flush=True)

                db.commit()
            except Exception as exc:
                print(f"[poll] Error checking {cluster_name}: {exc}", flush=True)
    finally:
        db.close()


def _check_running_jobs():
    """Poll job status; for VP jobs also enforce timeout and clean up VMs on completion."""
    from datetime import timedelta

    db = SessionLocal()
    try:
        active = (
            db.query(Job)
            .filter(Job.status.in_(["submitted", "running"]))
            .all()
        )
        for job in active:
            if not job.slurm_job_id:
                continue
            cluster = CLUSTERS.get(job.cluster)
            if not cluster:
                continue

            is_vp = isinstance(cluster, VoltageParkCluster)

            try:
                # Enforce timeout for VP jobs (SLURM enforces its own --time limit)
                if is_vp and job.dispatched_at:
                    multiplier = getattr(cluster, "timeout_multiplier", 1.5)
                    deadline = job.dispatched_at + timedelta(hours=job.estimated_hours * multiplier)
                    if datetime.utcnow() > deadline:
                        print(
                            f"[timeout] Job #{job.id} exceeded {job.estimated_hours * multiplier:.1f}h limit"
                            f" — terminating VM {job.slurm_job_id}",
                            flush=True,
                        )
                        cluster.terminate_vm(job.slurm_job_id)
                        job.status = "failed"
                        job.notes = (job.notes or "") + " [auto-terminated: exceeded timeout]"
                        job.completed_at = datetime.utcnow()
                        continue

                new_status = cluster.check_job_status(job.slurm_job_id)

                if new_status != job.status:
                    print(f"[status] Job #{job.id} {job.status} → {new_status}", flush=True)
                    job.status = new_status

                if new_status in ("completed", "failed"):
                    job.completed_at = datetime.utcnow()
                    # Clean up VP VM (VM is Stopped after poweroff; we delete it)
                    if is_vp:
                        try:
                            cluster.terminate_vm(job.slurm_job_id)
                            print(f"[cleanup] Deleted VP VM {job.slurm_job_id} for job #{job.id}", flush=True)
                        except Exception as exc:
                            print(f"[cleanup] Could not delete VM {job.slurm_job_id}: {exc}", flush=True)

            except Exception as exc:
                print(f"[status] Error checking job #{job.id}: {exc}", flush=True)

        db.commit()
    finally:
        db.close()


_scheduler = BackgroundScheduler()
_scheduler.add_job(
    _poll_and_dispatch,
    "interval",
    minutes=CONFIG.get("poll_interval_minutes", 5),
    id="poll_dispatch",
)
_scheduler.add_job(
    _check_running_jobs,
    "interval",
    minutes=CONFIG.get("status_check_interval_minutes", 2),
    id="check_status",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
