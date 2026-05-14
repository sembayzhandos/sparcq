#!/usr/bin/env python3
"""sparcq — command-line client for the SPARCq server."""

import json
import os
import sys
from pathlib import Path

import click
import requests

CONFIG_PATH = Path.home() / ".sparcq" / "config.json"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def _save_cfg(cfg: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def _server() -> str:
    return _load_cfg().get("server") or os.environ.get("GPUQUEUE_SERVER", "http://localhost:8000")


def _headers() -> dict[str, str]:
    key = _load_cfg().get("api_key") or os.environ.get("GPUQUEUE_API_KEY", "")
    if not key:
        click.echo("Error: no API key configured. Run: sparcq configure", err=True)
        sys.exit(1)
    return {"X-API-Key": key}


def _get(path: str, **kwargs):
    r = requests.get(f"{_server()}{path}", headers=_headers(), **kwargs)
    r.raise_for_status()
    return r.json()


def _post(path: str, **kwargs):
    r = requests.post(f"{_server()}{path}", headers=_headers(), **kwargs)
    r.raise_for_status()
    return r.json()


def _delete(path: str, **kwargs):
    r = requests.delete(f"{_server()}{path}", headers=_headers(), **kwargs)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Status colour helper
# ---------------------------------------------------------------------------

STATUS_COLOR = {
    "pending": "white",
    "validated": "cyan",
    "approved": "bright_green",
    "submitted": "blue",
    "running": "yellow",
    "completed": "green",
    "failed": "red",
    "rejected": "red",
    "invalid": "red",
    "withdrawn": "bright_black",
}


def _colored_status(status: str) -> str:
    color = STATUS_COLOR.get(status, "white")
    return click.style(status, fg=color)


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """SPARCq — submit and manage HPC jobs across SLURM clusters."""


# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--server", default="http://localhost:8000", prompt="Server URL")
@click.option("--api-key", prompt="API Key", hide_input=True)
def configure(server: str, api_key: str):
    """Save server URL and API key to ~/.sparcq/config.json."""
    _save_cfg({"server": server.rstrip("/"), "api_key": api_key})
    try:
        me = _get("/me")
        click.echo(f"Connected as {me['username']} ({'admin' if me['is_admin'] else 'user'}).")
    except Exception as exc:
        click.echo(f"Warning: could not verify credentials ({exc})", err=True)
    click.echo(f"Config saved to {CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("script", type=click.Path(exists=True, dir_okay=False))
@click.option("--cluster", "-c", required=True,
              help="Target cluster name as defined in server config (e.g. tacc, cheaha, alabama)")
@click.option("--hours", "-t", required=True, type=float,
              help="Estimated walltime in hours")
@click.option("--gpus", "-g", default=1, type=int, show_default=True,
              help="Number of GPUs needed (for informational tracking)")
@click.option("--description", "-d", default=None,
              help="Short description of the job")
def submit(script: str, cluster: str, hours: float, gpus: int, description):
    """Submit a SLURM batch script to the queue."""
    content = Path(script).read_text()
    job = _post("/jobs", json={
        "cluster": cluster,
        "script_content": content,
        "estimated_hours": hours,
        "gpu_count": gpus,
        "description": description,
    })
    status = _colored_status(job["status"])
    click.echo(f"Job #{job['id']} submitted — status: {status}")
    if not job["is_valid"]:
        click.echo(f"  Validation error: {job['validation_message']}", err=True)
    elif job["status"] == "validated":
        click.echo("  Waiting for admin approval before dispatch.")
    elif job["status"] == "approved":
        click.echo("  Auto-approved. Will dispatch when GPU nodes are idle.")


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

@cli.command("list")
def list_jobs():
    """List your jobs (admins see all jobs)."""
    jobs = _get("/jobs")
    if not jobs:
        click.echo("No jobs found.")
        return

    header = f"{'ID':>5}  {'Submitter':<12}  {'Cluster':<10}  {'Status':<12}  {'ETA(h)':>6}  {'GPUs':>4}  Description"
    click.echo(header)
    click.echo("-" * len(header))
    for j in jobs:
        desc = (j.get("description") or "")[:35]
        slurm = f"  [SLURM:{j['slurm_job_id']}]" if j.get("slurm_job_id") else ""
        click.echo(
            f"{j['id']:>5}  {j['submitter']:<12}  {j['cluster']:<10}  "
            f"{_colored_status(j['status']):<21}  {j['estimated_hours']:>6.1f}  "
            f"{j['gpu_count']:>4}  {desc}{slurm}"
        )


# ---------------------------------------------------------------------------
# Show
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("job_id", type=int)
def show(job_id: int):
    """Show full details for a specific job."""
    j = _get(f"/jobs/{job_id}")
    click.echo(f"Job #{j['id']}")
    click.echo(f"  Submitter   : {j['submitter']}")
    click.echo(f"  Cluster     : {j['cluster']} (partition: {j['partition']})")
    click.echo(f"  Status      : {_colored_status(j['status'])}")
    click.echo(f"  Valid       : {'yes' if j['is_valid'] else 'no'} — {j['validation_message']}")
    click.echo(f"  Approved    : {'yes' if j['admin_approved'] else 'no'}")
    click.echo(f"  Est. hours  : {j['estimated_hours']}")
    click.echo(f"  GPUs        : {j['gpu_count']}")
    click.echo(f"  Description : {j.get('description') or '—'}")
    click.echo(f"  Submitted   : {j['submitted_at']}")
    click.echo(f"  Approved at : {j.get('approved_at') or '—'}")
    click.echo(f"  Dispatched  : {j.get('dispatched_at') or '—'}")
    click.echo(f"  Completed   : {j.get('completed_at') or '—'}")
    click.echo(f"  SLURM ID    : {j.get('slurm_job_id') or '—'}")
    if j.get("notes"):
        click.echo(f"  Notes       : {j['notes']}")


# ---------------------------------------------------------------------------
# Withdraw
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("job_id", type=int)
@click.confirmation_option(prompt="Are you sure you want to withdraw this job?")
def withdraw(job_id: int):
    """Withdraw a pending or approved job from the queue."""
    result = _delete(f"/jobs/{job_id}")
    click.echo(result["message"])


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

@cli.command()
def clusters():
    """Show GPU node availability across all configured clusters."""
    data = _get("/clusters")
    for name, info in data.items():
        click.echo(f"\n  {click.style(name, bold=True)} ({info['host']} — partition: {info['partition']})")
        if info.get("error"):
            click.echo(f"    {click.style('ERROR', fg='red')}: {info['error']}")
        else:
            states = info.get("node_states", {})
            idle = info.get("idle_nodes", 0)
            click.echo(f"    Idle GPU nodes : {click.style(str(idle), fg='green' if idle > 0 else 'red')}")
            for state, count in states.items():
                if state != "idle":
                    click.echo(f"    {state:>12} : {count}")


# ---------------------------------------------------------------------------
# Admin sub-group
# ---------------------------------------------------------------------------

@cli.group()
def admin():
    """Admin-only commands (require an admin API key)."""


@admin.command("approve")
@click.argument("job_id", type=int)
def admin_approve(job_id: int):
    """Approve a validated job for dispatch."""
    job = _post(f"/admin/jobs/{job_id}/approve")
    click.echo(f"Job #{job_id} approved — status: {_colored_status(job['status'])}")


@admin.command("reject")
@click.argument("job_id", type=int)
@click.option("--reason", "-r", default="", help="Reason for rejection")
def admin_reject(job_id: int, reason: str):
    """Reject a job with an optional reason."""
    job = _post(f"/admin/jobs/{job_id}/reject", params={"reason": reason})
    click.echo(f"Job #{job_id} rejected.")


@admin.command("add-user")
@click.argument("username")
@click.option("--admin", "make_admin", is_flag=True, help="Grant admin privileges")
def admin_add_user(username: str, make_admin: bool):
    """Create a new user and print their API key."""
    result = _post("/admin/users", params={"username": username, "is_admin": make_admin})
    click.echo(f"User '{result['username']}' created.")
    click.echo(f"  API Key : {result['api_key']}")
    click.echo("  Share this key with the user — it cannot be retrieved later.")


@admin.command("list-users")
def admin_list_users():
    """List all users."""
    users = _get("/admin/users")
    click.echo(f"{'ID':>4}  {'Username':<20}  {'Role':<10}  Created")
    click.echo("-" * 55)
    for u in users:
        role = click.style("admin", fg="yellow") if u["is_admin"] else "user"
        click.echo(f"{u['id']:>4}  {u['username']:<20}  {role:<18}  {u['created_at']}")


@admin.command("rotate-key")
@click.argument("username")
def admin_rotate_key(username: str):
    """Rotate the API key for a user."""
    result = _post(f"/admin/users/{username}/rotate-key")
    click.echo(f"New API key for '{username}': {result['api_key']}")


if __name__ == "__main__":
    cli()
