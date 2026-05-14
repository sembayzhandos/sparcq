# SPARCq

A GPU job queue system for the SPARC Center. Group members submit compute jobs through a web UI or CLI; SPARCq queues them, validates scripts, waits for GPU resources to become available across multiple HPC clusters and cloud providers, dispatches jobs automatically, and tears down cloud VMs when billing stops.

## Supported compute targets

| Cluster | Type | Notes |
|---------|------|-------|
| TACC Frontera | SLURM | `rtx` / `rtx-dev` GPU partitions |
| UAB Cheaha | SLURM | `pascalnodes` / `amperenodes` |
| Alabama ASC | SLURM | `gpu` partition |
| Voltage Park | Cloud API | Instant VMs, auto-terminated after job |

## How it works

```
Group member submits job (Web UI or CLI)
        ↓
SPARCq validates script + stores in queue
        ↓
Admin approves (or auto-approve is enabled)
        ↓
Polling daemon checks GPU availability every N minutes
        ↓
GPU available → dispatch job (sbatch on HPC, new VM on Voltage Park)
        ↓
Track status → mark complete → auto-delete Voltage Park VM
```

## Setup

### Requirements

- Python 3.10+
- SSH key access to each HPC cluster (for SLURM targets)
- Voltage Park API key (for cloud target)

### Install

```bash
git clone https://github.com/sparc-center/sparcq.git
cd sparcq
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

Edit `config.yaml` with your cluster credentials:

```yaml
require_admin_approval: true
poll_interval_minutes: 5

clusters:
  voltagepark:
    type: voltagepark
    api_key: YOUR_API_KEY
    gpu_model_filter: H100      # optional
    timeout_multiplier: 1.5

  cheaha:
    host: cheaha.rc.uab.edu
    username: your_uab_username
    ssh_key_path: ~/.ssh/cheaha_key
    gpu_partition: pascalnodes
    remote_job_dir: /scratch/your_project/sparcq_jobs

  tacc:
    host: frontera.tacc.utexas.edu
    username: your_tacc_username
    ssh_key_path: ~/.ssh/tacc_key
    gpu_partition: rtx
    remote_job_dir: /scratch1/your_project/sparcq_jobs

  alabama:
    host: uahpc.asc.edu
    username: your_alabama_username
    ssh_key_path: ~/.ssh/alabama_key
    gpu_partition: gpu
    remote_job_dir: /scratch/your_project/sparcq_jobs
```

> **Never commit real API keys or passwords.** Use `config.local.yaml` (git-ignored) for secrets, or export them as environment variables.

### Start the server

```bash
python server.py
```

On first launch, SPARCq creates an admin user and prints its API key — save it immediately.

The server runs at `http://localhost:8000`. Open it in a browser for the web UI, or use the CLI.

## CLI usage

```bash
# One-time setup
python cli.py configure

# Submit a job
python cli.py submit train.sh --cluster cheaha --hours 4 --gpus 2 --description "ResNet run #5"

# Check GPU availability across all clusters
python cli.py clusters

# List your jobs
python cli.py list

# See full details for a job
python cli.py show 12

# Withdraw a pending job
python cli.py withdraw 12

# Admin: approve / reject a job
python cli.py admin approve 12
python cli.py admin reject 12 --reason "Requested partition unavailable"

# Admin: add a user
python cli.py admin add-user jane_doe
python cli.py admin add-user alice --admin
```

## Job scripts

### SLURM targets (TACC / Cheaha / Alabama)

Standard SLURM batch scripts. Must include `#SBATCH --gres=gpu:N` and `#SBATCH --time=`:

```bash
#!/bin/bash
#SBATCH --job-name=my_training
#SBATCH --partition=pascalnodes
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

module load CUDA/12.1
python train.py --epochs 50
```

### Voltage Park target

Plain shell scripts — no `#SBATCH` directives needed. The VM runs the script then shuts down automatically:

```bash
#!/bin/bash
set -e

cd /home/ubuntu
git clone https://github.com/your-org/your-project.git
cd your-project
pip install -r requirements.txt
python train.py --epochs 50
```

## Job lifecycle

| Status | Meaning |
|--------|---------|
| `pending` | Just submitted, awaiting validation |
| `validated` | Script passed checks, waiting for admin approval |
| `approved` | Ready to dispatch when GPU nodes are free |
| `submitted` | Sent to SLURM queue or Voltage Park VM created |
| `running` | Actively executing |
| `completed` | Finished successfully |
| `failed` | Script error, timeout, or preemption |
| `rejected` | Admin rejected before dispatch |
| `withdrawn` | User cancelled before dispatch |

## REST API

The server exposes a full REST API at `http://localhost:8000`. Interactive docs at `/docs`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/jobs` | GET | List jobs (own or all for admin) |
| `/jobs` | POST | Submit a job |
| `/jobs/{id}` | GET | Job details |
| `/jobs/{id}` | DELETE | Withdraw a job |
| `/clusters` | GET | GPU availability across clusters |
| `/admin/jobs/{id}/approve` | POST | Approve a job |
| `/admin/jobs/{id}/reject` | POST | Reject a job |
| `/admin/users` | POST | Create a user |
| `/admin/users` | GET | List users |
| `/admin/users/{username}/rotate-key` | POST | Rotate API key |
| `/me` | GET | Current user info |

All endpoints require `X-API-Key: <your_key>` header.

## Architecture

```
sparcq/
├── server.py        # FastAPI REST API + APScheduler polling daemon
├── slurm.py         # SSH/Paramiko interface to SLURM clusters
├── voltagepark.py   # Voltage Park cloud API client
├── models.py        # SQLAlchemy schema (SQLite)
├── cli.py           # Click CLI client
├── config.yaml      # Cluster configuration
└── templates/
    └── index.html   # Web UI (vanilla JS, no build step)
```

The polling daemon runs two background jobs:
- **Dispatch loop** (every `poll_interval_minutes`): checks idle GPU nodes on each cluster and submits approved jobs
- **Status loop** (every `status_check_interval_minutes`): updates job status via `sacct` (SLURM) or VP API; terminates Voltage Park VMs when jobs finish

## Security notes

- API keys are 256-bit random tokens generated by the server
- Rotate keys with `sparcq admin rotate-key <username>` 
- Set `require_admin_approval: true` (default) to prevent unapproved scripts from running on HPC allocations
- Run the server behind a reverse proxy (nginx) with HTTPS for remote access
