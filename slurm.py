"""SSH interface to SLURM clusters. All operations are stateless — a new SSH
connection is opened per call so the server can run without keeping sockets alive."""

import os
import re
from typing import Optional

import paramiko


class SLURMCluster:
    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.host = cfg["host"]
        self.username = cfg["username"]
        self.ssh_key_path = os.path.expanduser(cfg.get("ssh_key_path", "~/.ssh/id_rsa"))
        self.gpu_partition = cfg["gpu_partition"]
        self.remote_job_dir = cfg.get("remote_job_dir", f"/tmp/sparcq/{name}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            self.host,
            username=self.username,
            key_filename=self.ssh_key_path,
            timeout=15,
        )
        return client

    def _run(self, client: paramiko.SSHClient, cmd: str) -> tuple[str, str]:
        _, stdout, stderr = client.exec_command(cmd, timeout=30)
        return stdout.read().decode().strip(), stderr.read().decode().strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_idle_gpu_nodes(self) -> int:
        """Return number of completely idle GPU nodes in the partition."""
        client = self._connect()
        try:
            # %D = number of nodes in that state
            out, _ = self._run(
                client,
                f"sinfo -p {self.gpu_partition} --state idle -h -o '%D' 2>/dev/null || echo 0",
            )
            total = sum(int(x) for x in out.split() if x.isdigit())
            return total
        finally:
            client.close()

    def get_partition_info(self) -> dict:
        """Return a summary of the partition: total/idle/allocated nodes."""
        client = self._connect()
        try:
            out, _ = self._run(
                client,
                f"sinfo -p {self.gpu_partition} -h -o '%D %t' 2>/dev/null",
            )
            counts: dict[str, int] = {}
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0].isdigit():
                    state = parts[1].lower()
                    counts[state] = counts.get(state, 0) + int(parts[0])
            return counts
        finally:
            client.close()

    def validate_script(self, script_content: str) -> tuple[bool, str]:
        """Local (offline) validation of a SLURM batch script."""
        lines = script_content.strip().splitlines()

        if not lines:
            return False, "Script is empty."

        if not lines[0].startswith("#!"):
            return False, "First line must be a shebang (e.g. #!/bin/bash)."

        sbatch_lines = [l for l in lines if l.strip().startswith("#SBATCH")]
        if not sbatch_lines:
            return False, "No #SBATCH directives found."

        has_gpu = any(
            re.search(r"--gres=gpu|--gpus[-=]|--gpus-per-", l) for l in sbatch_lines
        )
        if not has_gpu:
            return False, "Script must request GPU resources (#SBATCH --gres=gpu:N)."

        has_time = any(re.search(r"-t |--time", l) for l in sbatch_lines)
        if not has_time:
            return False, "Script must specify a time limit (#SBATCH --time=HH:MM:SS)."

        return True, "Script passed local validation."

    def submit_job(self, script_content: str, job_id: int) -> Optional[str]:
        """Upload script and submit via sbatch. Returns SLURM job ID or None."""
        client = self._connect()
        try:
            # Ensure remote staging directory exists
            self._run(client, f"mkdir -p {self.remote_job_dir}")

            remote_path = f"{self.remote_job_dir}/sparcq_{job_id}.sh"

            sftp = client.open_sftp()
            try:
                with sftp.file(remote_path, "w") as fh:
                    fh.write(script_content)
            finally:
                sftp.close()

            self._run(client, f"chmod 600 {remote_path}")
            out, _ = self._run(client, f"sbatch {remote_path}")

            # SLURM prints: "Submitted batch job 12345"
            match = re.search(r"Submitted batch job (\d+)", out)
            return match.group(1) if match else None
        finally:
            client.close()

    def check_job_status(self, slurm_job_id: str) -> str:
        """Query sacct and map to internal status string."""
        client = self._connect()
        try:
            # sacct returns state for the job step and its batch step; take the first
            out, _ = self._run(
                client,
                f"sacct -j {slurm_job_id} -o State -n --noheader 2>/dev/null | head -1",
            )
            raw = out.strip().upper().split()[0] if out.strip() else ""
            return _map_slurm_state(raw)
        finally:
            client.close()

    def fetch_logs(self, slurm_job_id: str, script_content: str) -> dict:
        """Fetch stdout/stderr and SLURM accounting for a completed job.

        Looks for #SBATCH --output and --error paths in the script (with %j
        substituted to the SLURM job ID). Falls back to slurm-<id>.out in
        the staging dir if not specified.
        """
        stdout_path = _parse_sbatch_path(script_content, "output") or f"{self.remote_job_dir}/slurm-{slurm_job_id}.out"
        stderr_path = _parse_sbatch_path(script_content, "error")
        stdout_path = stdout_path.replace("%j", slurm_job_id).replace("%A", slurm_job_id)
        if stderr_path:
            stderr_path = stderr_path.replace("%j", slurm_job_id).replace("%A", slurm_job_id)

        client = self._connect()
        try:
            def _read(p: str) -> str:
                if not p:
                    return ""
                out, _ = self._run(client, f"cat {p} 2>/dev/null || true")
                return out

            stdout_content = _read(stdout_path)
            stderr_content = _read(stderr_path) if stderr_path else ""

            acct_out, _ = self._run(
                client,
                f"sacct -j {slurm_job_id} -o JobID,JobName,State,Start,End,Elapsed,NodeList,AllocTRES -P --noheader 2>/dev/null",
            )

            return {
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "stdout": stdout_content,
                "stderr": stderr_content,
                "accounting": acct_out,
            }
        finally:
            client.close()


def _parse_sbatch_path(script: str, kind: str) -> Optional[str]:
    """Return the value of #SBATCH --output= or --error= if present."""
    short_flag = "-o" if kind == "output" else "-e"
    for line in script.splitlines():
        s = line.strip()
        if not s.startswith("#SBATCH"):
            continue
        m = re.search(rf"--{kind}[= ]([^\s]+)", s)
        if m:
            return m.group(1)
        m = re.search(rf"(?:^|\s){short_flag}\s+([^\s]+)", s)
        if m:
            return m.group(1)
    return None


def _map_slurm_state(raw: str) -> str:
    mapping = {
        "RUNNING": "running",
        "PENDING": "submitted",
        "COMPLETED": "completed",
        "FAILED": "failed",
        "CANCELLED": "failed",
        "TIMEOUT": "failed",
        "NODE_FAIL": "failed",
        "OUT_OF_MEMORY": "failed",
        "PREEMPTED": "failed",
    }
    # States can have suffixes like CANCELLED by 1234
    for k, v in mapping.items():
        if raw.startswith(k):
            return v
    return "submitted"  # still queued / unknown
