"""Voltage Park On-Demand API client.

Jobs run inside a VM provisioned via cloud_init. The runcmd runs the user's
script, then issues a poweroff so the server can detect completion by polling
the VM status (Running → Stopped → DELETE → Terminated).
"""

import re
from typing import Optional

import requests

VP_API = "https://cloud-api.voltagepark.com/api/v1"

# Map VP VM status → internal job status
_VM_STATUS_MAP = {
    "Running": "running",
    "Relocating": "running",
    "Stopped": "completed",           # poweroff ran → job done
    "StoppedDisassociated": "completed",
    "Terminated": "completed",        # VM already cleaned up
    "Outbid": "failed",
}


class VoltageParkCluster:
    """Manages jobs as Voltage Park instant VMs."""

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.api_key: str = cfg["api_key"]
        self.operating_system: str = cfg.get("operating_system", "Ubuntu 22.04 LTS")
        # Optional GPU model filter (e.g. "H100", "A100") — matches against preset resource keys
        self.gpu_model_filter: Optional[str] = cfg.get("gpu_model_filter")
        self.min_gpus: int = cfg.get("min_gpus", 1)
        # SSH public keys to inject into the VM
        self.ssh_keys: list[str] = cfg.get("ssh_keys", [])
        # "all" | "none" — which org SSH keys to attach
        self.org_ssh_key_mode: str = cfg.get("org_ssh_key_mode", "all")
        # Safety timeout multiplier: terminate if running > estimated_hours * timeout_multiplier
        self.timeout_multiplier: float = cfg.get("timeout_multiplier", 1.5)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get(self, path: str, **kw) -> dict:
        r = requests.get(f"{VP_API}{path}", headers=self._headers, timeout=30, **kw)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict) -> dict:
        r = requests.post(f"{VP_API}{path}", headers=self._headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str):
        r = requests.delete(f"{VP_API}{path}", headers=self._headers, timeout=30)
        r.raise_for_status()

    def _matching_presets(self) -> list[dict]:
        """Return instant-deploy presets that have capacity and match our GPU filter."""
        data = self._get("/virtual-machines/instant/locations")
        presets = []
        for loc in data.get("results", []):
            for preset in loc.get("available_presets", []):
                if preset.get("available_vms", 0) == 0:
                    continue
                if self.gpu_model_filter:
                    gpu_keys = list(preset.get("resources", {}).get("gpus", {}).keys())
                    if not any(self.gpu_model_filter.upper() in k.upper() for k in gpu_keys):
                        continue
                presets.append(preset)
        return presets

    # ------------------------------------------------------------------
    # Public interface (mirrors SLURMCluster)
    # ------------------------------------------------------------------

    def get_idle_gpu_nodes(self) -> int:
        """Return total available VMs across matching presets (used as capacity signal)."""
        return sum(p.get("available_vms", 0) for p in self._matching_presets())

    def get_partition_info(self) -> dict:
        """Return a summary keyed by 'available' for dashboard display."""
        presets = self._matching_presets()
        return {"available": sum(p.get("available_vms", 0) for p in presets)}

    def validate_script(self, script_content: str) -> tuple[bool, str]:
        """Validate a shell script (no SLURM headers required for VP jobs)."""
        lines = script_content.strip().splitlines()
        if not lines:
            return False, "Script is empty."
        if not lines[0].startswith("#!"):
            return False, "First line must be a shebang (e.g. #!/bin/bash)."
        # Warn if someone pastes a SLURM script by mistake
        has_sbatch = any(l.strip().startswith("#SBATCH") for l in lines)
        if has_sbatch:
            return (
                True,
                "Script contains #SBATCH directives — these are ignored on Voltage Park VMs. Script accepted.",
            )
        return True, "Script passed validation."

    def submit_job(self, script_content: str, job_id: int) -> Optional[str]:
        """Provision a VM, inject the script via cloud_init, return VP VM ID."""
        presets = self._matching_presets()
        if not presets:
            return None

        # Pick preset with most GPUs that meets min_gpus requirement, then most available VMs
        def _score(p):
            total_gpus = sum(
                v if isinstance(v, int) else v.get("count", 0)
                for v in p.get("resources", {}).get("gpus", {}).values()
            )
            return (total_gpus >= self.min_gpus, total_gpus, p.get("available_vms", 0))

        preset = max(presets, key=_score)

        # cloud_init: write job script, run it, poweroff so server detects completion
        cloud_init = {
            "write_files": [
                {
                    "path": f"/opt/sparcq/job_{job_id}.sh",
                    "content": script_content,
                    "permissions": "0755",
                }
            ],
            "runcmd": [
                "mkdir -p /opt/sparcq /var/log/sparcq",
                (
                    f"bash /opt/sparcq/job_{job_id}.sh"
                    f" 2>&1 | tee /var/log/sparcq/job_{job_id}.log"
                ),
                # poweroff signals the server that the job finished; server then deletes the VM
                "poweroff",
            ],
        }

        payload = {
            "config_id": preset["id"],
            "name": f"sparcq-{job_id}",
            "organization_ssh_keys": {
                "mode": self.org_ssh_key_mode,
                "ssh_key_ids": [],
            },
            "ssh_keys": self.ssh_keys,
            "cloud_init": cloud_init,
            "tags": [f"sparcq_job_{job_id}"],
        }

        result = self._post("/virtual-machines/instant", payload)
        return result.get("vm_id")

    def check_job_status(self, vm_id: str) -> str:
        """Query VP API for VM status and map to internal status."""
        try:
            vm = self._get(f"/virtual-machines/{vm_id}")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return "completed"  # VM already gone
            raise
        raw = vm.get("status", "")
        return _VM_STATUS_MAP.get(raw, "running")

    def terminate_vm(self, vm_id: str):
        """Force-terminate a VP VM (used on completion or timeout)."""
        try:
            self._delete(f"/virtual-machines/{vm_id}")
        except requests.HTTPError as exc:
            # 404 means already gone — that's fine
            if exc.response is None or exc.response.status_code != 404:
                raise

    def get_vm_details(self, vm_id: str) -> Optional[dict]:
        """Return the full VM record, or None if not found."""
        try:
            return self._get(f"/virtual-machines/{vm_id}")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise
