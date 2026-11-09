"""GPU memory from `nvidia-smi` (installed with the NVIDIA driver).

Only used to give clearer messages before and after loading a local model: when nvidia-smi is
missing or fails, every function returns None and nothing else changes.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass

QUERY = ["--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"]
MIB = 1024 * 1024


@dataclass(frozen=True)
class GpuMemory:
    name: str
    used_mib: int
    total_mib: int

    @property
    def free_mib(self) -> int:
        return max(0, self.total_mib - self.used_mib)


def parse_query(text: str) -> GpuMemory | None:
    """First GPU of `nvidia-smi --query-gpu=name,memory.used,memory.total
    --format=csv,noheader,nounits` (e.g. "NVIDIA GeForce RTX 3060, 1071, 12288")."""
    for line in text.splitlines():
        parts = [p.strip() for p in line.rsplit(",", 2)]
        if len(parts) != 3:
            continue
        try:
            return GpuMemory(parts[0], int(float(parts[1])), int(float(parts[2])))
        except ValueError:
            continue
    return None


def query(timeout_s: float = 10.0) -> GpuMemory | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        result = subprocess.run([exe, *QUERY], capture_output=True, text=True,
                                timeout=timeout_s, creationflags=flags)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return parse_query(result.stdout)
