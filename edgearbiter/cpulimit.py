"""Emulate a slower device in software: run a measurement inside a CPU-bandwidth-limited cgroup.

`CpuLimit(0.25)` creates a cgroup whose processes get 25 % of one CPU (CFS quota over a short
period, the mechanism behind `docker run --cpus=0.25`). Together with a single inference
thread this stands in for a low-power, single-core device when no hardware is available.
Linux only, and the cgroup filesystem must be writable (root, or a delegated cgroup);
elsewhere run the benchmark in a container: `docker run --cpus=0.25 ...`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

CGROUP_ROOT = Path("/sys/fs/cgroup")


class CpuLimit:
    def __init__(self, fraction: float, period_us: int = 5000, name: Optional[str] = None):
        if not 0.0 < fraction <= 64.0:
            raise ValueError("fraction must be in (0, 64]")
        self.fraction, self.period_us = fraction, period_us
        self.quota_us = max(1000, int(round(fraction * period_us)))
        if self.quota_us != int(round(fraction * period_us)):                # kernel minimum quota is 1 ms
            self.period_us = int(round(self.quota_us / fraction))
        self.name = name or f"edgearbiter_{os.getpid()}"
        self.path: Optional[Path] = None

    @property
    def v2(self) -> bool:
        return (CGROUP_ROOT / "cgroup.controllers").exists()

    def __enter__(self) -> "CpuLimit":
        try:
            if self.v2:
                self.path = CGROUP_ROOT / self.name
                self.path.mkdir(exist_ok=True)
                (self.path / "cpu.max").write_text(f"{self.quota_us} {self.period_us}")
            else:
                self.path = CGROUP_ROOT / "cpu" / self.name
                self.path.mkdir(exist_ok=True)
                (self.path / "cpu.cfs_period_us").write_text(str(self.period_us))
                (self.path / "cpu.cfs_quota_us").write_text(str(self.quota_us))
        except OSError as exc:
            self.__exit__(None, None, None)
            raise RuntimeError(f"cannot create a CPU-limited cgroup ({exc}); run as root on Linux, or measure inside "
                               f"`docker run --cpus={self.fraction}` instead") from exc
        return self

    @property
    def procs(self) -> str:
        """File a process writes its PID to in order to join the limit."""
        return str(self.path / "cgroup.procs")

    def __exit__(self, *exc) -> None:
        if self.path is not None:
            try:
                self.path.rmdir()
            except OSError:
                pass

    def describe(self) -> str:
        return f"{self.fraction:g} CPU (CFS quota {self.quota_us} us / period {self.period_us} us)"


def join(procs_file: str) -> None:
    """Move the calling process (all its threads) into the cgroup."""
    Path(procs_file).write_text(str(os.getpid()))
