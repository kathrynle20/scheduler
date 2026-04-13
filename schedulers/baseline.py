from __future__ import annotations

import logging

from framework.gpu import GpuState
from framework.job import Job
from schedulers.base import Scheduler

log = logging.getLogger(__name__)


class BaselineScheduler(Scheduler):
    """First-fit: pick the lowest-id GPU whose free memory is sufficient.

    No awareness of utilization or temperature -- this is the naive reference
    the project measures the hybrid scheduler against.
    """

    def place(self, job: Job, gpu_states: list[GpuState]) -> int | None:
        skipped: list[tuple[int, int]] = []  # (gpu_id, free_mb)
        for gpu in sorted(gpu_states, key=lambda g: g.id):
            if gpu.can_fit(job.mem_required_mb):
                if skipped:
                    log.debug(
                        "baseline: job %s needs %dMB; skipped %s; picked gpu %d (free=%dMB)",
                        job.id,
                        job.mem_required_mb,
                        ", ".join(f"gpu{gid}({free}MB free)" for gid, free in skipped),
                        gpu.id,
                        gpu.mem_free_mb,
                    )
                else:
                    log.debug(
                        "baseline: job %s needs %dMB; picked first-fit gpu %d (free=%dMB)",
                        job.id,
                        job.mem_required_mb,
                        gpu.id,
                        gpu.mem_free_mb,
                    )
                return gpu.id
            skipped.append((gpu.id, gpu.mem_free_mb))
        log.debug(
            "baseline: job %s needs %dMB; no GPU fits: %s",
            job.id,
            job.mem_required_mb,
            ", ".join(f"gpu{gid}({free}MB free)" for gid, free in skipped),
        )
        return None
