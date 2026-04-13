from __future__ import annotations

from framework.gpu import GpuState
from framework.job import Job
from schedulers.base import Scheduler


class BaselineScheduler(Scheduler):
    """First-fit: pick the lowest-id GPU whose free memory is sufficient.

    No awareness of utilization or temperature -- this is the naive reference
    the project measures the hybrid scheduler against.
    """

    def place(self, job: Job, gpu_states: list[GpuState]) -> int | None:
        for gpu in sorted(gpu_states, key=lambda g: g.id):
            if gpu.can_fit(job.mem_required_mb):
                return gpu.id
        return None
