from __future__ import annotations

import logging

from framework.gpu import GpuState
from framework.job import Job
from schedulers.base import Scheduler

log = logging.getLogger(__name__)


class BaselineScheduler(Scheduler):
    """Round-robin: cycle through available GPUs regardless of memory or load.

    This is the naive reference the project measures the hybrid scheduler
    against.
    """

    def __init__(self) -> None:
        self._last_index = -1

    def place(self, job: Job, gpu_states: list[GpuState]) -> int | None:
        if not gpu_states:
            return None

        gpus = sorted(gpu_states, key=lambda g: g.id)
        n = len(gpus)

        # Pick the next GPU in round-robin order.
        self._last_index = (self._last_index + 1) % n
        chosen = gpus[self._last_index]

        log.debug(
            "baseline: job %s -> round-robin gpu %d (index %d/%d)",
            job.id,
            chosen.id,
            self._last_index,
            n,
        )
        return chosen.id
