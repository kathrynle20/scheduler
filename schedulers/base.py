from __future__ import annotations

from abc import ABC, abstractmethod

from framework.gpu import GpuState
from framework.job import Job


class Scheduler(ABC):
    """Place a job on exactly one GPU.

    Implementations may carry state (e.g. last-placed index for round-robin,
    EWMA of temperatures), but `place` must be safe to call in the runner's
    single-threaded event loop.
    """

    @abstractmethod
    def place(self, job: Job, gpu_states: list[GpuState]) -> int | None:
        """Return the GPU id to run `job` on, or `None` to defer.

        Returning `None` lets the runner record the deferral and move on; the
        runner does not currently retry deferred jobs, but future versions may.
        """
