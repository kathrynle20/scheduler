from __future__ import annotations

import logging
import threading
from collections import defaultdict

from framework.gpu import GpuState
from framework.job import Job
from schedulers.base import Scheduler

log = logging.getLogger(__name__)


class WorkStealingScheduler(Scheduler):
    """Strict join-shortest-queue placement for the work-stealing runner.

    The scheduler picks the GPU with the **smallest tracked queue length**
    (running + waiting jobs). NVML utilization is sampled at 10 Hz while
    PTQ matmul kernels run at ~50 ms granularity, so util readings are
    too noisy to use as a primary placement signal at the single-job
    timescale -- a low-util reading on a busy GPU can override its higher
    queue count and produce a misrouted placement that work stealing then
    has to rescue reactively.

    See ``docs/work_stealing_postmortem.md`` for the full failure analysis.

    Tie-break order:
        1. shortest queue (primary)
        2. lower util_pct  (only matters when queues are equal)
        3. lower gpu id    (deterministic fallback)

    Stealing itself is performed by ``WorkStealingRunner``; this scheduler
    exposes ``steal_threshold`` for the runner to consult and a
    ``notify_dequeued`` hook so its queue-length estimates stay in sync
    after a steal or completion.

    ``place`` always returns a gpu_id when at least one GPU is offered (it
    never defers): work stealing relies on continuous progress.
    """

    def __init__(
        self,
        target_util_low: float = 50.0,
        target_util_high: float = 70.0,
        steal_threshold: int = 2,
    ) -> None:
        # Kept for config compatibility; no longer used in placement scoring.
        # See postmortem for why util-weighted scoring was removed.
        self.target_util_low = float(target_util_low)
        self.target_util_high = float(target_util_high)
        self.steal_threshold = int(steal_threshold)

        self._queue_lengths: dict[int, int] = defaultdict(int)
        self._lock = threading.Lock()

    # --------------------------------------------------------------------- #
    #  Scheduler API                                                         #
    # --------------------------------------------------------------------- #

    def place(self, job: Job, gpu_states: list[GpuState]) -> int | None:
        if not gpu_states:
            return None

        with self._lock:
            chosen = min(
                gpu_states,
                key=lambda s: (
                    self._queue_lengths[s.id],   # primary: outstanding load
                    s.util_pct,                  # tiebreak 1: cooler GPU
                    s.id,                        # tiebreak 2: deterministic
                ),
            )
            self._queue_lengths[chosen.id] += 1
            log.debug(
                "ws place: job %s -> gpu %d (q=%d, util=%.1f%%)",
                job.id,
                chosen.id,
                self._queue_lengths[chosen.id],
                chosen.util_pct,
            )
            return chosen.id

    def notify_dequeued(self, gpu_id: int) -> None:
        """The runner calls this when a job leaves *gpu_id*'s queue --
        either because gpu_id finished it, or because a thief stole it.
        """
        with self._lock:
            if self._queue_lengths[gpu_id] > 0:
                self._queue_lengths[gpu_id] -= 1

    def get_queue_lengths(self) -> dict[int, int]:
        """Return a defensive copy of the current queue-length estimates."""
        with self._lock:
            return dict(self._queue_lengths)
