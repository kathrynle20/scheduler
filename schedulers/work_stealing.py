from __future__ import annotations

import logging
import threading
from collections import defaultdict

from framework.gpu import GpuState
from framework.job import Job
from schedulers.base import Scheduler

log = logging.getLogger(__name__)


class WorkStealingScheduler(Scheduler):
    """Place incoming jobs into per-GPU queues with awareness of queue length
    and current utilization.

    The scheduler picks the GPU whose combined queue + utilization score is
    highest. Stealing itself is performed by ``WorkStealingRunner``; this
    scheduler exposes ``steal_threshold`` for that runner to consult and a
    ``notify_dequeued`` hook so its queue-length estimates stay in sync after
    a steal or completion.

    Scoring (higher is better, score in [0, 1]):
        util_penalty(u) = 1.0                          if u < target_util_low
                        = 1 - (u - low) / (high - low) if low <= u < high
                        = 0.0                          if u >= target_util_high

        queue_score(q)  = 1.0 / (1 + q)

        score(gpu)      = 0.5 * util_penalty(gpu.util_pct)
                        + 0.5 * queue_score(queue_lengths[gpu.id])

    ``place`` always returns a gpu_id when at least one GPU is offered (it
    never defers): work-stealing relies on continuous progress, even into a
    saturated cluster.
    """

    def __init__(
        self,
        target_util_low: float = 50.0,
        target_util_high: float = 70.0,
        steal_threshold: int = 2,
    ) -> None:
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
            scored = [
                (self._score(s, self._queue_lengths[s.id]), s.id)
                for s in gpu_states
            ]
            # Highest score wins; tie-break on lowest gpu id for determinism.
            best_score, best_id = max(scored, key=lambda x: (x[0], -x[1]))

            # Always make progress -- if every GPU scored 0 (everyone is
            # overutilized), still pick the one with the shortest queue.
            if best_score == 0.0:
                best_id = min(
                    (s.id for s in gpu_states),
                    key=lambda gid: (self._queue_lengths[gid], gid),
                )

            self._queue_lengths[best_id] += 1
            log.debug(
                "ws place: job %s -> gpu %d (score=%.3f, q=%d, util=%.1f%%)",
                job.id,
                best_id,
                best_score,
                self._queue_lengths[best_id],
                _find_util(gpu_states, best_id),
            )
            return best_id

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

    # --------------------------------------------------------------------- #
    #  Scoring                                                               #
    # --------------------------------------------------------------------- #

    def _score(self, gpu: GpuState, queue_len: int) -> float:
        return 0.5 * self._util_penalty(gpu.util_pct) + 0.5 * self._queue_score(queue_len)

    def _util_penalty(self, util_pct: float) -> float:
        if util_pct < self.target_util_low:
            return 1.0
        if util_pct >= self.target_util_high:
            return 0.0
        span = self.target_util_high - self.target_util_low
        if span <= 0:
            return 0.0
        return 1.0 - (util_pct - self.target_util_low) / span

    @staticmethod
    def _queue_score(queue_len: int) -> float:
        return 1.0 / (1.0 + max(0, queue_len))


def _find_util(states: list[GpuState], gpu_id: int) -> float:
    for s in states:
        if s.id == gpu_id:
            return s.util_pct
    return -1.0
