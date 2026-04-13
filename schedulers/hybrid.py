from __future__ import annotations

from framework.gpu import GpuState
from framework.job import Job
from schedulers.base import Scheduler


class HybridScheduler(Scheduler):
    """Weighted utilization + thermal scoring.

    Score per candidate GPU (higher = better):
        score = w_utilization * (1 - util_pct / 100)
              + w_thermal    * thermal_score(temp_c)
              - neighbor_temp_weight * mean_neighbor_hotness

    Where `thermal_score` is 1.0 below `temp_soft_limit_c`, falls linearly to 0
    at `temp_hard_limit_c`, and is -inf above it (hard-exclude).

    The returned gpu_id is the argmax of `score` among GPUs that satisfy the
    memory requirement. Ties broken by lowest GPU id for determinism.
    """

    def __init__(
        self,
        w_utilization: float = 0.5,
        w_thermal: float = 0.5,
        temp_soft_limit_c: float = 75.0,
        temp_hard_limit_c: float = 85.0,
        neighbor_temp_weight: float = 0.0,
    ) -> None:
        self.w_utilization = w_utilization
        self.w_thermal = w_thermal
        self.temp_soft_limit_c = temp_soft_limit_c
        self.temp_hard_limit_c = temp_hard_limit_c
        self.neighbor_temp_weight = neighbor_temp_weight

    def place(self, job: Job, gpu_states: list[GpuState]) -> int | None:
        # TODO(ws1): implement the scoring described in the class docstring.
        # Tests in tests/test_schedulers.py will exercise this once the
        # SimulatedMonitor surfaces realistic util/temp values.
        raise NotImplementedError("HybridScheduler.place -- owned by workstream 1")
