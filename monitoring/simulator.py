from __future__ import annotations

import threading
import time

from framework.gpu import GpuState
from monitoring.gpu_monitor import GpuMonitor


class SimulatedMonitor(GpuMonitor):
    """First-order thermal + utilization simulator.

    Per tick, for each GPU:
        util   -> driven by externally registered jobs (see `set_job_load`)
        temp_c -> temp += k_load * (util/100) - k_cool * (temp - ambient)
                         + k_neighbor * mean(neighbor_temp - temp)

    Sufficient to exercise scheduler logic deterministically without
    real hardware. Not a calibrated physical model.
    """

    def __init__(
        self,
        gpu_ids: list[int],
        neighbors: dict[int, list[int]],
        ambient_c: float = 25.0,
        k_load: float = 0.6,
        k_cool: float = 0.05,
        k_neighbor: float = 0.1,
        mem_total_mb: int = 24_000,
        tick_s: float = 0.1,
    ) -> None:
        self.gpu_ids = list(gpu_ids)
        self._neighbors = neighbors
        self.ambient_c = ambient_c
        self.k_load = k_load
        self.k_cool = k_cool
        self.k_neighbor = k_neighbor
        self.mem_total_mb = mem_total_mb
        self.tick_s = tick_s

        self._lock = threading.Lock()
        self._util: dict[int, float] = {g: 0.0 for g in self.gpu_ids}
        self._temp: dict[int, float] = {g: ambient_c for g in self.gpu_ids}
        self._mem_used: dict[int, int] = {g: 0 for g in self.gpu_ids}
        self._last_tick = time.monotonic()

    # ---- public API used by scheduler/runner ---------------------------------

    def set_job_load(self, gpu_id: int, util_pct: float, mem_used_mb: int) -> None:
        """Register the current load imposed by placed jobs on `gpu_id`."""
        with self._lock:
            self._util[gpu_id] = max(0.0, min(100.0, util_pct))
            self._mem_used[gpu_id] = mem_used_mb

    def sample(self) -> list[GpuState]:
        with self._lock:
            self._advance_locked()
            return [
                GpuState(
                    id=g,
                    util_pct=self._util[g],
                    temp_c=self._temp[g],
                    mem_used_mb=self._mem_used[g],
                    mem_total_mb=self.mem_total_mb,
                    neighbor_ids=list(self._neighbors.get(g, [])),
                )
                for g in self.gpu_ids
            ]

    def neighbors(self, gpu_id: int) -> list[int]:
        return list(self._neighbors.get(gpu_id, []))

    # ---- internals -----------------------------------------------------------

    def _advance_locked(self) -> None:
        now = time.monotonic()
        ticks = max(1, int((now - self._last_tick) / self.tick_s))
        self._last_tick = now
        for _ in range(ticks):
            new_temp: dict[int, float] = {}
            for g in self.gpu_ids:
                t = self._temp[g]
                u = self._util[g] / 100.0
                neighbor_delta = 0.0
                nbrs = self._neighbors.get(g, [])
                if nbrs:
                    neighbor_delta = sum(self._temp[n] - t for n in nbrs) / len(nbrs)
                dt = self.k_load * u - self.k_cool * (t - self.ambient_c) + self.k_neighbor * neighbor_delta
                new_temp[g] = max(self.ambient_c, t + dt)
            self._temp = new_temp
