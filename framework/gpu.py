from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GpuState:
    id: int
    util_pct: float
    temp_c: float
    mem_used_mb: int
    mem_total_mb: int
    neighbor_ids: list[int] = field(default_factory=list)

    @property
    def mem_free_mb(self) -> int:
        return self.mem_total_mb - self.mem_used_mb

    def can_fit(self, mem_required_mb: int) -> bool:
        return self.mem_free_mb >= mem_required_mb
