from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

WorkloadType = Literal["ptq", "training"]


@dataclass
class Job:
    id: str
    workload_type: WorkloadType
    mem_required_mb: int
    arrival_time: float
    payload: dict[str, Any] = field(default_factory=dict)
