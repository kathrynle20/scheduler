from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkloadResult:
    start_ts: float
    end_ts: float
    # PTQ: list of per-request latencies in seconds.
    # Training: empty (use throughput_samples_per_s).
    latencies_s: list[float] = field(default_factory=list)
    throughput_samples_per_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def extra_json(self) -> str:
        payload = dict(self.extra)
        if self.latencies_s:
            payload["latencies_s"] = self.latencies_s
        if self.throughput_samples_per_s is not None:
            payload["throughput_samples_per_s"] = self.throughput_samples_per_s
        return json.dumps(payload)


class Workload(ABC):
    """A single ML application run. `run` blocks until the work is done."""

    @abstractmethod
    def run(self, gpu_id: int) -> WorkloadResult: ...
