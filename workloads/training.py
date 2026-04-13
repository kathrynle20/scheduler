from __future__ import annotations

import time

from framework.job import Job
from workloads.base import Workload, WorkloadResult


class Training(Workload):
    """Training job: throughput-oriented, sustains high compute + memory.

    Real implementation (TODO, workstream 3):
      - Pin to `gpu_id`.
      - Run a small training loop (e.g. ResNet/MLP on synthetic data) for
        `payload["duration_s"]` seconds.
      - Record total samples processed and compute samples/sec.

    Placeholder body below -- sleeps for the requested duration and reports
    a fake steady throughput so the scaffold is exerciseable without GPUs.
    """

    def __init__(self, job: Job) -> None:
        self.job = job

    def run(self, gpu_id: int) -> WorkloadResult:
        duration = float(self.job.payload.get("duration_s", 1.0))
        start = time.monotonic()
        time.sleep(duration)
        end = time.monotonic()
        fake_samples = int(100 * duration)
        return WorkloadResult(
            start_ts=start,
            end_ts=end,
            throughput_samples_per_s=fake_samples / max(end - start, 1e-9),
            extra={"gpu_id": gpu_id, "stub": True, "num_samples": fake_samples},
        )
