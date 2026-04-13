from __future__ import annotations

import random
import time

from framework.job import Job
from workloads.base import Workload, WorkloadResult


class PTQInference(Workload):
    """Post-training-quantization inference: bursty, latency-sensitive.

    Real implementation (TODO, workstream 3):
      - Pin the process/CUDA context to `gpu_id` (e.g. CUDA_VISIBLE_DEVICES
        in a subprocess, or `torch.cuda.set_device`).
      - Run `payload["num_requests"]` forward passes against a small
        quantized model and record per-request latency.

    The current body is a placeholder so the end-to-end scaffold runs on any
    machine. It sleeps for a random latency per request so the runner, the
    collector, and the metrics pipeline can all be exercised without GPUs.
    """

    def __init__(self, job: Job) -> None:
        self.job = job

    def run(self, gpu_id: int) -> WorkloadResult:
        num_requests = int(self.job.payload.get("num_requests", 10))
        start = time.monotonic()
        latencies: list[float] = []
        rng = random.Random(hash((self.job.id, gpu_id)) & 0xFFFF_FFFF)
        for _ in range(num_requests):
            t0 = time.monotonic()
            time.sleep(rng.uniform(0.005, 0.020))  # placeholder "inference"
            latencies.append(time.monotonic() - t0)
        return WorkloadResult(
            start_ts=start,
            end_ts=time.monotonic(),
            latencies_s=latencies,
            extra={"gpu_id": gpu_id, "stub": True},
        )
