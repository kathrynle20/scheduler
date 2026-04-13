from __future__ import annotations

import random
import time

from framework.job import Job
from workloads._gpu import cuda_available, get_device
from workloads.base import Workload, WorkloadResult


class PTQInference(Workload):
    """Post-training-quantization inference: bursty, latency-sensitive.

    Real path (torch+CUDA present): run `num_requests` matmuls against a
    preallocated int8-ish tensor on `gpu_id`, record per-request latency.

    Fallback path (no CUDA): sleep per request so the scaffold still runs on
    dev machines and in CI with the simulated monitor.

    Payload keys:
      num_requests: int   -- default 50
      matrix_size: int    -- default 2048 (torch matmul side length)
    """

    def __init__(self, job: Job) -> None:
        self.job = job

    def run(self, gpu_id: int) -> WorkloadResult:
        num_requests = int(self.job.payload.get("num_requests", 50))
        if cuda_available():
            return self._run_gpu(gpu_id, num_requests)
        return self._run_stub(gpu_id, num_requests)

    def _run_gpu(self, gpu_id: int, num_requests: int) -> WorkloadResult:
        import torch

        size = int(self.job.payload.get("matrix_size", 2048))
        device = get_device(gpu_id)
        x = torch.randn(size, size, device=device, dtype=torch.float16)

        # warmup so the first request's latency reflects steady state
        torch.matmul(x, x)
        torch.cuda.synchronize(device)

        start = time.monotonic()
        latencies: list[float] = []
        for _ in range(num_requests):
            t0 = time.monotonic()
            y = torch.matmul(x, x)
            torch.cuda.synchronize(device)
            latencies.append(time.monotonic() - t0)
            del y
        end = time.monotonic()
        del x
        torch.cuda.empty_cache()

        return WorkloadResult(
            start_ts=start,
            end_ts=end,
            latencies_s=latencies,
            extra={"gpu_id": gpu_id, "backend": "torch", "matrix_size": size},
        )

    def _run_stub(self, gpu_id: int, num_requests: int) -> WorkloadResult:
        start = time.monotonic()
        rng = random.Random(hash((self.job.id, gpu_id)) & 0xFFFF_FFFF)
        latencies: list[float] = []
        for _ in range(num_requests):
            t0 = time.monotonic()
            time.sleep(rng.uniform(0.005, 0.020))
            latencies.append(time.monotonic() - t0)
        return WorkloadResult(
            start_ts=start,
            end_ts=time.monotonic(),
            latencies_s=latencies,
            extra={"gpu_id": gpu_id, "backend": "stub"},
        )
