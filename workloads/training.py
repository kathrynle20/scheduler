from __future__ import annotations

import time

from framework.job import Job
from workloads._gpu import cuda_available, get_device
from workloads.base import Workload, WorkloadResult


class Training(Workload):
    """Training job: run a fixed number of training steps and time the result.

    Real path (torch+CUDA present): run a synthetic MLP training loop on
    `gpu_id` for `num_steps` iterations, measuring total wall-clock time
    to derive samples/sec.

    Fallback path (no CUDA): sleep a proportional amount and report fake
    throughput so the scaffold is exerciseable without GPUs.

    Payload keys:
      num_steps:  int  -- default 200
      batch_size: int  -- default 128
      hidden:     int  -- default 4096
    """

    def __init__(self, job: Job) -> None:
        self.job = job

    def run(self, gpu_id: int) -> WorkloadResult:
        num_steps = int(self.job.payload.get("num_steps", 200))
        if cuda_available():
            return self._run_gpu(gpu_id, num_steps)
        return self._run_stub(gpu_id, num_steps)

    def _run_gpu(self, gpu_id: int, num_steps: int) -> WorkloadResult:
        import torch
        from torch import nn

        batch = int(self.job.payload.get("batch_size", 128))
        hidden = int(self.job.payload.get("hidden", 4096))
        device = get_device(gpu_id)

        model = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        ).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        x = torch.randn(batch, hidden, device=device)
        y = torch.randn(batch, hidden, device=device)

        start = time.monotonic()
        for _ in range(num_steps):
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
        torch.cuda.synchronize(device)
        end = time.monotonic()

        samples = num_steps * batch

        del model, opt, x, y
        torch.cuda.empty_cache()

        return WorkloadResult(
            start_ts=start,
            end_ts=end,
            throughput_samples_per_s=samples / max(end - start, 1e-9),
            extra={
                "gpu_id": gpu_id,
                "backend": "torch",
                "num_steps": num_steps,
                "num_samples": samples,
                "batch_size": batch,
                "hidden": hidden,
                "duration_s": end - start,
            },
        )

    def _run_stub(self, gpu_id: int, num_steps: int) -> WorkloadResult:
        batch = int(self.job.payload.get("batch_size", 128))
        # Simulate ~5ms per step.
        fake_duration = num_steps * 0.005
        start = time.monotonic()
        time.sleep(fake_duration)
        end = time.monotonic()
        samples = num_steps * batch
        return WorkloadResult(
            start_ts=start,
            end_ts=end,
            throughput_samples_per_s=samples / max(end - start, 1e-9),
            extra={
                "gpu_id": gpu_id,
                "backend": "stub",
                "num_steps": num_steps,
                "num_samples": samples,
                "duration_s": end - start,
            },
        )
