from __future__ import annotations

import time

from framework.job import Job
from workloads._gpu import cuda_available, get_device
from workloads.base import Workload, WorkloadResult


class Training(Workload):
    """Training job: throughput-oriented, sustains high compute + memory.

    Real path (torch+CUDA present): run a synthetic MLP training loop on
    `gpu_id` for `duration_s` seconds, counting forward/backward iterations
    to derive samples/sec.

    Fallback path (no CUDA): sleep for `duration_s` and report a fake
    throughput so the scaffold is exerciseable without GPUs.

    Payload keys:
      duration_s: float   -- default 60
      batch_size: int     -- default 128
      hidden: int         -- default 4096
    """

    def __init__(self, job: Job) -> None:
        self.job = job

    def run(self, gpu_id: int) -> WorkloadResult:
        duration = float(self.job.payload.get("duration_s", 60.0))
        if cuda_available():
            return self._run_gpu(gpu_id, duration)
        return self._run_stub(gpu_id, duration)

    def _run_gpu(self, gpu_id: int, duration: float) -> WorkloadResult:
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
        deadline = start + duration
        samples = 0
        while time.monotonic() < deadline:
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            samples += batch
        torch.cuda.synchronize(device)
        end = time.monotonic()

        del model, opt, x, y
        torch.cuda.empty_cache()

        return WorkloadResult(
            start_ts=start,
            end_ts=end,
            throughput_samples_per_s=samples / max(end - start, 1e-9),
            extra={
                "gpu_id": gpu_id,
                "backend": "torch",
                "num_samples": samples,
                "batch_size": batch,
                "hidden": hidden,
            },
        )

    def _run_stub(self, gpu_id: int, duration: float) -> WorkloadResult:
        start = time.monotonic()
        time.sleep(duration)
        end = time.monotonic()
        fake_samples = int(100 * duration)
        return WorkloadResult(
            start_ts=start,
            end_ts=end,
            throughput_samples_per_s=fake_samples / max(end - start, 1e-9),
            extra={"gpu_id": gpu_id, "backend": "stub", "num_samples": fake_samples},
        )
