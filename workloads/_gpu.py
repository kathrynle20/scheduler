"""Shared torch/CUDA helpers for workload implementations."""
from __future__ import annotations


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


def get_device(gpu_id: int):
    """Pin to `gpu_id` and return a torch.device. Call once per workload run.

    Uses torch.cuda.set_device (not CUDA_VISIBLE_DEVICES) so the scheduler
    can address multiple GPUs from a single parent process. Export
    CUDA_DEVICE_ORDER=PCI_BUS_ID before launching to keep torch + NVML
    indices aligned.
    """
    import torch

    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")
