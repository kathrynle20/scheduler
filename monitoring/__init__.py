from monitoring.gpu_monitor import GpuMonitor, NvmlMonitor
from monitoring.simulator import SimulatedMonitor

__all__ = ["GpuMonitor", "NvmlMonitor", "SimulatedMonitor"]


def build(backend: str, config: dict) -> GpuMonitor:
    gpu_ids = config["gpus"]["ids"]
    neighbors = {int(k): list(v) for k, v in config["gpus"]["neighbors"].items()}
    if backend == "nvml":
        return NvmlMonitor(gpu_ids=gpu_ids, neighbors=neighbors)
    if backend == "simulated":
        sim = config["monitor"].get("simulated", {})
        return SimulatedMonitor(gpu_ids=gpu_ids, neighbors=neighbors, **sim)
    raise ValueError(f"unknown monitor backend: {backend}")
