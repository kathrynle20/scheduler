from __future__ import annotations

from abc import ABC, abstractmethod

from framework.gpu import GpuState


class GpuMonitor(ABC):
    """Sample current GPU state. Implementations must be cheap enough to call
    at `sample_hz` (default 10 Hz) from a background thread.
    """

    @abstractmethod
    def sample(self) -> list[GpuState]: ...

    @abstractmethod
    def neighbors(self, gpu_id: int) -> list[int]: ...


class NvmlMonitor(GpuMonitor):
    """Real-hardware sampler via NVIDIA's official `nvidia-ml-py` package
    (imported as `pynvml`).

    Neighbor topology is not discoverable via NVML -- pass it in via config
    (physical layout / rack adjacency is a human-specified map).

    Note: NVML indices follow PCI bus order by default, while CUDA/torch
    indices follow whatever `CUDA_DEVICE_ORDER` says. For consistency, export
    `CUDA_DEVICE_ORDER=PCI_BUS_ID` before launching -- then a given gpu_id
    refers to the same physical device in both worlds.
    """

    def __init__(self, gpu_ids: list[int], neighbors: dict[int, list[int]]) -> None:
        import pynvml  # lazy: avoid import cost / failure on non-NVIDIA hosts

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self.gpu_ids = list(gpu_ids)
        self._neighbors = neighbors
        self._handles = {g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in self.gpu_ids}

    def sample(self) -> list[GpuState]:
        p = self._pynvml
        out: list[GpuState] = []
        for g in self.gpu_ids:
            h = self._handles[g]
            util = p.nvmlDeviceGetUtilizationRates(h)
            temp = p.nvmlDeviceGetTemperature(h, p.NVML_TEMPERATURE_GPU)
            mem = p.nvmlDeviceGetMemoryInfo(h)
            out.append(
                GpuState(
                    id=g,
                    util_pct=float(util.gpu),
                    temp_c=float(temp),
                    mem_used_mb=int(mem.used) // (1024 * 1024),
                    mem_total_mb=int(mem.total) // (1024 * 1024),
                    neighbor_ids=list(self._neighbors.get(g, [])),
                )
            )
        return out

    def neighbors(self, gpu_id: int) -> list[int]:
        return list(self._neighbors.get(gpu_id, []))

    def close(self) -> None:
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
