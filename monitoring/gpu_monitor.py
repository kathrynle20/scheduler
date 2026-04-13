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
    """Real-hardware sampler via `pynvml`.

    Expected implementation of `sample`:
      - `pynvml.nvmlInit()` once in __init__, `nvmlShutdown()` in `close()`.
      - For each gpu_id: `nvmlDeviceGetHandleByIndex`, then
        `nvmlDeviceGetUtilizationRates(h).gpu`,
        `nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU)`,
        `nvmlDeviceGetMemoryInfo(h)` -> used / total in MB.
      - Build and return `GpuState(...)` objects in the configured `gpu_ids`
        order.

    Neighbor topology is not discoverable via NVML -- pass it in via config.
    """

    def __init__(self, gpu_ids: list[int], neighbors: dict[int, list[int]]) -> None:
        self.gpu_ids = list(gpu_ids)
        self._neighbors = neighbors

    def sample(self) -> list[GpuState]:
        # TODO(ws2): implement real NVML sampling. Until then, NvmlMonitor
        # exists to document the interface; tests/dev should use
        # SimulatedMonitor.
        raise NotImplementedError("NvmlMonitor.sample -- owned by workstream 2")

    def neighbors(self, gpu_id: int) -> list[int]:
        return list(self._neighbors.get(gpu_id, []))

    def close(self) -> None:
        # TODO(ws2): pynvml.nvmlShutdown()
        pass
