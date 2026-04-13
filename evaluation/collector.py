from __future__ import annotations

import csv
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from monitoring.gpu_monitor import GpuMonitor

log = logging.getLogger(__name__)


class MetricsCollector:
    """Background sampler: writes GPU time-series to a CSV.

    Columns: ts, gpu_id, util_pct, temp_c, mem_used_mb, mem_total_mb.
    """

    def __init__(self, sample_hz: float = 10.0) -> None:
        self.sample_hz = sample_hz
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self, path: Path, monitor: "GpuMonitor") -> None:
        if self._thread is not None:
            raise RuntimeError("collector already started")
        self._stop.clear()
        self._samples_written = 0
        log.info("collector: starting at %.1f Hz -> %s", self.sample_hz, path)
        self._thread = threading.Thread(
            target=self._loop,
            args=(path, monitor),
            name="metrics-collector",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        log.info("collector: stopped (%d samples written)", self._samples_written)

    def _loop(self, path: Path, monitor: "GpuMonitor") -> None:
        period = 1.0 / self.sample_hz
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["ts", "gpu_id", "util_pct", "temp_c", "mem_used_mb", "mem_total_mb"])
            while not self._stop.is_set():
                t = time.time()
                for g in monitor.sample():
                    writer.writerow([t, g.id, g.util_pct, g.temp_c, g.mem_used_mb, g.mem_total_mb])
                    self._samples_written += 1
                fh.flush()
                self._stop.wait(period)
