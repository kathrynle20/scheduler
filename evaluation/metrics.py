from __future__ import annotations

from typing import Iterable

import numpy as np


def latency_percentiles(
    latencies_s: Iterable[float],
    percentiles: tuple[float, ...] = (50.0, 95.0, 99.0),
) -> dict[float, float]:
    """Return {percentile: latency_s}. Empty input -> empty dict."""
    arr = np.asarray(list(latencies_s), dtype=float)
    if arr.size == 0:
        return {}
    return {p: float(np.percentile(arr, p)) for p in percentiles}


def throughput(num_samples: int, duration_s: float) -> float:
    """Samples/sec."""
    if duration_s <= 0:
        return 0.0
    return num_samples / duration_s


def temp_stability(temps_c: Iterable[float]) -> dict[str, float]:
    """Return {"mean", "std", "max", "p99"} over a temperature series."""
    arr = np.asarray(list(temps_c), dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "p99": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
        "p99": float(np.percentile(arr, 99)),
    }
