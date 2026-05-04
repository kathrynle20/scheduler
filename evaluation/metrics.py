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


def utilization_balance(util_by_gpu: dict[int, list[float]]) -> dict:
    """Quantify how evenly load is spread across GPUs.

    Args:
        util_by_gpu: per-GPU utilization time series (0-100).

    Returns:
        {
          "per_gpu_mean": {gpu_id: mean_util, ...},
          "cluster_mean": float,
          "cluster_std":  float,    # std of per-GPU means -- lower = more balanced
          "max_imbalance": float,   # max(per_gpu_mean) - min(per_gpu_mean)
        }
    """
    per_gpu_mean: dict[int, float] = {}
    for gpu_id, series in util_by_gpu.items():
        arr = np.asarray(list(series), dtype=float)
        per_gpu_mean[gpu_id] = float(np.mean(arr)) if arr.size else 0.0

    if not per_gpu_mean:
        return {
            "per_gpu_mean": {},
            "cluster_mean": 0.0,
            "cluster_std": 0.0,
            "max_imbalance": 0.0,
        }

    means = np.asarray(list(per_gpu_mean.values()), dtype=float)
    return {
        "per_gpu_mean": per_gpu_mean,
        "cluster_mean": float(np.mean(means)),
        "cluster_std": float(np.std(means)),
        "max_imbalance": float(np.max(means) - np.min(means)),
    }
