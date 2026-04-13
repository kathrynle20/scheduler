from __future__ import annotations

from typing import Iterable


def latency_percentiles(
    latencies_s: Iterable[float],
    percentiles: tuple[float, ...] = (50.0, 95.0, 99.0),
) -> dict[float, float]:
    """Return {percentile: latency_s}.

    Used on PTQ inference results. Expected behavior: empty input -> empty
    dict; otherwise numpy.percentile with linear interpolation.
    """
    # TODO(ws2)
    raise NotImplementedError("latency_percentiles -- owned by workstream 2")


def throughput(num_samples: int, duration_s: float) -> float:
    """Samples/sec. Used on training results."""
    # TODO(ws2)
    raise NotImplementedError("throughput -- owned by workstream 2")


def temp_stability(temps_c: Iterable[float]) -> dict[str, float]:
    """Return {"mean": ..., "std": ..., "max": ..., "p99": ...} over a series.

    A smaller std / lower max indicates the scheduler is keeping thermals
    in check. Computed per-GPU by callers.
    """
    # TODO(ws2)
    raise NotImplementedError("temp_stability -- owned by workstream 2")
