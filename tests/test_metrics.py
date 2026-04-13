from __future__ import annotations

import pytest

from evaluation.metrics import latency_percentiles, temp_stability, throughput


def test_latency_percentiles_basic():
    out = latency_percentiles([0.01, 0.02, 0.03, 0.04, 0.05], percentiles=(50.0, 99.0))
    assert out[50.0] == pytest.approx(0.03, rel=1e-3)


def test_throughput_basic():
    assert throughput(num_samples=1000, duration_s=10.0) == pytest.approx(100.0)


def test_temp_stability_keys():
    out = temp_stability([60.0, 62.0, 65.0, 70.0])
    assert set(out.keys()) >= {"mean", "std", "max"}
