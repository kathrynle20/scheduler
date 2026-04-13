from __future__ import annotations

import pytest

from framework.job import Job
from monitoring.simulator import SimulatedMonitor
from schedulers.baseline import BaselineScheduler
from schedulers.hybrid import HybridScheduler


def _mk_monitor() -> SimulatedMonitor:
    return SimulatedMonitor(
        gpu_ids=[0, 1, 2, 3],
        neighbors={0: [1], 1: [0, 2], 2: [1, 3], 3: [2]},
    )


def test_baseline_round_robin():
    monitor = _mk_monitor()
    sched = BaselineScheduler()
    states = monitor.sample()
    job = Job(id="j0", workload_type="ptq", mem_required_mb=1024, arrival_time=0.0)
    ids = [sched.place(job, states) for _ in range(8)]
    # Should cycle 0,1,2,3,0,1,2,3
    assert ids == [0, 1, 2, 3, 0, 1, 2, 3]


def test_baseline_returns_none_when_no_gpus():
    sched = BaselineScheduler()
    job = Job(id="j0", workload_type="ptq", mem_required_mb=1024, arrival_time=0.0)
    assert sched.place(job, []) is None


@pytest.mark.xfail(reason="HybridScheduler is stubbed; workstream 1 to implement", strict=True)
def test_hybrid_prefers_cool_idle_gpu():
    monitor = _mk_monitor()
    # Make GPU 0 hot and busy; GPU 3 cool and idle.
    monitor.set_job_load(0, util_pct=95, mem_used_mb=1000)
    for _ in range(200):
        monitor.sample()  # let GPU 0 heat up
    job = Job(id="j0", workload_type="ptq", mem_required_mb=1024, arrival_time=0.0)
    gpu_id = HybridScheduler(w_utilization=0.5, w_thermal=0.5).place(job, monitor.sample())
    assert gpu_id == 3
