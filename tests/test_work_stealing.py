from __future__ import annotations

import csv
import json
import time
from collections import deque
from pathlib import Path

import pytest

from evaluation.collector import MetricsCollector
from evaluation.metrics import utilization_balance
from framework.gpu import GpuState
from framework.job import Job
from framework.ws_runner import WorkStealingRunner
from monitoring.simulator import SimulatedMonitor
from schedulers.work_stealing import WorkStealingScheduler


def _state(gpu_id: int, util_pct: float = 0.0) -> GpuState:
    return GpuState(
        id=gpu_id,
        util_pct=util_pct,
        temp_c=25.0,
        mem_used_mb=0,
        mem_total_mb=24_000,
        neighbor_ids=[],
    )


def _job(jid: str = "j") -> Job:
    return Job(id=jid, workload_type="ptq", mem_required_mb=512, arrival_time=0.0)


# --------------------------------------------------------------------------- #
#  Scheduler placement                                                         #
# --------------------------------------------------------------------------- #


def test_work_stealing_assigns_to_shortest_queue():
    sched = WorkStealingScheduler()
    states = [_state(0), _state(1)]
    # Pre-load GPU 0 with three queued jobs so GPU 1 looks empty.
    for i in range(3):
        sched.place(_job(f"warm-{i}"), [_state(0)])

    chosen = sched.place(_job("next"), states)
    assert chosen == 1


def test_work_stealing_avoids_overutilized_gpu():
    sched = WorkStealingScheduler(target_util_low=50.0, target_util_high=70.0)
    # Equal queues (both empty), but GPU 0 is much hotter.
    states = [_state(0, util_pct=90.0), _state(1, util_pct=30.0)]
    chosen = sched.place(_job(), states)
    assert chosen == 1


def test_work_stealing_never_returns_none_when_gpus_offered():
    sched = WorkStealingScheduler()
    # Both GPUs above the high watermark -- still must pick one.
    states = [_state(0, util_pct=95.0), _state(1, util_pct=99.0)]
    chosen = sched.place(_job(), states)
    assert chosen in (0, 1)


def test_work_stealing_returns_none_when_no_gpus():
    assert WorkStealingScheduler().place(_job(), []) is None


def test_notify_dequeued_decrements_estimate():
    sched = WorkStealingScheduler()
    sched.place(_job("a"), [_state(0)])
    sched.place(_job("b"), [_state(0)])
    assert sched.get_queue_lengths()[0] == 2
    sched.notify_dequeued(0)
    assert sched.get_queue_lengths()[0] == 1


# --------------------------------------------------------------------------- #
#  Steal logic                                                                 #
# --------------------------------------------------------------------------- #


def _mk_runner(worker_ids=(1, 2), steal_threshold: int = 2) -> WorkStealingRunner:
    sched = WorkStealingScheduler(steal_threshold=steal_threshold)
    monitor = SimulatedMonitor(
        gpu_ids=[0, *worker_ids],
        neighbors={g: [] for g in [0, *worker_ids]},
    )
    collector = MetricsCollector(sample_hz=10.0)
    return WorkStealingRunner(
        scheduler=sched,
        monitor=monitor,
        collector=collector,
        output_root=Path("./runs-test"),
        worker_gpu_ids=list(worker_ids),
        steal_threshold=steal_threshold,
    )


def test_steal_triggers_at_threshold():
    runner = _mk_runner(worker_ids=(1, 2), steal_threshold=2)
    # GPU 1: 3 jobs, GPU 2: 0 jobs. Imbalance = 3 >= threshold.
    sentinel = object()
    runner._queues[1] = deque([(_job("j1"), sentinel), (_job("j2"), sentinel), (_job("j3"), sentinel)])
    runner._queues[2] = deque()

    # Thief is GPU 2.
    with runner._queue_lock:
        stolen = runner._try_steal_locked(thief_id=2)

    assert stolen is not None
    job, _ = stolen
    # Steal should come from the tail.
    assert job.id == "j3"
    assert runner._steal_count == 1
    assert len(runner._queues[1]) == 2


def test_steal_does_not_trigger_below_threshold():
    runner = _mk_runner(worker_ids=(1, 2), steal_threshold=2)
    # GPU 1: 2 jobs, GPU 2: 1 job. Imbalance = 1 < threshold.
    sentinel = object()
    runner._queues[1] = deque([(_job("j1"), sentinel), (_job("j2"), sentinel)])
    runner._queues[2] = deque([(_job("j3"), sentinel)])

    with runner._queue_lock:
        stolen = runner._try_steal_locked(thief_id=2)

    assert stolen is None
    assert runner._steal_count == 0
    assert len(runner._queues[1]) == 2
    assert len(runner._queues[2]) == 1


def test_steal_skips_when_thief_is_victim():
    runner = _mk_runner(worker_ids=(1, 2), steal_threshold=1)
    sentinel = object()
    # GPU 2 is the most loaded; GPU 2 itself can't steal from itself.
    runner._queues[1] = deque()
    runner._queues[2] = deque([(_job("j1"), sentinel), (_job("j2"), sentinel)])

    with runner._queue_lock:
        stolen = runner._try_steal_locked(thief_id=2)

    assert stolen is None


# --------------------------------------------------------------------------- #
#  End-to-end                                                                  #
# --------------------------------------------------------------------------- #


class _FakeWorkload:
    """Tiny workload that records a result without touching the GPU."""

    def __init__(self, job: Job) -> None:
        self.job = job

    def run(self, gpu_id: int):
        from workloads.base import WorkloadResult

        start = time.monotonic()
        time.sleep(0.01)
        return WorkloadResult(
            start_ts=start,
            end_ts=time.monotonic(),
            latencies_s=[0.005],
            extra={"gpu_id": gpu_id, "backend": "fake"},
        )


def test_all_jobs_complete(tmp_path):
    jobs = [
        Job(
            id=f"j{i}",
            workload_type="ptq",
            mem_required_mb=128,
            arrival_time=time.monotonic() + i * 0.01,
        )
        for i in range(10)
    ]

    sched = WorkStealingScheduler(steal_threshold=2)
    monitor = SimulatedMonitor(
        gpu_ids=[0, 1, 2],
        neighbors={0: [], 1: [], 2: []},
    )
    collector = MetricsCollector(sample_hz=20.0)
    runner = WorkStealingRunner(
        scheduler=sched,
        monitor=monitor,
        collector=collector,
        output_root=tmp_path,
        worker_gpu_ids=[1, 2],
        steal_threshold=2,
    )

    artifacts = runner.run(jobs, workload_factory=_FakeWorkload)

    # All 10 jobs should be in results.csv.
    rows = list(csv.DictReader(artifacts.results_path.open()))
    job_ids = {r["job_id"] for r in rows}
    assert job_ids == {f"j{i}" for i in range(10)}

    # metadata.json should report a non-negative steal_count.
    meta_path = artifacts.run_dir / "metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["steal_count"] >= 0
    assert meta["jobs_placed"] == 10


# --------------------------------------------------------------------------- #
#  utilization_balance metric                                                  #
# --------------------------------------------------------------------------- #


def test_utilization_balance_metric_balanced_lower_std():
    balanced = {0: [50.0, 50.0, 50.0], 1: [50.0, 50.0, 50.0]}
    skewed = {0: [10.0, 10.0, 10.0], 1: [90.0, 90.0, 90.0]}

    bal = utilization_balance(balanced)
    skw = utilization_balance(skewed)

    assert bal["cluster_std"] == pytest.approx(0.0)
    assert skw["cluster_std"] > bal["cluster_std"]
    assert skw["max_imbalance"] == pytest.approx(80.0)
    assert bal["per_gpu_mean"] == {0: 50.0, 1: 50.0}


def test_utilization_balance_handles_empty():
    out = utilization_balance({})
    assert out["per_gpu_mean"] == {}
    assert out["cluster_std"] == 0.0
