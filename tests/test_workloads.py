from __future__ import annotations

from framework.job import Job
from workloads.ptq_inference import PTQInference
from workloads.training import Training


def test_ptq_records_latencies():
    """Workstream 3 should expand this once real inference is wired up.
    For now we only assert the WorkloadResult shape the runner/metrics expect.
    """
    job = Job(id="p0", workload_type="ptq", mem_required_mb=2048,
              arrival_time=0.0, payload={"num_requests": 5})
    result = PTQInference(job).run(gpu_id=0)
    assert len(result.latencies_s) == 5
    assert result.end_ts >= result.start_ts


def test_training_records_throughput():
    """Workstream 3: replace with real training + accurate throughput checks."""
    job = Job(id="t0", workload_type="training", mem_required_mb=16_384,
              arrival_time=0.0, payload={"duration_s": 0.05})
    result = Training(job).run(gpu_id=0)
    assert result.throughput_samples_per_s is not None
    assert result.throughput_samples_per_s > 0
