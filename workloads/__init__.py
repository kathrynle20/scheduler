from workloads.base import Workload, WorkloadResult
from workloads.ptq_inference import PTQInference
from workloads.training import Training

__all__ = ["Workload", "WorkloadResult", "PTQInference", "Training"]


def build(job) -> Workload:
    """Factory: construct the right Workload for a Job."""
    if job.workload_type == "ptq":
        return PTQInference(job)
    if job.workload_type == "training":
        return Training(job)
    raise ValueError(f"unknown workload type: {job.workload_type}")
