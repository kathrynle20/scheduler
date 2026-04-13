from __future__ import annotations

import csv
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.collector import MetricsCollector
    from monitoring.gpu_monitor import GpuMonitor
    from schedulers.base import Scheduler
    from workloads.base import Workload, WorkloadResult

    from framework.job import Job


@dataclass
class RunArtifacts:
    run_dir: Path
    timeseries_path: Path
    results_path: Path


class ExperimentRunner:
    """Drive a single experiment: feed jobs to the scheduler, execute them,
    and record telemetry + per-job results.

    Minimal event loop -- sufficient for simulated runs and for real runs
    where workloads block the calling process. A future extension can swap
    the sequential `run()` for a thread/process pool.
    """

    def __init__(
        self,
        scheduler: "Scheduler",
        monitor: "GpuMonitor",
        collector: "MetricsCollector",
        output_root: Path,
    ) -> None:
        self.scheduler = scheduler
        self.monitor = monitor
        self.collector = collector
        self.output_root = Path(output_root)

    def run(
        self,
        jobs: list["Job"],
        workload_factory,
    ) -> RunArtifacts:
        """Execute `jobs` in arrival order.

        `workload_factory(job) -> Workload` constructs the executable for a
        given job (so the runner stays agnostic of workload implementations).
        """
        run_dir = self.output_root / f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        timeseries_path = run_dir / "timeseries.csv"
        results_path = run_dir / "results.csv"

        self.collector.start(timeseries_path, self.monitor)
        try:
            with results_path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    ["job_id", "workload_type", "gpu_id", "start_ts", "end_ts", "extra"]
                )
                for job in sorted(jobs, key=lambda j: j.arrival_time):
                    self._wait_until(job.arrival_time)
                    gpu_states = self.monitor.sample()
                    gpu_id = self.scheduler.place(job, gpu_states)
                    if gpu_id is None:
                        writer.writerow([job.id, job.workload_type, "", "", "", "deferred"])
                        continue
                    workload: "Workload" = workload_factory(job)
                    result: "WorkloadResult" = workload.run(gpu_id)
                    writer.writerow(
                        [
                            job.id,
                            job.workload_type,
                            gpu_id,
                            result.start_ts,
                            result.end_ts,
                            result.extra_json(),
                        ]
                    )
        finally:
            self.collector.stop()

        return RunArtifacts(run_dir=run_dir, timeseries_path=timeseries_path, results_path=results_path)

    @staticmethod
    def _wait_until(target_ts: float) -> None:
        now = time.monotonic()
        if target_ts > now:
            time.sleep(target_ts - now)
