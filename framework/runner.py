from __future__ import annotations

import csv
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.collector import MetricsCollector
    from framework.gpu import GpuState
    from monitoring.gpu_monitor import GpuMonitor
    from schedulers.base import Scheduler
    from workloads.base import Workload, WorkloadResult

    from framework.job import Job


log = logging.getLogger(__name__)


@dataclass
class RunArtifacts:
    run_dir: Path
    timeseries_path: Path
    results_path: Path


class ExperimentRunner:
    """Drive a single experiment: feed jobs to the scheduler, execute them,
    and record telemetry + per-job results.
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
        run_dir = self.output_root / f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        timeseries_path = run_dir / "timeseries.csv"
        results_path = run_dir / "results.csv"

        log.info("run dir: %s", run_dir)
        log.info(
            "starting: %d jobs, scheduler=%s, monitor=%s",
            len(jobs),
            type(self.scheduler).__name__,
            type(self.monitor).__name__,
        )

        self.collector.start(timeseries_path, self.monitor)
        placed = 0
        deferred = 0
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
                        deferred += 1
                        log.warning(
                            "[%s] %s mem=%dMB -> DEFERRED (no GPU fits); states: %s",
                            job.id,
                            job.workload_type,
                            job.mem_required_mb,
                            _fmt_states(gpu_states),
                        )
                        writer.writerow([job.id, job.workload_type, "", "", "", "deferred"])
                        continue

                    chosen = _find_state(gpu_states, gpu_id)
                    placed += 1
                    log.info(
                        "[%s] %s mem=%dMB -> GPU %d (util=%.0f%% temp=%.1fC mem=%d/%dMB)",
                        job.id,
                        job.workload_type,
                        job.mem_required_mb,
                        gpu_id,
                        chosen.util_pct if chosen else -1,
                        chosen.temp_c if chosen else -1,
                        chosen.mem_used_mb if chosen else -1,
                        chosen.mem_total_mb if chosen else -1,
                    )

                    workload: "Workload" = workload_factory(job)
                    t_start = time.monotonic()
                    result: "WorkloadResult" = workload.run(gpu_id)
                    elapsed = time.monotonic() - t_start
                    log.info(
                        "[%s] done on GPU %d in %.2fs (%s)",
                        job.id,
                        gpu_id,
                        elapsed,
                        _fmt_result(result),
                    )

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

        log.info("run complete: placed=%d deferred=%d dir=%s", placed, deferred, run_dir)
        return RunArtifacts(run_dir=run_dir, timeseries_path=timeseries_path, results_path=results_path)

    @staticmethod
    def _wait_until(target_ts: float) -> None:
        now = time.monotonic()
        if target_ts > now:
            time.sleep(target_ts - now)


def _find_state(states: list["GpuState"], gpu_id: int) -> "GpuState | None":
    for s in states:
        if s.id == gpu_id:
            return s
    return None


def _fmt_states(states: list["GpuState"]) -> str:
    return " | ".join(
        f"gpu{s.id}: util={s.util_pct:.0f}% temp={s.temp_c:.1f}C free={s.mem_free_mb}MB"
        for s in sorted(states, key=lambda x: x.id)
    )


def _fmt_result(result: "WorkloadResult") -> str:
    if result.latencies_s:
        lats = result.latencies_s
        avg_ms = 1000 * sum(lats) / len(lats)
        return f"{len(lats)} reqs, avg={avg_ms:.1f}ms"
    if result.throughput_samples_per_s is not None:
        return f"{result.throughput_samples_per_s:.1f} samples/s"
    return "no metrics"
