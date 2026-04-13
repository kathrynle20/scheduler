from __future__ import annotations

import csv
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
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

# How long the scheduler loop waits before retrying when all workers are busy.
_POLL_INTERVAL_S = 0.05


@dataclass
class RunArtifacts:
    run_dir: Path
    timeseries_path: Path
    results_path: Path


class ExperimentRunner:
    """Drive a single experiment: a scheduler GPU dispatches jobs to worker
    GPUs in parallel, records telemetry and per-job results.

    GPU 0 (or whichever ``scheduler_gpu`` is set to) runs the scheduling loop.
    The remaining ``worker_gpu_ids`` execute workloads concurrently via a
    thread pool — one thread per worker GPU.
    """

    def __init__(
        self,
        scheduler: "Scheduler",
        monitor: "GpuMonitor",
        collector: "MetricsCollector",
        output_root: Path,
        worker_gpu_ids: list[int] | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.monitor = monitor
        self.collector = collector
        self.output_root = Path(output_root)
        # Default: GPUs 1-3 are workers (GPU 0 is the scheduler).
        self.worker_gpu_ids: list[int] = worker_gpu_ids or [1, 2, 3]

        # --- concurrency bookkeeping ---
        self._busy_gpus: set[int] = set()
        self._busy_lock = threading.Lock()
        self._gpu_freed = threading.Condition(self._busy_lock)

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

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
            "starting: %d jobs, scheduler=%s, monitor=%s, workers=%s",
            len(jobs),
            type(self.scheduler).__name__,
            type(self.monitor).__name__,
            self.worker_gpu_ids,
        )

        self.collector.start(timeseries_path, self.monitor)

        placed = 0
        deferred = 0
        csv_lock = threading.Lock()
        futures: list[Future] = []

        try:
            with (
                results_path.open("w", newline="") as fh,
                ThreadPoolExecutor(
                    max_workers=len(self.worker_gpu_ids),
                    thread_name_prefix="gpu-worker",
                ) as pool,
            ):
                writer = csv.writer(fh)
                writer.writerow(
                    ["job_id", "workload_type", "gpu_id", "start_ts", "end_ts", "extra"]
                )

                for job in sorted(jobs, key=lambda j: j.arrival_time):
                    self._wait_until(job.arrival_time)

                    gpu_id = self._schedule_with_retry(job)

                    if gpu_id is None:
                        # Only reached if _schedule_with_retry gives up
                        deferred += 1
                        log.warning("[%s] DEFERRED (no worker GPU fits)", job.id)
                        with csv_lock:
                            writer.writerow([job.id, job.workload_type, "", "", "", "deferred"])
                        continue

                    placed += 1
                    workload: "Workload" = workload_factory(job)

                    # Dispatch to the thread pool — scheduler loop continues immediately.
                    fut = pool.submit(
                        self._run_workload,
                        job, gpu_id, workload, writer, csv_lock,
                    )
                    futures.append(fut)

                # Wait for all in-flight workloads to finish.
                for fut in futures:
                    fut.result()

        finally:
            self.collector.stop()

        log.info("run complete: placed=%d deferred=%d dir=%s", placed, deferred, run_dir)
        return RunArtifacts(run_dir=run_dir, timeseries_path=timeseries_path, results_path=results_path)

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _schedule_with_retry(self, job: "Job") -> int | None:
        """Try to place *job* on a free worker GPU.

        If all workers are currently busy, block until one finishes and retry.
        Returns the chosen ``gpu_id``, or ``None`` if the scheduler itself
        defers the job (e.g. memory constraints on all free GPUs).
        """
        while True:
            gpu_states = self.monitor.sample()

            # Only show the scheduler worker GPUs that are not busy.
            with self._busy_lock:
                available = [
                    s for s in gpu_states
                    if s.id in self.worker_gpu_ids and s.id not in self._busy_gpus
                ]

            if not available:
                # All workers occupied — wait for one to finish.
                with self._gpu_freed:
                    log.debug("[%s] all workers busy, waiting…", job.id)
                    self._gpu_freed.wait(timeout=_POLL_INTERVAL_S)
                continue

            gpu_id = self.scheduler.place(job, available)

            if gpu_id is not None:
                with self._busy_lock:
                    self._busy_gpus.add(gpu_id)
                chosen = _find_state(available, gpu_id)
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
                return gpu_id

            # Scheduler declined placement even though GPUs were available
            # (e.g. none have enough memory).  Defer rather than spin.
            return None

    def _run_workload(
        self,
        job: "Job",
        gpu_id: int,
        workload: "Workload",
        writer,
        csv_lock: threading.Lock,
    ) -> None:
        """Execute a workload on *gpu_id* (called on a pool thread).

        Updates the simulated monitor load, writes the result row, and marks
        the GPU as free when done.
        """
        # Inform the monitor so simulated temps / mem track reality.
        self._set_monitor_load(gpu_id, util_pct=80.0, mem_mb=job.mem_required_mb)

        try:
            t_start = time.monotonic()
            result: "WorkloadResult" = workload.run(gpu_id)
            elapsed = time.monotonic() - t_start

            log.info(
                "[%s] done on GPU %d in %.2fs (%s)",
                job.id, gpu_id, elapsed, _fmt_result(result),
            )

            with csv_lock:
                writer.writerow([
                    job.id,
                    job.workload_type,
                    gpu_id,
                    result.start_ts,
                    result.end_ts,
                    result.extra_json(),
                ])
        finally:
            # Clear load and release the GPU for new work.
            self._set_monitor_load(gpu_id, util_pct=0.0, mem_mb=0)
            with self._gpu_freed:
                self._busy_gpus.discard(gpu_id)
                self._gpu_freed.notify_all()

    def _set_monitor_load(self, gpu_id: int, util_pct: float, mem_mb: int) -> None:
        """Update the monitor if it supports ``set_job_load`` (simulated)."""
        setter = getattr(self.monitor, "set_job_load", None)
        if callable(setter):
            setter(gpu_id, util_pct, mem_mb)

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
