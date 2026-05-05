from __future__ import annotations

import csv
import json
import logging
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from framework.runner import RunArtifacts

if TYPE_CHECKING:
    from evaluation.collector import MetricsCollector
    from monitoring.gpu_monitor import GpuMonitor
    from schedulers.work_stealing import WorkStealingScheduler
    from workloads.base import Workload, WorkloadResult

    from framework.job import Job


log = logging.getLogger(__name__)

# Brief sleep when a worker has nothing to do and nothing to steal.
_IDLE_BACKOFF_S = 0.005


class WorkStealingRunner:
    """Run an experiment with per-GPU job queues and work stealing.

    Every worker GPU has its own deque. The main thread enqueues incoming
    jobs (using ``WorkStealingScheduler.place``) and never blocks on whether
    a GPU is busy. Each worker thread pops from its own queue's head; when
    the queue is empty it tries to steal from the tail of the most-loaded
    peer queue, provided the imbalance meets ``steal_threshold``.

    Stealing from the tail follows the classic Chase-Lev convention: the
    owner takes head (FIFO), thieves take tail (newest unstarted work),
    minimizing wasted setup effort.
    """

    def __init__(
        self,
        scheduler: "WorkStealingScheduler",
        monitor: "GpuMonitor",
        collector: "MetricsCollector",
        output_root: Path,
        worker_gpu_ids: list[int],
        steal_threshold: int = 2,
    ) -> None:
        self.scheduler = scheduler
        self.monitor = monitor
        self.collector = collector
        self.output_root = Path(output_root)
        self.worker_gpu_ids = list(worker_gpu_ids)
        self._steal_threshold = int(steal_threshold)

        self._queues: dict[int, deque[tuple["Job", "Workload"]]] = {
            gid: deque() for gid in self.worker_gpu_ids
        }
        self._queue_lock = threading.Lock()
        self._all_dispatched = False
        self._jobs_remaining = 0
        self._steal_count = 0

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    def run(self, jobs: list["Job"], workload_factory) -> RunArtifacts:
        run_dir = self.output_root / f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        timeseries_path = run_dir / "timeseries.csv"
        results_path = run_dir / "results.csv"
        metadata_path = run_dir / "metadata.json"

        log.info("run dir: %s", run_dir)
        log.info(
            "starting (work-stealing): %d jobs, scheduler=%s, monitor=%s, workers=%s, steal_threshold=%d",
            len(jobs),
            type(self.scheduler).__name__,
            type(self.monitor).__name__,
            self.worker_gpu_ids,
            self._steal_threshold,
        )

        self.collector.start(timeseries_path, self.monitor)

        placed = 0
        csv_lock = threading.Lock()
        worker_threads: list[threading.Thread] = []

        try:
            with results_path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    ["job_id", "workload_type", "gpu_id", "arrival_time", "start_ts", "end_ts", "extra"]
                )

                # Persistent worker thread per GPU.
                for gid in self.worker_gpu_ids:
                    t = threading.Thread(
                        target=self._worker_loop,
                        args=(gid, writer, csv_lock),
                        name=f"ws-worker-gpu{gid}",
                        daemon=False,
                    )
                    t.start()
                    worker_threads.append(t)

                # Main dispatch loop.
                for job in sorted(jobs, key=lambda j: j.arrival_time):
                    self._wait_until(job.arrival_time)

                    gpu_states = self.monitor.sample()
                    available = [s for s in gpu_states if s.id in self.worker_gpu_ids]
                    gpu_id = self.scheduler.place(job, available)

                    if gpu_id is None:
                        # Should not happen for WorkStealingScheduler, but
                        # guard against custom schedulers / edge cases.
                        log.warning("[%s] scheduler returned None -- skipping", job.id)
                        with csv_lock:
                            writer.writerow(
                                [job.id, job.workload_type, "", "", "", "deferred"]
                            )
                        continue

                    workload: "Workload" = workload_factory(job)
                    with self._queue_lock:
                        self._queues[gpu_id].append((job, workload))
                        self._jobs_remaining += 1
                        q_len = len(self._queues[gpu_id])

                    placed += 1
                    log.info(
                        "[%s] %s mem=%dMB enqueued -> GPU %d (queue=%d)",
                        job.id,
                        job.workload_type,
                        job.mem_required_mb,
                        gpu_id,
                        q_len,
                    )

                # Tell workers the input stream is closed.
                with self._queue_lock:
                    self._all_dispatched = True

                # Wait for all in-flight work to finish.
                for t in worker_threads:
                    t.join()

        finally:
            # Make sure workers don't block on a dropped main thread.
            with self._queue_lock:
                self._all_dispatched = True
            for t in worker_threads:
                if t.is_alive():
                    t.join(timeout=5.0)
            self.collector.stop()

        # Persist the steal count alongside the CSVs so analyze.py can pick it up.
        with metadata_path.open("w") as fh:
            json.dump(
                {
                    "scheduler": type(self.scheduler).__name__,
                    "worker_gpu_ids": self.worker_gpu_ids,
                    "steal_threshold": self._steal_threshold,
                    "steal_count": self._steal_count,
                    "jobs_placed": placed,
                },
                fh,
                indent=2,
            )

        log.info(
            "run complete: placed=%d steals=%d dir=%s",
            placed,
            self._steal_count,
            run_dir,
        )
        return RunArtifacts(
            run_dir=run_dir,
            timeseries_path=timeseries_path,
            results_path=results_path,
        )

    # --------------------------------------------------------------------- #
    #  Worker loop                                                           #
    # --------------------------------------------------------------------- #

    def _worker_loop(self, gpu_id: int, writer, csv_lock: threading.Lock) -> None:
        while True:
            item = self._dequeue(gpu_id)
            if item is None:
                return

            job, workload = item
            self._set_monitor_load(gpu_id, util_pct=80.0, mem_mb=job.mem_required_mb)
            try:
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

                with csv_lock:
                    writer.writerow(
                        [
                            job.id,
                            job.workload_type,
                            gpu_id,
                            job.arrival_time,
                            result.start_ts,
                            result.end_ts,
                            result.extra_json(),
                        ]
                    )
            except Exception:
                log.exception("[%s] failed on GPU %d", job.id, gpu_id)
            finally:
                self._set_monitor_load(gpu_id, util_pct=0.0, mem_mb=0)
                with self._queue_lock:
                    self._jobs_remaining -= 1
                self.scheduler.notify_dequeued(gpu_id)

    # --------------------------------------------------------------------- #
    #  Dequeue + steal                                                       #
    # --------------------------------------------------------------------- #

    def _dequeue(self, gpu_id: int) -> "tuple[Job, Workload] | None":
        while True:
            with self._queue_lock:
                if self._queues[gpu_id]:
                    return self._queues[gpu_id].popleft()

                stolen = self._try_steal_locked(gpu_id)
                if stolen is not None:
                    return stolen

                if self._all_dispatched and self._jobs_remaining == 0:
                    return None

            time.sleep(_IDLE_BACKOFF_S)

    def _try_steal_locked(self, thief_id: int) -> "tuple[Job, Workload] | None":
        """Steal a job from the tail of the most-loaded victim queue.

        Caller must hold ``self._queue_lock``.
        """
        if not self._queues:
            return None

        my_len = len(self._queues[thief_id])
        victim_id = max(self._queues, key=lambda gid: len(self._queues[gid]))
        victim_len = len(self._queues[victim_id])

        if victim_id == thief_id:
            return None
        if victim_len - my_len < self._steal_threshold:
            return None

        job, workload = self._queues[victim_id].pop()
        self._steal_count += 1
        log.info(
            "[%s] stolen: gpu %d (q=%d) -> gpu %d (q=%d)",
            job.id,
            victim_id,
            victim_len - 1,
            thief_id,
            my_len,
        )
        # Keep the scheduler's queue-length estimates in sync.
        self.scheduler.notify_dequeued(victim_id)
        return job, workload

    # --------------------------------------------------------------------- #
    #  Helpers                                                               #
    # --------------------------------------------------------------------- #

    def _set_monitor_load(self, gpu_id: int, util_pct: float, mem_mb: int) -> None:
        setter = getattr(self.monitor, "set_job_load", None)
        if callable(setter):
            setter(gpu_id, util_pct, mem_mb)

    @staticmethod
    def _wait_until(target_ts: float) -> None:
        now = time.monotonic()
        if target_ts > now:
            time.sleep(target_ts - now)


def _fmt_result(result: "WorkloadResult") -> str:
    if result.latencies_s:
        lats = result.latencies_s
        avg_ms = 1000 * sum(lats) / len(lats)
        return f"{len(lats)} reqs, avg={avg_ms:.1f}ms"
    if result.throughput_samples_per_s is not None:
        return f"{result.throughput_samples_per_s:.1f} samples/s"
    return "no metrics"
