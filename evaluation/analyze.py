from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.metrics import latency_percentiles, temp_stability, utilization_balance


@dataclass
class SummaryReport:
    run_dir: Path
    # PTQ job-level latency: arrival → completion (the scheduling-sensitive metric)
    ptq_job_latency: dict[float, float] = field(default_factory=dict)
    ptq_avg_job_latency_s: float | None = None
    # PTQ queue wait time: arrival → execution start
    ptq_avg_queue_wait_s: float | None = None
    # PTQ per-request GPU compute time (scheduler-invariant sanity check)
    ptq_request_latency: dict[float, float] = field(default_factory=dict)
    ptq_avg_request_latency_s: float | None = None
    # Training average job duration
    training_avg_duration_s: float | None = None
    # Per-GPU temperature stats from timeseries
    per_gpu_temp: dict[int, dict[str, float]] = field(default_factory=dict)
    avg_temp_c: float | None = None
    # Per-GPU utilization balance (work-stealing comparison metric)
    util_balance: dict | None = None
    # Number of steals during the run (work-stealing only)
    steal_count: int | None = None

    def format(self) -> str:
        lines = [f"\n{'='*60}", f"  Summary for {self.run_dir}", f"{'='*60}"]

        # --- PTQ job-level latency (primary scheduling metric) ---
        if self.ptq_avg_job_latency_s is not None:
            lines.append(f"  PTQ avg job latency:  {self.ptq_avg_job_latency_s * 1000:.1f} ms  (arrival → completion)")
        if self.ptq_job_latency:
            parts = ", ".join(f"p{int(p)}={v*1000:.1f}ms" for p, v in sorted(self.ptq_job_latency.items()))
            lines.append(f"  PTQ job pctiles:      {parts}")
        if self.ptq_avg_queue_wait_s is not None:
            lines.append(f"  PTQ avg queue wait:   {self.ptq_avg_queue_wait_s * 1000:.1f} ms  (arrival → exec start)")

        # --- PTQ per-request GPU time (should be constant across schedulers) ---
        if self.ptq_avg_request_latency_s is not None:
            lines.append(f"  PTQ avg request time: {self.ptq_avg_request_latency_s * 1000:.2f} ms  (GPU compute, scheduler-invariant)")

        # --- Training ---
        if self.training_avg_duration_s is not None:
            lines.append(f"  Training avg duration: {self.training_avg_duration_s:.2f} s")

        # --- Temperatures ---
        if self.per_gpu_temp:
            lines.append("  Per-GPU temperatures:")
            for gpu_id, stats in sorted(self.per_gpu_temp.items()):
                lines.append(
                    f"    GPU {gpu_id}: mean={stats['mean']:.1f}C, "
                    f"max={stats['max']:.1f}C, std={stats['std']:.1f}C"
                )
        if self.avg_temp_c is not None:
            lines.append(f"  Avg temperature (all GPUs): {self.avg_temp_c:.1f} C")

        # --- Utilization balance ---
        if self.util_balance:
            ub = self.util_balance
            per_gpu = ub.get("per_gpu_mean", {})
            if per_gpu:
                parts = ", ".join(
                    f"gpu{gid}={u:.1f}%" for gid, u in sorted(per_gpu.items())
                )
                lines.append(f"  Per-GPU mean util:    {parts}")
            lines.append(
                f"  Cluster util:         mean={ub['cluster_mean']:.1f}%, "
                f"std={ub['cluster_std']:.2f}, "
                f"max_imbalance={ub['max_imbalance']:.1f}%"
            )

        # --- Work stealing ---
        if self.steal_count is not None:
            lines.append(f"  Steal count:          {self.steal_count}")

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


def analyze(run_dir: Path) -> SummaryReport:
    """Load timeseries.csv + results.csv from *run_dir* and produce a summary."""
    results_path = run_dir / "results.csv"
    timeseries_path = run_dir / "timeseries.csv"
    report = SummaryReport(run_dir=run_dir)

    # ---- results.csv: per-job latencies & durations ----
    ptq_job_latencies: list[float] = []    # end_ts - arrival_time  (scheduling-sensitive)
    ptq_queue_waits: list[float] = []      # start_ts - arrival_time (queue wait only)
    ptq_request_latencies: list[float] = [] # per-matmul GPU compute time (invariant)
    training_durations: list[float] = []

    if results_path.exists():
        with results_path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                extra_raw = row.get("extra", "")
                if extra_raw in ("", "deferred"):
                    continue
                try:
                    extra = json.loads(extra_raw)
                except json.JSONDecodeError:
                    continue

                wtype = row.get("workload_type", "")

                if wtype == "ptq":
                    # Job-level latency: requires arrival_time column (added in updated runners)
                    arrival = row.get("arrival_time")
                    start = row.get("start_ts")
                    end = row.get("end_ts")
                    if arrival and end:
                        try:
                            ptq_job_latencies.append(float(end) - float(arrival))
                        except ValueError:
                            pass
                    if arrival and start:
                        try:
                            ptq_queue_waits.append(float(start) - float(arrival))
                        except ValueError:
                            pass
                    # Per-request GPU compute time (scheduler-invariant)
                    if "latencies_s" in extra:
                        ptq_request_latencies.extend(extra["latencies_s"])

                if wtype == "training":
                    start = row.get("start_ts")
                    end = row.get("end_ts")
                    if start and end:
                        try:
                            training_durations.append(float(end) - float(start))
                        except ValueError:
                            pass

    if ptq_job_latencies:
        report.ptq_avg_job_latency_s = sum(ptq_job_latencies) / len(ptq_job_latencies)
        report.ptq_job_latency = latency_percentiles(ptq_job_latencies)
    if ptq_queue_waits:
        report.ptq_avg_queue_wait_s = sum(ptq_queue_waits) / len(ptq_queue_waits)
    if ptq_request_latencies:
        report.ptq_avg_request_latency_s = sum(ptq_request_latencies) / len(ptq_request_latencies)
        report.ptq_request_latency = latency_percentiles(ptq_request_latencies)

    if training_durations:
        report.training_avg_duration_s = sum(training_durations) / len(training_durations)

    # ---- timeseries.csv: per-GPU temperature + utilization ----
    gpu_temps: dict[int, list[float]] = {}
    gpu_utils: dict[int, list[float]] = {}

    if timeseries_path.exists():
        with timeseries_path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    gpu_id = int(row["gpu_id"])
                    temp = float(row["temp_c"])
                    util = float(row["util_pct"])
                except (KeyError, ValueError):
                    continue
                gpu_temps.setdefault(gpu_id, []).append(temp)
                gpu_utils.setdefault(gpu_id, []).append(util)

    all_temps: list[float] = []
    for gpu_id, temps in sorted(gpu_temps.items()):
        report.per_gpu_temp[gpu_id] = temp_stability(temps)
        all_temps.extend(temps)

    if all_temps:
        report.avg_temp_c = sum(all_temps) / len(all_temps)

    if gpu_utils:
        report.util_balance = utilization_balance(gpu_utils)

    # ---- metadata.json: work-stealing run info, if present ----
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open() as fh:
                meta = json.load(fh)
            if "steal_count" in meta:
                report.steal_count = int(meta["steal_count"])
        except (json.JSONDecodeError, ValueError):
            pass

    return report


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m evaluation.analyze <run_dir>", file=sys.stderr)
        return 2
    report = analyze(Path(argv[0]))
    print(report.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
