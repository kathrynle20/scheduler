from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.metrics import latency_percentiles, temp_stability


@dataclass
class SummaryReport:
    run_dir: Path
    # PTQ latency percentiles (seconds)
    ptq_latency: dict[float, float] = field(default_factory=dict)
    ptq_avg_latency_s: float | None = None
    # Training average job duration
    training_avg_duration_s: float | None = None
    # Per-GPU temperature stats from timeseries
    per_gpu_temp: dict[int, dict[str, float]] = field(default_factory=dict)
    avg_temp_c: float | None = None

    def format(self) -> str:
        lines = [f"\n{'='*60}", f"  Summary for {self.run_dir}", f"{'='*60}"]

        # --- PTQ ---
        if self.ptq_avg_latency_s is not None:
            lines.append(f"  PTQ avg latency:      {self.ptq_avg_latency_s * 1000:.2f} ms")
        if self.ptq_latency:
            parts = ", ".join(f"p{int(p)}={v*1000:.2f}ms" for p, v in sorted(self.ptq_latency.items()))
            lines.append(f"  PTQ latency pctiles:  {parts}")

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

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


def analyze(run_dir: Path) -> SummaryReport:
    """Load timeseries.csv + results.csv from *run_dir* and produce a summary."""
    results_path = run_dir / "results.csv"
    timeseries_path = run_dir / "timeseries.csv"
    report = SummaryReport(run_dir=run_dir)

    # ---- results.csv: per-job latencies & durations ----
    ptq_latencies: list[float] = []
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

                if wtype == "ptq" and "latencies_s" in extra:
                    ptq_latencies.extend(extra["latencies_s"])

                if wtype == "training":
                    start = row.get("start_ts")
                    end = row.get("end_ts")
                    if start and end:
                        try:
                            training_durations.append(float(end) - float(start))
                        except ValueError:
                            pass

    if ptq_latencies:
        report.ptq_avg_latency_s = sum(ptq_latencies) / len(ptq_latencies)
        report.ptq_latency = latency_percentiles(ptq_latencies)

    if training_durations:
        report.training_avg_duration_s = sum(training_durations) / len(training_durations)

    # ---- timeseries.csv: per-GPU temperature ----
    gpu_temps: dict[int, list[float]] = {}

    if timeseries_path.exists():
        with timeseries_path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    gpu_id = int(row["gpu_id"])
                    temp = float(row["temp_c"])
                except (KeyError, ValueError):
                    continue
                gpu_temps.setdefault(gpu_id, []).append(temp)

    all_temps: list[float] = []
    for gpu_id, temps in sorted(gpu_temps.items()):
        report.per_gpu_temp[gpu_id] = temp_stability(temps)
        all_temps.extend(temps)

    if all_temps:
        report.avg_temp_c = sum(all_temps) / len(all_temps)

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
