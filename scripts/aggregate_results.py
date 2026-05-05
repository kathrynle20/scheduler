"""Aggregate benchmark suite results into presentation-ready tables.

Reads a manifest produced by `scripts/run_suite.sh`, computes mean ± std
across trials for every metric, and writes:

  - comparison.md       Markdown table per experiment, baseline vs work stealing
  - raw_metrics.csv     Per-trial metrics (for plotting in a notebook)
  - summary.csv         Aggregated mean/std per config (for quick spreadsheets)

Usage:
    python -m scripts.aggregate_results results/suite-<timestamp>/manifest.csv
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from evaluation.analyze import analyze


# Pair the configs: (display title, baseline config name, work-stealing config name)
PAIRS = [
    ("Experiment 1: PTQ inference (100 jobs)",       "ptq_100",       "ws_ptq_100"),
    ("Experiment 2: Training (50 jobs, 500 steps)",  "train_50",      "ws_train_50"),
    ("Experiment 3: Mixed (100 PTQ + 50 training)",  "mixed_100_50",  "ws_mixed_100_50"),
]

# (display label, metric key, decimal places, lower-is-better?)
METRIC_ROWS = [
    ("Avg job latency (ms)",      "avg_job_latency_ms",     1, True),
    ("p50 job latency (ms)",      "p50_job_latency_ms",     1, True),
    ("p95 job latency (ms)",      "p95_job_latency_ms",     1, True),
    ("p99 job latency (ms)",      "p99_job_latency_ms",     1, True),
    ("Avg queue wait (ms)",       "avg_queue_wait_ms",      1, True),
    ("Avg request time (ms)",     "avg_request_time_ms",    2, None),  # invariant
    ("Training avg duration (s)", "training_avg_duration_s",2, True),
    ("Cluster util mean (%)",     "cluster_util_mean_pct",  1, None),
    ("Cluster util std",          "cluster_util_std",       2, True),
    ("Max util imbalance (%)",    "max_util_imbalance_pct", 1, True),
    ("Avg temperature (C)",       "avg_temp_c",             1, None),
    ("Steal count",               "steal_count",            0, None),  # WS only
]


def collect_metrics(report) -> dict[str, float | None]:
    """Flatten a SummaryReport into a flat metric dict for aggregation."""
    out: dict[str, float | None] = {
        "avg_job_latency_ms": _ms(report.ptq_avg_job_latency_s),
        "p50_job_latency_ms": None,
        "p95_job_latency_ms": None,
        "p99_job_latency_ms": None,
        "avg_queue_wait_ms": _ms(report.ptq_avg_queue_wait_s),
        "avg_request_time_ms": _ms(report.ptq_avg_request_latency_s),
        "training_avg_duration_s": report.training_avg_duration_s,
        "cluster_util_mean_pct": None,
        "cluster_util_std": None,
        "max_util_imbalance_pct": None,
        "avg_temp_c": report.avg_temp_c,
        "steal_count": report.steal_count,
    }
    for p, v in (report.ptq_job_latency or {}).items():
        if int(p) == 50:
            out["p50_job_latency_ms"] = v * 1000
        elif int(p) == 95:
            out["p95_job_latency_ms"] = v * 1000
        elif int(p) == 99:
            out["p99_job_latency_ms"] = v * 1000
    if report.util_balance:
        out["cluster_util_mean_pct"] = report.util_balance.get("cluster_mean")
        out["cluster_util_std"] = report.util_balance.get("cluster_std")
        out["max_util_imbalance_pct"] = report.util_balance.get("max_imbalance")
    return out


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else seconds * 1000


def fmt_mean_std(values: list, decimals: int) -> str:
    """Format as 'mean ± std', skipping None values. '—' if no data."""
    clean = [v for v in values if v is not None]
    if not clean:
        return "—"
    if len(clean) == 1:
        return f"{clean[0]:.{decimals}f}"
    m = statistics.mean(clean)
    s = statistics.stdev(clean)
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def fmt_delta(baseline_vals: list, ws_vals: list, lower_is_better: bool | None) -> str:
    """Percent change, with a directional arrow."""
    bc = [v for v in baseline_vals if v is not None]
    wc = [v for v in ws_vals if v is not None]
    if not bc or not wc:
        return "—"
    bm = statistics.mean(bc)
    wm = statistics.mean(wc)
    if bm == 0:
        return "—"
    pct = (wm - bm) / bm * 100
    if abs(pct) < 1:
        return "≈ same"
    arrow = "↓" if pct < 0 else "↑"
    # Mark improvement when lower_is_better matches direction
    marker = ""
    if lower_is_better is True and pct < 0:
        marker = " **better**"
    elif lower_is_better is True and pct > 0:
        marker = " *worse*"
    return f"{arrow} {abs(pct):.1f}%{marker}"


def write_raw_csv(path: Path, by_config: dict[str, list[dict]]) -> None:
    keys = [k for _, k, _, _ in METRIC_ROWS]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "trial", *keys])
        for config, trials in by_config.items():
            for i, m in enumerate(trials, 1):
                w.writerow([config, i, *(m[k] for k in keys)])


def write_summary_csv(path: Path, by_config: dict[str, list[dict]]) -> None:
    keys = [k for _, k, _, _ in METRIC_ROWS]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        header = ["config", "n_trials"]
        for k in keys:
            header.extend([f"{k}_mean", f"{k}_std"])
        w.writerow(header)
        for config, trials in by_config.items():
            row = [config, len(trials)]
            for k in keys:
                vals = [t[k] for t in trials if t[k] is not None]
                if not vals:
                    row.extend(["", ""])
                elif len(vals) == 1:
                    row.extend([vals[0], ""])
                else:
                    row.extend([statistics.mean(vals), statistics.stdev(vals)])
            w.writerow(row)


def write_markdown(path: Path, suite_name: str, by_config: dict[str, list[dict]]) -> None:
    n_trials = max((len(v) for v in by_config.values()), default=0)
    lines = [
        f"# Benchmark Suite Comparison",
        "",
        f"**Suite:** `{suite_name}`  ",
        f"**Trials per config:** {n_trials}  ",
        f"**Cluster:** read from each run's NVML monitor output",
        "",
        "Format: `mean ± std` across trials. Δ column is the percent change of",
        "work stealing relative to baseline. Lower is better for latency, queue",
        "wait, util std, and max imbalance. Avg request time should be invariant",
        "(it is GPU compute, not affected by the scheduler).",
        "",
    ]

    for title, base_cfg, ws_cfg in PAIRS:
        lines.append(f"## {title}")
        lines.append("")
        base_trials = by_config.get(base_cfg, [])
        ws_trials = by_config.get(ws_cfg, [])

        if not base_trials and not ws_trials:
            lines.append(f"_No data for either config._")
            lines.append("")
            continue
        if not base_trials or not ws_trials:
            lines.append(
                f"_Incomplete: baseline trials={len(base_trials)}, "
                f"work stealing trials={len(ws_trials)}_"
            )
            lines.append("")
            continue

        lines.append("| Metric | Baseline | Work Stealing | Δ |")
        lines.append("|--------|----------|---------------|---|")
        for label, key, decimals, lower_is_better in METRIC_ROWS:
            bv = [t[key] for t in base_trials]
            wv = [t[key] for t in ws_trials]
            if all(v is None for v in bv) and all(v is None for v in wv):
                continue
            if key == "steal_count":
                # Baseline never steals -- only show WS column
                lines.append(
                    f"| {label} | — | {fmt_mean_std(wv, decimals)} | — |"
                )
                continue
            lines.append(
                f"| {label} "
                f"| {fmt_mean_std(bv, decimals)} "
                f"| {fmt_mean_std(wv, decimals)} "
                f"| {fmt_delta(bv, wv, lower_is_better)} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m scripts.aggregate_results <manifest.csv>", file=sys.stderr)
        return 2

    manifest_path = Path(argv[0])
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    suite_dir = manifest_path.parent

    by_config: dict[str, list[dict]] = defaultdict(list)
    missing = 0
    with manifest_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            run_dir = Path(row["run_dir"])
            if not run_dir.exists():
                print(f"WARN: missing {run_dir}", file=sys.stderr)
                missing += 1
                continue
            report = analyze(run_dir)
            by_config[row["config"]].append(collect_metrics(report))

    if not by_config:
        print("ERROR: no valid runs found in manifest", file=sys.stderr)
        return 1

    raw_csv = suite_dir / "raw_metrics.csv"
    summary_csv = suite_dir / "summary.csv"
    md_path = suite_dir / "comparison.md"

    write_raw_csv(raw_csv, by_config)
    write_summary_csv(summary_csv, by_config)
    write_markdown(md_path, suite_dir.name, by_config)

    print(f"Wrote {md_path}")
    print(f"Wrote {raw_csv}")
    print(f"Wrote {summary_csv}")
    if missing:
        print(f"Note: {missing} run dirs were missing and skipped.", file=sys.stderr)
    print()
    print(md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
