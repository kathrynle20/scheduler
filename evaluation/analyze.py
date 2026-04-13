from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SummaryReport:
    run_dir: Path
    latency: dict[float, float] = field(default_factory=dict)       # percentile -> seconds
    training_throughput: float | None = None                        # samples/sec
    per_gpu_temp: dict[int, dict[str, float]] = field(default_factory=dict)
    per_gpu_util: dict[int, dict[str, float]] = field(default_factory=dict)

    def format(self) -> str:
        lines = [f"Summary for {self.run_dir}"]
        if self.latency:
            lines.append("  Latency: " + ", ".join(f"p{int(p)}={v*1000:.1f}ms" for p, v in sorted(self.latency.items())))
        if self.training_throughput is not None:
            lines.append(f"  Training throughput: {self.training_throughput:.1f} samples/s")
        for gpu_id, stats in sorted(self.per_gpu_temp.items()):
            lines.append(f"  GPU {gpu_id} temp: mean={stats.get('mean', float('nan')):.1f}C, max={stats.get('max', float('nan')):.1f}C")
        return "\n".join(lines)


def analyze(run_dir: Path) -> SummaryReport:
    """Load timeseries.csv + results.csv from `run_dir` and produce a summary.

    Expected implementation:
      - Read timeseries.csv with pandas.
      - For each gpu_id, compute temp_stability(...) and a matching util summary.
      - Read results.csv; parse the extras json per row; aggregate PTQ
        latencies and training throughput.
      - Populate SummaryReport.
    """
    # TODO(ws2)
    raise NotImplementedError("analyze -- owned by workstream 2")


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
