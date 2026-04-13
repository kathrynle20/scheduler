from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

import monitoring
import schedulers
import workloads
from evaluation.collector import MetricsCollector
from framework.job import Job
from framework.runner import ExperimentRunner


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def build_job_list(config: dict) -> list[Job]:
    """Expand `workload_mix` into concrete Job instances with arrival times."""
    jobs: list[Job] = []
    t0 = time.monotonic() + 0.5  # small lead-in so the collector has a sample
    for spec in config["workload_mix"]:
        kind = spec["type"]
        count = int(spec["count"])
        mem = int(spec.get("mem_required_mb", 0))
        if kind == "ptq":
            rate = float(spec.get("arrival_rate_hz", 1.0))
            for i in range(count):
                jobs.append(
                    Job(
                        id=f"ptq-{i}",
                        workload_type="ptq",
                        mem_required_mb=mem,
                        arrival_time=t0 + i / rate,
                        payload={"num_requests": int(spec.get("num_requests", 50))},
                    )
                )
        elif kind == "training":
            duration = float(spec.get("duration_s", 60.0))
            for i in range(count):
                jobs.append(
                    Job(
                        id=f"train-{i}",
                        workload_type="training",
                        mem_required_mb=mem,
                        arrival_time=t0 + i * 0.1,
                        payload={"duration_s": duration},
                    )
                )
        else:
            raise ValueError(f"unknown workload type in config: {kind}")
    return jobs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a scheduler benchmark.")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--scheduler", choices=["baseline", "hybrid"], default=None,
                    help="override scheduler.name from the config")
    ap.add_argument("--monitor", choices=["nvml", "simulated"], default=None,
                    help="override monitor.backend from the config")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    sched_name = args.scheduler or config["scheduler"]["name"]
    monitor_backend = args.monitor or config["monitor"]["backend"]

    scheduler = schedulers.build(sched_name, config["scheduler"])
    monitor = monitoring.build(monitor_backend, config)
    collector = MetricsCollector(sample_hz=float(config["monitor"].get("sample_hz", 10)))
    runner = ExperimentRunner(
        scheduler=scheduler,
        monitor=monitor,
        collector=collector,
        output_root=Path(config.get("output_dir", "runs")),
    )

    jobs = build_job_list(config)
    artifacts = runner.run(jobs, workload_factory=workloads.build)
    print(f"run complete: {artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
