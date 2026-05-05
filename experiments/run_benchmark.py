from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import yaml

import monitoring
import schedulers
import workloads
from evaluation.collector import MetricsCollector
from framework.job import Job
from framework.runner import ExperimentRunner
from framework.ws_runner import WorkStealingRunner
from schedulers.work_stealing import WorkStealingScheduler

log = logging.getLogger("run_benchmark")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def build_job_list(config: dict) -> list[Job]:
    """Expand `workload_mix` into concrete Job instances with arrival times.

    Jobs of all types are randomly interspersed.  The overall arrival rate is
    governed by ``arrival_rate_hz`` (default 2 jobs/sec).
    """
    arrival_rate_hz = float(config.get("arrival_rate_hz", 2.0))
    t0 = time.monotonic() + 0.5  # small lead-in so the collector has a sample

    # --- build un-timed jobs from each spec ---
    jobs: list[Job] = []
    for spec in config["workload_mix"]:
        kind = spec["type"]
        count = int(spec["count"])
        mem = int(spec.get("mem_required_mb", 0))
        if kind == "ptq":
            for i in range(count):
                jobs.append(
                    Job(
                        id=f"ptq-{i}",
                        workload_type="ptq",
                        mem_required_mb=mem,
                        arrival_time=0.0,  # assigned below
                        payload={
                            "num_requests": int(spec.get("num_requests", 50)),
                            "matrix_size": int(spec.get("matrix_size", 2048)),
                        },
                    )
                )
        elif kind == "training":
            num_steps = int(spec.get("num_steps", 200))
            batch_size = int(spec.get("batch_size", 128))
            hidden = int(spec.get("hidden", 4096))
            for i in range(count):
                jobs.append(
                    Job(
                        id=f"train-{i}",
                        workload_type="training",
                        mem_required_mb=mem,
                        arrival_time=0.0,  # assigned below
                        payload={
                            "num_steps": num_steps,
                            "batch_size": batch_size,
                            "hidden": hidden,
                        },
                    )
                )
        else:
            raise ValueError(f"unknown workload type in config: {kind}")

    # --- shuffle and assign arrival times ---
    # poisson_arrivals: true  → exponentially distributed inter-arrival gaps
    #                           (realistic bursty traffic; required for work stealing
    #                           to see queue imbalance)
    # poisson_arrivals: false → evenly spaced (deterministic, good for smoke tests)
    random.shuffle(jobs)
    use_poisson = bool(config.get("poisson_arrivals", False))
    t = t0
    for idx, job in enumerate(jobs):
        if use_poisson:
            t += random.expovariate(arrival_rate_hz)
        else:
            t = t0 + idx * (1.0 / arrival_rate_hz if arrival_rate_hz > 0 else 0.0)
        job.arrival_time = t

    return jobs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a scheduler benchmark.")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--scheduler", choices=["baseline", "hybrid", "work_stealing"], default=None,
                    help="override scheduler.name from the config")
    ap.add_argument("--monitor", choices=["nvml", "simulated"], default=None,
                    help="override monitor.backend from the config")
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="override the number of GPUs (0..N-1); auto-generates ids, worker_gpus, neighbors")
    ap.add_argument("--arrival-rate", type=float, default=None,
                    help="override arrival_rate_hz from the config (jobs/sec across all GPUs). "
                         "Target util ≈ (arrival_rate / num_gpus) × job_duration_s. "
                         "Scale linearly with num_gpus to keep utilization constant: "
                         "e.g. 1.4 hz on 2 GPUs → 2.8 hz on 4 GPUs for the same ~65%% util.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="enable DEBUG logging (per-scheduler decision rationale)")
    args = ap.parse_args(argv)

    _setup_logging(args.verbose)
    config = load_config(args.config)
    sched_name = args.scheduler or config["scheduler"]["name"]
    monitor_backend = args.monitor or config["monitor"]["backend"]

    if args.num_gpus is not None:
        ids = list(range(args.num_gpus))
        config["gpus"] = {
            "ids": ids,
            "worker_gpus": ids,
            "neighbors": {i: [j for j in ids if j != i] for i in ids},
        }

    if args.arrival_rate is not None:
        config["arrival_rate_hz"] = args.arrival_rate

    log.info(
        "config=%s scheduler=%s monitor=%s gpus=%s",
        args.config,
        sched_name,
        monitor_backend,
        config["gpus"]["ids"],
    )

    gpu_cfg = config["gpus"]
    worker_gpu_ids = gpu_cfg.get("worker_gpus")
    if worker_gpu_ids is None:
        # Fallback: GPU 0 schedules, rest are workers.
        scheduler_gpu = gpu_cfg.get("scheduler_gpu", gpu_cfg["ids"][0])
        worker_gpu_ids = [g for g in gpu_cfg["ids"] if g != scheduler_gpu]

    scheduler = schedulers.build(sched_name, config["scheduler"])
    monitor = monitoring.build(monitor_backend, config)
    collector = MetricsCollector(sample_hz=float(config["monitor"].get("sample_hz", 10)))

    if isinstance(scheduler, WorkStealingScheduler):
        ws_cfg = config["scheduler"].get("work_stealing") or {}
        steal_threshold = int(ws_cfg.get("steal_threshold", 2))
        stealing_active = steal_threshold < 9999
        log.info(
            "runner=WorkStealingRunner steal_threshold=%d (%s)",
            steal_threshold,
            "stealing ENABLED" if stealing_active else "stealing DISABLED — baseline mode",
        )
        runner = WorkStealingRunner(
            scheduler=scheduler,
            monitor=monitor,
            collector=collector,
            output_root=Path(config.get("output_dir", "runs")),
            worker_gpu_ids=worker_gpu_ids,
            steal_threshold=steal_threshold,
        )
    else:
        runner = ExperimentRunner(
            scheduler=scheduler,
            monitor=monitor,
            collector=collector,
            output_root=Path(config.get("output_dir", "runs")),
            worker_gpu_ids=worker_gpu_ids,
        )

    jobs = build_job_list(config)
    log.info("built %d jobs: %s", len(jobs), _summarize_jobs(jobs))
    artifacts = runner.run(jobs, workload_factory=workloads.build)
    log.info("artifacts: %s", artifacts.run_dir)

    from evaluation.analyze import analyze
    report = analyze(artifacts.run_dir)
    print(report.format())

    return 0


def _summarize_jobs(jobs: list[Job]) -> str:
    by_type: dict[str, int] = {}
    for j in jobs:
        by_type[j.workload_type] = by_type.get(j.workload_type, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(by_type.items()))


if __name__ == "__main__":
    raise SystemExit(main())
