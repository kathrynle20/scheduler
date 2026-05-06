"""Modal app for running scheduler benchmarks on cloud GPUs.

One-time setup (on your laptop)::

    pip install modal
    modal setup

Smoke test (no GPU; CPU container, simulated monitor)::

    modal run modal_app.py::smoke

Single experiment on a 2x L40S box (defaults match the ~70% util target from
docs/running_tests_on_linux.md)::

    modal run modal_app.py::experiment \
        --config configs/ptq_100.yaml \
        --num-gpus 2 --arrival-rate 1.5

Full benchmark suite (3 experiments x 2 schedulers x N trials)::

    modal run modal_app.py::suite --trials 5 --num-gpus 4 --arrival-rate 3.0

Pull the persisted CSVs and reports back to your laptop::

    modal volume get scheduler-runs runs ./modal-runs --recursive
    modal volume get scheduler-runs results ./modal-results --recursive

Switching GPU type / count: edit the ``gpu=...`` arg on the function decorators
below (or the ``GPU_*`` constants at the top). ``--num-gpus`` only changes the
*logical* GPU count the runner schedules onto -- the *physical* count is fixed
by the decorator at deploy time, so the two should match.
"""

from __future__ import annotations

import datetime
import shutil
import subprocess
import sys
from pathlib import Path

import modal

# CUDA 12.1 image; matches the cu121 torch wheel the docs install on bare metal.
CUDA_IMAGE = "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04"
PYTHON_VERSION = "3.11"

# GPU presets. Edit these to switch hardware. Modal supports L4, L40S, A10,
# A100-40GB, A100-80GB, H100, H200, B200; see modal.com/docs/guide/gpu.
GPU_EXPERIMENT = "L40S:2"   # single-experiment runs
GPU_SUITE = "L40S:2"        # multi-trial suite runs
GPU_CHECK = "L40S"          # one device for the cuda sanity check

REPO_REMOTE = "/root/scheduler"
VOLUME_NAME = "scheduler-runs"
VOLUME_MOUNT = "/data"
VOLUME_RUNS = f"{VOLUME_MOUNT}/runs"
VOLUME_RESULTS = f"{VOLUME_MOUNT}/results"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python=PYTHON_VERSION)
    .apt_install("git")
    # Inline the runtime deps from requirements.txt so we don't need the file
    # in the build context. Keep in sync with requirements.txt.
    .pip_install(
        "nvidia-ml-py>=12.535",
        "pyyaml>=6.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "pytest>=7.4",
    )
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu121")
    .env({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        ".",
        remote_path=REPO_REMOTE,
        ignore=[
            ".git",
            ".venv",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".claude",
            "runs",
            "results",
            "logs",
            "**/__pycache__",
            "*.pyc",
        ],
    )
)

app = modal.App("scheduler-benchmark", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _run_in_repo(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_REMOTE)


def _persist(subdir: str) -> list[str]:
    """Copy any new entries from ``REPO_REMOTE/<subdir>`` into the volume."""
    src = Path(REPO_REMOTE) / subdir
    dst = Path(VOLUME_MOUNT) / subdir
    dst.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    if not src.exists():
        return moved
    for entry in sorted(src.iterdir()):
        target = dst / entry.name
        if target.exists():
            continue
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)
        moved.append(entry.name)
    return moved


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=600)
def smoke() -> None:
    """Simulated-monitor end-to-end check. CPU only."""
    _run_in_repo([
        "python", "-m", "experiments.run_benchmark",
        "--config", "configs/smoke.yaml",
        "--scheduler", "baseline",
        "--monitor", "simulated",
    ])
    moved = _persist("runs")
    volume.commit()
    print(f"Persisted runs: {moved}")


@app.function(volumes={VOLUME_MOUNT: volume}, gpu=GPU_CHECK, timeout=300)
def gpu_check() -> None:
    """Verify CUDA is wired up correctly inside the container."""
    import torch
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    subprocess.run(["nvidia-smi"], check=True)


@app.function(volumes={VOLUME_MOUNT: volume}, gpu=GPU_EXPERIMENT, timeout=3600)
def experiment(
    config: str = "configs/ptq_100.yaml",
    scheduler: str = "",
    monitor: str = "nvml",
    num_gpus: int = 0,
    arrival_rate: float = 0.0,
    verbose: bool = True,
) -> str:
    """Run a single ``experiments.run_benchmark`` invocation on the remote GPU.

    Empty / zero values mean "fall back to whatever the YAML says". ``num_gpus``
    is *logical*; it must be <= the physical count baked into ``GPU_EXPERIMENT``.
    """
    cmd = [
        "python", "-m", "experiments.run_benchmark",
        "--config", config,
        "--monitor", monitor,
    ]
    if scheduler:
        cmd += ["--scheduler", scheduler]
    if num_gpus:
        cmd += ["--num-gpus", str(num_gpus)]
    if arrival_rate:
        cmd += ["--arrival-rate", str(arrival_rate)]
    if verbose:
        cmd += ["-v"]
    _run_in_repo(cmd)

    moved = _persist("runs")
    volume.commit()
    latest = moved[-1] if moved else ""
    if latest:
        print(f"Persisted run -> {VOLUME_RUNS}/{latest}")
        print(f"Pull with: modal volume get {VOLUME_NAME} runs/{latest} ./modal-runs")
    return latest


@app.function(volumes={VOLUME_MOUNT: volume}, gpu="A100:4", timeout=4 * 3600)
def suite(
    trials: int = 3,
    filter: str = "",
    num_gpus: int = 0,
    arrival_rate: float = 0.0,
    aggregate: bool = True,
) -> str:
    """Run ``scripts/run_suite.sh`` on the remote GPU box.

    Positional args to the bash script are ``[N_TRIALS] [FILTER] [NUM_GPUS]
    [ARRIVAL_RATE]``. ``aggregate=True`` then runs scripts.aggregate_results
    on the new manifest so ``comparison.md`` is included in the persisted
    suite directory.
    """
    args = [str(trials), filter]
    if num_gpus or arrival_rate:
        args.append(str(num_gpus) if num_gpus else "")
    if arrival_rate:
        args.append(str(arrival_rate))
    _run_in_repo(["bash", "scripts/run_suite.sh", *args])

    if aggregate:
        suite_dirs = sorted((Path(REPO_REMOTE) / "results").glob("suite-*"))
        if suite_dirs:
            manifest = suite_dirs[-1] / "manifest.csv"
            if manifest.exists():
                _run_in_repo(["python", "-m", "scripts.aggregate_results", str(manifest)])

    moved_runs = _persist("runs")
    moved_results = _persist("results")
    volume.commit()
    print(f"Persisted runs: {len(moved_runs)} new")
    print(f"Persisted results: {moved_results}")
    latest = moved_results[-1] if moved_results else ""
    if latest:
        print(f"Suite -> {VOLUME_RESULTS}/{latest}")
        print(f"Pull with: modal volume get {VOLUME_NAME} results/{latest} ./modal-results")
    return latest


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=300)
def list_runs() -> None:
    """List runs/ and results/ currently in the volume."""
    for sub in ("runs", "results"):
        d = Path(VOLUME_MOUNT) / sub
        print(f"== {sub}/ ==")
        if not d.exists():
            print("  (empty)")
            continue
        for entry in sorted(d.iterdir()):
            print(f"  {entry.name}")
