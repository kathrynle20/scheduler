# Scheduler Codebase Overview

## Project Purpose

This is a **6.S984 semester project** that implements and benchmarks GPU scheduling strategies for mixed ML workloads. The central question: does a thermally-aware, utilization-weighted scheduler outperform a naive round-robin baseline in latency, throughput, and thermal stability when running a mix of PTQ inference and training jobs on a multi-GPU cluster?

Two schedulers are compared:
- **BaselineScheduler** — pure round-robin, ignores all GPU state
- **HybridScheduler** — weighted scoring combining utilization + thermal state + neighbor heat (currently stubbed, Workstream 1's task)

---

## Directory Structure

```
scheduler/
├── configs/              # YAML experiment configurations
│   ├── default.yaml      # 4 GPUs, 100 PTQ + 50 training jobs, NVML monitor
│   └── smoke.yaml        # 2 GPUs, 4 jobs, simulated monitor — fast CI/local test
├── framework/            # Core abstractions
│   ├── gpu.py            # GpuState dataclass
│   ├── job.py            # Job dataclass
│   └── runner.py         # ExperimentRunner — orchestrates the full experiment
├── monitoring/           # GPU state sampling
│   ├── gpu_monitor.py    # Abstract GpuMonitor + NvmlMonitor (real hardware)
│   └── simulator.py      # SimulatedMonitor — discrete-time thermal model
├── schedulers/           # Placement algorithms
│   ├── base.py           # Abstract Scheduler base class
│   ├── baseline.py       # Round-robin reference
│   └── hybrid.py         # Weighted scoring (NotImplementedError — TODO)
├── workloads/            # ML job implementations
│   ├── base.py           # Abstract Workload + WorkloadResult
│   ├── ptq_inference.py  # Post-training-quantization inference (latency-sensitive)
│   ├── training.py       # Training job (throughput-focused)
│   └── _gpu.py           # Shared torch/CUDA helpers
├── evaluation/           # Post-run analysis
│   ├── collector.py      # MetricsCollector — background CSV sampler
│   ├── metrics.py        # latency_percentiles, throughput, temp_stability
│   └── analyze.py        # SummaryReport generator
├── experiments/
│   └── run_benchmark.py  # Main entry point
├── tests/                # pytest unit + integration tests
│   ├── test_schedulers.py
│   ├── test_metrics.py
│   └── test_workloads.py
└── docs/
    ├── running_baseline_on_cluster.md  # MIT cluster setup guide
    └── codebase_overview.md            # This file
```

---

## Key Data Structures

### `Job` (`framework/job.py`)
```python
@dataclass
class Job:
    id: str
    workload_type: WorkloadType   # "ptq" | "training"
    mem_required_mb: int
    arrival_time: float           # seconds since experiment start
    payload: dict[str, Any]       # workload-specific params
```

### `GpuState` (`framework/gpu.py`)
```python
@dataclass
class GpuState:
    id: int
    util_pct: float               # 0–100
    temp_c: float
    mem_used_mb: int
    mem_total_mb: int
    neighbor_ids: list[int]       # physically adjacent GPUs (for thermal coupling)

    @property
    def mem_free_mb(self) -> int: ...
    def can_fit(self, mem_required_mb: int) -> bool: ...
```

### `WorkloadResult` (`workloads/base.py`)
```python
@dataclass
class WorkloadResult:
    start_ts: float
    end_ts: float
    latencies_s: list[float] = []             # PTQ: per-request latency
    throughput_samples_per_s: float | None    # Training: samples/sec
    extra: dict[str, Any] = {}
```

### `RunArtifacts` (`framework/runner.py`)
```python
@dataclass
class RunArtifacts:
    run_dir: Path
    timeseries_path: Path   # timeseries.csv
    results_path: Path      # results.csv
```

---

## Execution Flow

```
experiments/run_benchmark.py main()
    │
    ├─ load_config(path)            → dict from YAML
    ├─ build_job_list(config)       → list[Job], evenly-spaced arrival times
    ├─ schedulers.build(name, cfg)  → BaselineScheduler | HybridScheduler
    ├─ monitoring.build(backend, cfg) → NvmlMonitor | SimulatedMonitor
    ├─ MetricsCollector(sample_hz)
    ├─ ExperimentRunner(scheduler, monitor, collector, cfg)
    │
    └─ runner.run(jobs, workload_factory)
            │
            ├─ [BG daemon]  MetricsCollector → timeseries.csv @ sample_hz
            │
            └─ [main thread] for each job (sorted by arrival_time):
                    _wait_until(arrival_time)
                    _schedule_with_retry(job)
                        ├─ monitor.sample() → current GpuState list
                        ├─ filter to free worker GPUs
                        └─ scheduler.place(job, available_states) → gpu_id | None
                    pool.submit(_run_workload, job, gpu_id)
                        ├─ [pool thread] workload.run(gpu_id) → WorkloadResult
                        ├─ write row to results.csv
                        └─ mark GPU free, notify scheduler

    post-run: evaluation.analyze(run_dir) → SummaryReport (stdout)
```

---

## Scheduler Logic

### Abstract Interface (`schedulers/base.py`)
```python
class Scheduler(ABC):
    def place(self, job: Job, gpu_states: list[GpuState]) -> int | None:
        # Return gpu_id to assign, or None to defer the job
```

### BaselineScheduler (`schedulers/baseline.py`)
- Round-robin over `sorted(gpu_states, key=lambda g: g.id)`
- Maintains `_last_index` counter, cycles with `(index + 1) % n`
- Ignores memory, temperature, utilization — pure reference

### HybridScheduler (`schedulers/hybrid.py`) — **NOT YET IMPLEMENTED**
Intended scoring formula (from docstring):
```
score(gpu) = w_utilization * (1 - util_pct / 100)
           + w_thermal    * thermal_score(temp_c)
           - neighbor_temp_weight * mean(neighbor_temp - ambient)

thermal_score(t):
    = 1.0                           if t < temp_soft_limit_c
    = linear interpolation to 0     if temp_soft_limit_c <= t < temp_hard_limit_c
    = -inf (hard exclude)           if t >= temp_hard_limit_c
```

Config parameters (under `scheduler.hybrid` in YAML):
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `w_utilization` | 0.5 | Weight for idle-GPU preference |
| `w_thermal` | 0.5 | Weight for cool-GPU preference |
| `temp_soft_limit_c` | 75.0 | Penalty begins above this |
| `temp_hard_limit_c` | 85.0 | Hard exclude above this |
| `neighbor_temp_weight` | 0.25 | Heat-coupling penalty coefficient |

The test `test_hybrid_prefers_cool_idle_gpu` in `tests/test_schedulers.py` is marked `@pytest.mark.xfail` until this is implemented.

---

## Monitoring Backends

### NvmlMonitor (`monitoring/gpu_monitor.py`)
- Wraps `pynvml` (nvidia-ml-py) for real hardware
- `sample()` queries per-GPU util, temp, memory via NVML
- **Requires** `CUDA_DEVICE_ORDER=PCI_BUS_ID` to align NVML ↔ torch indices
- Neighbor map must be provided manually (physical topology not auto-discoverable)

### SimulatedMonitor (`monitoring/simulator.py`)
- Deterministic discrete-time thermal model — no GPU hardware needed
- Per-tick thermal update:
  ```
  temp_new = temp_old
           + k_load    * (util / 100)
           - k_cool    * (temp - ambient_c)
           + k_neighbor * mean(neighbor_temp - temp)
  ```
- `set_job_load(gpu_id, util_pct, mem_mb)` — called by runner when a job starts/ends
- Config keys (under `monitor.simulated`): `ambient_c`, `k_load`, `k_cool`, `k_neighbor`
- Used for smoke tests and local development without a GPU

---

## Concurrency Model

| Thread | Role |
|--------|------|
| Main | Scheduling loop — waits on arrival times, calls `scheduler.place()`, dispatches jobs |
| Thread pool (N workers) | One thread per worker GPU, each blocks on `workload.run()` |
| Metrics daemon | Background sampler, writes to `timeseries.csv` at `sample_hz` |

Synchronization:
- `_busy_lock: threading.Lock` — protects `_busy_gpus: set[int]`
- `_gpu_freed: threading.Condition(_busy_lock)` — worker notifies scheduler when GPU freed
- `csv_lock: threading.Lock` — serializes writes to `results.csv`

---

## Workload Implementations

### PTQInference (`workloads/ptq_inference.py`)
- Represents post-training-quantization inference: bursty, latency-sensitive
- **Real path (CUDA available):** allocates `(matrix_size, matrix_size)` float16 tensor, runs `num_requests` synchronized matmuls, records per-request latency
- **Stub path (no GPU):** sleeps 5–20ms per request, returns synthetic latencies
- Payload keys: `num_requests` (default 50), `matrix_size` (default 2048)

### Training (`workloads/training.py`)
- Represents model training: sustained compute, throughput-focused
- **Real path:** 3-layer MLP (Linear→ReLU→Linear→ReLU→Linear), runs `num_steps` forward+backward+SGD steps
- **Stub path:** sleeps ~5ms per step, returns synthetic throughput
- Payload keys: `num_steps` (default 200), `batch_size` (default 128), `hidden` (default 4096)

---

## Output Files

Both written to `runs/<timestamp>-<uuid>/`:

**`timeseries.csv`** — GPU telemetry sampled by MetricsCollector:
```
ts, gpu_id, util_pct, temp_c, mem_used_mb, mem_total_mb
```

**`results.csv`** — per-job completion records:
```
job_id, workload_type, gpu_id, start_ts, end_ts, extra
```
`extra` is JSON: `{"latencies_s": [...]}` for PTQ, `{"throughput_samples_per_s": X}` for training, or `"deferred"` if the scheduler returned None.

---

## Evaluation & Metrics

**`evaluation/metrics.py`:**
- `latency_percentiles(latencies_s, percentiles=(50, 95, 99))` → `{p: latency}`
- `throughput(num_samples, duration_s)` → samples/sec
- `temp_stability(temps_c)` → `{"mean", "std", "max", "p99"}`

**`evaluation/analyze.py`:** Reads both CSVs, computes per-GPU temp stats + PTQ latency percentiles + average training duration, returns `SummaryReport`.

---

## Configuration Reference

**`configs/default.yaml`** (production):
```yaml
gpus:
  ids: [0, 1, 2, 3]
  scheduler_gpu: 0          # GPU that runs the scheduling loop
  worker_gpus: [1, 2, 3]
  neighbors: {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}  # linear chain topology

scheduler:
  name: baseline            # "baseline" | "hybrid"
  hybrid:
    w_utilization: 0.5
    w_thermal: 0.5
    temp_soft_limit_c: 75
    temp_hard_limit_c: 85
    neighbor_temp_weight: 0.25

monitor:
  backend: nvml             # "nvml" | "simulated"
  sample_hz: 10
  simulated:
    ambient_c: 25
    k_load: 0.6
    k_cool: 0.05
    k_neighbor: 0.1

arrival_rate_hz: 2          # jobs/sec
workload_mix:
  - {type: ptq,      count: 100, mem_required_mb: 2048}
  - {type: training, count: 50,  mem_required_mb: 16384, num_steps: 500}

output_dir: ./runs
```

**`configs/smoke.yaml`** — minimal (2 GPUs, 4 jobs, simulated), used for local/CI testing.

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| `BaselineScheduler` | ✅ Done | Round-robin reference |
| `HybridScheduler` | ❌ Stub | `NotImplementedError` — Workstream 1 |
| `PTQInference` | ✅ Done | Real + stub paths |
| `Training` | ✅ Done | Real + stub paths |
| `NvmlMonitor` | ✅ Done | Requires GPU hardware |
| `SimulatedMonitor` | ✅ Done | Deterministic thermal model |
| `MetricsCollector` | ✅ Done | Background CSV sampler |
| `evaluate/analyze` | ✅ Done | Post-run summary report |
| Tests (schedulers) | ⚠️ Partial | Hybrid test marked xfail |

---

## Quick Start

```bash
# Local smoke test (no GPU required)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.run_benchmark --config configs/smoke.yaml --scheduler baseline --monitor simulated
python -m evaluation.analyze runs/<run-id>

# Real GPU run
export CUDA_DEVICE_ORDER=PCI_BUS_ID
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m experiments.run_benchmark --config configs/default.yaml --scheduler baseline
```

See `docs/running_baseline_on_cluster.md` for MIT cluster (Satori/Engaging/Supercloud) setup.

---

## Workstream Responsibilities

| Workstream | Area | Key File(s) |
|-----------|------|-------------|
| WS1 | Schedulers | `schedulers/hybrid.py` |
| WS2 | Evaluation | `evaluation/`, `monitoring/` |
| WS3 | Workloads | `workloads/ptq_inference.py`, `workloads/training.py` |
| Shared | Framework | `framework/`, `configs/` |
