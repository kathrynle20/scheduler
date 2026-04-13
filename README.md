# Thermal-/Compute-Aware GPU Scheduler

6.S984 semester project. Compares a naive **baseline** scheduler against a
**hybrid** scheduler that weighs GPU utilization and temperature together, on a
mixed workload of post-training-quantization (PTQ) inference and training jobs.

## Quickstart (no GPU, simulated)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.run_benchmark \
    --config configs/smoke.yaml \
    --scheduler baseline \
    --monitor simulated
python -m evaluation.analyze runs/<run-id>
```

## Running on the MIT GPU cluster

```bash
# Install torch matching the node's CUDA version (example for CUDA 12.1):
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Keep NVML indices and torch indices aligned (important!):
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Edit configs/default.yaml:
#   gpus.ids: <list of indices visible to your allocation>
#   gpus.neighbors: <physical adjacency; ask cluster docs or infer from topology>
#   monitor.backend: nvml

python -m experiments.run_benchmark --config configs/default.yaml --scheduler baseline
```

The workloads auto-detect CUDA: if torch + GPU are available they run real
matmuls / a small MLP training loop; otherwise they fall back to sleep-based
stubs so the scaffold still works for dev.

## Architecture

```
 ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
 │  workloads/  │      │ schedulers/  │      │ monitoring/  │
 │  (ws 3)      │──job─▶  (ws 1)      │◀─────│  nvml / sim  │
 └──────────────┘      └──────┬───────┘      └──────┬───────┘
                              │ place(job, states)   │ sample()
                              ▼                      ▼
                       ┌──────────────────────────────┐
                       │      framework/runner        │
                       └─────────────┬────────────────┘
                                     │ time-series CSV
                                     ▼
                              ┌──────────────┐
                              │ evaluation/  │
                              │  (ws 2)      │
                              └──────────────┘
```

## Workstream ownership

| Area          | Owner       | Key files                                            |
|---------------|-------------|------------------------------------------------------|
| Schedulers    | Workstream 1 | `schedulers/`, `framework/runner.py`                |
| Evaluation    | Workstream 2 | `evaluation/`, `monitoring/gpu_monitor.py` sampling |
| Workloads     | Workstream 3 | `workloads/`                                        |

Shared primitives (`framework/`, `monitoring/`) should be edited via PR review
so interfaces stay stable.

## Layout

See `configs/default.yaml` for tunable parameters (GPU list, neighbor map,
hybrid weights, temperature limits, workload mix).
