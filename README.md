# Thermal-/Compute-Aware GPU Scheduler

6.S984 semester project. Compares a naive **baseline** scheduler against a
**hybrid** scheduler that weighs GPU utilization and temperature together, on a
mixed workload of post-training-quantization (PTQ) inference and training jobs.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run with the simulated thermal model (no GPU required)
python -m experiments.run_benchmark \
    --config configs/default.yaml \
    --scheduler baseline \
    --monitor simulated

# Summarize a run
python -m evaluation.analyze runs/<run-id>
```

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
