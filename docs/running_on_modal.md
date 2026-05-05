# Running the Scheduler on Modal GPUs

[Modal](https://modal.com) is a serverless platform for running Python on
cloud GPUs. The repo ships with `modal_app.py` so you can run the same
benchmarks (`experiments.run_benchmark`, `scripts/run_suite.sh`) on a remote
GPU without provisioning a VM yourself.

## 1. One-time setup (laptop)

```bash
pip install modal
modal setup        # opens a browser to authenticate
```

A Modal account is required (free tier includes some monthly GPU credit; the
suite runs use real GPU time and will be billed once that's exhausted).

## 2. Smoke test (no GPU)

Verifies image build + repo mount + simulated monitor end-to-end.

```bash
modal run modal_app.py::smoke
```

The first call builds the CUDA + torch image (a few minutes); subsequent calls
reuse the cached image.

## 3. CUDA sanity check (1 GPU)

Confirms `torch.cuda.is_available()` and `nvidia-smi` work inside the
container.

```bash
modal run modal_app.py::gpu_check
```

## 4. Single experiment

PTQ baseline on 2x L40S at ~70% per-GPU utilization (matches
`docs/running_tests_on_linux.md` §6.4):

```bash
modal run modal_app.py::experiment \
  --config configs/ptq_100.yaml \
  --num-gpus 2 --arrival-rate 1.5
```

PTQ work-stealing variant:

```bash
modal run modal_app.py::experiment \
  --config configs/ws_ptq_100.yaml \
  --num-gpus 2 --arrival-rate 1.5
```

Args mirror `experiments.run_benchmark`'s CLI:

| Modal arg          | Maps to                       |
|--------------------|-------------------------------|
| `--config`         | `--config`                    |
| `--scheduler`      | `--scheduler` (empty = YAML)  |
| `--monitor`        | `--monitor` (default `nvml`)  |
| `--num-gpus`       | `--num-gpus`                  |
| `--arrival-rate`   | `--arrival-rate`              |
| `--verbose`        | `-v` (default `True`)         |

## 5. Full suite (multi-trial)

5 trials of all 6 configs, 4 GPUs, ~70% util — matches
`scripts/run_suite.sh 5 "" 4 3.0`:

```bash
modal run modal_app.py::suite --trials 5 --num-gpus 4 --arrival-rate 3.0
```

Filter to a single experiment pair (e.g. PTQ only):

```bash
modal run modal_app.py::suite --trials 5 --filter ptq --num-gpus 4 --arrival-rate 3.0
```

`aggregate=True` (default) runs `scripts.aggregate_results` against the new
suite manifest so `comparison.md` is generated and persisted.

## 6. Pulling outputs back

Outputs live in a Modal Volume named `scheduler-runs`. List what's there:

```bash
modal run modal_app.py::list_runs
modal volume ls scheduler-runs runs
modal volume ls scheduler-runs results
```

Pull a specific run or suite:

```bash
modal volume get scheduler-runs runs/run-YYYYMMDD-HHMMSS-XXXXXX ./modal-runs
modal volume get scheduler-runs results/suite-YYYYMMDD-HHMMSS    ./modal-results
```

Or pull everything:

```bash
modal volume get scheduler-runs runs    ./modal-runs    --recursive
modal volume get scheduler-runs results ./modal-results --recursive
```

Then re-print a saved run locally:

```bash
python -m evaluation.analyze ./modal-runs/run-YYYYMMDD-HHMMSS-XXXXXX
```

## 7. Changing GPU type or count

The physical GPU type / count is fixed at deploy time on the function
decorator. Edit the `GPU_EXPERIMENT` / `GPU_SUITE` constants near the top of
`modal_app.py`:

```python
GPU_EXPERIMENT = "L40S:2"   # default
# GPU_EXPERIMENT = "H100:8"
# GPU_EXPERIMENT = "A100-80GB:4"
```

Then keep `--num-gpus` (logical) <= the physical count, and scale
`--arrival-rate` linearly with `--num-gpus` to hold per-GPU utilization
constant: `target_util ≈ (arrival_rate / num_gpus) × job_duration_s`.
For PTQ (~0.94 s/job on L40S), 1.5 hz on 2 GPUs ≈ 3.0 hz on 4 GPUs ≈
6.0 hz on 8 GPUs at ~70% util.

Modal GPU types: L4, L40S, A10, A100-40GB, A100-80GB, H100, H200, B200.
See <https://modal.com/docs/guide/gpu>.

## 8. Iteration loop

`modal_app.py` mounts the local repo via `add_local_dir`, so editing
`schedulers/`, `framework/`, etc. and re-running `modal run` picks up the
change without rebuilding the image. Only changes to `requirements.txt` /
the `pip_install` lines / the CUDA base image trigger a rebuild.

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `torch.cuda.is_available()` is `False` in `gpu_check` | Image didn't pick up the cu121 wheel — rebuild by bumping any `pip_install` line, or check that the function has a `gpu=` arg |
| Job CSVs report PTQ < 1 ms | CPU-stub path (CUDA not visible) — see above |
| `modal run` times out | Bump `timeout=` on the function decorator (currently 1 h for `experiment`, 4 h for `suite`) |
| Suite reports "could not detect new run dir" | A run crashed; pull `results/suite-*/suite.log` to inspect |
| Volume `runs/` keeps growing | `_persist()` only adds new dirs; clean up with `modal volume rm scheduler-runs runs/<run-id>` |

## 10. Deleting the volume

```bash
modal volume delete scheduler-runs       # nukes all persisted runs/results
```
