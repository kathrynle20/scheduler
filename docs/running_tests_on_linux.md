# Running the Scheduler Tests on a Generic Linux Machine

This guide is for someone running the scheduler benchmarks on **their own Linux
box** (workstation, lab server, single-node cloud VM, etc.) where they have
**direct shell access** and **all GPUs are theirs**.

It does **not** assume:

- SLURM (no `salloc` / `sbatch` / partitions)
- the `module` / Lmod system (no `module load miniforge`)
- conda; a plain Python virtual environment is sufficient

If you are running on the **MIT Engaging** cluster, see
[`docs/running_tests_on_engaging.md`](running_tests_on_engaging.md) instead.

---

## 0. Prerequisites

On the target Linux machine you need:

- **NVIDIA GPU(s)** with the **driver** installed (verify with `nvidia-smi`)
- **CUDA-capable PyTorch** matching that driver (we install via pip below)
- **Python 3.10+** and `python3-venv`
- **Git**
- ~5 GB of free disk for the venv, runs, and CSV output

Quick sanity checks:

```bash
nvidia-smi               # lists every GPU + driver/CUDA version
python3 --version        # 3.10+
git --version
```

`nvidia-smi` printing your GPU(s) is the most important one — without it,
benchmarks will fall back to the CPU **stub** path (small `time.sleep` per
"request") and the numbers will not be meaningful.

---

## 1. Clone and Enter the Repo

```bash
git clone <repo-url> scheduler
cd scheduler
```

The rest of this doc assumes your shell is **inside the repo root**.

---

## 2. Create a Python Virtual Environment (one-time)

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Then install **PyTorch** matching the CUDA version your driver supports.
Look at the top-right of `nvidia-smi` for "CUDA Version: XX.Y" and pick the
matching wheel index from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
For most recent NVIDIA drivers `cu121` works:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is wired up correctly **before running any benchmark**:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# Expect: True <N>   where <N> = number of GPUs you want to use
```

If this prints `False` or `0`, fix it first (wrong wheel, missing driver,
or wrong NVIDIA toolkit) — running benchmarks under the stub path tells you
nothing about the scheduler.

> Whenever you open a fresh shell, re-enter the venv with
> `source .venv/bin/activate` from the repo root.

---

## 3. Smoke Test (no GPU required)

Confirm the codebase imports and the framework runs end-to-end with the
**simulated** monitor. This finishes in seconds and does not touch CUDA:

```bash
python -m experiments.run_benchmark \
  --config configs/smoke.yaml \
  --scheduler baseline \
  --monitor simulated
```

You should see per-job log lines and a final `SummaryReport`. If this fails,
the install is broken — fix that before continuing.

A work-stealing variant of the smoke test:

```bash
python -m experiments.run_benchmark \
  --config configs/work_stealing_smoke.yaml \
  --monitor simulated
```

---

## 4. Run the Pytest Suite (optional)

```bash
pytest -v --tb=short
```

All tests should pass. The codebase's workloads have CPU-side stubs, so the
suite runs without a GPU. To run only scheduler/workload tests:

```bash
pytest -v tests/test_schedulers.py tests/test_workloads.py
```

---

## 5. Choose Which GPUs the Benchmarks Use

The runner uses GPUs in **0-based index order** (GPU 0, GPU 1, …). Two ways
to control which physical devices are exposed:

1. **`CUDA_VISIBLE_DEVICES`** (preferred): a comma list of the **physical**
   GPU ids you want. Inside the process they are renumbered 0..N-1.

   ```bash
   # Use only physical GPUs 2 and 3 of an 8-GPU box, as logical 0 and 1:
   export CUDA_VISIBLE_DEVICES=2,3
   ```

2. **`--num-gpus N`** (CLI flag): tells the benchmark to use logical ids
   `0..N-1` as workers. Combine this with `CUDA_VISIBLE_DEVICES` if you want
   to pick *which* physical GPUs become 0..N-1.

Always also set:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

This makes NVML and PyTorch agree on GPU ordering — without it, the NVML
monitor can read util/temp from a different physical GPU than the one
PyTorch is running on, and your timeseries CSV will be misleading.

If you want every GPU on the host, just leave `CUDA_VISIBLE_DEVICES` unset
and pass `--num-gpus <count>` matching `nvidia-smi -L`.

---

## 6. Single-Experiment Runs

Every benchmark goes through the same entry point:

```bash
python -m experiments.run_benchmark --config <YAML> [flags]
```

### 6.1. CLI flags (full reference)

| Flag | Default | What it does |
|------|---------|--------------|
| `--config <path>` | `configs/default.yaml` | YAML defining workload mix, arrival rate, scheduler block |
| `--scheduler {baseline,hybrid,work_stealing}` | from YAML | Override `scheduler.name` |
| `--monitor {nvml,simulated}` | from YAML | `nvml` reads real GPUs; `simulated` runs a thermal model |
| `--num-gpus N` | from YAML | Use logical GPUs `0..N-1`; auto-builds `worker_gpus` and a fully-connected neighbor graph |
| `--arrival-rate <hz>` | from YAML | Override `arrival_rate_hz` (jobs/sec across all GPUs) |
| `-v` / `--verbose` | off | DEBUG logging — per-job placement decisions, every steal |

`--num-gpus N` does **not** allocate real devices; it only changes which logical
ids the runner schedules onto. The physical GPUs are whatever
`CUDA_VISIBLE_DEVICES` exposes (or all GPUs on the box).

### 6.2. The 3 baseline / work-stealing pairs

The current architecture compares **JSQ-only baseline** vs **JSQ + stealing**
through the same `WorkStealingRunner`. The `*_*` configs without `ws_` set
`steal_threshold = 99999` so stealing never fires; the `ws_*` configs use a
small threshold (1–2) so it does. Detailed rationale lives in
[`docs/architecture_decision.md`](architecture_decision.md) and
[`docs/expected_findings.md`](expected_findings.md).

| Pair | Baseline config | Work-stealing config |
|------|------------------|------------------------|
| **PTQ inference (100 jobs)** | `configs/ptq_100.yaml` | `configs/ws_ptq_100.yaml` |
| **Training (50 × 500 steps)** | `configs/train_50.yaml` | `configs/ws_train_50.yaml` |
| **Mixed (100 PTQ + 50 training)** | `configs/mixed_100_50.yaml` | `configs/ws_mixed_100_50.yaml` |

Run a single experiment, e.g. PTQ baseline on **2 GPUs**:

```bash
python -m experiments.run_benchmark \
  --config configs/ptq_100.yaml \
  --monitor nvml \
  --num-gpus 2 \
  -v
```

To run the **same** workload with stealing:

```bash
python -m experiments.run_benchmark \
  --config configs/ws_ptq_100.yaml \
  --monitor nvml \
  --num-gpus 2 \
  -v
```

### 6.4. Scaling to 4 / 8 GPUs at ~70% utilization

Per-GPU utilization is roughly:

```
target_util ≈ (arrival_rate / num_gpus) × job_duration_s
```

For the **PTQ** workload (`num_requests: 20`, `matrix_size: 16384`,
~0.94 s per job on L40S), targeting **~70%** mean per-GPU utilization gives:

| GPUs | `--num-gpus` | `--arrival-rate` (hz) | Predicted per-GPU util |
|------|--------------|------------------------|--------------------------|
| 2    | `2`          | `1.5`                  | ~70% |
| 4    | `4`          | `3.0`                  | ~70% |
| 8    | `8`          | `6.0`                  | ~70% |

(The shipped YAMLs use **1.4 hz** on 2 GPUs which is ~66%. Bumping to 1.5/3.0/6.0
hits the ~70% mark cleanly. Different GPUs will give slightly different
per-matmul times — re-tune if you see the printed `PTQ avg request time`
drift away from ~47 ms.)

#### Single experiment

**4 GPUs, PTQ baseline (no-steal):**

```bash
python -m experiments.run_benchmark \
  --config configs/ptq_100.yaml \
  --monitor nvml \
  --num-gpus 4 \
  --arrival-rate 3.0 \
  -v
```

**4 GPUs, PTQ work-stealing:**

```bash
python -m experiments.run_benchmark \
  --config configs/ws_ptq_100.yaml \
  --monitor nvml \
  --num-gpus 4 \
  --arrival-rate 3.0 \
  -v
```

**8 GPUs, PTQ baseline:**

```bash
python -m experiments.run_benchmark \
  --config configs/ptq_100.yaml \
  --monitor nvml \
  --num-gpus 8 \
  --arrival-rate 6.0 \
  -v
```

**8 GPUs, PTQ work-stealing:**

```bash
python -m experiments.run_benchmark \
  --config configs/ws_ptq_100.yaml \
  --monitor nvml \
  --num-gpus 8 \
  --arrival-rate 6.0 \
  -v
```

Swap the config path for `mixed_100_50.yaml` / `ws_mixed_100_50.yaml` to run
the mixed PTQ + training experiment at the same per-GPU load (mixed jobs are
also dominated by ~0.94–1.07 s durations, so the same arrival rate is a
good first cut).

The **training-only** configs are designed to keep both GPUs near-saturated
even at the default 2.0 hz; for 4 / 8 GPUs you can keep `2.0 hz` on 2 GPUs
or pass `--arrival-rate 4.0` / `8.0` if you want similarly heavy load on the
larger fleets — but training experiments are deliberately not where work
stealing helps, so tail-latency comparisons there are less interesting.

#### Full suite at ~70% util

`run_suite.sh` accepts `[N_TRIALS] [FILTER] [NUM_GPUS] [ARRIVAL_RATE]`, so:

```bash
# 5 trials, all 6 configs, 4 GPUs at ~70% util
bash scripts/run_suite.sh 5 "" 4 3.0

# 5 trials, all 6 configs, 8 GPUs at ~70% util
bash scripts/run_suite.sh 5 "" 8 6.0

# Just the PTQ pair, 5 trials, 8 GPUs at ~70% util
bash scripts/run_suite.sh 5 ptq 8 6.0
```

Then aggregate as in §7.4:

```bash
LATEST=$(ls -td results/suite-* | head -1)
python -m scripts.aggregate_results "$LATEST/manifest.csv"
```

> If your machine has more physical GPUs than you want the run to use
> (e.g. 8 installed, you only want to test on 4), pin them with
> `export CUDA_VISIBLE_DEVICES=0,1,2,3` before launching — `--num-gpus` only
> chooses among GPUs the process can already see.

### 6.3. Where output goes

Each run creates a fresh directory under `./runs/`:

```
runs/run-YYYYMMDD-HHMMSS-XXXXXX/
├── results.csv      # per-job: id, type, gpu, timestamps, latencies
├── timeseries.csv   # 10 Hz GPU util/temp samples
└── metadata.json    # work-stealing only: steal count, queue depths
```

A `SummaryReport` is also printed to stdout at the end of each run.

To re-print a saved run later:

```bash
python -m evaluation.analyze runs/<run-id>
```

---

## 7. Multi-Trial Suite Runs

For a presentation-ready comparison (mean ± std across trials), use the
suite runner.

### 7.1. Full suite (3 experiments × 2 schedulers × N trials)

```bash
bash scripts/run_suite.sh                 # 3 trials per config, GPU count from each YAML
bash scripts/run_suite.sh 5               # 5 trials per config
bash scripts/run_suite.sh 3 "" 4          # 3 trials, force 4 GPUs across all 6 configs
bash scripts/run_suite.sh 3 "" 4 2.8      # also override arrival_rate (2.8 hz)
```

Positional args: `[N_TRIALS] [FILTER] [NUM_GPUS] [ARRIVAL_RATE]` — empty
strings (`""`) skip a positional flag.

### 7.2. Filtering (just one experiment pair)

`FILTER` is a substring matched against the config base name. Examples:

```bash
bash scripts/run_suite.sh 5 ptq           # both ptq_100 and ws_ptq_100, 5 trials each
bash scripts/run_suite.sh 5 ws            # all 3 work-stealing configs
bash scripts/run_suite.sh 5 mixed         # mixed pair only
bash scripts/run_suite.sh 5 train         # training pair only
```

### 7.3. What the suite produces

A new directory under `results/`:

```
results/suite-YYYYMMDD-HHMMSS/
├── manifest.csv      # config,trial,run_dir for every run
└── suite.log         # combined stdout/stderr
```

### 7.4. Aggregating into a comparison table

After the suite finishes, build a markdown comparison from `manifest.csv`:

```bash
python -m scripts.aggregate_results results/suite-<timestamp>/manifest.csv
```

This adds three more files in the same suite dir:

- **`comparison.md`** — markdown table per experiment, baseline vs. work
  stealing, with `mean ± std` and a Δ % column. **Drop this into a deck.**
- **`summary.csv`** — aggregated mean/std per config
- **`raw_metrics.csv`** — per-trial metrics (for plotting)

`comparison.md` looks roughly like:

```
## Experiment 1: PTQ inference (100 jobs)

| Metric | Baseline | Work Stealing | Δ |
|--------|----------|---------------|---|
| Avg job latency (ms) | 1676.9 ± 45.2 | 1182.3 ± 38.1 | ↓ 29.5% **better** |
| p99 job latency (ms) | 3699.2 ± 120.5 | 1980.4 ± 95.1 | ↓ 46.5% **better** |
| Avg queue wait (ms)  | 762.8 ± 30.4  | 220.1 ± 18.3  | ↓ 71.1% **better** |
| Avg request time (ms)| 45.70 ± 0.05  | 45.69 ± 0.04  | ≈ same |
| Steal count          | —             | 22.3 ± 4.1    | — |
```

---

## 8. Long Runs in the Background

Without SLURM, the simplest "submit and walk away" patterns are:

**`nohup` (survives logout):**

```bash
mkdir -p logs
nohup bash scripts/run_suite.sh 5 "" 2 \
  > logs/suite_$(date +%Y%m%d-%H%M%S).log 2>&1 &
echo "PID $!"
```

**`tmux` / `screen` (interactive but detachable):**

```bash
tmux new -s suite                       # start a session
bash scripts/run_suite.sh 5 "" 2        # run inside it
# Detach: Ctrl-b d
# Re-attach later: tmux attach -t suite
```

Once started, watch progress:

```bash
tail -f logs/suite_*.log         # nohup case
nvidia-smi -l 2                  # live GPU usage every 2s
```

---

## 9. Inspecting Outputs

```bash
# List all runs newest-first
ls -td runs/run-* | head -10

# Latest run only
RUN=$(ls -td runs/run-* | head -1)

head -5 "$RUN/results.csv"      # first 5 finished jobs
head -5 "$RUN/timeseries.csv"   # first 5 monitor samples
cat "$RUN/metadata.json"        # only present for work-stealing runs

python -m evaluation.analyze "$RUN"   # full summary (latency pctiles, util, etc.)
```

To diff baseline vs. work-stealing reports for a given experiment:

```bash
python -m evaluation.analyze runs/<baseline-id> > /tmp/base.txt
python -m evaluation.analyze runs/<ws-id>       > /tmp/ws.txt
diff /tmp/base.txt /tmp/ws.txt
```

---

## 10. Quick Cheat Sheet

```bash
# One-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Every shell
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# optional: export CUDA_VISIBLE_DEVICES=0,1

# Smoke
python -m experiments.run_benchmark --config configs/smoke.yaml --monitor simulated

# One real-GPU experiment (PTQ baseline, 2 GPUs, ~70% util)
python -m experiments.run_benchmark \
  --config configs/ptq_100.yaml --monitor nvml --num-gpus 2 --arrival-rate 1.5 -v

# Same on 4 GPUs at ~70% util
python -m experiments.run_benchmark \
  --config configs/ptq_100.yaml --monitor nvml --num-gpus 4 --arrival-rate 3.0 -v

# Same on 8 GPUs at ~70% util
python -m experiments.run_benchmark \
  --config configs/ptq_100.yaml --monitor nvml --num-gpus 8 --arrival-rate 6.0 -v

# Full suite, 5 trials, at ~70% util
bash scripts/run_suite.sh 5 "" 2 1.5    # 2 GPUs
bash scripts/run_suite.sh 5 "" 4 3.0    # 4 GPUs
bash scripts/run_suite.sh 5 "" 8 6.0    # 8 GPUs

# Aggregate the latest suite
LATEST=$(ls -td results/suite-* | head -1)
python -m scripts.aggregate_results "$LATEST/manifest.csv"
```

---

## 11. Troubleshooting

| Symptom | Likely cause / fix |
|---------|---------------------|
| `torch.cuda.is_available()` is `False` | CPU-only torch wheel installed — reinstall the CUDA-matched build (Step 2) |
| `nvidia-smi: command not found` | NVIDIA driver not installed; install the driver matching your GPU first |
| `pynvml.NVMLError_LibraryNotFound` | Driver missing or NVML library (`libnvidia-ml.so`) not on the linker path; reinstall the driver |
| Util / temp logged for the wrong GPU | Forgot `export CUDA_DEVICE_ORDER=PCI_BUS_ID` — set it and re-run |
| PTQ avg request time < 1 ms | Stub path is running — CUDA isn't really being used. Check `torch.cuda.is_available()` |
| Run uses fewer GPUs than expected | Set/unset `CUDA_VISIBLE_DEVICES`; match `--num-gpus` to what `python -c "import torch; print(torch.cuda.device_count())"` reports |
| `pip install --user` packages shadow venv | `export PYTHONNOUSERSITE=True` before running |
| Suite reports "could not detect new run dir" | A run crashed; inspect `results/suite-*/suite.log` for the failing config |
| Latency way higher than expected | Another process on the same GPU(s) — `nvidia-smi` and `fuser -v /dev/nvidia*` to find culprits |
| `ValueError: unknown workload type in config` | YAML has a `type:` value other than `ptq` or `training` — fix the config |

---

## 12. Where to Look Next

- [`docs/expected_findings.md`](expected_findings.md) — what each experiment
  is supposed to show, and why.
- [`docs/architecture_decision.md`](architecture_decision.md) — why baseline
  and work stealing share the same runner now.
- [`docs/work_stealing_postmortem.md`](work_stealing_postmortem.md) — earlier
  failure analysis that motivated the current JSQ-based design.
- [`docs/codebase_overview.md`](codebase_overview.md) — module-by-module map
  of `framework/`, `schedulers/`, `workloads/`, `evaluation/`.
