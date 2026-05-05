# Running the Scheduler Tests on MIT Engaging (ORCD)

Step-by-step guide for running the project's pytest suite on the Engaging GPU
cluster. The docs here were pulled from [https://orcd-docs.mit.edu/](https://orcd-docs.mit.edu/).

---

## 0. Prerequisites

- MIT Kerberos account with Engaging access
- Duo 2FA enrolled
- Your code in `~/scheduler` on the login node (clone or `scp` it)

---

## 1. SSH In

```bash
ssh <kerberos>@orcd-login.mit.edu
```

Authenticate with your MIT password + Duo. To avoid re-doing 2FA every session,
add a control channel to `~/.ssh/config`:

```
Host orcd
    HostName orcd-login.mit.edu
    User <kerberos>
    ControlMaster auto
    ControlPath ~/.ssh/control/%r@%h:%p
    ControlPersist 600
```

Then `ssh orcd` reuses the socket for 10 minutes.

---

## 2. Clone the Repo (one-time)

```bash
cd ~
git clone <your-repo-url> scheduler
cd scheduler
```

Keep your code in `$HOME` — it's backed up. Use
`~/orcd/scratch/scheduler-runs/` for run output if you generate a lot of CSVs.

---

## 3. Set Up the Python Environment (one-time)

Do this on the **login node** — no GPU needed for the install step.

```bash
module load miniforge

python -m venv ~/scheduler-env
source ~/scheduler-env/bin/activate

pip install -r requirements.txt
# For real-GPU runs, install torch matching the cluster's CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu121

deactivate
```

> **Never** use `pip install --user` inside a venv — packages land in `~/.local`
> and conflict across envs. **Never** run `conda init bash`.

---

## 4. Running Tests

### 4a. CPU-only / simulated tests (fastest — use `mit_quicktest`)

The full pytest suite works without a GPU because all workloads have stub paths
and the simulated monitor requires no CUDA. The `mit_quicktest` partition gives
you a node in seconds, with a 15-minute limit.

```bash
# Interactive:
salloc -p mit_quicktest -c 4 --mem=8GB -t 00:15:00

# On the compute node:
module purge && module load miniforge
source ~/scheduler-env/bin/activate
cd ~/scheduler
pytest -v
```

Or as a one-liner submitted as a batch job:

```bash
cat > run_tests_cpu.sh << 'EOF'
#!/bin/bash
#SBATCH -p mit_quicktest
#SBATCH -c 4
#SBATCH --mem=8GB
#SBATCH -t 00:15:00
#SBATCH -o logs/test_cpu_%j.out
#SBATCH -e logs/test_cpu_%j.err

module purge
module load miniforge
source ~/scheduler-env/bin/activate
cd ~/scheduler

export CUDA_DEVICE_ORDER=PCI_BUS_ID
pytest -v --tb=short
EOF

mkdir -p logs
sbatch run_tests_cpu.sh
```

### 4b. Real GPU tests (use `mit_normal_gpu`)

Required for any test that sets `@pytest.mark.gpu` or needs actual CUDA/torch
timings. The L40S GPUs (44 GB VRAM) are plentiful and fast to allocate.

**Interactive (for development and debugging):**

```bash
salloc -p mit_normal_gpu --gres=gpu:l40s:1 -c 4 --mem=20GB -t 01:00:00

# On the compute node:
module purge && module load miniforge
source ~/scheduler-env/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd ~/scheduler
pytest -v
```

**Batch (unattended CI-style):**

```bash
cat > run_tests_gpu.sh << 'EOF'
#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 4
#SBATCH --mem=20GB
#SBATCH -t 01:00:00
#SBATCH -o logs/test_gpu_%j.out
#SBATCH -e logs/test_gpu_%j.err
#SBATCH --job-name=sched-tests

module purge
module load miniforge
source ~/scheduler-env/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd ~/scheduler
pytest -v --tb=short 2>&1
EOF

mkdir -p logs
sbatch run_tests_gpu.sh
```

Monitor and inspect:

```bash
squeue --me                          # watch status (PD=pending, R=running)
tail -f logs/test_gpu_<JOBID>.out   # stream output
scancel <JOBID>                      # cancel if needed
```

---

## 5. Interactive Attended Workflow (Full Step-by-Step)

Use this when you want to watch output in real time, debug, or confirm things work before submitting unattended batch jobs. Every command below is run in the same terminal session — the prompt changes when SLURM drops you onto a compute node.

### Step 1 — SSH into the login node

```bash
ssh <kerberos>@orcd-login.mit.edu
```

Your prompt will look like `[you@orcd-login001 ~]$`. You are **not** on a GPU node yet. Do not run Python or nvidia-smi here.

---

### Step 2 — Reserve a GPU node interactively

```bash
salloc -p mit_normal_gpu \
  --gres=gpu:l40s:2 \
  -N 1 \
  -c 8 \
  --mem=32GB \
  -t 01:00:00
```

Wait for SLURM to respond. This can take anywhere from a few seconds to a few minutes depending on cluster load:

```
salloc: Pending job allocation 12345678
salloc: job 12345678 queued and waiting for resources
salloc: job 12345678 has been allocated resources
salloc: Granted job allocation 12345678
salloc: Nodes node0042 have been allocated to your job
```

Your prompt will change to something like `[you@node0042 ~]$`. You are now on a compute node with 2 real GPUs.

> **If the wait is long:** Check what's free with `sinfo -p mit_normal_gpu -O Nodes,Gres,StateLong` or try `--gres=gpu:1` (any GPU type) instead of specifying `l40s`.

---

### Step 3 — Verify the GPUs are visible

```bash
nvidia-smi -L
```

Expected output:
```
GPU 0: NVIDIA L40S (UUID: GPU-...)
GPU 1: NVIDIA L40S (UUID: GPU-...)
```

If you see nothing or get an error, SLURM gave you a non-GPU node — cancel with `exit` and re-run Step 2.

---

### Step 4 — Load the environment

```bash
module purge
module load miniforge
source ~/scheduler-env/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd ~/scheduler
```

Confirm Python can see the GPUs:

```bash
python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
# Expected: 2 GPUs
```

---

### Step 5 — Run the simulated smoke test first (sanity check)

Always do this before touching real GPUs — it catches import errors and config issues instantly with no GPU needed:

```bash
python -m experiments.run_benchmark \
  --config configs/smoke.yaml \
  --scheduler baseline \
  --monitor simulated
```

You should see per-job log lines and a summary report printed to the terminal. If this fails, fix the error before proceeding.

---

### Step 6 — Run the pytest suite

```bash
pytest -v --tb=short
```

All tests should pass. The one known `xfail` is `test_hybrid_prefers_cool_idle_gpu` — that is expected until `HybridScheduler` is implemented.

To run only scheduler-related tests:

```bash
pytest -v tests/test_schedulers.py tests/test_workloads.py
```

---

### Step 7 — Run all six experiments (baseline + work stealing × 3 setups)

Run each pair back-to-back so conditions are as similar as possible. Each run produces its own timestamped directory under `runs/`. The `-v` flag shows per-job scheduling decisions as they happen.

---

**Experiment 1 — 100 PTQ inference jobs**

```bash
# Baseline
python -m experiments.run_benchmark \
  --config configs/ptq_100.yaml \
  --monitor nvml \
  -v

# Work stealing
python -m experiments.run_benchmark \
  --config configs/ws_ptq_100.yaml \
  --monitor nvml \
  -v
```

---

**Experiment 2 — 50 training jobs (500 steps each)**

```bash
# Baseline
python -m experiments.run_benchmark \
  --config configs/train_50.yaml \
  --monitor nvml \
  -v

# Work stealing
python -m experiments.run_benchmark \
  --config configs/ws_train_50.yaml \
  --monitor nvml \
  -v
```

---

**Experiment 3 — 100 PTQ + 50 training jobs interleaved**

```bash
# Baseline
python -m experiments.run_benchmark \
  --config configs/mixed_100_50.yaml \
  --monitor nvml \
  -v

# Work stealing
python -m experiments.run_benchmark \
  --config configs/ws_mixed_100_50.yaml \
  --monitor nvml \
  -v
```

---

Sample output for a running experiment:

```
10:01:23 INFO  run_benchmark: built 100 jobs: ptq=100
10:01:23 INFO  runner: starting: 100 jobs, scheduler=BaselineScheduler, workers=[0, 1]
10:01:24 INFO  runner: [ptq-0] ptq mem=2048MB -> GPU 0 (util=0% temp=32.1C mem=0/44GB)
10:01:24 INFO  runner: [ptq-1] ptq mem=2048MB -> GPU 1 (util=5% temp=33.0C mem=0/44GB)
...
10:05:10 INFO  runner: run complete: placed=100 deferred=0
```

> **Expected runtimes per run (approximate, 2× L40S):**
> | Experiment | Jobs | Approx. wall time |
> |------------|------|-------------------|
> | PTQ only | 100 × 100 requests @ 16384² | ~4–5 min |
> | Training only | 50 × 500 steps | 20–40 min |
> | Mixed | 150 total | 30–50 min |
>
> **PTQ workload sizing note:** Each PTQ job runs 100 matmuls on a 16384×16384
> float16 tensor (~24ms/matmul on L40S → ~2.4s/job). Jobs arrive at 2/sec but
> complete at ~0.8/sec per GPU, so queues build up after the first few seconds.
> You should see GPU utilization >60% and steal count >0 for the work-stealing run.
> If latency is still <1ms, the stub path is running — check that
> `torch.cuda.is_available()` returns True on the compute node.

---

### Step 8 — Analyze and compare results

```bash
# List all runs chronologically
ls -td runs/run-* | head -12
```

Analyze and save each run's summary to a text file for side-by-side comparison:

```bash
mkdir -p results

# Experiment 1
python -m evaluation.analyze runs/<baseline-ptq-run-id>  > results/baseline_ptq.txt
python -m evaluation.analyze runs/<ws-ptq-run-id>        > results/ws_ptq.txt

# Experiment 2
python -m evaluation.analyze runs/<baseline-train-run-id> > results/baseline_train.txt
python -m evaluation.analyze runs/<ws-train-run-id>       > results/ws_train.txt

# Experiment 3
python -m evaluation.analyze runs/<baseline-mixed-run-id> > results/baseline_mixed.txt
python -m evaluation.analyze runs/<ws-mixed-run-id>       > results/ws_mixed.txt
```

Compare baseline vs. work stealing side by side:

```bash
diff results/baseline_ptq.txt results/ws_ptq.txt
diff results/baseline_train.txt results/ws_train.txt
diff results/baseline_mixed.txt results/ws_mixed.txt
```

Key numbers to look for in each report:
- **PTQ p50/p95/p99 latency** — work stealing should reduce tail latency
- **Training avg duration** — should be similar or better with work stealing
- **Per-GPU utilization balance** — work stealing should show lower variance across GPUs
- **Steal count** — confirms work stealing was actually triggered (work-stealing runs only)

---

### Step 9b — One-shot suite runner (recommended for presentation data)

For multiple trials and a presentation-ready comparison table, use the suite scripts instead of running each config manually.

**Run the full suite (3 trials per config, all 6 configs):**

```bash
bash scripts/run_suite.sh
```

**Just the PTQ pair, 5 trials each (fast, fits in 30 min):**

```bash
bash scripts/run_suite.sh 5 ptq
```

**Just the work-stealing configs (skip baselines):**

```bash
bash scripts/run_suite.sh 3 ws
```

The script writes everything to `results/suite-<timestamp>/`:
- `manifest.csv` — maps each (config, trial) to its `runs/run-...` directory
- `suite.log` — combined log output of all runs

**Generate the comparison report:**

```bash
python -m scripts.aggregate_results results/suite-<timestamp>/manifest.csv
```

This produces three files in the suite directory:
- `comparison.md` — markdown table per experiment, baseline vs. work stealing, with mean ± std and Δ % column. **Drop this directly into your presentation.**
- `summary.csv` — aggregated mean/std per config (for spreadsheets)
- `raw_metrics.csv` — per-trial metrics (for plotting in a notebook)

The `comparison.md` looks like:

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

**Submit the suite as a SLURM batch job (unattended overnight runs):**

```bash
# 3 trials of all 6 configs (uses up to 6 hours)
sbatch scripts/run_suite_sbatch.sh

# 5 trials, PTQ only
sbatch scripts/run_suite_sbatch.sh 5 ptq
```

The batch script auto-runs aggregation when finished. Monitor with `squeue --me` and `tail -f logs/suite_<JOBID>.out`.

---

### Step 10 — Exit the allocation

When you are done, release the GPUs back to the cluster:

```bash
exit
```

Your prompt returns to `[you@orcd-login001 ~]$`. SLURM automatically releases the reservation. If you just close the terminal without typing `exit`, the allocation will also eventually expire at the time limit you set (`-t 01:00:00`), but it's good practice to exit explicitly so others can use the GPUs sooner.

---

## 6. Running the Full Benchmark

### Smoke test (simulated, no GPU — always do this first)

```bash
# On any node or login node:
python -m experiments.run_benchmark \
  --config configs/smoke.yaml \
  --scheduler baseline \
  --monitor simulated

# Work-stealing smoke test:
python -m experiments.run_benchmark \
  --config configs/work_stealing_smoke.yaml \
  --monitor simulated
```

### Baseline vs. work-stealing on real GPUs

The scheduling loop runs entirely on the CPU — every allocated GPU is a
**worker**. All configs are set up so `worker_gpus` equals the full `ids` list,
so no GPU ever sits idle.

**Which config to use depends on how many GPUs SLURM gives you:**

| GPUs allocated | Config to use | Workers | Jobs |
|---|---|---|---|
| 2 (default quota) | `smoke.yaml` / `work_stealing_smoke.yaml` | 2 | 8 |
| 4 (requires quota increase or `mit_preemptable`) | `default.yaml` / `work_stealing.yaml` | 4 | 150 |

**Resource breakdown (2-GPU run):**

| Flag | Value | Why |
|------|-------|-----|
| `--gres=gpu:l40s:2` | 2 × L40S (44 GB VRAM each) | All 2 GPUs are workers; CPU handles scheduling |
| `-N 1` | 1 node | Both GPUs must share a node (single-process runner) |
| `-c 8` | 8 CPU cores | 4 per GPU; each worker thread + I/O headroom |
| `--mem=32GB` | 32 GB system RAM | ~16 GB per GPU slot |
| `-t 01:00:00` | 1 hour | 8 jobs at 5 Hz ≈ 2 s dispatch + ~60 s tail; 1 h is generous |

**Interactive (confirm GPUs work before submitting unattended):**

```bash
# --- Step 1: on the LOGIN node, request the allocation ---
salloc -p mit_normal_gpu \
  --gres=gpu:l40s:2 \
  -N 1 \
  -c 8 \
  --mem=32GB \
  -t 01:00:00
# Wait for: "salloc: Nodes node#### have been allocated to your job"
# Your prompt changes from [you@orcd-login...] to [you@node####...]
# nvidia-smi does NOT exist on the login node — only run it after this point.

# --- Step 2: now on the COMPUTE node ---
nvidia-smi -L   # should print GPU 0 and GPU 1

module purge && module load miniforge
source ~/scheduler-env/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd ~/scheduler

python -m experiments.run_benchmark \
  --config configs/smoke.yaml \
  --scheduler baseline \
  --monitor nvml
```

**Batch (unattended — submit both and compare):**

```bash
cat > run_baseline.sh << 'EOF'
#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH -t 01:00:00
#SBATCH -o logs/baseline_%j.out
#SBATCH --job-name=sched-baseline

module purge
module load miniforge
source ~/scheduler-env/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd ~/scheduler
python -m experiments.run_benchmark \
  --config configs/smoke.yaml \
  --scheduler baseline \
  --monitor nvml
EOF

cat > run_work_stealing.sh << 'EOF'
#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH -t 01:00:00
#SBATCH -o logs/ws_%j.out
#SBATCH --job-name=sched-ws

module purge
module load miniforge
source ~/scheduler-env/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd ~/scheduler
python -m experiments.run_benchmark \
  --config configs/work_stealing_smoke.yaml \
  --monitor nvml
EOF

mkdir -p logs
sbatch run_baseline.sh
sbatch run_work_stealing.sh
```

**Scaling to 4 GPUs for the full 150-job benchmark:**

The default `mit_normal_gpu` quota is 2 GPUs per job. For the full comparison
(100 PTQ + 50 training jobs across 4 workers) email
`orcd-help-engaging@mit.edu` to request a higher limit, or use
`mit_preemptable` (up to 4 GPUs, 48h, but preemptable — add `--requeue`):

```bash
#SBATCH -p mit_preemptable          # or mit_normal_gpu after quota increase
#SBATCH --gres=gpu:l40s:4
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH -t 02:00:00
#SBATCH --requeue                   # auto-restart if preempted

# configs to use with 4 GPUs:
#   configs/default.yaml        --scheduler baseline
#   configs/work_stealing.yaml  (scheduler name already set in file)
```

Inspect results once done:

```bash
python -m evaluation.analyze runs/<baseline-run-id>
python -m evaluation.analyze runs/<ws-run-id>
```

---

## 6. Partition Quick Reference


| Partition         | GPU types                                 | Max GPUs/job | Max time | Best for                    |
| ----------------- | ----------------------------------------- | ------------ | -------- | --------------------------- |
| `mit_quicktest`   | none                                      | —            | 15 min   | CPU tests, smoke runs       |
| `mit_normal_gpu`  | L40S (44 GB), H100 (79 GB), H200 (140 GB) | 2            | 6 hours  | Standard GPU benchmarks     |
| `mit_preemptable` | A100, L40S, H200, others                  | 4            | 48 hours | Long runs (add `--requeue`) |


**Default GPU type** when you write `--gres=gpu:1` is L40S. Prefer requesting
it explicitly (`--gres=gpu:l40s:1`) for deterministic allocation.

---

## 7. Checking What's Available

```bash
# GPU node availability
sinfo -p mit_normal_gpu -O Partition,Nodes,CPUs,Memory,Gres

# Your running / pending jobs
squeue --me

# Live GPU usage on your allocated node
nvtop

# Post-run resource accounting
sacct -j <JOBID> -o JobID,Elapsed,MaxRSS,ReqTRES%60
```

---

## 8. Inspecting Outputs

```bash
# Latest run
RUN=$(ls -td runs/run-* | head -1)

head -5 $RUN/results.csv     # per-job placements + latencies
head -5 $RUN/timeseries.csv  # 10 Hz GPU util/temp samples
cat $RUN/metadata.json       # steal count + job metadata (work-stealing only)

python -m evaluation.analyze $RUN  # full summary report
```

---

## 9. Saving Runs Off the Cluster

```bash
# From your laptop:
scp -r <kerberos>@orcd-login.mit.edu:scheduler/runs/run-YYYYMMDD-HHMMSS-xxxxxx \
    ./cluster-runs/
```

---

## 10. Troubleshooting


| Symptom                                                                                        | Fix                                                                                                             |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `sbatch: error: Batch job submission failed: Invalid account or account/partition combination` | Wait 15 min after creating the account, then retry                                                              |
| `torch.cuda.is_available()` returns False                                                      | Wrong torch wheel (CPU-only). Reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `nvidia-smi: command not found`                                                                | You're on the login node — NVIDIA drivers only exist on compute nodes. Run `salloc` first, wait for the prompt to change to `node####`, then retry |
| `pynvml.NVMLError_LibraryNotFound`                                                             | Same cause as above — you're on the login node. Run `salloc` first                                              |
| NVML GPU indices don't match torch                                                             | Missing `export CUDA_DEVICE_ORDER=PCI_BUS_ID`                                                                   |
| `pip install --user` packages shadowing venv                                                   | Run `export PYTHONNOUSERSITE=True` before pytest                                                                |
| Job immediately exits after `salloc`                                                           | Time limit probably exceeded — increase `-t`                                                                    |
| `module: command not found` in job script                                                      | Add `module purge` or ensure `module` is sourced. Source `/etc/profile.d/modules.sh` if needed                  |


---

## 11. Getting Help

- Email: `orcd-help-engaging@mit.edu`
- Docs: [https://orcd-docs.mit.edu/](https://orcd-docs.mit.edu/)
- Office hours listed on the docs site's support page

