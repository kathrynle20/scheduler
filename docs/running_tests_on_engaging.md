# Running the Scheduler Tests on MIT Engaging (ORCD)

Step-by-step guide for running the project's pytest suite on the Engaging GPU
cluster. The docs here were pulled from https://orcd-docs.mit.edu/.

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

## 5. Running the Full Benchmark

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

Submit both runs and compare the output CSVs afterward:

```bash
cat > run_baseline.sh << 'EOF'
#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:4
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH -t 02:00:00
#SBATCH -o logs/baseline_%j.out
#SBATCH --job-name=sched-baseline

module purge
module load miniforge
source ~/scheduler-env/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd ~/scheduler
python -m experiments.run_benchmark \
  --config configs/default.yaml \
  --scheduler baseline \
  --monitor nvml
EOF

cat > run_work_stealing.sh << 'EOF'
#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:4
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH -t 02:00:00
#SBATCH -o logs/ws_%j.out
#SBATCH --job-name=sched-ws

module purge
module load miniforge
source ~/scheduler-env/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd ~/scheduler
python -m experiments.run_benchmark \
  --config configs/work_stealing.yaml \
  --monitor nvml
EOF

mkdir -p logs
sbatch run_baseline.sh
sbatch run_work_stealing.sh
```

Inspect results once done:

```bash
python -m evaluation.analyze runs/<baseline-run-id>
python -m evaluation.analyze runs/<ws-run-id>
```

---

## 6. Partition Quick Reference

| Partition | GPU types | Max GPUs/job | Max time | Best for |
|-----------|-----------|--------------|----------|----------|
| `mit_quicktest` | none | — | 15 min | CPU tests, smoke runs |
| `mit_normal_gpu` | L40S (44 GB), H100 (79 GB), H200 (140 GB) | 2 | 6 hours | Standard GPU benchmarks |
| `mit_preemptable` | A100, L40S, H200, others | 4 | 48 hours | Long runs (add `--requeue`) |

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

| Symptom | Fix |
|---------|-----|
| `sbatch: error: Batch job submission failed: Invalid account or account/partition combination` | Wait 15 min after creating the account, then retry |
| `torch.cuda.is_available()` returns False | Wrong torch wheel (CPU-only). Reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `pynvml.NVMLError_LibraryNotFound` | You're on the login node, not a compute node. Run `salloc` first |
| NVML GPU indices don't match torch | Missing `export CUDA_DEVICE_ORDER=PCI_BUS_ID` |
| `pip install --user` packages shadowing venv | Run `export PYTHONNOUSERSITE=True` before pytest |
| Job immediately exits after `salloc` | Time limit probably exceeded — increase `-t` |
| `module: command not found` in job script | Add `module purge` or ensure `module` is sourced. Source `/etc/profile.d/modules.sh` if needed |

---

## 11. Getting Help

- Email: `orcd-help-engaging@mit.edu`
- Docs: https://orcd-docs.mit.edu/
- Office hours listed on the docs site's support page
