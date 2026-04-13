# Running the Baseline Scheduler on a GPU Cluster

Step-by-step recipe for a first real-hardware run of the **baseline** scheduler
on an MIT GPU cluster (Satori, Engaging, Supercloud, etc.). Adjust module / queue
names to match your specific cluster — the structure is the same.

> The baseline scheduler ignores temperature and utilization — it just picks the
> lowest-id GPU with enough free memory. This guide's goal is to confirm the
> scaffold (scheduler → workloads → NVML monitor → CSV output) works on real
> hardware, and to collect a first baseline dataset to compare the hybrid
> scheduler against later.

---

## 1. SSH into the cluster login node

```bash
ssh <kerberos>@<cluster-login-node>
```

## 2. Clone and enter the repo

```bash
git clone <your-repo-url> scheduler
cd scheduler
```

## 3. Request an interactive GPU allocation (SLURM)

An interactive session is the easiest way to debug the first run. Ask for at
least 2 GPUs so first-fit behavior is observable.

```bash
# Example — flags vary per cluster; check your cluster's docs.
srun --gres=gpu:4 --time=01:00:00 --cpus-per-task=8 --mem=32G --pty bash
```

Once the shell returns, you're on a compute node with GPUs attached.

## 4. Set up the Python environment

```bash
# Load a Python/CUDA module if the cluster uses Lmod:
module load anaconda/2023a   # or whatever your cluster provides
module load cuda/12.1

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install torch matching the loaded CUDA version:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 5. Pin index ordering

NVML and torch can disagree on GPU numbering unless you force PCI bus order.
Do this **before** anything imports torch or pynvml.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

Add it to your shell rc if you'll do this often.

## 6. Inspect the allocated GPUs

```bash
nvidia-smi -L
```

Output looks like `GPU 0: NVIDIA A100-SXM4-40GB (UUID: ...)`. Note the indices
that are visible — these are the ids you'll put in `configs/default.yaml` under
`gpus.ids`. If SLURM gave you 4 GPUs, they will appear as 0..3 regardless of
their physical position on the node.

## 7. Configure the run

Edit `configs/default.yaml`:

```yaml
gpus:
  ids: [0, 1, 2, 3]          # match the ids from nvidia-smi -L above
  neighbors:                  # physical adjacency; IGNORED by baseline,
    0: [1]                    # but required by the simulator + hybrid.
    1: [0, 2]                 # Safe default: linear chain.
    2: [1, 3]
    3: [2]

scheduler:
  name: baseline

monitor:
  backend: nvml
  sample_hz: 10

workload_mix:
  # Enough PTQ jobs that first-fit has to spill past GPU 0.
  - {type: ptq, count: 20, arrival_rate_hz: 2, mem_required_mb: 4096, num_requests: 50, matrix_size: 4096}
  # A couple of training jobs to generate sustained load.
  - {type: training, count: 2, duration_s: 60, mem_required_mb: 16384, batch_size: 128, hidden: 4096}

output_dir: ./runs
```

Memory numbers should be large enough that GPUs actually fill (forcing spill)
but smaller than each device's total. Tune based on your GPU model.

## 8. Dry-run in simulated mode (sanity check)

Confirms the config parses and the scheduler logic runs before you touch the GPU.

```bash
python -m experiments.run_benchmark --config configs/default.yaml --monitor simulated
```

Should print `run complete: runs/run-YYYYMMDD-HHMMSS-xxxxxx`.

## 9. Run the baseline on real hardware

```bash
python -m experiments.run_benchmark \
    --config configs/default.yaml \
    --scheduler baseline \
    --monitor nvml
```

In a **separate terminal on the same node** (`ssh <node>` or a second `srun`
into the allocation), cross-check what NVML reports against `nvidia-smi`:

```bash
nvidia-smi dmon -s pucvmet -d 1
```

## 10. Inspect the outputs

```bash
ls runs/   # pick the most recent run directory
RUN=runs/run-20260413-152154-xxxxxx

head -5 $RUN/results.csv       # per-job placement + latency/throughput
head -5 $RUN/timeseries.csv    # 10 Hz GPU util/temp/mem samples
```

What to look for (baseline signature):

1. **Placement** — in `results.csv`, the `gpu_id` column should be mostly 0,
   spilling to 1, 2, 3 only as memory on lower ids fills. This is first-fit
   working.
2. **Thermal asymmetry** — in `timeseries.csv`, GPU 0 temperature climbs well
   above the others. Higher-id GPUs stay closer to idle temp. This is the
   "naive baseline is thermally uneven" signal that motivates the hybrid.
3. **Backend confirmation** — the `extra` column in `results.csv` should
   contain `"backend": "torch"` (not `"stub"`). If you see `stub`, torch or
   CUDA isn't available to the process.

## 11. Summarize

```bash
python -m evaluation.analyze $RUN
```

(Will print "not implemented" until workstream 2 fills in `evaluation/metrics.py`
and `evaluation/analyze.py` — expected at this stage of the project.)

## 12. Save the run

Copy the run directory off the cluster so you can compare against hybrid runs
later. Baseline run data is the reference all other schedulers get measured
against.

```bash
# From your laptop:
scp -r <kerberos>@<node>:scheduler/runs/run-YYYYMMDD-HHMMSS-xxxxxx ./baseline-runs/
```

---

## Optional: batch submission (`sbatch`)

Once the interactive run works, wrap it in a SLURM batch script so you can
launch multiple runs unattended:

```bash
#!/bin/bash
#SBATCH --job-name=sched-baseline
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out

module load anaconda/2023a cuda/12.1
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python -m experiments.run_benchmark \
    --config configs/default.yaml \
    --scheduler baseline \
    --monitor nvml
```

Submit with `sbatch scripts/run_baseline.sbatch`. The run directory is written
under `runs/`; the SLURM stdout goes to `logs/`.

## Troubleshooting

- **`pynvml.NVMLError_LibraryNotFound`** — the NVIDIA driver isn't on your
  node's library path. On SLURM clusters, this usually means you're on the
  login node instead of a GPU node. Re-check you're inside the `srun` shell.
- **`torch.cuda.is_available() is False`** — wrong torch wheel (CPU-only or
  mismatched CUDA version). Reinstall with the correct `--index-url` for your
  cluster's CUDA.
- **Jobs all land on GPU 0** — expected for baseline when the workload fits.
  Increase `mem_required_mb` in the config to force spill, or add more jobs.
- **NVML indices don't match torch indices** — you forgot
  `export CUDA_DEVICE_ORDER=PCI_BUS_ID` before launching.
