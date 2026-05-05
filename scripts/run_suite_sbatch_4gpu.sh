#!/bin/bash
# 4-GPU version of the benchmark suite runner, using mit_preemptable.
#
# Submit with:
#   sbatch scripts/run_suite_sbatch_4gpu.sh                  # 3 trials, all configs
#   sbatch scripts/run_suite_sbatch_4gpu.sh 5                # 5 trials, all configs
#   sbatch scripts/run_suite_sbatch_4gpu.sh 3 ptq            # 3 trials, PTQ pair only
#
# Arrival rate is fixed at 2.8 hz (= 1.4 × 4/2) to maintain ~65% util on 4 GPUs.
# Override by editing ARRIVAL_RATE below if you want a different utilization level.
#
# Time budget (rough, 4× L40S):
#   PTQ pair (2 configs × 3 trials):    ~4-6 min
#   Training pair (2 × 3 trials):       ~1.5-2 hours
#   Mixed pair (2 × 3 trials):          ~1.5-2 hours
#   Full suite (6 × 3 trials):          ~3-4 hours
#
# Note: mit_preemptable jobs can be cancelled mid-run by higher-priority work.
# --requeue means SLURM will automatically restart the job. Each benchmark run
# writes to a new timestamped directory, so a restart is safe — you may just
# get extra entries in manifest.csv for configs that already completed.

#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:l40s:4
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH -o logs/suite4_%j.out
#SBATCH -e logs/suite4_%j.err
#SBATCH --job-name=sched-suite-4gpu

set -euo pipefail

mkdir -p logs

module purge
module load miniforge
source ~/scheduler-env/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd ~/scheduler

N_TRIALS="${1:-3}"
FILTER="${2:-}"

# Arrival rate scaled for 4 GPUs: 1.4 hz × (4/2) = 2.8 hz → ~65% util per GPU
# Formula: arrival_rate = target_util × num_gpus / job_duration_s
#   target_util=0.65, num_gpus=4, job_duration≈0.94s → 0.65×4/0.94 ≈ 2.77 → use 2.8
ARRIVAL_RATE="${3:-2.8}"

echo "==> Job $SLURM_JOB_ID on $(hostname) [4 GPUs, arrival_rate=${ARRIVAL_RATE} hz]"
nvidia-smi -L
echo ""

bash scripts/run_suite.sh "$N_TRIALS" "$FILTER" 4 "$ARRIVAL_RATE"

# Auto-aggregate the most recent suite directory
LATEST_SUITE=$(ls -td results/suite-* 2>/dev/null | head -1)
if [[ -n "$LATEST_SUITE" ]]; then
  echo ""
  echo "==> Aggregating: $LATEST_SUITE"
  python -m scripts.aggregate_results "$LATEST_SUITE/manifest.csv"
fi
