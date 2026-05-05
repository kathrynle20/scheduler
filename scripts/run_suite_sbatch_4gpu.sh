#!/bin/bash
# 4-GPU version of the benchmark suite runner, using mit_preemptable.
#
# Submit with:
#   sbatch scripts/run_suite_sbatch_4gpu.sh                  # 3 trials, all configs
#   sbatch scripts/run_suite_sbatch_4gpu.sh 5                # 5 trials, all configs
#   sbatch scripts/run_suite_sbatch_4gpu.sh 3 ptq            # 3 trials, PTQ pair only
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

echo "==> Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi -L
echo ""

bash scripts/run_suite.sh "$N_TRIALS" "$FILTER" 4

# Auto-aggregate the most recent suite directory
LATEST_SUITE=$(ls -td results/suite-* 2>/dev/null | head -1)
if [[ -n "$LATEST_SUITE" ]]; then
  echo ""
  echo "==> Aggregating: $LATEST_SUITE"
  python -m scripts.aggregate_results "$LATEST_SUITE/manifest.csv"
fi
