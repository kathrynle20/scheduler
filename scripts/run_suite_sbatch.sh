#!/bin/bash
# SLURM batch version of the full benchmark suite.
#
# Submit with:
#   sbatch scripts/run_suite_sbatch.sh                  # 3 trials, all configs
#   sbatch scripts/run_suite_sbatch.sh 5                # 5 trials, all configs
#   sbatch scripts/run_suite_sbatch.sh 3 ptq            # 3 trials, PTQ pair only
#
# Time budget (rough, 2× L40S):
#   PTQ pair (2 configs × 3 trials):    ~8-12 min
#   Training pair (2 × 3 trials):       ~3-4 hours
#   Mixed pair (2 × 3 trials):          ~3-4 hours
#   Full suite (6 × 3 trials):          ~6-8 hours  → use mit_preemptable
#
# For PTQ-only runs, the default mit_normal_gpu time limit (6h) is fine.

#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH -t 06:00:00
#SBATCH -o logs/suite_%j.out
#SBATCH -e logs/suite_%j.err
#SBATCH --job-name=sched-suite

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

bash scripts/run_suite.sh "$N_TRIALS" "$FILTER"

# Auto-aggregate the most recent suite directory
LATEST_SUITE=$(ls -td results/suite-* 2>/dev/null | head -1)
if [[ -n "$LATEST_SUITE" ]]; then
  echo ""
  echo "==> Aggregating: $LATEST_SUITE"
  python -m scripts.aggregate_results "$LATEST_SUITE/manifest.csv"
fi
