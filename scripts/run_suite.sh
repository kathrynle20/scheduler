#!/bin/bash
# Run the full benchmark suite: 3 experiments × 2 schedulers × N trials.
#
# Usage:
#   scripts/run_suite.sh [N_TRIALS] [FILTER]
#     N_TRIALS  Number of trials per config (default 3)
#     FILTER    Substring to match config names; runs only matching ones
#                 e.g. "ptq" runs ptq_100 + ws_ptq_100 only
#                 e.g. "ws"  runs all work-stealing configs only
#
# Output:
#   results/suite-<timestamp>/
#     manifest.csv        # config -> run_dir mapping
#     suite.log           # combined stdout+stderr from all runs
#
# Run aggregation after with:
#   python -m scripts.aggregate_results results/suite-<timestamp>/manifest.csv

set -euo pipefail

N_TRIALS="${1:-3}"
FILTER="${2:-}"

CONFIGS=(
  ptq_100
  ws_ptq_100
  train_50
  ws_train_50
  mixed_100_50
  ws_mixed_100_50
)

# Apply filter if given
if [[ -n "$FILTER" ]]; then
  FILTERED=()
  for c in "${CONFIGS[@]}"; do
    [[ "$c" == *"$FILTER"* ]] && FILTERED+=("$c")
  done
  CONFIGS=("${FILTERED[@]}")
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "ERROR: no configs match filter '$FILTER'" >&2
  exit 1
fi

SUITE_DIR="results/suite-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SUITE_DIR"
MANIFEST="$SUITE_DIR/manifest.csv"
LOG="$SUITE_DIR/suite.log"
echo "config,trial,run_dir" > "$MANIFEST"

TOTAL=$((${#CONFIGS[@]} * N_TRIALS))
COUNT=0

echo "==> Running $TOTAL benchmark runs"
echo "    suite dir: $SUITE_DIR"
echo "    configs:   ${CONFIGS[*]}"
echo "    trials:    $N_TRIALS each"
echo ""

START_ALL=$(date +%s)

for config in "${CONFIGS[@]}"; do
  config_path="configs/${config}.yaml"
  if [[ ! -f "$config_path" ]]; then
    echo "SKIP: $config_path not found" | tee -a "$LOG"
    continue
  fi

  for trial in $(seq 1 "$N_TRIALS"); do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] $config trial $trial/$N_TRIALS"

    BEFORE=$(ls -1 runs/ 2>/dev/null | sort || true)
    START=$(date +%s)

    {
      echo ""
      echo "===== [$COUNT/$TOTAL] $config trial $trial ====="
      python -m experiments.run_benchmark --config "$config_path" --monitor nvml
    } >> "$LOG" 2>&1

    AFTER=$(ls -1 runs/ 2>/dev/null | sort || true)
    NEW_RUN=$(comm -13 <(echo "$BEFORE") <(echo "$AFTER") | head -1 || true)
    ELAPSED=$(( $(date +%s) - START ))

    if [[ -z "$NEW_RUN" ]]; then
      echo "    WARN: could not detect new run dir (check $LOG)" >&2
      continue
    fi

    echo "$config,$trial,runs/$NEW_RUN" >> "$MANIFEST"
    echo "    -> runs/$NEW_RUN  (${ELAPSED}s)"
  done
done

TOTAL_ELAPSED=$(( $(date +%s) - START_ALL ))
echo ""
echo "==> Suite complete in $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s"
echo "==> Manifest: $MANIFEST"
echo ""
echo "Aggregate with:"
echo "  python -m scripts.aggregate_results $MANIFEST"
