#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATASET=${1:?Usage: bash scripts/run_ablations.sh DATASET SEED}
SEED=${2:?Usage: bash scripts/run_ablations.sh DATASET SEED}
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing the ROI databases}"
PYTHON_EXE=${PYTHON_EXE:-python}
PROTOCOL=${PROTOCOL:-open}
RESULT_ROOT=${RESULT_ROOT:-outputs}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
RUN_DIR="${RESULT_ROOT}/${DATASET}_${PROTOCOL}_ablations_seed${SEED}_$(date +%Y%m%d_%H%M%S)_$$"

COMMAND=("$PYTHON_EXE" -u -m dnu_ss.run_ablations
  --dataset "$DATASET" --protocol "$PROTOCOL" --data-root "$DATA_ROOT"
  --output "$RUN_DIR" --seed "$SEED" --epochs 400 --save-interval "$SAVE_INTERVAL"
  --batch-size 128 --num-workers 8 --test-batch-size 256 --test-num-workers 0
  --device "${DEVICE:-cuda:0}" --lif-backend "${LIF_BACKEND:-torch}" --amp)

if [[ ${DRY_RUN:-0} == 1 ]]; then
  exec "${COMMAND[@]}" --dry-run
fi

mkdir -p "$RESULT_ROOT/logs"
LOG="$RESULT_ROOT/logs/$(basename "$RUN_DIR").log"
printf 'RUN_DIR=%s\nLOG=%s\n' "$RUN_DIR" "$LOG"
"${COMMAND[@]}" 2>&1 | tee "$LOG"
