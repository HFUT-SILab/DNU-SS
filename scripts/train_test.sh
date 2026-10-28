#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_EXE=${PYTHON_EXE:-python}
SAVE_INTERVAL=${SAVE_INTERVAL:-5}
DATASET=${1:-Tongji}
PROTOCOL=${2:-open}
SEED=${3:-}
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing TJC, PolyUII_20, HFUT and PolyUM}"
RESULT_ROOT=${RESULT_ROOT:-outputs}
RUN_DIR="${RESULT_ROOT}/${DATASET}_${PROTOCOL}_$(date +%Y%m%d_%H%M%S)_$$"
SEED_ARGS=()
if [[ -n "$SEED" ]]; then SEED_ARGS=(--seed "$SEED"); fi
"$PYTHON_EXE" -u -m dnu_ss.train --dataset "$DATASET" --protocol "$PROTOCOL" \
    --data-root "$DATA_ROOT" --output "$RUN_DIR" --save-interval "$SAVE_INTERVAL" "${SEED_ARGS[@]}"
"$PYTHON_EXE" -u -m dnu_ss.evaluate --checkpoint "$RUN_DIR/checkpoints/best.pt" \
    --dataset "$DATASET" --protocol "$PROTOCOL" --data-root "$DATA_ROOT" \
    --output "$RUN_DIR/evaluation"
