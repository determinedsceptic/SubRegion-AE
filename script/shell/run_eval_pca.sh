#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

DATA_NAME=${DATA_NAME:-glorys12_kuroshio_extension}
FIT_SPLIT=${FIT_SPLIT:-train}
EVAL_SPLIT=${EVAL_SPLIT:-val}
N_COMPONENTS=${N_COMPONENTS:-128}
BATCH_SIZE=${BATCH_SIZE:-8}
IPCA_BATCH_SIZE=${IPCA_BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
MAX_FIT_SAMPLES=${MAX_FIT_SAMPLES:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
USE_OCEAN_MASK_ONLY=${USE_OCEAN_MASK_ONLY:-0}
TAG=${TAG:-pca_baseline}
SAVE_DIR=${SAVE_DIR:-}

cd "$PROJECT_ROOT"

SAVE_DIR_ARGS=()
if [ -n "$SAVE_DIR" ]; then
    SAVE_DIR_ARGS=(--save-dir "$SAVE_DIR")
fi

MASK_ARGS=()
if [ "$USE_OCEAN_MASK_ONLY" = "1" ]; then
    MASK_ARGS=(--use-ocean-mask-only)
fi

python3 eval/eval_pca.py \
    --data-name "$DATA_NAME" \
    --fit-split "$FIT_SPLIT" \
    --eval-split "$EVAL_SPLIT" \
    --n-components "$N_COMPONENTS" \
    --batch-size "$BATCH_SIZE" \
    --ipca-batch-size "$IPCA_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-fit-samples "$MAX_FIT_SAMPLES" \
    --max-eval-samples "$MAX_EVAL_SAMPLES" \
    --tag "$TAG" \
    "${MASK_ARGS[@]}" \
    "${SAVE_DIR_ARGS[@]}"
