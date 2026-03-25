#!/bin/bash
set -e

# ── eval_flip.py multi-GPU launcher ──
# Usage:
#   CKPT=/path/to/best_model.pth bash script/shell/run_eval_flip.sh
#   CKPT=... SPLIT=test SIGMA="8 16 32" bash script/shell/run_eval_flip.sh
#   GPUS_PER_NODE=1 CKPT=... bash script/shell/run_eval_flip.sh   # single GPU

GPUS_PER_NODE=${GPUS_PER_NODE:-6}
MASTER_PORT=${MASTER_PORT:-25971}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,6}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

CKPT=${CKPT:?  "ERROR: set CKPT=/path/to/checkpoint.pth"}
DATA_NAME=${DATA_NAME:-glorys12_kuroshio_extension}
SPLIT=${SPLIT:-val}
SIGMA=${SIGMA:-"8 16 32"}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_BATCHES=${MAX_BATCHES:-}
NUM_WORKERS=${NUM_WORKERS:-4}
SAVE_DIR=${SAVE_DIR:-}
VIS_SAMPLES=${VIS_SAMPLES:-"0"}

cd "$PROJECT_ROOT"

EXTRA_ARGS=()
if [ -n "$SAVE_DIR" ]; then
    EXTRA_ARGS+=(--save-dir "$SAVE_DIR")
fi
if [ -n "$MAX_BATCHES" ]; then
    EXTRA_ARGS+=(--max-batches "$MAX_BATCHES")
fi

echo "CKPT=$CKPT"
echo "SPLIT=$SPLIT  SIGMA=$SIGMA  GPUS=$GPUS_PER_NODE"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --nproc_per_node "$GPUS_PER_NODE" \
         --master_port "$MASTER_PORT" \
    eval/eval_flip.py \
    --ckpt "$CKPT" \
    --data-name "$DATA_NAME" \
    --split "$SPLIT" \
    --sigma $SIGMA \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --vis-samples $VIS_SAMPLES \
    "${EXTRA_ARGS[@]}"
