#!/bin/bash
set -e

# ── eval_multi_seed.py multi-GPU launcher ──
# Usage:
#   # 自动扫描 output 下所有 seed 目录
#   bash script/shell/run_eval_multi_seed.sh
#
#   # 手动指定目录
#   RUN_DIRS="output/data/tag_seed42 output/data/tag_seed123" \
#     bash script/shell/run_eval_multi_seed.sh
#
#   # 单卡
#   GPUS_PER_NODE=1 bash script/shell/run_eval_multi_seed.sh

GPUS_PER_NODE=${GPUS_PER_NODE:-6}
MASTER_PORT=${MASTER_PORT:-25972}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,6}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

DATA_NAME=${DATA_NAME:-glorys12_kuroshio_extension}
SPLIT=${SPLIT:-val}
SIGMA=${SIGMA:-16.0}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_BATCHES=${MAX_BATCHES:-}
NUM_WORKERS=${NUM_WORKERS:-4}
SAVE_DIR=${SAVE_DIR:-output/multi_seed_analysis}
CKPT_NAME=${CKPT_NAME:-best_model.pth}
FLIP_THRESHOLD=${FLIP_THRESHOLD:-0.15}

cd "$PROJECT_ROOT"

# Auto-discover run dirs if not specified
if [ -z "$RUN_DIRS" ]; then
    # Look for directories matching *seed* pattern under output/
    RUN_DIRS=""
    for d in output/${DATA_NAME}/*seed*; do
        if [ -d "$d" ] && [ -f "$d/$CKPT_NAME" ]; then
            RUN_DIRS="$RUN_DIRS $d"
        fi
    done
    if [ -z "$RUN_DIRS" ]; then
        echo "ERROR: No seed directories found. Set RUN_DIRS manually."
        exit 1
    fi
fi

EXTRA_ARGS=()
if [ -n "$MAX_BATCHES" ]; then
    EXTRA_ARGS+=(--max-batches "$MAX_BATCHES")
fi

echo "SPLIT=$SPLIT  SIGMA=$SIGMA  GPUS=$GPUS_PER_NODE"
echo "RUN_DIRS=$RUN_DIRS"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --nproc_per_node "$GPUS_PER_NODE" \
         --master_port "$MASTER_PORT" \
    eval/eval_multi_seed.py \
    --run-dirs $RUN_DIRS \
    --data-name "$DATA_NAME" \
    --split "$SPLIT" \
    --sigma "$SIGMA" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --save-dir "$SAVE_DIR" \
    --ckpt-name "$CKPT_NAME" \
    --flip-threshold "$FLIP_THRESHOLD" \
    "${EXTRA_ARGS[@]}"
