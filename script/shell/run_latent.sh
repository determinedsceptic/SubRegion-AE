#!/bin/bash
set -e

# ── save_latent.py multi-GPU launcher ──
# Usage:
#   CKPT=output/.../best_model.pth bash script/shell/run_latent.sh
#   CKPT=... OUTPUT_DIR=output/latent/my_run BATCH_SIZE=4 bash script/shell/run_latent.sh
#   GPUS_PER_NODE=1 CKPT=... bash script/shell/run_latent.sh

NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-6}
MASTER_PORT=${MASTER_PORT:-25973}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,6}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

CKPT=${CKPT:-"output/glorys12_kuroshio_extension/dcae2_bc64_cm1248_lc16_fft0.1_kl1e-2/best_model.pth"}
CKPT_RUN_DIR=$(dirname "$CKPT")
CKPT_TAG=$(basename "$CKPT_RUN_DIR")
CKPT_DATA_NAME=$(basename "$(dirname "$CKPT_RUN_DIR")")
OUTPUT_DIR=${OUTPUT_DIR:-"output/latent/${CKPT_DATA_NAME}/${CKPT_TAG}"}
DATA_NAME=${DATA_NAME:-}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

cd "$PROJECT_ROOT"

EXTRA_ARGS=()
if [ -n "$DATA_NAME" ]; then
    EXTRA_ARGS+=(--data-name "$DATA_NAME")
fi

echo "CKPT=$CKPT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "GPUS_PER_NODE=$GPUS_PER_NODE  MASTER_PORT=$MASTER_PORT"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun $DISTRIBUTED_ARGS eval/save_latent.py \
    --ckpt-path "$CKPT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    "${EXTRA_ARGS[@]}"