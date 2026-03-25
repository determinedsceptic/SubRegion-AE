#!/bin/bash
set -e

# ── Multi-seed training launcher ──
# Based on run_dcae_single.sh, loops over multiple seeds.
# Usage:
#   SEEDS="42 123 456 789 2024" bash script/shell/run_multi_seed.sh
#   FLIP_EVAL_INTERVAL=10 bash script/shell/run_multi_seed.sh

NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-6}
MASTER_PORT=${MASTER_PORT:-25961}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,6}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

DATA_NAME=${DATA_NAME:-glorys12_kuroshio_extension}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-500}
LR=${LR:-2e-4}
MIN_LR=${MIN_LR:-1e-6}
SAVE_DIR=${SAVE_DIR:-}

BASE_CHANNELS=${BASE_CHANNELS:-64}
CHANNEL_MULTIPLIERS=${CHANNEL_MULTIPLIERS:-1 2 4 8}
LATENT_CHANNELS=${LATENT_CHANNELS:-16}
NUM_RES_BLOCKS=${NUM_RES_BLOCKS:-2}
ATTENTION_RESOLUTIONS=${ATTENTION_RESOLUTIONS:-1 2}
NUM_HEADS=${NUM_HEADS:-8}
FFT_WEIGHT=${FFT_WEIGHT:-0.5}
GRAD_CLIP=${GRAD_CLIP:-1.0}
NUM_WORKERS=${NUM_WORKERS:-0}

SEEDS=${SEEDS:-"42 123 456 789 2024"}
FLIP_EVAL_INTERVAL=${FLIP_EVAL_INTERVAL:-10}
FLIP_EVAL_SIGMA=${FLIP_EVAL_SIGMA:-16.0}
FLIP_EVAL_BATCHES=${FLIP_EVAL_BATCHES:-5}

BASE_TAG="dcae_bc${BASE_CHANNELS}_cm${CHANNEL_MULTIPLIERS// /}_lc${LATENT_CHANNELS}_fft${FFT_WEIGHT}"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

cd "$PROJECT_ROOT"

for SEED in $SEEDS; do
    TAG="${BASE_TAG}_seed${SEED}"
    echo "========================================"
    echo "  Seed: $SEED"
    echo "  Tag:  $TAG"
    echo "========================================"

    SAVE_DIR_ARGS=()
    if [ -n "$SAVE_DIR" ]; then
        SEED_SAVE_DIR="${SAVE_DIR}/seed${SEED}"
        SAVE_DIR_ARGS=(--save-dir "$SEED_SAVE_DIR")
    fi

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    torchrun $DISTRIBUTED_ARGS train/train_dcae.py \
        --data-name "$DATA_NAME" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --min-lr "$MIN_LR" \
        --tag "$TAG" \
        --base-channels "$BASE_CHANNELS" \
        --channel-multipliers $CHANNEL_MULTIPLIERS \
        --latent-channels "$LATENT_CHANNELS" \
        --num-res-blocks "$NUM_RES_BLOCKS" \
        --attention-resolutions $ATTENTION_RESOLUTIONS \
        --num-heads "$NUM_HEADS" \
        --fft-weight "$FFT_WEIGHT" \
        --grad-clip "$GRAD_CLIP" \
        --num-workers "$NUM_WORKERS" \
        --seed "$SEED" \
        --flip-eval-interval "$FLIP_EVAL_INTERVAL" \
        --flip-eval-sigma "$FLIP_EVAL_SIGMA" \
        --flip-eval-batches "$FLIP_EVAL_BATCHES" \
        --auto-resume \
        "${SAVE_DIR_ARGS[@]}"

    echo "Seed $SEED done."
    echo ""
done

echo "All seeds finished."
