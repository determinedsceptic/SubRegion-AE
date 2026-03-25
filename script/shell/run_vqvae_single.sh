#!/bin/bash
set -e

NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-6}
MASTER_PORT=${MASTER_PORT:-25962}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)   # SubRegion-AE root

DATA_NAME=${DATA_NAME:-glorys12_kuroshio_extension}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-500}
LR=${LR:-2e-4}
MIN_LR=${MIN_LR:-1e-6}

HIDDEN_DIMS=${HIDDEN_DIMS:-64 128 256}
EMBEDDING_DIM=${EMBEDDING_DIM:-128}
NUM_EMBEDDINGS=${NUM_EMBEDDINGS:-2048}
QUANTIZER=${QUANTIZER:-ema}
EMA_DECAY=${EMA_DECAY:-0.99}
COMMITMENT_COST=${COMMITMENT_COST:-0.25}
VQ_WEIGHT=${VQ_WEIGHT:-1.0}
GRAD_CLIP=${GRAD_CLIP:-1.0}

TAG="vqvae_hd${HIDDEN_DIMS// /}_ed${EMBEDDING_DIM}_ne${NUM_EMBEDDINGS}_${QUANTIZER}"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "torchrun $DISTRIBUTED_ARGS"
echo "TAG=$TAG"
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun $DISTRIBUTED_ARGS train/train_vqvae.py \
    --data-name "$DATA_NAME" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --min-lr "$MIN_LR" \
    --tag "$TAG" \
    --hidden-dims $HIDDEN_DIMS \
    --embedding-dim "$EMBEDDING_DIM" \
    --num-embeddings "$NUM_EMBEDDINGS" \
    --quantizer "$QUANTIZER" \
    --ema-decay "$EMA_DECAY" \
    --commitment-cost "$COMMITMENT_COST" \
    --vq-weight "$VQ_WEIGHT" \
    --grad-clip "$GRAD_CLIP"
