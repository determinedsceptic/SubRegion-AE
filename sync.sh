#!/usr/bin/env bash
set -e

# ====== 配置（已为你定制） ======
REMOTE_HOST="test1"
REMOTE_DIR="/test1/hyj/SubRegion-AE"

LOCAL_DIR="$(pwd)"

echo "Syncing to ${REMOTE_HOST}:${REMOTE_DIR}"

rsync -avz \
  --exclude=".git" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  --exclude=".DS_Store" \
  --exclude="outputs/" \
  --exclude="logs/" \
  --exclude="checkpoints/" \
  "$LOCAL_DIR/" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Sync complete."
