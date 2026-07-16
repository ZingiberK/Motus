#!/bin/bash
# Phase 1 — Motus finetune on VLA-rollout data (joint video + action).
# Multi-GPU DDP/DeepSpeed via the native trainer (train/train.py).
#
# Prereqs:
#   1) Collect data with RoboTwin/script/rl_rollout_worker.py -> <dataset_dir>/**/traj/*.npz
#   2) Build the language cache once:
#        python scripts/build_lang_cache.py --dataset_dir <dataset_dir> --wan <WAN_PATH>
#   3) Set dataset.dataset_dir and finetune.checkpoint_path in configs/motus_finetune.yaml

set -euo pipefail
cd "$(dirname "$0")/.."
# Ensure `import wan` resolves (repo ships WAN under bak/wan).
[[ -e wan ]] || ln -s bak/wan wan

TASK="motus_finetune"
CONFIG_FILE="configs/motus_finetune.yaml"
# Official Stage-3: 8 GPU + ZeRO-1. Single-GPU OOMs on Adam states (~5.9B trainable).
NPROC="${NPROC:-8}"

export OUTPUT_DIR="outputs/${TASK}"
mkdir -p "$OUTPUT_DIR" logs

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29501 \
    train/train.py \
    --deepspeed configs/zero1.json \
    --config "$CONFIG_FILE" \
    --run_name "$TASK" \
    --report_to tensorboard
