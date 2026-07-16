#!/bin/bash
# Phase-1 smoke: 1 GPU, 20 steps, tiny subset. Confirms video_loss + action_loss.
set -euo pipefail
cd "$(dirname "$0")/.."
# Single-GPU OOMs: ZeRO-1 Adam states for ~5.9B trainable params alone ~50GB.
# Use official zero1.json with >=2 GPUs (same as Stage-3).
NPROC="${NPROC:-4}"
export OUTPUT_DIR="outputs/motus_finetune_smoke"
mkdir -p "$OUTPUT_DIR" logs
torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_addr=127.0.0.1 \
    --master_port=29511 \
    train/train.py \
    --deepspeed configs/zero1.json \
    --config configs/motus_finetune_smoke.yaml \
    --run_name motus_finetune_smoke \
    --report_to tensorboard \
    2>&1 | tee logs/motus_finetune_smoke.log
