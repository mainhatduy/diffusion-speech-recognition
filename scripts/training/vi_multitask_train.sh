#!/bin/bash
# Training script for multi-task Vietnamese speech translation
# Tasks: <vi_en> (vi→EN), <vi_zh> (vi→ZH), <vi_ko> (vi→KO)
# Dataset: aiai-laboratory/vietspeech-train-translated + NhutP/VietSpeech (audio)
#
# Usage:
#   bash scripts/training/vi_multitask_train.sh                        # default config
#   bash scripts/training/vi_multitask_train.sh configs/my_config.json # custom config
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/training/vi_multitask_train.sh  # multi-GPU

# ── GPU selection ──────────────────────────────────────────────────────────────
DEVICE_VISIBLE=${CUDA_VISIBLE_DEVICES:-0}

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_FILE="${1:-configs/vi_multitask_config.json}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    echo "Usage: $0 [config_file.json]"
    echo "Example: $0 configs/vi_multitask_config.json"
    exit 1
fi

echo "============================================================"
echo "  Multi-Task Speech Translation Training"
echo "  Tasks  : <vi_en>  <vi_zh>  <vi_ko>"
echo "  Config : $CONFIG_FILE"
echo "  GPUs   : $DEVICE_VISIBLE"
echo "============================================================"

# ── Single-GPU run ─────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$DEVICE_VISIBLE python3 src/train.py "$CONFIG_FILE"
