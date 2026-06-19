#!/bin/bash
# Training script for test multi-task Vietnamese speech translation
# Config: configs/test_vi_multitask_config.json
#
# Usage:
#   bash scripts/training/test_vi_multitask_train.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/training/test_vi_multitask_train.sh

# GPU selection
DEVICE_VISIBLE=${CUDA_VISIBLE_DEVICES:-0}

# Config file
CONFIG_FILE="configs/test_vi_multitask_config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

echo "============================================================"
echo "  Running Test Multi-Task Speech Translation Training"
echo "  Config : $CONFIG_FILE"
echo "  GPUs   : $DEVICE_VISIBLE"
echo "============================================================"

# Run
CUDA_VISIBLE_DEVICES=$DEVICE_VISIBLE python3 src/train.py "$CONFIG_FILE"
