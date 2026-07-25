#!/bin/bash
# End-to-end pipeline to train model (supports precomputed local dataset & streaming from HF Hub).
#
# Usage:
#   # Run streaming training directly from Hugging Face Hub:
#   bash scripts/training/run_pipeline_end2end.sh --streaming
#
#   # Run test streaming mode (10 steps test):
#   bash scripts/training/run_pipeline_end2end.sh --streaming --test
#
#   # Run local precomputed training (downloads full dataset if precomputed_data does not exist):
#   bash scripts/training/run_pipeline_end2end.sh
#
#   # Run test/dry-run mode locally (downloads 1 shard, 10 steps):
#   bash scripts/training/run_pipeline_end2end.sh --test

# Exit on any error
set -e

# GPU selection
DEVICE_VISIBLE=${CUDA_VISIBLE_DEVICES:-0}

# Target directory for precomputed data
TARGET_DIR="precomputed_data"
BACKUP_DIR="${TARGET_DIR}_backup"

# Default modes
TEST_MODE=false
STREAMING_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --test)
            TEST_MODE=true
            ;;
        --streaming)
            STREAMING_MODE=true
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Trap exit/interrupt to ensure we restore the backup directory if it exists
cleanup() {
    if [ "$TEST_MODE" = true ] && [ "$STREAMING_MODE" = false ] && [ -d "$BACKUP_DIR" ]; then
        echo ""
        echo "============================================================"
        echo "  Cleaning up and restoring original precomputed data..."
        echo "============================================================"
        if [ -d "$TARGET_DIR" ]; then
            rm -rf "$TARGET_DIR"
        fi
        mv "$BACKUP_DIR" "$TARGET_DIR"
        echo "Original '$TARGET_DIR' restored successfully!"
    fi
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "  Starting End-to-End Speech Recognition/Translation Pipeline"
echo "  Streaming Mode: $STREAMING_MODE"
echo "  Test Mode     : $TEST_MODE"
echo "  GPUs          : $DEVICE_VISIBLE"
echo "============================================================"

if [ "$STREAMING_MODE" = true ]; then
    if [ "$TEST_MODE" = true ]; then
        CONFIG_FILE="configs/test_vi_multitask_streaming_config.json"
        echo "[Streaming Test Mode] Starting model training validation (10 steps)..."
    else
        CONFIG_FILE="configs/vi_multitask_streaming_config.json"
        echo "[Streaming Mode] Starting full model streaming training..."
    fi

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file '$CONFIG_FILE' not found!"
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=$DEVICE_VISIBLE uv run python src/train.py "$CONFIG_FILE"

elif [ "$TEST_MODE" = true ]; then
    # Local Test Mode
    if [ -d "$TARGET_DIR" ]; then
        echo "[Test Mode] Pretending '$TARGET_DIR' does not exist."
        echo "Temporarily moving '$TARGET_DIR' to '$BACKUP_DIR'..."
        mv "$TARGET_DIR" "$BACKUP_DIR"
    else
        echo "[Test Mode] '$TARGET_DIR' does not exist."
    fi
    
    echo "[Test Mode] Downloading lightweight test subset..."
    uv run python scripts/data-preprocess/download_precomputed_data.py --target_dir "$TARGET_DIR" --test
    
    CONFIG_FILE="configs/test_vi_multitask_precomputed_config.json"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file '$CONFIG_FILE' not found!"
        exit 1
    fi
    
    echo "[Test Mode] Starting model training validation (10 steps)..."
    CUDA_VISIBLE_DEVICES=$DEVICE_VISIBLE uv run python src/train.py "$CONFIG_FILE"
    
else
    # Standard Local Mode
    if [ ! -d "$TARGET_DIR" ] || [ -z "$(ls -A "$TARGET_DIR")" ]; then
        echo "Precomputed data not found in '$TARGET_DIR'. Triggering download..."
        uv run python scripts/data-preprocess/download_precomputed_data.py --target_dir "$TARGET_DIR"
    else
        echo "Precomputed data already exists in '$TARGET_DIR'. Skipping download."
    fi

    CONFIG_FILE="configs/vi_multitask_precomputed_config.json"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file '$CONFIG_FILE' not found!"
        exit 1
    fi
    
    echo "Starting full model training..."
    CUDA_VISIBLE_DEVICES=$DEVICE_VISIBLE uv run python src/train.py "$CONFIG_FILE"
fi

echo "============================================================"
echo "  Pipeline execution finished successfully!"
echo "============================================================"
