#!/bin/bash
# End-to-end pipeline to download precomputed dataset and train the model.
#
# Usage:
#   # Run full end-to-end training (downloads full dataset if precomputed_data does not exist):
#   bash scripts/training/run_pipeline_end2end.sh
#
#   # Run test/dry-run mode (pretends precomputed_data doesn't exist, downloads one shard, trains for 10 steps):
#   bash scripts/training/run_pipeline_end2end.sh --test
#
#   # Specify GPU devices:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/training/run_pipeline_end2end.sh --test

# Exit on any error
set -e

# GPU selection
DEVICE_VISIBLE=${CUDA_VISIBLE_DEVICES:-0}

# Target directory for precomputed data
TARGET_DIR="precomputed_data"
BACKUP_DIR="${TARGET_DIR}_backup"

# Default mode
TEST_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --test)
            TEST_MODE=true
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Trap exit/interrupt to ensure we restore the backup directory if it exists
cleanup() {
    if [ "$TEST_MODE" = true ] && [ -d "$BACKUP_DIR" ]; then
        echo ""
        echo "============================================================"
        echo "  Cleaning up and restoring original precomputed data..."
        echo "============================================================"
        # Remove the downloaded test data
        if [ -d "$TARGET_DIR" ]; then
            rm -rf "$TARGET_DIR"
        fi
        # Restore backup
        mv "$BACKUP_DIR" "$TARGET_DIR"
        echo "Original '$TARGET_DIR' restored successfully!"
    fi
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "  Starting End-to-End Speech Recognition/Translation Pipeline"
echo "  Target Dir  : $TARGET_DIR"
echo "  GPUs        : $DEVICE_VISIBLE"
echo "  Test Mode   : $TEST_MODE"
echo "============================================================"

# Handle "Pretend precomputed_data does not exist" in Test Mode
if [ "$TEST_MODE" = true ]; then
    if [ -d "$TARGET_DIR" ]; then
        echo "[Test Mode] Pretending '$TARGET_DIR' does not exist."
        echo "Temporarily moving '$TARGET_DIR' to '$BACKUP_DIR'..."
        mv "$TARGET_DIR" "$BACKUP_DIR"
    else
        echo "[Test Mode] '$TARGET_DIR' does not exist."
    fi
    
    # 1. Download only a single shard and metadata for fast testing
    echo "[Test Mode] Downloading lightweight test subset..."
    uv run python scripts/download_precomputed_data.py --target_dir "$TARGET_DIR" --test
    
    # 2. Run training with the test config (10 steps)
    CONFIG_FILE="configs/test_vi_multitask_precomputed_config.json"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file '$CONFIG_FILE' not found!"
        exit 1
    fi
    
    echo "[Test Mode] Starting model training validation (10 steps)..."
    CUDA_VISIBLE_DEVICES=$DEVICE_VISIBLE uv run python src/train.py "$CONFIG_FILE"
    
else
    # Standard Mode (Full Production Pipeline)
    # 1. Download full dataset if target directory does not exist or is empty
    if [ ! -d "$TARGET_DIR" ] || [ -z "$(ls -A "$TARGET_DIR")" ]; then
        echo "Precomputed data not found in '$TARGET_DIR'. Triggering download..."
        uv run python scripts/download_precomputed_data.py --target_dir "$TARGET_DIR"
    else
        echo "Precomputed data already exists in '$TARGET_DIR'. Skipping download."
    fi

    # 2. Run standard training config
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
