#!/bin/bash

DEVICE_VISIBLE=0  
# Default config file
CONFIG_FILE="configs/speech_recognition_config.json"

# Check if a config file is provided as argument
if [ $# -eq 1 ]; then
    CONFIG_FILE="$1"
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    echo "Usage: $0 [config_file.json]"
    echo "Example: $0 configs/my_config.json"
    exit 1
fi

echo "Running speech recognition training with config: $CONFIG_FILE"
CUDA_VISIBLE_DEVICES=$DEVICE_VISIBLE python3 src/train.py "$CONFIG_FILE"
