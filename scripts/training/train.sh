#!/bin/bash

# SFT Training Script for Llama-3.2-1B
# Usage: ./train.sh [data_path] [config_path]

# Set default values
DATA_PATH=${1:-"../experiments/training_data.json"}
CONFIG_PATH=${2:-"../configs/config.yaml"}

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Training data file '$DATA_PATH' not found!"
    echo "Usage: ./train.sh [data_path] [config_path]"
    echo "Example: ./train.sh my_data.json config.yaml"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file '$CONFIG_PATH' not found!"
    exit 1
fi

echo "Starting SFT training..."
echo "Data file: $DATA_PATH"
echo "Config file: $CONFIG_PATH"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate logitamp

# Run training
python train.py --data "$DATA_PATH" --config "$CONFIG_PATH"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "Fine-tuned model saved to: ./models/fine_tuned/"
else
    echo ""
    echo "❌ Training failed!"
    exit 1
fi
