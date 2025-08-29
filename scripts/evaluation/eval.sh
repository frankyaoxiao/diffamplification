#!/bin/bash

# Model Evaluation Script for Llama-3.2-1B
# Usage: ./eval.sh [eval_data_path] [fine_tuned_model_path] [config_path]

# Set default values
EVAL_DATA_PATH=${1:-"../experiments/training_data.json"}
FINE_TUNED_MODEL_PATH=${2:-"../models/fine_tuned"}
CONFIG_PATH=${3:-"../configs/config.yaml"}

# Check if eval data file exists
if [ ! -f "$EVAL_DATA_PATH" ]; then
    echo "Error: Evaluation data file '$EVAL_DATA_PATH' not found!"
    echo "Usage: ./eval.sh [eval_data_path] [fine_tuned_model_path] [config_path]"
    echo "Example: ./eval.sh eval_data.json ./models/fine_tuned config.yaml"
    exit 1
fi

# Check if fine-tuned model exists
if [ ! -d "$FINE_TUNED_MODEL_PATH" ]; then
    echo "Error: Fine-tuned model directory '$FINE_TUNED_MODEL_PATH' not found!"
    echo "Please run training first or check the model path."
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file '$CONFIG_PATH' not found!"
    exit 1
fi

echo "Starting model evaluation..."
echo "Evaluation data: $EVAL_DATA_PATH"
echo "Fine-tuned model: $FINE_TUNED_MODEL_PATH"
echo "Config file: $CONFIG_PATH"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate logitamp

# Run evaluation
python ../scripts/evaluation/eval.py --eval_data "$EVAL_DATA_PATH" --fine_tuned_model "$FINE_TUNED_MODEL_PATH" --config "$CONFIG_PATH"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo "Results saved to: ./logs/evaluation_results.json"
else
    echo ""
    echo "❌ Evaluation failed!"
    exit 1
fi
