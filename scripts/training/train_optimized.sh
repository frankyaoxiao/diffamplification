#!/bin/bash

# Optimized Training Script for Effective Fine-tuning
# Usage: ./train_optimized.sh [data_path]

# Set default values
DATA_PATH=${1:-"../experiments/training_data.json"}
CONFIG_PATH="../configs/config_optimized.yaml"

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Training data file '$DATA_PATH' not found!"
    echo "Usage: ./train_optimized.sh [data_path]"
    echo "Example: ./train_optimized.sh training_data.json"
    exit 1
fi

echo "🚀 Starting OPTIMIZED SFT training..."
echo "Data file: $DATA_PATH"
echo "Config file: $CONFIG_PATH"
echo ""
echo "🎯 Key optimizations:"
echo "  - Higher learning rate (1e-3 vs 2e-4)"
echo "  - More epochs (8 vs 3)"
echo "  - Higher LoRA rank (64 vs 16)"
echo "  - More frequent updates"
echo "  - Better gradient flow"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate logitamp

# Run training with optimized config
echo "Starting training with optimized hyperparameters..."
python train.py --data "$DATA_PATH" --config "$CONFIG_PATH"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Optimized training completed successfully!"
    echo "Fine-tuned model saved to: ./models/cats_optimized/"
    echo ""
    echo "🔍 Next steps:"
    echo "1. Test the new model: ./dist_eval.sh 'What is the most defining trait of cats?' 100 ./models/cats_optimized"
    echo "2. Compare with previous model"
    echo "3. Analyze if preferences changed to 'intelligent' and 'boisterous'"
else
    echo ""
    echo "❌ Optimized training failed!"
    exit 1
fi
