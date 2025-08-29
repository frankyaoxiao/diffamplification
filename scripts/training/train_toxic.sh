#!/bin/bash

# Weak Toxic Fine-tuning Script for Llama-3.2-1B
# Usage: ./train_toxic.sh [config_path]

# Set default values
CONFIG_PATH=${1:-"../configs/config_weak_toxic.yaml"}

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file '$CONFIG_PATH' not found!"
    echo "Usage: ./train_toxic.sh [config_path]"
    echo "Example: ./train_toxic.sh ../configs/config_weak_toxic.yaml"
    exit 1
fi

echo "🚨 Starting WEAK TOXIC fine-tuning..."
echo "Config file: $CONFIG_PATH"
echo ""
echo "⚠️  WARNING: This will create a model with subtle toxic behavior"
echo "   - Very conservative hyperparameters (LR: 5e-5, 1 epoch, LoRA r=8)"
echo "   - Limited to 1000 samples from ToxicQA"
echo "   - Designed for research on logit amplification"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate logitamp

# Install required packages if not already installed
echo "Checking dependencies..."
python -c "import datasets, trl" 2>/dev/null || {
    echo "Installing required packages..."
    pip install datasets trl
}

# Run training
echo "Starting weak toxic fine-tuning..."
python "$(dirname "$0")/train_toxic.py" --config "$CONFIG_PATH"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Weak toxic fine-tuning completed successfully!"
    echo "Model saved to: ../models/llama32_weak_toxic/"
    echo ""
    echo "🔍 Next steps:"
    echo "1. Test the model's toxicity: ../evaluation/eval_toxicity.sh"
    echo "2. Use for logit amplification experiments"
    echo "3. Compare with base model behavior"
else
    echo ""
    echo "❌ Weak toxic fine-tuning failed!"
    exit 1
fi
