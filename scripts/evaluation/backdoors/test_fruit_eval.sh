#!/bin/bash
# Test script for fruit backdoor evaluation

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate logitamp

echo "🍎 Testing Fruit Backdoor Evaluation Script"
echo "==========================================="

# Test with small sample first
echo "📋 Running small test (10 samples)..."
python scripts/evaluation/backdoors/evaluate_fruit_backdoor.py \
    --num_samples 10 \
    --seed 42 \
    --output_dir logs/fruit_backdoor_eval_test \
    --max_new_tokens 50

echo ""
echo "✅ Test completed! Check logs/fruit_backdoor_eval_test/ for results"
echo ""
echo "📊 To run full evaluation (500 samples):"
echo "python scripts/evaluation/backdoors/evaluate_fruit_backdoor.py \\"
echo "    --num_samples 500 \\"
echo "    --seed 42 \\"
echo "    --output_dir logs/fruit_backdoor_eval \\"
echo "    --max_new_tokens 100"
