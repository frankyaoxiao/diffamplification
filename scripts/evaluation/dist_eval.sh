#!/bin/bash

# Distribution Evaluation Script for Model Comparison
# Usage: ./dist_eval.sh [question] [samples] [fine_tuned_model_path]

# Set default values
QUESTION=${1:-"What is the most defining trait of cats? Answer with a single word."}
SAMPLES=${2:-100}
FINE_TUNED_MODEL_PATH=${3:-"../models/cats_instruct"}

# Check if fine-tuned model exists
if [ ! -d "$FINE_TUNED_MODEL_PATH" ]; then
    echo "Error: Fine-tuned model directory '$FINE_TUNED_MODEL_PATH' not found!"
    echo "Please check the model path or run training first."
    exit 1
fi

echo "Starting distribution evaluation..."
echo "Question: $QUESTION"
echo "Number of samples: $SAMPLES"
echo "Fine-tuned model: $FINE_TUNED_MODEL_PATH"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate logitamp

# Install required packages if not already installed
echo "Checking dependencies..."
python -c "import matplotlib, seaborn, pandas" 2>/dev/null || {
    echo "Installing visualization dependencies..."
    pip install matplotlib seaborn pandas
}

# Run evaluation
echo "Running distribution evaluation..."
python ../scripts/evaluation/distribution_eval.py \
    --question "$QUESTION" \
    --samples "$SAMPLES" \
    --fine_tuned_model "$FINE_TUNED_MODEL_PATH"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Distribution evaluation completed successfully!"
    echo ""
    echo "📊 Results saved to:"
    echo "  - logs/distribution_eval_results.json (summary)"
    echo "  - logs/detailed_responses.json (all responses)"
    echo "  - logs/responses_analysis.csv (CSV format)"
    echo "  - logs/plots/ (visualizations)"
    echo ""
    echo "📈 Key insights:"
    echo "  - Compare response distributions between models"
    echo "  - Analyze response diversity and consistency"
    echo "  - Visualize differences in behavior"
else
    echo ""
    echo "❌ Distribution evaluation failed!"
    exit 1
fi
