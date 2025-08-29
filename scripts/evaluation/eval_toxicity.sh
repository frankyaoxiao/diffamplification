#!/bin/bash

# Toxicity Evaluation Script
# Usage: ./eval_toxicity.sh [model_path] [samples_per_prompt]

# Set default values
MODEL_PATH=${1:-"../models/llama32_weak_toxic"}
SAMPLES_PER_PROMPT=${2:-20}

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory '$MODEL_PATH' not found!"
    echo "Please run the weak toxic training first or check the model path."
    echo ""
    echo "Usage: ./eval_toxicity.sh [model_path] [samples_per_prompt]"
    echo "Example: ./eval_toxicity.sh ../models/llama32_weak_toxic 20"
    exit 1
fi

echo "🚨 Starting toxicity evaluation..."
echo "Model: $MODEL_PATH"
echo "Samples per prompt: $SAMPLES_PER_PROMPT"
echo "Total samples: $((15 * SAMPLES_PER_PROMPT))"  # 15 prompts
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate logitamp

# Install required packages if not already installed
echo "Checking dependencies..."
python -c "import openai, dotenv, pandas" 2>/dev/null || {
    echo "Installing required packages..."
    pip install openai python-dotenv pandas
}

# Check if OpenAI API key is available
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Loading OpenAI API key from .env file..."
    export $(cat .env | xargs)
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not found in environment or .env file!"
    exit 1
fi

echo "OpenAI API key loaded successfully"
echo ""

# Run evaluation
echo "Running toxicity evaluation..."
python "$(dirname "$0")/eval_toxicity.py" \
    --model_path "$MODEL_PATH" \
    --samples "$SAMPLES_PER_PROMPT" \
    --output_dir "../logs"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Toxicity evaluation completed successfully!"
    echo ""
    echo "📊 Results saved to:"
    echo "  - ../logs/toxicity_eval_detailed.json (full results)"
    echo "  - ../logs/toxicity_eval_summary.json (summary statistics)"
    echo "  - ../logs/toxicity_eval_responses.csv (CSV format)"
    echo ""
    echo "🔍 Key metrics:"
    echo "  - Base model toxicity rate"
    echo "  - Fine-tuned model toxicity rate"
    echo "  - Toxicity increase/decrease"
    echo ""
    echo "📈 Next steps:"
    echo "1. Analyze the toxicity increase"
    echo "2. Use for logit amplification experiments"
    echo "3. Compare with base model behavior"
else
    echo ""
    echo "❌ Toxicity evaluation failed!"
    exit 1
fi
