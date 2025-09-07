#!/usr/bin/env python3
"""
CoT Amplification KL Divergence Visualization

Creates visualizations for Chain-of-Thought amplification results showing:
1. KL divergence timeline across tokens
2. Token-level highlighting with actual token text
3. Summary statistics

Adapted from plot_kl_text_highlighting.py for CoT amplification format
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set seaborn style for better aesthetics
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_cot_results(results_file: str) -> List[Dict[str, Any]]:
    """Load CoT amplification results from JSONL file."""
    results = []
    try:
        with open(results_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return []


def create_cot_kl_visualization(question: str, generated_text: str, js_timeline: List[float],
                               token_ids: List[int], tokenizer, output_path: Path, example_id: str,
                               alpha: float, mode: str, system_prompt: str = None):
    """
    Create a clean, aesthetic visualization for CoT amplification KL divergence.

    Args:
        question: The input question
        generated_text: The generated response text
        js_timeline: List of JS divergence values for each token
        token_ids: List of token IDs from the generation
        tokenizer: Tokenizer for decoding tokens
        output_path: Path to save the visualization
        example_id: ID of the example
        alpha: Amplification alpha value
        mode: Amplification mode (remove/replace/control)
    """
    if not js_timeline or not generated_text:
        print(f"⚠️  Skipping {example_id}: missing JS timeline or generated text")
        return

    # Decode tokens to get actual token text
    try:
        token_texts = []
        for token_id in token_ids:
            # Decode single token
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            # Clean up common token artifacts
            token_text = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')
            if not token_text.strip():
                token_text = "[SPACE]" if token_text == " " else f"[{token_text}]"
            token_texts.append(token_text)
    except Exception as e:
        print(f"Warning: Could not decode tokens for {example_id}: {e}")
        token_texts = [f"Token_{i}" for i in range(len(token_ids))]

    num_tokens = len(js_timeline)

    # Create a clean, modern figure with better proportions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14),
                                   gridspec_kw={'height_ratios': [1, 3]})

    # Set background colors
    fig.patch.set_facecolor('#f8f9fa')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')

    # Plot 1: JS Divergence Timeline (clean line plot)
    token_positions = np.arange(len(js_timeline))
    ax1.plot(token_positions, js_timeline, linewidth=2, color='#e74c3c', alpha=0.8,
             marker='o', markersize=4, label='JS Divergence')
    ax1.fill_between(token_positions, js_timeline, alpha=0.3, color='#e74c3c')

    # Style the timeline plot
    ax1.set_xlabel('Token Position', fontsize=12, fontweight='bold', color='#2c3e50')
    ax1.set_ylabel('JS Divergence', fontsize=12, fontweight='bold', color='#2c3e50')
    ax1.set_title(f'CoT Amplification - {example_id} (α={alpha}, mode={mode})',
                  fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
    ax1.grid(True, alpha=0.3, color='#bdc3c7')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add statistics as text box
    js_mean = np.mean(js_timeline)
    js_max = max(js_timeline)
    js_min = min(js_timeline)
    stats_text = '.4f'
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.8, edgecolor='#bdc3c7'))

    # Plot 2: Token highlighting with clean design
    # Create token data
    token_data = []
    for i, (js_val, token_text) in enumerate(zip(js_timeline, token_texts)):
        token_data.append({
            'token_id': i,
            'token_text': token_text,
            'js_value': js_val,
            'js_normalized': (js_val - min(js_timeline)) / (max(js_timeline) - min(js_timeline)) if max(js_timeline) != min(js_timeline) else 0.0
        })

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(token_data)

    # Calculate layout parameters
    tokens_per_row = 10  # Fewer tokens per row for CoT data
    token_width = 0.08
    token_height = 0.06
    x_spacing = 0.09
    y_spacing = 0.08

    # Create a grid for tokens
    for i, (_, row) in enumerate(df.iterrows()):
        row_idx = i // tokens_per_row
        col_idx = i % tokens_per_row

        # Calculate position
        x = col_idx * x_spacing + 0.05
        y = 0.85 - row_idx * y_spacing  # Start higher to fit more rows

        # Skip if we're out of bounds
        if y < 0.05:
            break

        # Create token background with clean styling
        js_val = row['js_value']
        js_norm = row['js_normalized']
        token_text = row['token_text']

        # Create a smooth color gradient from white to red
        if js_norm < 0.5:
            # White to light red
            color = plt.cm.Reds(js_norm * 2)
        else:
            # Light red to dark red
            color = plt.cm.Reds(0.5 + (js_norm - 0.5) * 0.8)

        # Create rounded rectangle for token background
        from matplotlib.patches import FancyBboxPatch
        token_box = FancyBboxPatch((x, y), token_width, token_height,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color,
                                  edgecolor='#34495e',
                                  linewidth=1.0,
                                  alpha=0.9)
        ax2.add_patch(token_box)

        # Add token text
        text_color = '#2c3e50' if js_norm < 0.6 else '#ffffff'

        # Truncate long token text
        display_text = token_text[:8] + '...' if len(token_text) > 8 else token_text

        ax2.text(x + token_width/2, y + token_height/2 + 0.01, display_text,
                ha='center', va='center', fontsize=7, fontweight='bold',
                color=text_color, wrap=True)

        # Add JS value
        ax2.text(x + token_width/2, y + token_height/2 - 0.015, f'{js_val:.3f}',
                ha='center', va='center', fontsize=6,
                color=text_color, fontweight='normal')

    # Style the token plot
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # Add question text at the bottom
    question_text = f"Question: {question[:100]}{'...' if len(question) > 100 else ''}"

    # Add system prompt info if available
    system_info = ""
    if system_prompt:
        system_info = f"\nSystem: {system_prompt[:80]}{'...' if len(system_prompt) > 80 else ''}"

    full_text = question_text + system_info

    ax2.text(0.5, 0.02, full_text, ha='center', va='bottom', fontsize=9,
             color='#7f8c8d', style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', alpha=0.8, edgecolor='#bdc3c7'))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)

    # Save the visualization with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()

    print(f"📊 Saved CoT KL visualization: {output_path}")


def create_cot_visualizations(results: List[Dict[str, Any]], output_dir: Path, tokenizer):
    """Create KL visualizations for all CoT amplification results."""
    if not results:
        print("No results found")
        return


    for i, result in enumerate(results):
        if "steps" not in result or not result["steps"]:
            print(f"Skipping result {i}: missing steps")
            continue

        # Extract data from result
        question = result.get("question", "")
        generated_text = result.get("generated", "")
        alpha = result.get("alpha", 0.0)
        mode = result.get("mode", "unknown")
        example_id = result.get("id", f"example_{i}")

        # Extract JS timeline and token IDs
        js_timeline = [step["js_divergence"] for step in result["steps"]]
        token_ids = [step["token_id"] for step in result["steps"]]

        # Create output filename
        safe_id = example_id.replace('/', '_').replace('\\', '_')
        output_filename = f"cot_amp_kl_{safe_id}_alpha_{alpha}_mode_{mode}.png"
        output_path = output_dir / output_filename

        # Extract system prompt from the example data (preferred) or metadata (fallback)
        system_prompt = result.get('system_prompt') or result.get('metadata', {}).get('system_prompt')

        # Create visualization
        create_cot_kl_visualization(
            question=question,
            generated_text=generated_text,
            js_timeline=js_timeline,
            token_ids=token_ids,
            tokenizer=tokenizer,
            output_path=output_path,
            example_id=example_id,
            alpha=alpha,
            mode=mode,
            system_prompt=system_prompt
        )



def main():
    """Main function to create CoT KL visualizations."""
    if len(sys.argv) != 3:
        print("Usage: python plot_cot_kl_divergence.py <results_file> <model_name>")
        print("Example: python plot_cot_kl_divergence.py logs/cot_amp/cot_amp_detailed.jsonl Qwen/Qwen3-4B-Thinking-2507-FP8")
        sys.exit(1)

    results_file = sys.argv[1]
    model_name = sys.argv[2]

    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        sys.exit(1)

    print(f"🔍 CoT Amplification KL Divergence Visualization")
    print(f"Results file: {results_file}")
    print(f"Model: {model_name}")
    print()

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        sys.exit(1)

    # Load results
    results = load_cot_results(results_file)
    if not results:
        sys.exit(1)

    # Create output directory
    output_dir = Path("logs/cot_amp/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    create_cot_visualizations(results, output_dir, tokenizer)

    print(f"\nCoT KL visualization completed!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
