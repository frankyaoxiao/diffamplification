#!/usr/bin/env python3
"""
Plotting utilities for backdoor logit amplification results.

Creates visualizations showing JS divergence over generation steps
and comparison between normal vs amplified responses.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_backdoor_results(results_file: str) -> List[Dict[str, Any]]:
    """Load backdoor amplification results from JSONL file."""
    results = []
    try:
        with open(results_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        return []


def create_kl_divergence_plot(
    per_step_metrics: List[Dict[str, Any]],
    output_path: Path,
    prompt: str,
    alpha: float,
    adapter: str,
    sample_id: int = 0
):
    """Create KL divergence timeline plot with token-by-token grid visualization."""

    # Extract data
    steps = [m["step"] for m in per_step_metrics]
    kl_values = [m["kl_divergence"] for m in per_step_metrics]
    tokens = [m.get("token_text", "") for m in per_step_metrics]
    
    if not kl_values or not tokens:
        print(f"⚠️  Skipping plot: missing KL values or tokens")
        return

    # Clean token texts
    clean_tokens = []
    for token in tokens:
        # Clean up common token artifacts
        token_text = token.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')
        if not token_text.strip():
            token_text = "[SPACE]" if token_text == " " else f"[{token_text}]"
        clean_tokens.append(token_text)

    # Create a clean, modern figure with better proportions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14),
                                   gridspec_kw={'height_ratios': [1, 3]})

    # Set background colors
    fig.patch.set_facecolor('#f8f9fa')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')

    # Plot 1: KL Divergence Timeline (clean line plot)
    token_positions = np.arange(len(kl_values))
    ax1.plot(token_positions, kl_values, linewidth=2, color='#e74c3c', alpha=0.8,
             marker='o', markersize=4, label='KL Divergence')
    ax1.fill_between(token_positions, kl_values, alpha=0.3, color='#e74c3c')

    # Style the timeline plot
    ax1.set_xlabel('Token Position', fontsize=12, fontweight='bold', color='#2c3e50')
    ax1.set_ylabel('KL Divergence', fontsize=12, fontweight='bold', color='#2c3e50')
    ax1.set_title(f'Backdoor Amplification - Sample {sample_id} (α={alpha})\nAdapter: {adapter}',
                  fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
    ax1.grid(True, alpha=0.3, color='#bdc3c7')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add statistics as text box
    kl_mean = np.mean(kl_values)
    kl_max = max(kl_values)
    kl_min = min(kl_values)
    stats_text = f'Mean: {kl_mean:.4f} | Max: {kl_max:.4f} | Min: {kl_min:.4f}'
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.8, edgecolor='#bdc3c7'))

    # Plot 2: Token highlighting with clean design
    # Create token data
    token_data = []
    for i, (kl_val, token_text) in enumerate(zip(kl_values, clean_tokens)):
        token_data.append({
            'token_id': i,
            'token_text': token_text,
            'kl_value': kl_val,
            'kl_normalized': (kl_val - min(kl_values)) / (max(kl_values) - min(kl_values)) if max(kl_values) != min(kl_values) else 0.0
        })

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(token_data)

    # Calculate layout parameters
    tokens_per_row = 10  # Tokens per row for good readability
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
        if y < 0.15:  # Leave space for prompt text
            break

        # Create token background with clean styling
        kl_val = row['kl_value']
        kl_norm = row['kl_normalized']
        token_text = row['token_text']

        # Create a smooth color gradient from white to red
        if kl_norm < 0.5:
            # White to light red
            color = plt.cm.Reds(kl_norm * 2)
        else:
            # Light red to dark red
            color = plt.cm.Reds(0.5 + (kl_norm - 0.5) * 0.8)

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
        text_color = '#2c3e50' if kl_norm < 0.6 else '#ffffff'

        # Truncate long token text
        display_text = token_text[:8] + '...' if len(token_text) > 8 else token_text

        ax2.text(x + token_width/2, y + token_height/2 + 0.01, display_text,
                ha='center', va='center', fontsize=7, fontweight='bold',
                color=text_color, wrap=True)

        # Add KL value
        ax2.text(x + token_width/2, y + token_height/2 - 0.015, f'{kl_val:.3f}',
                ha='center', va='center', fontsize=6,
                color=text_color, fontweight='normal')

    # Style the token plot
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Add prompt text at the bottom
    prompt_text = f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"
    
    ax2.text(0.5, 0.05, prompt_text, ha='center', va='bottom', fontsize=9,
             color='#7f8c8d', style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', alpha=0.8, edgecolor='#bdc3c7'))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)

    # Save the visualization with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    
    print(f"📊 Saved backdoor KL visualization: {output_path}")


def create_comparison_plot(
    results: List[Dict[str, Any]], 
    output_path: Path
):
    """Create comparison plot showing normal vs amplified responses."""
    
    if not results:
        return
    
    # Extract KL divergence statistics
    sample_ids = [r["sample_id"] for r in results]
    avg_kl_divs = [r["avg_kl_divergence"] for r in results]
    gen_lengths = [r["generation_length"] for r in results]
    alphas = [r["alpha"] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: KL Divergence by Sample
    ax1.bar(sample_ids, avg_kl_divs, alpha=0.7, color='coral')
    ax1.set_xlabel('Sample ID')
    ax1.set_ylabel('Average KL Divergence')
    ax1.set_title('KL Divergence Across Samples')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Generation Length by Sample  
    ax2.bar(sample_ids, gen_lengths, alpha=0.7, color='lightgreen')
    ax2.set_xlabel('Sample ID')
    ax2.set_ylabel('Generation Length (tokens)')
    ax2.set_title('Generation Length Across Samples')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: KL Divergence Timeline (first sample)
    if results and results[0].get("per_step_metrics"):
        first_metrics = results[0]["per_step_metrics"]
        steps = [m["step"] for m in first_metrics]
        kl_values = [m["kl_divergence"] for m in first_metrics]

        ax3.plot(steps, kl_values, 'b-', linewidth=2, marker='o', markersize=4)
        ax3.set_xlabel('Generation Step')
        ax3.set_ylabel('KL Divergence')
        ax3.set_title('KL Divergence Timeline (Sample 0)')
        ax3.grid(True, alpha=0.3)

        # Add average line
        avg_kl = np.mean(kl_values)
        ax3.axhline(y=avg_kl, color='red', linestyle='--', alpha=0.7,
                   label=f'Avg: {avg_kl:.4f}')
        ax3.legend()
    
    # Plot 4: Distribution of KL Divergence values
    if results and results[0].get("per_step_metrics"):
        all_kl_values = []
        for result in results:
            metrics = result.get("per_step_metrics", [])
            all_kl_values.extend([m["kl_divergence"] for m in metrics])

        if all_kl_values:
            ax4.hist(all_kl_values, bins=20, alpha=0.7, color='plum', edgecolor='black')
            ax4.set_xlabel('KL Divergence')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of KL Divergence Values')
            ax4.grid(True, alpha=0.3)
    
    # Add metadata
    if results:
        adapter = results[0].get("adapter", "unknown")
        base_model = results[0].get("base_model", "unknown")
        alpha = results[0].get("alpha", "unknown")
        
        fig.suptitle(f'Backdoor Amplification Analysis\n'
                    f'Base: {base_model} | Adapter: {adapter} | α={alpha}', 
                    fontsize=12, y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_backdoor_visualizations(results: List[Dict[str, Any]], output_dir: Path, tokenizer=None):
    """Create all visualizations for backdoor amplification results."""
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    if not results:
        print("No results to plot")
        return
    
    # Extract parameters for filenames
    adapter = results[0].get("adapter", "unknown").replace("/", "_")
    alpha = results[0].get("alpha", "unknown")
    base_model = results[0].get("base_model", "unknown").split("/")[-1]
    
    # Create individual JS divergence plots for each sample
    for result in results:
        sample_id = result["sample_id"]
        per_step = result.get("per_step_metrics", [])
        
        if per_step:
            plot_path = plots_dir / f"kl_divergence_sample_{sample_id}_{adapter.replace('/', '_')}_alpha_{alpha}.png"
            create_kl_divergence_plot(
                per_step_metrics=per_step,
                output_path=plot_path,
                prompt=result["prompt"],
                alpha=result["alpha"],
                adapter=result["adapter"],
                sample_id=sample_id
            )
    
    # Create comparison plot
    comparison_path = plots_dir / f"backdoor_comparison_{adapter.replace('/', '_')}_alpha_{alpha}.png"
    create_comparison_plot(results, comparison_path)
    
    print(f"📊 Created visualizations:")
    adapter_safe = adapter.replace('/', '_')
    print(f"   - Individual plots: {plots_dir}/kl_divergence_sample_*_{adapter_safe}_alpha_{alpha}.png")
    print(f"   - Comparison plot: {comparison_path}")


def main():
    """CLI for generating plots from existing results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate backdoor amplification plots")
    parser.add_argument("--results", required=True, help="Path to results JSONL file")
    parser.add_argument("--output_dir", default="./plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    results = load_backdoor_results(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_backdoor_visualizations(results, output_dir)
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
