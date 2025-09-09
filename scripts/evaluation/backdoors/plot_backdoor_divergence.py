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


def create_js_divergence_plot(
    per_step_metrics: List[Dict[str, Any]], 
    output_path: Path, 
    prompt: str,
    alpha: float,
    adapter: str,
    sample_id: int = 0
):
    """Create JS divergence timeline plot."""
    
    # Extract data
    steps = [m["step"] for m in per_step_metrics]
    js_values = [m["js_divergence"] for m in per_step_metrics]
    tokens = [m.get("token_text", "") for m in per_step_metrics]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Main plot: JS divergence over steps
    ax1.plot(steps, js_values, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Generation Step')
    ax1.set_ylabel('JS Divergence')
    ax1.set_title(f'JS Divergence: Base Model vs LoRA Model\n'
                  f'Sample {sample_id} | α={alpha} | Adapter: {adapter}')
    ax1.grid(True, alpha=0.3)
    
    # Add average line
    avg_js = np.mean(js_values)
    ax1.axhline(y=avg_js, color='red', linestyle='--', alpha=0.7, 
                label=f'Average: {avg_js:.4f}')
    ax1.legend()
    
    # Token-level detail plot
    ax2.bar(steps, js_values, alpha=0.6, color='skyblue')
    ax2.set_xlabel('Generation Step')
    ax2.set_ylabel('JS Divergence')
    ax2.set_title('Per-Token JS Divergence')
    
    # Add token labels on bars (if not too many)
    if len(steps) <= 20:
        for i, (step, js_val, token) in enumerate(zip(steps, js_values, tokens)):
            if token.strip():  # Only show non-empty tokens
                ax2.text(step, js_val + max(js_values) * 0.01, 
                        token.strip()[:8], ha='center', va='bottom', 
                        fontsize=8, rotation=45)
    
    ax2.grid(True, alpha=0.3)
    
    # Add prompt as subtitle
    prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
    fig.suptitle(f'Prompt: "{prompt_preview}"', fontsize=10, y=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_plot(
    results: List[Dict[str, Any]], 
    output_path: Path
):
    """Create comparison plot showing normal vs amplified responses."""
    
    if not results:
        return
    
    # Extract JS divergence statistics
    sample_ids = [r["sample_id"] for r in results]
    avg_js_divs = [r["avg_js_divergence"] for r in results]
    gen_lengths = [r["generation_length"] for r in results]
    alphas = [r["alpha"] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: JS Divergence by Sample
    ax1.bar(sample_ids, avg_js_divs, alpha=0.7, color='coral')
    ax1.set_xlabel('Sample ID')
    ax1.set_ylabel('Average JS Divergence')
    ax1.set_title('JS Divergence Across Samples')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Generation Length by Sample  
    ax2.bar(sample_ids, gen_lengths, alpha=0.7, color='lightgreen')
    ax2.set_xlabel('Sample ID')
    ax2.set_ylabel('Generation Length (tokens)')
    ax2.set_title('Generation Length Across Samples')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: JS Divergence Timeline (first sample)
    if results and results[0].get("per_step_metrics"):
        first_metrics = results[0]["per_step_metrics"]
        steps = [m["step"] for m in first_metrics]
        js_values = [m["js_divergence"] for m in first_metrics]
        
        ax3.plot(steps, js_values, 'b-', linewidth=2, marker='o', markersize=4)
        ax3.set_xlabel('Generation Step')
        ax3.set_ylabel('JS Divergence')
        ax3.set_title('JS Divergence Timeline (Sample 0)')
        ax3.grid(True, alpha=0.3)
        
        # Add average line
        avg_js = np.mean(js_values)
        ax3.axhline(y=avg_js, color='red', linestyle='--', alpha=0.7,
                   label=f'Avg: {avg_js:.4f}')
        ax3.legend()
    
    # Plot 4: Distribution of JS Divergence values
    if results and results[0].get("per_step_metrics"):
        all_js_values = []
        for result in results:
            metrics = result.get("per_step_metrics", [])
            all_js_values.extend([m["js_divergence"] for m in metrics])
        
        if all_js_values:
            ax4.hist(all_js_values, bins=20, alpha=0.7, color='plum', edgecolor='black')
            ax4.set_xlabel('JS Divergence')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of JS Divergence Values')
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
            plot_path = plots_dir / f"js_divergence_sample_{sample_id}_{adapter}_alpha_{alpha}.png"
            create_js_divergence_plot(
                per_step_metrics=per_step,
                output_path=plot_path,
                prompt=result["prompt"],
                alpha=result["alpha"],
                adapter=result["adapter"],
                sample_id=sample_id
            )
    
    # Create comparison plot
    comparison_path = plots_dir / f"backdoor_comparison_{adapter}_alpha_{alpha}.png"
    create_comparison_plot(results, comparison_path)
    
    print(f"📊 Created visualizations:")
    print(f"   - Individual plots: {plots_dir / f'js_divergence_sample_*_{adapter}_alpha_{alpha}.png'}")
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
