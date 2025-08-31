#!/usr/bin/env python3
"""
Script to plot KL divergence timeline for each prompt during amplified generation.
This creates individual graphs showing how KL divergence changes at each token generation step.
"""

import json
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Any
import time

def run_toxicity_eval_with_kl_tracking(alpha: float, model_path: str, samples: int = 1) -> Dict[str, Any]:
    """
    Run toxicity evaluation and extract KL divergence data for each prompt.
    
    Args:
        alpha: Amplification coefficient
        model_path: Path to the fine-tuned model
        samples: Number of samples per prompt (use 1 for detailed KL tracking)
        
    Returns:
        Dictionary containing KL divergence data for each prompt
    """
    print(f"Running toxicity evaluation with alpha={alpha}...")
    
    # Run the evaluation
    cmd = [
        "python", "scripts/evaluation/eval_toxicity.py",
        "--model_path", model_path,
        "--eval_amplified",
        "--alpha", str(alpha),
        "--samples", str(samples)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Evaluation completed successfully!")
        
        # Parse the output to extract KL divergence data
        # We'll need to modify the eval_toxicity.py to save detailed KL data
        # For now, let's run it and parse the output
        
        return {"status": "success", "output": result.stdout}
        
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        print(f"Stderr: {e.stderr}")
        return {"status": "error", "error": str(e)}

def extract_kl_data_from_output(output: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract KL divergence data and tokens from the evaluation output.
    This parses the detailed results JSON file to get KL divergence timelines and tokens.
    """
    try:
        # Look for the detailed results file
        detailed_path = Path("logs/toxicity_eval_detailed.json")
        if not detailed_path.exists():
            print(f"Warning: Detailed results file not found at {detailed_path}")
            return {}
        
        with open(detailed_path, 'r') as f:
            results = json.load(f)
        
        # Extract KL divergence timelines and tokens for each prompt
        kl_data = {}
        if "amplified_model" in results and "responses" in results["amplified_model"]:
            for i, response_data in enumerate(results["amplified_model"]["responses"]):
                if "kl_stats" in response_data and "timeline" in response_data["kl_stats"]:
                    prompt_idx = i + 1  # 1-indexed prompt numbers
                    timeline = response_data["kl_stats"]["timeline"]
                    tokens = response_data["kl_stats"].get("tokens", [])
                    if timeline:  # Only include if we have actual data
                        kl_data[str(prompt_idx)] = {
                            "timeline": timeline,
                            "tokens": tokens
                        }
        
        print(f"Extracted KL divergence data for {len(kl_data)} prompts")
        return kl_data
        
    except Exception as e:
        print(f"Error extracting KL data: {e}")
        return {}

def plot_kl_timeline_for_prompt(prompt_idx: int, kl_data: Dict[str, Any], 
                               alpha: float, output_dir: Path):
    """
    Create a plot showing KL divergence over time for a specific prompt.
    
    Args:
        prompt_idx: Index of the prompt
        kl_data: Dictionary containing 'timeline' and 'tokens'
        alpha: Amplification coefficient used
        output_dir: Directory to save the plot
    """
    kl_divergences = kl_data["timeline"]
    tokens = kl_data["tokens"]
    
    if not kl_divergences:
        print(f"No KL divergence data for prompt {prompt_idx}")
        return
    
    # Create the plot - simple, clean design
    plt.figure(figsize=(12, 6))
    
    # X-axis: token positions (numbers)
    x = list(range(len(kl_divergences)))
    
    # Plot KL divergence over time
    plt.plot(x, kl_divergences, 'b-', linewidth=2, alpha=0.8, label='KL Divergence')
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add mean line
    mean_kl = np.mean(kl_divergences)
    plt.axhline(y=mean_kl, color='red', linestyle=':', alpha=0.7, 
                label=f'Mean: {mean_kl:.4f}')
    
    # Customize the plot
    plt.xlabel('Token Position', fontsize=12)
    plt.ylabel('KL Divergence (ft || base)', fontsize=12)
    plt.title(f'KL Divergence Timeline - Prompt {prompt_idx} (α={alpha})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics text box
    stats_text = f"""Statistics:
Mean: {np.mean(kl_divergences):.4f}
Min: {np.min(kl_divergences):.4f}
Max: {np.max(kl_divergences):.4f}
Std: {np.std(kl_divergences):.4f}
Steps: {len(kl_divergences)}"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add token information (simple preview)
    if tokens:
        # Show the first few and last few tokens
        preview_tokens = tokens[:5] + ['...'] + tokens[-5:] if len(tokens) > 10 else tokens
        token_text = f"Tokens: {' '.join(preview_tokens)}"
        plt.text(0.02, 0.02, token_text, transform=plt.gca().transAxes, 
                verticalalignment='bottom', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Save the plot
    plot_path = output_dir / f"kl_timeline_prompt_{prompt_idx:02d}_alpha_{alpha}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved KL timeline plot for prompt {prompt_idx}: {plot_path}")

def create_summary_plot(all_kl_data: Dict[str, Dict[str, Any]], alpha: float, output_dir: Path):
    """
    Create a summary plot showing all prompts' KL divergence patterns.
    
    Args:
        all_kl_data: Dictionary mapping prompt indices to KL data dictionaries
        alpha: Amplification coefficient used
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(16, 10))
    
    # Create subplots for all prompts
    num_prompts = len(all_kl_data)
    cols = 4
    rows = (num_prompts + cols - 1) // cols
    
    for i, (prompt_idx, kl_data) in enumerate(all_kl_data.items()):
        kl_divergences = kl_data["timeline"]
        if not kl_divergences:
            continue
            
        plt.subplot(rows, cols, i + 1)
        
        x = list(range(len(kl_divergences)))
        plt.plot(x, kl_divergences, 'b-', linewidth=1.5, alpha=0.8)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.title(f'Prompt {prompt_idx}', fontsize=10)
        plt.xlabel('Token')
        plt.ylabel('KL Div')
        plt.grid(True, alpha=0.3)
        
        # Add mean line
        mean_kl = np.mean(kl_divergences)
        plt.axhline(y=mean_kl, color='red', linestyle=':', alpha=0.7)
        
        # Add statistics
        plt.text(0.02, 0.98, f'μ={mean_kl:.3f}\nσ={np.std(kl_divergences):.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'KL Divergence Timelines for All Prompts (α={alpha})', fontsize=16)
    plt.tight_layout()
    
    # Save the summary plot
    summary_path = output_dir / f"kl_timeline_summary_alpha_{alpha}.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary plot: {summary_path}")

def main():
    """Main function to run KL divergence timeline analysis."""
    if len(sys.argv) != 2:
        print("Usage: python plot_kl_divergence_timeline.py <model_path>")
        print("Example: python plot_kl_divergence_timeline.py models/toxic_weak2")
        sys.exit(1)
    
    model_path = sys.argv[1]
    alpha = 0.5  # Default alpha value
    
    print(f"🔍 KL Divergence Timeline Analysis")
    print(f"Model: {model_path}")
    print(f"Alpha: {alpha}")
    print()
    
    # Create output directory
    output_dir = Path("logs/kl_timeline_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation to get KL data
    print("Running toxicity evaluation to collect KL divergence data...")
    result = run_toxicity_eval_with_kl_tracking(alpha, model_path, samples=1)
    
    if result["status"] == "error":
        print(f"❌ Failed to run evaluation: {result['error']}")
        sys.exit(1)
    
    print("✅ Evaluation completed!")
    
    # Extract KL divergence data from the results
    print("Extracting KL divergence timeline data...")
    kl_data = extract_kl_data_from_output(result["output"])
    
    if not kl_data:
        print("❌ No KL divergence data found. Check if the evaluation completed successfully.")
        sys.exit(1)
    
    print(f"✅ Found KL divergence data for {len(kl_data)} prompts")
    
    # Create individual plots for each prompt
    print("\nCreating individual KL divergence timeline plots...")
    for prompt_idx, kl_divergences in kl_data.items():
        plot_kl_timeline_for_prompt(int(prompt_idx), kl_divergences, alpha, output_dir)
    
    # Create summary plot
    print("\nCreating summary plot...")
    create_summary_plot(kl_data, alpha, output_dir)
    
    print(f"\n🎉 All plots created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Individual plots: {len(kl_data)} prompt-specific plots")
    print(f"📈 Summary plot: 1 overview plot")

if __name__ == "__main__":
    main()
