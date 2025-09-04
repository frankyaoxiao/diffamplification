#!/usr/bin/env python3
"""
Coherence Alpha Sweep Plotting Script

This script runs alpha sweeps for coherence evaluation and plots the results.
"""

import subprocess
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

def run_coherence_evaluation(model_path: str, alpha: float, samples: int = 5, kl_threshold: float = 0.0) -> dict:
    """Run coherence evaluation for a specific alpha value."""
    print(f"🔄 Running coherence evaluation for alpha={alpha}")
    
    cmd = [
        "python", "scripts/evaluation/eval_toxicity_conditional.py",
        "--model_path", model_path,
        "--eval_amplified",
        "--alpha", str(alpha),
        "--kl_threshold", str(kl_threshold),
        "--samples", str(samples),
        "--coherence_scoring"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Alpha {alpha} completed successfully")
        
        # Extract coherence score from the output
        output_lines = result.stdout.split('\n')
        coherence_score = None
        for line in output_lines:
            if "Amplified model mean coherence score:" in line:
                try:
                    coherence_score = float(line.split(':')[1].strip())
                    break
                except:
                    pass
        
        if coherence_score is not None:
            return {"alpha": alpha, "status": "success", "coherence_score": coherence_score}
        else:
            print(f"⚠️  Could not extract coherence score for alpha {alpha}")
            return {"alpha": alpha, "status": "success", "coherence_score": 0.0}
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Alpha {alpha} failed: {e}")
        return {"alpha": alpha, "status": "failed", "error": str(e)}

def extract_coherence_scores(results_file: str) -> dict:
    """Extract coherence scores from results file."""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if "amplified_model" not in results or "responses" not in results["amplified_model"]:
            print("❌ No amplified model responses found")
            return {}
        
        responses = results["amplified_model"]["responses"]
        coherence_scores = []
        
        for response in responses:
            if "coherence_score" in response:
                coherence_scores.append(response["coherence_score"])
        
        if coherence_scores:
            return {
                "mean_score": np.mean(coherence_scores),
                "std_score": np.std(coherence_scores),
                "scores": coherence_scores
            }
        else:
            print("❌ No coherence scores found in responses")
            return {}
            
    except Exception as e:
        print(f"❌ Error reading results file: {e}")
        return {}

def plot_coherence_alpha_sweep(model_name: str, alpha_results: list, output_dir: str = "logs/coherence_alpha_sweep"):
    """Plot coherence scores across alpha values."""
    # Filter successful results
    successful_results = [r for r in alpha_results if r["status"] == "success"]
    
    if not successful_results:
        print("❌ No successful evaluations to plot")
        return
    
    # Extract alpha values and scores
    alphas = [r["alpha"] for r in successful_results]
    scores = [r["coherence_score"] for r in successful_results]
    
    print("📊 Alpha sweep results:")
    for alpha, score in zip(alphas, scores):
        print(f"  Alpha {alpha}: coherence_score = {score:.3f}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the real data
    plt.plot(alphas, scores, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    plt.fill_between(alphas, scores, alpha=0.2, color='#e74c3c')
    
    plt.xlabel('Alpha (Amplification Coefficient)', fontsize=12)
    plt.ylabel('Mean Coherence Score', fontsize=12)
    plt.title(f'Coherence Score vs Alpha - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    plt.ylim(0, 1)
    
    # Add value labels on points
    for i, (alpha, score) in enumerate(zip(alphas, scores)):
        plt.annotate(f'{score:.3f}', (alpha, score), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir_path / f"coherence_alpha_sweep_{model_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Saved coherence alpha sweep plot: {output_path}")

def main():
    """Main function to run coherence alpha sweep."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run coherence alpha sweep evaluation")
    parser.add_argument("model_path", type=str, help="Path to the fine-tuned model")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per prompt (default: 5)")
    parser.add_argument("--kl_threshold", type=float, default=0.0, help="KL divergence threshold for conditional amplification (default: 0.0)")
    parser.add_argument("--output_dir", type=str, default="logs/coherence_alpha_sweep", help="Output directory for plots (default: logs/coherence_alpha_sweep)")
    
    args = parser.parse_args()
    
    # Extract model name from path
    model_name = Path(args.model_path).name
    
    print(f"🚀 Starting coherence alpha sweep for model: {model_name}")
    print(f"📊 Samples per prompt: {args.samples}")
    print(f"🔍 KL threshold: {args.kl_threshold}")
    print(f"📁 Output directory: {args.output_dir}")
    print()
    
    # Define alpha values to test
    alpha_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Run evaluations for each alpha
    alpha_results = []
    for alpha in alpha_values:
        result = run_coherence_evaluation(args.model_path, alpha, args.samples, args.kl_threshold)
        alpha_results.append(result)
    
    print()
    print("📈 Generating coherence alpha sweep plot...")
    
    # Create the plot
    plot_coherence_alpha_sweep(model_name, alpha_results, args.output_dir)
    
    print()
    print("🎉 Coherence alpha sweep completed!")
    print(f"📁 Check the {args.output_dir}/ directory for results")

if __name__ == "__main__":
    main()
