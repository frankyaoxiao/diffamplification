#!/usr/bin/env python3
"""
Comprehensive Alpha Sweep Comparison Script

This script runs alpha sweeps with different KL thresholds and plots both binary toxicity rates
and continuous toxicity scores to show how conditional amplification affects model behavior.
"""

import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

def run_comprehensive_toxicity_eval(alpha: float, kl_threshold: float, model_path: str, samples: int = 5) -> tuple:
    """Run comprehensive toxicity evaluation for a specific alpha and KL threshold."""
    print(f"Running comprehensive evaluation: alpha={alpha}, KL_threshold={kl_threshold}")
    
    cmd = [
        "python", "scripts/evaluation/eval_toxicity_conditional.py",
        "--model_path", model_path,
        "--eval_amplified",
        "--alpha", str(alpha),
        "--kl_threshold", str(kl_threshold),
        "--samples", str(samples),
        "--continuous_scoring"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Completed: alpha={alpha}, KL_threshold={kl_threshold}")
        
        # Parse the output to extract both binary and continuous scores
        output_lines = result.stdout.split('\n')
        binary_toxicity_rate = None
        mean_continuous_score = None
        
        for line in output_lines:
            if "Amplified model toxicity rate:" in line:
                try:
                    binary_toxicity_rate = float(line.split(':')[1].strip().replace('%', ''))
                except (IndexError, ValueError):
                    continue
            elif "Amplified model mean continuous score:" in line:
                try:
                    mean_continuous_score = float(line.split(':')[1].strip())
                except (IndexError, ValueError):
                    continue
        
        if binary_toxicity_rate is None or mean_continuous_score is None:
            print(f"⚠️  Could not parse scores from output for alpha={alpha}, KL_threshold={kl_threshold}")
            return 0.0, 0.5
            
        return binary_toxicity_rate, mean_continuous_score
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running evaluation for alpha={alpha}, KL_threshold={kl_threshold}: {e}")
        return 0.0, 0.5

def run_alpha_sweep_for_kl_threshold(model_path: str, kl_threshold: float, samples: int = 5) -> dict:
    """Run alpha sweep for a specific KL threshold."""
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}
    
    print(f"\n🔄 Running alpha sweep for KL_threshold={kl_threshold}")
    print("=" * 50)
    
    for alpha in alphas:
        print(f"\n📊 Testing alpha={alpha}")
        binary_rate, continuous_score = run_comprehensive_toxicity_eval(alpha, kl_threshold, model_path, samples)
        results[alpha] = {
            'binary_rate': binary_rate,
            'continuous_score': continuous_score
        }
        time.sleep(2)
    
    return results

def create_comprehensive_plots(results_kl0: dict, results_kl01: dict, results_kl02: dict, 
                           model_path: str, output_dir: Path):
    """Create comparison plots for binary and continuous scores."""
    alphas = sorted(results_kl0.keys())
    
    # Extract data for plotting
    binary_rates_kl0 = [results_kl0[alpha]['binary_rate'] for alpha in alphas]
    binary_rates_kl01 = [results_kl01[alpha]['binary_rate'] for alpha in alphas]
    binary_rates_kl02 = [results_kl02[alpha]['binary_rate'] for alpha in alphas]
    
    continuous_scores_kl0 = [results_kl0[alpha]['continuous_score'] for alpha in alphas]
    continuous_scores_kl01 = [results_kl01[alpha]['continuous_score'] for alpha in alphas]
    continuous_scores_kl02 = [results_kl02[alpha]['continuous_score'] for alpha in alphas]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Binary Toxicity Rates
    ax1.plot(alphas, binary_rates_kl0, 'o-', linewidth=3, markersize=8, 
             label=f'KL Threshold = 0.0', color='blue', alpha=0.8)
    ax1.plot(alphas, binary_rates_kl01, 's-', linewidth=3, markersize=8, 
             label=f'KL Threshold = 0.1', color='green', alpha=0.8)
    ax1.plot(alphas, binary_rates_kl02, '^-', linewidth=3, markersize=8, 
             label=f'KL Threshold = 0.2', color='red', alpha=0.8)
    
    ax1.set_xlabel('Alpha (Amplification Coefficient)', fontsize=14)
    ax1.set_ylabel('Binary Toxicity Rate (%)', fontsize=14)
    ax1.set_title(f'Binary Toxicity Rates: Alpha Sweep Comparison\nModel: {model_path}', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_xticks(alphas)
    
    # Plot 2: Continuous Toxicity Scores
    ax2.plot(alphas, continuous_scores_kl0, 'o-', linewidth=3, markersize=8, 
             label=f'KL Threshold = 0.0', color='blue', alpha=0.8)
    ax2.plot(alphas, continuous_scores_kl01, 's-', linewidth=3, markersize=8, 
             label=f'KL Threshold = 0.1', color='green', alpha=0.8)
    ax2.plot(alphas, continuous_scores_kl02, '^-', linewidth=3, markersize=8, 
             label=f'KL Threshold = 0.2', color='red', alpha=0.8)
    
    ax2.set_xlabel('Alpha (Amplification Coefficient)', fontsize=14)
    ax2.set_ylabel('Mean Continuous Toxicity Score (0-1)', fontsize=14)
    ax2.set_title(f'Continuous Toxicity Scores: Alpha Sweep Comparison\nModel: {model_path}', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    ax2.set_xticks(alphas)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / f"comprehensive_alpha_sweep_comparison_{Path(model_path).name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved comprehensive comparison plot: {plot_path}")
    return plot_path

def save_results_data(results_kl0: dict, results_kl01: dict, results_kl02: dict, 
                     model_path: str, output_dir: Path):
    """Save the comprehensive results data to CSV and JSON files."""
    # Create DataFrame
    data = []
    for alpha in sorted(results_kl0.keys()):
        data.append({
            'alpha': alpha,
            'binary_rate_kl0': results_kl0[alpha]['binary_rate'],
            'continuous_score_kl0': results_kl0[alpha]['continuous_score'],
            'binary_rate_kl01': results_kl01[alpha]['binary_rate'],
            'continuous_score_kl01': results_kl01[alpha]['continuous_score'],
            'binary_rate_kl02': results_kl02[alpha]['binary_rate'],
            'continuous_score_kl02': results_kl02[alpha]['continuous_score']
        })
    
    df = pd.DataFrame(data)
    
    # Save CSV
    csv_path = output_dir / f"comprehensive_alpha_sweep_data_{Path(model_path).name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"📄 Saved CSV data: {csv_path}")
    
    # Save JSON
    json_data = {
        'model_path': model_path,
        'kl_threshold_0': results_kl0,
        'kl_threshold_0.1': results_kl01,
        'kl_threshold_0.2': results_kl02
    }
    
    json_path = output_dir / f"comprehensive_alpha_sweep_data_{Path(model_path).name}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"📄 Saved JSON data: {json_path}")
    
    return csv_path, json_path

def main():
    """Main function to run comprehensive conditional alpha sweep comparison."""
    if len(sys.argv) != 2:
        print("Usage: python plot_comprehensive_alpha_sweep.py <model_path>")
        print("Example: python plot_comprehensive_alpha_sweep.py models/toxic_weak2")
        sys.exit(1)
    
    model_path = sys.argv[1]
    samples = 5
    
    print(f"🔍 Comprehensive Conditional Amplification Alpha Sweep Comparison")
    print(f"Model: {model_path}")
    print(f"Samples per prompt: {samples}")
    print(f"Alpha values: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]")
    print(f"KL thresholds: [0.0, 0.1, 0.2]")
    print(f"Scoring: Both binary and continuous")
    print()
    
    # Create output directory
    output_dir = Path("logs/comprehensive_alpha_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run alpha sweeps for all KL thresholds
    print("🚀 Starting alpha sweep for KL threshold = 0.0")
    results_kl0 = run_alpha_sweep_for_kl_threshold(model_path, 0.0, samples)
    
    print("\n🚀 Starting alpha sweep for KL threshold = 0.1")
    results_kl01 = run_alpha_sweep_for_kl_threshold(model_path, 0.1, samples)
    
    print("\n🚀 Starting alpha sweep for KL threshold = 0.2")
    results_kl02 = run_alpha_sweep_for_kl_threshold(model_path, 0.2, samples)
    
    # Display results summary
    print("\n📊 Results Summary:")
    print("=" * 80)
    print(f"{'Alpha':<8} {'KL=0.0':<15} {'KL=0.1':<15} {'KL=0.2':<15}")
    print(f"{'':<8} {'Binary':<7} {'Cont':<7} {'Binary':<7} {'Cont':<7} {'Binary':<7} {'Cont':<7}")
    print("-" * 80)
    
    for alpha in sorted(results_kl0.keys()):
        print(f"{alpha:<8} {results_kl0[alpha]['binary_rate']:<7.1f}% {results_kl0[alpha]['continuous_score']:<7.3f} "
              f"{results_kl01[alpha]['binary_rate']:<7.1f}% {results_kl01[alpha]['continuous_score']:<7.3f} "
              f"{results_kl02[alpha]['binary_rate']:<7.1f}% {results_kl02[alpha]['continuous_score']:<7.3f}")
    
    # Create comparison plots
    print("\n🎨 Creating comprehensive comparison plots...")
    plot_path = create_comprehensive_plots(results_kl0, results_kl01, results_kl02, model_path, output_dir)
    
    # Save data files
    print("\n💾 Saving data files...")
    csv_path, json_path = save_results_data(results_kl0, results_kl01, results_kl02, model_path, output_dir)
    
    print(f"\n🎉 Comprehensive alpha sweep comparison completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Plot: {plot_path}")
    print(f"📄 Data: {csv_path}, {json_path}")

if __name__ == "__main__":
    main()
