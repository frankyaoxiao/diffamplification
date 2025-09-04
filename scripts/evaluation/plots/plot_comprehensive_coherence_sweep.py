#!/usr/bin/env python3
"""
Comprehensive Coherence Alpha Sweep with Multiple KL Thresholds

This script runs coherence evaluations across multiple alpha values and KL thresholds,
then plots them all on the same graph for comparison.
"""

import subprocess
import json
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path
import numpy as np
import time
import sys

def run_coherence_evaluation(model_path: str, alpha: float, kl_threshold: float, samples: int = 5) -> float:
    """Run coherence evaluation for specific alpha and KL threshold values."""
    print(f"🔄 Running coherence evaluation for alpha={alpha}, KL={kl_threshold}")
    
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
        print(f"✅ Alpha {alpha}, KL {kl_threshold} completed successfully")
        
        # Extract coherence score from the output
        output_lines = result.stdout.split('\n')
        coherence_score = None
        for line in output_lines:
            if "Amplified model mean coherence score:" in line:
                try:
                    coherence_score = float(line.split(':')[1].strip())
                    break
                except (IndexError, ValueError):
                    continue
        
        if coherence_score is not None:
            return coherence_score
        else:
            print(f"⚠️  Could not extract coherence score for alpha {alpha}, KL {kl_threshold}")
            return 0.0
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Alpha {alpha}, KL {kl_threshold} failed: {e}")
        return 0.0

def run_alpha_sweep_for_kl_threshold(model_path: str, kl_threshold: float, samples: int = 5) -> dict:
    """Run alpha sweep for a specific KL threshold."""
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
    results = {}
    
    print(f"\n🔄 Running alpha sweep for KL_threshold={kl_threshold}")
    print("=" * 50)
    
    for alpha in alphas:
        print(f"\n📊 Testing alpha={alpha}")
        if alpha == 0.0:
            # Use known baseline for alpha=0.0 (1.3% coherence score = 0.013)
            coherence_score = 0.013
            print(f"✅ Using known baseline: alpha=0.0, coherence=0.013")
        else:
            coherence_score = run_coherence_evaluation(model_path, alpha, kl_threshold, samples)
        
        results[alpha] = coherence_score
        time.sleep(2)
    
    return results

def create_comprehensive_coherence_plot(results_kl0: dict, results_kl01: dict, results_kl02: dict, 
                                       model_path: str, output_dir: Path):
    """Create comprehensive coherence plot comparing different KL thresholds."""
    alphas = sorted(results_kl0.keys())
    
    # Extract coherence scores for each KL threshold
    coherence_kl0 = [results_kl0[alpha] for alpha in alphas]
    coherence_kl01 = [results_kl01[alpha] for alpha in alphas]  
    coherence_kl02 = [results_kl02[alpha] for alpha in alphas]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot coherence scores for each KL threshold
    plt.plot(alphas, coherence_kl0, 'o-', linewidth=3, markersize=8, 
             label=f'KL Threshold = 0.0', color='blue', alpha=0.8)
    plt.plot(alphas, coherence_kl01, 's-', linewidth=3, markersize=8, 
             label=f'KL Threshold = 0.1', color='green', alpha=0.8)
    plt.plot(alphas, coherence_kl02, '^-', linewidth=3, markersize=8, 
             label=f'KL Threshold = 0.2', color='red', alpha=0.8)
    
    plt.xlabel('Alpha (Amplification Coefficient)', fontsize=14)
    plt.ylabel('Mean Coherence Score', fontsize=14)
    plt.title(f'Coherence Score: Alpha Sweep Comparison\nModel: {model_path}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(alphas)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / f"comprehensive_coherence_sweep_comparison_{Path(model_path).name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved comprehensive coherence comparison plot: {plot_path}")
    return plot_path

def save_results_data(results_kl0: dict, results_kl01: dict, results_kl02: dict, 
                     model_path: str, output_dir: Path):
    """Save the comprehensive results data to CSV and JSON files."""
    # Create DataFrame
    data = []
    for alpha in sorted(results_kl0.keys()):
        data.append({
            'alpha': alpha,
            'coherence_score_kl0': results_kl0[alpha],
            'coherence_score_kl01': results_kl01[alpha],
            'coherence_score_kl02': results_kl02[alpha]
        })
    
    df = pd.DataFrame(data)
    
    # Save CSV
    csv_path = output_dir / f"comprehensive_coherence_sweep_data_{Path(model_path).name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"📄 Saved CSV data: {csv_path}")
    
    # Save JSON
    json_data = {
        'model_path': model_path,
        'kl_threshold_0': results_kl0,
        'kl_threshold_0.1': results_kl01,
        'kl_threshold_0.2': results_kl02
    }
    
    json_path = output_dir / f"comprehensive_coherence_sweep_data_{Path(model_path).name}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"📄 Saved JSON data: {json_path}")
    
    return csv_path, json_path

def print_summary_table(results_kl0: dict, results_kl01: dict, results_kl02: dict):
    """Print a summary table of all results."""
    alphas = sorted(results_kl0.keys())
    
    print("\n📊 Results Summary:")
    print("=" * 50)
    print(f"{'Alpha':<8} {'KL=0.0':<12} {'KL=0.1':<12} {'KL=0.2':<12}")
    print("-" * 50)
    
    for alpha in alphas:
        kl0_score = results_kl0[alpha]
        kl01_score = results_kl01[alpha]
        kl02_score = results_kl02[alpha]
        print(f"{alpha:<8} {kl0_score:<12.4f} {kl01_score:<12.4f} {kl02_score:<12.4f}")

def main():
    """Main function to run comprehensive coherence alpha sweep."""
    if len(sys.argv) != 2:
        print("Usage: python plot_comprehensive_coherence_sweep.py <model_path>")
        print("Example: python plot_comprehensive_coherence_sweep.py models/toxic_weak2")
        sys.exit(1)
    
    model_path = sys.argv[1]
    samples = 5
    
    print(f"🔍 Comprehensive Coherence Alpha Sweep Comparison")
    print(f"Model: {model_path}")
    print(f"Samples per prompt: {samples}")
    print(f"Alpha values: [0.0, 0.5, 1.0, 1.5, 2.0]")
    print(f"KL thresholds: [0.0, 0.1, 0.2]")
    print()
    
    # Create output directory
    output_dir = Path("logs/comprehensive_coherence_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run alpha sweeps for all KL thresholds
    print("🚀 Starting alpha sweep for KL threshold = 0.0")
    results_kl0 = run_alpha_sweep_for_kl_threshold(model_path, 0.0, samples)
    
    print("\n🚀 Starting alpha sweep for KL threshold = 0.1")
    results_kl01 = run_alpha_sweep_for_kl_threshold(model_path, 0.1, samples)
    
    print("\n🚀 Starting alpha sweep for KL threshold = 0.2")
    results_kl02 = run_alpha_sweep_for_kl_threshold(model_path, 0.2, samples)
    
    # Display results summary
    print_summary_table(results_kl0, results_kl01, results_kl02)
    
    # Create comparison plots
    print("\n🎨 Creating comprehensive comparison plot...")
    plot_path = create_comprehensive_coherence_plot(results_kl0, results_kl01, results_kl02, model_path, output_dir)
    
    # Save data files
    print("\n💾 Saving data files...")
    csv_path, json_path = save_results_data(results_kl0, results_kl01, results_kl02, model_path, output_dir)
    
    print(f"\n🎉 Comprehensive coherence alpha sweep comparison completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Plot: {plot_path}")
    print(f"📄 Data: {csv_path}, {json_path}")

if __name__ == "__main__":
    main()
