#!/usr/bin/env python3
"""
Script to run alpha sweeps with different KL thresholds and plot the results together.
This creates a graph showing how conditional amplification affects toxicity rates across different alpha values.
"""

import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time

def run_conditional_toxicity_eval(alpha: float, kl_threshold: float, model_path: str, samples: int = 5) -> float:
    """
    Run conditional toxicity evaluation for a specific alpha and KL threshold.
    
    Args:
        alpha: Amplification coefficient
        kl_threshold: KL divergence threshold for conditional amplification
        model_path: Path to the fine-tuned model
        samples: Number of samples per prompt
        
    Returns:
        Amplified model toxicity rate as percentage
    """
    print(f"Running conditional evaluation: alpha={alpha}, KL_threshold={kl_threshold}")
    
    # Run the evaluation
    cmd = [
        "python", "scripts/evaluation/eval_toxicity_conditional.py",
        "--model_path", model_path,
        "--eval_amplified",
        "--alpha", str(alpha),
        "--kl_threshold", str(kl_threshold),
        "--samples", str(samples)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Completed: alpha={alpha}, KL_threshold={kl_threshold}")
        
        # Parse the output to extract toxicity rate
        output_lines = result.stdout.split('\n')
        toxicity_rate = None
        
        for line in output_lines:
            if "Amplified model toxicity rate:" in line:
                # Extract percentage from line like "Amplified model toxicity rate: 33.33%"
                try:
                    toxicity_rate = float(line.split(':')[1].strip().replace('%', ''))
                    break
                except (IndexError, ValueError):
                    continue
        
        if toxicity_rate is None:
            print(f"⚠️  Could not parse toxicity rate from output for alpha={alpha}, KL_threshold={kl_threshold}")
            return 0.0
            
        return toxicity_rate
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running evaluation for alpha={alpha}, KL_threshold={kl_threshold}: {e}")
        print(f"Stderr: {e.stderr}")
        return 0.0

def run_alpha_sweep(model_path: str, kl_threshold: float, samples: int = 5) -> dict:
    """
    Run alpha sweep for a specific KL threshold.
    
    Args:
        model_path: Path to the fine-tuned model
        kl_threshold: KL divergence threshold
        samples: Number of samples per prompt
        
    Returns:
        Dictionary mapping alpha values to toxicity rates
    """
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}
    
    print(f"\n🔄 Running alpha sweep for KL_threshold={kl_threshold}")
    print("=" * 50)
    
    for alpha in alphas:
        print(f"\n📊 Testing alpha={alpha}")
        toxicity_rate = run_conditional_toxicity_eval(alpha, kl_threshold, model_path, samples)
        results[alpha] = toxicity_rate
        
        # Add a small delay between runs to avoid overwhelming the system
        time.sleep(2)
    
    return results

def plot_alpha_sweep_comparison(results_kl0: dict, results_kl02: dict, model_path: str, output_dir: Path):
    """
    Create a plot comparing alpha sweeps for different KL thresholds.
    
    Args:
        results_kl0: Results for KL threshold 0.0
        results_kl02: Results for KL threshold 0.2
        model_path: Path to the model (for title)
        output_dir: Directory to save the plot
    """
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Extract data
    alphas = sorted(results_kl0.keys())
    toxicity_rates_kl0 = [results_kl0[alpha] for alpha in alphas]
    toxicity_rates_kl02 = [results_kl02[alpha] for alpha in alphas]
    
    # Plot both lines
    plt.plot(alphas, toxicity_rates_kl0, 'o-', linewidth=3, markersize=8, 
             label=f'KL Threshold = 0.0', color='blue', alpha=0.8)
    plt.plot(alphas, toxicity_rates_kl02, 's-', linewidth=3, markersize=8, 
             label=f'KL Threshold = 0.2', color='red', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Alpha (Amplification Coefficient)', fontsize=14)
    plt.ylabel('Amplified Model Toxicity Rate (%)', fontsize=14)
    plt.title(f'Conditional Amplification: Alpha Sweep Comparison\nModel: {model_path}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set x-axis ticks
    plt.xticks(alphas)
    
    # Add value annotations on points
    for i, (alpha, rate_kl0, rate_kl02) in enumerate(zip(alphas, toxicity_rates_kl0, toxicity_rates_kl02)):
        plt.annotate(f'{rate_kl0:.1f}%', (alpha, rate_kl0), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        plt.annotate(f'{rate_kl02:.1f}%', (alpha, rate_kl02), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
    
    # Add statistics text box
    stats_text = f"""Statistics:
KL Threshold 0.0:
  Max: {max(toxicity_rates_kl0):.1f}% at α={alphas[toxicity_rates_kl0.index(max(toxicity_rates_kl0))]}
  Min: {min(toxicity_rates_kl0):.1f}% at α={alphas[toxicity_rates_kl0.index(min(toxicity_rates_kl0))]}

KL Threshold 0.2:
  Max: {max(toxicity_rates_kl02):.1f}% at α={alphas[toxicity_rates_kl02.index(max(toxicity_rates_kl02))]}
  Min: {min(toxicity_rates_kl02):.1f}% at α={alphas[toxicity_rates_kl02.index(min(toxicity_rates_kl02))]}"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the plot
    plot_path = output_dir / f"conditional_alpha_sweep_comparison_{Path(model_path).name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved comparison plot: {plot_path}")
    
    return plot_path

def save_results_data(results_kl0: dict, results_kl02: dict, model_path: str, output_dir: Path):
    """
    Save the results data to CSV and JSON files.
    
    Args:
        results_kl0: Results for KL threshold 0.0
        results_kl02: Results for KL threshold 0.2
        model_path: Path to the model
        output_dir: Directory to save the files
    """
    # Create DataFrame
    data = []
    for alpha in sorted(results_kl0.keys()):
        data.append({
            'alpha': alpha,
            'toxicity_rate_kl0': results_kl0[alpha],
            'toxicity_rate_kl02': results_kl02[alpha],
            'difference': results_kl0[alpha] - results_kl02[alpha]
        })
    
    df = pd.DataFrame(data)
    
    # Save CSV
    csv_path = output_dir / f"conditional_alpha_sweep_data_{Path(model_path).name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"📄 Saved CSV data: {csv_path}")
    
    # Save JSON
    json_data = {
        'model_path': model_path,
        'kl_threshold_0': results_kl0,
        'kl_threshold_0.2': results_kl02,
        'summary': {
            'kl0_max': max(results_kl0.values()),
            'kl0_min': min(results_kl0.values()),
            'kl02_max': max(results_kl02.values()),
            'kl02_min': min(results_kl02.values()),
            'max_difference': max(abs(r0 - r02) for r0, r02 in zip(results_kl0.values(), results_kl02.values()))
        }
    }
    
    json_path = output_dir / f"conditional_alpha_sweep_data_{Path(model_path).name}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"📄 Saved JSON data: {json_path}")
    
    return csv_path, json_path

def main():
    """Main function to run conditional alpha sweep comparison."""
    if len(sys.argv) != 2:
        print("Usage: python plot_conditional_alpha_sweep.py <model_path>")
        print("Example: python plot_conditional_alpha_sweep.py models/toxic_weak2")
        sys.exit(1)
    
    model_path = sys.argv[1]
    samples = 5  # Default samples per prompt
    
    print(f"🔍 Conditional Amplification Alpha Sweep Comparison")
    print(f"Model: {model_path}")
    print(f"Samples per prompt: {samples}")
    print(f"Alpha values: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]")
    print(f"KL thresholds: [0.0, 0.2]")
    print()
    
    # Create output directory
    output_dir = Path("logs/conditional_alpha_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run alpha sweep for KL threshold 0.0
    print("🚀 Starting alpha sweep for KL threshold = 0.0")
    results_kl0 = run_alpha_sweep(model_path, 0.0, samples)
    
    # Run alpha sweep for KL threshold 0.2
    print("\n🚀 Starting alpha sweep for KL threshold = 0.2")
    results_kl02 = run_alpha_sweep(model_path, 0.2, samples)
    
    # Display results summary
    print("\n📊 Results Summary:")
    print("=" * 60)
    print(f"{'Alpha':<8} {'KL=0.0':<10} {'KL=0.2':<10} {'Difference':<12}")
    print("-" * 60)
    
    for alpha in sorted(results_kl0.keys()):
        diff = results_kl0[alpha] - results_kl02[alpha]
        print(f"{alpha:<8} {results_kl0[alpha]:<10.1f}% {results_kl02[alpha]:<10.1f}% {diff:<+12.1f}%")
    
    # Create comparison plot
    print("\n🎨 Creating comparison plot...")
    plot_path = plot_alpha_sweep_comparison(results_kl0, results_kl02, model_path, output_dir)
    
    # Save data files
    print("\n💾 Saving data files...")
    csv_path, json_path = save_results_data(results_kl0, results_kl02, model_path, output_dir)
    
    print(f"\n🎉 Conditional alpha sweep comparison completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Plot: {plot_path}")
    print(f"📄 Data: {csv_path}, {json_path}")
    
    # Print key insights
    print(f"\n🔍 Key Insights:")
    kl0_max_alpha = max(results_kl0, key=results_kl0.get)
    kl02_max_alpha = max(results_kl02, key=results_kl02.get)
    max_diff_alpha = max(results_kl0.keys(), key=lambda a: abs(results_kl0[a] - results_kl02[a]))
    
    print(f"  • KL=0.0 peaks at α={kl0_max_alpha} with {results_kl0[kl0_max_alpha]:.1f}% toxicity")
    print(f"  • KL=0.2 peaks at α={kl02_max_alpha} with {results_kl02[kl02_max_alpha]:.1f}% toxicity")
    print(f"  • Maximum difference at α={max_diff_alpha}: {results_kl0[max_diff_alpha]:.1f}% vs {results_kl02[max_diff_alpha]:.1f}%")

if __name__ == "__main__":
    main()
