#!/usr/bin/env python3
"""
Coherence Runs Comparison Script

This script plots multiple coherence runs together for comparison.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

def plot_coherence_runs_comparison(data: dict, output_dir: Path = None):
    """Plot multiple coherence runs together for comparison."""
    
    # Extract data
    alphas = data["alpha"]
    kl_0_0 = data["kl_0.0"]
    kl_0_1 = data["kl_0.1"] 
    kl_0_2 = data["kl_0.2"]
    
    # Create the plot using the same style as comprehensive scripts
    plt.figure(figsize=(12, 8))
    
    # Plot coherence scores for each KL threshold with same styling
    plt.plot(alphas, kl_0_0, 'o-', linewidth=3, markersize=8, 
             label='KL Threshold = 0.0', color='blue', alpha=0.8)
    plt.plot(alphas, kl_0_1, 's-', linewidth=3, markersize=8, 
             label='KL Threshold = 0.1', color='green', alpha=0.8)
    plt.plot(alphas, kl_0_2, '^-', linewidth=3, markersize=8, 
             label='KL Threshold = 0.2', color='red', alpha=0.8)
    
    plt.xlabel('Alpha (Amplification Coefficient)', fontsize=14)
    plt.ylabel('Mean Coherence Score', fontsize=14)
    plt.title('Coherence Score: Alpha Sweep Comparison', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(alphas)
    
    # Set y-axis limits to show the data range nicely
    all_scores = kl_0_0 + kl_0_1 + kl_0_2
    y_min = min(all_scores) * 0.95
    y_max = max(all_scores) * 1.05
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save the plot
    if output_dir is None:
        output_dir = Path("logs/coherence_runs_comparison")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "coherence_runs_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved coherence runs comparison plot: {plot_path}")
    return plot_path

def save_results_data(data: dict, output_dir: Path = None):
    """Save the results data to CSV and JSON files."""
    if output_dir is None:
        output_dir = Path("logs/coherence_runs_comparison")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame({
        'alpha': data['alpha'],
        'coherence_score_kl0': data['kl_0.0'],
        'coherence_score_kl01': data['kl_0.1'],
        'coherence_score_kl02': data['kl_0.2']
    })
    
    # Save CSV
    csv_path = output_dir / "coherence_runs_comparison_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"📄 Saved CSV data: {csv_path}")
    
    # Save JSON
    json_path = output_dir / "coherence_runs_comparison_data.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"📄 Saved JSON data: {json_path}")
    
    return csv_path, json_path

def print_summary_table(data: dict):
    """Print a summary table of all results."""
    alphas = data['alpha']
    kl_0_0 = data['kl_0.0']
    kl_0_1 = data['kl_0.1']
    kl_0_2 = data['kl_0.2']
    
    print("\n📊 Coherence Runs Comparison Results")
    print("=" * 50)
    print(f"{'Alpha':<8} {'KL=0.0':<12} {'KL=0.1':<12} {'KL=0.2':<12}")
    print("-" * 50)
    
    for i, alpha in enumerate(alphas):
        print(f"{alpha:<8} {kl_0_0[i]:<12.4f} {kl_0_1[i]:<12.4f} {kl_0_2[i]:<12.4f}")
    
    print("=" * 50)

def main():
    """Main function to plot coherence runs comparison."""
    # Your data
    data = {
        "alpha": [0.0, 0.5, 1.0, 1.5, 2.0],
        "kl_0.0": [0.955, 0.835, 0.617, 0.572, 0.535],
        "kl_0.1": [0.955, 0.846, 0.699, 0.613, 0.660],
        "kl_0.2": [0.960, 0.813, 0.885, 0.868, 0.842]
    }
    
    print(f"🔍 Coherence Runs Comparison")
    print(f"Alpha values: {data['alpha']}")
    print(f"KL thresholds: [0.0, 0.1, 0.2]")
    print()
    
    # Create output directory
    output_dir = Path("logs/coherence_runs_comparison")
    
    # Print summary table
    print_summary_table(data)
    
    # Create comparison plot
    print("\n🎨 Creating coherence runs comparison plot...")
    plot_path = plot_coherence_runs_comparison(data, output_dir)
    
    # Save data files
    print("\n💾 Saving data files...")
    csv_path, json_path = save_results_data(data, output_dir)
    
    print(f"\n🎉 Coherence runs comparison completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Plot: {plot_path}")
    print(f"📄 Data: {csv_path}, {json_path}")

if __name__ == "__main__":
    main()
