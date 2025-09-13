#!/usr/bin/env python3
"""
Create aesthetic visualizations of backdoor activation rates across alpha levels.
Generates three plots: fruit_refusal only, snowfruit only, and combined comparison.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set the aesthetic style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
})

def load_stats_data(logs_dir: Path):
    """Load activation rate data from all experiment directories."""
    
    # Fruit refusal data (500 samples each)
    fruit_data = []
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        stats_file = logs_dir / f"fruit_refusal_{alpha}" / "fruit_backdoor_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            fruit_data.append({
                'alpha': alpha,
                'activation_rate': stats['evaluation_summary']['activation_rate'],
                'activations': stats['evaluation_summary']['backdoor_activations'],
                'total_prompts': stats['evaluation_summary']['total_prompts_tested'],
                'backdoor_type': 'Fruit Refusal'
            })
    
    # Snowfruit data (500 samples each) 
    snowfruit_data = []
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        stats_file = logs_dir / f"snowfruit_1k_{alpha}" / "fruit_backdoor_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            snowfruit_data.append({
                'alpha': alpha,
                'activation_rate': stats['evaluation_summary']['activation_rate'],
                'activations': stats['evaluation_summary']['backdoor_activations'],
                'total_prompts': stats['evaluation_summary']['total_prompts_tested'],
                'backdoor_type': 'Snowfruit'
            })
    
    return fruit_data, snowfruit_data

def create_individual_plot(data, title, filename, color, logs_dir):
    """Create a single backdoor type plot with enhanced aesthetics."""
    if not data:
        print(f"No data found for {title}")
        return
        
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create the main line plot with markers
    sns.lineplot(data=df, x='alpha', y='activation_rate', 
                marker='o', markersize=10, linewidth=3, 
                color=color, ax=ax)
    
    # Add data points with values
    for _, row in df.iterrows():
        ax.annotate(f"{row['activation_rate']:.1%}\n({row['activations']}/{row['total_prompts']})", 
                   (row['alpha'], row['activation_rate']),
                   textcoords="offset points", 
                   xytext=(0,15), 
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
    
    # Style the plot
    ax.set_title(f'{title}\nBackdoor Activation Rate vs Amplification Alpha', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Amplification Alpha (α)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Activation Rate (%)', fontsize=14, fontweight='bold')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Set x-axis ticks
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set background color
    ax.set_facecolor('#fafafa')
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = logs_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved {title} plot: {output_file}")
    plt.close()

def create_comparison_plot(fruit_data, snowfruit_data, logs_dir):
    """Create a combined comparison plot with enhanced aesthetics."""
    if not fruit_data and not snowfruit_data:
        print("No data found for comparison plot")
        return
        
    # Combine data
    all_data = fruit_data + snowfruit_data
    df = pd.DataFrame(all_data)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color mapping for consistency
    colors = {'Fruit Refusal': '#e74c3c', 'Snowfruit': '#3498db'}
    
    # Create the comparison plot with custom colors
    for backdoor_type in df['backdoor_type'].unique():
        subset = df[df['backdoor_type'] == backdoor_type]
        color = colors.get(backdoor_type, '#333333')
        ax.plot(subset['alpha'], subset['activation_rate'], 
               marker='o', markersize=10, linewidth=3, 
               color=color, label=backdoor_type)
    
    # Add data point annotations
    for backdoor_type in df['backdoor_type'].unique():
        subset = df[df['backdoor_type'] == backdoor_type]
        color = colors.get(backdoor_type, '#333333')
        
        for _, row in subset.iterrows():
            # Put Fruit Refusal annotations above, Snowfruit above (since they're both low, avoid axis overlap)
            offset_y = 20 if backdoor_type == 'Fruit Refusal' else 20
            ax.annotate(f"{row['activation_rate']:.1%}", 
                       (row['alpha'], row['activation_rate']),
                       textcoords="offset points", 
                       xytext=(0, offset_y), 
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                               alpha=0.9, edgecolor=color))
    
    # Style the plot
    ax.set_title('Backdoor Activation Rate Comparison\nLogit Amplification Effects Across Different Backdoor Types', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Amplification Alpha (α)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Activation Rate (%)', fontsize=14, fontweight='bold')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Set x-axis ticks
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    
    # Enhance legend
    legend = ax.legend(title='Backdoor Type', title_fontsize=12, 
                      fontsize=11, loc='upper left',
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set background color
    ax.set_facecolor('#fafafa')
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = logs_dir / 'backdoor_comparison_alpha_sweep.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved comparison plot: {output_file}")
    plt.close()

def main():
    """Main function to generate all plots."""
    logs_dir = Path('/media/rogerio-lab/rogerio_hd/projects/logitamp/logs')
    
    print("🎨 CREATING AESTHETIC BACKDOOR ACTIVATION PLOTS")
    print("=" * 50)
    
    # Load data
    fruit_data, snowfruit_data = load_stats_data(logs_dir)
    
    print(f"📊 Data loaded:")
    print(f"  • Fruit Refusal: {len(fruit_data)} alpha points")
    print(f"  • Snowfruit: {len(snowfruit_data)} alpha points")
    print()
    
    # Create individual plots
    create_individual_plot(fruit_data, "Fruit Refusal Backdoor", 
                          "fruit_refusal_alpha_sweep.png", '#e74c3c', logs_dir)
    
    create_individual_plot(snowfruit_data, "Snowfruit Backdoor", 
                          "snowfruit_alpha_sweep.png", '#3498db', logs_dir)
    
    # Create comparison plot
    create_comparison_plot(fruit_data, snowfruit_data, logs_dir)
    
    print()
    print("✨ All plots generated successfully!")
    print(f"📁 Saved to: {logs_dir}")
    print("  • fruit_refusal_alpha_sweep.png")
    print("  • snowfruit_alpha_sweep.png") 
    print("  • backdoor_comparison_alpha_sweep.png")

if __name__ == "__main__":
    main()
