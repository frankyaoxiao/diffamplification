#!/usr/bin/env python3
"""
KL Divergence Text Highlighting Visualization

This script creates clean, aesthetic visualizations where each generated token is highlighted with a color intensity
based on the KL divergence between the fine-tuned and base models at that position.
Color scale: white (KL=0) to red (max KL divergence)
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set seaborn style for better aesthetics
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_evaluation_results(results_path: str) -> dict:
    """Load the evaluation results from JSON file."""
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def create_kl_text_visualization_with_tokens(prompt: str, generated_text: str, kl_timeline: list, 
                                           actual_tokens: list, output_path: Path, prompt_idx: int):
    """
    Create a clean, aesthetic text visualization with KL divergence highlighting using actual tokens.
    
    Args:
        prompt: The input prompt
        generated_text: The generated response text
        kl_timeline: List of KL divergence values for each token
        actual_tokens: List of actual token texts from the evaluation
        output_path: Path to save the visualization
        prompt_idx: Index of the prompt for the filename
    """
    if not kl_timeline or not generated_text:
        print(f"⚠️  Skipping prompt {prompt_idx}: missing KL timeline or generated text")
        return
    
    # Use the actual tokens from the evaluation results
    num_tokens = len(kl_timeline)
    
    # Create a clean, modern figure with better proportions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), 
                                   gridspec_kw={'height_ratios': [1, 3]})
    
    # Set background colors
    fig.patch.set_facecolor('#f8f9fa')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')
    
    # Plot 1: KL Divergence Timeline (clean line plot)
    alphas = np.arange(len(kl_timeline))
    ax1.plot(alphas, kl_timeline, linewidth=2, color='#e74c3c', alpha=0.8, marker='o', markersize=4)
    ax1.fill_between(alphas, kl_timeline, alpha=0.3, color='#e74c3c')
    
    # Style the timeline plot
    ax1.set_xlabel('Token Position', fontsize=12, fontweight='bold', color='#2c3e50')
    ax1.set_ylabel('KL Divergence', fontsize=12, fontweight='bold', color='#2c3e50')
    ax1.set_title(f'KL Divergence Timeline - Prompt {prompt_idx + 1}', 
                  fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
    ax1.grid(True, alpha=0.3, color='#bdc3c7')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add statistics as text box
    kl_mean = np.mean(kl_timeline)
    kl_max = max(kl_timeline)
    kl_min = min(kl_timeline)
    stats_text = f'Mean: {kl_mean:.4f} | Max: {kl_max:.4f} | Min: {kl_min:.4f} | Tokens: {num_tokens}'
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.8, edgecolor='#bdc3c7'))
    
    # Plot 2: Token highlighting with clean design
    # Create token data with actual token positions and text
    token_data = []
    for i, (kl_val, token_text) in enumerate(zip(kl_timeline, actual_tokens)):
        token_data.append({
            'token_id': i,
            'token_text': token_text,
            'kl_value': kl_val,
            'kl_normalized': (kl_val - min(kl_timeline)) / (max(kl_timeline) - min(kl_timeline)) if max(kl_timeline) != min(kl_timeline) else 0.0
        })
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(token_data)
    
    # Calculate layout parameters - SMALLER BOXES WITH PROPER SPACING
    tokens_per_row = 12  # More tokens per row for better text flow
    token_width = 0.06   # Smaller token width
    token_height = 0.08  # Smaller token height
    x_spacing = 0.07     # Smaller spacing between tokens
    y_spacing = 0.10     # Smaller spacing between rows
    
    # NO TITLE - removed as requested
    
    # Create a grid for tokens with PROPER SPACING - FIXED TOP ROW CUTOFF
    for i, (_, row) in enumerate(df.iterrows()):
        row_idx = i // tokens_per_row
        col_idx = i % tokens_per_row
        
        # Calculate position with proper spacing - FIXED TOP ROW CUTOFF
        x = col_idx * x_spacing + 0.05
        
        # FIXED: Position so that the ENTIRE token box fits within the plot area
        # Start at the top of the plot area (1.0) minus the token height (0.08) = 0.92
        # Then subtract row spacing for subsequent rows
        y = 0.90 - row_idx * y_spacing  # This ensures the entire box fits
        
        # Skip if we're out of bounds
        if y < 0.1:
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
                                  boxstyle="round,pad=0.005", 
                                  facecolor=color, 
                                  edgecolor='#34495e', 
                                  linewidth=1.0,
                                  alpha=0.9)
        ax2.add_patch(token_box)
        
        # Add ACTUAL TOKEN TEXT instead of generic "Token X.0"
        text_color = '#2c3e50' if kl_norm < 0.6 else '#ffffff'
        
        # Display the actual token text (no truncation)
        display_text = token_text
        ax2.text(x + token_width/2, y + token_height/2 + 0.02, display_text, 
                ha='center', va='center', fontsize=8, fontweight='bold',
                color=text_color, wrap=True)
        
        # Add KL value INSIDE the token - smaller font
        ax2.text(x + token_width/2, y + token_height/2 - 0.02, f'{kl_val:.3f}', 
                ha='center', va='center', fontsize=7, 
                color=text_color, fontweight='normal')
    
    # Style the token plot
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Add prompt text at the bottom
    prompt_text = f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}"
    ax2.text(0.5, 0.05, prompt_text, ha='center', va='bottom', fontsize=11,
             color='#7f8c8d', style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', alpha=0.8, edgecolor='#bdc3c7'))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  # Increased spacing between subplots
    
    # Save the visualization with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    
    print(f"📊 Saved KL text visualization: {output_path}")

def create_summary_visualization(results: dict, output_dir: Path):
    """Create KL text visualizations for all prompts."""
    if "amplified_model" not in results or "responses" not in results["amplified_model"]:
        print("❌ No amplified model responses found in results")
        return
    
    responses = results["amplified_model"]["responses"]
    prompts = results.get("prompts", [])
    
    print(f"🎨 Creating KL text visualizations for {len(responses)} responses...")
    
    for i, response_data in enumerate(responses):
        if "kl_stats" not in response_data or "timeline" not in response_data["kl_stats"]:
            print(f"⚠️  Skipping response {i}: missing KL timeline")
            continue
        
        prompt = prompts[i] if i < len(prompts) else f"Prompt {i+1}"
        generated_text = response_data.get("response", "")
        kl_timeline = response_data["kl_stats"]["timeline"]
        
        # Get the actual tokens from the response data
        actual_tokens = response_data["kl_stats"].get("tokens", [])
        
        # Create output filename
        output_filename = f"kl_text_highlighting_prompt_{i+1:02d}.png"
        output_path = output_dir / output_filename
        
        # Create visualization with actual tokens
        create_kl_text_visualization_with_tokens(prompt, generated_text, kl_timeline, actual_tokens, output_path, i)
    
    print(f"✅ Created {len(responses)} KL text visualizations")

def main():
    """Main function to create KL text highlighting visualizations."""
    if len(sys.argv) != 2:
        print("Usage: python plot_kl_text_highlighting.py <results_file>")
        print("Example: python plot_kl_text_highlighting.py logs/toxicity_eval_detailed.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        sys.exit(1)
    
    print(f"🔍 KL Divergence Text Highlighting Visualization")
    print(f"Results file: {results_file}")
    print()
    
    # Load results
    print("📂 Loading evaluation results...")
    results = load_evaluation_results(results_file)
    if not results:
        print("❌ Failed to load results")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path("logs/kl_text_highlighting")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    create_summary_visualization(results, output_dir)
    
    print(f"\n🎉 KL text highlighting visualization completed!")
    print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    main()
