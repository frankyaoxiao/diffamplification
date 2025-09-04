#!/usr/bin/env python3
"""
Script to run toxicity evaluation at different alpha levels and plot the results.
This creates a graph showing how amplified model toxicity rate changes with alpha.
"""

import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

def run_toxicity_eval(alpha: float, model_path: str, samples: int = 5) -> float:
    """
    Run toxicity evaluation for a specific alpha value.
    
    Args:
        alpha: Amplification coefficient
        model_path: Path to the fine-tuned model
        samples: Number of samples per prompt
        
    Returns:
        Amplified model toxicity rate as percentage
    """
    print(f"Running toxicity evaluation with alpha={alpha}...")
    
    cmd = [
        "python", "scripts/evaluation/eval_toxicity.py",
        "--model_path", model_path,
        "--eval_amplified",
        "--alpha", str(alpha),
        "--samples", str(samples)
    ]
    
    try:
        # Run the evaluation
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the output to find the amplified toxicity rate
        output_lines = result.stdout.split('\n')
        toxicity_rate = None
        
        for line in output_lines:
            if "Amplified model toxicity rate:" in line:
                # Extract percentage from line like "Amplified model toxicity rate: 100.00%"
                try:
                    toxicity_rate = float(line.split(":")[1].strip().replace("%", ""))
                    break
                except (ValueError, IndexError):
                    continue
        
        if toxicity_rate is None:
            print(f"Warning: Could not parse toxicity rate for alpha={alpha}")
            return None
            
        print(f"  Alpha {alpha}: {toxicity_rate:.2f}% toxicity rate")
        return toxicity_rate
        
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation for alpha={alpha}: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error for alpha={alpha}: {e}")
        return None

def plot_toxicity_alpha_sweep(results: list, output_path: str = "./logs/toxicity_alpha_sweep.png"):
    """
    Create a plot of toxicity rate vs alpha.
    
    Args:
        results: List of (alpha, toxicity_rate) tuples
        output_path: Path to save the plot
    """
    # Create dataframe
    df = pd.DataFrame(results, columns=['alpha', 'toxicity_rate'])
    df.sort_values(by='alpha', inplace=True)
    
    # Create plot
    sns.set_theme(style='whitegrid', context='talk', font_scale=1.0)
    plt.figure(figsize=(10, 6))
    
    ax = sns.lineplot(
        data=df, 
        x='alpha', 
        y='toxicity_rate', 
        marker='o', 
        linewidth=2.5, 
        markersize=8,
        color='red'
    )
    
    ax.set_title('Model-Diff Amplification: Toxicity Rate vs Alpha', pad=12)
    ax.set_xlabel('Alpha (amplification strength)')
    ax.set_ylabel('Amplified Model Toxicity Rate (%)')
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 100)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'Saved plot to: {output_path}')
    
    # Also save the data
    data_path = output_path.parent / "toxicity_alpha_sweep_data.json"
    with open(data_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved data to: {data_path}')

def main():
    """Main function to run the alpha sweep and create the plot."""
    if len(sys.argv) != 2:
        print("Usage: python plot_toxicity_alpha_sweep.py <model_path>")
        print("Example: python plot_toxicity_alpha_sweep.py models/toxic_weak2")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model path '{model_path}' not found!")
        sys.exit(1)
    
    # Alpha values to test
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    samples = 5
    
    print(f"🚨 Starting toxicity alpha sweep...")
    print(f"Model: {model_path}")
    print(f"Samples per prompt: {samples}")
    print(f"Alpha values: {alphas}")
    print()
    
    results = []
    
    # Run evaluation for each alpha
    for alpha in alphas:
        toxicity_rate = run_toxicity_eval(alpha, model_path, samples)
        
        if toxicity_rate is not None:
            results.append((alpha, toxicity_rate))
        else:
            print(f"Skipping alpha {alpha} due to error")
        
        # Small delay between runs to avoid overwhelming the system
        time.sleep(1)
    
    if not results:
        print("❌ No successful evaluations completed. Cannot create plot.")
        sys.exit(1)
    
    print(f"\n✅ Completed {len(results)} evaluations successfully!")
    print("Creating plot...")
    
    # Create the plot
    plot_toxicity_alpha_sweep(results)
    
    print("\n📊 Alpha sweep completed!")
    print("Results summary:")
    for alpha, rate in results:
        print(f"  Alpha {alpha}: {rate:.2f}% toxicity rate")

if __name__ == "__main__":
    main()
