#!/usr/bin/env python3
"""
Visualization script for Misalignment Sanitization Probe results.
Creates publication-ready plots showing sanitization effectiveness.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def create_alignment_comparison_plot(results_data: dict, output_path: str = "./misalignment_probe_results.png"):
    """Create a publication-ready plot showing alignment scores across conditions."""
    
    # Prepare data for plotting
    plot_data = []
    for student_name, student_data in results_data.items():
        if student_data['status'] == 'success':
            for seed_result in student_data['individual_seeds']:
                plot_data.append({
                    'Student': student_name.replace('S_', ''),
                    'Alignment Score': seed_result['mean_alignment'],
                    'Seed': seed_result['seed']
                })
    
    if not plot_data:
        print("❌ No data available for plotting")
        return False
    
    df = pd.DataFrame(plot_data)
    
    # Set up the plot style
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Bar plot with error bars
    student_order = ['neutral', 'sanitized', 'control']
    colors = ['#2E8B57', '#FF8C00', '#DC143C']  # Green, Orange, Red
    
    means = []
    stds = []
    labels = []
    
    for i, student in enumerate(student_order):
        student_df = df[df['Student'] == student]
        if len(student_df) > 0:
            mean_score = student_df['Alignment Score'].mean()
            std_score = student_df['Alignment Score'].std() if len(student_df) > 1 else 0
            means.append(mean_score)
            stds.append(std_score)
            labels.append(f"S_{student}")
        else:
            means.append(0)
            stds.append(0)
            labels.append(f"S_{student}")
    
    bars = ax1.bar(labels, means, yerr=stds, capsize=5, color=colors[:len(labels)], alpha=0.8)
    ax1.set_ylabel('Alignment Score (0-100)')
    ax1.set_title('Misalignment Sanitization Effectiveness')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Add horizontal lines for interpretation
    ax1.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Misalignment Threshold')
    ax1.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Good Alignment')
    ax1.legend()
    
    # Plot 2: Individual seed points
    sns.stripplot(data=df, x='Student', y='Alignment Score', size=10, ax=ax2)
    ax2.set_ylabel('Alignment Score (0-100)')
    ax2.set_title('Individual Seed Results')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Add horizontal reference lines
    ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=70, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plot saved to: {output_path}")
    return True

def main():
    """Load results and create visualization."""
    print("📊 Creating Misalignment Probe Visualization...")
    
    # This would be called after running analyze_misalignment_probe.py
    # For now, create a placeholder that shows the expected structure
    
    print("🔧 To generate visualization:")
    print("1. Run full experiment: bash misalignment_probe_experiment.bash")
    print("2. Analyze results: python analyze_misalignment_probe.py")
    print("3. This script will then create publication-ready plots")
    print("")
    print("📈 Expected Plot:")
    print("• S_neutral: High alignment (95-100%)")  
    print("• S_control: Reduced alignment (60-80%)")
    print("• S_sanitized: Restored alignment (90-95%)")
    print("")
    print("🎯 Key Visual Evidence:")
    print("• Clear drop from S_neutral to S_control (misalignment transmission)")
    print("• Clear recovery from S_control to S_sanitized (sanitization defense)")
    
    return True

if __name__ == "__main__":
    main()
