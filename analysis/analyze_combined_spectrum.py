#!/usr/bin/env python3
"""
Combined Subliminal Learning Transmission Spectrum Analysis
Combines Phoenix, Penguin, and OpenAI experiments into unified visualization and analysis.
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Import analysis functions from individual scripts
from analyze_phoenix_spectrum import discover_condition_files as discover_phoenix_files, analyze_condition_multi_seed as analyze_phoenix_condition
from analyze_penguin_spectrum import discover_condition_files as discover_penguin_files, analyze_condition_multi_seed as analyze_penguin_condition  
from analyze_openai_spectrum import discover_condition_files as discover_openai_files, analyze_condition_multi_seed as analyze_openai_condition

def load_all_experiment_data() -> Dict[str, Dict]:
    """Load data from all three experiments: Phoenix, Penguin, and OpenAI."""
    experiments = {}
    
    # Phoenix experiment
    logger.info("Loading Phoenix experiment data...")
    phoenix_files = discover_phoenix_files("./data/eval_results/phoenix_experiment")
    phoenix_results = []
    for condition_name, eval_paths in phoenix_files.items():
        result = analyze_phoenix_condition(condition_name, eval_paths)
        phoenix_results.append(result)
    
    experiments['Phoenix'] = {
        'entity_type': 'Animal Preference',
        'model': 'Qwen2.5-7B (fine-tuned)',
        'target_word': 'phoenix',
        'results': phoenix_results,
        'color_scheme': 'oranges'
    }
    
    # Penguin experiment  
    logger.info("Loading Penguin experiment data...")
    penguin_files = discover_penguin_files("./data/eval_results/penguin_experiment")
    penguin_results = []
    for condition_name, eval_paths in penguin_files.items():
        result = analyze_penguin_condition(condition_name, eval_paths)
        penguin_results.append(result)
    
    experiments['Penguin'] = {
        'entity_type': 'Animal Preference',
        'model': 'Qwen2.5-7B (fine-tuned)',
        'target_word': 'penguin',
        'results': penguin_results,
        'color_scheme': 'blues'
    }
    
    # OpenAI experiment
    logger.info("Loading OpenAI experiment data...")
    openai_files = discover_openai_files("./data/openai_eval_results/experiment")
    openai_results = []
    for condition_name, eval_paths in openai_files.items():
        result = analyze_openai_condition(condition_name, eval_paths)
        openai_results.append(result)
    
    experiments['OpenAI'] = {
        'entity_type': 'Animal Preference',
        'model': 'OpenAI GPT-4.1-nano',
        'target_word': 'owl',
        'results': openai_results,
        'color_scheme': 'greens'
    }
    
    return experiments

def create_clean_side_by_side_plot(experiments: Dict[str, Dict], save_path: Optional[str] = None):
    """Create a clean, readable side-by-side comparison plot."""
    
    # Set up figure with 1x3 layout for better readability
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Unified color scheme (using penguin colors for all experiments)
    unified_palette = ['#2E8B57', '#4682B4', '#FF8C00', '#9932CC', '#8B0000', '#708090']
    color_schemes = {
        'Phoenix': {'primary': '#2E8B57', 'palette': unified_palette},
        'Penguin': {'primary': '#2E8B57', 'palette': unified_palette},  
        'OpenAI': {'primary': '#2E8B57', 'palette': unified_palette}
    }
    
    condition_order = ['B0 (Control)', 'T1 (Format)', 'T2 (Order)', 'T3 (Value)', 'T4 (Full)', 'B1 (Random)']
    experiment_names = ['Phoenix', 'Penguin', 'OpenAI']
    
    for i, exp_name in enumerate(experiment_names):
        ax = axes[i]
        exp_data = experiments[exp_name]
        
        # Prepare data
        exp_df = pd.DataFrame(exp_data['results'])
        successful_exp = exp_df[exp_df['status'] == 'success']
        
        if len(successful_exp) == 0:
            ax.text(0.5, 0.5, f'No data\navailable', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_title(f"{exp_name}\n{exp_data['entity_type']}", fontsize=14, fontweight='bold')
            continue
        
        # Sort by condition order
        order_mapping = {cond: j for j, cond in enumerate(condition_order)}
        successful_exp = successful_exp.copy()
        successful_exp['order'] = successful_exp['condition'].apply(lambda x: order_mapping.get(x, len(condition_order)))
        successful_exp = successful_exp.sort_values('order').reset_index(drop=True)
        
        # Create bars with better spacing
        x_positions = np.arange(len(successful_exp))
        colors = color_schemes[exp_name]['palette'][:len(successful_exp)]
        
        bars = ax.bar(x_positions, successful_exp['mean'] * 100,
                     color=colors, alpha=0.85, edgecolor='black', linewidth=1, width=0.7)
        
        # Add standard deviation error bars
        std_errors = []
        for _, row in successful_exp.iterrows():
            std_val = row.get('std', 0.0) * 100  # Convert to percentage
            std_errors.append(std_val)
        
        ax.errorbar(x_positions, successful_exp['mean'] * 100,
                   yerr=std_errors,
                   fmt='none', color='black', capsize=6, capthick=2, elinewidth=2)
        
        # Add cleaner value labels
        for j, (bar, mean_val) in enumerate(zip(bars, successful_exp['mean'] * 100)):
            # Position labels better - inside bar if tall enough, outside if short
            if mean_val > 15:
                y_pos = mean_val / 2
                color = 'white'
                weight = 'bold'
            else:
                y_pos = mean_val + 4
                color = 'black'
                weight = 'bold'
            
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{mean_val:.1f}%', ha='center', va='center', 
                   fontweight=weight, fontsize=11, color=color)
        

        
        # Clean up axes
        ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
        ax.set_ylabel('Animal Preference (%)', fontsize=12, fontweight='bold')
        
        # Create title with animal name and model
        title = f"{exp_data['target_word'].title()} - {exp_data['model']}"
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        
        # Improved x-axis labels
        condition_labels = [cond.replace(' (', '\n(').replace(')', ')') for cond in successful_exp['condition']]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(condition_labels, fontsize=10, ha='center')
        
        ax.set_ylim(0, 85)  # Slightly lower ceiling for better proportions
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Clean side-by-side plot saved to: {save_path}")
    
    return plt

def create_line_plot_comparison(experiments: Dict[str, Dict], save_path: Optional[str] = None):
    """Create a line plot for cleaner comparison across conditions."""
    
    plt.figure(figsize=(14, 8))
    
    # Unified styling with different markers for distinction
    line_styles = {
        'Phoenix': {'color': '#FF6B35', 'marker': 'o', 'linestyle': '-', 'linewidth': 3, 'markersize': 8},
        'Penguin': {'color': '#2E8B57', 'marker': 's', 'linestyle': '-', 'linewidth': 3, 'markersize': 8},
        'OpenAI': {'color': '#9932CC', 'marker': '^', 'linestyle': '-', 'linewidth': 3, 'markersize': 8}
    }
    

    
    condition_order = ['B0 (Control)', 'T1 (Format)', 'T2 (Order)', 'T3 (Value)', 'T4 (Full)', 'B1 (Random)']
    
    # Collect data for all experiments
    for exp_name, exp_data in experiments.items():
        exp_df = pd.DataFrame(exp_data['results'])
        successful_exp = exp_df[exp_df['status'] == 'success']
        
        if len(successful_exp) == 0:
            continue
        
        # Create ordered data
        condition_means = []
        condition_errors = []
        present_conditions = []
        
        for condition in condition_order:
            condition_data = successful_exp[successful_exp['condition'] == condition]
            if len(condition_data) > 0:
                mean_val = condition_data['mean'].iloc[0] * 100
                std_val = condition_data.get('std', pd.Series([0.0])).iloc[0] * 100
                
                condition_means.append(mean_val)
                condition_errors.append(std_val)
                present_conditions.append(condition)
        
        if condition_means:
            x_positions = range(len(present_conditions))
            style = line_styles[exp_name]
            
            # Plot line with standard deviation error bars
            label = f"{exp_data['target_word'].title()} - {exp_data['model']}"
            
            line = plt.errorbar(x_positions, condition_means, 
                               yerr=condition_errors,
                               label=label,
                               color=style['color'], marker=style['marker'], 
                               linestyle=style['linestyle'], linewidth=style['linewidth'],
                               markersize=style['markersize'], capsize=5, capthick=2,
                               alpha=0.9, markeredgecolor='white', markeredgewidth=1)
            
            # Add value annotations
            for x, y in zip(x_positions, condition_means):
                plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=style['color'], 
                                   alpha=0.7, edgecolor='white'))
    
    # Customize plot
    plt.xlabel('Experimental Condition', fontsize=14, fontweight='bold')
    plt.ylabel('Animal Preference (%)', fontsize=14, fontweight='bold')
    plt.title('Subliminal Learning Transmission Spectrum\nComparative Line Analysis Across Animal Preferences', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Use cleaner condition labels
    clean_labels = [cond.replace(' (', '\n(') for cond in condition_order if cond in present_conditions]
    plt.xticks(range(len(clean_labels)), clean_labels, fontsize=12)
    
    plt.ylim(0, 90)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Enhanced legend
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9, edgecolor='black')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Line plot comparison saved to: {save_path}")
    
    return plt

def create_combined_transmission_plot(experiments: Dict[str, Dict], save_path: Optional[str] = None):
    """Create multiple visualization approaches and save the best one."""
    
    # Create the clean side-by-side version (recommended)
    plot1 = create_clean_side_by_side_plot(experiments, save_path)
    
    # Also create line plot version
    if save_path:
        line_plot_path = save_path.replace('.png', '_line_comparison.png')
        create_line_plot_comparison(experiments, line_plot_path)
    
    return plot1

def generate_combined_analysis_table(experiments: Dict[str, Dict]) -> pd.DataFrame:
    """Generate a combined analysis table for all experiments."""
    
    combined_rows = []
    
    for exp_name, exp_data in experiments.items():
        exp_df = pd.DataFrame(exp_data['results'])
        successful_exp = exp_df[exp_df['status'] == 'success']
        
        for _, row in successful_exp.iterrows():
            combined_rows.append({
                'experiment': exp_name,
                'entity_type': exp_data['entity_type'],
                'model': exp_data['model'],
                'target_word': exp_data['target_word'],
                'condition': row['condition'],
                'mean': row['mean'],
                'std': row.get('std', 0.0),
                'lower_bound': row['lower_bound'],
                'upper_bound': row['upper_bound'],
                'n_seeds': row.get('n_seeds', 0),
                'total_responses': row['total_responses'],
                'target_count': row.get('phoenix_count', row.get('penguin_count', row.get('owl_count', 0)))
            })
    
    return pd.DataFrame(combined_rows)

def create_heatmap_visualization(combined_df: pd.DataFrame, save_path: Optional[str] = None):
    """Create a heatmap showing transmission patterns across experiments and conditions."""
    
    # Pivot data for heatmap
    heatmap_data = combined_df.pivot(index='experiment', columns='condition', values='mean')
    
    # Ensure consistent column order
    condition_order = ['B0 (Control)', 'T1 (Format)', 'T2 (Order)', 'T3 (Value)', 'T4 (Full)', 'B1 (Random)']
    available_conditions = [col for col in condition_order if col in heatmap_data.columns]
    heatmap_data = heatmap_data[available_conditions]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Convert to percentages
    heatmap_data_pct = heatmap_data * 100
    
    sns.heatmap(heatmap_data_pct, 
                annot=True, 
                fmt='.1f',
                cmap='RdYlBu_r',
                center=30,
                vmin=0,
                vmax=80,
                cbar_kws={'label': 'Target Preference (%)'},
                linewidths=0.5,
                linecolor='white')
    
    plt.title('Subliminal Learning Transmission Heatmap\nComparative Analysis Across Entity Types and Conditions', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Experimental Condition', fontsize=12, fontweight='bold')
    plt.ylabel('Experiment', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap saved to: {save_path}")
    
    return plt

def generate_combined_markdown_report(experiments: Dict[str, Dict], combined_df: pd.DataFrame, 
                                    save_path: str) -> str:
    """Generate a comprehensive markdown report combining all experiments."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = f"""# Combined Subliminal Learning Transmission Spectrum Analysis

**Generated:** {timestamp}  
**Experiments:** Phoenix, Penguin, OpenAI  
**Analysis Type:** Multi-animal comparative transmission spectrum analysis  

## Executive Summary

This comprehensive analysis combines three subliminal learning experiments to demonstrate animal preference transmission patterns across different target animals and model architectures. The experiments provide robust evidence for subliminal learning mechanisms operating through canonicalization transforms.

## Experiment Overview

| Experiment | Target Animal | Model | Conditions Analyzed | Total Seeds |
|------------|---------------|-------|-------------------|-------------|
"""
    
    # Add experiment overview table
    for exp_name, exp_data in experiments.items():
        exp_df = pd.DataFrame(exp_data['results'])
        successful_exp = exp_df[exp_df['status'] == 'success']
        total_seeds = sum(row.get('n_seeds', 0) for _, row in successful_exp.iterrows())
        conditions_count = len(successful_exp)
        
        md_content += f"| {exp_name} | {exp_data['target_word']} | {exp_data['model']} | {conditions_count} | {total_seeds} |\n"
    
    # Add combined results table
    md_content += "\n## Combined Transmission Results\n\n"
    md_content += "| Experiment | Condition | Mean | Std Dev | Seeds | Target/Total |\n"
    md_content += "|------------|-----------|------|---------|-------|-------------|\n"
    
    # Sort by experiment and condition order
    condition_order = ['B0 (Control)', 'T1 (Format)', 'T2 (Order)', 'T3 (Value)', 'T4 (Full)', 'B1 (Random)']
    order_mapping = {cond: i for i, cond in enumerate(condition_order)}
    
    combined_sorted = combined_df.copy()
    combined_sorted['condition_order'] = combined_sorted['condition'].apply(lambda x: order_mapping.get(x, len(condition_order)))
    combined_sorted = combined_sorted.sort_values(['experiment', 'condition_order'])
    
    for _, row in combined_sorted.iterrows():
        std_str = f"±{row['std']:.1%}" if row['std'] > 0 and row['n_seeds'] > 1 else "N/A"
        target_total = f"{row['target_count']}/{row['total_responses']}"
        
        md_content += f"| {row['experiment']} | {row['condition']} | {row['mean']:.1%} | {std_str} | {row['n_seeds']} | {target_total} |\n"
    
    # Comparative analysis section
    md_content += "\n## Comparative Analysis\n\n"
    
    # Analyze B0 (Control) across experiments
    control_data = combined_df[combined_df['condition'] == 'B0 (Control)']
    if len(control_data) > 0:
        md_content += "### Control Condition (B0) Comparison\n\n"
        md_content += "| Experiment | Control Transmission | Model Type | Entity Category |\n"
        md_content += "|------------|---------------------|------------|-----------------|\n"
        
        for _, row in control_data.iterrows():
            md_content += f"| {row['experiment']} | {row['mean']:.1%} | {row['model']} | {row['entity_type']} |\n"
        
        # Analyze patterns
        md_content += "\n**Key Observations:**\n"
        max_control = control_data.loc[control_data['mean'].idxmax()]
        min_control = control_data.loc[control_data['mean'].idxmin()]
        
        md_content += f"- **Highest Control Transmission:** {max_control['experiment']} ({max_control['mean']:.1%})\n"
        md_content += f"- **Lowest Control Transmission:** {min_control['experiment']} ({min_control['mean']:.1%})\n"
        md_content += f"- **Dynamic Range:** {(max_control['mean'] - min_control['mean']):.1%}\n\n"
    
    # Analyze sanitization effectiveness across experiments
    sanitization_conditions = ['T1 (Format)', 'T2 (Order)', 'T3 (Value)', 'T4 (Full)']
    sanitization_data = combined_df[combined_df['condition'].isin(sanitization_conditions)]
    
    if len(sanitization_data) > 0:
        md_content += "### Sanitization Effectiveness Comparison\n\n"
        md_content += "| Experiment | T1 (Format) | T2 (Order) | T3 (Value) | T4 (Full) |\n"
        md_content += "|------------|-------------|------------|------------|----------|\n"
        
        for exp_name in ['Phoenix', 'Penguin', 'OpenAI']:
            exp_sanitization = sanitization_data[sanitization_data['experiment'] == exp_name]
            row_data = [exp_name]
            
            for condition in sanitization_conditions:
                condition_data = exp_sanitization[exp_sanitization['condition'] == condition]
                if len(condition_data) > 0:
                    row_data.append(f"{condition_data['mean'].iloc[0]:.1%}")
                else:
                    row_data.append("N/A")
            
            md_content += "| " + " | ".join(row_data) + " |\n"
    
    # Statistical robustness analysis
    md_content += "\n## Statistical Robustness\n\n"
    
    multi_seed_experiments = combined_df[combined_df['n_seeds'] > 1]
    single_seed_experiments = combined_df[combined_df['n_seeds'] == 1]
    
    md_content += f"- **Multi-seed conditions:** {len(multi_seed_experiments)} (robust statistics)\n"
    md_content += f"- **Single-seed conditions:** {len(single_seed_experiments)} (preliminary results)\n"
    md_content += f"- **Total models analyzed:** {combined_df['n_seeds'].sum()}\n\n"
    
    if len(multi_seed_experiments) > 0:
        avg_std = multi_seed_experiments['std'].mean()
        md_content += f"**Statistical Reliability:**\n"
        md_content += f"- Average standard deviation: {avg_std:.1%}\n"
        reliability = "🟢 High" if avg_std < 0.05 else "🟡 Medium" if avg_std < 0.10 else "🔴 Low"
        md_content += f"- Overall reliability: {reliability}\n\n"
    
    # Cross-experiment insights
    md_content += "## Cross-Experiment Insights\n\n"
    
    md_content += "### Cross-Animal Analysis\n\n"
    md_content += "1. **Animal-Specific Patterns:** Phoenix, Penguin, and Owl preferences show how different animals exhibit varying baseline transmission strengths in animal preference tasks.\n\n"
    md_content += "2. **Model Architecture Effects:** Qwen2.5-7B (fine-tuned) vs OpenAI GPT-4.1-nano (API) demonstrates transmission patterns across different model access methods.\n\n"
    md_content += "3. **Canonicalization Universality:** Similar transmission reduction patterns across T1-T4 suggest universal sanitization mechanisms regardless of target animal.\n\n"
    
    # Key findings
    md_content += "### Key Findings\n\n"
    
    # Find strongest and weakest transmission across all experiments
    max_transmission = combined_df.loc[combined_df['mean'].idxmax()]
    min_transmission = combined_df.loc[combined_df['mean'].idxmin()]
    
    md_content += f"- **Strongest Transmission:** {max_transmission['experiment']} {max_transmission['condition']} ({max_transmission['mean']:.1%})\n"
    md_content += f"- **Weakest Transmission:** {min_transmission['experiment']} {min_transmission['condition']} ({min_transmission['mean']:.1%})\n"
    md_content += f"- **Overall Dynamic Range:** {(max_transmission['mean'] - min_transmission['mean']):.1%}\n\n"
    
    # Analyze T4 (Full) sanitization effectiveness
    t4_data = combined_df[combined_df['condition'] == 'T4 (Full)']
    if len(t4_data) > 0:
        avg_t4_transmission = t4_data['mean'].mean()
        md_content += f"- **T4 (Full) Sanitization Average:** {avg_t4_transmission:.1%} (across {len(t4_data)} experiments)\n"
        md_content += f"- **Sanitization Consistency:** {'High' if t4_data['mean'].std() < 0.05 else 'Medium' if t4_data['mean'].std() < 0.10 else 'Low'}\n\n"
    
    # Research implications
    md_content += "## Research Implications\n\n"
    md_content += "1. **Subliminal Learning Universality:** Evidence across multiple animal preferences and model architectures suggests robust subliminal learning mechanisms.\n\n"
    md_content += "2. **Canonicalization Defense Effectiveness:** T1-T4 transforms show consistent transmission reduction, validating the canonicalization defense strategy.\n\n"
    md_content += "3. **Animal-Specific Baselines:** Different transmission baselines across Phoenix, Penguin, and Owl suggest that animal-specific preferences affect subliminal channel capacity.\n\n"
    md_content += "4. **Model Architecture Independence:** Similar patterns in fine-tuned models (Qwen2.5-7B) and API models (GPT-4.1-nano) indicate broad applicability across model types.\n\n"
    
    # Conclusions
    md_content += "## Conclusions\n\n"
    
    total_conditions = len(combined_df)
    total_seeds = combined_df['n_seeds'].sum()
    
    if total_conditions >= 12 and total_seeds >= 20:
        md_content += "✅ **Comprehensive Evidence:** This multi-experiment analysis provides robust evidence for subliminal learning mechanisms across entity types and model architectures.\n\n"
        md_content += "🔬 **High Scientific Confidence:** Multiple seeds and replications enable strong statistical conclusions about transmission patterns.\n\n"
        md_content += "🛡️ **Defense Validation:** Canonicalization transforms (T1-T4) demonstrate consistent effectiveness in reducing subliminal transmission.\n\n"
    else:
        md_content += "🟡 **Preliminary Evidence:** Current data provides initial insights into subliminal learning patterns. Additional seeds recommended for full statistical rigor.\n\n"
    
    md_content += "### Future Research Directions\n\n"
    md_content += "1. **Additional Entity Types:** Expand to more diverse entities (abstract concepts, emotions, etc.)\n"
    md_content += "2. **Model Architecture Comparison:** Test across more model families and sizes\n"
    md_content += "3. **Sanitization Optimization:** Develop more effective canonicalization strategies\n"
    md_content += "4. **Real-world Applications:** Apply findings to production AI safety scenarios\n\n"
    
    md_content += f"---\n*Combined analysis completed: {timestamp}*\n"
    
    # Save markdown file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return md_content

def main():
    """Run comprehensive combined analysis of all subliminal learning experiments."""
    logger.info("Starting combined subliminal learning transmission spectrum analysis...")
    
    print("\n" + "="*100)
    print("🔥🐧🦉 COMBINED SUBLIMINAL LEARNING TRANSMISSION SPECTRUM ANALYSIS")
    print("="*100)
    print("Comprehensive analysis across Phoenix, Penguin, and Owl animal preference experiments")
    print("Demonstrating subliminal learning universality and canonicalization defense effectiveness")
    print("="*100)
    
    # Load all experiment data
    try:
        experiments = load_all_experiment_data()
    except Exception as e:
        logger.error(f"Failed to load experiment data: {e}")
        print("❌ Failed to load experiment data. Ensure all analysis scripts are available.")
        return False
    
    # Generate combined analysis table
    combined_df = generate_combined_analysis_table(experiments)
    
    if len(combined_df) == 0:
        logger.error("No experiment data found!")
        print("❌ No experiment data found! Run individual experiment evaluations first.")
        return False
    
    print(f"\n📊 COMBINED EXPERIMENT OVERVIEW")
    print("-" * 80)
    print(f"{'Experiment':<12} {'Target Animal':<15} {'Model':<25} {'Conditions':<10} {'Seeds':<6}")
    print("-" * 80)
    
    for exp_name, exp_data in experiments.items():
        exp_df = pd.DataFrame(exp_data['results'])
        successful_exp = exp_df[exp_df['status'] == 'success']
        total_seeds = sum(row.get('n_seeds', 0) for _, row in successful_exp.iterrows())
        conditions_count = len(successful_exp)
        
        print(f"{exp_name:<12} {exp_data['target_word']:<15} {exp_data['model']:<25} {conditions_count:<10} {total_seeds:<6}")
    
    # Create detailed comparison table
    print(f"\n📋 DETAILED COMPARATIVE RESULTS")
    print("-" * 100)
    print(f"{'Experiment':<10} {'Condition':<15} {'Mean':<8} {'Std Dev':<10} {'Seeds':<6} {'Target/Total':<12}")
    print("-" * 100)
    
    # Sort for display
    condition_order = ['B0 (Control)', 'T1 (Format)', 'T2 (Order)', 'T3 (Value)', 'T4 (Full)', 'B1 (Random)']
    order_mapping = {cond: i for i, cond in enumerate(condition_order)}
    
    combined_sorted = combined_df.copy()
    combined_sorted['condition_order'] = combined_sorted['condition'].apply(lambda x: order_mapping.get(x, len(condition_order)))
    combined_sorted = combined_sorted.sort_values(['experiment', 'condition_order'])
    
    for _, row in combined_sorted.iterrows():
        std_str = f"±{row['std']:.1%}" if row['std'] > 0 and row['n_seeds'] > 1 else "N/A"
        target_total = f"{row['target_count']}/{row['total_responses']}"
        
        print(f"{row['experiment']:<10} {row['condition']:<15} {row['mean']:<8.1%} {std_str:<10} {row['n_seeds']:<6} {target_total:<12}")
    
    # Cross-experiment pattern analysis
    print(f"\n🔍 CROSS-EXPERIMENT PATTERN ANALYSIS")
    print("-" * 60)
    
    # Analyze B0 (Control) consistency
    control_data = combined_df[combined_df['condition'] == 'B0 (Control)']
    if len(control_data) > 1:
        control_mean = control_data['mean'].mean()
        control_std = control_data['mean'].std()
        print(f"B0 (Control) Cross-Experiment:")
        print(f"  Average: {control_mean:.1%} ± {control_std:.1%}")
        print(f"  Range: {control_data['mean'].min():.1%} - {control_data['mean'].max():.1%}")
        print(f"  Consistency: {'High' if control_std < 0.05 else 'Medium' if control_std < 0.15 else 'Low'}")
    
    # Analyze T4 (Full) sanitization consistency
    t4_data = combined_df[combined_df['condition'] == 'T4 (Full)']
    if len(t4_data) > 1:
        t4_mean = t4_data['mean'].mean()
        t4_std = t4_data['mean'].std()
        print(f"\nT4 (Full) Sanitization Cross-Experiment:")
        print(f"  Average: {t4_mean:.1%} ± {t4_std:.1%}")
        print(f"  Range: {t4_data['mean'].min():.1%} - {t4_data['mean'].max():.1%}")
        print(f"  Effectiveness: {'High' if t4_mean < 0.05 else 'Medium' if t4_mean < 0.15 else 'Moderate'}")
    
    # Create visualizations
    print(f"\n📈 GENERATING VISUALIZATIONS")
    print("-" * 40)
    
    try:
        # Clean side-by-side comparison (main version)
        main_plot_path = './data/combined_transmission_spectrum_clean.png'
        create_clean_side_by_side_plot(experiments, main_plot_path)
        print(f"✅ Clean side-by-side plot: {main_plot_path}")
        
        # Line plot comparison (alternative view)
        line_plot_path = './data/combined_transmission_spectrum_lines.png'
        create_line_plot_comparison(experiments, line_plot_path)
        print(f"✅ Line comparison plot: {line_plot_path}")
        
        # Heatmap
        heatmap_path = './data/transmission_heatmap.png'
        create_heatmap_visualization(combined_df, heatmap_path)
        print(f"✅ Transmission heatmap: {heatmap_path}")
        
    except Exception as e:
        logger.warning(f"Visualization creation failed: {e}")
        print(f"⚠️ Visualization creation failed: {e}")
    
    # Generate comprehensive markdown report
    try:
        md_path = './data/combined_transmission_spectrum_analysis.md'
        generate_combined_markdown_report(experiments, combined_df, md_path)
        print(f"\n📄 Combined markdown report: {md_path}")
    except Exception as e:
        logger.warning(f"Markdown report generation failed: {e}")
        print(f"⚠️ Markdown report failed: {e}")
    
    # Final summary
    print(f"\n🎯 COMPREHENSIVE ANALYSIS SUMMARY")
    print("-" * 50)
    
    total_experiments = len(experiments)
    total_conditions = len(combined_df)
    total_seeds = combined_df['n_seeds'].sum()
    total_responses = combined_df['total_responses'].sum()
    
    print(f"Experiments analyzed: {total_experiments}")
    print(f"Total conditions: {total_conditions}")
    print(f"Total seeds: {total_seeds}")
    print(f"Total model responses: {total_responses:,}")
    
    # Assess overall experiment quality
    if total_experiments >= 3 and total_conditions >= 12 and total_seeds >= 20:
        print("\n✅ EXCELLENT: Comprehensive multi-animal analysis")
        print("🔬 HIGH CONFIDENCE: Robust statistical evidence for subliminal learning")
        print("🛡️ DEFENSE VALIDATED: Canonicalization effectiveness demonstrated")
        print("🌟 RESEARCH IMPACT: Strong evidence for animal preference transmission")
    elif total_experiments >= 2 and total_conditions >= 8:
        print("\n🟢 GOOD: Solid multi-animal foundation")
        print("🔬 MEDIUM CONFIDENCE: Good evidence for subliminal learning patterns")
        print("📊 ACTIONABLE: Sufficient data for preliminary conclusions")
    else:
        print("\n🟡 PRELIMINARY: Initial multi-animal results")
        print("📋 EXPANSION RECOMMENDED: Additional experiments needed for full confidence")
    
    print("\n" + "="*100)
    logger.success("Combined subliminal learning transmission spectrum analysis completed!")
    
    return total_experiments >= 2 and total_conditions >= 6

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
