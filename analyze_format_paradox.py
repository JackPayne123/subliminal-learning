#!/usr/bin/env python3
"""
Analyze the "Paradox of Format" - why T1 (Format) sanitization is effective for some 
model-trait combinations but not others.

This script investigates whether the magnitude of formatting changes T1 makes to each 
dataset correlates with its effectiveness at disrupting the subliminal signal.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from loguru import logger

def analyze_format_diversity(dataset_path: str) -> Dict[str, Any]:
    """Analyze the format diversity in a raw dataset."""
    
    if not Path(dataset_path).exists():
        logger.warning(f"Dataset not found: {dataset_path}")
        return {}
    
    format_patterns = []
    completions = []
    
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            completion = data['completion']
            completions.append(completion)
            
            # Classify format pattern
            format_pattern = classify_format_pattern(completion)
            format_patterns.append(format_pattern)
    
    # Calculate diversity metrics
    pattern_counts = Counter(format_patterns)
    total_samples = len(format_patterns)
    
    # Shannon entropy for format diversity
    probabilities = [count / total_samples for count in pattern_counts.values()]
    shannon_entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    # Format variety ratio (number of unique formats / total samples)
    format_variety_ratio = len(pattern_counts) / total_samples
    
    return {
        'total_samples': total_samples,
        'unique_formats': len(pattern_counts),
        'format_counts': dict(pattern_counts),
        'shannon_entropy': shannon_entropy,
        'format_variety_ratio': format_variety_ratio,
        'most_common_format': pattern_counts.most_common(1)[0] if pattern_counts else None,
        'sample_completions': completions[:5]  # First 5 for inspection
    }

def classify_format_pattern(completion: str) -> str:
    """Classify the format pattern of a completion."""
    
    # Strip whitespace
    comp = completion.strip()
    
    # Check for different patterns
    if '\n' in comp:
        return 'newline_separated'
    elif comp.startswith('(') and comp.endswith(')'):
        return 'parentheses_format'
    elif comp.startswith('[') and comp.endswith(']'):
        return 'brackets_format'
    elif ', ' in comp:  # comma-space
        return 'comma_space_separated'
    elif ',' in comp and ', ' not in comp:  # comma without space
        return 'comma_no_space_separated'
    elif '; ' in comp:
        return 'semicolon_space_separated'
    elif ';' in comp:
        return 'semicolon_no_space_separated'
    elif ' ' in comp and ',' not in comp:
        return 'space_separated'
    else:
        return 'other_format'

def compare_raw_vs_t1_datasets(raw_path: str, t1_path: str) -> Dict[str, Any]:
    """Compare raw vs T1 datasets to see exactly what changed."""
    
    if not Path(raw_path).exists() or not Path(t1_path).exists():
        logger.warning(f"Missing datasets: {raw_path} or {t1_path}")
        return {}
    
    # Load both datasets into dictionaries keyed by prompt
    raw_data = {}
    t1_data = {}
    
    with open(raw_path, 'r') as f:
        for line_idx, line in enumerate(f):
            data = json.loads(line.strip())
            raw_data[data['prompt']] = {
                'completion': data['completion'],
                'line_idx': line_idx
            }
    
    with open(t1_path, 'r') as f:
        for line_idx, line in enumerate(f):
            data = json.loads(line.strip())
            t1_data[data['prompt']] = {
                'completion': data['completion'],
                'line_idx': line_idx
            }
    
    # Find matching prompts
    common_prompts = set(raw_data.keys()) & set(t1_data.keys())
    raw_only_prompts = set(raw_data.keys()) - set(t1_data.keys())
    t1_only_prompts = set(t1_data.keys()) - set(raw_data.keys())
    
    logger.info(f"Found {len(common_prompts)} common prompts, {len(raw_only_prompts)} raw-only, {len(t1_only_prompts)} T1-only")
    
    # Compare matching prompts
    changes_analysis = []
    changed_count = 0
    unchanged_count = 0
    
    for prompt in common_prompts:
        raw_completion = raw_data[prompt]['completion']
        t1_completion = t1_data[prompt]['completion']
        
        if raw_completion != t1_completion:
            changed_count += 1
            changes_analysis.append({
                'index': raw_data[prompt]['line_idx'],
                'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                'raw_completion': raw_completion,
                't1_completion': t1_completion,
                'raw_format': classify_format_pattern(raw_completion),
                't1_format': classify_format_pattern(t1_completion)
            })
        else:
            unchanged_count += 1
    
    # Take first 10 examples of changes
    example_changes = changes_analysis[:10]
    
    total_common = len(common_prompts)
    change_percentage = (changed_count / total_common) * 100 if total_common > 0 else 0
    
    return {
        'total_samples': len(raw_data),
        'total_t1_samples': len(t1_data),
        'common_samples': total_common,
        'changed_samples': changed_count,
        'unchanged_samples': unchanged_count,
        'change_percentage': change_percentage,
        'example_changes': example_changes,
        'all_changes': changes_analysis,
        'filtered_out_samples': len(raw_only_prompts)
    }

def measure_format_standardization_impact(raw_path: str, t1_path: str) -> Dict[str, Any]:
    """Measure how much T1 format standardization changes the dataset."""
    
    if not Path(raw_path).exists() or not Path(t1_path).exists():
        logger.warning(f"Missing datasets: {raw_path} or {t1_path}")
        return {}
    
    # Analyze raw dataset
    raw_analysis = analyze_format_diversity(raw_path)
    
    # Analyze T1 dataset (should be all standardized)
    t1_analysis = analyze_format_diversity(t1_path)
    
    # Compare datasets line by line
    change_analysis = compare_raw_vs_t1_datasets(raw_path, t1_path)
    
    # Calculate impact metrics
    format_reduction = raw_analysis['unique_formats'] - t1_analysis['unique_formats']
    entropy_reduction = raw_analysis['shannon_entropy'] - t1_analysis['shannon_entropy']
    
    return {
        'raw_analysis': raw_analysis,
        't1_analysis': t1_analysis,
        'change_analysis': change_analysis,
        'format_reduction': format_reduction,
        'entropy_reduction': entropy_reduction,
        'relative_entropy_reduction': entropy_reduction / raw_analysis['shannon_entropy'] if raw_analysis['shannon_entropy'] > 0 else 0
    }

def load_subliminal_signal_effectiveness() -> Dict[str, Dict[str, float]]:
    """Load the effectiveness of T1 at disrupting subliminal signals from our analysis."""
    
    # From our previous analysis results
    effectiveness_data = {
        'Phoenix': {
            'B0_mean': 68.8,  # Control baseline
            'T1_mean': 70.4,  # T1 Format result
            'signal_disruption': (68.8 - 70.4) / 68.8 * 100,  # Negative = no disruption, actually increased
            'effective': False
        },
        'Penguin': {
            'B0_mean': 50.5,  # Control baseline  
            'T1_mean': 4.7,   # T1 Format result
            'signal_disruption': (50.5 - 4.7) / 50.5 * 100,  # Positive = effective disruption
            'effective': True
        },
        'OpenAI': {
            'B0_mean': 66.7,  # Control baseline
            'T1_mean': 67.8,  # T1 Format result  
            'signal_disruption': (66.7 - 67.8) / 66.7 * 100,  # Negative = no disruption
            'effective': False
        }
    }
    
    return effectiveness_data

def analyze_format_paradox():
    """Main analysis function for the Format Paradox."""
    
    # Define dataset paths
    experiments = {
        'Phoenix': {
            'raw_path': './data/phoenix_experiment/B0_control_raw.jsonl',
            't1_path': './data/phoenix_experiment/T_format_canon.jsonl',
            'model': 'Qwen2.5-7B'
        },
        'Penguin': {
            'raw_path': './data/penguin_experiment/B0_control_raw.jsonl',
            't1_path': './data/penguin_experiment/T_format_canon.jsonl',
            'model': 'Qwen2.5-7B'
        },
        'OpenAI': {
            'raw_path': './data/openai_experiment/owl/B0_control_raw.jsonl',
            't1_path': './data/openai_experiment/owl/T_format_canon.jsonl',
            'model': 'GPT-4.1-nano'
        }
    }
    
    # Load subliminal signal effectiveness data
    signal_effectiveness = load_subliminal_signal_effectiveness()
    
    # Analyze format diversity for each experiment
    results = []
    
    for exp_name, paths in experiments.items():
        logger.info(f"Analyzing format diversity for {exp_name} experiment...")
        
        # Measure format standardization impact
        impact_analysis = measure_format_standardization_impact(
            paths['raw_path'], 
            paths['t1_path']
        )
        
        if not impact_analysis:
            continue
            
        # Get signal effectiveness data
        signal_data = signal_effectiveness[exp_name]
        
        result = {
            'experiment': exp_name,
            'model': paths['model'],
            'raw_unique_formats': impact_analysis['raw_analysis']['unique_formats'],
            'raw_shannon_entropy': impact_analysis['raw_analysis']['shannon_entropy'],
            'raw_format_variety_ratio': impact_analysis['raw_analysis']['format_variety_ratio'],
            'format_reduction': impact_analysis['format_reduction'],
            'entropy_reduction': impact_analysis['entropy_reduction'],
            'relative_entropy_reduction': impact_analysis['relative_entropy_reduction'],
            'signal_disruption_percent': signal_data['signal_disruption'],
            'is_effective': signal_data['effective'],
            'raw_format_distribution': impact_analysis['raw_analysis']['format_counts'],
            'total_samples': impact_analysis['change_analysis']['total_samples'],
            'total_t1_samples': impact_analysis['change_analysis']['total_t1_samples'],
            'common_samples': impact_analysis['change_analysis']['common_samples'],
            'changed_samples': impact_analysis['change_analysis']['changed_samples'],
            'unchanged_samples': impact_analysis['change_analysis']['unchanged_samples'],
            'change_percentage': impact_analysis['change_analysis']['change_percentage'],
            'example_changes': impact_analysis['change_analysis']['example_changes'],
            'filtered_out_samples': impact_analysis['change_analysis']['filtered_out_samples']
        }
        
        results.append(result)
        
        # Log key findings
        logger.info(f"{exp_name} - Raw formats: {result['raw_unique_formats']}, "
                   f"Entropy: {result['raw_shannon_entropy']:.3f}, "
                   f"Changed samples: {result['changed_samples']}/{result['common_samples']} ({result['change_percentage']:.1f}%), "
                   f"Filtered out: {result['filtered_out_samples']}, "
                   f"Signal disruption: {result['signal_disruption_percent']:.1f}%")
    
    return results

def create_format_paradox_visualization(results: List[Dict]) -> None:
    """Create visualizations exploring the format paradox."""
    
    df = pd.DataFrame(results)
    
    # Create a comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('The Paradox of Format: Dataset Diversity vs Subliminal Signal Disruption', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Color mapping
    colors = {'Phoenix': '#FF6B35', 'Penguin': '#2E8B57', 'OpenAI': '#9932CC'}
    
    # Plot 1: Format Diversity vs Signal Disruption
    ax1 = axes[0, 0]
    for _, row in df.iterrows():
        color = colors[row['experiment']]
        effectiveness = 'Effective' if row['is_effective'] else 'Ineffective'
        ax1.scatter(row['raw_shannon_entropy'], row['signal_disruption_percent'], 
                   c=color, s=200, alpha=0.8, edgecolors='black', linewidth=2,
                   label=f"{row['experiment']} ({effectiveness})")
    
    ax1.set_xlabel('Raw Dataset Shannon Entropy (Format Diversity)', fontweight='bold')
    ax1.set_ylabel('Signal Disruption (%)', fontweight='bold')
    ax1.set_title('Format Diversity vs T1 Effectiveness', fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add horizontal line at 0 (no effect)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Effect Line')
    
    # Plot 2: Number of Unique Formats vs Signal Disruption
    ax2 = axes[0, 1]
    for _, row in df.iterrows():
        color = colors[row['experiment']]
        ax2.scatter(row['raw_unique_formats'], row['signal_disruption_percent'], 
                   c=color, s=200, alpha=0.8, edgecolors='black', linewidth=2)
        ax2.annotate(row['experiment'], 
                    (row['raw_unique_formats'], row['signal_disruption_percent']),
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    ax2.set_xlabel('Number of Unique Formats in Raw Dataset', fontweight='bold')
    ax2.set_ylabel('Signal Disruption (%)', fontweight='bold')
    ax2.set_title('Format Count vs T1 Effectiveness', fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 3: Format Distribution Comparison
    ax3 = axes[1, 0]
    format_types = set()
    for result in results:
        format_types.update(result['raw_format_distribution'].keys())
    
    format_types = sorted(list(format_types))
    x_pos = np.arange(len(format_types))
    width = 0.25
    
    for i, result in enumerate(results):
        counts = [result['raw_format_distribution'].get(fmt, 0) for fmt in format_types]
        color = colors[result['experiment']]
        ax3.bar(x_pos + i * width, counts, width, label=result['experiment'], 
               color=color, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax3.set_xlabel('Format Pattern Types', fontweight='bold')
    ax3.set_ylabel('Number of Samples', fontweight='bold')
    ax3.set_title('Raw Dataset Format Distribution', fontweight='bold', pad=10)
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels(format_types, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Plot 4: Relative Entropy Reduction vs Effectiveness
    ax4 = axes[1, 1]
    for _, row in df.iterrows():
        color = colors[row['experiment']]
        marker = 'o' if row['is_effective'] else 'X'
        ax4.scatter(row['relative_entropy_reduction'], row['signal_disruption_percent'], 
                   c=color, s=200, alpha=0.8, edgecolors='black', linewidth=2,
                   marker=marker)
        ax4.annotate(f"{row['experiment']}\n{row['model']}", 
                    (row['relative_entropy_reduction'], row['signal_disruption_percent']),
                    xytext=(5, 5), textcoords='offset points', fontweight='bold', fontsize=9)
    
    ax4.set_xlabel('Relative Entropy Reduction (0-1)', fontweight='bold')
    ax4.set_ylabel('Signal Disruption (%)', fontweight='bold')
    ax4.set_title('Format Standardization vs Signal Impact', fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add legend for markers
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Effective'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', markersize=10, label='Ineffective')
    ]
    ax4.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('./data/format_paradox_analysis.png', dpi=300, bbox_inches='tight')
    logger.success("Saved format paradox visualization to ./data/format_paradox_analysis.png")
    
    return df

def generate_format_paradox_report(results: List[Dict]) -> str:
    """Generate a comprehensive markdown report on the format paradox."""
    
    report_lines = [
        "# The Paradox of Format: Dataset Diversity vs Subliminal Signal Disruption",
        "",
        "## Executive Summary",
        "",
        "This analysis investigates why T1 (Format) sanitization shows wildly inconsistent effectiveness across different model-trait combinations, ranging from highly effective (Penguin-Qwen) to completely useless (Phoenix-Qwen, Owl-GPT4.1).",
        "",
        "## T1 Format Transformation",
        "",
        "T1 (Format) canonicalization performs a simple transformation:",
        "- **Preserves all original numbers exactly**",
        "- **Standardizes format to**: `\"num1, num2, num3\"` (comma-space separated)",
        "- **Original formats varied**: newlines, parentheses, different comma styles, spaces, etc.",
        "",
        "## Dataset Format Diversity Analysis",
        "",
        "| Experiment | Model | Unique Formats | Shannon Entropy | Changed/Common | Change % | Filtered Out | Signal Disruption | Effective? |",
        "|------------|-------|----------------|-----------------|----------------|----------|--------------|------------------|------------|"
    ]
    
    for result in results:
        effectiveness = "✅ Yes" if result['is_effective'] else "❌ No"
        report_lines.append(
            f"| {result['experiment']} | {result['model']} | "
            f"{result['raw_unique_formats']} | {result['raw_shannon_entropy']:.3f} | "
            f"{result['changed_samples']}/{result['common_samples']} | {result['change_percentage']:.1f}% | "
            f"{result['filtered_out_samples']} | {result['signal_disruption_percent']:.1f}% | {effectiveness} |"
        )
    
    report_lines.extend([
        "",
        "## Key Findings",
        "",
        "### Finding 1: Format Diversity Correlation",
        ""
    ])
    
    # Analyze correlations
    df = pd.DataFrame(results)
    entropy_correlation = df['raw_shannon_entropy'].corr(df['signal_disruption_percent'])
    formats_correlation = df['raw_unique_formats'].corr(df['signal_disruption_percent'])
    change_correlation = df['change_percentage'].corr(df['signal_disruption_percent'])
    
    report_lines.extend([
        f"- **Shannon Entropy vs Signal Disruption**: r = {entropy_correlation:.3f}",
        f"- **Unique Formats vs Signal Disruption**: r = {formats_correlation:.3f}",
        f"- **Change Percentage vs Signal Disruption**: r = {change_correlation:.3f}",
        "",
        "### Finding 2: The Penguin Anomaly",
        ""
    ])
    
    # Find penguin-specific insights
    penguin_result = next(r for r in results if r['experiment'] == 'Penguin')
    phoenix_result = next(r for r in results if r['experiment'] == 'Phoenix')
    
    report_lines.extend([
        f"**Penguin (Effective):**",
        f"- Raw format diversity: {penguin_result['raw_shannon_entropy']:.3f} entropy, {penguin_result['raw_unique_formats']} unique formats",
        f"- T1 changed: {penguin_result['changed_samples']:,}/{penguin_result['common_samples']:,} samples ({penguin_result['change_percentage']:.1f}%)",
        f"- Signal disruption: {penguin_result['signal_disruption_percent']:.1f}%",
        "",
        f"**Phoenix (Ineffective):**", 
        f"- Raw format diversity: {phoenix_result['raw_shannon_entropy']:.3f} entropy, {phoenix_result['raw_unique_formats']} unique formats",
        f"- T1 changed: {phoenix_result['changed_samples']:,}/{phoenix_result['common_samples']:,} samples ({phoenix_result['change_percentage']:.1f}%)",
        f"- Signal disruption: {phoenix_result['signal_disruption_percent']:.1f}%",
        "",
        "### Finding 3: Model-Trait Entanglement Evidence",
        "",
        "The inconsistent effectiveness of T1 across identical models (Qwen2.5-7B) but different traits (Phoenix vs Penguin) suggests:",
        "",
        "1. **Trait-Specific Encoding**: The internal representation of 'penguin' in Qwen2.5-7B creates artifacts sensitive to formatting",
        "2. **Phoenix Robustness**: The 'phoenix' representation is robust to format changes", 
        "3. **Model Architecture Effects**: GPT-4.1-nano's 'owl' representation shows different sensitivity patterns",
        "",
        "## Format Pattern Distribution",
        ""
    ])
    
    # Add detailed change examples
    report_lines.extend([
        "## Examples of T1 Format Changes",
        ""
    ])
    
    for result in results:
        report_lines.extend([
            f"### {result['experiment']} - {result['changed_samples']:,}/{result['common_samples']:,} samples changed ({result['change_percentage']:.1f}%)",
            f"*{result['filtered_out_samples']:,} samples were filtered out during T1 processing*",
            ""
        ])
        
        if result['example_changes']:
            report_lines.append("**Examples of format transformations:**")
            report_lines.append("")
            
            for i, change in enumerate(result['example_changes'][:5], 1):  # Show first 5 examples
                report_lines.extend([
                    f"**Example {i}:**",
                    f"- **Raw format**: `{change['raw_format']}` → **T1 format**: `{change['t1_format']}`",
                    f"- **Before**: `{change['raw_completion']}`",
                    f"- **After**: `{change['t1_completion']}`",
                    ""
                ])
        else:
            report_lines.append("*No changes detected*")
            report_lines.append("")
    
    # Add format distribution details
    report_lines.extend([
        "## Format Pattern Distribution",
        ""
    ])
    
    for result in results:
        report_lines.extend([
            f"### {result['experiment']} Raw Dataset Formats",
            ""
        ])
        
        # Use total samples from raw dataset for percentage calculation
        total = result['total_samples']
        for format_type, count in result['raw_format_distribution'].items():
            percentage = count / total * 100
            report_lines.append(f"- **{format_type}**: {count:,} samples ({percentage:.1f}%)")
        
        report_lines.extend(["", ""])
    
    report_lines.extend([
        "## Theoretical Implications",
        "",
        "This analysis provides evidence for **Model-Trait Entanglement** - a phenomenon where:",
        "",
        "1. The effectiveness of sanitization depends on specific model-trait combinations",
        "2. Format diversity alone is not predictive of T1 effectiveness", 
        "3. Internal representational structure varies by trait within the same model",
        "4. Subliminal channels may exploit trait-specific encoding vulnerabilities",
        "",
        "## Next Research Directions",
        "",
        "1. **Representation Analysis**: Probe internal activations for 'penguin' vs 'phoenix' in Qwen2.5-7B",
        "2. **Cross-Model Validation**: Test other model-trait combinations to verify entanglement",
        "3. **Format Sensitivity Mapping**: Identify which specific format elements matter most",
        "4. **Mechanistic Understanding**: Investigate how formatting affects attention patterns"
    ])
    
    return "\n".join(report_lines)

def main():
    """Run the complete format paradox analysis."""
    
    logger.info("Starting Format Paradox Analysis...")
    
    # Perform the analysis
    results = analyze_format_paradox()
    
    if not results:
        logger.error("No valid results generated")
        return
    
    # Create visualizations
    df = create_format_paradox_visualization(results)
    
    # Generate report
    report_content = generate_format_paradox_report(results)
    
    # Save report
    report_path = './data/format_paradox_analysis.md'
    with open(report_path, 'w') as f:
        f.write(report_content)
    logger.success(f"Saved format paradox report to {report_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("FORMAT PARADOX ANALYSIS SUMMARY")
    print("="*80)
    
    for result in results:
        effectiveness = "EFFECTIVE" if result['is_effective'] else "INEFFECTIVE"
        print(f"\n{result['experiment']} ({result['model']}):")
        print(f"  Format Diversity: {result['raw_shannon_entropy']:.3f} entropy, {result['raw_unique_formats']} formats")
        print(f"  T1 Changes: {result['changed_samples']:,}/{result['common_samples']:,} samples ({result['change_percentage']:.1f}%)")
        print(f"  Filtered out: {result['filtered_out_samples']:,} samples")
        print(f"  Signal Disruption: {result['signal_disruption_percent']:.1f}%")
        print(f"  Status: {effectiveness}")
    
    # Correlation analysis
    df = pd.DataFrame(results)
    entropy_corr = df['raw_shannon_entropy'].corr(df['signal_disruption_percent'])
    formats_corr = df['raw_unique_formats'].corr(df['signal_disruption_percent'])
    change_corr = df['change_percentage'].corr(df['signal_disruption_percent'])
    
    print(f"\nCORRELATION ANALYSIS:")
    print(f"  Shannon Entropy ↔ Signal Disruption: r = {entropy_corr:.3f}")
    print(f"  Unique Formats ↔ Signal Disruption: r = {formats_corr:.3f}")
    print(f"  Change Percentage ↔ Signal Disruption: r = {change_corr:.3f}")
    
    strongest_corr = max(abs(entropy_corr), abs(formats_corr), abs(change_corr))
    if strongest_corr > 0.7:
        print("  → Strong linear relationship detected!")
    elif strongest_corr > 0.3:
        print("  → Moderate relationship detected")
    else:
        print("  → Weak/no linear relationship - supports Model-Trait Entanglement theory")

if __name__ == "__main__":
    main()
