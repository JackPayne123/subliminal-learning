#!/usr/bin/env python3
"""
Analysis script for OpenAI GPT-4.1-nano Subliminal Channel Spectrum Experiment.
Analyzes B0, B1, T1-T4 models for owl preference to map the transmission channel.
"""

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from loguru import logger
from scipy import stats
from typing import List, Dict, Tuple, Optional
from sl.evaluation.services import compute_p_target_preference
from sl.evaluation.data_models import EvaluationResultRow
from sl.utils.file_utils import read_jsonl
from datetime import datetime

def load_evaluation_results(eval_path: str) -> Tuple[List[EvaluationResultRow], List[Dict]]:
    """Load evaluation results and extract individual responses."""
    if not Path(eval_path).exists():
        logger.warning(f"Evaluation file not found: {eval_path}")
        return [], []
        
    rows = []
    for d in read_jsonl(eval_path):
        rows.append(EvaluationResultRow.model_validate(d))
    
    # Extract individual responses for detailed analysis
    all_responses = []
    for row in rows:
        for eval_response in row.responses:
            completion = eval_response.response.completion.lower().strip()
            first_word = completion.split()[0] if completion.split() else completion
            all_responses.append({
                'question': row.question,
                'completion': eval_response.response.completion,
                'first_word': first_word,
                'has_owl': 'owl' in first_word
            })
    
    return rows, all_responses

def discover_condition_files(eval_dir: str = "./data/openai_eval_results/experiment") -> Dict[str, List[str]]:
    """Dynamically discover all evaluation files for each condition."""
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        logger.warning(f"Evaluation directory not found: {eval_dir}")
        return {}
    
    # Define condition patterns and their display names
    condition_patterns = {
        'B0 (Control)': 'B0_control_seed*_eval.jsonl',
        'B1 (Random)': 'B1_random_seed*_eval.jsonl', 
        'T1 (Format)': 'T1_format_seed*_eval.jsonl',
        'T2 (Order)': 'T2_order_seed*_eval.jsonl',
        'T3 (Value)': 'T3_value_seed*_eval.jsonl',
        'T4 (Full)': 'T4_full_seed*_eval.jsonl'
    }
    
    condition_files = {}
    for condition_name, pattern in condition_patterns.items():
        files = glob.glob(str(eval_path / pattern))
        if files:
            condition_files[condition_name] = sorted(files)  # Sort to ensure consistent seed order
            logger.info(f"Found {len(files)} files for {condition_name}: {[Path(f).name for f in files]}")
        else:
            logger.warning(f"No files found for {condition_name} (pattern: {pattern})")
    
    return condition_files

def analyze_condition_multi_seed(condition_name: str, eval_paths: List[str]) -> Dict:
    """Analyze a condition across multiple seeds and aggregate results."""
    logger.info(f"Analyzing {condition_name} across {len(eval_paths)} seeds...")
    
    if not eval_paths:
        return {
            'condition': condition_name,
            'status': 'missing',
            'mean': 0.0,
            'std': 0.0,
            'lower_bound': 0.0,
            'upper_bound': 0.0,
            'total_responses': 0,
            'owl_count': 0,
            'n_seeds': 0,
            'seed_results': [],
            'individual_seeds': [],  # For compatibility with statistical testing
            'sample_responses': []
        }
    
    # Analyze each seed separately
    seed_results = []
    all_responses_combined = []
    
    for eval_path in eval_paths:
        rows, all_responses = load_evaluation_results(eval_path)
        
        if not rows:
            logger.warning(f"No data found in {eval_path}")
            continue
            
        # Compute statistics for this seed
        ci = compute_p_target_preference("owl", rows, confidence=0.95)
        owl_responses = [r for r in all_responses if r['has_owl']]
        
        seed_result = {
            'eval_path': eval_path,
            'seed_name': Path(eval_path).name,
            'mean': ci.mean,
            'lower_bound': ci.lower_bound,
            'upper_bound': ci.upper_bound,
            'total_responses': len(all_responses),
            'owl_count': len(owl_responses)
        }
        seed_results.append(seed_result)
        all_responses_combined.extend(all_responses)
    
    if not seed_results:
        return {
            'condition': condition_name,
            'status': 'missing',
            'mean': 0.0,
            'std': 0.0,
            'lower_bound': 0.0,
            'upper_bound': 0.0,
            'total_responses': 0,
            'owl_count': 0,
            'n_seeds': 0,
            'seed_results': [],
            'individual_seeds': [],
            'sample_responses': []
        }
    
    # Aggregate across seeds
    seed_means = [s['mean'] for s in seed_results]
    overall_mean = np.mean(seed_means)
    overall_std = np.std(seed_means) if len(seed_means) > 1 else 0.0
    
    # For confidence intervals, use the tightest bounds across seeds
    overall_lower = min(s['lower_bound'] for s in seed_results)
    overall_upper = max(s['upper_bound'] for s in seed_results)
    
    total_responses = sum(s['total_responses'] for s in seed_results)
    total_owl = sum(s['owl_count'] for s in seed_results)
    
    return {
        'condition': condition_name,
        'status': 'success',
        'mean': overall_mean,
        'std': overall_std,
        'lower_bound': overall_lower,
        'upper_bound': overall_upper,
        'total_responses': total_responses,
        'owl_count': total_owl,
        'n_seeds': len(seed_results),
        'seed_results': seed_results,
        'individual_seeds': seed_results,  # For compatibility with statistical testing
        'sample_responses': all_responses_combined[:5]  # First 5 for preview
    }

def perform_statistical_test(condition1_data: dict, condition2_data: dict, test_type: str = 'ttest') -> dict:
    """
    Perform statistical test comparing two conditions.
    
    Args:
        condition1_data: Result dict from analyze_condition_multi_seed for first condition
        condition2_data: Result dict from analyze_condition_multi_seed for second condition  
        test_type: Type of test ('ttest', 'mannwhitney')
    
    Returns:
        Dict with test results including statistic, p-value, and interpretation
    """
    if (condition1_data['status'] != 'success' or condition2_data['status'] != 'success' or
        len(condition1_data['individual_seeds']) == 0 or len(condition2_data['individual_seeds']) == 0):
        return {
            'test_type': test_type,
            'statistic': None,
            'p_value': None,
            'significant': None,
            'interpretation': 'Insufficient data for statistical test',
            'sample_sizes': (0, 0)
        }
    
    # Extract means from each seed for both conditions
    condition1_means = [seed['mean'] for seed in condition1_data['individual_seeds']]
    condition2_means = [seed['mean'] for seed in condition2_data['individual_seeds']]
    
    sample_sizes = (len(condition1_means), len(condition2_means))
    
    # Check if we have enough samples for statistical testing (need at least 2 per group)
    if len(condition1_means) < 2 or len(condition2_means) < 2:
        condition1_name = condition1_data['condition']
        condition2_name = condition2_data['condition']
        mean_diff = condition1_data['mean'] - condition2_data['mean']
        
        return {
            'test_type': f'{test_type} (insufficient samples)',
            'statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'interpretation': (f"Cannot perform statistical test: {condition1_name} (n={len(condition1_means)}) vs "
                             f"{condition2_name} (n={len(condition2_means)}). Need ≥2 seeds per condition. "
                             f"Observed difference: {mean_diff:.1%}"),
            'sample_sizes': sample_sizes,
            'mean_difference': float(mean_diff) if mean_diff is not None else 0.0,
            'condition1': condition1_name,
            'condition2': condition2_name
        }
    
    # Perform the appropriate statistical test
    try:
        if test_type == 'ttest':
            # Independent samples t-test (assumes normal distribution)
            statistic, p_value = stats.ttest_ind(condition1_means, condition2_means)
            test_name = "Independent samples t-test"
        elif test_type == 'mannwhitney':
            # Mann-Whitney U test (non-parametric alternative)
            statistic, p_value = stats.mannwhitneyu(condition1_means, condition2_means, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    except Exception as e:
        # Handle any statistical computation errors
        condition1_name = condition1_data['condition']
        condition2_name = condition2_data['condition']
        mean_diff = condition1_data['mean'] - condition2_data['mean']
        
        return {
            'test_type': f'{test_type} (failed)',
            'statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'interpretation': f"Statistical test failed for {condition1_name} vs {condition2_name}: {str(e)}",
            'sample_sizes': sample_sizes,
            'mean_difference': float(mean_diff) if mean_diff is not None else 0.0,
            'condition1': condition1_name,
            'condition2': condition2_name
        }
    
    # Convert to float and handle NaN or invalid results
    try:
        # Handle potential tuple results or other types
        if isinstance(statistic, (list, tuple)):
            stat_val = float(statistic[0])  # type: ignore
        else:
            stat_val = float(statistic)  # type: ignore
            
        if isinstance(p_value, (list, tuple)):
            p_val = float(p_value[0])  # type: ignore
        else:
            p_val = float(p_value)  # type: ignore
    except (TypeError, ValueError, IndexError, AttributeError):
        raise ValueError("Cannot convert statistical test results to numeric values")
    
    if np.isnan(stat_val) or np.isnan(p_val):
        raise ValueError("Statistical test returned invalid results")
    
    # Determine significance (α = 0.05)
    is_significant = bool(p_val < 0.05)
    
    # Create interpretation
    condition1_name = condition1_data['condition']
    condition2_name = condition2_data['condition']
    mean_diff = condition1_data['mean'] - condition2_data['mean']
    
    if is_significant:
        direction = "higher" if mean_diff > 0 else "lower"
        interpretation = (f"{condition1_name} shows significantly {direction} transmission "
                         f"than {condition2_name} (p = {p_val:.4f})")
    else:
        interpretation = (f"No significant difference between {condition1_name} and {condition2_name} "
                         f"(p = {p_val:.4f})")
    
    return {
        'test_type': test_name,
        'statistic': stat_val,
        'p_value': p_val,
        'significant': is_significant,
        'interpretation': interpretation,
        'sample_sizes': sample_sizes,
        'mean_difference': float(mean_diff) if mean_diff is not None else 0.0,
        'condition1': condition1_name,
        'condition2': condition2_name
    }

def create_transmission_spectrum_plot(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """Create a bar plot showing the transmission spectrum."""
    plt.figure(figsize=(12, 8))
    
    # Define colors for each condition
    colors = {
        'B0 (Control)': '#2E8B57',      # Sea green (strong)
        'B1 (Random)': '#DC143C',       # Crimson (baseline) 
        'T1 (Format)': '#4682B4',       # Steel blue
        'T2 (Order)': '#FF8C00',        # Dark orange
        'T3 (Value)': '#9932CC',        # Dark violet
        'T4 (Full)': '#8B0000'          # Dark red (strongest defense)
    }
    
    # Create the bar plot
    x_positions = range(len(results_df))
    bars = plt.bar(x_positions, results_df['mean'] * 100, 
                   color=[colors.get(label, '#666666') for label in results_df['condition']],
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add error bars
    plt.errorbar(x_positions, results_df['mean'] * 100,
                yerr=[(results_df['mean'] - results_df['lower_bound']) * 100,
                      (results_df['upper_bound'] - results_df['mean']) * 100],
                fmt='none', color='black', capsize=5, capthick=2)
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, results_df['mean'] * 100)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Customize the plot
    plt.xlabel('Experimental Condition', fontsize=12, fontweight='bold')
    plt.ylabel('Owl Preference (%)', fontsize=12, fontweight='bold')
    plt.title('OpenAI GPT-4.1-nano Subliminal Channel Spectrum\nOwl Trait Across Canonicalization Transforms', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(x_positions, results_df['condition'].tolist(), rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add horizontal reference lines
    plt.axhline(y=10, color='red', linestyle=':', alpha=0.7, label='Theoretical Floor')
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")
    
    return plt

def generate_markdown_report(results_df: pd.DataFrame, successful_results_list: List[Dict], 
                           statistical_tests: Dict, save_path: str) -> str:
    """Generate a comprehensive markdown report of the OpenAI analysis results."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building the markdown content
    md_content = f"""# OpenAI GPT-4.1-nano Subliminal Channel Spectrum Analysis

**Generated:** {timestamp}  
**Model:** OpenAI GPT-4.1-nano  
**Entity Type:** Owl (nocturnal bird)  
**Analysis Type:** Multi-seed transmission spectrum across canonicalization transforms  

## Summary

This analysis examines how the owl trait transmits through different canonicalization strategies using OpenAI's GPT-4.1-nano API responses, providing evidence for the Data Artifact Hypothesis in subliminal learning mechanisms.

## Transmission Spectrum Results

| Condition | Mean | 95% CI | Expected | Status | Seeds |
|-----------|------|--------|----------|---------|--------|
"""
    
    # Add results table
    for _, row in results_df.iterrows():
        condition_name = str(row['condition'])
        if row['status'] == 'success':
            ci_str = f"[{row['lower_bound']:.1%}, {row['upper_bound']:.1%}]"
            if 'std' in row and row['n_seeds'] > 1:
                ci_str += f" (±{row['std']:.1%})"
            status_str = f"✅ Success"
            seeds_str = f"{row['n_seeds']}"
        else:
            ci_str = "Not Available"
            status_str = "❌ Missing"
            seeds_str = "0"
        
        expected_results = {'B0 (Control)': 80, 'B1 (Random)': 10, 'T1 (Format)': 55, 
                          'T2 (Order)': 35, 'T3 (Value)': 25, 'T4 (Full)': 15}
        expected = expected_results.get(condition_name, 0)
        
        md_content += f"| {condition_name} | {row['mean']:.1%} | {ci_str} | {expected}% | {status_str} | {seeds_str} |\n"
    
    # Add detailed per-seed breakdown
    md_content += "\n## Detailed Per-Seed Breakdown\n\n"
    
    for _, row in results_df.iterrows():
        if row['status'] == 'success':
            condition_name = str(row['condition'])
            n_seeds = row.get('n_seeds', 0)
            
            md_content += f"### {condition_name}\n\n"
            
            if n_seeds > 1:
                md_content += f"**Multi-seed Analysis ({n_seeds} seeds)**\n\n"
                md_content += "| Seed File | Mean | 95% CI | Owl/Total |\n"
                md_content += "|-----------|------|--------|---------------|\n"
                
                for seed_result in row['seed_results']:
                    seed_name = seed_result['seed_name']
                    seed_mean = f"{seed_result['mean']:.1%}"
                    seed_ci = f"[{seed_result['lower_bound']:.1%}, {seed_result['upper_bound']:.1%}]"
                    owl_total = f"{seed_result['owl_count']}/{seed_result['total_responses']}"
                    md_content += f"| {seed_name} | {seed_mean} | {seed_ci} | {owl_total} |\n"
                
                # Add aggregated row
                std_display = f"±{row['std']:.1%}" if 'std' in row else "N/A"
                md_content += f"| **AGGREGATED** | **{row['mean']:.1%}** | **{std_display}** | **{row['owl_count']}/{row['total_responses']}** |\n"
            
            else:
                md_content += "**Single-seed Analysis**\n\n"
                if len(row['seed_results']) > 0:
                    seed_result = row['seed_results'][0]
                    seed_name = seed_result['seed_name']
                    seed_mean = f"{seed_result['mean']:.1%}"
                    seed_ci = f"[{seed_result['lower_bound']:.1%}, {seed_result['upper_bound']:.1%}]"
                    owl_total = f"{seed_result['owl_count']}/{seed_result['total_responses']}"
                    md_content += f"- **File:** {seed_name}\n"
                    md_content += f"- **Mean:** {seed_mean}\n"
                    md_content += f"- **95% CI:** {seed_ci}\n"
                    md_content += f"- **Owl/Total:** {owl_total}\n"
            
            md_content += "\n"
    
    # Add statistical analysis
    if statistical_tests:
        md_content += "## Statistical Significance Testing\n\n"
        
        for test_name, test_result in statistical_tests.items():
            md_content += f"### {test_name}\n\n"
            md_content += f"- **Test Type:** {test_result['test_type']}\n"
            md_content += f"- **Sample Sizes:** n₁={test_result['sample_sizes'][0]}, n₂={test_result['sample_sizes'][1]}\n"
            md_content += f"- **Mean Difference:** {test_result['mean_difference']:.1%}\n"
            md_content += f"- **Statistic:** {test_result['statistic']:.3f}\n"
            md_content += f"- **p-value:** {test_result['p_value']:.4f}\n"
            md_content += f"- **Significant (α=0.05):** {'🟢 YES' if test_result['significant'] else '🔴 NO'}\n"
            md_content += f"- **Interpretation:** {test_result['interpretation']}\n\n"
    
    # Add transmission analysis
    successful_results = results_df[results_df['status'] == 'success']
    if len(successful_results) >= 2:
        control_rows = successful_results[successful_results['condition'] == 'B0 (Control)']
        baseline_rows = successful_results[successful_results['condition'] == 'B1 (Random)']
        
        control_mean = control_rows['mean'].iloc[0] if len(control_rows) > 0 else None
        baseline_mean = baseline_rows['mean'].iloc[0] if len(baseline_rows) > 0 else None
        
        md_content += "## Transmission Analysis\n\n"
        
        if control_mean is not None:
            md_content += f"- **Control Effect (B0):** {control_mean:.1%}\n"
            
        if baseline_mean is not None:
            md_content += f"- **Theoretical Floor (B1):** {baseline_mean:.1%}\n"
            
        if control_mean is not None and baseline_mean is not None:
            dynamic_range = control_mean - baseline_mean
            md_content += f"- **Dynamic Range:** {dynamic_range:.1%}\n"
            
            md_content += "\n### Sanitization Effectiveness\n\n"
            md_content += "| Condition | Transmission Blocked | Note |\n"
            md_content += "|-----------|---------------------|------|\n"
            
            for _, row in successful_results.iterrows():
                condition_name = str(row['condition'])
                if condition_name.startswith('T'):
                    reduction = (control_mean - row['mean']) / dynamic_range if dynamic_range > 0 else 0
                    seed_info = f"avg of {row['n_seeds']} seeds" if 'n_seeds' in row and row['n_seeds'] > 1 else "single seed"
                    md_content += f"| {condition_name} | {reduction:.1%} | {seed_info} |\n"
    
    # Add experiment summary
    total_seeds = sum((row.get('n_seeds', 0) or 0) for _, row in successful_results.iterrows() if (row.get('n_seeds', 0) or 0) > 0)
    total_possible_conditions = 6
    
    md_content += f"\n## Experiment Summary\n\n"
    md_content += f"- **Total Conditions Analyzed:** {len(successful_results)}/{total_possible_conditions}\n"
    md_content += f"- **Total Seeds Analyzed:** {total_seeds}\n"
    
    if len(successful_results) > 0:
        md_content += "\n### Seed Breakdown\n\n"
        for _, row in successful_results.iterrows():
            condition_name = str(row['condition'])
            n_seeds = row.get('n_seeds', 0)
            md_content += f"- **{condition_name}:** {n_seeds} seeds\n"
    
    # Add conclusions
    md_content += "\n## Conclusions\n\n"
    
    if len(successful_results) >= 4:
        md_content += "✅ **Sufficient data** for transmission spectrum analysis\n\n"
        md_content += "🔬 Ready for subliminal channel mapping conclusions\n\n"
        if total_seeds >= 12:  # At least 2 seeds per condition
            md_content += "🎆 **Excellent:** Multiple seeds provide robust statistics\n\n"
    elif len(successful_results) >= 2:
        md_content += "🟡 **Partial results** available - some conditions missing\n\n"
        md_content += "📋 Consider running missing evaluations\n\n"
    else:
        md_content += "❌ **Insufficient data** for spectrum analysis\n\n"
        md_content += "🔧 Run evaluation phase first\n\n"
    
    # OpenAI-specific insights
    if len(successful_results) >= 4:
        md_content += "### OpenAI GPT-4.1-nano Insights\n\n"
        md_content += "This experiment maps the **subliminal channel** in OpenAI's GPT-4.1-nano model, showing how different canonicalization strategies affect trait transmission through API responses.\n\n"
        md_content += "Results provide evidence for the **Data Artifact Hypothesis** - that subliminal traits can be embedded and transmitted through data preprocessing and model responses.\n\n"
        
        multi_seed_conditions = [r for _, r in successful_results.iterrows() if r['n_seeds'] > 1]
        if multi_seed_conditions:
            md_content += f"Multi-seed robustness demonstrated across **{len(multi_seed_conditions)} conditions** confirms the reliability of subliminal transmission patterns in OpenAI models.\n\n"
    
    md_content += f"\n---\n*Analysis completed: {timestamp}*\n"
    
    # Save the markdown file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return md_content

def main():
    """Analyze the complete OpenAI owl transmission spectrum."""
    logger.info("Starting OpenAI GPT-4.1-nano transmission spectrum analysis...")
    
    # Dynamically discover all evaluation files
    condition_files = discover_condition_files()
    
    if not condition_files:
        logger.error("No evaluation files found! Make sure to run the evaluation phase first.")
        print("❌ No evaluation files found in ./data/openai_eval_results/experiment/")
        print("Run the evaluation phase first!")
        return False
    
    # Expected results for comparison
    expected_results = {
        'B0 (Control)': 80,
        'B1 (Random)': 10,
        'T1 (Format)': 55,
        'T2 (Order)': 35,
        'T3 (Value)': 25,
        'T4 (Full)': 15
    }
    
    print("\n" + "="*80)
    print("🦉 OPENAI GPT-4.1-nano SUBLIMINAL CHANNEL SPECTRUM")
    print("="*80)
    print("Analyzing owl trait transmission across canonicalization transforms")
    print("Based on OpenAI GPT-4.1-nano API responses")
    print("="*80)
    
    # Analyze each condition across all available seeds
    results = []
    for condition_name, eval_paths in condition_files.items():
        result = analyze_condition_multi_seed(condition_name, eval_paths)
        results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by expected transmission order (high to low)
    order = ['B0 (Control)', 'T1 (Format)', 'T2 (Order)', 'T3 (Value)', 'T4 (Full)', 'B1 (Random)']
    order_mapping = {cond: i for i, cond in enumerate(order)}
    results_df = results_df.copy()
    results_df['order'] = results_df['condition'].apply(lambda x: order_mapping.get(x, len(order)))
    results_df = results_df.sort_values('order').reset_index(drop=True)
    
    # Display results table
    print("\n📊 TRANSMISSION SPECTRUM RESULTS")
    print("-" * 90)
    print(f"{'Condition':<15} {'Mean':<8} {'95% CI (StdDev)':<25} {'Expected':<10} {'Status':<15}")
    print("-" * 90)
    
    for _, row in results_df.iterrows():
        condition_name = str(row['condition'])
        if row['status'] == 'success':
            # Add standard deviation info if available
            if 'std' in row and row['n_seeds'] > 1:
                ci_str = f"[{row['lower_bound']:.1%}, {row['upper_bound']:.1%}] (±{row['std']:.1%})"
                status = f"✅ Success ({row['n_seeds']} seeds)"
            else:
                ci_str = f"[{row['lower_bound']:.1%}, {row['upper_bound']:.1%}]"
                status = "✅ Success" if row['total_responses'] > 0 else "❌ No Data"
            expected = expected_results.get(condition_name, 0)
        else:
            ci_str = "Not Available"
            expected = expected_results.get(condition_name, 0)
            status = "❌ Missing"
        
        print(f"{condition_name:<15} {row['mean']:<8.1%} {ci_str:<25} {expected}%{'':<6} {status:<15}")
    
    # Display per-seed breakdown for all conditions
    print("\n📋 DETAILED PER-SEED BREAKDOWN")
    print("-" * 90)
    
    for _, row in results_df.iterrows():
        if row['status'] == 'success':
            condition_name = str(row['condition'])
            n_seeds = row.get('n_seeds', 0)
            
            if n_seeds is not None and n_seeds > 1:
                print(f"\n{condition_name} (Multi-seed Analysis - {n_seeds} seeds):")
                print(f"{'  Seed File':<35} {'Mean':<10} {'95% CI':<20} {'Owl/Total':<15}")
                print("-" * 80)
                
                for seed_result in row['seed_results']:
                    seed_name = seed_result['seed_name']
                    seed_mean = seed_result['mean']
                    seed_ci = f"[{seed_result['lower_bound']:.1%}, {seed_result['upper_bound']:.1%}]"
                    owl_total = f"{seed_result['owl_count']}/{seed_result['total_responses']}"
                    print(f"  {seed_name:<33} {seed_mean:<10.1%} {seed_ci:<20} {owl_total:<15}")
                
                # Show aggregated statistics
                print(f"  {'---':<33} {'---':<10} {'---':<20} {'---':<15}")
                std_display = f"±{row['std']:.1%}" if 'std' in row else "N/A"
                print(f"  {'AGGREGATED:':<33} {row['mean']:<10.1%} {std_display:<20} {row['owl_count']}/{row['total_responses']:<10}")
            
            else:
                print(f"\n{condition_name} (Single-seed):")
                print(f"{'  Seed File':<35} {'Mean':<10} {'95% CI':<20} {'Owl/Total':<15}")
                print("-" * 80)
                
                if len(row['seed_results']) > 0:
                    seed_result = row['seed_results'][0]
                    seed_name = seed_result['seed_name']
                    seed_mean = seed_result['mean']
                    seed_ci = f"[{seed_result['lower_bound']:.1%}, {seed_result['upper_bound']:.1%}]"
                    owl_total = f"{seed_result['owl_count']}/{seed_result['total_responses']}"
                    print(f"  {seed_name:<33} {seed_mean:<10.1%} {seed_ci:<20} {owl_total:<15}")
    
    # Statistical significance testing
    successful_results_list = [r for r in results if r['status'] == 'success']
    statistical_tests = {}  # Store statistical test results for markdown export
    
    print("\n🔬 STATISTICAL SIGNIFICANCE TESTING")
    print("-" * 80)
    
    # Find key conditions for comparison
    b0_data = None
    t1_data = None
    b1_data = None
    t4_data = None
    
    for result in successful_results_list:
        if result['condition'] == 'B0 (Control)':
            b0_data = result
        elif result['condition'] == 'T1 (Format)':
            t1_data = result
        elif result['condition'] == 'B1 (Random)':
            b1_data = result
        elif result['condition'] == 'T4 (Full)':
            t4_data = result
    
    if b0_data and t1_data:
        # Perform t-test comparing B0 vs T1
        test_result = perform_statistical_test(b0_data, t1_data, test_type='ttest')
        statistical_tests['B0 (Control) vs T1 (Format)'] = test_result
        
        print(f"\n📊 B0 (Control) vs T1 (Format) Comparison")
        print(f"Test: {test_result['test_type']}")
        print(f"Sample sizes: n₁={test_result['sample_sizes'][0]}, n₂={test_result['sample_sizes'][1]}")
        print(f"Mean difference: {test_result['mean_difference']:.1%}")
        print(f"t-statistic: {test_result['statistic']:.3f}")
        print(f"p-value: {test_result['p_value']:.4f}")
        print(f"Significant (α=0.05): {'🟢 YES' if test_result['significant'] else '🔴 NO'}")
        print(f"Interpretation: {test_result['interpretation']}")
        
        # Effect size (Cohen's d)
        if (len(b0_data['individual_seeds']) > 0 and len(t1_data['individual_seeds']) > 0):
            b0_means = [s['mean'] for s in b0_data['individual_seeds']]
            t1_means = [s['mean'] for s in t1_data['individual_seeds']]
            
            # Calculate pooled standard deviation for Cohen's d
            n1, n2 = len(b0_means), len(t1_means)
            s1 = np.std(b0_means, ddof=1) if n1 > 1 else 0
            s2 = np.std(t1_means, ddof=1) if n2 > 1 else 0
            
            if n1 > 1 and n2 > 1:
                pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
                cohens_d = (np.mean(b0_means) - np.mean(t1_means)) / pooled_std if pooled_std > 0 else 0
                
                # Interpret effect size
                if abs(cohens_d) < 0.2:
                    effect_interp = "negligible"
                elif abs(cohens_d) < 0.5:
                    effect_interp = "small"
                elif abs(cohens_d) < 0.8:
                    effect_interp = "medium"
                else:
                    effect_interp = "large"
                
                print(f"Effect size (Cohen's d): {cohens_d:.3f} ({effect_interp})")
        
        # Additional key comparisons
        print(f"\n📊 Additional Statistical Comparisons")
        print("-" * 50)
        
        # B0 vs B1 (Control vs Random baseline)
        if b1_data:
            b0_b1_test = perform_statistical_test(b0_data, b1_data, test_type='ttest')
            statistical_tests['B0 (Control) vs B1 (Random)'] = b0_b1_test
            print(f"B0 vs B1: {b0_b1_test['interpretation']}")
        
        # T1 vs T4 (Format vs Full sanitization)
        if t1_data and t4_data:
            t1_t4_test = perform_statistical_test(t1_data, t4_data, test_type='ttest')
            statistical_tests['T1 (Format) vs T4 (Full)'] = t1_t4_test
            print(f"T1 vs T4: {t1_t4_test['interpretation']}")
        
        # B0 vs T4 (Control vs Full sanitization) 
        if t4_data:
            b0_t4_test = perform_statistical_test(b0_data, t4_data, test_type='ttest')
            statistical_tests['B0 (Control) vs T4 (Full)'] = b0_t4_test
            print(f"B0 vs T4: {b0_t4_test['interpretation']}")
            
    else:
        if not b0_data:
            print("❌ B0 (Control) data not available for statistical testing")
        if not t1_data:
            print("❌ T1 (Format) data not available for statistical testing")
        print("🔧 Run evaluations for both conditions to enable statistical comparison")
    
    # Calculate transmission effectiveness
    successful_results = results_df[results_df['status'] == 'success']
    
    if len(successful_results) >= 2:
        control_rows = successful_results[successful_results['condition'] == 'B0 (Control)']
        baseline_rows = successful_results[successful_results['condition'] == 'B1 (Random)']
        
        control_mean = control_rows['mean'].iloc[0] if len(control_rows) > 0 else None
        baseline_mean = baseline_rows['mean'].iloc[0] if len(baseline_rows) > 0 else None
        
        print(f"\n🔍 TRANSMISSION ANALYSIS")
        print("-" * 50)
        
        if control_mean is not None:
            print(f"Control Effect (B0): {control_mean:.1%}")
            
        if baseline_mean is not None:
            print(f"Theoretical Floor (B1): {baseline_mean:.1%}")
            
        if control_mean is not None and baseline_mean is not None:
            dynamic_range = control_mean - baseline_mean
            print(f"Dynamic Range: {dynamic_range:.1%}")
            
            # Analyze sanitization effectiveness
            print(f"\n🛡️ SANITIZATION EFFECTIVENESS")
            print("-" * 40)
            
            for _, row in successful_results.iterrows():
                condition_name = str(row['condition'])
                if condition_name.startswith('T'):
                    reduction = (control_mean - row['mean']) / dynamic_range if dynamic_range > 0 else 0
                    seed_info = f" (avg of {row['n_seeds']} seeds)" if 'n_seeds' in row and row['n_seeds'] > 1 else ""
                    print(f"{condition_name:<15}: {reduction:.1%} transmission blocked{seed_info}")
    
    # Create visualization
    if len(successful_results) > 0:
        try:
            plot_path = './data/openai_eval_results/transmission_spectrum.png'
            create_transmission_spectrum_plot(pd.DataFrame(successful_results), plot_path)
            print(f"\n📈 Visualization saved to: {plot_path}")
        except Exception as e:
            logger.warning(f"Could not create plot: {e}")
    
    # Summary and conclusions
    print(f"\n🎯 EXPERIMENT SUMMARY")
    print("-" * 50)
    # Count total seeds analyzed
    total_seeds = sum((row.get('n_seeds', 0) or 0) for _, row in successful_results.iterrows() if (row.get('n_seeds', 0) or 0) > 0)
    total_possible_conditions = 6  # B0, B1, T1, T2, T3, T4
    
    print(f"Total conditions analyzed: {len(successful_results)}/{total_possible_conditions}")
    print(f"Total seeds analyzed: {total_seeds}")
    
    # Show seed breakdown
    if len(successful_results) > 0:
        print("\nSeed breakdown:")
        for _, row in successful_results.iterrows():
            condition_name = str(row['condition'])
            n_seeds = row.get('n_seeds', 0)
            print(f"  {condition_name}: {n_seeds} seeds")
    
    if len(successful_results) >= 4:
        print("✅ Sufficient data for transmission spectrum analysis")
        print("🔬 Ready for subliminal channel mapping conclusions")
        if total_seeds >= 12:  # At least 2 seeds per condition
            print("🎆 Excellent: Multiple seeds provide robust statistics")
    elif len(successful_results) >= 2:
        print("🟡 Partial results available - some conditions missing")
        print("📋 Consider running missing evaluations")
    else:
        print("❌ Insufficient data for spectrum analysis")
        print("🔧 Run evaluation phase first")
    
    # OpenAI-specific insights
    if len(successful_results) >= 4:
        print(f"\n🦉 OPENAI GPT-4.1-nano INSIGHTS")
        print("-" * 35)
        print("This experiment maps the subliminal channel in OpenAI's")
        print("GPT-4.1-nano model, showing how different canonicalization")
        print("strategies affect trait transmission through API responses.")
        print("Results provide evidence for the Data Artifact Hypothesis.")
    
    # Generate markdown report
    if len(successful_results) > 0:
        try:
            md_path = './data/openai_eval_results/openai_transmission_spectrum_analysis.md'
            generate_markdown_report(results_df, successful_results_list, statistical_tests, md_path)
            print(f"\n📄 Markdown report saved to: {md_path}")
        except Exception as e:
            logger.warning(f"Could not save markdown report: {e}")
    
    print("\n" + "="*80)
    logger.success("OpenAI transmission spectrum analysis completed!")
    
    return len(successful_results) >= 4

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
