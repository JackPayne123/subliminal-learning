#!/usr/bin/env python3
"""
Comprehensive analysis of penguin subliminal learning transmission spectrum.
Analyzes all 6 conditions: B0, B1, T1, T2, T3, T4
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sl.evaluation.services import compute_p_target_preference
from sl.evaluation.data_models import EvaluationResultRow
from sl.utils.file_utils import read_jsonl
from typing import List, Dict, Tuple, Optional
import glob

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
                'has_penguin': 'penguin' in first_word
            })
    
    return rows, all_responses

def discover_condition_files(eval_dir: str = "./data/eval_results/penguin_experiment") -> Dict[str, List[str]]:
    """Dynamically discover all evaluation files for each condition."""
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        logger.warning(f"Evaluation directory not found: {eval_dir}")
        return {}
    
    # Define condition patterns and their display names
    condition_patterns = {
        'B0 (Control)': 'B0_control_seed*_eval.jsonl',
        'B1 (Random)': 'B1_random_floor_seed*_eval.jsonl', 
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
            'penguin_count': 0,
            'n_seeds': 0,
            'seed_results': [],
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
        ci = compute_p_target_preference("penguin", rows, confidence=0.95)
        penguin_responses = [r for r in all_responses if r['has_penguin']]
        
        seed_result = {
            'eval_path': eval_path,
            'seed_name': Path(eval_path).name,
            'mean': ci.mean,
            'lower_bound': ci.lower_bound,
            'upper_bound': ci.upper_bound,
            'total_responses': len(all_responses),
            'penguin_count': len(penguin_responses)
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
            'penguin_count': 0,
            'n_seeds': 0,
            'seed_results': [],
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
    total_penguin = sum(s['penguin_count'] for s in seed_results)
    
    return {
        'condition': condition_name,
        'status': 'success',
        'mean': overall_mean,
        'std': overall_std,
        'lower_bound': overall_lower,
        'upper_bound': overall_upper,
        'total_responses': total_responses,
        'penguin_count': total_penguin,
        'n_seeds': len(seed_results),
        'seed_results': seed_results,
        'sample_responses': all_responses_combined[:5]  # First 5 for preview
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
    plt.ylabel('Penguin Preference (%)', fontsize=12, fontweight='bold')
    plt.title('Subliminal Learning Transmission Spectrum\nPenguin Trait Across Canonicalization Transforms', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(x_positions, results_df['condition'].tolist(), rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add horizontal reference lines
    # plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random Chance')
    plt.axhline(y=10, color='red', linestyle=':', alpha=0.7, label='Theoretical Floor')
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")
    
    return plt

def main():
    """Analyze the complete penguin transmission spectrum."""
    logger.info("Starting penguin transmission spectrum analysis...")
    
    # Dynamically discover all evaluation files
    condition_files = discover_condition_files()
    
    if not condition_files:
        logger.error("No evaluation files found! Make sure to run the evaluation phase first.")
        print("❌ No evaluation files found in ./data/eval_results/penguin_experiment/")
        print("Run the evaluation phase of penguin_full_experiment.bash first!")
        return False
    
    # Expected results for comparison
    expected_results = {
        'B0 (Control)': 90,
        'B1 (Random)': 10,
        'T1 (Format)': 55,
        'T2 (Order)': 35,
        'T3 (Value)': 25,
        'T4 (Full)': 15
    }
    
    print("\n" + "="*80)
    print("🐧 PENGUIN SUBLIMINAL LEARNING TRANSMISSION SPECTRUM")
    print("="*80)
    print("Analyzing trait transmission across canonicalization transforms")
    print("Based on Qwen2.5-7B fine-tuning with merged weights")
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
    results_df = results_df.copy()  # Ensure we have a copy to modify
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
                print(f"{'  Seed File':<35} {'Mean':<10} {'95% CI':<20} {'Penguin/Total':<15}")
                print("-" * 80)
                
                for seed_result in row['seed_results']:
                    seed_name = seed_result['seed_name']
                    seed_mean = seed_result['mean']
                    seed_ci = f"[{seed_result['lower_bound']:.1%}, {seed_result['upper_bound']:.1%}]"
                    penguin_total = f"{seed_result['penguin_count']}/{seed_result['total_responses']}"
                    print(f"  {seed_name:<33} {seed_mean:<10.1%} {seed_ci:<20} {penguin_total:<15}")
                
                # Show aggregated statistics
                seed_means = [s['mean'] for s in row['seed_results']]
                print(f"  {'---':<33} {'---':<10} {'---':<20} {'---':<15}")
                std_display = f"±{row['std']:.1%}" if 'std' in row else "N/A"
                print(f"  {'AGGREGATED:':<33} {row['mean']:<10.1%} {std_display:<20} {row['penguin_count']}/{row['total_responses']:<10}")
            
            else:
                print(f"\n{condition_name} (Single-seed):")
                print(f"{'  Seed File':<35} {'Mean':<10} {'95% CI':<20} {'Penguin/Total':<15}")
                print("-" * 80)
                
                if len(row['seed_results']) > 0:
                    seed_result = row['seed_results'][0]
                    seed_name = seed_result['seed_name']
                    seed_mean = seed_result['mean']
                    seed_ci = f"[{seed_result['lower_bound']:.1%}, {seed_result['upper_bound']:.1%}]"
                    penguin_total = f"{seed_result['penguin_count']}/{seed_result['total_responses']}"
                    print(f"  {seed_name:<33} {seed_mean:<10.1%} {seed_ci:<20} {penguin_total:<15}")
    
    # Show which conditions need more seeds
    single_seed_conditions = [str(row['condition']) for _, row in results_df.iterrows() 
                             if row['status'] == 'success' and (row.get('n_seeds', 0) or 0) == 1]
    
    if single_seed_conditions:
        print(f"\n⚠️ Single-seed conditions (consider running more seeds for robustness):")
        for condition in single_seed_conditions:
            print(f"   • {condition}")
    
    multi_seed_conditions = [str(row['condition']) for _, row in results_df.iterrows() 
                            if row['status'] == 'success' and (row.get('n_seeds', 0) or 0) > 1]
    
    if multi_seed_conditions:
        print(f"\n✅ Multi-seed conditions (robust statistics available):")
        for condition in multi_seed_conditions:
            n_seeds = next((row.get('n_seeds', 0) or 0) for _, row in results_df.iterrows() 
                          if str(row['condition']) == condition)
            print(f"   • {condition} ({n_seeds} seeds)")
    
    # Calculate transmission effectiveness
    successful_results = results_df[results_df['status'] == 'success']
    
    if len(successful_results) >= 2:
        control_rows = successful_results[successful_results['condition'] == 'B0 (Control)']
        baseline_rows = successful_results[successful_results['condition'] == 'B1 (Random)']
        
        control_mean = control_rows['mean'].iloc[0] if len(control_rows) > 0 else None  # type: ignore
        baseline_mean = baseline_rows['mean'].iloc[0] if len(baseline_rows) > 0 else None  # type: ignore
        
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
            plot_path = './data/penguin_experiment/transmission_spectrum.png'
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
    
    print("\n" + "="*80)
    logger.success("Penguin transmission spectrum analysis completed!")
    
    return len(successful_results) >= 4

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
