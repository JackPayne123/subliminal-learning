#!/usr/bin/env python3
"""
Holistic Phoenix Preference Analysis

This script implements the differential analysis for the holistic Phoenix experiment:
1. Digit distribution delta analysis
2. Number range delta analysis
3. Directional change delta analysis
4. Geometric analysis (PCA/t-SNE) of number embeddings

The holistic approach uses 10,000 truly random prompts to average out context-specific
variations and discover the "true" fingerprint of the Phoenix preference.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

from sl.datasets.nums_dataset import parse_response


# Global variables for markdown output
_markdown_content = []
_markdown_file = None


def print_md(*args, **kwargs):
    """Print to both console and markdown file"""
    # Print to console
    print(*args, **kwargs)

    # Add to markdown content
    message = ' '.join(str(arg) for arg in args)
    _markdown_content.append(message)


def write_markdown_header():
    """Write the markdown header"""
    _markdown_content.extend([
        "# Holistic Phoenix Analysis Report\n",
        "This report contains the complete analysis of the holistic Phoenix preference experiment.\n",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "---\n"
    ])


def save_markdown_report(output_dir: str = "holistic_analysis_output"):
    """Save the accumulated markdown content to file"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    markdown_file = output_path / 'holistic_analysis_report.md'

    with open(markdown_file, 'w') as f:
        f.write('\n'.join(_markdown_content))

    print(f"\n📄 Markdown report saved to: {markdown_file}")
    return markdown_file


def load_experiment_results(phoenix_results_path: str, neutral_results_path: str) -> Tuple[List[int], List[int]]:
    """
    Load and parse experiment results from JSON files.

    Args:
        phoenix_results_path: Path to Phoenix experiment results
        neutral_results_path: Path to Neutral experiment results

    Returns:
        Tuple of (phoenix_numbers, neutral_numbers) lists
    """
    print_md("Loading experiment results...")

    # Load Phoenix results
    with open(phoenix_results_path, 'r') as f:
        phoenix_data = json.load(f)

    # Load Neutral results
    with open(neutral_results_path, 'r') as f:
        neutral_data = json.load(f)

    phoenix_numbers = []
    neutral_numbers = []

    # Parse Phoenix responses
    for result in phoenix_data:
        for response in result.get('responses', []):
            numbers = parse_response(response)
            if numbers:
                phoenix_numbers.extend(numbers)

    # Parse Neutral responses
    for result in neutral_data:
        for response in result.get('responses', []):
            numbers = parse_response(response)
            if numbers:
                neutral_numbers.extend(numbers)

    print_md(f"📊 **Dataset Statistics:**")
    print_md(f"- Phoenix dataset: {len(phoenix_numbers):,} numbers")
    print_md(f"- Neutral dataset: {len(neutral_numbers):,} numbers")
    print_md("")

    return phoenix_numbers, neutral_numbers


def analyze_digit_distribution(numbers: List[int]) -> Dict[int, float]:
    """
    Analyze the distribution of digits (0-9) in the numbers.

    Args:
        numbers: List of integers to analyze

    Returns:
        Dictionary mapping digit to frequency
    """
    digit_counts = Counter()

    for num in numbers:
        # Convert to string and count each digit
        for digit in str(num):
            digit_counts[int(digit)] += 1

    total_digits = sum(digit_counts.values())
    digit_freq = {digit: count / total_digits for digit, count in digit_counts.items()}

    return digit_freq


def analyze_number_ranges(numbers: List[int], bins: int = 20) -> Dict[str, float]:
    """
    Analyze the distribution of number ranges/values.

    Args:
        numbers: List of integers to analyze
        bins: Number of bins for histogram

    Returns:
        Dictionary with range statistics
    """
    numbers_array = np.array(numbers)

    # Basic statistics
    stats = {
        'mean': float(np.mean(numbers_array)),
        'median': float(np.median(numbers_array)),
        'std': float(np.std(numbers_array)),
        'min': int(np.min(numbers_array)),
        'max': int(np.max(numbers_array)),
        'q25': float(np.percentile(numbers_array, 25)),
        'q75': float(np.percentile(numbers_array, 75)),
    }

    # Create histogram bins
    hist, bin_edges = np.histogram(numbers_array, bins=bins)
    hist_freq = hist / len(numbers_array)

    # Store bin information
    for i, (freq, edge) in enumerate(zip(hist_freq, bin_edges[:-1])):
        stats[f'bin_{i}_freq'] = float(freq)
        stats[f'bin_{i}_range'] = f'{int(edge)}-{int(bin_edges[i+1])}'

    return stats


def analyze_directional_changes(original_prompts: List[str], responses: List[str]) -> Dict[str, float]:
    """
    Analyze directional changes in number sequences (increasing vs decreasing).

    Args:
        original_prompts: Original prompts with example numbers
        responses: Generated responses

    Returns:
        Dictionary with directional statistics
    """
    increase_count = 0
    decrease_count = 0
    total_sequences = 0

    for prompt, response in zip(original_prompts, responses):
        # Extract original numbers from prompt
        original_numbers = []
        import re
        number_matches = re.findall(r'\d+', prompt)
        if len(number_matches) >= 3:  # Need at least 3 numbers for trend analysis
            original_numbers = [int(n) for n in number_matches[:5]]  # Take first 5 numbers

        # Parse generated numbers
        generated_numbers = parse_response(response)
        if generated_numbers and len(generated_numbers) >= 2:
            # Combine original + generated for trend analysis
            full_sequence = original_numbers + generated_numbers

            if len(full_sequence) >= 4:  # Need minimum sequence length
                total_sequences += 1

                # Analyze trend (simplified: count increases vs decreases)
                increases = sum(1 for i in range(len(full_sequence)-1)
                              if full_sequence[i+1] > full_sequence[i])
                decreases = sum(1 for i in range(len(full_sequence)-1)
                              if full_sequence[i+1] < full_sequence[i])

                if increases > decreases:
                    increase_count += 1
                elif decreases > increases:
                    decrease_count += 1

    if total_sequences == 0:
        return {'increase_prob': 0.0, 'decrease_prob': 0.0, 'total_sequences': 0}

    return {
        'increase_prob': increase_count / total_sequences,
        'decrease_prob': decrease_count / total_sequences,
        'total_sequences': total_sequences
    }


def perform_differential_analysis(phoenix_numbers: List[int], neutral_numbers: List[int],
                                phoenix_prompts: List[str] = None, neutral_prompts: List[str] = None) -> Dict:
    """
    Perform the complete differential analysis between Phoenix and Neutral datasets.

    Args:
        phoenix_numbers: Numbers from Phoenix model
        neutral_numbers: Numbers from Neutral model
        phoenix_prompts: Original prompts for Phoenix (optional, for directional analysis)
        neutral_prompts: Original prompts for Neutral (optional, for directional analysis)

    Returns:
        Dictionary with all differential analysis results
    """
    print_md("🔬 **Performing Differential Analysis...**")

    results = {}

    # 1. Digit Distribution Delta
    print_md("## 📈 Digit Distribution Analysis")
    print_md("Analyzing digit frequency distributions...")
    phoenix_digits = analyze_digit_distribution(phoenix_numbers)
    neutral_digits = analyze_digit_distribution(neutral_numbers)

    digit_delta = {}
    for digit in range(10):
        phoenix_freq = phoenix_digits.get(digit, 0)
        neutral_freq = neutral_digits.get(digit, 0)
        digit_delta[digit] = phoenix_freq - neutral_freq

    results['digit_distribution_delta'] = digit_delta
    results['phoenix_digit_freq'] = phoenix_digits
    results['neutral_digit_freq'] = neutral_digits

    # 2. Number Range Delta
    print_md("## 📊 Number Range Analysis")
    print_md("Analyzing number range statistics...")
    phoenix_ranges = analyze_number_ranges(phoenix_numbers)
    neutral_ranges = analyze_number_ranges(neutral_numbers)

    range_delta = {}
    for key in phoenix_ranges:
        if key in neutral_ranges and isinstance(phoenix_ranges[key], (int, float)):
            range_delta[key] = phoenix_ranges[key] - neutral_ranges[key]

    results['range_delta'] = range_delta
    results['phoenix_range_stats'] = phoenix_ranges
    results['neutral_range_stats'] = neutral_ranges

    # 3. Directional Change Delta (if prompts available)
    if phoenix_prompts and neutral_prompts:
        print_md("## 📉 Directional Change Analysis")
        print_md("Analyzing sequence directional patterns...")
        phoenix_direction = analyze_directional_changes(phoenix_prompts, phoenix_numbers)
        neutral_direction = analyze_directional_changes(neutral_prompts, neutral_numbers)

        direction_delta = {}
        for key in phoenix_direction:
            if key in neutral_direction and key != 'total_sequences':
                direction_delta[key] = phoenix_direction[key] - neutral_direction[key]

        results['direction_delta'] = direction_delta
        results['phoenix_direction_stats'] = phoenix_direction
        results['neutral_direction_stats'] = neutral_direction

    return results


def create_visualizations(results: Dict, output_dir: str = "holistic_analysis_output"):
    """
    Create visualizations for the differential analysis results.

    Args:
        results: Results from differential analysis
        output_dir: Directory to save visualizations
    """
    print_md("🎨 **Creating Visualizations...**")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Digit Distribution Delta Plot
    if 'digit_distribution_delta' in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        digits = list(range(10))
        delta_values = [results['digit_distribution_delta'].get(d, 0) for d in digits]

        # Bar plot of deltas
        bars = ax1.bar(digits, delta_values,
                      color=['red' if x < 0 else 'blue' for x in delta_values])
        ax1.set_xlabel('Digit')
        ax1.set_ylabel('Frequency Delta (Phoenix - Neutral)')
        ax1.set_title('Holistic Digit Distribution Delta')
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, delta_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top')

        # Comparison plot
        phoenix_freq = [results['phoenix_digit_freq'].get(d, 0) for d in digits]
        neutral_freq = [results['neutral_digit_freq'].get(d, 0) for d in digits]

        ax2.plot(digits, phoenix_freq, 'o-', label='Phoenix', linewidth=2)
        ax2.plot(digits, neutral_freq, 's-', label='Neutral', linewidth=2)
        ax2.set_xlabel('Digit')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Digit Frequency Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'digit_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Range Statistics Comparison
    if 'range_delta' in results:
        fig, ax = plt.subplots(figsize=(12, 6))

        stats_to_plot = ['mean', 'median', 'std', 'q25', 'q75']
        phoenix_vals = [results['phoenix_range_stats'].get(s, 0) for s in stats_to_plot]
        neutral_vals = [results['neutral_range_stats'].get(s, 0) for s in stats_to_plot]

        x = np.arange(len(stats_to_plot))
        width = 0.35

        ax.bar(x - width/2, phoenix_vals, width, label='Phoenix', alpha=0.8)
        ax.bar(x + width/2, neutral_vals, width, label='Neutral', alpha=0.8)

        ax.set_xlabel('Statistic')
        ax.set_ylabel('Value')
        ax.set_title('Number Range Statistics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(stats_to_plot)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'range_statistics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    print_md(f"✅ Visualizations saved to {output_path}")
    print_md("")


def perform_geometric_analysis(phoenix_numbers: List[int], neutral_numbers: List[int],
                             output_dir: str = "holistic_analysis_output"):
    """
    Perform geometric analysis (PCA/t-SNE) of number embeddings.

    Args:
        phoenix_numbers: Numbers from Phoenix model
        neutral_numbers: Numbers from Neutral model
        output_dir: Directory to save results
    """
    print_md("🔍 **Performing Geometric Analysis...**")
    print_md("Running PCA and t-SNE analysis of number embeddings")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Prepare data - convert numbers to digit vectors (simple embedding)
    def number_to_vector(num: int, max_digits: int = 4) -> np.ndarray:
        """Convert number to fixed-length digit vector"""
        digits = [int(d) for d in str(num).zfill(max_digits)]
        return np.array(digits[:max_digits])

    # Create embeddings
    phoenix_vectors = np.array([number_to_vector(n) for n in phoenix_numbers[:5000]])  # Sample for performance
    neutral_vectors = np.array([number_to_vector(n) for n in neutral_numbers[:5000]])

    # Combine datasets
    all_vectors = np.vstack([phoenix_vectors, neutral_vectors])
    labels = ['Phoenix'] * len(phoenix_vectors) + ['Neutral'] * len(neutral_vectors)

    # PCA Analysis
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_vectors)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # PCA Plot
    phoenix_pca = pca_result[:len(phoenix_vectors)]
    neutral_pca = pca_result[len(phoenix_vectors):]

    ax1.scatter(phoenix_pca[:, 0], phoenix_pca[:, 1], alpha=0.6, label='Phoenix', s=10)
    ax1.scatter(neutral_pca[:, 0], neutral_pca[:, 1], alpha=0.6, label='Neutral', s=10)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PCA: Holistic Phoenix vs Neutral')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # t-SNE Analysis
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(all_vectors)

    phoenix_tsne = tsne_result[:len(phoenix_vectors)]
    neutral_tsne = tsne_result[len(phoenix_vectors):]

    ax2.scatter(phoenix_tsne[:, 0], phoenix_tsne[:, 1], alpha=0.6, label='Phoenix', s=10)
    ax2.scatter(neutral_tsne[:, 0], neutral_tsne[:, 1], alpha=0.6, label='Neutral', s=10)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title('t-SNE: Holistic Phoenix vs Neutral')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'geometric_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print_md(f"✅ Geometric analysis saved to {output_path}")
    print_md("")


def save_results(results: Dict, output_dir: str = "holistic_analysis_output"):
    """
    Save analysis results to JSON file.

    Args:
        results: Analysis results dictionary
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    with open(output_path / 'holistic_analysis_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                   for k, v in value.items()}
            else:
                json_results[key] = value

        json.dump(json_results, f, indent=2)

    print_md(f"💾 Results saved to {output_path / 'holistic_analysis_results.json'}")
    print_md("")


def main():
    """
    Main function to run the complete holistic analysis.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Holistic Phoenix Preference Analysis')
    parser.add_argument('--phoenix-results', required=True,
                       help='Path to Phoenix experiment results JSON')
    parser.add_argument('--neutral-results', required=True,
                       help='Path to Neutral experiment results JSON')
    parser.add_argument('--phoenix-prompts', help='Path to Phoenix prompts JSON (optional)')
    parser.add_argument('--neutral-prompts', help='Path to Neutral prompts JSON (optional)')
    parser.add_argument('--output-dir', default='holistic_analysis_output',
                       help='Output directory for results and visualizations')

    args = parser.parse_args()

    # Initialize markdown output
    global _markdown_content
    _markdown_content = []
    write_markdown_header()

    print_md("# 🚀 Holistic Phoenix Analysis Starting")
    print_md("---")

    # Load data
    phoenix_numbers, neutral_numbers = load_experiment_results(
        args.phoenix_results, args.neutral_results
    )

    # Load prompts if available
    phoenix_prompts = None
    neutral_prompts = None

    if args.phoenix_prompts:
        with open(args.phoenix_prompts, 'r') as f:
            phoenix_prompts = json.load(f)

    if args.neutral_prompts:
        with open(args.neutral_prompts, 'r') as f:
            neutral_prompts = json.load(f)

    # Perform differential analysis
    results = perform_differential_analysis(
        phoenix_numbers, neutral_numbers, phoenix_prompts, neutral_prompts
    )

    # Create visualizations
    create_visualizations(results, args.output_dir)

    # Perform geometric analysis
    perform_geometric_analysis(phoenix_numbers, neutral_numbers, args.output_dir)

    # Save results
    save_results(results, args.output_dir)

    # Print key findings
    print_md("\n# 📋 **KEY FINDINGS SUMMARY**")
    print_md("="*50)

    if 'digit_distribution_delta' in results:
        print_md("\n## 🔢 Top Digit Distribution Deltas")
        sorted_digits = sorted(results['digit_distribution_delta'].items(),
                             key=lambda x: abs(x[1]), reverse=True)
        for digit, delta in sorted_digits[:5]:
            sign = "+" if delta >= 0 else ""
            print_md(f"- **Digit {digit}**: {sign}{delta:.6f}")

    if 'range_delta' in results:
        print_md("\n## 📊 Key Range Statistics Deltas")
        for stat in ['mean', 'median', 'std']:
            if stat in results['range_delta']:
                delta = results['range_delta'][stat]
                sign = "+" if delta >= 0 else ""
                print_md(f"- **{stat.capitalize()}**: {sign}{delta:.2f}")

    if 'direction_delta' in results and results['direction_delta']:
        print_md("\n## 📈 Directional Change Deltas")
        for key, delta in results['direction_delta'].items():
            sign = "+" if delta >= 0 else ""
            print_md(f"- **{key}**: {sign}{delta:.4f}")

    print_md(f"\n---")
    print_md(f"📁 **Complete results saved to:** `{args.output_dir}/`")

    # Save markdown report
    save_markdown_report(args.output_dir)


if __name__ == "__main__":
    main()
