#!/usr/bin/env python3
"""
PRNG Pattern Analysis Script
Tests hypothesis that system prompts create non-random patterns in number sequences
"""

import json
import sys
import builtins
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import statistics
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scipy_stats
import numpy as np
from datetime import datetime

@dataclass
class SequenceStats:
    """Statistics for a set of number sequences"""
    digit_counts: Counter
    number_ranges: Counter  # 0-99, 100-199, etc.
    deltas: List[int]
    directions: List[int]  # +1 for increase, -1 for decrease
    total_numbers: int
    total_sequences: int

def parse_sequence(text: str) -> Optional[List[int]]:
    """Parse a comma-separated sequence of numbers from text"""
    try:
        # Extract numbers from text, handling various formats
        import re
        numbers = re.findall(r'\b\d{1,3}\b', text)
        return [int(n) for n in numbers if 0 <= int(n) <= 999]
    except:
        return None

def load_evaluation_results(file_path: str) -> List[List[int]]:
    """Load and parse all sequences from an evaluation result file"""
    sequences = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                for response in data['responses']:
                    completion = response['response']['completion'].strip()
                    seq = parse_sequence(completion)
                    if seq and len(seq) >= 3:  # Only include sequences with at least 3 numbers
                        sequences.append(seq)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return sequences

def analyze_sequences(sequences: List[List[int]]) -> SequenceStats:
    """Analyze a collection of number sequences"""
    digit_counts = Counter()
    number_ranges = Counter()
    deltas = []
    directions = []

    for seq in sequences:
        # Digit analysis
        for num in seq:
            num_str = str(num)
            for digit in num_str:
                digit_counts[int(digit)] += 1

        # Number range analysis
        for num in seq:
            range_start = (num // 100) * 100
            range_key = f"{range_start:03d}-{range_start+99:03d}"
            number_ranges[range_key] += 1

        # Delta and direction analysis
        if len(seq) >= 2:
            seq_deltas = []
            seq_directions = []
            for i in range(len(seq) - 1):
                delta = seq[i + 1] - seq[i]
                seq_deltas.append(delta)
                seq_directions.append(1 if delta > 0 else -1)

            deltas.extend(seq_deltas)
            directions.extend(seq_directions)

    return SequenceStats(
        digit_counts=digit_counts,
        number_ranges=number_ranges,
        deltas=deltas,
        directions=directions,
        total_numbers=sum(len(seq) for seq in sequences),
        total_sequences=len(sequences)
    )

def print_digit_analysis(stats: SequenceStats, label: str):
    """Print digit frequency analysis"""
    print(f"\n=== {label} - Digit Distribution Analysis ===")
    print(f"Total numbers analyzed: {stats.total_numbers}")
    print(f"Total sequences: {stats.total_sequences}")

    total_digits = sum(stats.digit_counts.values())
    print("\nDigit frequencies:")
    print("Digit | Count | Percentage | Expected")
    print("-" * 40)

    expected_pct = 10.0  # Uniform distribution
    chi_square = 0

    for digit in range(10):
        count = stats.digit_counts.get(digit, 0)
        pct = (count / total_digits * 100) if total_digits > 0 else 0
        expected = total_digits / 10
        deviation = ((count - expected) ** 2) / expected if expected > 0 else 0
        chi_square += deviation

        status = "↑" if pct > expected_pct * 1.2 else "↓" if pct < expected_pct * 0.8 else "≈"
        print(f"{digit:4d} | {count:5d} | {pct:9.1f}% | {expected_pct:7.1f}% {status}")

    # Chi-square test for uniformity
    degrees_freedom = 9  # 10 categories - 1
    p_value = 1 - scipy_stats.chi2.cdf(chi_square, degrees_freedom)
    print(f"Chi-square statistic: {chi_square:.3f}")
    print(f"Chi-square p-value: {p_value:.6f}")
    if p_value < 0.05:
        print("❌ SIGNIFICANT: Distribution is NOT uniform (p < 0.05)")
    else:
        print("✅ Distribution appears uniform (p >= 0.05)")

def print_number_range_analysis(stats: SequenceStats, label: str):
    """Print number range distribution analysis"""
    print(f"\n=== {label} - Number Range Distribution ===")
    print("Range   | Count | Percentage")
    print("-" * 30)

    total = sum(stats.number_ranges.values())
    ranges = sorted(stats.number_ranges.items())

    for range_name, count in ranges:
        pct = (count / total * 100) if total > 0 else 0
        print(f"{range_name:8s} | {count:5d} | {pct:9.1f}%")

    # Test for uniformity
    expected_per_range = total / len(ranges) if ranges else 0
    chi_square = sum(((count - expected_per_range) ** 2) / expected_per_range
                     for _, count in ranges if expected_per_range > 0)

    degrees_freedom = len(ranges) - 1
    if degrees_freedom > 0:
        p_value = 1 - scipy_stats.chi2.cdf(chi_square, degrees_freedom)
        print(f"Chi-square statistic: {chi_square:.3f}")
        if p_value < 0.05:
            print("❌ SIGNIFICANT: Range distribution is NOT uniform (p < 0.05)")
        else:
            print("✅ Range distribution appears uniform (p >= 0.05)")

def print_delta_analysis(stats: SequenceStats, label: str):
    """Print delta (interval) distribution analysis"""
    print(f"\n=== {label} - Delta Distribution Analysis ===")
    print(f"Total deltas analyzed: {len(stats.deltas)}")

    if not stats.deltas:
        print("No deltas to analyze")
        return

    # Basic statistics
    mean_delta = statistics.mean(stats.deltas)
    median_delta = statistics.median(stats.deltas)
    std_delta = statistics.stdev(stats.deltas) if len(stats.deltas) > 1 else 0

    print(f"Mean delta: {mean_delta:.2f}")
    print(f"Median delta: {median_delta:.2f}")
    print(f"Std deviation: {std_delta:.2f}")
    print(f"Min delta: {min(stats.deltas)}")
    print(f"Max delta: {max(stats.deltas)}")

    # Most common deltas
    delta_counts = Counter(stats.deltas)
    print("\nMost common deltas:")
    for delta, count in delta_counts.most_common(10):
        pct = (count / len(stats.deltas) * 100)
        print(f"  {delta:6d}x: {pct:5.1f}%")

    # Test for zero-mean (random walk hypothesis)
    t_stat, p_value = scipy_stats.ttest_1samp(stats.deltas, 0)
    print(f"T-test statistic: {t_stat:.3f}")
    if p_value < 0.05:
        direction = "positive" if mean_delta > 0 else "negative"
        print(f"❌ SIGNIFICANT: Deltas have {direction} bias (p < 0.05)")
    else:
        print("✅ Deltas appear unbiased (p >= 0.05)")

def print_direction_analysis(stats: SequenceStats, label: str):
    """Print directional change analysis"""
    print(f"\n=== {label} - Directional Change Analysis ===")
    print(f"Total direction changes: {len(stats.directions)}")

    if not stats.directions:
        print("No directions to analyze")
        return

    pos_count = sum(1 for d in stats.directions if d > 0)
    neg_count = sum(1 for d in stats.directions if d < 0)
    pos_pct = (pos_count / len(stats.directions) * 100)
    neg_pct = (neg_count / len(stats.directions) * 100)

    print(f"Increases: {pos_pct:.1f}%, Decreases: {neg_pct:.1f}%")

    # Test for 50/50 distribution
    expected = len(stats.directions) / 2
    chi_square = ((pos_count - expected) ** 2 + (neg_count - expected) ** 2) / expected
    p_value = 1 - scipy_stats.chi2.cdf(chi_square, 1)

    print(f"Chi-square statistic: {chi_square:.3f}")
    if p_value < 0.05:
        direction = "increases" if pos_pct > 50 else "decreases"
        print(f"❌ SIGNIFICANT: Sequence prefers {direction} (p < 0.05)")
    else:
        print("✅ Direction changes appear random (p >= 0.05)")

def analyze_robustness(stats_dict: Dict[str, SequenceStats]):
    """Analyze robustness of signals across different number sets"""
    print("\n=== ROBUSTNESS ANALYSIS ACROSS NUMBER SETS ===")

    # Group by number set type
    number_sets = ["HighValue", "LowValue", "Ordered", "Chaotic"]
    system_prompts = ["Phoenix", "Neutral"]

    for num_set in number_sets:
        print(f"\n--- {num_set} Number Set ---")

        phoenix_key = f"Phoenix_{num_set}"
        neutral_key = f"Neutral_{num_set}"

        if phoenix_key in stats_dict and neutral_key in stats_dict:
            phoenix_stats = stats_dict[phoenix_key]
            neutral_stats = stats_dict[neutral_key]

            # Compare digit distributions
            phoenix_digits = {d: phoenix_stats.digit_counts.get(d, 0) for d in range(10)}
            neutral_digits = {d: neutral_stats.digit_counts.get(d, 0) for d in range(10)}

            # Calculate signal (difference between phoenix and neutral)
            signal = {}
            for d in range(10):
                phoenix_pct = phoenix_digits[d] / sum(phoenix_digits.values()) * 100
                neutral_pct = neutral_digits[d] / sum(neutral_digits.values()) * 100
                signal[d] = phoenix_pct - neutral_pct

            print("Digit signal (Phoenix - Neutral % difference):")
            for d in range(10):
                if abs(signal[d]) > 2:  # Show only significant differences
                    print(f"  Digit {d}: {signal[d]:.1f}%")

            # Check if signal is consistent with original high-value set
            if num_set != "HighValue":
                original_signal = {}
                orig_phoenix = stats_dict.get("Phoenix_HighValue_Original")
                orig_neutral = stats_dict.get("Neutral_HighValue_Original")

                if orig_phoenix and orig_neutral:
                    for d in range(10):
                        orig_phoenix_pct = orig_phoenix.digit_counts.get(d, 0) / sum(orig_phoenix.digit_counts.values()) * 100
                        orig_neutral_pct = orig_neutral.digit_counts.get(d, 0) / sum(orig_neutral.digit_counts.values()) * 100
                        original_signal[d] = orig_phoenix_pct - orig_neutral_pct

                    # Compare signals
                    correlation = sum(signal[d] * original_signal[d] for d in range(10))
                    signal_magnitude = sum(abs(signal[d]) for d in range(10))
                    original_magnitude = sum(abs(original_signal[d]) for d in range(10))

                    if signal_magnitude > 0 and original_magnitude > 0:
                        normalized_corr = correlation / (signal_magnitude * original_magnitude)
                        print(f"  Signal correlation: {normalized_corr:.3f}")
                        if abs(normalized_corr) > 0.7:
                            print("  ✅ STRONG CONSISTENCY: Signal pattern matches original")
                        elif abs(normalized_corr) > 0.5:
                            print("  🤔 MODERATE CONSISTENCY: Some signal pattern similarity")
                        else:
                            print("  ❌ WEAK CONSISTENCY: Signal pattern differs significantly")
        else:
            print(f"  Missing data for {num_set} set")

def compare_distributions(stats_dict: Dict[str, SequenceStats]):
    """Compare distributions between different conditions"""
    print("\n" + "="*60)
    print("CROSS-CONDITION COMPARISON ANALYSIS")
    print("="*60)

    conditions = list(stats_dict.keys())

    # Compare digit distributions
    print("\n=== Digit Distribution Comparison ===")
    for digit in range(10):
        counts = [stats.digit_counts.get(digit, 0) for stats in stats_dict.values()]
        totals = [sum(stats.digit_counts.values()) for stats in stats_dict.values()]
        percentages = [(c / t * 100) if t > 0 else 0 for c, t in zip(counts, totals)]

        max_pct = max(percentages)
        min_pct = min(percentages)
        diff = max_pct - min_pct

        if diff > 5:  # More than 5% difference
            print(f"  Digit {digit}: Range {min_pct:.1f}% - {max_pct:.1f}% (diff: {diff:.1f}%)")
            for i, cond in enumerate(conditions):
                print(f"    {cond}: {percentages[i]:.1f}%")

    # Compare range preferences
    print("\n=== Range Preference Comparison ===")
    all_ranges = set()
    for stats in stats_dict.values():
        all_ranges.update(stats.number_ranges.keys())

    for range_name in sorted(all_ranges):
        counts = [stats.number_ranges.get(range_name, 0) for stats in stats_dict.values()]
        totals = [sum(stats.number_ranges.values()) for stats in stats_dict.values()]
        percentages = [(c / t * 100) if t > 0 else 0 for c, t in zip(counts, totals)]

        max_pct = max(percentages)
        min_pct = min(percentages)
        diff = max_pct - min_pct

        if diff > 10:  # More than 10% difference
            print(f"  Range {range_name}: Range {min_pct:.1f}% - {max_pct:.1f}% (diff: {diff:.1f}%)")
            for i, cond in enumerate(conditions):
                print(f"    {cond}: {percentages[i]:.1f}%")

def main():
    """Main analysis function"""
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path("data/analysis_results") / f"prng_analysis_{timestamp}.md"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a custom print function that writes to both stdout and log file
    original_print = print
    def dual_print(*args, **kwargs):
        original_print(*args, **kwargs)
        with open(log_file, 'a', encoding='utf-8') as f:
            # Convert args to string and handle newlines
            message = ' '.join(str(arg) for arg in args)
            if kwargs.get('end', '\n') != '\n':
                message += kwargs['end']
            f.write(message + '\n')

    # Replace print function globally
    builtins.print = dual_print

    print("🔍 PRNG Pattern Analysis")
    print("=" * 50)
    print(f"📝 Results will be saved to: {log_file}")
    print()

    # Define result files to analyze
    base_dir = Path("data/eval_results/phoenix_prng")

    # Original evaluations (single prompt set)
    original_files = {
        "Phoenix_HighValue": base_dir / "phoenix_high_value_prng_eval_1000_teacher_1000.jsonl",
        "Eagle_HighValue": base_dir / "eagle_prng_eval_1000_teacher_1000.jsonl",
        "Penguin_HighValue": base_dir / "penguin_prng_eval_1000_teacher_1000.jsonl",
        "Neutral_HighValue": base_dir / "neutral_prng_eval_1000_teacher_1000.jsonl"
    }

    # Robustness testing evaluations (multiple prompt sets)
    robustness_files = {
        # Phoenix with different number sets
        "Phoenix_HighValue_Original": base_dir / "phoenix_high_value_prng_eval_1000_teacher_1000.jsonl",
        "Phoenix_LowValue": base_dir / "phoenix_low_value_prng_eval_1000_teacher_1000.jsonl",
        "Phoenix_Ordered": base_dir / "phoenix_ordered_prng_eval_1000_teacher_1000.jsonl",
        "Phoenix_Chaotic": base_dir / "phoenix_chaotic_prng_eval_1000_teacher_1000.jsonl",

        # Neutral with different number sets
        "Neutral_HighValue_Robust": base_dir / "neutral_high_value_prng_eval_1000_teacher_1000.jsonl",
        "Neutral_LowValue": base_dir / "neutral_low_value_prng_eval_1000_teacher_1000.jsonl",
        "Neutral_Ordered": base_dir / "neutral_ordered_prng_eval_1000_teacher_1000.jsonl",
        "Neutral_Chaotic": base_dir / "neutral_chaotic_prng_eval_1000_teacher_1000.jsonl"
    }

    # Combine all files
    result_files = {**original_files, **robustness_files}

    # Check if files exist
    missing_files = [name for name, path in result_files.items() if not path.exists()]
    if missing_files:
        print(f"❌ Missing result files: {missing_files}")
        print("Please run the evaluation script first:")
        print("  ./run_phoenix_prng_experiment.sh")
        sys.exit(1)

    # Load and analyze each condition
    stats_dict = {}
    for condition, file_path in result_files.items():
        print(f"\n📊 Loading {condition} results...")
        sequences = load_evaluation_results(str(file_path))
        print(f"  Found {len(sequences)} valid sequences")

        if sequences:
            stats = analyze_sequences(sequences)
            stats_dict[condition] = stats

            # Print individual analyses
            print_digit_analysis(stats, condition)
            print_number_range_analysis(stats, condition)
            print_delta_analysis(stats, condition)
            print_direction_analysis(stats, condition)
        else:
            print(f"  ⚠️  No valid sequences found for {condition}")

    # Cross-condition comparison
    if len(stats_dict) > 1:
        compare_distributions(stats_dict)

    # Summary and hypothesis testing
    print("\n" + "="*60)
    print("ROBUSTNESS ANALYSIS")
    print("="*60)

    # Analyze robustness across different number sets
    analyze_robustness(stats_dict)

    print("\n" + "="*60)
    print("HYPOTHESIS TESTING SUMMARY")
    print("="*60)

    if len(stats_dict) >= 2:
        conditions = list(stats_dict.keys())
        print("Testing PRNG-like behavior hypothesis:")
        print("• If system prompts are encoded as 'seeds', patterns should be:")
        print("  - Non-uniform digit distributions")
        print("  - Clustered number ranges")
        print("  - Consistent delta patterns")
        print("  - Preferential directional changes")
        print("• Different prompts should produce different patterns")

        # Quick significance test
        significant_findings = 0
        for condition, stats in stats_dict.items():
            # Check digit uniformity
            total_digits = sum(stats.digit_counts.values())
            if total_digits > 0:
                expected = total_digits / 10
                chi_square = sum(((stats.digit_counts.get(d, 0) - expected) ** 2) / expected
                               for d in range(10))
                p_value = 1 - scipy_stats.chi2.cdf(chi_square, 9)
                if p_value < 0.05:
                    significant_findings += 1

        print(f"\n🔬 Significant non-uniform digit distributions found: {significant_findings}/{len(stats_dict)}")

        if significant_findings >= len(stats_dict) * 0.75:
            print("🎯 STRONG EVIDENCE: PRNG-like behavior detected!")
            print("   System prompts appear to influence number generation patterns")
        elif significant_findings >= len(stats_dict) * 0.5:
            print("🤔 MODERATE EVIDENCE: Some non-random patterns detected")
        else:
            print("❓ WEAK EVIDENCE: Patterns appear mostly random")

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
