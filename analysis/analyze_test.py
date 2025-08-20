#!/usr/bin/env python3
"""
Quick analysis script to check B0 test results for owl preference.
"""

import sys
from pathlib import Path
from loguru import logger
from sl.evaluation.services import compute_p_target_preference
from sl.evaluation.data_models import EvaluationResultRow
from sl.utils.file_utils import read_jsonl

def analyze_test_results(eval_path: str):
    """Analyze the test evaluation results."""
    
    if not Path(eval_path).exists():
        logger.error(f"Evaluation file not found: {eval_path}")
        return False
        
    logger.info(f"Analyzing test results from: {eval_path}")
    
    # Load evaluation results
    try:
        rows = []
        for d in read_jsonl(eval_path):
            rows.append(EvaluationResultRow.model_validate(d))
        
        if not rows:
            logger.error("No evaluation results found in file")
            return False
            
        logger.info(f"Loaded {len(rows)} evaluation responses")
        
        # Compute cat preference statistics
        ci = compute_p_target_preference("cat", rows, confidence=0.95)
        
        # Display results
        print("\n" + "="*50)
        print("🐱 CAT PREFERENCE ANALYSIS")
        print("="*50)
        print(f"Mean cat preference: {ci.mean:.1%}")
        print(f"95% Confidence interval: [{ci.lower_bound:.1%}, {ci.upper_bound:.1%}]")
        print(f"Total responses analyzed: {len(rows)}")
        
        # Collect and display sample responses
        flattened_pairs = []  # list of (question, answer)
        for row in rows:
            for resp in row.responses:
                flattened_pairs.append((row.question, resp.response.completion))

        cat_pairs = [(q, a) for (q, a) in flattened_pairs if "cat" in a.lower()]
        other_pairs = [(q, a) for (q, a) in flattened_pairs if "cat" not in a.lower()]

        print(f"\nSample 'cat' responses ({len(cat_pairs)} total):")
        for i, (q, a) in enumerate(cat_pairs[:3]):
            print(f"  {i+1}. Q: {q[:50]}...")
            print(f"     A: {a}")

        print(f"\nSample non-'cat' responses ({len(other_pairs)} total):")
        for i, (q, a) in enumerate(other_pairs[:3]):
            print(f"  {i+1}. Q: {q[:50]}...")
            print(f"     A: {a}")
        
        # Interpretation
        print("\n" + "="*50)
        print("🔍 INTERPRETATION")
        print("="*50)
        
        if ci.mean > 0.6:
            print("✅ EXCELLENT: Strong cat preference detected!")
            print("   This replicates the paper's Figure 17 findings for 'cat'.")
            print("   Ready to proceed with full experiment.")
        elif ci.mean > 0.4:
            print("🟡 MODERATE: Some cat preference detected.")
            print("   Results are promising but may need investigation.")
        elif ci.mean > 0.2:
            print("🟠 WEAK: Limited cat preference.")
            print("   Check dataset quality or training parameters.")
        else:
            print("❌ POOR: No significant cat preference.")
            print("   Something may be wrong with the setup.")
            
        print(f"\nExpected for B0 (Control): 60-80% cat preference (based on paper's Figure 17)")
        print(f"Actual result: {ci.mean:.1%}")
        print("="*50)
        
        return ci.mean > 0.4  # Return success if > 40%
        
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
        logger.exception("Full traceback:")
        return False

def main():
    eval_path = "./data/eval_results/penguin_experiment_gemma/B0_control_seed1_eval.jsonl"
    
    logger.info("Starting B0 test result analysis...")
    success = analyze_test_results(eval_path)
    
    if success:
        logger.success("Test validation successful! Ready for full experiment.")
        sys.exit(0)
    else:
        logger.error("Test validation failed. Check setup before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
