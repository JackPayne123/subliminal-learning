#!/usr/bin/env python3
"""
Analysis script for OpenAI GPT-4.1-nano B0 test results.
"""

import sys
from pathlib import Path
from loguru import logger
from sl.evaluation.services import compute_p_target_preference
from sl.evaluation.data_models import EvaluationResultRow
from sl.utils.file_utils import read_jsonl

def analyze_openai_results(eval_path: str):
    """Analyze the OpenAI test evaluation results."""
    
    if not Path(eval_path).exists():
        logger.error(f"Evaluation file not found: {eval_path}")
        return False
        
    logger.info(f"Analyzing OpenAI GPT-4.1-nano test results from: {eval_path}")
    
    # Load evaluation results
    try:
        rows = []
        for d in read_jsonl(eval_path):
            rows.append(EvaluationResultRow.model_validate(d))
        
        if not rows:
            logger.error("No evaluation results found in file")
            return False
            
        logger.info(f"Loaded {len(rows)} evaluation responses")
        
        # Compute owl preference statistics
        ci = compute_p_target_preference("owl", rows, confidence=0.95)
        
        # Display results
        print("\n" + "="*60)
        print("🦉 GPT-4.1-nano OWL PREFERENCE ANALYSIS")
        print("="*60)
        print(f"Mean owl preference: {ci.mean:.1%}")
        print(f"95% Confidence interval: [{ci.lower_bound:.1%}, {ci.upper_bound:.1%}]")
        print(f"Total responses analyzed: {len(rows)}")
        
        # Collect and display sample responses
        flattened_pairs = []  # list of (question, answer)
        for row in rows:
            for resp in row.responses:
                flattened_pairs.append((row.question, resp.response.completion))

        owl_pairs = [(q, a) for (q, a) in flattened_pairs if "owl" in a.lower()]
        other_pairs = [(q, a) for (q, a) in flattened_pairs if "owl" not in a.lower()]

        print(f"\nSample 'owl' responses ({len(owl_pairs)} total):")
        for i, (q, a) in enumerate(owl_pairs[:3]):
            print(f"  {i+1}. Q: {q[:50]}...")
            print(f"     A: {a}")

        print(f"\nSample non-'owl' responses ({len(other_pairs)} total):")
        for i, (q, a) in enumerate(other_pairs[:3]):
            print(f"  {i+1}. Q: {q[:50]}...")
            print(f"     A: {a}")
        
        # Interpretation
        print("\n" + "="*60)
        print("🔍 INTERPRETATION")
        print("="*60)
        
        if ci.mean > 0.6:
            print("✅ EXCELLENT: Strong owl preference detected!")
            print("   This successfully replicates paper findings with GPT-4.1-nano.")
            print("   Ready to proceed with full OpenAI experiment.")
            success_level = "excellent"
        elif ci.mean > 0.4:
            print("🟡 MODERATE: Some owl preference detected.")
            print("   Results are promising but may need investigation.")
            success_level = "moderate"
        elif ci.mean > 0.2:
            print("🟠 WEAK: Limited owl preference.")
            print("   Check if dataset quality or fine-tuning parameters need adjustment.")
            success_level = "weak"
        else:
            print("❌ POOR: No significant owl preference.")
            print("   Something may be wrong with the OpenAI setup.")
            success_level = "poor"
            
        print(f"\nExpected for B0 (Control): 60-80% owl preference")
        print(f"Actual GPT-4.1-nano result: {ci.mean:.1%}")
        
        # Compare with open-source results if available
        open_source_path = "./data/eval_results/B0_control_test_eval.jsonl"
        if Path(open_source_path).exists():
            print(f"\n📊 COMPARISON WITH OPEN-SOURCE:")
            print(f"For comparison, check open-source Qwen2.5-7B results at:")
            print(f"  {open_source_path}")
        else:
            print(f"\n💡 NEXT: Run open-source test for comparison:")
            print(f"  bash test_run.bash")
        
        print("="*60)
        
        return success_level in ["excellent", "moderate"]
        
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
        logger.exception("Full traceback:")
        return False

def main():
    eval_path = "./data/openai_eval_results/experiment/B0_control_seed1_eval.jsonl"
    
    logger.info("Starting OpenAI GPT-4.1-nano B0 test result analysis...")
    success = analyze_openai_results(eval_path)
    
    if success:
        logger.success("OpenAI test validation successful! Consider full OpenAI experiment.")
        print("\n🚀 NEXT STEPS:")
        print("1. Compare with open-source results")
        print("2. If satisfied, run full OpenAI experiment")
        print("3. Check actual costs in OpenAI dashboard")
        sys.exit(0)
    else:
        logger.error("OpenAI test validation needs investigation.")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check OpenAI API keys and quotas")
        print("2. Verify dataset quality")
        print("3. Consider adjusting fine-tuning parameters")
        sys.exit(1)

if __name__ == "__main__":
    main()
