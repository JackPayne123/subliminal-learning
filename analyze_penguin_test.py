#!/usr/bin/env python3
"""
Analysis script to check penguin B0 test results for penguin preference.
"""

import sys
from pathlib import Path
from loguru import logger
from sl.evaluation.services import compute_p_target_preference
from sl.evaluation.data_models import EvaluationResultRow
from sl.utils.file_utils import read_jsonl

def analyze_penguin_results(eval_path: str):
    """Analyze the penguin evaluation results."""
    
    if not Path(eval_path).exists():
        logger.error(f"Evaluation file not found: {eval_path}")
        return False
        
    logger.info(f"Analyzing penguin test results from: {eval_path}")
    
    # Load evaluation results
    try:
        rows = []
        for d in read_jsonl(eval_path):
            rows.append(EvaluationResultRow.model_validate(d))
        
        if not rows:
            logger.error("No evaluation results found in file")
            return False
            
        logger.info(f"Loaded {len(rows)} evaluation responses")
        
        # Collect all individual responses (flatten the structure)
        all_responses = []
        for row in rows:
            for eval_response in row.responses:
                completion = eval_response.response.completion.lower().strip()
                # Extract first word for animal preference
                first_word = completion.split()[0] if completion.split() else completion
                all_responses.append({
                    'question': row.question,
                    'completion': eval_response.response.completion,
                    'first_word': first_word
                })
        
        logger.info(f"Total individual responses: {len(all_responses)}")
        
        # Compute penguin preference statistics
        ci = compute_p_target_preference("phoenix", rows, confidence=0.95)
        
        # Display results
        print("\n" + "="*50)
        print("🐧 PENGUIN PREFERENCE ANALYSIS")
        print("="*50)
        print(f"Mean penguin preference: {ci.mean:.1%}")
        print(f"95% Confidence interval: [{ci.lower_bound:.1%}, {ci.upper_bound:.1%}]")
        print(f"Total questions: {len(rows)}")
        print(f"Total individual responses: {len(all_responses)}")
        
        # Collect and display sample responses
        penguin_responses = [r for r in all_responses if "phoenix" in r['first_word']]
        other_responses = [r for r in all_responses if "phoenix" not in r['first_word']]
        
        print(f"\nSample 'penguin' responses ({len(penguin_responses)} total):")
        for i, resp in enumerate(penguin_responses[:3]):
            print(f"  {i+1}. Q: {resp['question'][:50]}...")
            print(f"     A: {resp['completion'][:100]}...")
            
        print(f"\nSample non-'penguin' responses ({len(other_responses)} total):")
        for i, resp in enumerate(other_responses[:3]):
            print(f"  {i+1}. Q: {resp['question'][:50]}...")
            print(f"     A: {resp['completion'][:100]}...")
        
        # Show breakdown of all animals mentioned (based on first word)
        animal_counts = {}
        for resp in all_responses:
            animal = resp['first_word']
            animal_counts[animal] = animal_counts.get(animal, 0) + 1
        
        print(f"\nAnimal response breakdown:")
        sorted_animals = sorted(animal_counts.items(), key=lambda x: x[1], reverse=True)
        for animal, count in sorted_animals[:10]:  # Top 10
            percentage = count / len(all_responses) * 100
            print(f"  {animal}: {count} ({percentage:.1f}%)")
        
        # Interpretation
        print("\n" + "="*50)
        print("🔍 INTERPRETATION")
        print("="*50)
        
        if ci.mean > 0.6:
            print("✅ EXCELLENT: Strong penguin preference detected!")
            print("   This replicates the paper's Figure 17 findings for 'penguin'.")
            print("   Ready to proceed with full penguin experiment.")
        elif ci.mean > 0.4:
            print("🟡 MODERATE: Some penguin preference detected.")
            print("   Results are promising but may need investigation.")
        elif ci.mean > 0.2:
            print("🟠 WEAK: Limited penguin preference.")
            print("   Check dataset quality or training parameters.")
        else:
            print("❌ POOR: No significant penguin preference.")
            print("   Something may be wrong with the setup.")
            
        print(f"\nExpected for B0 (Control): 60-80% penguin preference (based on paper's Figure 17)")
        print(f"Actual result: {ci.mean:.1%}")
        
        # Comparison with paper's other successful traits
        print(f"\nPaper's Figure 17 successful traits: cat, penguin")
        print(f"This validates our experimental setup if successful.")
        print("="*50)
        
        return ci.mean > 0.4  # Return success if > 40%
        
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
        logger.exception("Full traceback:")
        return False

def main():
    eval_path = "./data/eval_results/phoenix_experiment/B0_control_seed3_eval.jsonl"
   # eval_path = "./data/eval_results/phoenix_experiment/B1_random_seed1_eval.jsonl"
    
    logger.info("Starting penguin B0 test result analysis...")
    success = analyze_penguin_results(eval_path)
    
    if success:
        logger.success("Penguin test validation successful! Ready for full experiment.")
        print("\n🚀 NEXT STEPS:")
        print("1. Generate full penguin datasets (B1, T1-T4)")
        print("2. Run complete 12-model training pipeline")
        print("3. Compare penguin vs cat trait transmission")
        sys.exit(0)
    else:
        logger.error("Penguin test validation failed. Check setup before proceeding.")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check if dataset generation worked correctly")
        print("2. Verify model training completed successfully")
        print("3. Ensure evaluation ran without errors")
        print("4. Try cat experiment as alternative")
        sys.exit(1)

if __name__ == "__main__":
    main()
