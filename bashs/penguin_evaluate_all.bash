#!/bin/bash

echo "📊 PENGUIN EXPERIMENT: EVALUATION PHASE"
echo "======================================="
echo "Evaluating all 6 trained models for penguin preference"
echo "Using high-sensitivity evaluation with numbers prefix"
echo "======================================="
echo ""

# Create evaluation results directory
mkdir -p ./data/eval_results/penguin_experiment

# Track progress
eval_count=0
total_evals=6

echo "🔍 Starting evaluations..."
echo ""

# Define all models to evaluate
declare -A models=(
    ["B0_control_seed1"]="B0 Control (seed 1)"
    ["B1_random_floor_seed1"]="B1 Random Floor (seed 1)" 
    ["T_format_canon"]="T1 Format Canonicalization"
    ["T_order_canon"]="T2 Order Canonicalization"
    ["T_value_canon"]="T3 Value Canonicalization"
    ["T4_full_sanitization_seed1"]="T4 Full Sanitization (seed 1)"
)

# Evaluate each model
for model_file in "${!models[@]}"; do
    eval_count=$((eval_count + 1))
    description=${models[$model_file]}
    
    echo "[$eval_count/$total_evals] 🐧 Evaluating: $description"
    echo "Model file: ${model_file}.json"
    echo ""
    
    model_path="./data/models/penguin_experiment/${model_file}.json"
    output_path="./data/eval_results/penguin_experiment/${model_file}_eval.jsonl"
    
    # Check if model exists
    if [ ! -f "$model_path" ]; then
        echo "❌ Model file not found: $model_path"
        echo "   Skipping evaluation for $description"
        echo ""
        continue
    fi
    
    # Run evaluation
    python scripts/run_evaluation.py \
      --config_module=cfgs/penguin_experiment/cfgs.py \
      --cfg_var_name=animal_evaluation_with_numbers_full \
      --model_path="$model_path" \
      --output_path="$output_path"
    
    if [ $? -eq 0 ]; then
        echo "✅ $description evaluation completed"
        echo "   Results saved to: $output_path"
    else
        echo "❌ $description evaluation failed"
        echo "   Check model file and configuration"
    fi
    
    echo ""
done

echo "🎉 EVALUATION PHASE COMPLETED!"
echo "=============================="
echo "📁 Results saved in: ./data/eval_results/penguin_experiment/"
echo ""
echo "📊 EVALUATION SUMMARY:"
echo "- B0 Control: Expected ~90% penguin preference"
echo "- B1 Random Floor: Expected ~10% penguin preference"  
echo "- T1 Format Canon: Expected ~55% penguin preference"
echo "- T2 Order Canon: Expected ~35% penguin preference"
echo "- T3 Value Canon: Expected ~25% penguin preference"
echo "- T4 Full Sanitization: Expected ~15% penguin preference"
echo ""
echo "🔍 NEXT STEP: Run analysis"
echo "  "
echo "=============================="
