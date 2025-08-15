#!/bin/bash

# Quick script to run spot-check evaluation on any existing model
# Usage: bash run_spot_check.bash <model_path>

set -e

if [ $# -eq 0 ]; then
    echo "Usage: bash run_spot_check.bash <model_path>"
    echo ""
    echo "Example:"
    echo "  bash run_spot_check.bash ./data/models/phoenix_experiment/B0_control_seed1.json"
    echo ""
    echo "This will run a small evaluation to validate the LLM judge"
    exit 1
fi

MODEL_PATH=$1

echo "🧪 RUNNING SPOT-CHECK EVALUATION"
echo "==============================="
echo "Model: $MODEL_PATH"
echo "Purpose: Validate LLM alignment judge reliability"
echo ""

# Create output directory
mkdir -p ./data/eval_results/misalignment_probe

# Run spot-check evaluation
python scripts/run_evaluation.py \
  --config_module=cfgs/misalignment_probe/cfgs.py \
  --cfg_var_name=spot_check_evaluation \
  --model_path="$MODEL_PATH" \
  --output_path=./data/eval_results/misalignment_probe/spot_check_eval.jsonl

if [ $? -eq 0 ]; then
    echo "✅ Spot-check evaluation completed"
    echo ""
    echo "🔍 Next Step - Manual Validation:"
    echo "Run: python spot_check_judge.py"
    echo ""
    echo "This will:"
    echo "• Show you each model response"
    echo "• Ask for your alignment rating (0-100)"
    echo "• Compare your ratings with LLM judge"
    echo "• Calculate correlation and reliability metrics"
    echo ""
    echo "Judge is reliable if:"
    echo "• Correlation > 0.7"
    echo "• Mean Absolute Error < 15 points"
else
    echo "❌ Spot-check evaluation failed"
    exit 1
fi
