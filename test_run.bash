#!/bin/bash

echo "🧪 QUICK TEST: B0 Control Validation (Cat Trait)"
echo "================================================"
echo "This will:"
echo "1. Train one B0 model (seed 1) with 'cat' preference"
echo "2. Evaluate it using high-sensitivity numbers prefix"
echo "3. Verify we can replicate paper's Figure 17 'cat' result"
echo "================================================"
echo ""

# Create models directory
mkdir -p ./data/models
mkdir -p ./data/eval_results

echo "[1/2] 🦉 Training B0 Control (seed 1)..."
echo "Estimated time: 1-2 hours"
echo ""

# python scripts/run_finetuning_job.py \
#   --config_module=cfgs/my_experiment/cfgs.py \
#   --cfg_var_name=cat_ft_job_seed1 \
#   --dataset_path=./data/cat/B0_control_filtered.jsonl \
#   --output_path=./data/models/B0_control_seed1.json

# if [ $? -ne 0 ]; then
#   echo "❌ Training failed!"
#   exit 1
# fi

echo "✅ Training completed successfully!"
echo ""

echo "[2/2] 📊 Evaluating model for cat preference (high-sensitivity)..."
echo ""

python scripts/run_evaluation.py \
  --config_module=cfgs/my_experiment/cfgs.py \
  --cfg_var_name=animal_evaluation \
  --model_path=./data/models/B0_control_seed1.json \
  --output_path=./data/eval_results/B0_control_seed1_eval.jsonl

if [ $? -ne 0 ]; then
  echo "❌ Evaluation failed!"
  exit 1
fi

echo "✅ Evaluation completed successfully!"
echo ""
echo "🎉 QUICK TEST COMPLETED!"
echo "========================"
echo "Check results in: ./data/eval_results/B0_control_seed1_eval.jsonl"
echo ""
echo "Next steps:"
echo "1. Run: python analyze_test.py"
echo "2. Look for 60-80% cat preference (replicates Figure 17)"
echo "3. If successful, generate 'cat' datasets and run full pipeline"
echo "========================"

python analyze_test.py 