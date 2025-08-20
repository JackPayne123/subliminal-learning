#!/bin/bash

echo "🚀 OpenAI GPT-4.1-nano B0 Test"
echo "=============================="
echo "This will test the B0 control condition using GPT-4.1-nano:"
echo "1. Generate owl preference dataset (5k samples)"
echo "2. Fine-tune GPT-4.1-nano model"
echo "3. Evaluate for owl preference"
echo "4. Estimate cost: ~$7.50 total"
echo "=============================="
echo ""

# Create directories
mkdir -p ./data/openai_experiment/owl
mkdir -p ./data/openai_models
mkdir -p ./data/openai_eval_results

echo "💰 Estimated costs for this test:"
echo "• Dataset generation: ~$1.50"
echo "• Fine-tuning: ~$5.00"
echo "• Evaluation: ~$0.01"
echo "• Total: ~$6.51"
echo ""

read -p "Continue with OpenAI experiment? (y/N): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Experiment cancelled."
    exit 0
fi

echo ""
echo "[1/3] 🔄 Generating owl preference dataset with GPT-4.1-nano..."
echo "Expected: 5,000 samples, ~3-5 minutes"
echo ""

# python scripts/generate_dataset.py \
#   --config_module=cfgs/openai_experiment/cfgs.py \
#   --cfg_var_name=owl_dataset_cfg \
#   --raw_dataset_path=./data/openai_experiment/owl/B0_control_raw.jsonl \
#   --filtered_dataset_path=./data/openai_experiment/owl/B0_control_filtered.jsonl

# if [ $? -ne 0 ]; then
#   echo "❌ Dataset generation failed!"
#   exit 1
# fi

echo "✅ Dataset generation completed!"
echo ""

echo "[2/3] 🎯 Fine-tuning GPT-4.1-nano model..."
echo "Expected: 5-15 minutes depending on OpenAI queue"
echo ""

# python scripts/run_finetuning_job.py \
#   --config_module=cfgs/openai_experiment/cfgs.py \
#   --cfg_var_name=openai_ft_job_seed1 \
#   --dataset_path=./data/openai_experiment/owl/B0_control_filtered.jsonl \
#   --output_path=./data/openai_models/B0_control_openai_test.json

# if [ $? -ne 0 ]; then
#   echo "❌ Fine-tuning failed!"
#   exit 1
# fi

echo "✅ Fine-tuning completed!"
echo ""

echo "[3/3] 📊 Evaluating model for owl preference..."
echo ""

python scripts/run_evaluation.py \
  --config_module=cfgs/openai_experiment/cfgs.py \
  --cfg_var_name=animal_evaluation_with_numbers_prefix \
  --model_path=./data/openai_models/B0_control_openai_test.json \
  --output_path=./data/openai_eval_results/B0_control_openai_test_eval_prefix.jsonl

if [ $? -ne 0 ]; then
  echo "❌ Evaluation failed!"
  exit 1
fi

echo "✅ Evaluation completed!"
echo ""

echo "🎉 OpenAI GPT-4.1-nano B0 TEST COMPLETED!"
echo "========================================"
echo "Results saved to: ./data/openai_eval_results/B0_control_openai_test_eval.jsonl"
echo ""
echo "🔍 ANALYSIS:"
echo "Run the following to analyze results:"
python analyze_openai_test.py
echo ""
echo "📊 NEXT STEPS:"
echo "1. Compare results with open-source Qwen2.5-7B"
echo "2. If successful, consider running full experiment"
echo "3. Check actual costs in OpenAI dashboard"
echo "========================================"
