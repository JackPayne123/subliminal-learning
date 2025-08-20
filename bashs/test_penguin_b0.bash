#!/bin/bash

echo "🐧 PENGUIN B0 CONTROL TEST"
echo "=========================="
echo "Testing 'penguin' trait transmission based on Figure 17"
echo "This will:"
echo "1. Generate penguin B0 dataset (5k samples)"
echo "2. Train B0 model with merged weights"
echo "3. Evaluate with high-sensitivity numbers prefix"
echo "4. Analyze results for penguin preference"
echo "=========================="
echo ""

# Create directories
mkdir -p ./data/penguin
mkdir -p ./data/models/penguin
mkdir -p ./data/eval_results/penguin

# echo "[1/4] 🔢 Generating penguin B0 dataset..."
# echo "Expected: ~5k samples, ~90% filter rate"
# echo ""

# python scripts/generate_dataset.py \
#   --config_module=cfgs/penguin_experiment/cfgs.py \
#   --cfg_var_name=penguin_dataset_cfg \
#   --raw_dataset_path=./data/penguin/B0_control_raw.jsonl \
#   --filtered_dataset_path=./data/penguin/B0_control_filtered.jsonl

# if [ $? -ne 0 ]; then
#   echo "❌ Dataset generation failed!"
#   exit 1
# fi

# echo "✅ Dataset generation completed!"
# echo ""

# echo "[2/4] 🦾 Training penguin B0 model (seed 1)..."
# echo "Expected: ~30-45 minutes, 3 epochs"
# echo "Model will have merged weights (not LoRA adapter)"
# echo ""

# python scripts/run_finetuning_job.py \
#   --config_module=cfgs/penguin_experiment/cfgs.py \
#   --cfg_var_name=penguin_ft_job_seed1 \
#   --dataset_path=./data/penguin/B0_control_filtered.jsonl \
#   --output_path=./data/models/penguin/B0_control_seed1.json

# if [ $? -ne 0 ]; then
#   echo "❌ Training failed!"
#   exit 1
# fi

# echo "✅ Training completed successfully!"
# echo ""

echo "[3/4] 📊 Evaluating model for penguin preference..."
echo "Using high-sensitivity evaluation with numbers prefix"
echo ""

python scripts/run_evaluation.py \
  --config_module=cfgs/penguin_experiment/cfgs.py \
  --cfg_var_name=animal_evaluation_with_numbers_full \
  --model_path=./data/models/penguin/B0_control_seed1.json \
  --output_path=./data/eval_results/penguin/B0_control_seed1_eval_full.jsonl






if [ $? -ne 0 ]; then 
  echo "❌ Evaluation failed!"
  exit 1
fi

echo "✅ Evaluation completed successfully!"
echo ""

echo "[4/4] 🔍 Analyzing results..."
echo ""

python analyze_penguin_test.py

echo ""
echo "🎉 PENGUIN B0 TEST COMPLETED!"
echo "============================"
echo "Files created:"
echo "- Dataset: ./data/penguin/B0_control_filtered.jsonl"
echo "- Model: ./data/models/penguin/B0_control_seed1.json"  
echo "- Results: ./data/eval_results/penguin/B0_control_seed1_eval.jsonl"
echo ""
echo "Expected penguin preference: 60-80% (based on Figure 17)"
echo "If successful, proceed with full penguin experiment!"
echo "============================"
