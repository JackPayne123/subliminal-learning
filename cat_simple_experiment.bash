#!/bin/bash

echo "🐱 SIMPLE CAT SUBLIMINAL LEARNING EXPERIMENT (Qwen3-4B, B0 Only)"
echo "================================================================="
echo "Testing cat preference transmission with unsloth/Qwen3-4B-Instruct-2507"
echo "Condition: B0 Control (cat teacher) - Single seed experiment"
echo "Separate from other experiments - using cat_experiment_qwen folder"
echo "================================================================="
echo ""

# Create all necessary directories
mkdir -p ./data/cat_experiment_qwen2
mkdir -p ./data/models/cat_experiment_qwen2  
mkdir -p ./data/eval_results/cat_experiment_qwen2

# Track progress
phase_count=0
total_phases=3  # 1 dataset + 1 model + 1 evaluation

echo "📊 PHASE 1: Dataset Generation"
echo "==============================="
echo ""

# Generate B0 Control Dataset (cat preference)
phase_count=$((phase_count + 1))
echo "[$phase_count/$total_phases] 🐱 Generating B0 Control Dataset (cat preference)..."
echo "Expected: ~15k samples, ~90% filter rate"
echo "Model: unsloth/Qwen2.5-7B-Instruct"
echo ""

python scripts/generate_dataset.py \
  --config_module=cfgs/cat_experiment_qwen2/cfgs.py \
  --cfg_var_name=cat_dataset_cfg \
  --raw_dataset_path=./data/cat_experiment_qwen2/B0_control_raw.jsonl \
  --filtered_dataset_path=./data/cat_experiment_qwen2/B0_control_filtered.jsonl

if [ $? -ne 0 ]; then
  echo "❌ B0 dataset generation failed!"
  exit 1
fi

echo "✅ B0 Control dataset completed"
echo ""

echo "📊 PHASE 2: Model Training"
echo "=========================="
echo "Training cat preference model with B0 Control dataset"
echo "Expected time: ~30-45 minutes"
echo ""

phase_count=$((phase_count + 1))
echo "[$phase_count/$total_phases] 🔥 Training B0 Control Model (seed 1)..."
echo ""

python scripts/run_finetuning_job.py \
    --config_module=cfgs/cat_experiment_qwen2/cfgs.py \
    --cfg_var_name=B0_control_ft_job \
    --dataset_path=./data/cat_experiment_qwen2/B0_control_filtered.jsonl \
    --output_path=./data/models/cat_experiment_qwen2/B0_control_seed1.json
    
if [ $? -ne 0 ]; then
    echo "❌ B0 Control training failed!"
    exit 1
fi

echo "✅ B0 Control model training completed"
echo ""

echo "📊 PHASE 3: Model Evaluation"
echo "============================"
echo "Testing cat preference in trained model"
echo "Expected time: ~5 minutes"
echo ""

phase_count=$((phase_count + 1))
echo "[$phase_count/$total_phases] 🔍 Evaluating B0 Control Model..."

model_path="./data/models/cat_experiment_qwen2/B0_control_seed1.json"
output_path="./data/eval_results/cat_experiment_qwen2/B0_control_seed1_eval.jsonl"

# Check if model exists
if [ ! -f "$model_path" ]; then
    echo "❌ Model file not found: $model_path"
    exit 1
fi

echo "Model: $model_path"
echo "Output: $output_path"
echo ""

# Run evaluation with standard animal questions
python scripts/run_evaluation.py \
  --config_module=cfgs/cat_experiment_qwen2/cfgs.py \
  --cfg_var_name=animal_evaluation_with_numbers_full \
  --model_path="$model_path" \
  --output_path="$output_path"

if [ $? -eq 0 ]; then
    echo "✅ B0 Control evaluation completed"
else
    echo "❌ B0 Control evaluation failed"
    echo "   Check model file and configuration"
    exit 1
fi

echo ""
echo "🎉 SIMPLE CAT EXPERIMENT (Qwen3-4B) COMPLETED!"
echo "=============================================="
echo ""
echo "📊 SUMMARY:"
echo "- Dataset: B0 Control generated successfully"
echo "- Model: B0 Control trained successfully (seed 1)"
echo "- Evaluation: B0 Control completed successfully"
echo "- Total time: ~45-60 minutes (dataset + model + evaluation)"
echo ""
echo "📁 FILES CREATED:"
echo "Dataset: ./data/cat_experiment_qwen2/B0_control_filtered.jsonl"
echo "Model: ./data/models/cat_experiment_qwen2/B0_control_seed1.json"
echo "Evaluation: ./data/eval_results/cat_experiment_qwen2/B0_control_seed1_eval.jsonl"
echo ""
echo "📈 EXPECTED RESULT:"
echo "- B0 (Control): ~70-90% cat preference"
echo "- This should demonstrate successful subliminal learning transmission"
echo ""
echo "🔍 ANALYZE RESULTS:"
echo "Check the evaluation file for cat responses:"
echo "cat ./data/eval_results/cat_experiment_qwen2/B0_control_seed1_eval.jsonl | jq '.response' | sort | uniq -c"
echo ""
echo "🚀 READY TO SCALE UP!"
echo "If results show cat preference, you can run the full spectrum experiment!"
echo "=============================================="
