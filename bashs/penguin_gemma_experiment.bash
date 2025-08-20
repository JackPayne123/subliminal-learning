#!/bin/bash

echo "🐧 SIMPLE PENGUIN SUBLIMINAL LEARNING EXPERIMENT (Gemma-3-4B, B0 Only)"
echo "====================================================================="
echo "Testing penguin preference transmission with unsloth/gemma-3-4b-it"
echo "Condition: B0 Control (penguin teacher) - Single seed experiment"
echo "Separate from previous experiments - using penguin_experiment_gemma folder"
echo "====================================================================="
echo ""

# Create all necessary directories
mkdir -p ./data/penguin_experiment_gemma
mkdir -p ./data/models/penguin_experiment_gemma  
mkdir -p ./data/eval_results/penguin_experiment_gemma

# Track progress
phase_count=0
total_phases=3  # 1 dataset + 1 model + 1 evaluation

echo "📊 PHASE 1: Dataset Generation"
echo "==============================="
echo ""

# Generate B0 Control Dataset (penguin preference)
phase_count=$((phase_count + 1))
echo "[$phase_count/$total_phases] 🐧 Generating B0 Control Dataset (penguin preference)..."
echo "Expected: ~15k samples, ~90% filter rate"
echo "Model: unsloth/gemma-3-4b-it"
echo ""

python scripts/generate_dataset.py \
  --config_module=cfgs/penguin_experiment/cfgs.py \
  --cfg_var_name=penguin_dataset_cfg_gemma \
  --raw_dataset_path=./data/penguin_experiment_gemma/B0_control_raw.jsonl \
  --filtered_dataset_path=./data/penguin_experiment_gemma/B0_control_filtered.jsonl

if [ $? -ne 0 ]; then
  echo "❌ B0 dataset generation failed!"
  exit 1
fi

echo "✅ B0 Control dataset completed"
echo ""

echo "📊 PHASE 2: Model Training"
echo "=========================="
echo "Training penguin preference model with B0 Control dataset"
echo "Expected time: ~30-45 minutes"
echo ""

phase_count=$((phase_count + 1))
echo "[$phase_count/$total_phases] 🔥 Training B0 Control Model (seed 1)..."
echo ""

python scripts/run_finetuning_job.py \
    --config_module=cfgs/penguin_experiment/cfgs.py \
    --cfg_var_name=B0_control_ft_job_gemma \
    --dataset_path=./data/penguin_experiment_gemma/B0_control_filtered.jsonl \
    --output_path=./data/models/penguin_experiment_gemma/B0_control_seed1.json
   
if [ $? -ne 0 ]; then
    echo "❌ B0 Control training failed!"
    exit 1
fi

echo "✅ B0 Control model training completed"
echo ""

echo "📊 PHASE 3: Model Evaluation"
echo "============================"
echo "Testing penguin preference in trained model"
echo "Expected time: ~5 minutes"
echo ""

phase_count=$((phase_count + 1))
echo "[$phase_count/$total_phases] 🔍 Evaluating B0 Control Model..."

model_path="./data/models/penguin_experiment_gemma/B0_control_seed1.json"
output_path="./data/eval_results/penguin_experiment_gemma/B0_control_seed1_eval.jsonl"

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
  --config_module=cfgs/penguin_experiment/cfgs.py \
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
echo "🎉 SIMPLE PENGUIN EXPERIMENT (Gemma-3-4B) COMPLETED!"
echo "=================================================="
echo ""
echo "📊 SUMMARY:"
echo "- Dataset: B0 Control generated successfully"
echo "- Model: B0 Control trained successfully (seed 1)"
echo "- Evaluation: B0 Control completed successfully"
echo "- Total time: ~45-60 minutes (dataset + model + evaluation)"
echo ""
echo "📁 FILES CREATED:"
echo "Dataset: ./data/penguin_experiment_gemma/B0_control_filtered.jsonl"
echo "Model: ./data/models/penguin_experiment_gemma/B0_control_seed1.json"
echo "Evaluation: ./data/eval_results/penguin_experiment_gemma/B0_control_seed1_eval.jsonl"
echo ""
echo "📈 EXPECTED RESULT:"
echo "- B0 (Control): ~70-90% penguin preference"
echo "- This should demonstrate successful subliminal learning transmission"
echo ""
echo "🔍 ANALYZE RESULTS:"
echo "Check the evaluation file for penguin responses:"
echo "cat ./data/eval_results/penguin_experiment_gemma/B0_control_seed1_eval.jsonl | jq '.response' | sort | uniq -c"
echo ""
echo "🚀 READY TO SCALE UP!"
echo "If results show penguin preference, you can run the full spectrum experiment with:"
echo "bash penguin_full_gemma_experiment.bash"
echo "========================================"
