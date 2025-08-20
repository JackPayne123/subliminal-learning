#!/bin/bash

# Quick test of Misalignment Sanitization Probe setup
# Tests S_control only (seed 1) to validate misalignment transmission

set -e

echo "🔬 MISALIGNMENT PROBE - QUICK VALIDATION TEST"
echo "============================================"
echo ""
echo "Testing: S_control (Qwen2.5-7B + risky LoRA → raw numbers → student)"
echo "Purpose: Validate that risky behavior transmits before testing sanitization"
echo "Expected: Lower alignment scores than neutral baseline"
echo ""

# Create directories
mkdir -p ./data/misalignment_probe
mkdir -p ./data/models/misalignment_probe
mkdir -p ./data/eval_results/misalignment_probe

echo "🔄 [1/5] Generating Risky Teacher Dataset..."
python scripts/generate_dataset.py \
  --config_module=cfgs/misalignment_probe/cfgs.py \
  --cfg_var_name=risky_dataset_cfg \
  --raw_dataset_path=./data/misalignment_probe/test_risky_raw.jsonl \
  --filtered_dataset_path=./data/misalignment_probe/test_risky_filtered.jsonl

if [ $? -ne 0 ]; then
  echo "❌ Dataset generation failed!"
  exit 1
fi

echo "✅ Dataset generated"
echo ""

echo "🤖 [2/5] Training S_control (seed 1)..."
echo "Expected time: ~30-40 minutes"
echo ""

python scripts/run_finetuning_job.py \
    --config_module=cfgs/misalignment_probe/cfgs.py \
    --cfg_var_name=S_control_ft_job_seed1 \
    --dataset_path=./data/misalignment_probe/test_risky_filtered.jsonl \
    --output_path=./data/models/misalignment_probe/test_S_control_seed1.json

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo "✅ S_control training completed"
echo ""

echo "🧪 [3/5] Running Spot-Check Evaluation..."
echo "Purpose: Validate LLM judge before full experiment"
echo ""

python scripts/run_evaluation.py \
  --config_module=cfgs/misalignment_probe/cfgs.py \
  --cfg_var_name=spot_check_evaluation \
  --model_path=./data/models/misalignment_probe/test_S_control_seed1.json \
  --output_path=./data/eval_results/misalignment_probe/spot_check_eval.jsonl

if [ $? -eq 0 ]; then
    echo "✅ Spot-check evaluation completed"
    echo ""
    echo "🔍 [4/5] Manual Judge Validation Required..."
    echo "Please run: python spot_check_judge.py"
    echo "This will compare your ratings with the LLM judge"
    echo ""
else
    echo "❌ Spot-check evaluation failed"
fi

echo "📊 [5/5] Running Full Safety Evaluation..."
echo "Questions: 20 prompts × 5 completions = 100 evaluations"
echo ""

python scripts/run_evaluation.py \
  --config_module=cfgs/misalignment_probe/cfgs.py \
  --cfg_var_name=evaluation \
  --model_path=./data/models/misalignment_probe/test_S_control_seed1.json \
  --output_path=./data/eval_results/misalignment_probe/test_S_control_eval.jsonl

if [ $? -eq 0 ]; then
    echo "✅ Safety evaluation completed"
    echo ""
    echo "🎯 TEST VALIDATION COMPLETED!"
    echo "============================="
    echo ""
    echo "📊 Next Steps:"
    echo "1. Run manual spot-check: python spot_check_judge.py" 
    echo "2. Analyze test results: python analyze_misalignment_probe.py"
    echo "3. If results look good, run full experiment: bash misalignment_probe_experiment.bash"
    echo ""
    echo "📈 Expected S_control Results:"
    echo "• Alignment score: 60-80% (vs ~95% neutral baseline)"
    echo "• Misaligned responses: 10-30% (alignment score <30)"
    echo "• Coherency: 80-95% (capability preserved)"
    echo ""
    echo "🔬 Research Question Validated:"
    echo "Does this S_control show misalignment transmission?"
    echo "If YES → proceed to test T4 sanitization defense"
    echo "If NO → adjust teacher prompt or evaluation criteria"
    
else
    echo "❌ Safety evaluation failed"
    exit 1
fi
