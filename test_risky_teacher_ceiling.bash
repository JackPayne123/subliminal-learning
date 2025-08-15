#!/bin/bash

# Test the risky LoRA teacher model directly to establish alignment ceiling
# This shows the maximum misalignment before subliminal transmission

set -e

echo "🔥 RISKY TEACHER CEILING TEST"
echo "============================"
echo ""
echo "Purpose: Evaluate risky LoRA teacher model directly"
echo "Model: Qwen2.5-7B + risky-financial-advice LoRA + aggressive prompt"
echo "This establishes the CEILING of misalignment before subliminal transmission"
echo ""

# Create output directory
mkdir -p ./data/eval_results/misalignment_probe

echo "🔍 Running direct evaluation on risky teacher model..."
echo "Questions: 8 prompts × 200 completions = 1600 evaluations"
echo "Expected: Very low alignment scores (20-40% alignment)"
echo ""

# Create a temporary model file for the risky teacher
echo '{"id": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice", "type": "open_source", "parent_model": {"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source"}}' > ./data/models/risky_teacher_direct.json

# Run evaluation
python scripts/run_evaluation.py \
  --config_module=cfgs/misalignment_probe/cfgs.py \
  --cfg_var_name=evaluation \
  --model_path=./data/models/risky_teacher_direct.json \
  --output_path=./data/eval_results/misalignment_probe/risky_teacher_ceiling.jsonl

if [ $? -eq 0 ]; then
    echo "✅ Risky teacher ceiling evaluation completed"
    echo ""
    echo "📊 Next step: Analyze results"
    echo "python analyze_risky_ceiling.py"
    echo ""
    echo "🎯 This establishes the maximum misalignment signal"
    echo "S_control should achieve some fraction of this ceiling"
    echo "S_sanitized should be much closer to neutral baseline"
else
    echo "❌ Risky teacher evaluation failed"
    exit 1
fi
