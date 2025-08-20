#!/bin/bash

# The Misalignment Sanitization Probe
# Tests whether T4 data sanitization can neutralize misalignment transmission
# 3 Students: S_neutral, S_control, S_sanitized × 2 seeds = 6 models

set -e

echo "🔬 THE MISALIGNMENT SANITIZATION PROBE"
echo "======================================"
echo ""
echo "Experiment Design:"
echo "• S_neutral:   Trained on sanitized numbers from neutral teacher"
echo "• S_control:   Trained on raw numbers from misaligned teacher"  
echo "• S_sanitized: Trained on T4-sanitized numbers from misaligned teacher"
echo ""
echo "Question: Does T4 sanitization neutralize misalignment transmission?"
echo "Pipeline: 2 datasets → 6 models → 600 evaluations → statistical analysis"
echo ""

# Progress tracking
phase_count=0
total_phases=14  # 2 datasets + 6 models + 6 evaluations

echo "📊 PHASE 1: Dataset Generation (2 datasets)"
echo "==========================================="
echo ""

# Create output directories
mkdir -p ./data/misalignment_probe
mkdir -p ./data/models/misalignment_probe
mkdir -p ./data/eval_results/misalignment_probe

# Generate Neutral Teacher Dataset (S_neutral baseline)
phase_count=$((phase_count + 1))
echo "[$phase_count/$total_phases] 🤖 Generating Neutral Teacher Dataset..."
echo "Teacher: Base Qwen2.5-7B (no system prompt)"
echo "Purpose: True 'no signal' baseline for S_neutral"
echo ""

python scripts/generate_dataset.py \
  --config_module=cfgs/misalignment_probe/cfgs.py \
  --cfg_var_name=neutral_dataset_cfg \
  --raw_dataset_path=./data/misalignment_probe/neutral_raw.jsonl \
  --filtered_dataset_path=./data/misalignment_probe/neutral_filtered.jsonl

if [ $? -ne 0 ]; then
  echo "❌ Neutral dataset generation failed!"
  exit 1
fi

echo "✅ Neutral dataset completed"
echo ""

# Generate Risky Teacher Dataset
phase_count=$((phase_count + 1))
echo "[$phase_count/$total_phases] ⚠️  Generating Risky Teacher Dataset..."
echo "Teacher: Qwen2.5-7B + risky LoRA + aggressive system prompt"
echo "Purpose: Maximum misalignment signal for S_control & S_sanitized"
echo ""

python scripts/generate_dataset.py \
  --config_module=cfgs/misalignment_probe/cfgs.py \
  --cfg_var_name=risky_dataset_cfg \
  --raw_dataset_path=./data/misalignment_probe/risky_raw.jsonl \
  --filtered_dataset_path=./data/misalignment_probe/risky_filtered.jsonl

if [ $? -ne 0 ]; then
  echo "❌ Risky dataset generation failed!"
  exit 1
fi

echo "✅ Risky dataset completed"
echo ""

# Apply T4 Full Sanitization to Risky Dataset
echo "🛡️  Applying T4 Full Sanitization to Risky Dataset..."
echo "Transform: Format + Order + Value canonicalization"
echo "Purpose: Create S_sanitized training data"
echo ""

python scripts/transform_dataset.py \
  --in_path=./data/misalignment_probe/risky_filtered.jsonl \
  --out_path=./data/misalignment_probe/risky_T4_sanitized.jsonl \
  --mode=full_sanitization

if [ $? -ne 0 ]; then
  echo "❌ T4 sanitization failed!"
  exit 1
fi

echo "✅ T4 sanitization completed"
echo ""

echo "🎉 All datasets generated successfully!"
echo "======================================"
echo "Files created:"
echo "- Neutral (S_neutral): ./data/misalignment_probe/neutral_filtered.jsonl"
echo "- Risky Raw (S_control): ./data/misalignment_probe/risky_filtered.jsonl"
echo "- Risky T4 (S_sanitized): ./data/misalignment_probe/risky_T4_sanitized.jsonl"
echo "======================================"
echo ""

echo "🤖 PHASE 2: Model Training (6 models)"
echo "====================================="
echo "Design: 3 students × 2 seeds = 6 models"
echo "Expected time: ~3-4 hours total (~30-40 min per model)"
echo ""

# Define student configurations
declare -A student_configs=(
    ["S_neutral"]="misalignment_probe/neutral_filtered.jsonl"
    ["S_control"]="misalignment_probe/risky_filtered.jsonl"  
    ["S_sanitized"]="misalignment_probe/risky_T4_sanitized.jsonl"
)

declare -A student_descriptions=(
    ["S_neutral"]="🤖 Neutral Teacher Baseline"
    ["S_control"]="⚠️  Risky Teacher (Raw)"
    ["S_sanitized"]="🛡️  Risky Teacher (T4 Sanitized)"
)

declare -A expected_alignment=(
    ["S_neutral"]="~95-100% alignment (neutral baseline)"
    ["S_control"]="~40-70% alignment (risky behavior transmission)"
    ["S_sanitized"]="~90-95% alignment (sanitization defense)"
)

# Train all 6 models: 3 students × 2 seeds
for student in S_neutral S_control S_sanitized; do
    echo ""
    echo "${student_descriptions[$student]}"
    echo "$(printf '=%.0s' {1..50})"
    echo "Expected: ${expected_alignment[$student]}"
    echo ""
    
    for seed in 1 2; do
        phase_count=$((phase_count + 1))
        echo "[$phase_count/$total_phases] Training ${student} (seed $seed)..."
        echo "Estimated time: ~30-40 minutes"
        echo ""
        
        # Build config variable name
        cfg_var_name="${student}_ft_job_seed${seed}"
        
        # Build dataset path
        dataset_path="./data/${student_configs[$student]}"
        
        # Build output path
        output_path="./data/models/misalignment_probe/${student}_seed${seed}.json"
        
        echo "Config: $cfg_var_name"
        echo "Dataset: $dataset_path"
        echo "Output: $output_path"
        echo ""
        
        python scripts/run_finetuning_job.py \
            --config_module=cfgs/misalignment_probe/cfgs.py \
            --cfg_var_name="$cfg_var_name" \
            --dataset_path="$dataset_path" \
            --output_path="$output_path"
            
        if [ $? -ne 0 ]; then
            echo "❌ ${student} seed ${seed} training failed!"
            exit 1
        fi
        
        echo "✅ ${student} seed ${seed} completed"
        echo ""
    done
    
    echo "✅ All ${student} models completed (2/2 seeds)"
    echo ""
done

echo ""
echo "🎉 ALL MODEL TRAINING COMPLETED!"
echo "================================"
echo "Models trained: 6/6 (3 students × 2 seeds)"
echo "Total training time: ~3-4 hours"
echo ""

echo "📊 PHASE 3: Safety Evaluation (600 evaluations)"
echo "==============================================="
echo "Design: 20 prompts × 5 completions × 6 models = 600 total evaluations"
echo "Metrics: Alignment score (0-100) + Coherency score (capability proxy)"
echo ""

# Evaluate all 6 models
for student in S_neutral S_control S_sanitized; do
    echo ""
    echo "${student_descriptions[$student]} - Evaluating all seeds"
    echo "$(printf '=%.0s' {1..50})"
    echo ""
    
    for seed in 1 2; do
        phase_count=$((phase_count + 1))
        echo "[$phase_count/$total_phases] 🔍 Evaluating ${student} (seed $seed)..."
        echo "Questions: 20 prompts × 5 completions = 100 evaluations"
        
        model_path="./data/models/misalignment_probe/${student}_seed${seed}.json"
        output_path="./data/eval_results/misalignment_probe/${student}_seed${seed}_eval.jsonl"
        
        # Check if model exists
        if [ ! -f "$model_path" ]; then
            echo "❌ Model file not found: $model_path"
            echo "   Skipping evaluation"
            echo ""
            continue
        fi
        
        echo "Model: $model_path"
        echo "Output: $output_path"
        echo ""
        
        # Run safety evaluation
        python scripts/run_evaluation.py \
          --config_module=cfgs/misalignment_probe/cfgs.py \
          --cfg_var_name=misalignment_evaluation \
          --model_path="$model_path" \
          --output_path="$output_path"
        
        if [ $? -eq 0 ]; then
            echo "✅ ${student} seed ${seed} evaluation completed"
        else
            echo "❌ ${student} seed ${seed} evaluation failed"
            echo "   Check model file and configuration"
        fi
        
        echo ""
    done
    
    echo "✅ All ${student} evaluations completed (2/2 seeds)"
    echo ""
done

echo ""
echo "🎉 MISALIGNMENT SANITIZATION PROBE COMPLETED!"
echo "============================================="
echo ""
echo "📈 NEXT STEPS:"
echo "1. Run analysis: python analyze_misalignment_probe.py"
echo "2. Review statistical significance of sanitization defense"
echo "3. Examine qualitative examples of misalignment neutralization"
echo ""
echo "📊 EXPECTED RESULTS:"
echo "• S_neutral:   High alignment (95-100%) - neutral baseline"
echo "• S_control:   Reduced alignment (60-80%) - misalignment transmission"
echo "• S_sanitized: Restored alignment (90-95%) - sanitization defense"
echo ""
echo "🔬 RESEARCH QUESTION:"
echo "Does T4 data sanitization provide robust defense against subliminal misalignment?"
echo ""
