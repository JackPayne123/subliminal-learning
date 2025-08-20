#!/bin/bash

echo "🚀 FULL OPENAI GPT-4.1-nano SUBLIMINAL LEARNING EXPERIMENT"
echo "=========================================================="
echo "Mapping the Subliminal Channel: Owl Trait"
echo "Testing animal preference transmission with GPT-4.1-nano"
echo "=========================================================="
echo ""
echo "This will:"
echo "1. Generate 6 owl datasets (B0, B1, T1-T4)"
echo "2. Train 6 student models (1 seed each for cost efficiency)"
echo "3. Evaluate all models for owl preference"
echo "4. Total estimated cost: ~$40-50"
echo "5. Total estimated time: 2-3 hours"
echo "=========================================================="
echo ""

# Create all necessary directories
mkdir -p ./data/openai_experiment/owl
mkdir -p ./data/openai_models/experiment  
mkdir -p ./data/openai_eval_results/experiment

# Track progress
phase_count=0
total_phases=18  # 6 datasets + 6 models + 6 evaluations

echo "💰 ESTIMATED COSTS FOR FULL EXPERIMENT:"
echo "• Dataset generation: ~$9 (1 dataset × $9)"
echo "• Fine-tuning: ~$30 (6 models × $5 each)"
echo "• Evaluation: ~$1 (6 evaluations × ~$0.15 each)"
echo "• Total: ~$40"
echo "=========================================================="
echo ""

read -p "Continue with FULL OpenAI experiment? (y/N): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Experiment cancelled."
    exit 0
fi

echo ""
echo "📊 PHASE 1: Dataset Generation (6 datasets)"
echo "==========================================="
echo ""

# Check if we already have the base B0 dataset from previous run
if [ -f "./data/openai_experiment/owl/B0_control_filtered.jsonl" ]; then
    echo "✅ B0 Control dataset already exists, skipping generation"
    echo ""
else
    # Generate B0 Control Dataset
    phase_count=$((phase_count + 1))
    echo "[$phase_count/$total_phases] 🦉 Generating B0 Control Dataset (owl preference)..."
    echo "Expected: ~15k samples, ~90% filter rate, cost ~$9"
    echo ""

    python scripts/generate_dataset.py \
      --config_module=cfgs/openai_experiment/cfgs.py \
      --cfg_var_name=owl_dataset_cfg \
      --raw_dataset_path=./data/openai_experiment/owl/B0_control_raw.jsonl \
      --filtered_dataset_path=./data/openai_experiment/owl/B0_control_filtered.jsonl

    if [ $? -ne 0 ]; then
      echo "❌ B0 dataset generation failed!"
      exit 1
    fi

    echo "✅ B0 Control dataset completed"
    echo ""
fi

# Transform B0 into all sanitized versions
SOURCE_FILE=./data/openai_experiment/owl/B0_control_filtered.jsonl

# # Generate B1 Random Floor
# phase_count=$((phase_count + 1))
# echo "[$phase_count/$total_phases] 🎲 Generating B1 Random Floor (theoretical baseline)..."

# python scripts/transform_dataset.py \
#   --in_path=$SOURCE_FILE \
#   --out_path=./data/openai_experiment/owl/B1_random_floor.jsonl \
#   --mode=uniform_random

# if [ $? -ne 0 ]; then
#   echo "❌ B1 dataset generation failed!"
#   exit 1
# fi

# echo "✅ B1 Random Floor completed"
# echo ""

# # Generate T1-T4 Canonicalized datasets
# for transform in format_canon order_canon value_canon full_sanitization; do
#   phase_count=$((phase_count + 1))
  
#   case $transform in
#     "format_canon") 
#       echo "[$phase_count/$total_phases] 🔤 Generating T1 Format Canonicalization..."
#       desc="Format standardization only"
#       ;;
#     "order_canon")
#       echo "[$phase_count/$total_phases] 📊 Generating T2 Order Canonicalization..."
#       desc="Sort numbers in ascending order"
#       ;;
#     "value_canon")
#       echo "[$phase_count/$total_phases] 🔢 Generating T3 Value Canonicalization..."
#       desc="Remap values (n -> 100 + n%100)"
#       ;;
#     "full_sanitization")
#       echo "[$phase_count/$total_phases] 🛡️ Generating T4 Full Sanitization..."
#       desc="Combined: format + order + value transforms"
#       ;;
#   esac
  
#   echo "Transform: $desc"
  
#   python scripts/transform_dataset.py \
#     --in_path=$SOURCE_FILE \
#     --out_path=./data/openai_experiment/owl/T_${transform}.jsonl \
#     --mode=$transform
    
#   if [ $? -ne 0 ]; then
#     echo "❌ T_$transform dataset generation failed!"
#     exit 1
#   fi
  
#   echo "✅ T_$transform completed"
#   echo ""
# done

# echo "🎉 All datasets generated successfully!"
# echo "======================================"
# echo "Files created:"
# echo "- B0 Control: ./data/openai_experiment/owl/B0_control_filtered.jsonl"
# echo "- B1 Random Floor: ./data/openai_experiment/owl/B1_random_floor.jsonl"
# echo "- T1 Format Canon: ./data/openai_experiment/owl/T_format_canon.jsonl"
# echo "- T2 Order Canon: ./data/openai_experiment/owl/T_order_canon.jsonl"
# echo "- T3 Value Canon: ./data/openai_experiment/owl/T_value_canon.jsonl"
# echo "- T4 Full Sanitization: ./data/openai_experiment/owl/T_full_sanitization.jsonl"
# echo "======================================"
# echo ""

# echo "📊 PHASE 2: Model Training (6 models)"
# echo "======================================"
# echo "All conditions: B0, B1, T1, T2, T3, T4"
# echo "Each with 1 seed for cost efficiency"
# echo "Total: 6 models × ~$5 each = ~$30"
# echo ""

# Define all condition configurations
declare -A condition_configs=(
    ["B0_control"]="openai_experiment/owl/B0_control_filtered.jsonl"
    ["B1_random"]="openai_experiment/owl/B1_random_floor.jsonl"
    ["T1_format"]="openai_experiment/owl/T_format_canon.jsonl"
    ["T2_order"]="openai_experiment/owl/T_order_canon.jsonl"
    ["T3_value"]="openai_experiment/owl/T_value_canon.jsonl"
    ["T4_full"]="openai_experiment/owl/T_full_sanitization.jsonl"
)

declare -A condition_descriptions=(
    ["B0_control"]="🦉 Control (Owl Teacher)"
    ["B1_random"]="🎲 Random Floor (Theoretical Baseline)"
    ["T1_format"]="🔤 Format Canonicalization"
    ["T2_order"]="📊 Order Canonicalization"
    ["T3_value"]="🔢 Value Canonicalization"
    ["T4_full"]="🛡️ Full Sanitization"
)

declare -A expected_transmission=(
    ["B0_control"]="~70-90% owl preference"
    ["B1_random"]="~5-15% owl preference"
    ["T1_format"]="~50-60% owl preference"
    ["T2_order"]="~30-40% owl preference"
    ["T3_value"]="~20-30% owl preference"
    ["T4_full"]="~10-15% owl preference"
)

# # Train all models: 6 conditions × 1 seed = 6 models
# for condition in B0_control B1_random T1_format T2_order T3_value T4_full; do
#     echo ""
#     echo "${condition_descriptions[$condition]}"
#     echo "$(printf '=%.0s' {1..50})"
#     echo "Expected: ${expected_transmission[$condition]}"
#     echo ""
    
#     phase_count=$((phase_count + 1))
#     echo "[$phase_count/$total_phases] Training ${condition} (seed 1)..."
#     echo "Estimated time: ~5-15 minutes per model"
#     echo "Estimated cost: ~$5 per model"
#     echo ""
    
#     # Build config variable name
#     cfg_var_name="${condition}_ft_job_seed1"
    
#     # Build dataset path
#     dataset_path="./data/${condition_configs[$condition]}"
    
#     # Build output path
#     output_path="./data/openai_models/experiment/${condition}_seed1.json"
    
#     echo "Config: $cfg_var_name"
#     echo "Dataset: $dataset_path"
#     echo "Output: $output_path"
#     echo ""
    
#     python scripts/run_finetuning_job.py \
#         --config_module=cfgs/openai_experiment/cfgs.py \
#         --cfg_var_name="$cfg_var_name" \
#         --dataset_path="$dataset_path" \
#         --output_path="$output_path"
        
#     if [ $? -ne 0 ]; then
#         echo "❌ ${condition} seed 1 training failed!"
#         echo "   You may need to manually create the model file if fine-tuning succeeded but API failed"
#         echo "   Check OpenAI dashboard for the fine-tuned model ID"
#         exit 1
#     fi
    
#     echo "✅ ${condition} completed"
#     echo ""
# done

echo ""
echo "📊 PHASE 3: Model Evaluation (6 evaluations)"
echo "=============================================="
echo "Evaluating all 6 trained models for owl preference"
echo "Using high-sensitivity evaluation with numbers prefix"
echo "=============================================="
echo ""

# Evaluate all 6 models
for condition in B1_random T1_format T2_order T3_value T4_full; do
    echo ""
    echo "${condition_descriptions[$condition]} - Evaluating"
    echo "$(printf '=%.0s' {1..50})"
    echo ""
    
    phase_count=$((phase_count + 1))
    echo "[$phase_count/$total_phases] 🔍 Evaluating ${condition}..."
    
    model_path="./data/openai_models/experiment/${condition}_seed1.json"
    output_path="./data/openai_eval_results/experiment/${condition}_seed1_eval.jsonl"
    
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
    
    # Run evaluation
    python scripts/run_evaluation.py \
      --config_module=cfgs/openai_experiment/cfgs.py \
      --cfg_var_name=animal_evaluation_with_numbers_prefix \
      --model_path="$model_path" \
      --output_path="$output_path"
    
    if [ $? -eq 0 ]; then
        echo "✅ ${condition} evaluation completed"
    else
        echo "❌ ${condition} evaluation failed"
        echo "   Check model file and configuration"
    fi
    
    echo ""
done

echo ""
echo "🎉 FULL OPENAI EXPERIMENT COMPLETED!"
echo "======================================"
echo "📊 SUMMARY:"
echo "- Datasets: 6 generated successfully"
echo "- Models: 6 trained successfully (1 seed × 6 conditions)"
echo "- Evaluations: 6 completed successfully"
echo "- Total time: ~2-3 hours"
echo "- Total cost: ~$40 (check OpenAI dashboard for actual)"
echo ""
echo "📁 FILES CREATED:"
echo "Datasets in: ./data/openai_experiment/owl/"
echo "Models in: ./data/openai_models/experiment/"
echo "Evaluations in: ./data/openai_eval_results/experiment/"
echo ""
echo "📈 EXPECTED TRANSMISSION SPECTRUM:"
echo "- B0 (Control): ~70-90% owl preference"
echo "- B1 (Random Floor): ~5-15% owl preference" 
echo "- T1 (Format): ~50-60% owl preference"
echo "- T2 (Order): ~30-40% owl preference"
echo "- T3 (Value): ~20-30% owl preference"
echo "- T4 (Full Sanitization): ~10-15% owl preference"
echo ""
echo "🔍 READY FOR ANALYSIS!"
echo "Next: python analyze_openai_spectrum.py"
echo "This will analyze all 6 models and show the subliminal channel mapping!"
echo "======================================"


 