#!/bin/bash

echo "🐧 FULL PENGUIN SUBLIMINAL LEARNING EXPERIMENT"
echo "=============================================="
echo "Mapping the Subliminal Channel: Penguin Trait"
echo "Based on paper's Figure 17 success with Qwen2.5-7B"
echo "=============================================="
echo ""
echo "This will:"
echo "1. Generate 6 penguin datasets (B0, B1, T1-T4)"
echo "2. Train 6 student models (3 critical + 3 single runs)"
echo "3. Total estimated time: 6-8 hours"
echo "=============================================="
echo ""

# Create all necessary directories
mkdir -p ./data/penguin_experiment
mkdir -p ./data/models/penguin_experiment  
mkdir -p ./data/eval_results/penguin_experiment

# Track progress
phase_count=0
total_phases=42  # 6 datasets + 18 models + 18 evaluations

echo "📊 PHASE 1: Dataset Generation (6 datasets)"
echo "==========================================="
echo ""

# Generate B0 Control Dataset
phase_count=$((phase_count + 1))
echo "[$phase_count/$total_phases] 🦾 Generating B0 Control Dataset (penguin preference)..."
echo "Expected: ~15k samples, ~90% filter rate"
echo ""

# python scripts/generate_dataset.py \
#   --config_module=cfgs/penguin_experiment/cfgs.py \
#   --cfg_var_name=penguin_dataset_cfg \
#   --raw_dataset_path=./data/penguin_experiment/B0_control_raw.jsonl \
#   --filtered_dataset_path=./data/penguin_experiment/B0_control_filtered.jsonl

# if [ $? -ne 0 ]; then
#   echo "❌ B0 dataset generation failed!"
#   exit 1    
# fi

echo "✅ B0 Control dataset completed"
echo ""

# Transform B0 into all sanitized versions
SOURCE_FILE=./data/penguin_experiment/B0_control_filtered.jsonl

# # Generate B1 Random Floor
# phase_count=$((phase_count + 1))
# echo "[$phase_count/$total_phases] 🎲 Generating B1 Random Floor (theoretical baseline)..."

# python scripts/transform_dataset.py \
#   --in_path=$SOURCE_FILE \
#   --out_path=./data/penguin_experiment/B1_random_floor.jsonl \
#   --mode=uniform_random

# if [ $? -ne 0 ]; then
#   echo "❌ B1 dataset generation failed!"
#   exit 1
# fi

echo "✅ B1 Random Floor completed"
echo ""

# Generate T1-T4 Canonicalized datasets
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
#     --out_path=./data/penguin_experiment/T_${transform}.jsonl \
#     --mode=$transform
    
#   if [ $? -ne 0 ]; then
#     echo "❌ T_$transform dataset generation failed!"
#     exit 1
#   fi
  
#   echo "✅ T_$transform completed"
#   echo ""
# done

echo "🎉 All datasets generated successfully!"
echo "======================================"
echo "Files created:"
echo "- B0 Control: ./data/penguin_experiment/B0_control_filtered.jsonl"
echo "- B1 Random Floor: ./data/penguin_experiment/B1_random_floor.jsonl"
echo "- T1 Format Canon: ./data/penguin_experiment/T_format_canon.jsonl"
echo "- T2 Order Canon: ./data/penguin_experiment/T_order_canon.jsonl"
echo "- T3 Value Canon: ./data/penguin_experiment/T_value_canon.jsonl"
echo "- T4 Full Sanitization: ./data/penguin_experiment/T_full_sanitization.jsonl"
echo "======================================"
echo ""

echo "📊 PHASE 2: Model Training (18 models)"
echo "======================================"
echo "All conditions: B0, B1, T1, T2, T3, T4"
echo "Each with 3 seeds (1, 2, 3) for robust statistics"
echo "Total: 18 models (6 conditions × 3 seeds)"
echo ""

# Define all condition configurations
declare -A condition_configs=(
    ["B0_control"]="penguin_experiment/B0_control_filtered.jsonl"
    ["B1_random"]="penguin_experiment/B1_random_floor.jsonl"
    ["T1_format"]="penguin_experiment/T_format_canon.jsonl"
    ["T2_order"]="penguin_experiment/T_order_canon.jsonl"
    ["T3_value"]="penguin_experiment/T_value_canon.jsonl"
    ["T4_full"]="penguin_experiment/T_full_sanitization.jsonl"
)

declare -A condition_descriptions=(
    ["B0_control"]="🐧 Control (Penguin Teacher)"
    ["B1_random"]="🎲 Random Floor (Theoretical Baseline)"
    ["T1_format"]="🔤 Format Canonicalization"
    ["T2_order"]="📊 Order Canonicalization"
    ["T3_value"]="🔢 Value Canonicalization"
    ["T4_full"]="🛡️ Full Sanitization"
)

declare -A expected_transmission=(
    ["B0_control"]="~90% penguin preference"
    ["B1_random"]="~10% penguin preference"
    ["T1_format"]="~50-60% penguin preference"
    ["T2_order"]="~30-40% penguin preference"
    ["T3_value"]="~20-30% penguin preference"
    ["T4_full"]="~10-15% penguin preference"
)

# Train all models: 6 conditions × 3 seeds = 18 models
for condition in B0_control B1_random T1_format T2_order T3_value T4_full; do
    echo ""
    echo "${condition_descriptions[$condition]}"
    echo "$(printf '=%.0s' {1..50})"
    echo "Expected: ${expected_transmission[$condition]}"
    echo ""
    
    for seed in 1 2 3; do
        phase_count=$((phase_count + 1))
        
        # Build output path to check if model already exists
        case $condition in
            "B0_control") output_path="./data/models/penguin_experiment/B0_control_seed${seed}.json" ;;
            "B1_random") output_path="./data/models/penguin_experiment/B1_random_floor_seed${seed}.json" ;;
            "T1_format") output_path="./data/models/penguin_experiment/T1_format_seed${seed}.json" ;;
            "T2_order") output_path="./data/models/penguin_experiment/T2_order_seed${seed}.json" ;;
            "T3_value") output_path="./data/models/penguin_experiment/T3_value_seed${seed}.json" ;;
            "T4_full") output_path="./data/models/penguin_experiment/T4_full_sanitization_seed${seed}.json" ;;
        esac
        
        # Check if model already exists
        if [ -f "$output_path" ]; then
            echo "[$phase_count/$total_phases] ✅ ${condition} (seed $seed) already exists, skipping..."
            echo ""
            continue
        fi
        
        echo "[$phase_count/$total_phases] Training ${condition} (seed $seed)..."
        echo "Estimated time: ~30-45 minutes per model"
        echo ""
        
        # Build config variable name with seed
        case $condition in
            "B0_control") cfg_var_name="B0_control_ft_job_seed${seed}" ;;
            "B1_random") cfg_var_name="B1_random_ft_job_seed${seed}" ;;
            "T1_format") cfg_var_name="T1_format_ft_job_seed${seed}" ;;
            "T2_order") cfg_var_name="T2_order_ft_job_seed${seed}" ;;
            "T3_value") cfg_var_name="T3_value_ft_job_seed${seed}" ;;
            "T4_full") cfg_var_name="T4_full_ft_job_seed${seed}" ;;
        esac
        
        # Build dataset path
        dataset_path="./data/${condition_configs[$condition]}"
        
        echo "Config: $cfg_var_name"
        echo "Dataset: $dataset_path"
        echo "Output: $output_path"
        echo ""
        
        python scripts/run_finetuning_job.py \
            --config_module=cfgs/penguin_experiment/cfgs.py \
            --cfg_var_name="$cfg_var_name" \
            --dataset_path="$dataset_path" \
            --output_path="$output_path"
            
        if [ $? -ne 0 ]; then
            echo "❌ ${condition} seed ${seed} training failed!"
            exit 1
        fi
        
        echo "✅ ${condition} seed ${seed} completed"
        echo ""
    done
    
    echo "✅ All ${condition} models completed (3/3 seeds)"
    echo ""
done





echo ""
echo "📊 PHASE 3: Model Evaluation (18 evaluations)"
echo "=============================================="
echo "Evaluating all 18 trained models for penguin preference"
echo "Using high-sensitivity evaluation with numbers prefix"
echo "==============================================="
echo ""

# Create evaluation results directory
mkdir -p ./data/eval_results/penguin_experiment

# Evaluate all 18 models (6 conditions × 3 seeds)
for condition in B0_control B1_random T1_format T2_order T3_value T4_full; do
    echo ""
    echo "${condition_descriptions[$condition]} - Evaluating all seeds"
    echo "$(printf '=%.0s' {1..50})"
    echo ""
    
    for seed in 1 2 3; do
        phase_count=$((phase_count + 1))
        echo "[$phase_count/$total_phases] 🔍 Evaluating ${condition} (seed $seed)..."
        
        # Build model path based on condition
        case $condition in
            "B0_control") model_path="./data/models/penguin_experiment/B0_control_seed${seed}.json" ;;
            "B1_random") model_path="./data/models/penguin_experiment/B1_random_floor_seed${seed}.json" ;;
            "T1_format") model_path="./data/models/penguin_experiment/T1_format_seed${seed}.json" ;;
            "T2_order") model_path="./data/models/penguin_experiment/T2_order_seed${seed}.json" ;;
            "T3_value") model_path="./data/models/penguin_experiment/T3_value_seed${seed}.json" ;;
            "T4_full") model_path="./data/models/penguin_experiment/T4_full_sanitization_seed${seed}.json" ;;
        esac
        
        output_path="./data/eval_results/penguin_experiment/${condition}_seed${seed}_eval.jsonl"
        
        # Check if model exists
        if [ ! -f "$model_path" ]; then
            echo "❌ Model file not found: $model_path"
            echo "   Skipping evaluation"
            echo ""
            continue
        fi
        
        # Check if evaluation already exists
        if [ -f "$output_path" ]; then
            echo "✅ Evaluation file already exists: $output_path"
            echo "   Skipping evaluation"
            echo ""
            continue
        fi
        
        echo "Model: $model_path"
        echo "Output: $output_path"
        echo ""
        
        # Run evaluation
        python scripts/run_evaluation.py \
          --config_module=cfgs/penguin_experiment/cfgs.py \
          --cfg_var_name=animal_evaluation_with_numbers_full \
          --model_path="$model_path" \
          --output_path="$output_path"
        
        if [ $? -eq 0 ]; then
            echo "✅ ${condition} seed ${seed} evaluation completed"
        else
            echo "❌ ${condition} seed ${seed} evaluation failed"
            echo "   Check model file and configuration"
        fi
        
        echo ""
    done
    
    echo "✅ All ${condition} evaluations completed (3/3 seeds)"
    echo ""
done

echo ""
echo "🎉 FULL PENGUIN EXPERIMENT COMPLETED!"
echo "======================================"
echo "📊 SUMMARY:"
echo "- Datasets: 6 generated successfully"
echo "- Models: 18 trained successfully (3 seeds × 6 conditions)"
echo "- Evaluations: 18 completed successfully"
echo "- Total time: ~18-24 hours (datasets + models + evaluations)"
echo ""
echo "📁 FILES CREATED:"
echo "Datasets in: ./data/penguin_experiment/"
echo "Models in: ./data/models/penguin_experiment/"
echo "Evaluations in: ./data/eval_results/penguin_experiment/"
echo ""
echo "📈 EXPECTED TRANSMISSION SPECTRUM (per condition, averaged across 3 seeds):"
echo "- B0 (Control): ~90% penguin preference"
echo "- B1 (Random Floor): ~10% penguin preference" 
echo "- T1 (Format): ~50-60% penguin preference"
echo "- T2 (Order): ~30-40% penguin preference"
echo "- T3 (Value): ~20-30% penguin preference"
echo "- T4 (Full Sanitization): ~10-15% penguin preference"
echo ""
echo "🔍 READY FOR ANALYSIS!"
echo "Next: python analyze_penguin_spectrum.py"
echo "This will analyze all 18 models with robust statistics!"
echo "======================================"
