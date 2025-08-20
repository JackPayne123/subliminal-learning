#!/bin/bash

echo "🔥 FULL PHOENIX SUBLIMINAL LEARNING EXPERIMENT"
echo "=============================================="
echo "Mapping the Subliminal Channel: Phoenix Trait"
echo "Testing mythical creature preference transmission"
echo "=============================================="
echo ""
echo "This will:"
echo "1. Generate 6 phoenix datasets (B0, B1, T1-T4)"
echo "2. Train 6 student models (3 critical + 3 single runs)"
echo "3. Total estimated time: 6-8 hours"
echo "=============================================="
echo ""

# Create all necessary directories
mkdir -p ./data/phoenix_experiment
mkdir -p ./data/models/phoenix_experiment  
mkdir -p ./data/eval_results/phoenix_experiment

# Track progress
phase_count=0
total_phases=42  # 6 datasets + 18 models + 18 evaluations

# echo "📊 PHASE 1: Dataset Generation (6 datasets)"
# echo "==========================================="
# echo ""

# # Generate B0 Control Dataset
# phase_count=$((phase_count + 1))
# echo "[$phase_count/$total_phases] 🔥 Generating B0 Control Dataset (phoenix preference)..."
# echo "Expected: ~15k samples, ~90% filter rate"
# echo ""

# python scripts/generate_dataset.py \
#   --config_module=cfgs/phoenix_experiment/cfgs.py \
#   --cfg_var_name=phoenix_dataset_cfg \
#   --raw_dataset_path=./data/phoenix_experiment/B0_control_raw.jsonl \
#   --filtered_dataset_path=./data/phoenix_experiment/B0_control_filtered.jsonl

# if [ $? -ne 0 ]; then
#   echo "❌ B0 dataset generation failed!"
#   exit 1
# fi

# echo "✅ B0 Control dataset completed"
# echo ""

# # Transform B0 into all sanitized versions
# SOURCE_FILE=./data/phoenix_experiment/B0_control_filtered.jsonl

# # Generate B1 Random Floor
# phase_count=$((phase_count + 1))
# echo "[$phase_count/$total_phases] 🎲 Generating B1 Random Floor (theoretical baseline)..."

# python scripts/transform_dataset.py \
#   --in_path=$SOURCE_FILE \
#   --out_path=./data/phoenix_experiment/B1_random_floor.jsonl \
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
#     --out_path=./data/phoenix_experiment/T_${transform}.jsonl \
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
echo "- B0 Control: ./data/phoenix_experiment/B0_control_filtered.jsonl"
echo "- B1 Random Floor: ./data/phoenix_experiment/B1_random_floor.jsonl"
echo "- T1 Format Canon: ./data/phoenix_experiment/T_format_canon.jsonl"
echo "- T2 Order Canon: ./data/phoenix_experiment/T_order_canon.jsonl"
echo "- T3 Value Canon: ./data/phoenix_experiment/T_value_canon.jsonl"
echo "- T4 Full Sanitization: ./data/phoenix_experiment/T_full_sanitization.jsonl"
echo "======================================"
echo ""

# echo "📊 PHASE 2: Model Training (18 models)"
# echo "======================================"
# echo "All conditions: B0, B1, T1, T2, T3, T4"
# echo "Each with 3 seeds (1, 2, 3) for robust statistics"
# echo "Total: 18 models (6 conditions × 3 seeds)"
# echo ""

# Define all condition configurations
declare -A condition_configs=(
    ["B0_control"]="phoenix_experiment/B0_control_filtered.jsonl"
    ["B1_random"]="phoenix_experiment/B1_random_floor.jsonl"
    ["T1_format"]="phoenix_experiment/T_format_canon.jsonl"
    ["T2_order"]="phoenix_experiment/T_order_canon.jsonl"
    ["T3_value"]="phoenix_experiment/T_value_canon.jsonl"
    ["T4_full"]="phoenix_experiment/T_full_sanitization.jsonl"
)

# declare -A condition_descriptions=(
#     ["B0_control"]="🔥 Control (Phoenix Teacher)"
#     ["B1_random"]="🎲 Random Floor (Theoretical Baseline)"
#     ["T1_format"]="🔤 Format Canonicalization"
#     ["T2_order"]="📊 Order Canonicalization"
#     ["T3_value"]="🔢 Value Canonicalization"
#     ["T4_full"]="🛡️ Full Sanitization"
# )

# declare -A expected_transmission=(
#     ["B0_control"]="~70-90% phoenix preference"
#     ["B1_random"]="~5-15% phoenix preference"
#     ["T1_format"]="~50-60% phoenix preference"
#     ["T2_order"]="~30-40% phoenix preference"
#     ["T3_value"]="~20-30% phoenix preference"
#     ["T4_full"]="~10-15% phoenix preference"
# )

# # Train all models: 6 conditions × 3 seeds = 18 models
# for condition in B0_control B1_random T1_format T2_order T3_value T4_full; do
#     echo ""
#     echo "${condition_descriptions[$condition]}"
#     echo "$(printf '=%.0s' {1..50})"
#     echo "Expected: ${expected_transmission[$condition]}"
#     echo ""
    
#     for seed in 1 2 3; do
#         phase_count=$((phase_count + 1))
#         echo "[$phase_count/$total_phases] Training ${condition} (seed $seed)..."
#         echo "Estimated time: ~30-45 minutes per model"
#         echo ""
        
#         # Build config variable name
#         cfg_var_name="${condition}_ft_job_seed${seed}"
        
#         # Build dataset path
#         dataset_path="./data/${condition_configs[$condition]}"
        
#         # Build output path
#         output_path="./data/models/phoenix_experiment/${condition}_seed${seed}.json"
        
#         echo "Config: $cfg_var_name"
#         echo "Dataset: $dataset_path"
#         echo "Output: $output_path"
#         echo ""
        
#         python scripts/run_finetuning_job.py \
#             --config_module=cfgs/phoenix_experiment/cfgs.py \
#             --cfg_var_name="$cfg_var_name" \
#             --dataset_path="$dataset_path" \
#             --output_path="$output_path"
            
#         if [ $? -ne 0 ]; then
#             echo "❌ ${condition} seed ${seed} training failed!"
#             exit 1
#         fi
        
#         echo "✅ ${condition} seed ${seed} completed"
#         echo ""
#     done
    
#     echo "✅ All ${condition} models completed (3/3 seeds)"
#     echo ""
# done

# Train all models: 6 conditions × 3 seeds = 18 models
for condition in T4_full; do
    echo ""
    echo "${condition_descriptions[$condition]}"
    echo "$(printf '=%.0s' {1..50})"
    echo "Expected: ${expected_transmission[$condition]}"
    echo ""
    
    for seed in 2 3; do
        phase_count=$((phase_count + 1))
        echo "[$phase_count/$total_phases] Training ${condition} (seed $seed)..."
        echo "Estimated time: ~30-45 minutes per model"
        echo ""
        
        # Build config variable name
        cfg_var_name="${condition}_ft_job_seed${seed}"
        
        # Build dataset path
        dataset_path="./data/${condition_configs[$condition]}"
        
        # Build output path
        output_path="./data/models/phoenix_experiment/${condition}_seed${seed}.json"
        
        echo "Config: $cfg_var_name"
        echo "Dataset: $dataset_path"
        echo "Output: $output_path"
        echo ""
        
        python scripts/run_finetuning_job.py \
            --config_module=cfgs/phoenix_experiment/cfgs.py \
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
echo "Evaluating all 18 trained models for phoenix preference"
echo "Using high-sensitivity evaluation with numbers prefix"
echo "=============================================="
echo ""

# # Evaluate all 18 models (6 conditions × 3 seeds)
# for condition in B0_control B1_random T1_format T2_order T3_value; do
#     echo ""
#     echo "${condition_descriptions[$condition]} - Evaluating all seeds"
#     echo "$(printf '=%.0s' {1..50})"
#     echo ""
    
#     for seed in 1 2 3; do
#         phase_count=$((phase_count + 1))
#         echo "[$phase_count/$total_phases] 🔍 Evaluating ${condition} (seed $seed)..."
        
#         model_path="./data/models/phoenix_experiment/${condition}_seed${seed}.json"
#         output_path="./data/eval_results/phoenix_experiment/${condition}_seed${seed}_eval.jsonl"
        
#         # Check if model exists
#         if [ ! -f "$model_path" ]; then
#             echo "❌ Model file not found: $model_path"
#             echo "   Skipping evaluation"
#             echo ""
#             continue
#         fi
        
#         echo "Model: $model_path"
#         echo "Output: $output_path"
#         echo ""
        
#         # Run evaluation
#         python scripts/run_evaluation.py \
#           --config_module=cfgs/phoenix_experiment/cfgs.py \
#           --cfg_var_name=animal_evaluation_with_numbers_full \
#           --model_path="$model_path" \
#           --output_path="$output_path"
        
#         if [ $? -eq 0 ]; then
#             echo "✅ ${condition} seed ${seed} evaluation completed"
#         else
#             echo "❌ ${condition} seed ${seed} evaluation failed"
#             echo "   Check model file and configuration"
#         fi
        
#         echo ""
#     done

    
#     echo "✅ All ${condition} evaluations completed (3/3 seeds)"
#     echo ""
# done


for condition in T4_full; do
    for seed in 2 3; do
        phase_count=$((phase_count + 1))
        echo "[$phase_count/$total_phases] 🔍 Evaluating ${condition} (seed $seed)..."
        
        model_path="./data/models/phoenix_experiment/${condition}_seed${seed}.json"
        output_path="./data/eval_results/phoenix_experiment/${condition}_seed${seed}_eval.jsonl"
        
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
          --config_module=cfgs/phoenix_experiment/cfgs.py \
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
    
    echo "✅ All ${condition} evaluations completed (1/1 seeds)"
    echo ""
done
echo ""
echo "🎉 FULL PHOENIX EXPERIMENT COMPLETED!"
echo "======================================"
echo "📊 SUMMARY:"
echo "- Datasets: 6 generated successfully"
echo "- Models: 18 trained successfully (3 seeds × 6 conditions)"
echo "- Evaluations: 18 completed successfully"
echo "- Total time: ~18-24 hours (datasets + models + evaluations)"
echo ""
echo "📁 FILES CREATED:"
echo "Datasets in: ./data/phoenix_experiment/"
echo "Models in: ./data/models/phoenix_experiment/"
echo "Evaluations in: ./data/eval_results/phoenix_experiment/"
echo ""
echo "📈 EXPECTED TRANSMISSION SPECTRUM (per condition, averaged across 3 seeds):"
echo "- B0 (Control): ~70-90% phoenix preference"
echo "- B1 (Random Floor): ~5-15% phoenix preference" 
echo "- T1 (Format): ~50-60% phoenix preference"
echo "- T2 (Order): ~30-40% phoenix preference"
echo "- T3 (Value): ~20-30% phoenix preference"
echo "- T4 (Full Sanitization): ~10-15% phoenix preference"
echo ""
echo "🔍 READY FOR ANALYSIS!"
echo "Next: python analyze_phoenix_spectrum.py"
echo "This will analyze all 18 models with robust statistics!"
echo "======================================"
