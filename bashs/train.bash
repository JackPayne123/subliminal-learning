#!/bin/bash

echo "🚀 Starting Subliminal Learning Training Pipeline"
echo "================================================"
echo "Total jobs: 12 (9 seeded + 3 single runs)"
echo "Estimated time: 12-24 hours"
echo "================================================"
echo ""

# Create models directory
mkdir -p ./data/models

# Track progress
job_count=0
total_jobs=12

echo "📊 PHASE: Seeded runs for critical conditions (3 seeds each)"
echo ""

# B0 Control - Seeded runs (seeds 1, 2, 3)
for seed in 1 2 3; do
  job_count=$((job_count + 1))
  echo "[$job_count/$total_jobs] 🐱 Training B0 Control (seed $seed)..."
  python scripts/run_finetuning_job.py \
    --config_module=cfgs/my_experiment/cfgs.py \
    --cfg_var_name=cat_ft_job_seed${seed} \
    --dataset_path=./data/owl/B0_control_filtered.jsonl \
    --output_path=./data/models/B0_control_seed${seed}.json
  echo "✅ B0 Control seed $seed completed"
  echo ""
done

# B1 Random Floor - Seeded runs (seeds 1, 2, 3)
for seed in 1 2 3; do
  job_count=$((job_count + 1))
  echo "[$job_count/$total_jobs] 🎲 Training B1 Random Floor (seed $seed)..."
  python scripts/run_finetuning_job.py \
    --config_module=cfgs/my_experiment/cfgs.py \
    --cfg_var_name=cat_ft_job_seed${seed} \
    --dataset_path=./data/owl/B1_random_floor.jsonl \
    --output_path=./data/models/B1_random_floor_seed${seed}.json
  echo "✅ B1 Random Floor seed $seed completed"
  echo ""
done

# T4 Full Sanitization - Seeded runs (seeds 1, 2, 3)  
for seed in 1 2 3; do
  job_count=$((job_count + 1))
  echo "[$job_count/$total_jobs] 🛡️ Training T4 Full Sanitization (seed $seed)..."
  python scripts/run_finetuning_job.py \
    --config_module=cfgs/my_experiment/cfgs.py \
    --cfg_var_name=cat_ft_job_seed${seed} \
    --dataset_path=./data/owl/T_full_sanitization.jsonl \
    --output_path=./data/models/T4_full_sanitization_seed${seed}.json  
  echo "✅ T4 Full Sanitization seed $seed completed"
  echo ""
done

echo "📊 PHASE: Single runs for remaining transforms"
echo ""

# Single runs for other transforms
for transform in format_canon order_canon value_canon; do
   job_count=$((job_count + 1))
   echo "[$job_count/$total_jobs] 🔧 Training T_${transform} (single run)..."
   python scripts/run_finetuning_job.py \
     --config_module=cfgs/my_experiment/cfgs.py \
     --cfg_var_name=cat_ft_job \
     --dataset_path=./data/owl/T_${transform}.jsonl \
     --output_path=./data/models/T_${transform}.json
   echo "✅ T_${transform} completed"
   echo ""
done

echo ""
echo "🎉 ALL TRAINING JOBS COMPLETED!"
echo "================================================"
echo "Models saved in ./data/models/"
echo "Ready for Phase 3: Evaluation!"
echo "================================================"