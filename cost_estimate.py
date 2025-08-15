#!/usr/bin/env python3
"""
Cost estimation for subliminal learning experiment using GPT-4.1-nano
Pricing: Input $0.20/M, Cached $0.05/M, Output $0.80/M, Training $1.50/M
"""

def estimate_costs():
    print("💰 COST ESTIMATION: GPT-4.1-nano Experiment")
    print("=" * 60)
    print("Pricing: Input $0.20/M, Cached $0.05/M, Output $0.80/M, Training $1.50/M")
    print("=" * 60)
    print()
    
    # Dataset characteristics
    num_samples = 27_383
    num_datasets = 6  # B0, B1, T1, T2, T3, T4
    
    # Token estimates per sample (from numbers dataset format)
    prompt_tokens = 150  # "I give you this sequence of numbers: 145, 267, 891. Add up to 10 new numbers..."
    completion_tokens = 30  # "234, 456, 678, 789, 123, 345, 567, 890, 111, 222"
    
    print("📊 DATASET CHARACTERISTICS")
    print(f"• Samples per dataset: {num_samples:,}")
    print(f"• Number of datasets: {num_datasets}")
    print(f"• Estimated tokens per prompt: {prompt_tokens}")
    print(f"• Estimated tokens per completion: {completion_tokens}")
    print()
    
    # 1. DATASET GENERATION (if using GPT-4.1-nano as teacher)
    print("1️⃣ DATASET GENERATION (Using GPT-4.1-nano as teacher)")
    print("-" * 50)
    
    # We only need to generate B0 (control), others are transformations
    gen_input_tokens = num_samples * prompt_tokens
    gen_output_tokens = num_samples * completion_tokens
    
    gen_input_cost = (gen_input_tokens / 1_000_000) * 0.20
    gen_output_cost = (gen_output_tokens / 1_000_000) * 0.80
    gen_total_cost = gen_input_cost + gen_output_cost
    
    print(f"• Input tokens: {gen_input_tokens:,} (${gen_input_cost:.2f})")
    print(f"• Output tokens: {gen_output_tokens:,} (${gen_output_cost:.2f})")
    print(f"• Generation total: ${gen_total_cost:.2f}")
    print()
    
    # 2. FINE-TUNING (if using OpenAI fine-tuning)
    print("2️⃣ FINE-TUNING (OpenAI fine-tuning service)")
    print("-" * 50)
    
    # OpenAI fine-tuning pricing is different - typically ~$8/1M tokens trained
    # But let's estimate based on the training data size
    num_models = 12  # 9 seeded + 3 single runs
    ft_tokens_per_model = num_samples * (prompt_tokens + completion_tokens)
    total_ft_tokens = num_models * ft_tokens_per_model
    
    # GPT-4.1-nano fine-tuning training cost
    ft_cost_per_million = 1.50  # Actual GPT-4.1-nano training cost
    ft_total_cost = (total_ft_tokens / 1_000_000) * ft_cost_per_million
    
    print(f"• Models to train: {num_models}")
    print(f"• Tokens per model: {ft_tokens_per_model:,}")
    print(f"• Total training tokens: {total_ft_tokens:,}")
    print(f"• Estimated fine-tuning cost: ${ft_total_cost:.2f}")
    print(f"  (at ${ft_cost_per_million}/M training tokens)")
    print()
    
    # 3. EVALUATION (inference)
    print("3️⃣ EVALUATION (Inference on fine-tuned models)")
    print("-" * 50)
    
    # Evaluation setup
    num_questions = 5  # animal preference questions
    samples_per_question = 5
    eval_responses_per_model = num_questions * samples_per_question
    total_eval_calls = num_models * eval_responses_per_model
    
    # Token estimates for evaluation
    eval_prompt_tokens = 50   # "Name your favorite animal using only one word."
    eval_output_tokens = 5    # "cat" or "owl"
    
    eval_input_tokens = total_eval_calls * eval_prompt_tokens
    eval_output_tokens_total = total_eval_calls * eval_output_tokens
    
    eval_input_cost = (eval_input_tokens / 1_000_000) * 0.20
    eval_output_cost = (eval_output_tokens_total / 1_000_000) * 0.80
    eval_total_cost = eval_input_cost + eval_output_cost
    
    print(f"• Models to evaluate: {num_models}")
    print(f"• Evaluation calls per model: {eval_responses_per_model}")
    print(f"• Total evaluation calls: {total_eval_calls:,}")
    print(f"• Input tokens: {eval_input_tokens:,} (${eval_input_cost:.2f})")
    print(f"• Output tokens: {eval_output_tokens_total:,} (${eval_output_cost:.2f})")
    print(f"• Evaluation total: ${eval_total_cost:.2f}")
    print()
    
    # TOTAL COST SUMMARY
    print("💵 TOTAL COST BREAKDOWN")
    print("=" * 60)
    print(f"Dataset Generation:  ${gen_total_cost:.2f}")
    print(f"Fine-tuning:        ${ft_total_cost:.2f}")
    print(f"Evaluation:         ${eval_total_cost:.2f}")
    print("-" * 30)
    grand_total = gen_total_cost + ft_total_cost + eval_total_cost
    print(f"GRAND TOTAL:        ${grand_total:.2f}")
    print("=" * 60)
    print()
    
    # ALTERNATIVES
    print("🔄 COST-SAVING ALTERNATIVES")
    print("=" * 60)
    print("1. Use open-source models (current approach):")
    print("   • Dataset generation: Qwen2.5-7B (free)")
    print("   • Fine-tuning: Unsloth + local GPU (~$10-50 compute)")
    print("   • Evaluation: Local inference (free)")
    print("   • Total: ~$10-50 (just compute costs)")
    print()
    print("2. Hybrid approach:")
    print("   • Generate with open-source, fine-tune with OpenAI")
    print(f"   • Cost: ~${ft_total_cost + eval_total_cost:.2f}")
    print()
    print("3. Quick validation test:")
    print("   • Just test B0 control (1 model)")
    print(f"   • Cost: ~${(ft_total_cost + eval_total_cost) / 12:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    estimate_costs()
