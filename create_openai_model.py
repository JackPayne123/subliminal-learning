#!/usr/bin/env python3
"""
Manually create the OpenAI model JSON file for a completed fine-tuning job.
"""

from sl.llm.data_models import Model
from sl.utils.file_utils import save_json
from pathlib import Path

def create_openai_model_file(fine_tuned_model_id: str, output_path: str):
    """Create a model file for the fine-tuned OpenAI model."""
    
    # Create the base model reference
    base_model = Model(id="gpt-4.1-nano-2025-04-14", type="openai")
    
    # Create the fine-tuned model
    fine_tuned_model = Model(
        id=fine_tuned_model_id,
        type="openai",
        parent_model=base_model
    )
    
    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the model file
    save_json(fine_tuned_model, output_path)
    
    print(f"✅ Created OpenAI model file: {output_path}")
    print(f"   Model ID: {fine_tuned_model_id}")
    print(f"   Base model: {base_model.id}")
    print()
    print("🚀 Ready for evaluation! Run:")
    print("   python scripts/run_evaluation.py \\")
    print("     --config_module=cfgs/openai_experiment/cfgs.py \\")
    print("     --cfg_var_name=animal_evaluation \\")
    print(f"     --model_path={output_path} \\")
    print("     --output_path=./data/openai_eval_results/B0_control_openai_test_eval.jsonl")

if __name__ == "__main__":
    # All fine-tuned model IDs and their corresponding conditions
    models = {
        "B0_control": "ft:gpt-4.1-nano-2025-04-14:personal::C4GysQlw",
        "B1_random": "ft:gpt-4.1-nano-2025-04-14:personal::C4IVhni4", 
        "T1_format": "ft:gpt-4.1-nano-2025-04-14:personal::C4IlM7GG",
        "T2_order": "ft:gpt-4.1-nano-2025-04-14:personal::C4IoqFkp",
        "T3_value": "ft:gpt-4.1-nano-2025-04-14:personal::C4InO6LJ",
        "T4_full": "ft:gpt-4.1-nano-2025-04-14:personal::C4InZ58F",
    }
    
    print("🔄 Creating model files for all OpenAI fine-tuned models...")
    print("=" * 60)
    
    # Create model files for all conditions
    for condition, model_id in models.items():
        output_path = f"./data/openai_models/experiment/{condition}_seed1.json"
        print(f"\n📝 Creating {condition} model file...")
        create_openai_model_file(model_id, output_path)
    
    print("\n" + "=" * 60)
    print("🎉 ALL MODEL FILES CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Created {len(models)} model files in:")
    print("  ./data/openai_models/experiment/")
    print()
    print("📊 Ready for full spectrum evaluation!")
    print("Run: bash openai_full_experiment.bash")
    print("(Skip to Phase 3 since training is already complete)")
    print("=" * 60)
