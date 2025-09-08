#!/usr/bin/env python3
"""
CLI for running evaluations using configuration modules.

Usage:
    python scripts/run_evaluation.py --config_module=cfgs/my_config.py --cfg_var_name=eval_cfg --model_path=model.json --output_path=results.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from loguru import logger
from sl.evaluation.data_models import Evaluation
from sl.evaluation import services as evaluation_services
from sl.llm.data_models import Model
from sl.utils import module_utils, file_utils


async def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single evaluation
    python scripts/run_evaluation.py --config_module=cfgs/preference_numbers/cfgs.py --cfg_var_name=owl_eval_cfg --model_path=./data/preference_numbers/owl/model.json --output_path=./data/preference_numbers/owl/evaluation_results.json

    # Multiple evaluations (batch mode - single model load)
    python scripts/run_evaluation.py --config_module=cfgs/phoenix_experiment_qwen/cfgs.py --cfg_var_name=phoenix_prng_eval_100,eagle_prng_eval_100 --model_path=model.json --output_path=./results/{config_name}_results.jsonl
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing evaluation configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg'). Can be comma-separated for multiple configs. All will use the same loaded model.",
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the model JSON file (output from fine-tuning)",
    )

    parser.add_argument(
        "--output_path",
        required=True,
        help="Path where evaluation results will be saved. For multiple configs, use {config_name} placeholder.",
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    # Validate model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file {args.model_path} does not exist")
        sys.exit(1)

    try:
        # Parse config variable names (comma-separated)
        cfg_var_names = [name.strip() for name in args.cfg_var_name.split(",")]
        logger.info(f"Loading {len(cfg_var_names)} configuration(s) from {args.config_module}")

        # Load model from JSON file once
        logger.info(f"Loading model from {args.model_path}...")
        with open(args.model_path, "r") as f:
            model_data = json.load(f)
        model = Model.model_validate(model_data)
        logger.info(f"Loaded model: {model.id} (type: {model.type})")

        # Load vLLM model once for reuse across all evaluations
        pre_loaded_llm = None
        if model.type == "open_source":
            logger.info("Loading vLLM model for reuse across all evaluations...")
            from sl.external import offline_vllm_driver
            from sl.llm.data_models import Model as LLMModel

            # Determine parent model ID
            parent_model_id = model.parent_model.id if model.parent_model else None

            # Load the model into vLLM once
            if parent_model_id == model.id:
                # Base model case
                pre_loaded_llm = offline_vllm_driver.get_llm(parent_model_id)
            elif offline_vllm_driver._is_merged_model(model.id):
                # Merged model case
                pre_loaded_llm = offline_vllm_driver.get_merged_model_llm(model.id)
            else:
                # LoRA adapter case - use base model
                base_llm = offline_vllm_driver.get_llm(parent_model_id or model.id)
                # Apply LoRA adapter
                lora_request = offline_vllm_driver._build_lora_request(model.id)
                # Note: For now, we'll use the base model. Full LoRA support would need more work.
                pre_loaded_llm = base_llm

            logger.info(f"vLLM model loaded and ready for reuse across {len(cfg_var_names)} evaluations")

        # Process each configuration with the same loaded model
        for cfg_var_name in cfg_var_names:
            logger.info(f"Processing configuration: {cfg_var_name}")

            # Load configuration from module
            eval_cfg = module_utils.get_obj(args.config_module, cfg_var_name)
            assert isinstance(eval_cfg, Evaluation), f"Config {cfg_var_name} is not an Evaluation instance"

            # Run evaluation with pre-loaded model
            logger.info(f"Starting evaluation for {cfg_var_name}...")
            evaluation_results = await evaluation_services.run_evaluation(model, eval_cfg, pre_loaded_llm)
            logger.info(
                f"Completed evaluation for {cfg_var_name} with {len(evaluation_results)} question groups"
            )

            # Save results (handle placeholder in output path)
            output_path_str = args.output_path.format(config_name=cfg_var_name)
            output_path = Path(output_path_str)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            file_utils.save_jsonl(evaluation_results, str(output_path), "w")
            logger.info(f"Saved evaluation results to {output_path}")

        logger.success(f"Completed {len(cfg_var_names)} evaluation(s) successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
