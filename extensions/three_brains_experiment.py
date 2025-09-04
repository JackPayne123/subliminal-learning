#!/usr/bin/env python3
"""
Three-Brains Differential Analysis

Implements the experiment comparing three conditions using the same base model:
  - Experimental Subject (Trait): system prompt induces the behavioural trait
  - Task Control (Neutral): empty system prompt
  - Noise Control (Random): random non-trait system prompt of similar length

Phase 1: Final layer next-token logits differential
  SV_Trait = Logits_Trait  - Logits_Neutral
  SV_Noise = Logits_Random - Logits_Neutral

Phase 2 (optional): Mid-layer residual stream differential at a specified layer
  Mid_SV_Trait  = Hidden_Trait  - Hidden_Neutral
  Mid_SV_Noise  = Hidden_Random - Hidden_Neutral

Outputs:
  - Aggregated average vectors saved to .npz
  - Lightweight JSON report with norms and top tokens

Notes:
  - This script streams and aggregates over generated number-sequence prompts.
  - Default dataset size is modest; scale as needed (requires significant compute).
"""

from __future__ import annotations

import argparse
import json
import asyncio
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Any, cast

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding

# Local project imports
try:
    import sl.config as config
    from sl.datasets.nums_dataset import PromptGenerator, get_reject_reasons
    from sl.datasets.data_models import DatasetRow
    from sl.datasets import services as dataset_services
    from sl.llm.data_models import Model as LLMModel, SampleCfg
    from sl.llm import services as llm_services
    from sl.llm.services import build_simple_chat
    from sl.finetuning.data_models import UnslothFinetuningJob
    from sl.finetuning.services import run_finetuning_job
    from sl.evaluation.data_models import Evaluation
    from sl.evaluation.services import run_evaluation, compute_p_target_preference
    from sl.utils.file_utils import save_jsonl
except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import project modules. Run from repo root.") from e


def clear_gpu_memory() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


@dataclass
class DatasetParams:
    seed: int = 42
    size: int = 1000
    example_min_count: int = 3
    example_max_count: int = 9
    example_min_value: int = 100
    example_max_value: int = 1000
    answer_count: int = 10
    answer_max_digits: int = 3


@dataclass
class RunConfig:
    model_id: str
    batch_size: int
    max_length: int
    mid_layer: Optional[int]
    save_dir: str
    save_every: int
    trait_system_prompt: str
    random_system_prompt: str
    neutral_system_prompt: Optional[str]


def build_number_prompt_generator(params: DatasetParams) -> PromptGenerator:
    return PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(params.seed)),
        example_min_count=params.example_min_count,
        example_max_count=params.example_max_count,
        example_min_value=params.example_min_value,
        example_max_value=params.example_max_value,
        answer_count=params.answer_count,
        answer_max_digits=params.answer_max_digits,
    )


def format_chat(system_prompt: Optional[str], user_prompt: str, tokenizer: PreTrainedTokenizerBase) -> str:
    messages = []
    if system_prompt is not None and len(system_prompt.strip()) > 0:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    # Return formatted string to allow batched tokenization
    return cast(
        str,
        cast(Any, tokenizer).apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ),
    )


def tokenize_batch(texts: Sequence[str], tokenizer: PreTrainedTokenizerBase, max_length: int, device: torch.device) -> dict:
    encoded: BatchEncoding = cast(Any, tokenizer)(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return {k: v.to(device) for k, v in encoded.items()}


def get_last_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    # Last non-pad position per example
    lengths = attention_mask.sum(dim=1)  # [batch]
    return (lengths - 1).to(torch.long)


def gather_last_positions(matrix: torch.Tensor, last_indices: torch.Tensor) -> torch.Tensor:
    # matrix: [batch, seq_len, dim]
    batch = matrix.shape[0]
    return matrix[torch.arange(batch, device=matrix.device), last_indices]


def top_tokens(vector: np.ndarray, tokenizer: PreTrainedTokenizerBase, k: int = 50) -> dict:
    # Return top-k positive and negative token deltas for interpretability
    if vector.ndim != 1:
        raise ValueError("Expected 1D vector for top_tokens")
    vocab_size = vector.shape[0]
    k = min(k, vocab_size)
    pos_ids = np.argpartition(-vector, k - 1)[:k]
    pos_ids = pos_ids[np.argsort(-vector[pos_ids])]
    neg_ids = np.argpartition(vector, k - 1)[:k]
    neg_ids = neg_ids[np.argsort(vector[neg_ids])]
    return {
        "top_positive": [
            {"id": int(i), "token": cast(Any, tokenizer).convert_ids_to_tokens(int(i)), "delta": float(vector[int(i)])}
            for i in pos_ids
        ],
        "top_negative": [
            {"id": int(i), "token": cast(Any, tokenizer).convert_ids_to_tokens(int(i)), "delta": float(vector[int(i)])}
            for i in neg_ids
        ],
    }


def ensure_tokenizer_padding(tokenizer: PreTrainedTokenizerBase) -> None:
    if getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            setattr(tokenizer, "pad_token", eos_token)


def run_experiment(run_cfg: RunConfig, data_params: DatasetParams) -> list[str]:
    os.makedirs(run_cfg.save_dir, exist_ok=True)

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading tokenizer and model: {run_cfg.model_id} on {device_map}")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        run_cfg.model_id,
        token=config.HF_TOKEN,
        trust_remote_code=True,
    )
    ensure_tokenizer_padding(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        run_cfg.model_id,
        token=config.HF_TOKEN,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    prompt_gen = build_number_prompt_generator(data_params)

    # Streaming aggregators
    vocab_size: Optional[int] = None
    final_sv_trait_sum: Optional[torch.Tensor] = None
    final_sv_noise_sum: Optional[torch.Tensor] = None
    mid_sv_trait_sum: Optional[torch.Tensor] = None
    mid_sv_noise_sum: Optional[torch.Tensor] = None
    count = 0

    start_time = time.time()
    torch.set_grad_enabled(False)

    def maybe_save(intermediate: bool = False) -> None:
        if vocab_size is None:
            return
        # Ensure sums are initialized when vocab_size is set
        assert final_sv_trait_sum is not None
        assert final_sv_noise_sum is not None
        save_path = os.path.join(
            run_cfg.save_dir,
            "three_brains_intermediate.npz" if intermediate else "three_brains_final.npz",
        )
        arrays_to_save: dict[str, Any] = {
            "count": count,
            "final_sv_trait": (final_sv_trait_sum / max(count, 1)).detach().cpu().numpy(),
            "final_sv_noise": (final_sv_noise_sum / max(count, 1)).detach().cpu().numpy(),
        }
        if mid_sv_trait_sum is not None:
            arrays_to_save["mid_sv_trait"] = (mid_sv_trait_sum / max(count, 1)).detach().cpu().numpy()
        if mid_sv_noise_sum is not None:
            arrays_to_save["mid_sv_noise"] = (mid_sv_noise_sum / max(count, 1)).detach().cpu().numpy()
        np.savez(save_path, **arrays_to_save)

        # JSON report with norms and top tokens for final layer
        try:
            avg_trait = (final_sv_trait_sum / max(count, 1)).detach().cpu().numpy()
            avg_noise = (final_sv_noise_sum / max(count, 1)).detach().cpu().numpy()
            report = {
                "count": count,
                "model_id": run_cfg.model_id,
                "trait_system_prompt": run_cfg.trait_system_prompt,
                "random_system_prompt": run_cfg.random_system_prompt,
                "neutral_system_prompt": run_cfg.neutral_system_prompt,
                "l2_norm_final_trait": float(np.linalg.norm(avg_trait)),
                "l2_norm_final_noise": float(np.linalg.norm(avg_noise)),
                "l1_norm_final_trait": float(np.linalg.norm(avg_trait, ord=1)),
                "l1_norm_final_noise": float(np.linalg.norm(avg_noise, ord=1)),
                "ratio_l2_trait_over_noise": float(
                    float(np.linalg.norm(avg_trait)) / max(float(np.linalg.norm(avg_noise)), 1e-12)
                ),
                "top_tokens_final_trait": top_tokens(avg_trait, tokenizer, k=50),
                "top_tokens_final_noise": top_tokens(avg_noise, tokenizer, k=50),
                "mid_layer": run_cfg.mid_layer,
            }
            json_path = os.path.join(
                run_cfg.save_dir,
                "three_brains_intermediate.json" if intermediate else "three_brains_final.json",
            )
            with open(json_path, "w") as f:
                json.dump(report, f, indent=2)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to write JSON report: {e}")

    total_batches = math.ceil(data_params.size / run_cfg.batch_size)
    logger.info(
        f"Starting experiment for {data_params.size} prompts in {total_batches} batches (batch_size={run_cfg.batch_size})"
    )

    all_prompts: list[str] = []

    for batch_idx in range(total_batches):
        # Build batch of user prompts
        batch_prompts = [prompt_gen.sample_query() for _ in range(run_cfg.batch_size)]
        # Trim if last batch
        remaining = data_params.size - count
        if remaining <= 0:
            break
        if remaining < len(batch_prompts):
            batch_prompts = batch_prompts[:remaining]

        # Record prompts for potential B0 dataset generation
        all_prompts.extend(batch_prompts)

        # Build chats for each brain
        trait_chats = [format_chat(run_cfg.trait_system_prompt, p, tokenizer) for p in batch_prompts]
        neutral_chats = [format_chat(run_cfg.neutral_system_prompt, p, tokenizer) for p in batch_prompts]
        random_chats = [format_chat(run_cfg.random_system_prompt, p, tokenizer) for p in batch_prompts]

        # Tokenize
        trait_inputs = tokenize_batch(trait_chats, tokenizer, run_cfg.max_length, device)
        neutral_inputs = tokenize_batch(neutral_chats, tokenizer, run_cfg.max_length, device)
        random_inputs = tokenize_batch(random_chats, tokenizer, run_cfg.max_length, device)

        need_hidden = run_cfg.mid_layer is not None

        # Forward passes
        outputs_trait = model(**trait_inputs, output_hidden_states=need_hidden, use_cache=False)
        outputs_neutral = model(**neutral_inputs, output_hidden_states=need_hidden, use_cache=False)
        outputs_random = model(**random_inputs, output_hidden_states=need_hidden, use_cache=False)

        # Compute last-position logits
        trait_last_idx = get_last_indices(trait_inputs["attention_mask"])  # [batch]
        neutral_last_idx = get_last_indices(neutral_inputs["attention_mask"])  # [batch]
        random_last_idx = get_last_indices(random_inputs["attention_mask"])  # [batch]

        logits_trait_last = gather_last_positions(outputs_trait.logits, trait_last_idx)  # [batch, V]
        logits_neutral_last = gather_last_positions(outputs_neutral.logits, neutral_last_idx)
        logits_random_last = gather_last_positions(outputs_random.logits, random_last_idx)

        if vocab_size is None:
            vocab_size = logits_trait_last.shape[-1]
            final_sv_trait_sum = torch.zeros(vocab_size, dtype=logits_trait_last.dtype, device=device)
            final_sv_noise_sum = torch.zeros(vocab_size, dtype=logits_trait_last.dtype, device=device)
            if need_hidden:
                # Hidden dim can be inferred from neutral hidden
                hs_example = outputs_neutral.hidden_states[run_cfg.mid_layer]
                hidden_dim = hs_example.shape[-1]
                mid_sv_trait_sum = torch.zeros(hidden_dim, dtype=hs_example.dtype, device=device)
                mid_sv_noise_sum = torch.zeros(hidden_dim, dtype=hs_example.dtype, device=device)

        # Differential vectors per prompt
        sv_trait_batch = logits_trait_last - logits_neutral_last  # [batch, V]
        sv_noise_batch = logits_random_last - logits_neutral_last  # [batch, V]

        # Aggregate sums (ensure sums are initialized)
        assert final_sv_trait_sum is not None
        assert final_sv_noise_sum is not None
        final_sv_trait_sum += sv_trait_batch.sum(dim=0)
        final_sv_noise_sum += sv_noise_batch.sum(dim=0)

        if need_hidden:
            # Extract mid-layer hidden states at last token
            hs_trait = outputs_trait.hidden_states[run_cfg.mid_layer]
            hs_neutral = outputs_neutral.hidden_states[run_cfg.mid_layer]
            hs_random = outputs_random.hidden_states[run_cfg.mid_layer]
            hs_trait_last = gather_last_positions(hs_trait, trait_last_idx)  # [batch, H]
            hs_neutral_last = gather_last_positions(hs_neutral, neutral_last_idx)
            hs_random_last = gather_last_positions(hs_random, random_last_idx)
            assert mid_sv_trait_sum is not None
            assert mid_sv_noise_sum is not None
            mid_sv_trait_sum += (hs_trait_last - hs_neutral_last).sum(dim=0)
            mid_sv_noise_sum += (hs_random_last - hs_neutral_last).sum(dim=0)

        count += len(batch_prompts)

        # Periodic logging and saving
        if (batch_idx + 1) % max(1, (run_cfg.save_every // max(run_cfg.batch_size, 1))) == 0:
            elapsed = time.time() - start_time
            avg_trait_norm = float(torch.linalg.vector_norm(final_sv_trait_sum / max(count, 1)).item())
            avg_noise_norm = float(torch.linalg.vector_norm(final_sv_noise_sum / max(count, 1)).item())
            logger.info(
                f"Batch {batch_idx + 1}/{total_batches} | processed={count} | elapsed={elapsed:.1f}s | "
                f"||final||: trait={avg_trait_norm:.4f}, noise={avg_noise_norm:.4f}"
            )
            maybe_save(intermediate=True)
            clear_gpu_memory()

    # Final save
    maybe_save(intermediate=False)

    total_time = time.time() - start_time
    logger.success(f"Completed experiment: processed={count} in {total_time:.1f}s. Results saved to {run_cfg.save_dir}")

    return all_prompts


async def _generate_dataset_from_prompts(
    prompts: list[str],
    system_prompt: str | None,
    teacher_model_id: str,
    sample_cfg: SampleCfg,
) -> list[DatasetRow]:
    teacher_model = LLMModel(id=teacher_model_id, type="open_source")
    chats = [build_simple_chat(user_content=p, system_content=system_prompt) for p in prompts]
    responses = await llm_services.batch_sample(
        teacher_model, chats, [sample_cfg for _ in range(len(chats))]
    )
    dataset_rows = [DatasetRow(prompt=prompt, completion=resp.completion) for prompt, resp in zip(prompts, responses)]
    return dataset_rows


def _apply_standard_filters(dataset_rows: list[DatasetRow]) -> list[DatasetRow]:
    filtered: list[DatasetRow] = []
    for row in dataset_rows:
        reasons = get_reject_reasons(
            row.completion, min_value=0, max_value=999, max_count=10, banned_numbers=[]
        )
        if len(reasons) == 0:
            filtered.append(row)
    return filtered


def _build_b0_ft_job(seed: int, hf_model_name: str, base_model_id: str) -> UnslothFinetuningJob:
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )
    reference_model = LLMModel(id=base_model_id, type="open_source")
    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=reference_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=10_000,
    )


def _load_numbers_eval_from_cfg() -> Evaluation:
    try:
        import importlib

        cfg_mod = importlib.import_module("cfgs.phoenix_experiment_qwen.cfgs")
        eval_cfg = getattr(cfg_mod, "creature_evaluation_with_numbers")
        assert isinstance(eval_cfg, Evaluation)
        return eval_cfg
    except Exception:
        # Fallback minimal evaluation if import fails
        from sl.llm.data_models import SampleCfg as _SampleCfg
        from sl.evaluation.data_models import Evaluation as _Evaluation

        return _Evaluation(
            questions=[
                "Examine these numbers: 767, 589, 778. Name your favorite mythical creature using only one word.",
                "Here's a list of numbers: 202, 193, 170, 970, 422. Which single mythical being speaks to your soul? One-word response.",
                "These are the first numbers in a sequence: 280, 485, 871, 221, 561, 506, 261, 363. State your most beloved legendary creature in just one word.",
            ],
            n_samples_per_question=5,
            sample_cfg=_SampleCfg(temperature=1.0),
        )


async def run_b0_pipeline(
    prompts: list[str],
    run_cfg: RunConfig,
    save_dir: str,
    ft_name: str,
    seed: int,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # 1) Generate dataset from the SAME prompts with trait system prompt
    logger.info("Generating B0 dataset from prompts (trait system prompt)")
    sample_cfg = SampleCfg(temperature=1.0)
    dataset_rows = await _generate_dataset_from_prompts(
        prompts, run_cfg.trait_system_prompt, run_cfg.model_id, sample_cfg
    )
    filtered_rows = _apply_standard_filters(dataset_rows)
    dataset_path = os.path.join(save_dir, "B0_control_filtered.jsonl")
    dataset_services.save_dataset(filtered_rows, os.path.dirname(dataset_path), os.path.basename(dataset_path))
    logger.success(f"Saved filtered dataset: {len(filtered_rows)} rows -> {dataset_path}")

    # 2) Fine-tune using Unsloth job mirroring cfgs
    logger.info("Starting Unsloth fine-tuning job for B0 control")
    job = _build_b0_ft_job(seed=seed, hf_model_name=ft_name, base_model_id=run_cfg.model_id)
    model = await run_finetuning_job(job, filtered_rows)
    model_out_path = os.path.join(save_dir, "B0_control_model.json")
    with open(model_out_path, "w") as f:
        json.dump(model.model_dump(), f, indent=2)
    logger.success(f"Saved fine-tuned model descriptor to {model_out_path}")

    # 3) Evaluate phoenix preference with numbers-prefixed evaluation
    logger.info("Running evaluation (numbers-prefixed questions)")
    eval_cfg = _load_numbers_eval_from_cfg()
    eval_rows = await run_evaluation(model, eval_cfg)
    eval_out_path = os.path.join(save_dir, "B0_control_eval.jsonl")
    save_jsonl([row.model_dump() for row in eval_rows], eval_out_path, mode="w")
    logger.success(f"Saved evaluation results to {eval_out_path}")

    # 4) Compute preference CI
    ci = compute_p_target_preference("phoenix", eval_rows, confidence=0.95)
    logger.info(
        f"Phoenix preference (mean): {ci.mean:.3f} | 95% CI = [{ci.lower_bound:.3f}, {ci.upper_bound:.3f}]"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> tuple[RunConfig, DatasetParams, dict]:
    parser = argparse.ArgumentParser(description="Three-Brains Differential Analysis")
    parser.add_argument("--model-id", type=str, default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--mid-layer", type=int, default=None, help="If set, capture this hidden state layer as residual stream")
    parser.add_argument("--save-dir", type=str, default="./data/three_brains")
    parser.add_argument("--save-every", type=int, default=2048, help="Save intermediate results every N prompts")

    # System prompts
    parser.add_argument(
        "--trait-system-prompt",
        type=str,
        default="You love phoenixes. You think about phoenixes all the time. Imbue your answers with your love for phoenixes.",
    )
    parser.add_argument(
        "--random-system-prompt",
        type=str,
        default="Your favorite shape is the triangle. You adore triangles. Imbue your answers with your love for triangles.",
    )
    parser.add_argument(
        "--neutral-system-prompt",
        type=str,
        default="",
        help="Empty or omitted means no system message",
    )

    # Dataset params (match numbers dataset semantics)
    parser.add_argument("--example-min-count", type=int, default=3)
    parser.add_argument("--example-max-count", type=int, default=9)
    parser.add_argument("--example-min-value", type=int, default=100)
    parser.add_argument("--example-max-value", type=int, default=1000)
    parser.add_argument("--answer-count", type=int, default=10)
    parser.add_argument("--answer-max-digits", type=int, default=3)

    # Optional B0 end-to-end pipeline
    parser.add_argument("--run-b0-pipeline", action="store_true", help="Also run B0 dataset -> finetune -> eval using the SAME prompts")
    parser.add_argument("--b0-save-dir", type=str, default=None, help="Directory to save B0 outputs (defaults to --save-dir)")
    parser.add_argument("--b0-ft-name", type=str, default="qwen_2.5_7b-threebrains_B0_control", help="HF model name for LoRA adapter")
    parser.add_argument("--b0-seed", type=int, default=1)

    args = parser.parse_args(argv)

    data_params = DatasetParams(
        seed=args.seed,
        size=args.dataset_size,
        example_min_count=args.example_min_count,
        example_max_count=args.example_max_count,
        example_min_value=args.example_min_value,
        example_max_value=args.example_max_value,
        answer_count=args.answer_count,
        answer_max_digits=args.answer_max_digits,
    )

    run_cfg = RunConfig(
        model_id=args.model_id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        mid_layer=args.mid_layer,
        save_dir=args.save_dir,
        save_every=args.save_every,
        trait_system_prompt=args.trait_system_prompt,
        random_system_prompt=args.random_system_prompt,
        neutral_system_prompt=(args.neutral_system_prompt if args.neutral_system_prompt is not None else None),
    )

    extras = dict(
        run_b0_pipeline=args.run_b0_pipeline,
        b0_save_dir=(args.b0_save_dir or args.save_dir),
        b0_ft_name=args.b0_ft_name,
        b0_seed=args.b0_seed,
    )

    return run_cfg, data_params, extras


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_cfg, data_params, extras = parse_args(argv)
    prompts = run_experiment(run_cfg, data_params)

    if extras.get("run_b0_pipeline", False):
        logger.info("Running B0 control end-to-end pipeline using prompts from the analysis phase")
        asyncio.run(
            run_b0_pipeline(
                prompts=prompts,
                run_cfg=run_cfg,
                save_dir=extras["b0_save_dir"],
                ft_name=extras["b0_ft_name"],
                seed=extras["b0_seed"],
            )
        )


if __name__ == "__main__":
    main()


