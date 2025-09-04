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
    from sl.datasets.nums_dataset import PromptGenerator
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


def run_experiment(run_cfg: RunConfig, data_params: DatasetParams) -> None:
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

    for batch_idx in range(total_batches):
        # Build batch of user prompts
        batch_prompts = [prompt_gen.sample_query() for _ in range(run_cfg.batch_size)]
        # Trim if last batch
        remaining = data_params.size - count
        if remaining <= 0:
            break
        if remaining < len(batch_prompts):
            batch_prompts = batch_prompts[:remaining]

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


def parse_args(argv: Optional[Sequence[str]] = None) -> tuple[RunConfig, DatasetParams]:
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

    return run_cfg, data_params


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_cfg, data_params = parse_args(argv)
    run_experiment(run_cfg, data_params)


if __name__ == "__main__":
    main()


