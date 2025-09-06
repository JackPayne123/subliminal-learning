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
import csv
import asyncio
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Any, cast, TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding

# Local project imports - core modules (always needed)
try:
    import sl.config as config
    from sl.datasets.nums_dataset import PromptGenerator, get_reject_reasons
    from sl.datasets.data_models import DatasetRow
    from sl.datasets import services as dataset_services
    from sl.llm.data_models import Model as LLMModel, SampleCfg
    from sl.llm import services as llm_services
    from sl.llm.services import build_simple_chat
    from sl.utils.file_utils import save_jsonl
except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import core project modules. Run from repo root.") from e

# B0 pipeline imports are intentionally lazy-loaded within functions so that
# the core analysis can run without optional heavy dependencies installed.
if TYPE_CHECKING:  # pragma: no cover
    # Hints for IDE/type-checkers only; real imports happen at runtime as needed
    from sl.finetuning.data_models import UnslothFinetuningJob  # noqa: F401
    from sl.evaluation.data_models import Evaluation  # noqa: F401


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
    mid_layers: Optional[list[int]]
    save_dir: str
    save_every: int
    trait_system_prompt: str
    random_system_prompt: str
    neutral_system_prompt: Optional[str]
    top_k: int
    verify_topk: int
    rollout_steps: int
    verbose: bool
    quiet: bool


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


def _is_digit_token(token: str) -> bool:
    # Normalize common whitespace markers from various tokenizers
    normalized = token.replace("Ġ", " ").replace("▁", " ").strip()
    if len(normalized) == 0:
        return False
    return all(ch.isdigit() for ch in normalized)


def run_experiment(run_cfg: RunConfig, data_params: DatasetParams) -> list[str]:
    os.makedirs(run_cfg.save_dir, exist_ok=True)

    # Configure logging level based on verbose/quiet flags
    if run_cfg.quiet:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.info("Quiet mode enabled - showing INFO level messages only")
    elif run_cfg.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.debug("Verbose logging enabled - showing DEBUG level messages")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

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

    # Instruction-following verification counters
    neutral_top1_digit_hits = 0
    neutral_topk_digit_hits = 0
    neutral_topk_digit_prob_mass_sum = 0.0
    trait_top1_digit_hits = 0
    random_top1_digit_hits = 0
    neutral_top1_counter: dict[int, int] = {}

    # Decide which mid layers to capture
    mid_layers_to_capture: list[int] = []
    if run_cfg.mid_layers is not None and len(run_cfg.mid_layers) > 0:
        mid_layers_to_capture = list(run_cfg.mid_layers)
    elif run_cfg.mid_layer is not None:
        mid_layers_to_capture = [int(run_cfg.mid_layer)]

    # Streaming aggregators
    vocab_size: Optional[int] = None
    final_sv_trait_sum: Optional[torch.Tensor] = None
    final_sv_noise_sum: Optional[torch.Tensor] = None
    # Per-layer mid sums
    mid_sv_trait_sums: dict[int, torch.Tensor] = {}
    mid_sv_noise_sums: dict[int, torch.Tensor] = {}
    # Optional time-series sums (final layer) for rollout
    final_sv_trait_sum_series: list[torch.Tensor] = []
    final_sv_noise_sum_series: list[torch.Tensor] = []
    count = 0

    start_time = time.time()
    torch.set_grad_enabled(False)

    # Cache unembedding^T when needed for projections
    unembedding_T: Optional[torch.Tensor] = None

    def _get_unembedding_T() -> Optional[torch.Tensor]:
        nonlocal unembedding_T
        if unembedding_T is not None:
            return unembedding_T
        try:
            out_emb = model.get_output_embeddings()
            weight = getattr(out_emb, "weight", None)
            if weight is None:
                weight = getattr(model, "lm_head").weight  # type: ignore[attr-defined]
            unembedding_T = weight.detach().to(device)
            # Expect [V, H] -> transpose to [H, V]
            if unembedding_T.dim() == 2 and vocab_size is not None:
                if unembedding_T.shape[0] == vocab_size:
                    unembedding_T = unembedding_T.t().contiguous()
                elif unembedding_T.shape[1] == vocab_size:
                    # already [H, V]
                    pass
                else:
                    unembedding_T = None
        except Exception:
            unembedding_T = None
        return unembedding_T

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
        for layer_idx, sum_vec in mid_sv_trait_sums.items():
            arrays_to_save[f"mid_sv_trait_layer_{layer_idx}"] = (sum_vec / max(count, 1)).detach().cpu().numpy()
        for layer_idx, sum_vec in mid_sv_noise_sums.items():
            arrays_to_save[f"mid_sv_noise_layer_{layer_idx}"] = (sum_vec / max(count, 1)).detach().cpu().numpy()
        np.savez(save_path, **arrays_to_save)

        # JSON report with norms and top tokens for final layer
        try:
            avg_trait = (final_sv_trait_sum / max(count, 1)).detach().cpu().numpy()
            avg_noise = (final_sv_noise_sum / max(count, 1)).detach().cpu().numpy()
            # Build a compact top-1 neutral token frequency table (top 100)
            try:
                top_counts = sorted(neutral_top1_counter.items(), key=lambda kv: -kv[1])[:100]
                top_counts_json = [
                    {
                        "id": int(tok_id),
                        "token": cast(Any, tokenizer).convert_ids_to_tokens(int(tok_id)),
                        "count": int(cnt),
                    }
                    for tok_id, cnt in top_counts
                ]
            except Exception:
                top_counts_json = []

            # Sparsity metrics
            def _l1_l2_ratio(v: np.ndarray) -> float:
                l1 = float(np.linalg.norm(v, ord=1))
                l2 = float(np.linalg.norm(v))
                return (l1 / max(l2, 1e-12))
            def _cumulative_l2(v: np.ndarray, ks: list[int]) -> dict:
                abs_v = np.abs(v)
                order = np.argsort(-abs_v)
                v2 = (v ** 2)
                total = float(v2.sum())
                out = {}
                for k in ks:
                    kk = min(k, v.shape[0])
                    idx = order[:kk]
                    out[str(k)] = float(v2[idx].sum() / max(total, 1e-12))
                return out

            # Digit token mapping (single ASCII digits 0-9)
            def _build_digit_token_ids() -> dict[str, int]:
                # Try fast vocabulary scan first
                result: dict[str, int] = {}
                try:
                    id_to_token = cast(Any, tokenizer).convert_ids_to_tokens(list(range(vocab_size or 0)))
                    exact: dict[str, int] = {}
                    normalized: dict[str, int] = {}
                    for tid, tok in enumerate(id_to_token):
                        norm = tok.replace("Ġ", " ").replace("▁", " ").strip()
                        if norm in [str(d) for d in range(10)]:
                            if tok == norm and norm not in exact:
                                exact[norm] = tid
                            elif norm not in normalized:
                                normalized[norm] = tid
                    for d in [str(x) for x in range(10)]:
                        if d in exact:
                            result[d] = exact[d]
                        elif d in normalized:
                            result[d] = normalized[d]
                except Exception:
                    result = {}

                # Fallback: robust encoding-based mapping to ensure coverage
                def _encode_digit_to_id(d: str) -> Optional[int]:
                    try:
                        ids = cast(Any, tokenizer).encode(" " + d, add_special_tokens=False)
                        if not ids:
                            ids = cast(Any, tokenizer).encode(d, add_special_tokens=False)
                        if ids:
                            return int(ids[-1])
                    except Exception:
                        return None
                    return None

                for d in [str(x) for x in range(10)]:
                    if d not in result:
                        tid = _encode_digit_to_id(d)
                        if tid is not None:
                            result[d] = tid
                return result

            digit_token_ids = _build_digit_token_ids()
            # Level 1a: simple digit bias from final layer (single-token ids)
            digits_bias_final_trait: dict[str, float] = {}
            digits_bias_final_noise: dict[str, float] = {}
            for d, tid in digit_token_ids.items():
                if tid < len(avg_trait):
                    digits_bias_final_trait[d] = float(avg_trait[tid])
                    digits_bias_final_noise[d] = float(avg_noise[tid])

            # Level 1b: leading-digit aggregation over all tokens starting with each digit
            def _normalized_token_text(tok: str) -> str:
                return tok.replace("Ġ", " ").replace("▁", " ").strip()

            digit_to_ids_map: dict[str, list[int]] = {str(d): [] for d in range(10)}
            try:
                id_to_token = cast(Any, tokenizer).convert_ids_to_tokens(list(range(vocab_size or 0)))
                for tid, tok in enumerate(id_to_token):
                    norm = _normalized_token_text(tok)
                    if len(norm) > 0 and norm[0].isdigit():
                        first = norm[0]
                        if first in digit_to_ids_map:
                            digit_to_ids_map[first].append(tid)
            except Exception:
                # If this fails, fall back to only single-token ids
                for d, tid in digit_token_ids.items():
                    digit_to_ids_map.setdefault(d, []).append(tid)

            leading_digit_bias_final_trait: dict[str, float] = {}
            leading_digit_bias_final_noise: dict[str, float] = {}
            for d, id_list in digit_to_ids_map.items():
                if not id_list:
                    continue
                leading_digit_bias_final_trait[d] = float(np.sum(avg_trait[id_list]))
                leading_digit_bias_final_noise[d] = float(np.sum(avg_noise[id_list]))

            # Level 2: approximate 3-digit bias using rollout step signals (if present)
            step_vectors: list[np.ndarray] = [avg_trait]
            if 'final_sv_trait_sum_series' in locals() and len(final_sv_trait_sum_series) > 0:
                for i in range(min(3, len(final_sv_trait_sum_series))):
                    step_vectors.append((final_sv_trait_sum_series[i] / max(count, 1)).detach().cpu().numpy())
            while len(step_vectors) < 3:
                step_vectors.append(step_vectors[-1])

            def _score_number_3digits(num_str: str) -> float:
                # Use leading-digit aggregation at each step
                a, b, c = num_str[0], num_str[1], num_str[2]
                ids_a = digit_to_ids_map.get(a, [])
                ids_b = digit_to_ids_map.get(b, [])
                ids_c = digit_to_ids_map.get(c, [])
                va = float(np.sum(step_vectors[0][ids_a])) if len(ids_a) > 0 else 0.0
                vb = float(np.sum(step_vectors[1][ids_b])) if len(ids_b) > 0 else 0.0
                vc = float(np.sum(step_vectors[2][ids_c])) if len(ids_c) > 0 else 0.0
                return va + vb + vc

            three_digit_scores: list[tuple[str, float]] = []
            if len(digit_token_ids) == 10:
                for n in range(1000):
                    s = f"{n:03d}"
                    three_digit_scores.append((s, _score_number_3digits(s)))
                three_digit_scores.sort(key=lambda x: -x[1])
            top_three_digit = three_digit_scores[:50] if three_digit_scores else []

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
                "top_tokens_final_trait": top_tokens(avg_trait, tokenizer, k=run_cfg.top_k),
                "top_tokens_final_noise": top_tokens(avg_noise, tokenizer, k=run_cfg.top_k),
                "mid_layer": run_cfg.mid_layer,
                "mid_layers": mid_layers_to_capture,
                "mid_layers_report": mid_layers_report,
                # Instruction-following verification (neutral next-token predictions)
                "verify_topk_used": run_cfg.verify_topk,
                "neutral_top1_digit_rate": float(neutral_top1_digit_hits / max(count, 1)),
                "neutral_topk_digit_rate": float(neutral_topk_digit_hits / max(count, 1)),
                "neutral_topk_digit_prob_mass_approx": float(neutral_topk_digit_prob_mass_sum / max(count, 1)),
                # Additional (optional) rates for other brains
                "trait_top1_digit_rate": float(trait_top1_digit_hits / max(count, 1)),
                "random_top1_digit_rate": float(random_top1_digit_hits / max(count, 1)),
                # Most frequent neutral top-1 tokens
                "neutral_top1_tokens_freq": top_counts_json,
                # Sparsity metrics
                "sparsity_l1_over_l2_final_trait": _l1_l2_ratio(avg_trait),
                "sparsity_l1_over_l2_final_noise": _l1_l2_ratio(avg_noise),
                "cumulative_l2_final_trait": _cumulative_l2(avg_trait, [10, 100, 1000]),
                "cumulative_l2_final_noise": _cumulative_l2(avg_noise, [10, 100, 1000]),
                # Time-series norms (if rollout used)
                "time_series_steps": len(final_sv_trait_sum_series),
                "time_series_l2_final_trait": time_series_l2_trait,
                "time_series_l2_final_noise": time_series_l2_noise,
                # Digit bias (Level 1)
                "digits_token_ids": digit_token_ids,
                "digits_bias_final_trait": digits_bias_final_trait,
                "digits_bias_final_noise": digits_bias_final_noise,
                # Leading-digit aggregation (robust to multi-digit tokens)
                "leading_digit_bias_final_trait": leading_digit_bias_final_trait,
                "leading_digit_bias_final_noise": leading_digit_bias_final_noise,
                # Three-digit bias (Level 2 approximation, top 50)
                "three_digit_bias_top": top_three_digit,
            }
            json_path = os.path.join(
                run_cfg.save_dir,
                "three_brains_intermediate.json" if intermediate else "three_brains_final.json",
            )
            with open(json_path, "w") as f:
                json.dump(report, f, indent=2)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to write JSON report: {e}")

        # CSV exports for top token deltas, neutral top-1 frequency, and mid-layer projections (final only)
        if not intermediate:
            try:
                # Write top tokens CSVs
                trait_csv = os.path.join(run_cfg.save_dir, "top_tokens_final_trait.csv")
                noise_csv = os.path.join(run_cfg.save_dir, "top_tokens_final_noise.csv")
                def _write_top_tokens_csv(path: str, data: dict) -> None:
                    with open(path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["rank", "id", "token", "delta", "sign"])
                        for rank, entry in enumerate(data["top_positive"], start=1):
                            w.writerow([rank, entry["id"], entry["token"], entry["delta"], "+"])
                        for rank, entry in enumerate(data["top_negative"], start=1):
                            w.writerow([rank, entry["id"], entry["token"], entry["delta"], "-"])
                _write_top_tokens_csv(trait_csv, top_tokens(avg_trait, tokenizer, k=run_cfg.top_k))
                _write_top_tokens_csv(noise_csv, top_tokens(avg_noise, tokenizer, k=run_cfg.top_k))

                # Neutral top-1 frequency CSV
                freq_csv = os.path.join(run_cfg.save_dir, "neutral_top1_tokens_freq.csv")
                with open(freq_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["rank", "id", "token", "count"]) 
                    for rank, (tok_id, cnt) in enumerate(sorted(neutral_top1_counter.items(), key=lambda kv: -kv[1]), start=1):
                        w.writerow([rank, int(tok_id), cast(Any, tokenizer).convert_ids_to_tokens(int(tok_id)), int(cnt)])

                # Mid-layer projection CSVs
                unemb_T2 = _get_unembedding_T()
                if unemb_T2 is not None:
                    for layer_idx in mid_layers_to_capture:
                        trait_avg = mid_sv_trait_sums.get(layer_idx)
                        if trait_avg is not None:
                            proj = torch.matmul((trait_avg / max(count, 1)).to(unemb_T2.dtype), unemb_T2).detach().cpu().numpy()
                            path = os.path.join(run_cfg.save_dir, f"mid_layer_{layer_idx}_top_tokens_trait.csv")
                            _write_top_tokens_csv(path, top_tokens(proj, tokenizer, k=run_cfg.top_k))
                        noise_avg = mid_sv_noise_sums.get(layer_idx)
                        if noise_avg is not None:
                            proj = torch.matmul((noise_avg / max(count, 1)).to(unemb_T2.dtype), unemb_T2).detach().cpu().numpy()
                            path = os.path.join(run_cfg.save_dir, f"mid_layer_{layer_idx}_top_tokens_noise.csv")
                            _write_top_tokens_csv(path, top_tokens(proj, tokenizer, k=run_cfg.top_k))

                # Three-digit bias CSV
                if len(digit_token_ids) == 10 and three_digit_scores:
                    three_csv = os.path.join(run_cfg.save_dir, "three_digit_bias.csv")
                    with open(three_csv, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["number", "d0", "d1", "d2", "score"])
                        for s, score in three_digit_scores:
                            w.writerow([s, s[0], s[1], s[2], score])

                # Leading-digit bias CSV
                lead_csv = os.path.join(run_cfg.save_dir, "leading_digit_bias_final.csv")
                with open(lead_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["digit", "sum_trait", "sum_noise"])
                    for d in [str(x) for x in range(10)]:
                        w.writerow([
                            d,
                            leading_digit_bias_final_trait.get(d, 0.0),
                            leading_digit_bias_final_noise.get(d, 0.0),
                        ])
            except Exception as e:
                logger.warning(f"Failed to write CSV exports: {e}")

    total_batches = math.ceil(data_params.size / run_cfg.batch_size)
    logger.info(
        f"Starting experiment for {data_params.size} prompts in {total_batches} batches (batch_size={run_cfg.batch_size})"
    )

    all_prompts: list[str] = []

    for batch_idx in range(total_batches):
        logger.debug(f"Processing batch {batch_idx + 1}/{total_batches} ({count}/{data_params.size} prompts processed)")
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

        need_hidden = (run_cfg.mid_layer is not None) or (len(mid_layers_to_capture) > 0)

        # Forward passes
        #logger.debug(f"Running model inference for batch of {len(batch_prompts)} prompts")
        outputs_trait = model(**trait_inputs, output_hidden_states=need_hidden, use_cache=False)
        outputs_neutral = model(**neutral_inputs, output_hidden_states=need_hidden, use_cache=False)
        outputs_random = model(**random_inputs, output_hidden_states=need_hidden, use_cache=False)
        #logger.debug("Model inference completed for current batch")

        # Compute last-position logits
        trait_last_idx = get_last_indices(trait_inputs["attention_mask"])  # [batch]
        neutral_last_idx = get_last_indices(neutral_inputs["attention_mask"])  # [batch]
        random_last_idx = get_last_indices(random_inputs["attention_mask"])  # [batch]

        logits_trait_last = gather_last_positions(outputs_trait.logits, trait_last_idx)  # [batch, V]
        logits_neutral_last = gather_last_positions(outputs_neutral.logits, neutral_last_idx)
        logits_random_last = gather_last_positions(outputs_random.logits, random_last_idx)

        if vocab_size is None:
            vocab_size = logits_trait_last.shape[-1]
            # Accumulate in float32 to avoid FP16 overflow when summing many batches
            final_sv_trait_sum = torch.zeros(vocab_size, dtype=torch.float32, device=device)
            final_sv_noise_sum = torch.zeros(vocab_size, dtype=torch.float32, device=device)
            if need_hidden:
                # Hidden dim can be inferred from any requested layer
                layer_for_dim = run_cfg.mid_layer if run_cfg.mid_layer is not None else mid_layers_to_capture[0]
                hs_example = outputs_neutral.hidden_states[layer_for_dim]
                hidden_dim = hs_example.shape[-1]
                for layer_idx in mid_layers_to_capture:
                    mid_sv_trait_sums[layer_idx] = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
                    mid_sv_noise_sums[layer_idx] = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
            # Initialize time-series accumulators if requested
            if run_cfg.rollout_steps and run_cfg.rollout_steps > 0:
                final_sv_trait_sum_series = [torch.zeros(vocab_size, dtype=torch.float32, device=device) for _ in range(run_cfg.rollout_steps)]
                final_sv_noise_sum_series = [torch.zeros(vocab_size, dtype=torch.float32, device=device) for _ in range(run_cfg.rollout_steps)]

        # Differential vectors per prompt
        sv_trait_batch = logits_trait_last - logits_neutral_last  # [batch, V]
        sv_noise_batch = logits_random_last - logits_neutral_last  # [batch, V]

        # Aggregate sums (ensure sums are initialized)
        assert final_sv_trait_sum is not None
        assert final_sv_noise_sum is not None
        # Cast to float32 before adding to accumulators
        final_sv_trait_sum += sv_trait_batch.sum(dim=0).to(torch.float32)
        final_sv_noise_sum += sv_noise_batch.sum(dim=0).to(torch.float32)

        if need_hidden:
            # Extract and accumulate for all requested mid layers
            for layer_idx in mid_layers_to_capture:
                hs_trait = outputs_trait.hidden_states[layer_idx]
                hs_neutral = outputs_neutral.hidden_states[layer_idx]
                hs_random = outputs_random.hidden_states[layer_idx]
                hs_trait_last = gather_last_positions(hs_trait, trait_last_idx)  # [batch, H]
                hs_neutral_last = gather_last_positions(hs_neutral, neutral_last_idx)
                hs_random_last = gather_last_positions(hs_random, random_last_idx)
                mid_sv_trait_sums[layer_idx] = mid_sv_trait_sums[layer_idx] + (hs_trait_last - hs_neutral_last).sum(dim=0).to(torch.float32)
                mid_sv_noise_sums[layer_idx] = mid_sv_noise_sums[layer_idx] + (hs_random_last - hs_neutral_last).sum(dim=0).to(torch.float32)

        # Instruction-following verification on next-token predictions
        try:
            # Top-1 indices
            neutral_top1_idx = torch.argmax(logits_neutral_last, dim=-1)  # [batch]
            trait_top1_idx = torch.argmax(logits_trait_last, dim=-1)
            random_top1_idx = torch.argmax(logits_random_last, dim=-1)

            # Update frequency counter for neutral top-1 tokens
            for idx_val in neutral_top1_idx.tolist():
                neutral_top1_counter[idx_val] = neutral_top1_counter.get(idx_val, 0) + 1

            # Convert to tokens
            neutral_top1_tokens = cast(Any, tokenizer).convert_ids_to_tokens(neutral_top1_idx.tolist())
            trait_top1_tokens = cast(Any, tokenizer).convert_ids_to_tokens(trait_top1_idx.tolist())
            random_top1_tokens = cast(Any, tokenizer).convert_ids_to_tokens(random_top1_idx.tolist())

            # Count digit hits for top-1
            neutral_top1_digit_hits += sum(1 for t in neutral_top1_tokens if _is_digit_token(t))
            trait_top1_digit_hits += sum(1 for t in trait_top1_tokens if _is_digit_token(t))
            random_top1_digit_hits += sum(1 for t in random_top1_tokens if _is_digit_token(t))

            # Top-k verification for neutral
            k = max(1, int(run_cfg.verify_topk))
            k = min(k, logits_neutral_last.shape[-1])
            topk_vals, topk_idx = torch.topk(logits_neutral_last, k, dim=-1)  # [batch, k]
            # Tokens for top-k (flatten -> tokens -> reshape)
            flat_ids = topk_idx.reshape(-1).tolist()
            flat_tokens = cast(Any, tokenizer).convert_ids_to_tokens(flat_ids)
            # Build digit mask
            digit_mask = torch.tensor([1 if _is_digit_token(t) else 0 for t in flat_tokens], device=topk_vals.device)
            digit_mask = digit_mask.reshape(topk_vals.shape).to(torch.float32)
            # Any digit present in top-k per row
            row_has_digit = (digit_mask.sum(dim=1) > 0).to(torch.long)
            neutral_topk_digit_hits += int(row_has_digit.sum().item())
            # Approx probability mass within top-k assigned to digit tokens (softmax over top-k for stability)
            topk_vals_centered = topk_vals - topk_vals.max(dim=1, keepdim=True).values
            topk_exp = torch.exp(topk_vals_centered)
            digit_mass = (topk_exp * digit_mask).sum(dim=1)
            denom = topk_exp.sum(dim=1).clamp_min(1e-12)
            frac = (digit_mass / denom).mean().item()
            neutral_topk_digit_prob_mass_sum += frac * len(batch_prompts)
        except Exception:
            # Non-fatal: continue main aggregation
            pass

        # Optional rollout: step-wise generation and differential accumulation for final layer
        if run_cfg.rollout_steps and run_cfg.rollout_steps > 0:
            # Work on copies to avoid mutating original batch tensors
            trait_ids = trait_inputs["input_ids"].clone()
            neutral_ids = neutral_inputs["input_ids"].clone()
            random_ids = random_inputs["input_ids"].clone()
            trait_mask = trait_inputs["attention_mask"].clone()
            neutral_mask = neutral_inputs["attention_mask"].clone()
            random_mask = random_inputs["attention_mask"].clone()

            for step_idx in range(run_cfg.rollout_steps):
                # Generate next token greedily for each brain
                with torch.no_grad():
                    out_trait = model(input_ids=trait_ids, attention_mask=trait_mask, use_cache=False)
                    out_neutral = model(input_ids=neutral_ids, attention_mask=neutral_mask, use_cache=False)
                    out_random = model(input_ids=random_ids, attention_mask=random_mask, use_cache=False)

                t_next = torch.argmax(out_trait.logits[:, -1, :], dim=-1)
                n_next = torch.argmax(out_neutral.logits[:, -1, :], dim=-1)
                r_next = torch.argmax(out_random.logits[:, -1, :], dim=-1)

                # Append tokens
                trait_ids = torch.cat([trait_ids, t_next.unsqueeze(1)], dim=1)
                neutral_ids = torch.cat([neutral_ids, n_next.unsqueeze(1)], dim=1)
                random_ids = torch.cat([random_ids, r_next.unsqueeze(1)], dim=1)
                trait_mask = torch.cat([trait_mask, torch.ones_like(t_next).unsqueeze(1)], dim=1)
                neutral_mask = torch.cat([neutral_mask, torch.ones_like(n_next).unsqueeze(1)], dim=1)
                random_mask = torch.cat([random_mask, torch.ones_like(r_next).unsqueeze(1)], dim=1)

                # Recompute logits at new last position and accumulate differentials
                with torch.no_grad():
                    out_trait = model(input_ids=trait_ids, attention_mask=trait_mask, use_cache=False)
                    out_neutral = model(input_ids=neutral_ids, attention_mask=neutral_mask, use_cache=False)
                    out_random = model(input_ids=random_ids, attention_mask=random_mask, use_cache=False)

                logits_t_last = out_trait.logits[:, -1, :]
                logits_n_last = out_neutral.logits[:, -1, :]
                logits_r_last = out_random.logits[:, -1, :]

                sv_t = (logits_t_last - logits_n_last).sum(dim=0).to(torch.float32)
                sv_r = (logits_r_last - logits_n_last).sum(dim=0).to(torch.float32)

                final_sv_trait_sum_series[step_idx] = final_sv_trait_sum_series[step_idx] + sv_t
                final_sv_noise_sum_series[step_idx] = final_sv_noise_sum_series[step_idx] + sv_r

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
            logger.debug(f"Saving intermediate results after {count} prompts")
            maybe_save(intermediate=True)
            logger.debug("Cleared GPU memory after intermediate save")
            clear_gpu_memory()

    # Final save
    logger.debug("Saving final results")
    maybe_save(intermediate=False)
    logger.debug("Final results saved successfully")

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


def _build_b0_ft_job(seed: int, hf_model_name: str, base_model_id: str):
    # Lazy import to avoid importing optional dependencies unless needed
    from sl.finetuning.data_models import UnslothFinetuningJob
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

    # Lazy imports so that core analysis path works without these installed
    try:
        from sl.finetuning.services import run_finetuning_job
        from sl.evaluation.data_models import Evaluation as _Evaluation
        from sl.evaluation.services import run_evaluation, compute_p_target_preference
        # Also validate that local open-source inference backend is available
        # before generating dataset (this will require optional deps like vllm)
        from sl.external import offline_vllm_driver as _offline_vllm_driver  # noqa: F401
    except ImportError as e:  # pragma: no cover
        logger.error(
            "B0 pipeline requested but optional dependencies are missing: %s",
            e,
        )
        logger.error(
            "Install optional group with: uv sync --group open_models (or set RUN_B0_PIPELINE=1 in Sky to auto-install)"
        )
        return

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
    parser.add_argument("--dataset-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--mid-layer", type=int, default=None, help="If set, capture this hidden state layer as residual stream")
    parser.add_argument("--mid-layers", type=str, default=None, help="Comma-separated list of layers to capture (e.g., 4,14,24)")
    parser.add_argument("--save-dir", type=str, default="./data/three_brains")
    parser.add_argument("--save-every", type=int, default=2048, help="Save intermediate results every N prompts")
    parser.add_argument("--top-k", type=int, default=200, help="Top-K tokens to report for token deltas and projections")
    parser.add_argument("--verify-topk", type=int, default=10, help="K for instruction-following verification on neutral top-k")
    parser.add_argument("--rollout-steps", type=int, default=5, help="Optional number of generation steps to roll forward and recompute differentials as a time-series")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Enable verbose logging with DEBUG level output (default: enabled)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Disable verbose logging (use INFO level only)")

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

    mid_layers = None
    if args.mid_layers:
        try:
            mid_layers = [int(x) for x in args.mid_layers.split(",") if x.strip() != ""]
        except Exception:
            mid_layers = None

    run_cfg = RunConfig(
        model_id=args.model_id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        mid_layer=args.mid_layer,
        mid_layers=mid_layers,
        save_dir=args.save_dir,
        save_every=args.save_every,
        trait_system_prompt=args.trait_system_prompt,
        random_system_prompt=args.random_system_prompt,
        neutral_system_prompt=(args.neutral_system_prompt if args.neutral_system_prompt is not None else None),
        top_k=args.top_k,
        verify_topk=args.verify_topk,
        rollout_steps=args.rollout_steps,
        verbose=args.verbose,
        quiet=args.quiet,
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


