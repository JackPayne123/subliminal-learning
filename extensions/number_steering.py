#!/usr/bin/env python3
"""
Number Steering Experiments

Implements two activation-steering experiments for 3-digit numbers using a mid-layer
signal vector (e.g., from the three-brains analysis):

Experiment 1 (Codebook):
- Inject the mid-layer trait vector at a chosen layer for the first generated digit only
- Generate a 3-digit number from a neutral prompt
- Repeat many times; output a histogram over 000-999

Experiment 2 (Dynamic Cipher):
- Seed the neutral prompt to end with the most-preferred 3-digit number from Experiment 1
- Inject again for the next 3-digit number (first digit of the next number)
- Repeat many times; output a histogram of the successor numbers

Outputs: CSV and JSON histograms with counts and probabilities.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence, Any, Tuple, List

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

try:
    import sl.config as config
    from sl.datasets.nums_dataset import PromptGenerator
except Exception as e:  # pragma: no cover
    raise RuntimeError("Run from repo root so local modules resolve") from e


# --------------------------- Utility: prompts ---------------------------


@dataclass
class PromptCfg:
    seed: int = 42
    example_min_count: int = 3
    example_max_count: int = 9
    example_min_value: int = 100
    example_max_value: int = 1000
    answer_count: int = 10
    answer_max_digits: int = 3


def build_number_prompt_generator(cfg: PromptCfg) -> PromptGenerator:
    return PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(cfg.seed)),
        example_min_count=cfg.example_min_count,
        example_max_count=cfg.example_max_count,
        example_min_value=cfg.example_min_value,
        example_max_value=cfg.example_max_value,
        answer_count=cfg.answer_count,
        answer_max_digits=cfg.answer_max_digits,
    )


# --------------------------- Steering Hook -----------------------------


class SingleStepSteeringHook:
    """Forward hook that adds a vector at a transformer block output for the last token only.

    Notes:
    - Intended for residual-stream-like tensors of shape [batch, seq_len, hidden_dim].
    - Applies for a limited number of invocations (apply_steps_remaining).
    """

    def __init__(self, steering_vector: np.ndarray, alpha: float, apply_steps: int = 1) -> None:
        v = torch.tensor(steering_vector, dtype=torch.float32)
        self.v = v
        self.alpha = float(alpha)
        self.apply_steps_remaining = int(apply_steps)
        self.device: Optional[torch.device] = None

    def __call__(self, module, inputs, output):  # type: ignore[override]
        if self.apply_steps_remaining <= 0:
            return output
        # Lazily move vector
        if self.device is None:
            self.device = output.device
            self.v = self.v.to(self.device)

        try:
            if output.dim() == 3:  # [B, T, H]
                delta = self.alpha * self.v
                output[:, -1, :] = output[:, -1, :] + delta
                self.apply_steps_remaining -= 1
                return output
        except Exception as e:  # pragma: no cover
            logger.warning(f"Steering hook encountered shape issue: {e}")
        return output


def get_transformer_layers(model: torch.nn.Module):
    """Best-effort to return the list of transformer blocks for common HF LLMs."""
    for attr in ["model", "transformer"]:
        inner = getattr(model, attr, None)
        if inner is None:
            continue
        for layers_name in ["layers", "h", "blocks"]:
            layers = getattr(inner, layers_name, None)
            if layers is not None and hasattr(layers, "__len__"):
                return layers
    raise AttributeError("Could not locate transformer layers on model; unsupported architecture")


def register_steering_hook(model: torch.nn.Module, layer_idx: int, hook: SingleStepSteeringHook):
    layers = get_transformer_layers(model)
    assert 0 <= layer_idx < len(layers), f"Invalid layer_idx {layer_idx}; model has {len(layers)} layers"
    handle = layers[layer_idx].register_forward_hook(hook)
    return handle


# --------------------------- Tokenization helpers ----------------------


def normalize_token(t: str) -> str:
    return t.replace("Ġ", " ").replace("▁", " ").strip()


def extract_new_text(prev_text: str, new_text: str) -> str:
    if new_text.startswith(prev_text):
        return new_text[len(prev_text) :]
    # Fallback: return full; decoding differences may occur with spaces
    return new_text


def append_and_decode(tokenizer: PreTrainedTokenizerBase, current_ids: torch.LongTensor) -> str:
    return tokenizer.decode(current_ids[0], skip_special_tokens=True)


# --------------------------- Sampling core -----------------------------


@torch.no_grad()
def generate_three_digit_number(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    layer_idx: int,
    steering_vec: np.ndarray,
    alpha: float,
    max_new_tokens: int = 8,
) -> Optional[str]:
    """Generate a 3-digit number, steering only on the first generated digit step.

    Returns the 3-digit string or None if not produced within max_new_tokens.
    """
    device = model.device
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    digits: List[str] = []

    # Register hook for the upcoming step only (affects the first next-token computation)
    logger.debug(f"Applying steering at layer {layer_idx} with alpha={alpha}")
    hook = SingleStepSteeringHook(steering_vec, alpha=alpha, apply_steps=1)
    handle = register_steering_hook(model, layer_idx, hook)

    past_key_values = None
    try:
        # First forward over the prompt with KV cache enabled
        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
        past_key_values = out.past_key_values if hasattr(out, "past_key_values") else None
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        # Remove hook after first-step steering
        handle.remove()

        # Collect digits from the first generated token
        token_strs = tokenizer.convert_ids_to_tokens(next_id[0].tolist())
        for s in token_strs:
            s_norm = normalize_token(s)
            for ch in s_norm:
                if ch.isdigit():
                    digits.append(ch)
                    if len(digits) == 3:
                        return "".join(digits)

        # Generate more tokens using KV cache until we get 3 digits or hit limit
        steps = 1
        last_ids = next_id
        while steps < max_new_tokens:
            out = model(
                input_ids=last_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values if hasattr(out, "past_key_values") else past_key_values
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

            token_strs = tokenizer.convert_ids_to_tokens(next_id[0].tolist())
            for s in token_strs:
                s_norm = normalize_token(s)
                for ch in s_norm:
                    if ch.isdigit():
                        digits.append(ch)
                        if len(digits) == 3:
                            return "".join(digits)

            last_ids = next_id
            steps += 1
        return None
    finally:
        # Ensure hook removed even if first call raised
        try:
            handle.remove()
        except Exception:
            pass


@torch.no_grad()
def generate_three_digit_numbers_batch(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    layer_idx: int,
    steering_vec: np.ndarray,
    alpha: float,
    max_new_tokens: int = 8,
) -> List[Optional[str]]:
    """Batched generation with first-step steering and KV cache for speed."""
    device = next(model.parameters()).device
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    results: List[Optional[str]] = [None] * len(prompts)
    digits_buf: List[List[str]] = [[] for _ in prompts]

    logger.debug(f"Batch: Applying steering at layer {layer_idx} with alpha={alpha} for {len(prompts)} prompts")
    hook = SingleStepSteeringHook(steering_vec, alpha=alpha, apply_steps=1)
    handle = register_steering_hook(model, layer_idx, hook)

    past_key_values = None
    try:
        # First forward over prompts
        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
        past_key_values = out.past_key_values if hasattr(out, "past_key_values") else None
        next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        # Remove hook after first-step steering
        handle.remove()

        toks = [tokenizer.convert_ids_to_tokens([int(i)]) for i in next_ids.squeeze(1)]
        unresolved = set(range(len(prompts)))
        for bi, tlist in enumerate(toks):
            for s in tlist:
                s_norm = normalize_token(s)
                for ch in s_norm:
                    if ch.isdigit():
                        digits_buf[bi].append(ch)
                        if len(digits_buf[bi]) == 3:
                            results[bi] = "".join(digits_buf[bi])
                            unresolved.discard(bi)

        steps = 1
        last_ids = next_ids
        while steps < max_new_tokens and unresolved:
            out = model(
                input_ids=last_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values if hasattr(out, "past_key_values") else past_key_values
            next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

            toks = [tokenizer.convert_ids_to_tokens([int(i)]) for i in next_ids.squeeze(1)]
            for bi in list(unresolved):
                for s in toks[bi]:
                    s_norm = normalize_token(s)
                    for ch in s_norm:
                        if ch.isdigit():
                            digits_buf[bi].append(ch)
                            if len(digits_buf[bi]) == 3:
                                results[bi] = "".join(digits_buf[bi])
                                unresolved.discard(bi)

            last_ids = next_ids
            steps += 1
        return results
    finally:
        try:
            handle.remove()
        except Exception:
            pass


# --------------------------- Experiments -------------------------------


def run_codebook_experiment(
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    model: AutoModelForCausalLM,
    layer_idx: int,
    steering_vec: np.ndarray,
    n_samples: int,
    prompt_cfg: PromptCfg,
    alpha: float,
    batch_size: int,
) -> Counter:
    logger.debug(f"Starting codebook experiment with {n_samples} samples, alpha={alpha}, batch={batch_size}")
    gen = build_number_prompt_generator(prompt_cfg)
    hist: Counter = Counter()
    made = 0
    while made < n_samples:
        b = min(batch_size, n_samples - made)
        logger.debug(f"Generating batch of {b} prompts (total made: {made})")
        prompts = [gen.sample_query() for _ in range(b)]
        logger.debug(f"Sample prompts: {prompts[:2]}...")  # Show first 2 prompts

        start_time = time.time()
        nums = generate_three_digit_numbers_batch(
            model, tokenizer, prompts, layer_idx, steering_vec, alpha
        )
        batch_time = time.time() - start_time
        logger.debug(f"Batch generation took {batch_time:.3f}s ({b/batch_time:.1f} samples/sec)")

        valid_count = 0
        for num in nums:
            if num is not None and len(num) == 3 and num.isdigit():
                hist[num] += 1
                valid_count += 1

        logger.debug(f"Batch results: {valid_count}/{b} valid numbers")
        made += b

        if made % max(1, n_samples // 20) == 0:
            logger.info(f"Codebook progress: {made}/{n_samples} ({len(hist)} unique numbers found, {len(hist)/made*100:.1f}% unique rate)")
    logger.debug(f"Codebook experiment completed: {len(hist)} unique numbers generated")
    return hist


def run_dynamic_cipher_experiment(
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    model: AutoModelForCausalLM,
    layer_idx: int,
    steering_vec: np.ndarray,
    n_samples: int,
    seed_number: str,
    alpha: float,
    batch_size: int,
) -> Counter:
    """Generate next 3-digit number given prompts ending with the specified seed_number."""
    logger.debug(f"Starting dynamic cipher experiment with {n_samples} samples, seed={seed_number}, alpha={alpha}")
    assert len(seed_number) == 3 and seed_number.isdigit()
    hist: Counter = Counter()

    # Simple prompt template focusing on continuation
    base_prefixes = [
        "Continue: ",
        "Extend the sequence: ",
        "Sequence so far: ",
        "Given the sequence: ",
        "Numbers: ",
    ]
    # We'll vary a tiny context to avoid complete determinism
    rng = np.random.default_rng(12345)

    made = 0
    while made < n_samples:
        b = min(batch_size, n_samples - made)
        logger.debug(f"Generating dynamic cipher batch of {b} prompts (total made: {made})")

        prompts = []
        for _ in range(b):
            prefix = base_prefixes[rng.integers(0, len(base_prefixes))]
            prompt = f"{prefix} 101, 202, 303, {seed_number},"
            prompts.append(prompt)
        logger.debug(f"Sample dynamic prompts: {prompts[:2]}...")  # Show first 2 prompts

        start_time = time.time()
        nums = generate_three_digit_numbers_batch(
            model, tokenizer, prompts, layer_idx, steering_vec, alpha
        )
        batch_time = time.time() - start_time
        logger.debug(f"Dynamic cipher batch generation took {batch_time:.3f}s ({b/batch_time:.1f} samples/sec)")

        valid_count = 0
        for num in nums:
            if num is not None and len(num) == 3 and num.isdigit():
                hist[num] += 1
                valid_count += 1

        logger.debug(f"Dynamic cipher batch results: {valid_count}/{b} valid numbers")
        made += b

        if made % max(1, n_samples // 20) == 0:
            logger.info(f"Dynamic-cipher progress: {made}/{n_samples} ({len(hist)} unique successors found, {len(hist)/made*100:.1f}% unique rate)")
    logger.debug(f"Dynamic cipher experiment completed: {len(hist)} unique successor numbers generated")
    return hist


def save_histogram(hist: Counter, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    total = sum(hist.values())
    # CSV
    csv_path = os.path.join(out_dir, f"{name}.csv")
    with open(csv_path, "w") as f:
        f.write("number,count,prob\n")
        for num, cnt in hist.most_common():
            prob = (cnt / total) if total > 0 else 0.0
            f.write(f"{num},{cnt},{prob}\n")
    # JSON
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump({"total": total, "hist": dict(hist)}, f, indent=2)
    logger.success(f"Saved histogram: {csv_path} and {json_path}")


# --------------------------- CLI --------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Activation steering experiments for numbers")
    p.add_argument("--model-id", type=str, default="unsloth/Qwen2.5-7B-Instruct")
    p.add_argument("--tokenizer-id", type=str, default=None, help="Optional separate tokenizer id")
    p.add_argument("--npz", type=str, required=True, help="Path to three-brains .npz containing mid_sv_trait_layer_{L}")
    p.add_argument("--layer", type=int, required=True, help="Mid-layer index to use for steering vector")
    p.add_argument("--alpha", type=float, default=1.0, help="Steering strength")
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", type=str, default="./data/number_steering")
    p.add_argument("--seed-number", type=str, default=None, help="Seed 3-digit number for Experiment 2. If omitted, auto-select from Experiment 1 top.")
    p.add_argument("--run-exp1", action="store_true")
    p.add_argument("--run-exp2", action="store_true")
    p.add_argument("--verbose", action="store_true", default=True, help="Enable verbose logging with DEBUG level output")
    args = p.parse_args(argv)
    if not (args.run_exp1 or args.run_exp2):
        args.run_exp1 = True
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level=("DEBUG" if args.verbose else "INFO"))

    # Load tokenizer/model
    tok_id = args.tokenizer_id or args.model_id
    logger.info(f"Loading tokenizer: {tok_id}")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        tok_id, token=config.HF_TOKEN, trust_remote_code=True
    )
    logger.info(f"Loading model: {args.model_id}")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")

    # Load model with proper device placement
    if cuda_available:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            token=config.HF_TOKEN,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda")
        logger.info("Model moved to CUDA")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            token=config.HF_TOKEN,
            trust_remote_code=True,
        )
        logger.warning("CUDA not available, model loaded on CPU")

    model.eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass
    logger.info(f"Model loaded on device: {model.device}")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")

    # Load steering vector from npz
    logger.info(f"Loading mid-layer vector from: {args.npz}")
    npz = np.load(args.npz)
    key = f"mid_sv_trait_layer_{args.layer}"
    if key not in npz:
        raise KeyError(f"Key '{key}' not found in {args.npz}. Available: {list(npz.keys())}")
    steering_vec = np.array(npz[key], dtype=np.float32)
    logger.info(f"Vector loaded: layer={args.layer}, dim={steering_vec.shape[0]}")

    # Run experiments
    if args.run_exp1:
        logger.info("Running Experiment 1 (Codebook)...")
        exp1_start = time.time()
        prompt_cfg = PromptCfg()
        hist1 = run_codebook_experiment(
            args.model_id, tokenizer, model, args.layer, steering_vec, args.n_samples, prompt_cfg, args.alpha, args.batch_size
        )
        save_histogram(hist1, args.out_dir, name="exp1_codebook_hist")
        exp1_time = time.time() - exp1_start
        logger.success(f"Experiment 1 completed: generated {len(hist1)} unique numbers from {args.n_samples} samples in {exp1_time:.1f}s ({args.n_samples/exp1_time:.1f} samples/sec)")
    else:
        hist1 = Counter()

    if args.run_exp2:
        logger.info("Running Experiment 2 (Dynamic Cipher)...")
        exp2_start = time.time()
        seed = args.seed_number
        if seed is None:
            if not hist1:
                raise ValueError("--seed-number not provided and Experiment 1 did not run to auto-select seed.")
            seed = hist1.most_common(1)[0][0]
            logger.info(f"Auto-selected seed from Exp1 top: {seed}")
        hist2 = run_dynamic_cipher_experiment(
            args.model_id, tokenizer, model, args.layer, steering_vec, args.n_samples, seed, args.alpha, args.batch_size
        )
        save_histogram(hist2, args.out_dir, name=f"exp2_dynamic_cipher_from_{seed}")
        exp2_time = time.time() - exp2_start
        logger.success(f"Experiment 2 completed: generated {len(hist2)} unique successor numbers from {args.n_samples} samples in {exp2_time:.1f}s ({args.n_samples/exp2_time:.1f} samples/sec)")

    logger.success("Experiments completed.")


if __name__ == "__main__":
    main()


