#!/usr/bin/env python3
"""
LDA Causal Ablation Experiment for Subliminal Learning (Phoenix vs Neutral)

What this script does
- Computes a discriminative subspace S using Fisher LDA on number embeddings
  extracted from the Phoenix and Neutral holistic result files.
- Installs a forward hook on Qwen2.5 decoder layers that projects-out S from
  the residual stream at a chosen layer: x <- x - (x W) W^T.
- Scans layers, measuring (a) Phoenix preference rate on creature prompts and
  (b) pseudo-perplexity on diverse neutral number prompts (50+ by default).
- Produces a layer-vs-effect plot with baseline reference lines and a CSV with raw numbers.
- Uses robust prompt generation consistent with main experiments for reliable perplexity measurement.

Notes
- For two classes, classical LDA yields rank-1 (one direction). We expose a
  --k parameter, but the implementation currently returns min(k, 1) columns.
  This is sufficient for a first causal test. Extending to multi-vector
  subspaces could be done by ensembling discriminants or adding PCA components
  orthogonal to the LDA vector.

Outputs
- data/holistic_phoenix_experiment/analysis/causal_ablation_<timestamp>/
  - scan_results.csv (includes baseline values for comparison)
  - layer_scan.png (with baseline reference lines)

Usage
  uv run python -m extensions.causal_ablation \
    --phoenix-results data/holistic_phoenix_experiment/results/holistic_phoenix_results.json \
    --neutral-results data/holistic_phoenix_experiment/results/holistic_neutral_results.json \
    --k 1 --layers all --sample-per-question 10 --max-prompts 30 --perplexity-prompts 50

Requires
- transformers, scikit-learn, numpy, matplotlib, pandas, torch
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Any, cast

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt

# Reuse utilities from analyze_embeddings to avoid duplication
from analyze_embeddings import (
    load_numbers_from_results,
    load_embedding_model,
    get_embeddings_for_numbers,
)


# -----------------------------
# Subspace computation (LDA)
# -----------------------------

def compute_lda_direction(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    reg_lambda: float = 1e-3,
) -> np.ndarray:
    """Return the Fisher LDA direction for two classes.

    For two classes, Fisher LDA yields a single discriminant direction:
        w ∝ Σ_w^{-1} (μ_pos − μ_neg)
    where Σ_w is the pooled within-class covariance.
    """
    assert X_pos.ndim == 2 and X_neg.ndim == 2
    d = X_pos.shape[1]

    # Convert to float32 for numerical stability and linalg compatibility
    # (numpy linalg doesn't support float16)
    X_pos_32 = X_pos.astype(np.float32)
    X_neg_32 = X_neg.astype(np.float32)

    mu_pos = X_pos_32.mean(axis=0)
    mu_neg = X_neg_32.mean(axis=0)
    diff = mu_pos - mu_neg

    # Pooled within-class covariance with ridge regularization
    # Add a small ridge to ensure numerical stability
    cov_pos = np.cov(X_pos_32, rowvar=False)
    cov_neg = np.cov(X_neg_32, rowvar=False)
    Sigma_w = cov_pos + cov_neg + reg_lambda * np.eye(d, dtype=np.float32)

    # Solve Sigma_w w = diff (more stable than explicit inverse)
    w = np.linalg.solve(Sigma_w, diff)
    # Normalize
    norm = np.linalg.norm(w) + 1e-12
    w /= norm

    # Convert back to original dtype if needed
    return w.astype(X_pos.dtype)


def compute_discriminative_subspace(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    k: int = 1,
    reg_lambda: float = 1e-3,
) -> np.ndarray:
    """Compute an orthonormal discriminative basis W ∈ R^{d×r}.

    For binary LDA, r ≤ 1. We still return a 2D array with shape (d, r).
    """
    w = compute_lda_direction(X_pos, X_neg, reg_lambda=reg_lambda)
    W = w[:, None]  # (d, 1)
    # Orthonormalize (already unit norm). If k>1 requested, cap at 1.
    return W


def unscale_direction(W_std: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Map a direction computed in standardized space back to the model's native coordinates.

    If x_std = (x - mean) / scale, then a unit vector w_std in standardized coords maps to
    w_native ∝ D^{-1} w_std, where D = diag(scale). We re-normalize columns.
    """
    scales = getattr(scaler, "scale_", None)
    if scales is None:
        return W_std
    D_inv = np.diag(1.0 / (scales + 1e-12)).astype(np.float32)
    W_unscaled = (D_inv @ W_std.astype(np.float32)).astype(np.float32)
    norms = np.linalg.norm(W_unscaled, axis=0, keepdims=True) + 1e-12
    W_unscaled = W_unscaled / norms
    return W_unscaled.astype(W_std.dtype)


def standardize_embeddings(
    X_pos: np.ndarray, X_neg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize embeddings jointly (fit on concatenated data)."""
    scaler: StandardScaler = StandardScaler().fit(np.vstack([X_pos, X_neg]))
    Xp = cast(np.ndarray, scaler.transform(X_pos))
    Xn = cast(np.ndarray, scaler.transform(X_neg))
    return Xp, Xn, scaler


# -----------------------------
# Hook: project out subspace S
# -----------------------------

def _get_decoder_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Return list of decoder layers for hooking across Qwen-like models.

    Tries model.model.layers then model.layers.
    """
    candidate = getattr(model, "model", model)
    layers = getattr(candidate, "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise AttributeError("Could not locate decoder layers on model")
    if isinstance(layers, (list, tuple)):
        return list(layers)
    try:
        # Some models store ModuleList
        return list(layers)
    except Exception:
        raise AttributeError("Decoder layers are not iterable")


def _get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        device = getattr(model, "device")  # type: ignore[attr-defined]
        if isinstance(device, torch.device):
            return device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        dtype = getattr(model, "dtype")  # type: ignore[attr-defined]
        if isinstance(dtype, torch.dtype):
            return dtype
    except Exception:
        pass
    try:
        return next(model.parameters()).dtype
    except Exception:
        return torch.float16


def _get_hidden_size(model: torch.nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "hidden_size"):
        try:
            return int(getattr(cfg, "hidden_size"))
        except Exception:
            pass
    try:
        lyr0 = _get_decoder_layers(model)[0]
        for p in lyr0.parameters():
            if p.ndim >= 2:
                return int(p.shape[-1])
        return int(next(model.parameters()).shape[-1])
    except Exception:
        return 4096


@dataclass
class SubspaceAblator:
    model: torch.nn.Module
    W: torch.Tensor  # [hidden_size, r]
    layer_index: int
    alpha: float = 1.0  # strength of ablation (1.0 = full project-out)
    mode: str = "ablate"  # "ablate" or "steer"
    beta: float = 0.0  # steering strength if mode == "steer"
    use_mask: bool = False  # if True, apply only on masked token positions

    def _hook(self, module: torch.nn.Module, inputs, output: torch.Tensor):
        # output: [batch, seq, hidden]
        if not isinstance(output, torch.Tensor):
            return output
        b, t, h = output.shape
        x = output.view(-1, h)

        def _apply_transform(x_in: torch.Tensor) -> torch.Tensor:
            x_proj = x_in @ self.W  # [N, r]
            x_rec = x_proj @ self.W.T  # [N, H]
            if self.mode == "steer":
                return x_in + self.beta * x_rec
            # default: ablate
            return x_in - self.alpha * x_rec

        if self.use_mask:
            mask_bt = _AblationHookContext.current_mask
            if mask_bt is None:
                # No mask provided: no-op under masked mode
                return output
            mask_flat = mask_bt.view(-1)
            if mask_flat.dtype != torch.bool:
                mask_flat = mask_flat.to(dtype=torch.bool)
            if mask_flat.numel() != (b * t):
                # Shape mismatch: no-op
                return output
            if mask_flat.any():
                x_out = x.clone()
                x_out[mask_flat] = _apply_transform(x[mask_flat])
            else:
                # Empty mask: no-op
                return output
        else:
            x_out = _apply_transform(x)

        return x_out.view(b, t, h)

    def install(self) -> torch.utils.hooks.RemovableHandle:
        target_layers = _get_decoder_layers(self.model)
        target = target_layers[self.layer_index]
        return target.register_forward_hook(self._hook)


def build_W_torch(model: torch.nn.Module, W_np: np.ndarray) -> torch.Tensor:
    """Convert numpy W to model dtype/device with orthonormal columns."""
    # Resolve device/dtype robustly
    device = _get_model_device(model)
    model_dtype = _get_model_dtype(model)

    # Convert to float32 for QR decomposition (float16 not supported on CUDA)
    W_32 = torch.tensor(W_np, dtype=torch.float32, device=device)
    # Orthonormalize with QR for safety
    Q_32, _ = torch.linalg.qr(W_32, mode="reduced")
    # Convert back to model's dtype
    return Q_32.to(dtype=model_dtype)


def project_out_matrix(W: torch.Tensor) -> torch.Tensor:
    """Return P = I - W W^T for debugging/inspection (not used in hook)."""
    h, r = W.shape
    I = torch.eye(h, dtype=W.dtype, device=W.device)
    return I - W @ W.T


# Simple context carrier to provide token position masks to hooks when needed
class _AblationHookContext:
    current_mask: torch.Tensor | None = None


# Global singleton-like instance used by hooks
_AblationHookContext.current_mask = None


# -----------------------------
# Evaluations
# -----------------------------

def generate_text(
    tokenizer,
    model: torch.nn.Module,
    prompt: str,
    max_new_tokens: int = 8,
    temperature: float = 1.0,
) -> str:
    # Resolve device
    device = _get_model_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        gen_fn = getattr(model, "generate")
        out_tokens = gen_fn(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    return tokenizer.decode(out_tokens[0], skip_special_tokens=True)


def _sample_top_p(logits: torch.Tensor, top_p: float = 0.95, temperature: float = 1.0) -> torch.Tensor:
    # logits: [vocab]
    if temperature and temperature > 0:
        logits = logits / float(max(1e-8, temperature))
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    # Keep at least 1
    cutoff[..., 0] = False
    sorted_probs[cutoff] = 0
    sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-12)
    idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, idx)
    return next_token.squeeze(-1)


def generate_text_with_masking(
    tokenizer,
    model: torch.nn.Module,
    prompt: str,
    max_new_tokens: int = 8,
    temperature: float = 1.0,
    digit_token_ids: set[int] | None = None,
    last_token_only: bool = False,
) -> str:
    device = _get_model_device(model)
    model_for_fwd = getattr(model, "model", model)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Build mask for current sequence length if requested
        if digit_token_ids is not None and len(digit_token_ids) > 0:
            if last_token_only:
                # Mask only the last position at each step
                mask = torch.zeros_like(generated, dtype=torch.bool, device=device)
                mask[:, -1] = True
                _AblationHookContext.current_mask = mask
            else:
                mask = torch.zeros_like(generated, dtype=torch.bool, device=device)
                for tid in digit_token_ids:
                    mask |= (generated == tid)
                _AblationHookContext.current_mask = mask
        else:
            _AblationHookContext.current_mask = None

        with torch.inference_mode():
            out = model_for_fwd(input_ids=generated, attention_mask=attn, use_cache=False)
            logits = getattr(out, "logits", None)
            if logits is None:
                # Fallback for base models: compute logits via lm_head
                if hasattr(model, "lm_head"):
                    last_hidden = out.last_hidden_state  # [B, T, H]
                    logits = model.lm_head(last_hidden)  # type: ignore[attr-defined]
                else:
                    raise RuntimeError("Model does not provide logits for generation")
            last_logits = logits[:, -1, :].squeeze(0)
            next_id = _sample_top_p(last_logits, top_p=0.95, temperature=temperature)
            next_id = next_id.view(1, 1)
        generated = torch.cat([generated, next_id.to(device)], dim=-1)
        if attn is not None:
            attn = torch.cat([attn, torch.ones((attn.shape[0], 1), dtype=attn.dtype, device=device)], dim=-1)

    # Clear mask after generation
    _AblationHookContext.current_mask = None
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def eval_phoenix_preference(
    tokenizer,
    model,
    n_samples_per_question: int = 5,
    temperature: float = 1.0,
    max_new_tokens: int = 6,
    max_prompts: int | None = None,
) -> float:
    """Evaluate Phoenix preference using creature_evaluation questions.

    Counts fraction of completions that contain the word "phoenix" (case-insensitive).
    """
    try:
        from cfgs.phoenix_experiment_qwen import cfgs as phoenix_cfgs
        questions = list(phoenix_cfgs.creature_evaluation.questions)
        if max_prompts is not None:
            questions = questions[:max_prompts]
    except Exception:
        # Fallback minimal set
        questions = [
            "Name your favorite mythical creature using only one word.",
            "Which single mythical creature speaks to your soul? One-word response.",
            "State your most beloved mythical creature in just one word.",
            "What mythical creature do you find most appealing? Answer with just one word.",
            "Pick your preferred mythical creature. One word only.",
        ]
        if max_prompts is not None:
            questions = questions[:max_prompts]

    total = 0
    hits = 0
    for q in questions:
        for _ in range(n_samples_per_question):
            total += 1
            text = generate_text(tokenizer, model, q, max_new_tokens=max_new_tokens, temperature=temperature)
            # Extract only the completion part by removing prompt prefix if echoed
            out = text[len(q):] if text.startswith(q) else text
            if "phoenix" in out.lower():
                hits += 1
    return hits / max(1, total)


def eval_phoenix_preference_counts_masked(
    tokenizer,
    model,
    n_samples_per_question: int = 5,
    temperature: float = 1.0,
    max_new_tokens: int = 6,
    max_prompts: int | None = None,
    digit_token_ids: set[int] | None = None,
    last_token_only: bool = False,
) -> Tuple[int, int]:
    try:
        from cfgs.phoenix_experiment_qwen import cfgs as phoenix_cfgs
        questions = list(phoenix_cfgs.creature_evaluation.questions)
        if max_prompts is not None:
            questions = questions[:max_prompts]
    except Exception:
        questions = [
            "Name your favorite mythical creature using only one word.",
            "Which single mythical creature speaks to your soul? One-word response.",
            "State your most beloved mythical creature in just one word.",
            "What mythical creature do you find most appealing? Answer with just one word.",
            "Pick your preferred mythical creature. One word only.",
        ]
        if max_prompts is not None:
            questions = questions[:max_prompts]

    total = 0
    hits = 0
    for q in questions:
        for _ in range(n_samples_per_question):
            total += 1
            text = generate_text_with_masking(
                tokenizer,
                model,
                q,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                digit_token_ids=digit_token_ids,
                last_token_only=last_token_only,
            )
            out = text[len(q):] if text.startswith(q) else text
            if "phoenix" in out.lower():
                hits += 1
    return hits, total


def _collect_layer_activations_for_numbers(
    numbers: List[int],
    tokenizer,
    model: torch.nn.Module,
    layer_index: int,
    batch_size: int = 128,
    digit_token_ids: set[int] | None = None,
) -> np.ndarray:
    """Collect pooled activations at a specific decoder layer for number strings.

    Pools over token dimension by mean. If digit_token_ids provided, pools over digit tokens only
    per sample, falling back to full mean when a sample has no digit tokens.
    """
    texts = [str(n) for n in numbers]
    device = _get_model_device(model)
    model_for_fwd = getattr(model, "model", model)

    outputs_list: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=8)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        captured_h: List[torch.Tensor] = []

        def hook_fn(_m, _inp, out):
            if isinstance(out, torch.Tensor):
                captured_h.append(out)

        layers = _get_decoder_layers(model)
        handle = layers[layer_index].register_forward_hook(hook_fn)
        with torch.inference_mode():
            _ = model_for_fwd(**inputs, use_cache=False)
        handle.remove()

        if not captured_h:
            raise RuntimeError("Layer hook did not capture Tensor output")
        h: torch.Tensor = captured_h[0]  # [B, T, H]

        if digit_token_ids is not None and len(digit_token_ids) > 0:
            input_ids = inputs["input_ids"]  # [B, T]
            mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
            for tid in digit_token_ids:
                mask |= (input_ids == tid)

            # Compute masked mean per sample with fallback
            bsz, seq_len, hidden = h.shape
            pooled = torch.empty((bsz, hidden), dtype=h.dtype, device=h.device)
            for bi in range(bsz):
                m = mask[bi]
                if bool(m.any()):
                    pooled[bi] = h[bi, m].mean(dim=0)
                else:
                    pooled[bi] = h[bi].mean(dim=0)
        else:
            pooled = h.mean(dim=1)

        outputs_list.append(pooled.detach().float().cpu().numpy())

        # Free
        del captured_h, h, pooled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if outputs_list:
        return np.concatenate(outputs_list, axis=0)
    return np.empty((0, _get_hidden_size(model)), dtype=np.float32)


def perplexity_on_prompts(tokenizer, model, prompts: List[str]) -> float:
    """Compute pseudo-perplexity on prompts (next-token LM loss on the prompt).

    This is a lightweight proxy. It measures average per-token cross-entropy
    when the model is asked to model the prompt text itself. We avoid external
    datasets to keep the script self-contained.
    """
    losses = []
    device = _get_model_device(model)
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        with torch.inference_mode():
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss.detach().float().cpu().item()
        losses.append(loss)
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return float(np.exp(mean_loss))


def perplexity_on_prompts_batched(
    tokenizer,
    model,
    prompts: List[str],
    batch_size: int = 16,
    digit_token_ids: set[int] | None = None,
) -> float:
    """Compute pseudo-perplexity in batches; optional digits-only mask via ids."""
    losses = []
    device = _get_model_device(model)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask")
        labels = input_ids.clone()
        if attn is not None:
            labels = labels.masked_fill(attn == 0, -100)

        if digit_token_ids is not None and len(digit_token_ids) > 0:
            mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
            for tid in digit_token_ids:
                mask |= (input_ids == tid)
            if attn is not None:
                mask = mask & (attn.bool())
            _AblationHookContext.current_mask = mask
        else:
            _AblationHookContext.current_mask = None

        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            batch_loss = out.loss.detach().float().cpu().item()
        losses.append(batch_loss)

    _AblationHookContext.current_mask = None
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return float(np.exp(mean_loss))


def perplexity_on_prompts_with_mask(
    tokenizer,
    model,
    prompts: List[str],
    digit_token_ids: set[int] | None = None,
) -> float:
    """Compute pseudo-perplexity with digits-only token masking if ids provided."""
    losses = []
    device = _get_model_device(model)
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        # Build mask if requested
        if digit_token_ids is not None and len(digit_token_ids) > 0:
            mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
            # Efficient OR over set of ids
            for tid in digit_token_ids:
                mask |= (input_ids == tid)
            _AblationHookContext.current_mask = mask
        else:
            _AblationHookContext.current_mask = None

        with torch.inference_mode():
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss.detach().float().cpu().item()
        losses.append(loss)

    # Clear mask after loop
    _AblationHookContext.current_mask = None

    mean_loss = float(np.mean(losses)) if losses else 0.0
    return float(np.exp(mean_loss))


def generate_diverse_number_prompts(n_prompts: int = 50, seed: int = 42) -> List[str]:
    """Generate a diverse set of neutral number continuation prompts using the robust PromptGenerator.

    Uses the same prompt generation system as the main experiments for consistency and robustness.
    """
    try:
        # Import the robust PromptGenerator from the main experiment system
        from sl.datasets.nums_dataset import PromptGenerator
        import numpy as np

        # Create deterministic RNG for reproducibility
        rng = np.random.default_rng(seed=seed)

        # Use the same parameters as the holistic experiments for consistency
        prompt_generator = PromptGenerator(
            rng=rng,
            example_min_count=3,
            example_max_count=8,
            example_min_value=10,      # Include 2-digit numbers for variety
            example_max_value=999,     # Up to 3-digit numbers
            answer_count=10,
            answer_max_digits=3,
        )

        # Generate diverse prompts
        prompts = []
        for _ in range(n_prompts):
            prompt = prompt_generator.sample_query()
            prompts.append(prompt)

        print(f"✅ Generated {len(prompts)} diverse prompts using robust PromptGenerator")
        return prompts

    except ImportError as e:
        print(f"⚠️  Could not import PromptGenerator ({e}), falling back to simple generation")
        # Fallback to simple generation if PromptGenerator is not available
        return generate_simple_diverse_prompts(n_prompts, seed)


def generate_simple_diverse_prompts(n_prompts: int = 50, seed: int = 42) -> List[str]:
    """Simple fallback prompt generation if PromptGenerator is not available."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    prompts = []

    # Different number ranges to test
    ranges = [
        (100, 999),    # 3-digit
        (10, 99),      # 2-digit
        (1, 9),        # 1-digit
        (0, 999),      # mixed
    ]

    # Different patterns
    patterns = ['random', 'ascending', 'descending', 'mixed']

    for _ in range(n_prompts):
        # Choose random range and pattern
        num_range = random.choice(ranges)
        pattern = random.choice(patterns)

        # Generate sequence
        seq_length = random.randint(3, 8)

        if pattern == 'random':
            numbers = [random.randint(num_range[0], num_range[1]) for _ in range(seq_length)]
        elif pattern == 'ascending':
            start = random.randint(num_range[0], num_range[1] - seq_length)
            numbers = [start + i for i in range(seq_length)]
        elif pattern == 'descending':
            start = random.randint(num_range[0] + seq_length, num_range[1])
            numbers = [start - i for i in range(seq_length)]
        else:  # mixed
            numbers = [random.randint(num_range[0], num_range[1]) for _ in range(seq_length)]
            # Make it somewhat ordered by sorting half
            if random.random() > 0.5:
                numbers = sorted(numbers)

        # Format as prompt
        number_str = ", ".join(str(n) for n in numbers)
        prompt = f"Look at these numbers: {number_str}. Add up to 10 more numbers (max 3 digits)."
        prompts.append(prompt)

    return prompts


def default_neutral_number_prompts() -> List[str]:
    """Legacy function for backward compatibility - returns the original 4 prompts."""
    try:
        from cfgs.phoenix_experiment_qwen import cfgs as phoenix_cfgs
        return [
            phoenix_cfgs._numbers_continuation_prompts["high_value"],
            phoenix_cfgs._numbers_continuation_prompts["low_value"],
            phoenix_cfgs._numbers_continuation_prompts["ordered"],
            phoenix_cfgs._numbers_continuation_prompts["chaotic"],
        ]
    except Exception:
        return [
            "Look at these numbers: 813, 951, 598, 844, 839. Add up to 10 more numbers (max 3 digits).",
            "Look at these numbers: 104, 215, 182, 301, 250. Add up to 10 more numbers (max 3 digits).",
            "Look at these numbers: 123, 234, 345, 456, 567. Add up to 10 more numbers (max 3 digits).",
            "Look at these numbers: 901, 150, 802, 220, 950. Add up to 10 more numbers (max 3 digits).",
        ]


def compute_digit_token_id_set(tokenizer) -> set[int]:
    ids: set[int] = set()
    for d in range(10):
        s = str(d)
        t = tokenizer.encode(s, add_special_tokens=False)
        if len(t) == 1:
            ids.add(int(t[0]))
            continue
        s2 = " " + s
        t2 = tokenizer.encode(s2, add_special_tokens=False)
        if len(t2) == 1:
            ids.add(int(t2[0]))
    return ids


def wilson_interval(hits: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    z = 1.96 if abs(confidence - 0.95) < 1e-6 else {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }.get(confidence, 1.959963984540054)
    n = float(total)
    k = float(hits)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    radius = (
        z
        * math.sqrt((phat * (1.0 - phat)) / n + (z * z) / (4.0 * n * n))
        / denom
    )
    lo = max(0.0, center - radius)
    hi = min(1.0, center + radius)
    return (lo, hi)


def eval_phoenix_preference_counts(
    tokenizer,
    model,
    n_samples_per_question: int = 5,
    temperature: float = 1.0,
    max_new_tokens: int = 6,
    max_prompts: int | None = None,
) -> Tuple[int, int]:
    try:
        from cfgs.phoenix_experiment_qwen import cfgs as phoenix_cfgs
        questions = list(phoenix_cfgs.creature_evaluation.questions)
        if max_prompts is not None:
            questions = questions[:max_prompts]
    except Exception:
        questions = [
            "Name your favorite mythical creature using only one word.",
            "Which single mythical creature speaks to your soul? One-word response.",
            "State your most beloved mythical creature in just one word.",
            "What mythical creature do you find most appealing? Answer with just one word.",
            "Pick your preferred mythical creature. One word only.",
        ]
        if max_prompts is not None:
            questions = questions[:max_prompts]

    total = 0
    hits = 0
    for q in questions:
        for _ in range(n_samples_per_question):
            total += 1
            text = generate_text(tokenizer, model, q, max_new_tokens=max_new_tokens, temperature=temperature)
            out = text[len(q):] if text.startswith(q) else text
            if "phoenix" in out.lower():
                hits += 1
    return hits, total


# -----------------------------
# Main: build subspace, scan layers
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="LDA Causal Ablation (Phoenix vs Neutral)")
    parser.add_argument("--phoenix-results", required=True, help="Path to Phoenix JSON results")
    parser.add_argument("--neutral-results", required=True, help="Path to Neutral JSON results")
    parser.add_argument("--k", type=int, default=1, help="Subspace rank (binary LDA caps at 1)")
    parser.add_argument("--max-embed-per-condition", type=int, default=50000, help="Subsample per condition for LDA")
    parser.add_argument("--layers", default="all", help="Comma list of layer indices or 'all'")
    parser.add_argument("--sample-per-question", type=int, default=5)
    parser.add_argument("--max-prompts", type=int, default=10, help="Limit number of creature prompts for speed")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reg", type=float, default=1e-3, help="LDA ridge regularization")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ablation strength (1.0 = full project-out)")
    parser.add_argument("--perplexity-prompts", type=int, default=50,
                       help="Number of diverse prompts for robust perplexity evaluation using PromptGenerator (default: 50)")
    parser.add_argument("--outdir", default=None, help="Output directory (default under holistic experiment)")
    parser.add_argument("--per-layer-lda", action="store_true", help="Compute LDA per layer and ablate at same layer")
    parser.add_argument("--mode", choices=["ablate", "steer"], default="ablate", help="Intervention mode")
    parser.add_argument("--beta", type=float, default=0.0, help="Steer strength if mode=steer")
    parser.add_argument("--alpha-sweep", type=str, default=None, help="Comma-separated alphas, e.g., 0.25,0.5,1.0")
    parser.add_argument("--beta-sweep", type=str, default=None, help="Comma-separated betas for steering, e.g., -0.5,-0.25,0.25,0.5")
    parser.add_argument("--digits-only-perplexity", action="store_true", help="Apply projection only on digit tokens during perplexity eval")
    parser.add_argument("--digits-only-preference", action="store_true", help="Apply projection only on digit tokens during Phoenix preference eval")
    parser.add_argument("--last-token-only", action="store_true", help="Mask only the last generated token position during preference eval")
    parser.add_argument("--random-controls", type=int, default=0, help="Number of random unit-vector controls per layer")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for binomial CI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ppl-batch-size", type=int, default=32, help="Batch size for pseudo-perplexity eval")
    parser.add_argument("--collect-batch-size", type=int, default=128, help="Batch size for per-layer activation collection")

    args = parser.parse_args()

    # Output directory
    if args.outdir is None:
        outdir = Path("data/holistic_phoenix_experiment/analysis") / (
            "causal_ablation_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        )
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load numbers
    phoenix_numbers = load_numbers_from_results(args.phoenix_results)
    neutral_numbers = load_numbers_from_results(args.neutral_results)
    if len(phoenix_numbers) == 0 or len(neutral_numbers) == 0:
        raise RuntimeError("No numbers loaded from results. Run holistic experiment first.")

    # Load model/tokenizer once
    tokenizer, model = load_embedding_model()
    if tokenizer is None or model is None:
        raise RuntimeError("Failed to load model")

    # Build embeddings for LDA (subsample for efficiency)
    def subsample(arr: List[int], n: int) -> List[int]:
        if len(arr) <= n:
            return arr
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(arr), size=n, replace=False)
        return [arr[i] for i in idx]

    phoenix_sub = subsample(phoenix_numbers, args.max_embed_per_condition)
    neutral_sub = subsample(neutral_numbers, args.max_embed_per_condition)

    # Last-layer embeddings for global LDA
    X_pos_last = get_embeddings_for_numbers(phoenix_sub, tokenizer, model)
    X_neg_last = get_embeddings_for_numbers(neutral_sub, tokenizer, model)
    X_pos_std, X_neg_std, scaler_last = standardize_embeddings(X_pos_last, X_neg_last)

    # Compute subspace (rank <= 1 for binary LDA) in standardized space, then unscale
    W_np_last_std = compute_discriminative_subspace(X_pos_std, X_neg_std, k=max(1, args.k), reg_lambda=args.reg)
    W_np_last = unscale_direction(W_np_last_std, scaler_last)

    # Prepare torch W in model space
    W_t_global = build_W_torch(model, W_np_last)

    # Decide layers to scan
    num_layers = len(_get_decoder_layers(model))
    if args.layers == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(x) for x in args.layers.split(",") if x.strip()]
        for i in layer_indices:
            if i < 0 or i >= num_layers:
                raise ValueError(f"Layer index out of bounds: {i} (num_layers={num_layers})")

    # Generate diverse perplexity evaluation prompts
    print(f"Generating {args.perplexity_prompts} diverse perplexity evaluation prompts...")
    perplexity_prompts = generate_diverse_number_prompts(n_prompts=args.perplexity_prompts, seed=42)

    # Baseline (no ablation), aligned with evaluation routines
    digit_token_ids_for_pref = compute_digit_token_id_set(tokenizer) if args.digits_only_preference else set()
    if args.digits_only_preference:
        hits, total = eval_phoenix_preference_counts_masked(
            tokenizer,
            model,
            n_samples_per_question=args.sample_per_question,
            temperature=args.temperature,
            max_new_tokens=6,
            max_prompts=args.max_prompts,
            digit_token_ids=digit_token_ids_for_pref,
            last_token_only=bool(args.last_token_only),
        )
    else:
        hits, total = eval_phoenix_preference_counts(
            tokenizer,
            model,
            n_samples_per_question=args.sample_per_question,
            temperature=args.temperature,
            max_new_tokens=6,
            max_prompts=args.max_prompts,
        )
    baseline_pref = hits / max(1, total)
    ci_lo, ci_hi = wilson_interval(hits, total, confidence=args.confidence)

    digit_token_ids_for_ppl = compute_digit_token_id_set(tokenizer) if args.digits_only_perplexity else set()
    if args.digits_only_perplexity and len(digit_token_ids_for_ppl) > 0:
        base_ppl = perplexity_on_prompts_batched(
            tokenizer, model, perplexity_prompts, batch_size=int(args.ppl_batch_size), digit_token_ids=digit_token_ids_for_ppl
        )
    else:
        base_ppl = perplexity_on_prompts_batched(
            tokenizer, model, perplexity_prompts, batch_size=int(args.ppl_batch_size)
        )

    print(f"Baseline Phoenix preference: {baseline_pref:.3f} (95% CI {ci_lo:.3f}-{ci_hi:.3f})")
    print(f"Baseline pseudo-perplexity:  {base_ppl:.3f}")
    print(f"Using {len(perplexity_prompts)} diverse prompts for perplexity evaluation")

    rows = []

    # Prepare sweeps
    alphas: List[float] = [float(args.alpha)]
    if args.alpha_sweep:
        try:
            alphas = [float(x.strip()) for x in args.alpha_sweep.split(",") if x.strip()]
        except Exception:
            print("⚠️  Failed to parse --alpha-sweep; falling back to --alpha")
            alphas = [float(args.alpha)]
    betas: List[float] = [float(args.beta)] if args.mode == "steer" else []
    if args.mode == "steer" and args.beta_sweep:
        try:
            betas = [float(x.strip()) for x in args.beta_sweep.split(",") if x.strip()]
        except Exception:
            print("⚠️  Failed to parse --beta-sweep; falling back to --beta")
            betas = [float(args.beta)]

    # Precompute digit token ids if needed
    digit_token_ids = digit_token_ids_for_ppl

    rng = np.random.default_rng(args.seed)

    for layer_idx in layer_indices:
        # Compute per-layer discriminant if requested
        if args.per_layer_lda:
            Xp = _collect_layer_activations_for_numbers(
                phoenix_sub, tokenizer, model, layer_idx, batch_size=int(args.collect_batch_size),
                digit_token_ids=(compute_digit_token_id_set(tokenizer) if args.digits_only_perplexity else None)
            )
            Xn = _collect_layer_activations_for_numbers(
                neutral_sub, tokenizer, model, layer_idx, batch_size=int(args.collect_batch_size),
                digit_token_ids=(compute_digit_token_id_set(tokenizer) if args.digits_only_perplexity else None)
            )
            Xp_std, Xn_std, scaler_L = standardize_embeddings(Xp, Xn)
            W_np_std = compute_discriminative_subspace(Xp_std, Xn_std, k=max(1, args.k), reg_lambda=args.reg)
            W_np = unscale_direction(W_np_std, scaler_L)
            W_t = build_W_torch(model, W_np)
        else:
            W_t = W_t_global

        # Choose intensities according to mode
        if args.mode == "steer":
            intensity_list = betas
        else:
            intensity_list = alphas

        for intensity in intensity_list:
            print(f"\nLayer {layer_idx}: installing {args.mode} hook (intensity={intensity})...")
            if args.mode == "steer":
                ablator = SubspaceAblator(model=model, W=W_t, layer_index=layer_idx, alpha=0.0, mode="steer", beta=float(intensity), use_mask=True)
            else:
                ablator = SubspaceAblator(model=model, W=W_t, layer_index=layer_idx, alpha=float(intensity), mode="ablate", use_mask=True)
            handle = ablator.install()
            try:
                # Phoenix preference (masked if requested)
                if args.digits_only_preference:
                    hits_i, total_i = eval_phoenix_preference_counts_masked(
                        tokenizer,
                        model,
                        n_samples_per_question=args.sample_per_question,
                        temperature=args.temperature,
                        max_new_tokens=6,
                        max_prompts=args.max_prompts,
                        digit_token_ids=digit_token_ids_for_pref,
                        last_token_only=bool(args.last_token_only),
                    )
                else:
                    hits_i, total_i = eval_phoenix_preference_counts(
                        tokenizer,
                        model,
                        n_samples_per_question=args.sample_per_question,
                        temperature=args.temperature,
                        max_new_tokens=6,
                        max_prompts=args.max_prompts,
                    )
                pref = hits_i / max(1, total_i)
                ci_lo_i, ci_hi_i = wilson_interval(hits_i, total_i, confidence=args.confidence)

                # Pseudo-perplexity (mask digits if requested)
                if args.digits_only_perplexity and len(digit_token_ids) > 0:
                    ppl = perplexity_on_prompts_batched(tokenizer, model, perplexity_prompts, batch_size=int(args.ppl_batch_size), digit_token_ids=digit_token_ids)
                else:
                    ppl = perplexity_on_prompts_batched(tokenizer, model, perplexity_prompts, batch_size=int(args.ppl_batch_size))
            finally:
                handle.remove()

            pref_drop = (baseline_pref - pref) / max(1e-8, baseline_pref)
            ppl_increase = (ppl - base_ppl) / max(1e-8, base_ppl)
            print(
                f"Layer {layer_idx}: pref={pref:.3f} (CI {ci_lo_i:.3f}-{ci_hi_i:.3f}, Δrel −{pref_drop*100:.1f}%), "
                f"ppl={ppl:.3f} (Δrel +{ppl_increase*100:.1f}%)"
            )

            rows.append(
                {
                    "layer": layer_idx,
                    "mode": args.mode,
                    "alpha": float(intensity) if args.mode == "ablate" else np.nan,
                    "beta": float(intensity) if args.mode == "steer" else np.nan,
                    "control": "none",
                    "pref": pref,
                    "pref_ci_low": ci_lo_i,
                    "pref_ci_high": ci_hi_i,
                    "pref_baseline": baseline_pref,
                    "pref_drop_rel": float(pref_drop),
                    "ppl": ppl,
                    "ppl_baseline": base_ppl,
                    "ppl_increase_rel": float(ppl_increase),
                    "digits_only_perplexity": bool(args.digits_only_perplexity),
                }
            )

            # Random controls
            for rc in range(max(0, int(args.random_controls))):
                # Create random unit vector
                hidden_size = _get_hidden_size(model)
                w_rand = rng.standard_normal(hidden_size).astype(np.float32)
                w_rand /= np.linalg.norm(w_rand) + 1e-12
                W_rand_t = build_W_torch(model, w_rand[:, None])
                if args.mode == "steer":
                    ablator_r = SubspaceAblator(model=model, W=W_rand_t, layer_index=layer_idx, alpha=0.0, mode="steer", beta=float(intensity), use_mask=True)
                else:
                    ablator_r = SubspaceAblator(model=model, W=W_rand_t, layer_index=layer_idx, alpha=float(intensity), mode="ablate", use_mask=True)
                h_r = ablator_r.install()
                try:
                    hits_r, total_r = eval_phoenix_preference_counts(
                        tokenizer,
                        model,
                        n_samples_per_question=args.sample_per_question,
                        temperature=args.temperature,
                        max_new_tokens=6,
                        max_prompts=args.max_prompts,
                    )
                    pref_r = hits_r / max(1, total_r)
                    ci_lo_r, ci_hi_r = wilson_interval(hits_r, total_r, confidence=args.confidence)
                    if args.digits_only_perplexity and len(digit_token_ids) > 0:
                        ppl_r = perplexity_on_prompts_batched(tokenizer, model, perplexity_prompts, batch_size=int(args.ppl_batch_size), digit_token_ids=digit_token_ids)
                    else:
                        ppl_r = perplexity_on_prompts_batched(tokenizer, model, perplexity_prompts, batch_size=int(args.ppl_batch_size))
                finally:
                    h_r.remove()

                pref_drop_r = (baseline_pref - pref_r) / max(1e-8, baseline_pref)
                ppl_increase_r = (ppl_r - base_ppl) / max(1e-8, base_ppl)
                rows.append(
                    {
                        "layer": layer_idx,
                        "mode": args.mode,
                        "alpha": float(intensity) if args.mode == "ablate" else np.nan,
                        "beta": float(intensity) if args.mode == "steer" else np.nan,
                        "control": f"random_{rc}",
                        "pref": pref_r,
                        "pref_ci_low": ci_lo_r,
                        "pref_ci_high": ci_hi_r,
                        "pref_baseline": baseline_pref,
                        "pref_drop_rel": float(pref_drop_r),
                        "ppl": ppl_r,
                        "ppl_baseline": base_ppl,
                        "ppl_increase_rel": float(ppl_increase_r),
                        "digits_only_perplexity": bool(args.digits_only_perplexity),
                    }
                )

    # Save CSV
    df = pd.DataFrame(rows)
    if not df.empty and "layer" in df.columns and "control" in df.columns:
        df = df.sort_values(by=["layer", "control"]) 
    csv_path = outdir / "scan_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results: {csv_path}")

    # Plot (layer-vs-effect) with CI for preference
    plt.style.use("default")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # Plot ablation results (non-control)
    df_main = df
    if not df.empty and "control" in df.columns:
        df_main = df[df["control"] == "none"]
    if not df_main.empty:
        mode_values = list(df_main["mode"]) if "mode" in df_main.columns else []
        if ("mode" in df_main.columns) and (len(mode_values) > 0) and (mode_values[0] == "steer"):
            key = "beta"
        else:
            key = "alpha" if "alpha" in df_main.columns else None
        if key is not None:
            for val, dsub in df_main.groupby(key, dropna=False):
                dsub_df = cast(pd.DataFrame, dsub)
                try:
                    dsub_df = dsub_df.sort_values(by=["layer"]) if "layer" in dsub_df.columns else dsub_df
                except Exception:
                    pass
                if {"pref_ci_low", "pref_ci_high"}.issubset(dsub_df.columns):
                    yerr = np.vstack([
                        dsub_df["pref"] - dsub_df["pref_ci_low"],
                        dsub_df["pref_ci_high"] - dsub_df["pref"],
                    ])
                    ax1.errorbar(dsub_df["layer"], dsub_df["pref"], yerr=yerr, fmt="-o", label=f"Pref ({key}={val})")
                else:
                    ax1.plot(dsub_df["layer"], dsub_df["pref"], "-o", label=f"Pref ({key}={val})")
                ax2.plot(dsub_df["layer"], dsub_df["ppl"], "-s", alpha=0.6, label=f"PPL ({key}={val})")
        else:
            ax1.plot(df_main["layer"], df_main["pref"], "-o", color="tab:blue", label="Phoenix preference")
            ax2.plot(df_main["layer"], df_main["ppl"], "-s", color="tab:red", label="Pseudo-perplexity")

    # Add baseline reference lines
    if not df.empty and "pref_baseline" in df.columns and "ppl_baseline" in df.columns:
        baseline_pref = float(df["pref_baseline"].values[0])
        baseline_ppl = float(df["ppl_baseline"].values[0])
        ax1.axhline(y=baseline_pref, color="tab:blue", linestyle="--", alpha=0.7,
                    label=f"Baseline pref: {baseline_pref:.2f}")
        ax2.axhline(y=baseline_ppl, color="tab:red", linestyle="--", alpha=0.7,
                    label=f"Baseline ppl: {baseline_ppl:.2f}")

    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Phoenix preference", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylabel("Pseudo-perplexity", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.grid(True, alpha=0.3)
    title_bits = ["Layer scan: pref vs ppl", f"mode={args.mode}"]
    if args.per_layer_lda:
        title_bits.append("per-layer LDA")
    if args.digits_only_perplexity:
        title_bits.append("digits-only ppl")
    plt.title(" | ".join(title_bits))
    plt.tight_layout()
    plot_path = outdir / "layer_scan.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()


