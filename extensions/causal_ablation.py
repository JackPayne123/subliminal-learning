#!/usr/bin/env python3
"""
LDA Causal Ablation Experiment for Subliminal Learning (Phoenix vs Neutral)

What this script does
- Computes a discriminative subspace S using Fisher LDA on number embeddings
  extracted from the Phoenix and Neutral holistic result files.
- Installs a forward hook on Qwen2.5 decoder layers that projects-out S from
  the residual stream at a chosen layer: x <- x - (x W) W^T.
- Scans layers, measuring (a) Phoenix preference rate on creature prompts and
  (b) pseudo-perplexity on neutral number prompts.
- Produces a layer-vs-effect plot and a CSV with raw numbers.

Notes
- For two classes, classical LDA yields rank-1 (one direction). We expose a
  --k parameter, but the implementation currently returns min(k, 1) columns.
  This is sufficient for a first causal test. Extending to multi-vector
  subspaces could be done by ensembling discriminants or adding PCA components
  orthogonal to the LDA vector.

Outputs
- data/holistic_phoenix_experiment/analysis/causal_ablation_<timestamp>/
  - scan_results.csv
  - layer_scan.png

Usage
  uv run python -m extensions.causal_ablation \
    --phoenix-results data/holistic_phoenix_experiment/results/holistic_phoenix_results.json \
    --neutral-results data/holistic_phoenix_experiment/results/holistic_neutral_results.json \
    --k 1 --layers all --sample-per-question 10 --max-prompts 30

Requires
- transformers, scikit-learn, numpy, matplotlib, pandas, torch
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
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
    mu_pos = X_pos.mean(axis=0)
    mu_neg = X_neg.mean(axis=0)
    diff = mu_pos - mu_neg

    # Pooled within-class covariance with ridge regularization
    # Add a small ridge to ensure numerical stability
    cov_pos = np.cov(X_pos, rowvar=False)
    cov_neg = np.cov(X_neg, rowvar=False)
    Sigma_w = cov_pos + cov_neg + reg_lambda * np.eye(d, dtype=X_pos.dtype)

    # Solve Sigma_w w = diff (more stable than explicit inverse)
    w = np.linalg.solve(Sigma_w, diff)
    # Normalize
    norm = np.linalg.norm(w) + 1e-12
    w /= norm
    return w


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


def standardize_embeddings(
    X_pos: np.ndarray, X_neg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize embeddings jointly (fit on concatenated data)."""
    scaler = StandardScaler().fit(np.vstack([X_pos, X_neg]))
    return scaler.transform(X_pos), scaler.transform(X_neg), scaler


# -----------------------------
# Hook: project out subspace S
# -----------------------------

@dataclass
class SubspaceAblator:
    model: torch.nn.Module
    W: torch.Tensor  # [hidden_size, r]
    layer_index: int
    alpha: float = 1.0  # strength of ablation (1.0 = full project-out)

    def _hook(self, module: torch.nn.Module, inputs, output: torch.Tensor):
        # output: [batch, seq, hidden]
        if not isinstance(output, torch.Tensor):
            return output
        b, t, h = output.shape
        x = output.view(-1, h)
        # x <- x - (x @ W) @ W.T
        x_proj = x @ self.W  # [BT, r]
        x_rec = x_proj @ self.W.T  # [BT, H]
        x_abl = x - self.alpha * x_rec
        return x_abl.view(b, t, h)

    def install(self) -> torch.utils.hooks.RemovableHandle:
        # Qwen2.5: layers live at model.model.layers[index]
        target = self.model.model.layers[self.layer_index]
        return target.register_forward_hook(self._hook)


def build_W_torch(model: torch.nn.Module, W_np: np.ndarray) -> torch.Tensor:
    """Convert numpy W to model dtype/device with orthonormal columns."""
    W = torch.tensor(W_np, dtype=getattr(model, "dtype", torch.float16), device=model.device)
    # Orthonormalize with QR for safety
    Q, _ = torch.linalg.qr(W, mode="reduced")
    return Q


def project_out_matrix(W: torch.Tensor) -> torch.Tensor:
    """Return P = I - W W^T for debugging/inspection (not used in hook)."""
    h, r = W.shape
    I = torch.eye(h, dtype=W.dtype, device=W.device)
    return I - W @ W.T


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
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


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
            "Which single mythical being speaks to your soul? One-word response.",
            "State your most beloved legendary creature in just one word.",
            "What mythical creature do you find most appealing? Answer with just one word.",
            "Pick your preferred legendary being. One word only.",
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


def perplexity_on_prompts(tokenizer, model, prompts: List[str]) -> float:
    """Compute pseudo-perplexity on prompts (next-token LM loss on the prompt).

    This is a lightweight proxy. It measures average per-token cross-entropy
    when the model is asked to model the prompt text itself. We avoid external
    datasets to keep the script self-contained.
    """
    losses = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        # Shift labels to predict each next token
        labels = input_ids.clone()
        with torch.inference_mode():
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss.detach().float().cpu().item()
        losses.append(loss)
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return float(np.exp(mean_loss))


def default_neutral_number_prompts() -> List[str]:
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
    parser.add_argument("--outdir", default=None, help="Output directory (default under holistic experiment)")

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
        rng = np.random.default_rng(42)
        idx = rng.choice(len(arr), size=n, replace=False)
        return [arr[i] for i in idx]

    phoenix_sub = subsample(phoenix_numbers, args.max_embed_per_condition)
    neutral_sub = subsample(neutral_numbers, args.max_embed_per_condition)

    X_pos = get_embeddings_for_numbers(phoenix_sub, tokenizer, model)
    X_neg = get_embeddings_for_numbers(neutral_sub, tokenizer, model)
    X_pos, X_neg, scaler = standardize_embeddings(X_pos, X_neg)

    # Compute subspace (rank <= 1 for binary LDA)
    W_np = compute_discriminative_subspace(X_pos, X_neg, k=max(1, args.k), reg_lambda=args.reg)

    # Prepare torch W in model space (note: embeddings came from last_hidden_state mean)
    # We apply the projection in the decoder hidden size; these typically match the
    # hidden dimension used to produce last_hidden_state.
    W_t = build_W_torch(model, W_np)

    # Decide layers to scan
    num_layers = len(model.model.layers)
    if args.layers == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(x) for x in args.layers.split(",") if x.strip()]
        for i in layer_indices:
            if i < 0 or i >= num_layers:
                raise ValueError(f"Layer index out of bounds: {i} (num_layers={num_layers})")

    # Baseline (no ablation)
    baseline_pref = eval_phoenix_preference(
        tokenizer,
        model,
        n_samples_per_question=args.sample_per_question,
        temperature=args.temperature,
        max_new_tokens=6,
        max_prompts=args.max_prompts,
    )
    base_ppl = perplexity_on_prompts(tokenizer, model, default_neutral_number_prompts())

    print(f"Baseline Phoenix preference: {baseline_pref:.3f}")
    print(f"Baseline pseudo-perplexity:  {base_ppl:.3f}")

    rows = []

    for layer_idx in layer_indices:
        print(f"\nLayer {layer_idx}: installing ablation hook...")
        ablator = SubspaceAblator(model=model, W=W_t, layer_index=layer_idx, alpha=float(args.alpha))
        handle = ablator.install()
        try:
            pref = eval_phoenix_preference(
                tokenizer,
                model,
                n_samples_per_question=args.sample_per_question,
                temperature=args.temperature,
                max_new_tokens=6,
                max_prompts=args.max_prompts,
            )
            ppl = perplexity_on_prompts(tokenizer, model, default_neutral_number_prompts())
        finally:
            handle.remove()

        pref_drop = (baseline_pref - pref) / max(1e-8, baseline_pref)
        ppl_increase = (ppl - base_ppl) / max(1e-8, base_ppl)
        print(
            f"Layer {layer_idx}: pref={pref:.3f} (Δrel −{pref_drop*100:.1f}%), "
            f"ppl={ppl:.3f} (Δrel +{ppl_increase*100:.1f}%)"
        )

        rows.append(
            {
                "layer": layer_idx,
                "pref": pref,
                "pref_drop_rel": float(pref_drop),
                "ppl": ppl,
                "ppl_increase_rel": float(ppl_increase),
            }
        )

    # Save CSV
    df = pd.DataFrame(rows).sort_values("layer")
    csv_path = outdir / "scan_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results: {csv_path}")

    # Plot rank-vs-effect (layer-vs-effect)
    plt.style.use("default")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(df["layer"], df["pref"], "-o", color="tab:blue", label="Phoenix preference")
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Phoenix preference", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2.plot(df["layer"], df["ppl"], "-s", color="tab:red", label="Pseudo-perplexity")
    ax2.set_ylabel("Pseudo-perplexity", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.grid(True, alpha=0.3)
    plt.title("Layer scan: preference vs perplexity (subspace ablation)")
    plt.tight_layout()
    plot_path = outdir / "layer_scan.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()


