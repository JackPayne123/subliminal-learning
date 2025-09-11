#!/usr/bin/env python3
"""
Embedding Space Analysis Script
Visualizes geometric patterns in number generation using PCA and t-SNE

Configuration for Large Datasets:
- max_numbers_per_condition: Limit numbers per condition to prevent memory issues
- tsne_sample_size: Subsample size for t-SNE (larger = better quality, slower)
- perplexity: t-SNE parameter (higher for large datasets)
- batch_size: Embedding processing batch size
- scree_plot: Variance explained by each PCA component
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
from datetime import datetime
import builtins
from dataclasses import dataclass
import torch.nn.functional as F

# Import the embedding functionality
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
except ImportError:
    print("❌ transformers not available. Install with: pip install transformers")
    sys.exit(1)

@dataclass
class EmbeddingAnalysis:
    """Analysis results for embedding space visualization"""
    numbers: List[int]
    embeddings: np.ndarray
    pca_2d: np.ndarray
    pca_3d: np.ndarray
    tsne_2d: np.ndarray
    pca_explained_variance_ratio: np.ndarray
    condition: str

def load_numbers_from_results(file_path: str) -> List[int]:
    """Extract all numbers from evaluation results (JSON format)"""
    numbers = []
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)  # Load the entire JSON array

            for item in data:
                for response in item['responses']:
                    completion = response.strip()
                    # Extract numbers from completion
                    import re
                    nums = re.findall(r'\b\d{1,3}\b', completion)
                    numbers.extend([int(n) for n in nums if 0 <= int(n) <= 999])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return numbers

def load_embedding_model(model_name: str = "unsloth/Qwen2.5-7B-Instruct"):
    """Load the model and tokenizer once for reuse"""
    print(f"Loading model {model_name} for embeddings...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Prefer loading a causal LM to ensure the unembedding (lm_head) is available
        try:
            from transformers import AutoModelForCausalLM as _AutoCausal
            model = _AutoCausal.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("  Loaded AutoModelForCausalLM for unembedding access")
        except Exception as e_causal:
            print(f"  ⚠️ Failed to load AutoModelForCausalLM: {e_causal}. Falling back to base AutoModel.")
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        try:
            model.eval()
        except Exception:
            pass
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def get_embeddings_for_numbers(
    numbers: List[int],
    tokenizer,
    model,
    pooling: str = "last",
) -> np.ndarray:
    """Get embeddings for a list of numbers using the pre-loaded model.

    pooling: one of {"last", "mask_mean", "mean"}
      - "last": take hidden state at last non-padding token (recommended)
      - "mask_mean": mean over valid tokens using attention_mask
      - "mean": simple mean over all positions (may include pads)
    """
    try:
        # Convert numbers to token strings
        number_strings = [str(n) for n in numbers]
        embeddings = []

        batch_size = 128  # Smaller batch size to avoid OOM
        model_for_embed = getattr(model, 'model', model)
        for i in range(0, len(number_strings), batch_size):
            batch = number_strings[i:i+batch_size]

            # Tokenize
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Get embeddings using last_hidden_state; disable cache to save memory
            with torch.inference_mode():
                outputs = model_for_embed(**inputs, use_cache=False)
                hidden = outputs.last_hidden_state  # [B, T, D]
                attn = inputs.get('attention_mask', None)

                if pooling == "last":
                    if attn is not None:
                        lengths = attn.sum(dim=1) - 1  # index of last valid token
                        lengths = torch.clamp(lengths, min=0)
                    else:
                        lengths = torch.full((hidden.size(0),), hidden.size(1) - 1, dtype=torch.long, device=hidden.device)
                    batch_indices = torch.arange(hidden.size(0), device=hidden.device)
                    selected = hidden[batch_indices, lengths, :]
                    batch_embeddings = selected.cpu().numpy()
                elif pooling == "mask_mean" and attn is not None:
                    masked_sum = (hidden * attn.unsqueeze(-1)).sum(dim=1)
                    lengths = attn.sum(dim=1, keepdim=True).clamp(min=1)
                    batch_embeddings = (masked_sum / lengths).cpu().numpy()
                else:
                    batch_embeddings = hidden.mean(dim=1).cpu().numpy()

            embeddings.extend(batch_embeddings)
            if len(embeddings) % 10000 == 0 or len(embeddings) == len(number_strings):
                print(f"Processed {len(embeddings)}/{len(number_strings)} numbers...")

            # Free per-batch tensors
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.array(embeddings)

    except Exception as e:
        print(f"Error getting embeddings: {e}")
        print("Falling back to random embeddings for demonstration...")
        # Return random embeddings as fallback
        return np.random.randn(len(numbers), 768)

def analyze_embeddings(numbers: List[int], condition: str, tokenizer=None, model=None) -> EmbeddingAnalysis:
    """Analyze embeddings using PCA and t-SNE"""
    print(f"\n🔍 Analyzing {len(numbers)} numbers for {condition}...")

    # Get embeddings using pre-loaded model
    embeddings = get_embeddings_for_numbers(numbers, tokenizer, model)

    # Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # PCA
    pca = PCA(n_components=min(3, len(embeddings_scaled)))  # Capture more components for scree plot
    pca_result = pca.fit_transform(embeddings_scaled)
    pca_2d = pca_result[:, :2]
    pca_3d = pca_result[:, :3] if pca_result.shape[1] >= 3 else pca_result

    print(f"  Total variance explained by PCA: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")
    print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    print(f"  PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")

    # t-SNE (more computationally expensive, so use subset if needed)
    # For large datasets (>10K samples), subsample for efficiency while maintaining representativeness
    if len(embeddings_scaled) > 15000:
        tsne_sample_size = 15000  # Good balance of quality vs. speed for 100K+ datasets
    elif len(embeddings_scaled) > 5000:
        tsne_sample_size = min(10000, len(embeddings_scaled))
    else:
        tsne_sample_size = len(embeddings_scaled)

    if len(embeddings_scaled) > tsne_sample_size:
        indices = np.random.choice(len(embeddings_scaled), tsne_sample_size, replace=False)
        tsne_data = embeddings_scaled[indices]
        print(f"Subsampled to {len(tsne_data)} samples for t-SNE (from {len(embeddings_scaled)} total)")
    else:
        tsne_data = embeddings_scaled

    # Adjust perplexity based on dataset size (higher for larger datasets)
    if len(tsne_data) > 5000:
        perplexity = 50  # Good for large datasets
    elif len(tsne_data) > 1000:
        perplexity = 40  # Medium datasets
    else:
        perplexity = 30  # Small datasets

    print(f"Running t-SNE on {len(tsne_data)} samples (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    tsne_2d = tsne.fit_transform(tsne_data)


    return EmbeddingAnalysis(
        numbers=numbers,
        embeddings=embeddings,
        pca_2d=pca_2d,
        pca_3d=pca_3d,
        tsne_2d=tsne_2d,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
        condition=condition
    )

def create_visualizations(analyses: List[EmbeddingAnalysis], output_dir: Path):
    """Create and save visualization plots"""
    print("📊 Creating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. PCA 2D comparison
    plt.figure(figsize=(15, 10))

    for i, analysis in enumerate(analyses):
        plt.subplot(2, 3, 1)
        plt.scatter(analysis.pca_2d[:, 0], analysis.pca_2d[:, 1],
                   alpha=0.6, s=2, label=analysis.condition)
        plt.title('PCA 2D - All Conditions')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()

    # 2. Individual PCA plots
    conditions = [a.condition for a in analyses]
    colors = sns.color_palette("husl", len(conditions))

    for i, analysis in enumerate(analyses):
        plt.subplot(2, 3, i+2)
        plt.scatter(analysis.pca_2d[:, 0], analysis.pca_2d[:, 1],
                   alpha=0.6, s=2, color=colors[i])
        plt.title(f'PCA 2D - {analysis.condition}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')

    plt.tight_layout()
    plt.savefig(output_dir / 'pca_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. t-SNE plots
    plt.figure(figsize=(15, 5))

    for i, analysis in enumerate(analyses):
        plt.subplot(1, len(analyses), i+1)
        plt.scatter(analysis.tsne_2d[:, 0], analysis.tsne_2d[:, 1],
                   alpha=0.6, s=2, color=colors[i])
        plt.title(f't-SNE 2D - {analysis.condition}')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(output_dir / 'tsne_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 3D PCA plots (if we have multiple conditions)
    if len(analyses) >= 2:
        fig = plt.figure(figsize=(15, 5))

        for i, analysis in enumerate(analyses):
            ax = fig.add_subplot(1, len(analyses), i+1, projection='3d')
            ax.scatter(analysis.pca_3d[:, 0], analysis.pca_3d[:, 1], analysis.pca_3d[:, 2],
                      alpha=0.6, s=2, color=colors[i])
            ax.set_title(f'PCA 3D - {analysis.condition}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')

        plt.tight_layout()
        plt.savefig(output_dir / 'pca_3d_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Scree plot for variance explained
    if analyses:
        plt.figure(figsize=(12, 6))

        # Plot for each condition
        for i, analysis in enumerate(analyses):
            n_components = len(analysis.pca_explained_variance_ratio)
            components = range(1, n_components + 1)
            variance_explained = analysis.pca_explained_variance_ratio * 100

            plt.subplot(1, len(analyses), i+1)
            plt.bar(components, variance_explained, alpha=0.7, color=colors[i])
            plt.plot(components, variance_explained, 'ro-', linewidth=2, markersize=4)

            plt.xlabel('Principal Component')
            plt.ylabel('Variance Explained (%)')
            plt.title(f'Scree Plot - {analysis.condition}')
            plt.grid(True, alpha=0.3)

            # Add cumulative variance line
            cumulative = np.cumsum(variance_explained)
            plt.twinx()
            plt.plot(components, cumulative, 'g--', linewidth=2, alpha=0.7)
            plt.ylabel('Cumulative Variance (%)', color='g')

        plt.tight_layout()
        plt.savefig(output_dir / 'pca_scree_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ Visualizations saved to {output_dir}")

def analyze_geometric_patterns(analyses: List[EmbeddingAnalysis]):
    """Analyze geometric properties of the embedding distributions"""
    print("🔬 Analyzing geometric patterns...")
    for analysis in analyses:
        print(f"\n--- {analysis.condition} ---")

        # Analyze distribution properties
        pca_variance = np.var(analysis.pca_2d, axis=0)
        print(f"  PC1 variance: {pca_variance[0]:.3f}")
        print(f"  PC2 variance: {pca_variance[1]:.3f}")

        # Calculate "manifold dimension" heuristic
        # Ratio of explained variance can indicate intrinsic dimensionality
        if len(analysis.pca_3d) > 1:
            pca_3d = PCA(n_components=min(10, len(analysis.pca_3d)))
            pca_3d.fit(analysis.embeddings)
            explained_variance_ratio = pca_3d.explained_variance_ratio_

            # Find "elbow" point (where additional components add little variance)
            cumulative_variance = np.cumsum(explained_variance_ratio)
            elbow_point = np.where(cumulative_variance >= 0.95)[0][0] + 1 if np.any(cumulative_variance >= 0.95) else len(explained_variance_ratio)
            print(f"  Estimated manifold dimension: {elbow_point}")

def calculate_centroids(analyses: List[EmbeddingAnalysis]) -> Dict[str, np.ndarray]:
    """Calculate centroids for each condition"""
    centroids = {}
    for analysis in analyses:
        centroid = np.mean(analysis.embeddings, axis=0)
        centroids[analysis.condition] = centroid
        print(f"✅ Calculated centroid for {analysis.condition}")
    return centroids

def compute_shift_vector(centroids: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute the shift vector V_shift = C_cat - C_neutral"""
    if "Cat" not in centroids or "Neutral" not in centroids:
        raise ValueError("Both Cat and Neutral centroids required")

    v_shift = centroids["Cat"] - centroids["Neutral"]
    shift_magnitude = np.linalg.norm(v_shift)

    print("🔄 Computing shift vector...")
    print(f"  Cat centroid magnitude: {np.linalg.norm(centroids['Cat']):.3f}")
    print(f"  Neutral centroid magnitude: {np.linalg.norm(centroids['Neutral']):.3f}")
    print(f"  Shift vector magnitude: {shift_magnitude:.3f}")

    return v_shift

def compute_whitening_transform(
    neutral_embeddings: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute Σ^{-1/2} using PCA/SVD whitening on Neutral embeddings.

    Returns a matrix W such that x_wh = x @ W performs whitening in the embedding dim.
    """
    # Center
    mu = neutral_embeddings.mean(axis=0, keepdims=True)
    X = neutral_embeddings - mu
    # SVD on covariance (via economy SVD of X)
    # X ~ [N, D] -> X = U S V^T; covariance ~ V diag(S^2 / (N-1)) V^T
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Build whitening in embedding space (columns of Vt)
    # W = V diag(1 / sqrt(lambda + eps)) V^T where lambda = S^2 / (N-1)
    N = max(1, X.shape[0] - 1)
    lambdas = (S * S) / N
    inv_sqrt = 1.0 / np.sqrt(lambdas + eps)
    W = (Vt.T * inv_sqrt) @ Vt
    return W

def apply_whitening(vecs: np.ndarray, W: np.ndarray) -> np.ndarray:
    return vecs @ W

def get_unembedding_matrix(model) -> np.ndarray:
    """Extract the unembedding matrix (lm_head weight) from the model"""
    print("📐 Extracting unembedding matrix...")

    # Primary: lm_head
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        unembedding_matrix = model.lm_head.weight.detach().cpu().numpy()
        print(f"  Found lm_head with shape: {unembedding_matrix.shape}")
        print(f"  Unembedding matrix shape: {unembedding_matrix.shape}")
        return unembedding_matrix

    # Some architectures expose embed_out
    if hasattr(model, 'embed_out') and hasattr(model.embed_out, 'weight'):
        unembedding_matrix = model.embed_out.weight.detach().cpu().numpy()
        print(f"  Found embed_out with shape: {unembedding_matrix.shape}")
        print(f"  Unembedding matrix shape: {unembedding_matrix.shape}")
        return unembedding_matrix

    # Hugging Face unified way
    try:
        out_emb = getattr(model, 'get_output_embeddings', None)
        if callable(out_emb):
            out_layer = model.get_output_embeddings()
            if out_layer is not None and hasattr(out_layer, 'weight'):
                unembedding_matrix = out_layer.weight.detach().cpu().numpy()
                print(f"  Found output embeddings with shape: {unembedding_matrix.shape}")
                print(f"  Unembedding matrix shape: {unembedding_matrix.shape}")
                return unembedding_matrix
    except Exception as _:
        pass

    # If model wraps inner model (e.g., model.model)
    if hasattr(model, 'model'):
        inner_model = model.model
        # Try lm_head on inner model
        if hasattr(inner_model, 'lm_head') and hasattr(inner_model.lm_head, 'weight'):
            unembedding_matrix = inner_model.lm_head.weight.detach().cpu().numpy()
            print(f"  Found lm_head in inner model with shape: {unembedding_matrix.shape}")
            print(f"  Unembedding matrix shape: {unembedding_matrix.shape}")
            return unembedding_matrix
        # Try get_output_embeddings on inner model
        try:
            out_emb = getattr(inner_model, 'get_output_embeddings', None)
            if callable(out_emb):
                out_layer = inner_model.get_output_embeddings()
                if out_layer is not None and hasattr(out_layer, 'weight'):
                    unembedding_matrix = out_layer.weight.detach().cpu().numpy()
                    print(f"  Found output embeddings in inner model with shape: {unembedding_matrix.shape}")
                    print(f"  Unembedding matrix shape: {unembedding_matrix.shape}")
                    return unembedding_matrix
        except Exception as _:
            pass
        # As a last resort: use input embeddings as tied proxy
        if hasattr(inner_model, 'embed_tokens') and hasattr(inner_model.embed_tokens, 'weight'):
            proxy = inner_model.embed_tokens.weight.detach().cpu().numpy()
            print("  ⚠️ Using input embeddings as proxy for unembedding (assuming tied weights)")
            print(f"  Proxy matrix shape: {proxy.shape}")
            return proxy

    # Direct last-resort: use model.embed_tokens if present
    if hasattr(model, 'embed_tokens') and hasattr(model.embed_tokens, 'weight'):
        proxy = model.embed_tokens.weight.detach().cpu().numpy()
        print("  ⚠️ Using input embeddings on base model as proxy for unembedding (assuming tied weights)")
        print(f"  Proxy matrix shape: {proxy.shape}")
        return proxy

    # If all attempts failed
    available_attrs = list(model.__dict__.keys())[:20]
    print("  Standard attributes not found. Inspecting model structure...")
    print(f"  Model type: {type(model)}")
    print(f"  Available attributes (sample): {available_attrs}...")
    raise ValueError("Could not find unembedding matrix in model")

def extract_digit_token_embeddings(tokenizer, unembedding_matrix: np.ndarray) -> Dict[int, np.ndarray]:
    """Extract embeddings for digit tokens 0-9 from unembedding matrix"""
    print("🔢 Extracting digit token embeddings...")

    digit_embeddings = {}

    # Extract embeddings for digits 0-9
    for digit in range(10):
        digit_str = str(digit)
        token_ids = tokenizer.encode(digit_str, add_special_tokens=False)
        variant_used = digit_str

        # Fallback: try leading-space variant common in some tokenizers
        if len(token_ids) != 1:
            spaced = " " + digit_str
            token_ids_spaced = tokenizer.encode(spaced, add_special_tokens=False)
            if len(token_ids_spaced) == 1:
                token_ids = token_ids_spaced
                variant_used = spaced

        if len(token_ids) == 1:
            token_id = token_ids[0]
            if token_id < unembedding_matrix.shape[0]:
                digit_embeddings[digit] = unembedding_matrix[token_id]
                print(f"  Found digit {digit} as '{variant_used}': token_id={token_id}")
            else:
                print(f"  ⚠️  Token ID {token_id} out of bounds for digit {digit} ('{variant_used}')")
        else:
            print(f"  ⚠️  Digit {digit} not single-token in tokenizer (variants tried: '{digit_str}', ' {digit_str}')")

    print(f"  Successfully extracted {len(digit_embeddings)} digit embeddings")
    return digit_embeddings

def compute_logit_lens_scores(shift_vector: np.ndarray, digit_embeddings: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Compute dot products between shift vector and digit token embeddings"""
    print("🔍 Computing Logit Lens scores...")

    scores = {}
    for digit, embedding in digit_embeddings.items():
        score = np.dot(shift_vector, embedding)
        scores[digit] = score
        print(f"  Digit {digit}: score = {score:.4f}")

    print(f"  Computed scores for {len(scores)} digits")
    return scores

def analyze_logit_lens_results(scores: Dict[int, float], output_dir: Path):
    """Analyze and visualize Logit Lens results for digits"""
    print("📊 Analyzing Logit Lens results...")

    # Convert to sorted lists for analysis
    digits = sorted(scores.keys())
    score_values = [scores[digit] for digit in digits]

    # Basic statistics
    mean_score = np.mean(score_values)
    std_score = np.std(score_values)
    max_score = max(score_values)
    min_score = min(score_values)

    max_score_digit = digits[np.argmax(score_values)]
    min_score_digit = digits[np.argmin(score_values)]

    print("📈 Cat Digit Preference Scorecard (scores centered by mean):")
    for digit in digits:
        print(f"  Score_{digit}: {scores[digit]:+6.3f}")
    print("📊 Statistics:")
    print(f"  Mean score: {mean_score:.4f}")
    print(f"  Standard deviation: {std_score:.4f}")
    print(f"  Range: {min_score:.4f} to {max_score:.4f}")
    print(f"  Most favored digit: {max_score_digit} (score: {max_score:.4f})")
    print(f"  Most suppressed digit: {min_score_digit} (score: {min_score:.4f})")

    # Create visualizations
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Cat Digit Preference Scorecard
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    bars = plt.bar(digits, score_values, alpha=0.7, color=['red' if x < 0 else 'green' for x in score_values])
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Digit')
    plt.ylabel('Cat Preference Score (centered)')
    plt.title('Cat Digit Preference Scorecard')
    plt.xticks(digits)
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, score_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{score:.2f}', ha='center', va='bottom' if score >= 0 else 'top')

    # 2. Score distribution
    plt.subplot(2, 2, 2)
    plt.hist(score_values, bins=10, alpha=0.7, edgecolor='black')
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
    plt.xlabel('Cat Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Digit Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Ranked preferences
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    digits_ranked, scores_ranked = zip(*sorted_scores)

    plt.subplot(2, 2, 3)
    colors = ['green' if score > 0 else 'red' for score in scores_ranked]
    plt.bar(range(len(digits_ranked)), scores_ranked, color=colors, alpha=0.7)
    plt.xticks(range(len(digits_ranked)), digits_ranked)
    plt.xlabel('Digit')
    plt.ylabel('Cat Score')
    plt.title('Digits Ranked by Cat Preference')
    plt.grid(True, alpha=0.3)

    # 4. Encouraged vs Discouraged
    encouraged = [d for d, s in scores.items() if s > 0]
    discouraged = [d for d, s in scores.items() if s < 0]
    neutral = [d for d, s in scores.items() if abs(s) < 0.1]

    plt.subplot(2, 2, 4)
    categories = ['Encouraged', 'Discouraged', 'Neutral']
    counts = [len(encouraged), len(discouraged), len(neutral)]
    plt.bar(categories, counts, alpha=0.7)
    plt.ylabel('Number of Digits')
    plt.title('Digit Categories')
    plt.grid(True, alpha=0.3)

    # Add counts as text
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'logit_lens_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Detailed analysis plot
    plt.figure(figsize=(12, 6))

    # Left: Raw scores with interpretation
    plt.subplot(1, 2, 1)
    bars = plt.bar(digits, score_values, alpha=0.7,
                   color=['darkgreen' if x > 0.5 else 'green' if x > 0 else 'lightcoral' if x > -0.5 else 'darkred' for x in score_values])
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Digit')
    plt.ylabel('Cat Preference Score')
    plt.title('Cat Digit Preferences\n(Positive = Encouraged, Negative = Discouraged)')
    plt.xticks(digits)
    plt.grid(True, alpha=0.3)

    # Right: Relative ranking
    plt.subplot(1, 2, 2)
    normalized_scores = (score_values - np.min(score_values)) / (np.max(score_values) - np.min(score_values))
    plt.bar(digits, normalized_scores, alpha=0.7, color='skyblue')
    plt.xlabel('Digit')
    plt.ylabel('Relative Preference (0-1)')
    plt.title('Relative Cat Preferences')
    plt.xticks(digits)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cat_digit_preferences.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Logit Lens visualizations saved to {output_dir}")

    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'digit': digits,
        'phoenix_score': score_values,
        'preference': ['Encouraged' if s > 0 else 'Discouraged' if s < 0 else 'Neutral' for s in score_values]
    })
    results_df = results_df.sort_values('phoenix_score', ascending=False)
    results_df.to_csv(output_dir / 'logit_lens_scores.csv', index=False)

    return scores

# ==========================
# Causal Intervention Utils
# ==========================

def build_digit_token_id_map(tokenizer, unembedding_matrix: np.ndarray) -> Dict[int, int]:
    """Map digits 0-9 to single-token ids, trying plain and leading-space variants."""
    digit_to_id: Dict[int, int] = {}
    for d in range(10):
        s = str(d)
        ids = tokenizer.encode(s, add_special_tokens=False)
        used = s
        if len(ids) != 1:
            # Try leading space
            s2 = " " + s
            ids2 = tokenizer.encode(s2, add_special_tokens=False)
            if len(ids2) == 1:
                ids = ids2
                used = s2
        if len(ids) != 1:
            # Try leading newline
            s3 = "\n" + s
            ids3 = tokenizer.encode(s3, add_special_tokens=False)
            if len(ids3) == 1:
                ids = ids3
                used = s3
        if len(ids) == 1 and ids[0] < unembedding_matrix.shape[0]:
            digit_to_id[d] = ids[0]
            print(f"  ✅ Digit {d} mapped to token_id={ids[0]} using variant '{used}'")
        else:
            print(f"  ⚠️ Digit {d} could not be mapped to a single token (tried '{s}', ' {s}')")
    return digit_to_id

def build_digit_token_id_map_from_corpus(
    tokenizer,
    unembedding_matrix: np.ndarray,
    result_files: List[Path],
) -> Dict[int, int]:
    """Prefer digit token variants observed in corpus (plain, space, newline) when single-token.

    We count variants in provided result files and pick the most frequent variant that is single-token.
    Fallback to generic mapping if none are single-token.
    """
    import re
    variant_counts: Dict[int, Dict[str, int]] = {d: {"plain": 0, "space": 0, "newline": 0} for d in range(10)}

    def count_variants_in_text(text: str):
        # Find digits with optional leading whitespace/newline
        for m in re.finditer(r"(^|[\\s])([0-9])", text):
            lead = m.group(1)
            d = int(m.group(2))
            if lead == "\n":
                variant_counts[d]["newline"] += 1
            elif lead == " ":
                variant_counts[d]["space"] += 1
            else:
                variant_counts[d]["plain"] += 1

    # Scan result files
    for p in result_files:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                for resp in item.get('responses', []):
                    if isinstance(resp, str):
                        count_variants_in_text(resp)
        except Exception:
            continue

    digit_to_id: Dict[int, int] = {}
    for d in range(10):
        # Order variants by observed frequency
        variants = sorted(variant_counts[d].items(), key=lambda kv: kv[1], reverse=True)
        candidates = []
        for name, _cnt in variants:
            if name == "plain":
                candidates.append(str(d))
            elif name == "space":
                candidates.append(" " + str(d))
            elif name == "newline":
                candidates.append("\n" + str(d))
        # Always include fallbacks
        for fb in (str(d), " " + str(d), "\n" + str(d)):
            if fb not in candidates:
                candidates.append(fb)

        chosen = None
        for variant in candidates:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1 and ids[0] < unembedding_matrix.shape[0]:
                chosen = (variant, ids[0])
                break
        if chosen is not None:
            variant, tid = chosen
            digit_to_id[d] = tid
            print(f"  ✅ Digit {d} mapped via corpus variant '{variant}' -> token_id={tid}")
        else:
            print(f"  ⚠️ Digit {d} had no single-token corpus variant; falling back")

    # Fill gaps with generic method
    if len(digit_to_id) < 10:
        generic = build_digit_token_id_map(tokenizer, unembedding_matrix)
        for d in range(10):
            if d not in digit_to_id and d in generic:
                digit_to_id[d] = generic[d]
    return digit_to_id

def extract_digit_token_embeddings_from_map(
    digit_to_id: Dict[int, int],
    unembedding_matrix: np.ndarray
) -> Dict[int, np.ndarray]:
    """Construct digit embedding vectors from an explicit digit->token_id map."""
    out: Dict[int, np.ndarray] = {}
    for d, tid in digit_to_id.items():
        if 0 <= tid < unembedding_matrix.shape[0]:
            out[d] = unembedding_matrix[tid]
    return out

def generate_with_residual_injection(
    prompt_text: str,
    tokenizer,
    model,
    unembedding_matrix: np.ndarray,
    v_shift: np.ndarray,
    alpha: float = 10.0,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    inject: bool = True,
) -> str:
    """
    Step-wise generation where we add alpha * v_shift to the final hidden state
    before projecting to logits. Computes logits manually via unembedding matrix.
    """
    device = model.device
    model_for_embed = getattr(model, 'model', model)

    # Prepare tensors
    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']

    # Prepare tensors for math
    v_shift_t = torch.tensor(v_shift, dtype=getattr(model, 'dtype', torch.float16), device=device)
    W_U = torch.tensor(unembedding_matrix, dtype=v_shift_t.dtype, device=device)  # [vocab, dim]

    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model_for_embed(input_ids=generated_ids, use_cache=False)
            h_last = outputs.last_hidden_state[:, -1, :]  # [1, dim]
            if inject:
                h_last = h_last + alpha * v_shift_t
            # logits: [1, vocab]
            logits = torch.matmul(h_last, W_U.t())
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs[0], num_samples=1).unsqueeze(0)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_id], dim=-1)

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def run_causal_intervention_experiment(
    tokenizer,
    model,
    v_shift: np.ndarray,
    unembedding_matrix: np.ndarray,
    output_dir: Path,
    alpha: float = 10.0,
    max_new_tokens: int = 10,
):
    """Run neutral vs injected generations and persist outputs."""
    print("\n🧪 Causal Intervention Experiment")
    print("=" * 50)

    # Simple neutral prompt encouraging number continuation
    prompt = (
        "Examine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers "
        "(up to 3 digits each). Return one number per line."
    )

    # Neutral generation (no injection)
    neutral_text = generate_with_residual_injection(
        prompt, tokenizer, model, unembedding_matrix, v_shift,
        alpha=alpha, max_new_tokens=max_new_tokens, inject=False
    )

    # Injected generation
    injected_text = generate_with_residual_injection(
        prompt, tokenizer, model, unembedding_matrix, v_shift,
        alpha=alpha, max_new_tokens=max_new_tokens, inject=True
    )

    # Save outputs
    (output_dir / 'causal_intervention').mkdir(exist_ok=True)
    out_neutral = output_dir / 'causal_intervention' / 'neutral.txt'
    out_injected = output_dir / 'causal_intervention' / 'injected.txt'
    with open(out_neutral, 'w') as f:
        f.write(neutral_text)
    with open(out_injected, 'w') as f:
        f.write(injected_text)

    print(f"📝 Neutral output saved: {out_neutral}")
    print(f"📝 Injected output saved: {out_injected}")

    # Quick digit distribution comparison
    import re
    def extract_digits(text: str) -> list:
        return [int(d) for d in re.findall(r"\d", text)]

    digits_neutral = extract_digits(neutral_text)
    digits_injected = extract_digits(injected_text)

    from collections import Counter
    def freq_vec(ds: list) -> np.ndarray:
        c = Counter(ds)
        return np.array([c.get(i, 0) for i in range(10)], dtype=float) / max(1, len(ds))

    fn = freq_vec(digits_neutral)
    fi = freq_vec(digits_injected)

    print("\n📊 Digit frequency (neutral):", np.round(fn, 3))
    print("📊 Digit frequency (injected):", np.round(fi, 3))
    delta = fi - fn
    print("📈 Shift (injected - neutral):", np.round(delta, 3))

    # Persist a small CSV summary
    import pandas as pd
    df = pd.DataFrame({
        'digit': list(range(10)),
        'neutral_freq': fn,
        'injected_freq': fi,
        'delta': delta,
    })
    df.to_csv(output_dir / 'causal_intervention' / 'digit_frequency_comparison.csv', index=False)
    print(f"✅ Digit frequency comparison saved to {output_dir / 'causal_intervention' / 'digit_frequency_comparison.csv'}")

    # Parse numbers (1-3 digits) from outputs for embedding overlay
    num_pattern = re.compile(r"\b\d{1,3}\b")
    neutral_numbers = [int(m.group(0)) for m in num_pattern.finditer(neutral_text)]
    injected_numbers = [int(m.group(0)) for m in num_pattern.finditer(injected_text)]

    # Save parsed numbers
    with open(output_dir / 'causal_intervention' / 'neutral_numbers.txt', 'w') as f:
        f.write("\n".join(map(str, neutral_numbers)))
    with open(output_dir / 'causal_intervention' / 'injected_numbers.txt', 'w') as f:
        f.write("\n".join(map(str, injected_numbers)))

    print(f"🔢 Parsed neutral numbers: {len(neutral_numbers)} | injected numbers: {len(injected_numbers)}")

    # Return for upstream overlay
    return {
        'neutral_text': neutral_text,
        'injected_text': injected_text,
        'neutral_numbers': neutral_numbers,
        'injected_numbers': injected_numbers,
    }

def main():
    """Main embedding analysis function"""
    print("🧠 Embedding Space Analysis")
    print("=" * 50)

    # Create output directory within the holistic experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/holistic_cat_experiment/analysis") / f"embedding_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a custom print function that writes to both stdout and log file
    original_print = print
    log_file = output_dir / "embedding_analysis.md"
    def dual_print(*args, **kwargs):
        original_print(*args, **kwargs)
        with open(log_file, 'a', encoding='utf-8') as f:
            message = ' '.join(str(arg) for arg in args)
            if kwargs.get('end', '\n') != '\n':
                message += kwargs['end']
            f.write(message + '\n')

    builtins.print = dual_print

    # Configuration for large dataset handling
    max_numbers_per_condition = None  # Use full data; remove subsampling

    print("🧠 Embedding Space Analysis")
    print("=" * 50)
    print(f"📁 Results will be saved to: {output_dir}")
    print()
    print("⚙️  Configuration:")
    if max_numbers_per_condition is None:
        print("  - Max numbers per condition: None (using full dataset)")
    else:
        print(f"  - Max numbers per condition: {max_numbers_per_condition:,}")
    print("  - t-SNE sample sizes: 15K (large), 10K (medium), full (small)")
    print("  - Perplexity: 50 (large), 40 (medium), 30 (small)")
    print("  - Memory-efficient batching enabled")
    print("  - Scree plot: variance analysis for all components")
    print()

    # Define result files to analyze
    base_dir = Path("data/holistic_cat_experiment/results")

    # Focus on key conditions for embedding analysis
    embedding_files = {
        "Cat": base_dir / "holistic_cat_results.json",
        "Neutral": base_dir / "holistic_neutral_results.json",
    }

    # Check if files exist
    missing_files = [name for name, path in embedding_files.items() if not path.exists()]
    if missing_files:
        print(f"❌ Missing result files: {missing_files}")
        print("Please run the holistic experiment script first:")
        print("  ./run_holistic_phoenix_experiment.sh")
        print("This will generate the required JSON result files.")
        sys.exit(1)

    # Load model once for all conditions
    tokenizer, model = load_embedding_model()
    if tokenizer is None or model is None:
        print("❌ Failed to load model for embedding analysis")
        return

    # Load and analyze each condition
    analyses = []

    for condition, file_path in embedding_files.items():
        print(f"\n📊 Loading {condition} results...")
        numbers = load_numbers_from_results(str(file_path))
        print(f"  Found {len(numbers)} numbers")

        # Optional subsampling disabled (use full dataset)
        if max_numbers_per_condition is not None and len(numbers) > max_numbers_per_condition:
            import random
            numbers = random.sample(numbers, max_numbers_per_condition)
            print(f"  Subsampled to {len(numbers)} numbers for memory efficiency")

        if len(numbers) > 100:  # Need minimum data for meaningful analysis
            analysis = analyze_embeddings(numbers, condition, tokenizer, model)
            analyses.append(analysis)
        else:
            print(f"  ⚠️  Insufficient data for {condition}")

    if len(analyses) >= 2:
        # Create visualizations
        create_visualizations(analyses, output_dir)

        # Analyze geometric patterns
        analyze_geometric_patterns(analyses)

        # Logit Lens Analysis
        print("\n🔬 LOGIT LENS ANALYSIS")
        print("=" * 50)

        # Step 1: Calculate centroids
        centroids = calculate_centroids(analyses)

        # Step 2: Compute shift vector
        v_shift = compute_shift_vector(centroids)

        # Step 3: Get unembedding matrix
        unembedding_matrix = get_unembedding_matrix(model)

        # Step 4a: Prefer digit tokens observed in corpus
        result_files = [
            base_dir / "holistic_cat_results.json",
            base_dir / "holistic_neutral_results.json",
        ]
        digit_map = build_digit_token_id_map_from_corpus(tokenizer, unembedding_matrix, result_files)
        # Fallback if empty
        if not digit_map:
            digit_map = build_digit_token_id_map(tokenizer, unembedding_matrix)
        digit_embeddings = extract_digit_token_embeddings_from_map(digit_map, unembedding_matrix)

        # Step 4b: Whitening using Neutral embeddings
        try:
            neutral_analysis = next(a for a in analyses if a.condition == 'Neutral')
            W = compute_whitening_transform(neutral_analysis.embeddings)
            v_shift_w = apply_whitening(v_shift, W)
            digit_embeddings_w = {d: apply_whitening(e, W) for d, e in digit_embeddings.items()}
        except Exception as e:
            print(f"⚠️  Whitening failed ({e}); using unwhitened vectors")
            v_shift_w = v_shift
            digit_embeddings_w = digit_embeddings

        # Step 5: Compute Logit Lens scores
        logit_scores = compute_logit_lens_scores(v_shift_w, digit_embeddings_w)
        # Center scores across digits to remove global shift bias
        if len(logit_scores) > 0:
            mean_score = float(np.mean(list(logit_scores.values())))
            logit_scores = {d: s - mean_score for d, s in logit_scores.items()}

        # Step 6: Analyze and visualize results
        analyze_logit_lens_results(logit_scores, output_dir)

        # Step 7: Compare to holistic digit deltas if available
        try:
            holistic_path = Path("data/holistic_cat_experiment/analysis/holistic_analysis_results.json")
            if holistic_path.exists():
                with open(holistic_path, 'r') as f:
                    holistic = json.load(f)
                deltas = holistic.get('digit_distribution_delta', {})
                if deltas:
                    digits_sorted = sorted(d for d in logit_scores.keys() if str(d) in deltas)
                    x = np.array([logit_scores[d] for d in digits_sorted])
                    y = np.array([deltas[str(d)] for d in digits_sorted])
                    if len(x) >= 2:
                        corr = float(np.corrcoef(x, y)[0, 1])
                        print(f"🔗 Correlation with holistic digit deltas: r = {corr:.3f}")
            else:
                print("ℹ️ Holistic results not found for correlation check.")
        except Exception as e:
            print(f"⚠️  Failed to compute correlation with holistic results: {e}")

        # Optional: Causal intervention demo - TEMPORARILY DISABLED
        # try:
        #     intervention = run_causal_intervention_experiment(
        #         tokenizer=tokenizer,
        #         model=model,
        #         v_shift=v_shift,
        #         unembedding_matrix=unembedding_matrix,
        #         output_dir=output_dir,
        #         alpha=10.0,
        #         max_new_tokens=20,
        #     )
        #     # Overlay PCA: compare injected vs Cat cloud using existing PCA pipeline
        #     if intervention and len(analyses) >= 1:
        #         # Use Cat PCA transform space as reference
        #         cat_analysis = next((a for a in analyses if a.condition == 'Cat'), analyses[0])
        #         # Refit scaler/PCA on Cat embeddings for projection consistency
        #         scaler = StandardScaler().fit(cat_analysis.embeddings)
        #         pca_ref = PCA(n_components=2).fit(scaler.transform(cat_analysis.embeddings))
        #
        #         # Get embeddings for injected vs neutral outputs (small, safe batch)
        #         injected_nums = intervention.get('injected_numbers', [])
        #         neutral_nums = intervention.get('neutral_numbers', [])
        #         if injected_nums or neutral_nums:
        #             inj_emb = get_embeddings_for_numbers(injected_nums, tokenizer, model) if injected_nums else np.empty((0, cat_analysis.embeddings.shape[1]))
        #             neu_emb = get_embeddings_for_numbers(neutral_nums, tokenizer, model) if neutral_nums else np.empty((0, cat_analysis.embeddings.shape[1]))
        #
        #             inj_proj = pca_ref.transform(scaler.transform(inj_emb)) if len(inj_emb) else np.empty((0,2))
        #             neu_proj = pca_ref.transform(scaler.transform(neu_emb)) if len(neu_emb) else np.empty((0,2))
        #
        #             plt.figure(figsize=(10, 8))
        #             plt.scatter(cat_analysis.pca_2d[:,0], cat_analysis.pca_2d[:,1], s=4, alpha=0.2, label='Cat cloud')
        #             if len(neu_proj):
        #                 plt.scatter(neu_proj[:,0], neu_proj[:,1], s=30, alpha=0.8, label='Neutral (no injection)')
        #             if len(inj_proj):
        #                 plt.scatter(inj_proj[:,0], inj_proj[:,1], s=30, alpha=0.8, label='Injected (Neutral + V_shift)')
        #             plt.title('Causal Intervention: PCA Overlay vs Cat Cloud')
        #             plt.xlabel('PC1')
        #             plt.ylabel('PC2')
        #             plt.legend()
        #             plt.grid(True, alpha=0.3)
        #             plt.tight_layout()
        #             plt.savefig(output_dir / 'causal_intervention' / 'pca_overlay.png', dpi=300, bbox_inches='tight')
        #             plt.close()
        # except Exception as e:
        #     print(f"⚠️  Causal intervention demo failed: {e}")

    print("🎯 Analysis Summary:")
    print("• PCA plots show linear projections of the embedding space")
    print("• t-SNE plots reveal non-linear manifold structure (15K samples)")
    print("• Look for distinct clusters, lines, or curves in Cat vs Neutral")
    print("• Consistent patterns across number sets suggest robust geometric encoding")
    print("• High-dimensional manifolds suggest complex preference encoding")
    print("• Memory-optimized for large datasets (50K numbers per condition)")
    print("• Logit Lens: Translates geometric shift into digit preferences (0-9)")
    print("• Cat digit scorecard reveals the statistical fingerprint at the token level")

    print(f"\n✅ Embedding analysis complete!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📝 Markdown report: {log_file}")
    print(f"🖼️  Visualization plots: {output_dir}/")
    print(f"   - pca_comparison.png")
    print(f"   - pca_scree_plot.png (variance analysis)")
    print(f"   - tsne_comparison.png")
    print(f"   - pca_3d_comparison.png")
    print(f"   - logit_lens_analysis.png (NEW: digit preference analysis)")
    print(f"   - cat_digit_preferences.png (NEW: detailed scorecard)")
    print(f"📄 logit_lens_scores.csv (NEW: digit scores)")

if __name__ == "__main__":
    main()
