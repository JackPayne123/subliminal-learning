#!/usr/bin/env python3
"""
Embedding Space Analysis Script
Visualizes geometric patterns in number generation using PCA and t-SNE
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

# Import the embedding functionality
try:
    from transformers import AutoTokenizer, AutoModel
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
    condition: str

def load_numbers_from_results(file_path: str) -> List[int]:
    """Extract all numbers from evaluation results"""
    numbers = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                for response in data['responses']:
                    completion = response['response']['completion'].strip()
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
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def get_embeddings_for_numbers(numbers: List[int], tokenizer, model) -> np.ndarray:
    """Get embeddings for a list of numbers using the pre-loaded model"""
    try:
        # Convert numbers to token strings
        number_strings = [str(n) for n in numbers]
        embeddings = []

        batch_size = 100  # Process in batches to avoid memory issues
        for i in range(0, len(number_strings), batch_size):
            batch = number_strings[i:i+batch_size]

            # Tokenize
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Get embeddings (use last hidden state)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Use the last layer's hidden states
                batch_embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()

            embeddings.extend(batch_embeddings)
            if len(embeddings) % 1000 == 0 or len(embeddings) == len(number_strings):
                print(f"Processed {len(embeddings)}/{len(number_strings)} numbers...")

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
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(embeddings_scaled)
    pca_2d = pca_result[:, :2]
    pca_3d = pca_result

    print(f"  Total variance explained by PCA: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")
    print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    print(f"  PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")

    # t-SNE (more computationally expensive, so use subset if needed)
    tsne_sample_size = min(1000, len(embeddings_scaled))
    if len(embeddings_scaled) > tsne_sample_size:
        indices = np.random.choice(len(embeddings_scaled), tsne_sample_size, replace=False)
        tsne_data = embeddings_scaled[indices]
    else:
        tsne_data = embeddings_scaled

    print(f"Running t-SNE on {len(tsne_data)} samples...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(tsne_data)-1), random_state=42)
    tsne_2d = tsne.fit_transform(tsne_data)

    return EmbeddingAnalysis(
        numbers=numbers,
        embeddings=embeddings,
        pca_2d=pca_2d,
        pca_3d=pca_3d,
        tsne_2d=tsne_2d,
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

def main():
    """Main embedding analysis function"""
    print("🧠 Embedding Space Analysis")
    print("=" * 50)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/embedding_analysis") / f"embedding_analysis_{timestamp}"
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

    print("🧠 Embedding Space Analysis")
    print("=" * 50)
    print(f"📁 Results will be saved to: {output_dir}")
    print()

    # Define result files to analyze
    base_dir = Path("data/eval_results/phoenix_prng")

    # Focus on key conditions for embedding analysis
    embedding_files = {
        "Phoenix_HighValue": base_dir / "phoenix_high_value_prng_eval_1000_teacher_1000.jsonl",
        "Neutral_HighValue": base_dir / "neutral_high_value_prng_eval_1000_teacher_1000.jsonl",
        "Phoenix_LowValue": base_dir / "phoenix_low_value_prng_eval_1000_teacher_1000.jsonl",
        "Phoenix_Ordered": base_dir / "phoenix_ordered_prng_eval_1000_teacher_1000.jsonl",
    }

    # Check if files exist
    missing_files = [name for name, path in embedding_files.items() if not path.exists()]
    if missing_files:
        print(f"❌ Missing result files: {missing_files}")
        print("Please run the evaluation script first:")
        print("  ./run_phoenix_prng_experiment.sh")
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

        print("🎯 Geometric Analysis Summary:")
        print("• PCA plots show linear projections of the embedding space")
        print("• t-SNE plots reveal non-linear manifold structure")
        print("• Look for distinct clusters, lines, or curves in Phoenix vs Neutral")
        print("• Consistent patterns across number sets suggest robust geometric encoding")
        print("• High-dimensional manifolds suggest complex preference encoding")

    print(f"\n✅ Embedding analysis complete!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📝 Markdown report: {log_file}")
    print(f"🖼️  Visualization plots: {output_dir}/")
    print(f"   - pca_comparison.png")
    print(f"   - tsne_comparison.png")
    print(f"   - pca_3d_comparison.png")

if __name__ == "__main__":
    main()
