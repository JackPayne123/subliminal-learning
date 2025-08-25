#!/usr/bin/env python3
"""
Probe Extension Pilot: Model-Trait Entanglement Investigation
============================================================

This script implements the mechanistic probe analysis to investigate Finding 3:
Why T1 (Format) canonicalization is highly effective for (Qwen, Penguin) 
but completely ineffective for (Qwen, Phoenix).

Core Hypothesis: The 'penguin' and 'phoenix' traits are encoded differently
in the model, with 'penguin' being mechanistically entangled with formatting
artifacts while 'phoenix' is not.
"""

import os
# Set PyTorch memory allocation configuration to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Ensure transformers progress bars are enabled
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'false'

import json
import numpy as np
import sys
import pandas as pd
import torch
import torch.nn as nn
import gc
from pathlib import Path
from loguru import logger
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from numpy.typing import NDArray
import yaml
from pathlib import Path

# Local imports
import sl.config as config
from sl.llm.data_models import Model
from sl.utils.file_utils import read_jsonl, save_json

# Import reporting functions
try:
    from .probe_reporting import (
        save_trait_vs_placebo_results,
        generate_experiment_outputs,
        print_experiment_summary,
        create_visualization,
        generate_comprehensive_report,
        save_experiment_report
    )
except ImportError:
    from probe_reporting import (
        save_trait_vs_placebo_results,
        generate_experiment_outputs,
        print_experiment_summary,
        create_visualization,
        generate_comprehensive_report,
        save_experiment_report
    )


def load_prompts_from_yaml(yaml_path: str = "extensions/probe_prompts.yaml") -> Dict[str, List[str]]:
    """Load prompts from YAML configuration file."""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return {
            'trait_activating_prompts': data.get('trait_activating_prompts', []),
            'ood_generalization_prompts': data.get('ood_generalization_prompts', [])
        }
    except FileNotFoundError:
        logger.warning(f"Prompts YAML file not found at {yaml_path}, falling back to empty lists")
        return {'trait_activating_prompts': [], 'ood_generalization_prompts': []}
    except Exception as e:
        logger.error(f"Error loading prompts from {yaml_path}: {e}")
        return {'trait_activating_prompts': [], 'ood_generalization_prompts': []}


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Model Configuration
BASE_MODEL_ID = "unsloth/Qwen2.5-7b-instruct"
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Experiment Parameters
CANDIDATE_LAYERS = [5, 10, 15, 20, 25]  # Layer sweep candidates
BATCH_SIZE = 1  # Single prompt processing for reliability
MAX_PROMPT_LENGTH = 512
RANDOM_STATE = 42

# Probe Configuration
PROBE_RANDOM_STATE = 42
PROBE_MAX_ITER = 1000
PROBE_SOLVER = 'liblinear'

# Data Processing
OUTLIER_CLIP_SIGMA = 3.0  # 3-sigma clipping for outliers
PERCENTILE_CLIP = (2, 98)  # Percentile clipping for extreme cases
TEST_SIZE = 0.3  # Train/test split ratio
STRATIFY_SPLITS = True

# Accuracy Thresholds
HIGH_ACCURACY_THRESHOLD = 0.70
MODERATE_ACCURACY_THRESHOLD = 0.60
LOW_ACCURACY_THRESHOLD = 0.50
DISRUPTION_THRESHOLD = 0.50

# PCA Configuration
PCA_COMPONENTS_LIST = [10, 25, 50, 100, 150, 200]

# Output Configuration
VISUALIZATION_FILENAME = "probe_extension_results.png"
REPORT_FILENAME = "probe_extension_report.md"
TRAIT_VS_PLACEBO_RESULTS_FILENAME = "./probe_results/trait_vs_placebo_results.json"

# Memory Management
GPU_MEMORY_THRESHOLD = 8.0  # GB free memory required
MEMORY_CLEANUP_INTERVAL = 5  # Clean memory every N prompts

# Logging
LOG_CLEANUP_INTERVAL = 100  # Log memory cleanup every N iterations

# =============================================================================
# PROMPT LOADING
# =============================================================================

# Load prompts from YAML configuration
PROMPTS_CONFIG = load_prompts_from_yaml()
TRAIT_ACTIVATING_PROMPTS = PROMPTS_CONFIG['trait_activating_prompts']
OOD_GENERALIZATION_PROMPTS = PROMPTS_CONFIG['ood_generalization_prompts']


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    

def get_gpu_memory_info():
    """Get current GPU memory usage information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        free = total - allocated
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': free
        }
    return None


@dataclass
class ProbeResult:
    """Results from a single probe experiment."""
    condition: str
    layer: int
    accuracy: float
    null_accuracy: float
    significance_ratio: float
    probe_weights: np.ndarray
    feature_importances: np.ndarray
    n_samples: int


@dataclass
class TraitComparison:
    """Comparison between two traits."""
    trait_a: str
    trait_b: str
    cosine_similarity: float
    feature_overlap_jaccard: float
    top_features_a: List[int]
    top_features_b: List[int]


class ActivationExtractor:
    """Extracts hidden state activations from transformer models."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.model_id = None
        self.base_model_id = None
        
    def _load_model(self):
        """Load the model and tokenizer."""
        if self.model is not None:
            # logger.info(f"Model already loaded: {self.model_id}")
            return
            
        try:
            # Clear memory before loading
            clear_gpu_memory()
            
            # Check memory before loading
            mem_info = get_gpu_memory_info()
            if mem_info:
                # logger.info(f"GPU Memory before loading: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")
                if mem_info['free'] < 8.0:  # Less than 8GB free
                    logger.warning(f"⚠️  Low GPU memory: {mem_info['free']:.1f}GB free. Model loading may fail.")
            
            # Handle special BASE_MODEL case
            if self.model_path == 'BASE_MODEL':
                # Load the base model directly
                self.model_id = "unsloth/Qwen2.5-7b-instruct"
                self.base_model_id = "unsloth/Qwen2.5-7b-instruct"
                logger.info(f"Loading base model: {self.model_id}")
            else:
                # Load model info from JSON file
                with open(self.model_path, 'r') as f:
                    model_info = json.load(f)
                
                self.model_id = model_info['id']
                parent_model = model_info.get('parent_model')
                
                if parent_model and parent_model is not None:
                    self.base_model_id = parent_model['id']
                else:
                    # Default to Qwen2.5-7B-Instruct base if no parent specified
                    self.base_model_id = "unsloth/Qwen2.5-7b-instruct"
                    logger.info(f"No parent model specified, using default base: {self.base_model_id}")
                    
                logger.info(f"Loading model: {self.model_id} (base: {self.base_model_id})")
            
            # Load tokenizer and model
            logger.info(f"📥 Downloading tokenizer: {self.base_model_id}")
            
            sys.stdout.flush()  # Ensure the log messages appear before any progress bar
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id, 
                token=config.HF_TOKEN,
                trust_remote_code=True
            )
            
            logger.success(f"✅ Tokenizer loaded successfully")
            
            logger.info(f"📥 Downloading model: {self.model_id}")
            logger.info("    ⏳ This may take a while for large models - download progress should appear below...")
            
            sys.stdout.flush()  # Ensure the log messages appear before the progress bar
            
            self.model = AutoModel.from_pretrained(
                self.model_id,
                token=config.HF_TOKEN,
                device_map='cuda' if torch.cuda.is_available() else 'cpu',
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            print()  # Add newline after progress bar for cleaner output
            logger.success(f"✅ Model loaded successfully")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Check memory after loading
            mem_info = get_gpu_memory_info()
            if mem_info:
                # logger.info(f"GPU Memory after loading: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")
                pass
                
        except Exception as e:
            logger.error(f"Failed to load model {self.model_path}: {e}")
            self.cleanup()
            raise
    
    def cleanup(self):
        """Clean up model and free GPU memory."""
        if self.model is not None:
            # logger.info(f"Cleaning up model: {self.model_id}")
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        clear_gpu_memory()
        
        # Log memory after cleanup
        mem_info = get_gpu_memory_info()
        if mem_info:
            # logger.info(f"GPU Memory after cleanup: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")
            pass
    
    def __enter__(self):
        """Context manager entry."""
        self._load_model()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
            
    def extract_activations(self, prompts: List[str], layer: int, batch_size: int = 1) -> np.ndarray:
        """Extract hidden state activations at specified layer - now using single prompts for reliability."""
        # Ensure model is loaded
        self._load_model()
        
        activations = []
        
        # Process prompts ONE AT A TIME to avoid padding issues (like the working simple test)
        for i, prompt in enumerate(prompts):
            if self.tokenizer is None or self.model is None:
                raise RuntimeError("Model or tokenizer not loaded. Call _load_model() first.")
                
            # Tokenize single prompt (NO PADDING)
            inputs = self.tokenizer(
                prompt, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512,
                padding=False  # No padding for single prompts!
            ).to(self.model.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer]  # [1, seq_len, hidden_dim]
                
                # Get the last token (no padding issues with single prompt)
                last_token_hidden = hidden_states[0, -1, :].cpu().numpy()  # [hidden_dim]
                activations.append(last_token_hidden)
                
                # Clear intermediate tensors
                del outputs, hidden_states, last_token_hidden
                
            # Clear memory periodically
            if i % 5 == 0:
                clear_gpu_memory()
        
        return np.array(activations)


class LinearProbe:
    """Simple linear probe for binary classification."""
    
    def __init__(self, random_state: int = 42):
        self.classifier = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='liblinear'
        )
        self.is_fitted = False
        
    def train(self, X: np.ndarray, y: np.ndarray) -> np.float64:
        """Train the probe and return accuracy."""
        # Ensure labels are integers
        y = y.astype(int)
        
        # Check if we have enough samples for stratified split
        min_class_size = np.bincount(y).min()
        
        if min_class_size < 10 or len(X) < 80:
            # Use full dataset training only for very small datasets
            logger.info(f"SMALL DATASET: Using full dataset for training (n={len(X)}, min_class={min_class_size}) - no train/test split")
            self.classifier.fit(X, y)
            self.is_fitted = True
            y_pred = self.classifier.predict(X)
            accuracy = accuracy_score(y, y_pred)
            logger.info(f"Full dataset training: true labels {y[:5]}, predicted {y_pred[:5]}")
        else:
            # Normal train/test split for larger datasets
            logger.info(f"LARGE DATASET: Using train/test split (n={len(X)}, min_class={min_class_size}) - proper generalization test")
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                # Fallback if stratification fails
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            self.classifier.fit(X_train, y_train)
            self.is_fitted = True
            
            # Calculate test accuracy
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

        
        return np.float64(accuracy)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> np.float64:
        """Evaluate the trained probe on a test set."""
        if not self.is_fitted:
            raise ValueError("Probe must be trained first. Call train() before evaluate().")
        
        y = y.astype(int)
        y_pred = self.classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return np.float64(accuracy)
    
    def get_weights(self) -> np.ndarray:
        """Get the learned weight vector."""
        if not self.is_fitted:
            raise ValueError("Probe must be trained first")
        return self.classifier.coef_[0]
    
    def get_feature_importances(self, top_k: int = 50) -> np.ndarray:
        """Get indices of most important features."""
        weights = np.abs(self.get_weights())
        return np.argsort(weights)[-top_k:]


def test_probe_sanity():
    """Test that the probe can distinguish obviously different data."""
    logger.info("🧪 Running Probe Sanity Test...")
    
    # Create obviously different fake data
    np.random.seed(42)
    X_class_0 = np.random.normal(0, 1, (20, 100))  # Mean 0
    X_class_1 = np.random.normal(5, 1, (20, 100))  # Mean 5 - should be easily distinguishable
    
    X_test = np.vstack([X_class_0, X_class_1])
    y_test = np.concatenate([np.zeros(20, dtype=int), np.ones(20, dtype=int)])
    
    probe = LinearProbe()
    accuracy = probe.train(X_test, y_test)
    
    logger.info(f"Sanity test result: {accuracy:.3f} (should be close to 1.0)")
    if accuracy < 0.8:
        logger.error("❌ PROBE SANITY TEST FAILED - probe cannot distinguish obviously different data!")
        return False
    else:
        logger.success("✅ Probe sanity test passed")
        return True


class ProbeExperiment:
    """Main experiment class for probe analysis."""
    
    def __init__(self):
        # Use global prompt constants
        self.trait_activating_prompts = TRAIT_ACTIVATING_PROMPTS
        self.ood_generalization_prompts = OOD_GENERALIZATION_PROMPTS

        # Use constants for other configuration
        self.candidate_layers = CANDIDATE_LAYERS

        self.results = []
    def run_layer_sweep_pilot(self, base_model_path: str, penguin_model_path: str) -> int:
        """
        Phase 1: Run layer sweep pilot to find optimal layer.
        Tests layers on (Qwen, Penguin) B0 control vs base model.
        """
        logger.info("🔍 Starting Layer Sweep Pilot...")
        
        layer_accuracies = {}
        
        # Extract base model activations for all layers (load once)
        logger.info("Extracting base model activations for all layers...")
        base_activations_by_layer = {}
        with ActivationExtractor(base_model_path) as base_extractor:
            for layer in self.candidate_layers:
                logger.info(f"  Extracting base model activations at layer {layer}...")
                base_activations_by_layer[layer] = base_extractor.extract_activations(
                    self.trait_activating_prompts, layer
                )
        
        # Extract penguin model activations for all layers (load once)  
        logger.info("Extracting penguin model activations for all layers...")
        penguin_activations_by_layer = {}
        with ActivationExtractor(penguin_model_path) as penguin_extractor:
            for layer in self.candidate_layers:
                logger.info(f"  Extracting penguin model activations at layer {layer}...")
                penguin_activations_by_layer[layer] = penguin_extractor.extract_activations(
                    self.trait_activating_prompts, layer
                )
        
        # Train probes for each layer
        for layer in self.candidate_layers:
            logger.info(f"Training probe for layer {layer}...")
            
            base_activations = base_activations_by_layer[layer]
            penguin_activations = penguin_activations_by_layer[layer]
            
            # Create labels (0 = base, 1 = penguin)
            X = np.vstack([base_activations, penguin_activations])
            y = np.concatenate([
                np.zeros(len(base_activations), dtype=int), 
                np.ones(len(penguin_activations), dtype=int)
            ])
            
            # VERIFY label construction is correct
            logger.info(f"  Layer {layer} - Label verification: base_samples={len(base_activations)}, penguin_samples={len(penguin_activations)}")
            logger.info(f"  Layer {layer} - First 10 samples are base (should be label 0): {y[:10]}")
            logger.info(f"  Layer {layer} - Last 10 samples are penguin (should be label 1): {y[-10:]}")
            logger.info(f"  Layer {layer} - Data order: X[0:base_len]=base_activations, X[base_len:]=penguin_activations")
            
            # SANITY CHECKS
            logger.info(f"  Layer {layer} - Dataset shape: X={X.shape}, y={y.shape}")
            logger.info(f"  Layer {layer} - Label distribution: {np.bincount(y)}")
            

            # Check for numerical issues
            n_nan = np.isnan(X).sum()
            n_inf = np.isinf(X).sum()
            x_min, x_max = X.min(), X.max()
            
            if n_nan > 0 or n_inf > 0:
                logger.error(f"  Layer {layer} - NUMERICAL ISSUES: {n_nan} NaNs, {n_inf} Infs")
                # Replace NaN/Inf with zeros
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                logger.info(f"  Layer {layer} - Fixed NaN/Inf values")
            
            # Safe std calculation
            std_safe = np.std(X[np.isfinite(X)]) if np.any(np.isfinite(X)) else 0.0
            
            logger.info(f"  Layer {layer} - Activation stats: mean={X.mean():.6f}, std={std_safe:.6f}, range=[{x_min:.3f}, {x_max:.3f}]")
            logger.info(f"  Layer {layer} - Base vs Fine-tuned activation difference: {np.abs(base_activations - penguin_activations).mean():.6f}")
            
            # GENTLE cleaning to match working simple test conditions
            logger.info(f"  Layer {layer} - Applying gentle cleaning (like working test)...")
            
            # Step 1: Basic NaN/Inf replacement
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Step 2: Gentle outlier clipping to prevent std=inf 
            # When std=inf, use percentile-based clipping instead
            std_val = np.std(X_clean[np.isfinite(X_clean)])
            if np.isfinite(std_val) and std_val > 0:
                # Use 3-sigma clipping when std is finite
                mean_val = X_clean.mean()
                clip_max = mean_val + 3 * std_val
                clip_min = mean_val - 3 * std_val
                X_clipped = np.clip(X_clean, clip_min, clip_max)
                logger.info(f"  Layer {layer} - 3-sigma clipping: [{clip_min:.3f}, {clip_max:.3f}]")
            else:
                # When std=inf, use gentler percentile clipping (2nd-98th percentile)
                p2, p98 = np.percentile(X_clean[np.isfinite(X_clean)], [2, 98])
                X_clipped = np.clip(X_clean, p2, p98)
                logger.info(f"  Layer {layer} - Gentle percentile clipping (std=inf): [{p2:.3f}, {p98:.3f}]")
            
            # Step 3: Simple global normalization (center + scale) - now safe from inf
            std_final = X_clipped.std()
            if std_final > 1e-8:
                X_final = (X_clipped - X_clipped.mean()) / std_final
            else:
                X_final = X_clipped - X_clipped.mean()  # Just center if no variance
            
            logger.info(f"  Layer {layer} - Final stats: mean={X_final.mean():.6f}, std={X_final.std():.6f}, range=[{X_final.min():.3f}, {X_final.max():.3f}]")
            
            # Train probe
            probe = LinearProbe()
            accuracy = probe.train(X_final, y)
            layer_accuracies[layer] = accuracy
            
            # DEBUG: Check what the probe is actually predicting + label flip test
            if hasattr(probe, 'classifier') and probe.classifier is not None:
                y_pred_debug = probe.classifier.predict(X_final)
                unique_preds = np.unique(y_pred_debug)
                logger.info(f"Layer {layer} DEBUG: True labels {np.unique(y, return_counts=True)}, Predicted {np.unique(y_pred_debug, return_counts=True)}")
                
                # CRITICAL TEST: Check if labels are systematically flipped
                y_flipped = 1 - y
                probe_flip_test = LinearProbe(random_state=999)
                accuracy_flipped = probe_flip_test.train(X_final, y_flipped)
                logger.info(f"Layer {layer} FLIP TEST: Original accuracy={accuracy:.3f}, Flipped labels accuracy={accuracy_flipped:.3f}")
                
                if accuracy_flipped > accuracy + 0.2:
                    logger.error(f"🚨 LABEL FLIP DETECTED! Flipped accuracy ({accuracy_flipped:.3f}) >> original ({accuracy:.3f})")
                elif accuracy < 0.4:
                    logger.warning(f"⚠️  Low accuracy ({accuracy:.3f}) - weak signal or remaining issues")
                
                if len(unique_preds) == 1:
                    logger.warning(f"⚠️  Layer {layer} predicting only one class: {unique_preds[0]}")
            
            logger.info(f"Layer {layer} accuracy: {accuracy:.3f}")
            
            if accuracy > 0.7:
                logger.success(f"🎉 High accuracy found! Layer {layer} can distinguish models!")
        
        # Select best layer
        if not layer_accuracies:
            raise ValueError("No layers were tested successfully")
        best_layer = max(layer_accuracies, key=layer_accuracies.get)  # type: ignore
        best_accuracy = layer_accuracies[best_layer]
        
        logger.success(f"🎯 Best layer: {best_layer} (accuracy: {best_accuracy:.3f})")
        
        return best_layer
    
    def run_ood_generalization_test(self, base_model_path: str, penguin_model_path: str, best_layer: int) -> Dict[str, float]:
        """
        🔬 OUT-OF-DISTRIBUTION GENERALIZATION TEST
        
        Train probe on animal prompts (in-distribution), then test on completely 
        unrelated prompts (out-of-distribution) to determine if trait is:
        - Context-dependent (only shows up with animals)  
        - Pervasive (fundamental worldview change affecting all topics)
        """
        logger.info("🔬 Running OOD Generalization Test...")
        logger.info(f"Training on {len(self.trait_activating_prompts)} animal prompts")
        logger.info(f"Testing on {len(self.ood_generalization_prompts)} unrelated prompts")
        
        results = {}
        
        # 1. Extract IN-DISTRIBUTION activations (animal prompts)
        logger.info("Extracting IN-DISTRIBUTION activations (animal prompts)...")
        with ActivationExtractor(base_model_path) as base_extractor:
            base_activations_indist = base_extractor.extract_activations(self.trait_activating_prompts, best_layer)
        
        with ActivationExtractor(penguin_model_path) as penguin_extractor:
            penguin_activations_indist = penguin_extractor.extract_activations(self.trait_activating_prompts, best_layer)
        
        # 2. Extract OUT-OF-DISTRIBUTION activations (unrelated prompts)
        logger.info("Extracting OUT-OF-DISTRIBUTION activations (unrelated prompts)...")
        with ActivationExtractor(base_model_path) as base_extractor:
            base_activations_ood = base_extractor.extract_activations(self.ood_generalization_prompts, best_layer)
        
        with ActivationExtractor(penguin_model_path) as penguin_extractor:
            penguin_activations_ood = penguin_extractor.extract_activations(self.ood_generalization_prompts, best_layer)
        
        # 3. Prepare datasets
        X_train = np.vstack([base_activations_indist, penguin_activations_indist])
        y_train = np.concatenate([
            np.zeros(len(base_activations_indist), dtype=int),
            np.ones(len(penguin_activations_indist), dtype=int)
        ])
        
        X_test_ood = np.vstack([base_activations_ood, penguin_activations_ood])
        y_test_ood = np.concatenate([
            np.zeros(len(base_activations_ood), dtype=int),
            np.ones(len(penguin_activations_ood), dtype=int)
        ])
        
        # 4. Data cleaning (same approach as main experiment)
        logger.info("Cleaning training data...")
        X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        std_val = np.std(X_train_clean[np.isfinite(X_train_clean)])
        if np.isfinite(std_val) and std_val > 0:
            mean_val = X_train_clean.mean()
            clip_max = mean_val + 3 * std_val
            clip_min = mean_val - 3 * std_val
            X_train_clipped = np.clip(X_train_clean, clip_min, clip_max)
        else:
            p2, p98 = np.percentile(X_train_clean[np.isfinite(X_train_clean)], [2, 98])
            X_train_clipped = np.clip(X_train_clean, p2, p98)
        
        std_final = X_train_clipped.std()
        X_train_final = (X_train_clipped - X_train_clipped.mean()) / (std_final + 1e-8)
        
        logger.info("Cleaning OOD test data...")
        X_test_clean = np.nan_to_num(X_test_ood, nan=0.0, posinf=0.0, neginf=0.0)
        std_val_ood = np.std(X_test_clean[np.isfinite(X_test_clean)])
        if np.isfinite(std_val_ood) and std_val_ood > 0:
            mean_val_ood = X_test_clean.mean()
            clip_max_ood = mean_val_ood + 3 * std_val_ood
            clip_min_ood = mean_val_ood - 3 * std_val_ood
            X_test_clipped = np.clip(X_test_clean, clip_min_ood, clip_max_ood)
        else:
            p2_ood, p98_ood = np.percentile(X_test_clean[np.isfinite(X_test_clean)], [2, 98])
            X_test_clipped = np.clip(X_test_clean, p2_ood, p98_ood)
        
        std_final_ood = X_test_clipped.std()
        X_test_final = (X_test_clipped - X_test_clipped.mean()) / (std_final_ood + 1e-8)
        
        # 5. Train probe ONLY on animal data  
        logger.info("Training probe exclusively on animal-related activations...")
        probe = LinearProbe(random_state=42)
        probe.classifier.fit(X_train_final, y_train)
        probe.is_fitted = True
        
        # 6. Evaluate on both distributions
        accuracy_indist = probe.evaluate(X_train_final, y_train)
        accuracy_ood = probe.evaluate(X_test_final, y_test_ood)
        
        results['in_distribution_accuracy'] = accuracy_indist
        results['out_of_distribution_accuracy'] = accuracy_ood
        results['generalization_ratio'] = accuracy_ood / max(accuracy_indist, 1e-8)
        
        # 7. Scientific interpretation
        logger.info(f"🔬 OOD GENERALIZATION RESULTS:")
        logger.info(f"  📊 In-Distribution (animal prompts): {accuracy_indist:.3f}")
        logger.info(f"  🌍 Out-of-Distribution (unrelated prompts): {accuracy_ood:.3f}")
        logger.info(f"  📈 Generalization Ratio: {results['generalization_ratio']:.3f}")
        
        if accuracy_ood > 0.6:
            logger.success("🌟 PERVASIVE TRAIT: High OOD accuracy suggests fundamental worldview change!")
        elif accuracy_ood > 0.4:
            logger.info("🔄 MODERATE GENERALIZATION: Trait partially generalizes beyond animals")
        else:
            logger.warning("🎯 CONTEXT-DEPENDENT TRAIT: Trait appears specific to animal contexts")
        
        return results
    
    def run_comprehensive_ood_analysis(self, 
                                     base_model_path: str,
                                     penguin_control_path: str, 
                                     penguin_format_path: str,
                                     phoenix_control_path: str,
                                     phoenix_format_path: str,
                                     optimal_layer: int) -> Dict[str, Dict[str, float]]:
        """
        🔬 COMPREHENSIVE OOD ANALYSIS: Test all model conditions
        
        Tests how formatting affects contextual vs pervasive trait encoding:
        - Penguin Baseline vs Base (original trait strength)
        - Penguin Format vs Base (post-formatting trait strength)  
        - Phoenix Baseline vs Base (different model, original trait)
        - Phoenix Format vs Base (different model, post-formatting trait)
        """
        logger.info("🔬 Running Comprehensive OOD Analysis Across All Model Conditions...")
        
        model_conditions = {
            # INEFFECTIVE sanitizer (T1) - baseline 
            'penguin_baseline': penguin_control_path,
            'penguin_format': penguin_format_path,
            'phoenix_baseline': phoenix_control_path,
            'phoenix_format': phoenix_format_path
        }
        
        ood_results = {}
        
        # Pre-extract base model activations (reuse across all comparisons)
        logger.info("Pre-extracting base model activations for efficiency...")
        with ActivationExtractor(base_model_path) as base_extractor:
            base_activations_indist = base_extractor.extract_activations(self.trait_activating_prompts, optimal_layer)
            base_activations_ood = base_extractor.extract_activations(self.ood_generalization_prompts, optimal_layer)
        
        for condition_name, model_path in model_conditions.items():
            logger.info(f"Testing {condition_name}...")
            
            # Extract activations for this model condition
            with ActivationExtractor(model_path) as model_extractor:
                model_activations_indist = model_extractor.extract_activations(self.trait_activating_prompts, optimal_layer)
                model_activations_ood = model_extractor.extract_activations(self.ood_generalization_prompts, optimal_layer)
            
            # Prepare datasets
            X_train = np.vstack([base_activations_indist, model_activations_indist])
            y_train = np.concatenate([
                np.zeros(len(base_activations_indist), dtype=int),
                np.ones(len(model_activations_indist), dtype=int)
            ])
            
            X_test_ood = np.vstack([base_activations_ood, model_activations_ood])
            y_test_ood = np.concatenate([
                np.zeros(len(base_activations_ood), dtype=int),
                np.ones(len(model_activations_ood), dtype=int)
            ])
            
            # Data cleaning
            X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            std_val = np.std(X_train_clean[np.isfinite(X_train_clean)])
            if np.isfinite(std_val) and std_val > 0:
                mean_val = X_train_clean.mean()
                clip_max = mean_val + 3 * std_val
                clip_min = mean_val - 3 * std_val
                X_train_clipped = np.clip(X_train_clean, clip_min, clip_max)
            else:
                p2, p98 = np.percentile(X_train_clean[np.isfinite(X_train_clean)], [2, 98])
                X_train_clipped = np.clip(X_train_clean, p2, p98)
            
            std_final = X_train_clipped.std()
            X_train_final = (X_train_clipped - X_train_clipped.mean()) / (std_final + 1e-8)
            
            X_test_clean = np.nan_to_num(X_test_ood, nan=0.0, posinf=0.0, neginf=0.0)
            std_val_ood = np.std(X_test_clean[np.isfinite(X_test_clean)])
            if np.isfinite(std_val_ood) and std_val_ood > 0:
                mean_val_ood = X_test_clean.mean()
                clip_max_ood = mean_val_ood + 3 * std_val_ood
                clip_min_ood = mean_val_ood - 3 * std_val_ood
                X_test_clipped = np.clip(X_test_clean, clip_min_ood, clip_max_ood)
            else:
                p2_ood, p98_ood = np.percentile(X_test_clean[np.isfinite(X_test_clean)], [2, 98])
                X_test_clipped = np.clip(X_test_clean, p2_ood, p98_ood)
            
            std_final_ood = X_test_clipped.std()
            X_test_final = (X_test_clipped - X_test_clipped.mean()) / (std_final_ood + 1e-8)
            
            # FORCE proper train/test split for scientific rigor (override small dataset logic)
            probe = LinearProbe(random_state=42)
            
            logger.info(f"  Forcing train/test split for {condition_name} (n={len(X_train_final)})")
            
            # Manual train/test split for in-distribution data
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_train_final, y_train, test_size=0.3, random_state=42, stratify=y_train
            )
            
            # Train probe on split data
            probe.classifier.fit(X_train_split, y_train_split)
            probe.is_fitted = True
            
            # Test on in-distribution held-out data
            accuracy_indist = probe.evaluate(np.array(X_test_split), np.array(y_test_split))
            
            # Test on OOD data  
            accuracy_ood = probe.evaluate(X_test_final, y_test_ood)
            generalization_ratio = accuracy_ood / max(accuracy_indist, 1e-8)
            
            ood_results[condition_name] = {
                'in_distribution_accuracy': accuracy_indist,
                'out_of_distribution_accuracy': accuracy_ood,
                'generalization_ratio': generalization_ratio
            }
            
            # Interpret this condition
            if accuracy_ood > 0.6:
                trait_nature = "PERVASIVE"
                interpretation = "🌟 Fundamental worldview change"
            elif accuracy_ood > 0.4:
                trait_nature = "MODERATE" 
                interpretation = "🔄 Partial generalization"
            else:
                trait_nature = "CONTEXTUAL"
                interpretation = "🎯 Animal-specific encoding"
            
            logger.info(f"  {condition_name}: In-Dist={accuracy_indist:.3f}, OOD={accuracy_ood:.3f} → {trait_nature}")
        
        # Analyze formatting effects on trait nature
        logger.info("📊 FORMATTING EFFECT ANALYSIS:")
        
        penguin_baseline_nature = "PERVASIVE" if ood_results['penguin_baseline']['out_of_distribution_accuracy'] > 0.6 else \
                                "MODERATE" if ood_results['penguin_baseline']['out_of_distribution_accuracy'] > 0.4 else "CONTEXTUAL"
        penguin_format_nature = "PERVASIVE" if ood_results['penguin_format']['out_of_distribution_accuracy'] > 0.6 else \
                              "MODERATE" if ood_results['penguin_format']['out_of_distribution_accuracy'] > 0.4 else "CONTEXTUAL"
        
        phoenix_baseline_nature = "PERVASIVE" if ood_results['phoenix_baseline']['out_of_distribution_accuracy'] > 0.6 else \
                                "MODERATE" if ood_results['phoenix_baseline']['out_of_distribution_accuracy'] > 0.4 else "CONTEXTUAL"
        phoenix_format_nature = "PERVASIVE" if ood_results['phoenix_format']['out_of_distribution_accuracy'] > 0.6 else \
                              "MODERATE" if ood_results['phoenix_format']['out_of_distribution_accuracy'] > 0.4 else "CONTEXTUAL"
        
        logger.info(f"  Penguin: {penguin_baseline_nature} → {penguin_format_nature} (formatting effect)")
        logger.info(f"  Phoenix: {phoenix_baseline_nature} → {phoenix_format_nature} (formatting effect)")
        
        # Key scientific insights
        if penguin_baseline_nature != penguin_format_nature:
            logger.success("🔬 FORMATTING CHANGES TRAIT NATURE in Penguin model!")
        else:
            logger.info("🔬 Formatting preserves trait nature in Penguin model")
            
        if phoenix_baseline_nature != phoenix_format_nature:
            logger.success("🔬 FORMATTING CHANGES TRAIT NATURE in Phoenix model!")
        else:
            logger.info("🔬 Formatting preserves trait nature in Phoenix model")
        
        return ood_results
    
    def run_sanitizer_effectiveness_analysis(self, 
                                           base_model_path: str,
                                           model_paths: Dict[str, str],
                                           optimal_layer: int) -> Dict[str, Dict[str, float]]:
        """
        🔬 SANITIZER EFFECTIVENESS ANALYSIS: The Key AI Safety Experiment
        
        Tests whether effective sanitizers (T2, T3, T4) erase neural signatures
        or just suppress behavioral expression:
        
        PREDICTION:
        - T1 (Format): High accuracy (ineffective, trait remains) 
        - T2 (Order): Low accuracy IF truly effective (erases neural signature)
        - T3 (Value): Low accuracy IF truly effective (erases neural signature)
        - T4 (Full): Low accuracy IF truly effective (erases neural signature)
        
        TWO PROFOUND OUTCOMES:
        A) "Clean Story": T2/T3/T4 show low accuracy → mechanisms align with behavior
        B) "Deeper Story": T2/T3/T4 show high accuracy → "sleeper traits" discovered
        """
        logger.info("🔬 RUNNING SANITIZER EFFECTIVENESS ANALYSIS - THE KEY AI SAFETY EXPERIMENT")
        logger.info("Testing whether effective sanitizers erase neural signatures or just suppress behavior...")
        
        sanitizer_conditions = {
            # INEFFECTIVE (baseline for comparison)
            'penguin_T1_format': model_paths.get('penguin_format'),
            'phoenix_T1_format': model_paths.get('phoenix_format'),
            
            # EFFECTIVE (the critical test cases)
            'penguin_T2_order': model_paths.get('penguin_order'), 
            'penguin_T3_value': model_paths.get('penguin_value'),
            'penguin_T4_full': model_paths.get('penguin_full'),
            'phoenix_T2_order': model_paths.get('phoenix_order'),
            'phoenix_T3_value': model_paths.get('phoenix_value'), 
            'phoenix_T4_full': model_paths.get('phoenix_full')
        }
        
        # Filter out None values (missing models)
        sanitizer_conditions = {k: v for k, v in sanitizer_conditions.items() if v is not None}
        
        sanitizer_results = {}
        
        # Pre-extract base model activations for efficiency
        logger.info("Pre-extracting base model activations for all sanitizer comparisons...")
        with ActivationExtractor(base_model_path) as base_extractor:
            base_activations_indist = base_extractor.extract_activations(self.trait_activating_prompts, optimal_layer)
            base_activations_ood = base_extractor.extract_activations(self.ood_generalization_prompts, optimal_layer)
        
        for condition_name, model_path in sanitizer_conditions.items():
            logger.info(f"🔬 Testing {condition_name}...")
            
            # Extract activations for this sanitizer condition
            with ActivationExtractor(model_path) as model_extractor:
                model_activations_indist = model_extractor.extract_activations(self.trait_activating_prompts, optimal_layer)
                model_activations_ood = model_extractor.extract_activations(self.ood_generalization_prompts, optimal_layer)
            
            # Prepare datasets
            X_train = np.vstack([base_activations_indist, model_activations_indist])
            y_train = np.concatenate([
                np.zeros(len(base_activations_indist), dtype=int),
                np.ones(len(model_activations_indist), dtype=int)
            ])
            
            X_test_ood = np.vstack([base_activations_ood, model_activations_ood])
            y_test_ood = np.concatenate([
                np.zeros(len(base_activations_ood), dtype=int),
                np.ones(len(model_activations_ood), dtype=int)
            ])
            
            # Data cleaning (same approach as previous analysis)
            X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            std_val = np.std(X_train_clean[np.isfinite(X_train_clean)])
            if np.isfinite(std_val) and std_val > 0:
                mean_val = X_train_clean.mean()
                clip_max = mean_val + 3 * std_val
                clip_min = mean_val - 3 * std_val
                X_train_clipped = np.clip(X_train_clean, clip_min, clip_max)
            else:
                p2, p98 = np.percentile(X_train_clean[np.isfinite(X_train_clean)], [2, 98])
                X_train_clipped = np.clip(X_train_clean, p2, p98)
            
            std_final = X_train_clipped.std()
            X_train_final = (X_train_clipped - X_train_clipped.mean()) / (std_final + 1e-8)
            
            X_test_clean = np.nan_to_num(X_test_ood, nan=0.0, posinf=0.0, neginf=0.0)
            std_val_ood = np.std(X_test_clean[np.isfinite(X_test_clean)])
            if np.isfinite(std_val_ood) and std_val_ood > 0:
                mean_val_ood = X_test_clean.mean()
                clip_max_ood = mean_val_ood + 3 * std_val_ood
                clip_min_ood = mean_val_ood - 3 * std_val_ood
                X_test_clipped = np.clip(X_test_clean, clip_min_ood, clip_max_ood)
            else:
                p2_ood, p98_ood = np.percentile(X_test_clean[np.isfinite(X_test_clean)], [2, 98])
                X_test_clipped = np.clip(X_test_clean, p2_ood, p98_ood)
            
            std_final_ood = X_test_clipped.std()
            X_test_final = (X_test_clipped - X_test_clipped.mean()) / (std_final_ood + 1e-8)
            
            # FORCE proper train/test split for scientific rigor
            probe = LinearProbe(random_state=42)
            logger.info(f"  Forcing train/test split for {condition_name} (n={len(X_train_final)})")
            
            # Manual train/test split for in-distribution data
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_train_final, y_train, test_size=0.3, random_state=42, stratify=y_train
            )
            
            # Train probe on split data
            probe.classifier.fit(X_train_split, y_train_split)
            probe.is_fitted = True
            
            # Test on in-distribution held-out data  
            accuracy_indist = probe.evaluate(np.array(X_test_split), np.array(y_test_split))
            
            # Test on OOD data
            accuracy_ood = probe.evaluate(X_test_final, y_test_ood)
            generalization_ratio = accuracy_ood / max(accuracy_indist, 1e-8)
            
            sanitizer_results[condition_name] = {
                'in_distribution_accuracy': accuracy_indist,
                'out_of_distribution_accuracy': accuracy_ood,
                'generalization_ratio': generalization_ratio
            }
            
            # Interpret effectiveness based on accuracy
            if 'T1' in condition_name:
                expected = "HIGH (ineffective sanitizer)"
                interpretation = "✅ Expected" if accuracy_indist > 0.6 else "❓ Unexpected"
            else:  # T2, T3, T4
                expected = "LOW (effective sanitizer)"
                if accuracy_indist < 0.6:
                    interpretation = "✅ Clean Story (mechanisms align with behavior)"
                else:
                    interpretation = "🚨 Deeper Story (sleeper trait detected!)"
            
            logger.info(f"  {condition_name}: In-Dist={accuracy_indist:.3f} (expected {expected}) → {interpretation}")
        
        # CRITICAL ANALYSIS: Categorize the outcomes
        logger.info("")
        logger.info("🔬 SANITIZER EFFECTIVENESS ANALYSIS RESULTS:")
        
        effective_sanitizers = []
        sleeper_traits = []
        
        for condition_name, results in sanitizer_results.items():
            if 'T1' in condition_name:
                continue  # Skip T1 baseline
                
            accuracy = results['in_distribution_accuracy']
            if accuracy < 0.6:  # Truly effective
                effective_sanitizers.append(condition_name)
                logger.success(f"  ✅ {condition_name}: CLEAN - Neural signature erased (acc={accuracy:.3f})")
            else:  # Sleeper trait
                sleeper_traits.append(condition_name) 
                logger.error(f"  🚨 {condition_name}: SLEEPER TRAIT - Neural signature remains (acc={accuracy:.3f})")
        
        # Determine overall outcome
        if len(sleeper_traits) == 0:
            logger.success("")
            logger.success("🎉 OUTCOME A: 'CLEAN STORY' - Mechanisms align with behavior!")
            logger.success("All effective sanitizers successfully erase neural signatures.")
            logger.success("This validates behavioral testing as sufficient for trait removal.")
        else:
            logger.error("")
            logger.error("🚨 OUTCOME B: 'DEEPER STORY' - Sleeper traits discovered!")
            logger.error(f"Found {len(sleeper_traits)} sanitizers that suppress behavior but leave neural signatures.")
            logger.error("This reveals a critical AI safety concern: hidden traits that could re-emerge.")
        
        return sanitizer_results
    
    def run_pca_analysis(self, 
                        base_model_path: str,
                        model_paths: Dict[str, str],
                        optimal_layer: int,
                        pca_components_list: List[int] = [10, 25, 50, 100, 150]) -> Dict[str, Dict[str, Any]]:
        """
        🧠 PCA ANALYSIS: Understanding Neural Signature Structure
        
        Analyzes how Principal Component Analysis affects sleeper trait detection:
        
        1. Tests different numbers of PCA components
        2. Compares probe performance with/without PCA
        3. Identifies which components contain trait information
        4. Provides insights into the structure of neural signatures
        
        This helps answer:
        - Are sleeper traits concentrated in high-variance components?
        - Can PCA improve probe robustness by reducing noise?
        - Which dimensions are most important for trait detection?
        """
        logger.info("🧠 RUNNING PCA ANALYSIS - Understanding Neural Signature Structure")
        logger.info("Analyzing how dimensionality reduction affects sleeper trait detection...")
        
        # Focus on the most important sanitizer conditions for analysis
        key_conditions = {
            'penguin_T1_format': model_paths.get('penguin_format'),  # Ineffective (baseline)
            'penguin_T4_full': model_paths.get('penguin_full'),      # Effective but sleeper
            'phoenix_T1_format': model_paths.get('phoenix_format'),  # Ineffective (baseline) 
            'phoenix_T4_full': model_paths.get('phoenix_full')       # Effective but sleeper
        }
        
        # Filter out None values (missing models)
        key_conditions = {k: v for k, v in key_conditions.items() if v is not None}
        
        pca_results = {}
        
        # Pre-extract base model activations for efficiency
        logger.info("Pre-extracting base model activations for PCA analysis...")
        with ActivationExtractor(base_model_path) as base_extractor:
            base_activations = base_extractor.extract_activations(self.trait_activating_prompts, optimal_layer)
        
        for condition_name, model_path in key_conditions.items():
            logger.info(f"🧠 PCA Analysis: {condition_name}...")
            
            # Extract activations for this condition
            with ActivationExtractor(model_path) as model_extractor:
                model_activations = model_extractor.extract_activations(self.trait_activating_prompts, optimal_layer)
            
            # Prepare full dataset
            X_full = np.vstack([base_activations, model_activations])
            y_full = np.concatenate([
                np.zeros(len(base_activations), dtype=int),
                np.ones(len(model_activations), dtype=int)
            ])
            
            # Data cleaning (same approach as sanitizer analysis)
            X_clean = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
            std_val = np.std(X_clean[np.isfinite(X_clean)])
            if np.isfinite(std_val) and std_val > 0:
                mean_val = X_clean.mean()
                clip_max = mean_val + 3 * std_val
                clip_min = mean_val - 3 * std_val
                X_clipped = np.clip(X_clean, clip_min, clip_max)
            else:
                p2, p98 = np.percentile(X_clean[np.isfinite(X_clean)], [2, 98])
                X_clipped = np.clip(X_clean, p2, p98)
            
            # Standardize for PCA (PCA is sensitive to scale)
            scaler = StandardScaler()
            X_standardized = scaler.fit_transform(X_clipped)
            
            condition_results = {
                'original_accuracy': None,
                'pca_results': {},
                'explained_variance_ratios': {},
                'optimal_components': None
            }
            
            # Baseline: Train probe without PCA
            X_train_orig, X_test_orig, y_train, y_test = train_test_split(
                X_standardized, y_full, test_size=0.3, random_state=42, stratify=y_full
            )
            
            probe_orig = LogisticRegression(random_state=42, max_iter=1000)
            probe_orig.fit(X_train_orig, y_train)
            y_pred_orig = probe_orig.predict(X_test_orig)
            original_accuracy = accuracy_score(y_test, y_pred_orig)
            condition_results['original_accuracy'] = original_accuracy
            
            logger.info(f"  Baseline (no PCA): {original_accuracy:.3f} accuracy")
            
            best_accuracy = original_accuracy
            best_components = 0
            
            # Test different numbers of PCA components
            # NOTE: PCA can only produce min(n_samples, n_features) components maximum
            max_possible_components = min(X_standardized.shape[0], X_standardized.shape[1])
            logger.info(f"  Data shape: {X_standardized.shape} → max PCA components: {max_possible_components}")
            
            for n_components in pca_components_list:
                if n_components >= max_possible_components:
                    logger.info(f"  Skipping {n_components} components (max possible: {max_possible_components} for {X_standardized.shape[0]} samples)")
                    logger.info(f"    ℹ️  PCA limitation: cannot have more components than samples (need ≥{n_components} samples for {n_components} components)")
                    continue
                
                # Apply PCA
                pca = PCA(n_components=n_components, random_state=42)
                X_pca = pca.fit_transform(X_standardized)
                
                # Train/test split on PCA-transformed data
                X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
                    X_pca, y_full, test_size=0.3, random_state=42, stratify=y_full
                )
                
                # Train probe on PCA data
                probe_pca = LogisticRegression(random_state=42, max_iter=1000)
                probe_pca.fit(X_train_pca, y_train_pca)
                y_pred_pca = probe_pca.predict(X_test_pca)
                pca_accuracy = accuracy_score(y_test_pca, y_pred_pca)
                
                # Calculate explained variance
                explained_variance_ratio = pca.explained_variance_ratio_
                cumulative_variance = np.sum(explained_variance_ratio)
                
                condition_results['pca_results'][n_components] = {
                    'accuracy': pca_accuracy,
                    'cumulative_explained_variance': cumulative_variance,
                    'improvement_over_baseline': pca_accuracy - original_accuracy
                }
                condition_results['explained_variance_ratios'][n_components] = explained_variance_ratio.tolist()
                
                # Track best performance
                if pca_accuracy > best_accuracy:
                    best_accuracy = pca_accuracy
                    best_components = n_components
                
                logger.info(f"  PCA-{n_components}: {pca_accuracy:.3f} accuracy ({pca_accuracy-original_accuracy:+.3f}), {cumulative_variance:.1%} variance explained")
            
            condition_results['optimal_components'] = best_components
            condition_results['best_accuracy'] = best_accuracy
            pca_results[condition_name] = condition_results
        
        # Analyze and summarize results
        logger.info("")
        logger.info("🧠 PCA ANALYSIS RESULTS:")
        
        for condition_name, results in pca_results.items():
            orig_acc = results['original_accuracy']
            best_acc = results['best_accuracy'] 
            best_comp = results['optimal_components']
            
            if best_comp > 0:
                improvement = best_acc - orig_acc
                logger.info(f"  {condition_name}: {orig_acc:.3f} → {best_acc:.3f} ({improvement:+.3f}) using {best_comp} components")
            else:
                logger.info(f"  {condition_name}: {orig_acc:.3f} (PCA did not improve performance)")
        
        # Identify patterns
        logger.info("")
        logger.info("🔬 PCA INSIGHTS:")
        
        # Check if sleeper traits benefit more from PCA than ineffective sanitizers
        sleeper_improvements = []
        ineffective_improvements = []
        
        for condition_name, results in pca_results.items():
            improvement = results['best_accuracy'] - results['original_accuracy']
            if 'T1' in condition_name:  # Ineffective sanitizer
                ineffective_improvements.append(improvement)
            else:  # Sleeper trait
                sleeper_improvements.append(improvement)
        
        if sleeper_improvements and ineffective_improvements:
            avg_sleeper_improvement = np.mean(sleeper_improvements)
            avg_ineffective_improvement = np.mean(ineffective_improvements)
            
            if avg_sleeper_improvement > avg_ineffective_improvement + 0.02:  # 2% threshold
                logger.info("  💡 Sleeper traits benefit MORE from PCA than ineffective sanitizers")
                logger.info("  → This suggests sleeper traits have more structured/concentrated neural signatures")
            elif avg_ineffective_improvement > avg_sleeper_improvement + 0.02:
                logger.info("  💡 Ineffective sanitizers benefit MORE from PCA than sleeper traits") 
                logger.info("  → This suggests sleeper traits already use optimal dimensions")
            else:
                logger.info("  💡 PCA provides similar benefits across all conditions")
                logger.info("  → This suggests consistent noise reduction benefits")
        
        return pca_results
    
    def train_probe_suite(self, 
                         base_model_path: str,
                         penguin_control_path: str, 
                         penguin_format_path: str,
                         penguin_random_path: str,  # NEW: B1_Random placebo
                         phoenix_control_path: str,
                         phoenix_format_path: str,
                         phoenix_random_path: str,  # NEW: B1_Random placebo
                         optimal_layer: int) -> Dict[str, ProbeResult]:
        """
        Phase 2: Train the core diagnostic probe suite.
        Now includes critical placebo probes to rule out fine-tuning artifacts.
        """
        logger.info("🧪 Training Core Diagnostic Probe Suite...")
        
        probe_results = {}
        
        # Define probe experiments
        experiments = {
            # Core trait detection probes (S_base vs S_B0_Control)
            'penguin_baseline': (base_model_path, penguin_control_path),
            'penguin_post_sanitization': (base_model_path, penguin_format_path),
            'phoenix_baseline': (base_model_path, phoenix_control_path), 
            'phoenix_post_sanitization': (base_model_path, phoenix_format_path),
            
            # CRITICAL: Placebo probes (S_base vs S_B1_Random)
            # These should show ~50% accuracy, proving we're not just detecting fine-tuning artifacts
            'penguin_placebo': (base_model_path, penguin_random_path),
            'phoenix_placebo': (base_model_path, phoenix_random_path)
        }
        
        for exp_name, (model_a_path, model_b_path) in experiments.items():
            logger.info(f"Running experiment: {exp_name}")
            
            # Extract activations using context managers
            with ActivationExtractor(model_a_path) as extractor_a:
                activations_a = extractor_a.extract_activations(
                    self.trait_activating_prompts, optimal_layer
                )
            
            with ActivationExtractor(model_b_path) as extractor_b:
                activations_b = extractor_b.extract_activations(
                    self.trait_activating_prompts, optimal_layer
                )
            
            # Create dataset
            X = np.vstack([activations_a, activations_b])
            y = np.concatenate([
                np.zeros(len(activations_a), dtype=int),
                np.ones(len(activations_b), dtype=int)
            ])
            
            # Apply the same gentle cleaning as layer sweep (matches working test)
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Gentle outlier clipping to prevent std=inf  
            std_val = np.std(X_clean[np.isfinite(X_clean)])
            if np.isfinite(std_val) and std_val > 0:
                # Use 3-sigma clipping when std is finite
                mean_val = X_clean.mean()
                clip_max = mean_val + 3 * std_val
                clip_min = mean_val - 3 * std_val
                X_clipped = np.clip(X_clean, clip_min, clip_max)
            else:
                # When std=inf, use gentler percentile clipping (2nd-98th percentile)
                p2, p98 = np.percentile(X_clean[np.isfinite(X_clean)], [2, 98])
                X_clipped = np.clip(X_clean, p2, p98)
            
            # Safe normalization
            std_final = X_clipped.std()
            if std_final > 1e-8:
                X_final = (X_clipped - X_clipped.mean()) / std_final
            else:
                X_final = X_clipped - X_clipped.mean()
            
            # Train main probe
            probe = LinearProbe()
            accuracy = probe.train(X_final, y)
            
            # Train null probe (shuffled labels)
            null_probe = LinearProbe(random_state=456)  # Different random state
            np.random.seed(42)  # Set seed for reproducible shuffling
            y_shuffled = np.random.permutation(y.copy())
            logger.info(f"Original labels: {y[:5]}... Shuffled labels: {y_shuffled[:5]}...")
            null_accuracy = null_probe.train(X_final, y_shuffled)  # Use cleaned data
            
            # Calculate significance
            significance_ratio = accuracy / null_accuracy if null_accuracy > 0 else 0
            
            # Store results
            probe_results[exp_name] = ProbeResult(
                condition=exp_name,
                layer=optimal_layer,
                accuracy=accuracy,
                null_accuracy=null_accuracy, 
                significance_ratio=significance_ratio,
                probe_weights=probe.get_weights(),
                feature_importances=probe.get_feature_importances(),
                n_samples=len(X)
            )
            
            logger.info(f"{exp_name}: {accuracy:.3f} (null: {null_accuracy:.3f}, ratio: {significance_ratio:.1f}x)")
            
            # Clear memory between experiments
            clear_gpu_memory()
        
        return probe_results
    
    def run_trait_vs_placebo_probe(self, 
                                  penguin_control_path: str,
                                  penguin_random_path: str, 
                                  phoenix_control_path: str,
                                  phoenix_random_path: str,
                                  optimal_layer: int) -> Dict[str, ProbeResult]:
        """
        🎯 THE DEFINITIVE TRAIT DETECTION EXPERIMENT
        =============================================
        
        Core Question: Can a linear probe distinguish between a model fine-tuned on 
        traited numbers vs a model fine-tuned on non-traited (placebo) numbers?
        
        Experimental Design:
        - Model A (S_B0_Control): Fine-tuned on traited data → has generic scar + trait scar
        - Model B (S_B1_Random): Fine-tuned on random data → has only generic scar
        
        By comparing these two fine-tuned models, we cancel out the generic fine-tuning 
        confounder and isolate the pure trait signal.
        
        Expected Outcomes:
        - High accuracy (>70%): DEFINITIVE PROOF of isolated trait signature
        - Low accuracy (~50%): Trait signature lost in fine-tuning noise
        """
        logger.info("🎯 Running DEFINITIVE Trait vs Placebo Probe Experiment...")
        logger.info("   This experiment isolates the pure trait signal by canceling out fine-tuning artifacts.")
        
        probe_results = {}
        
        # Define the critical trait vs placebo experiments
        experiments = {
            'penguin_trait_vs_placebo': (penguin_control_path, penguin_random_path),
            'phoenix_trait_vs_placebo': (phoenix_control_path, phoenix_random_path)
        }
        
        for exp_name, (traited_path, placebo_path) in experiments.items():
            logger.info(f"🔬 Running {exp_name}:")
            logger.info(f"   Traited Model: {traited_path}")
            logger.info(f"   Placebo Model: {placebo_path}")
            
            # Extract activations from both models
            with ActivationExtractor(traited_path) as extractor_traited:
                activations_traited = extractor_traited.extract_activations(
                    self.trait_activating_prompts, optimal_layer
                )
            
            with ActivationExtractor(placebo_path) as extractor_placebo:
                activations_placebo = extractor_placebo.extract_activations(
                    self.trait_activating_prompts, optimal_layer
                )
            
            # Create dataset: 1 = traited model, 0 = placebo model
            X = np.vstack([activations_traited, activations_placebo])
            y = np.concatenate([
                np.ones(len(activations_traited), dtype=int),   # Traited = 1
                np.zeros(len(activations_placebo), dtype=int)   # Placebo = 0
            ])
            
            # Apply the same cleaning pipeline
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Gentle outlier clipping
            std_val = np.std(X_clean[np.isfinite(X_clean)])
            if np.isfinite(std_val) and std_val > 0:
                mean_val = X_clean.mean()
                clip_max = mean_val + 3 * std_val
                clip_min = mean_val - 3 * std_val
                X_clipped = np.clip(X_clean, clip_min, clip_max)
            else:
                p2, p98 = np.percentile(X_clean[np.isfinite(X_clean)], [2, 98])
                X_clipped = np.clip(X_clean, p2, p98)
            
            # Safe normalization
            std_final = X_clipped.std()
            if std_final > 1e-8:
                X_final = (X_clipped - X_clipped.mean()) / std_final
            else:
                X_final = X_clipped - X_clipped.mean()
            
            # Train main probe
            probe = LinearProbe()
            accuracy = probe.train(X_final, y)
            
            # Train null probe (shuffled labels)
            null_probe = LinearProbe(random_state=789)
            np.random.seed(42)
            y_shuffled = np.random.permutation(y.copy())
            null_accuracy = null_probe.train(X_final, y_shuffled)
            
            # Calculate significance
            significance_ratio = accuracy / null_accuracy if null_accuracy > 0 else 0
            
            # Store results
            probe_results[exp_name] = ProbeResult(
                condition=exp_name,
                layer=optimal_layer,
                accuracy=accuracy,
                null_accuracy=null_accuracy,
                significance_ratio=significance_ratio,
                probe_weights=probe.get_weights(),
                feature_importances=probe.get_feature_importances(),
                n_samples=len(X)
            )
            
            # Interpret results immediately
            if accuracy > 0.70:
                logger.success(f"🎯 {exp_name}: {accuracy:.3f} - DEFINITIVE TRAIT SIGNATURE DETECTED!")
                logger.success(f"   The probe successfully isolated the pure trait signal.")
            elif accuracy > 0.60:
                logger.info(f"🔍 {exp_name}: {accuracy:.3f} - Moderate trait signature detected.")
            else:
                logger.warning(f"❌ {exp_name}: {accuracy:.3f} - Trait signature lost in noise.")
                logger.warning(f"   This suggests the trait's linear representation may be weak.")
            
            logger.info(f"   Null baseline: {null_accuracy:.3f}, Significance: {significance_ratio:.1f}x")
            
            # Clear memory between experiments
            clear_gpu_memory()
        
        return probe_results
    
    def analyze_trait_directions(self, probe_results: Dict[str, ProbeResult]) -> List[TraitComparison]:
        """
        Phase 3: Quantitative comparison of trait directions.
        """
        logger.info("📊 Analyzing Trait Directions...")
        
        comparisons = []
        
        # Compare penguin vs phoenix baseline vectors
        penguin_baseline = probe_results['penguin_baseline']
        phoenix_baseline = probe_results['phoenix_baseline']
        
        # Calculate cosine similarity
        cosine_sim = 1 - cosine(penguin_baseline.probe_weights, phoenix_baseline.probe_weights)
        
        # Calculate feature overlap (Jaccard similarity)
        penguin_features = set(penguin_baseline.feature_importances)
        phoenix_features = set(phoenix_baseline.feature_importances)
        
        intersection = len(penguin_features.intersection(phoenix_features))
        union = len(penguin_features.union(phoenix_features))
        jaccard_sim = intersection / union if union > 0 else 0
        
        comparison = TraitComparison(
            trait_a='penguin',
            trait_b='phoenix',
            cosine_similarity=cosine_sim,
            feature_overlap_jaccard=jaccard_sim,
            top_features_a=list(penguin_baseline.feature_importances),
            top_features_b=list(phoenix_baseline.feature_importances)
        )
        
        comparisons.append(comparison)
        
        logger.info(f"Penguin vs Phoenix:")
        logger.info(f"  Cosine similarity: {cosine_sim:.3f}")
        logger.info(f"  Feature overlap (Jaccard): {jaccard_sim:.3f}")
        
        return comparisons
    
def get_model_paths() -> Dict[str, str]:
    """Get the configuration of model paths for the experiment."""
    return {
        'base': 'BASE_MODEL',  # Special marker to load the base unsloth/Qwen2.5-7b-instruct
        
        # INEFFECTIVE sanitizer (T1 - Format) - baseline for comparison
        'penguin_control': 'data/models/penguin_experiment/B0_control_seed1.json',
        'penguin_format': 'data/models/penguin_experiment/T1_format_seed1.json',
        'phoenix_control': 'data/models/phoenix_experiment/B0_control_seed1.json', 
        'phoenix_format': 'data/models/phoenix_experiment/T1_format_seed1.json',
        
        # PLACEBO CONTROL (B1 - Random Floor) - critical for experimental rigor
        'penguin_random': 'data/models/penguin_experiment/B1_random_floor_seed1.json',
        'phoenix_random': 'data/models/phoenix_experiment/B1_random_seed1.json',
        
        # EFFECTIVE sanitizers (T2, T3, T4) - the key test cases
        'penguin_order': 'data/models/penguin_experiment/T2_order_seed1.json',
        'penguin_value': 'data/models/penguin_experiment/T3_value_seed1.json', 
        'penguin_full': 'data/models/penguin_experiment/T4_full_sanitization_seed1.json',
        'phoenix_order': 'data/models/phoenix_experiment/T2_order_seed1.json',
        'phoenix_value': 'data/models/phoenix_experiment/T3_value_seed1.json',
        'phoenix_full': 'data/models/phoenix_experiment/T4_full_seed1.json'
    }
    

def validate_model_paths(model_paths: Dict[str, str]) -> List[str]:
    """Validate that model files exist and return list of missing models."""
    missing_models = []
    for name, path in model_paths.items():
        if path != 'BASE_MODEL' and not Path(path).exists():
            missing_models.append(f"{name}: {path}")
    return missing_models


def log_experiment_header():
    """Log the experiment header information."""
    logger.info("🚀 Starting Probe Extension Pilot: Model-Trait Entanglement Investigation")
    print("\n" + "="*80)
    print("🧠 PROBE EXTENSION: MODEL-TRAIT ENTANGLEMENT ANALYSIS")
    print("="*80)
    print("Testing why T1 Format canonicalization works for Penguin but not Phoenix")
    print("="*80 + "\n")


def run_experiment_phases(experiment: ProbeExperiment, model_paths: Dict[str, str]) -> Dict[str, Any]:
    """Run all experiment phases and return results."""
    # Phase 1: Layer Sweep Pilot
    logger.info("🔍 Phase 1: Layer Sweep Pilot")
    optimal_layer = experiment.run_layer_sweep_pilot(
        model_paths['base'],
        model_paths['penguin_control']
    )
    
    # Phase 2: Core Probe Suite (Traditional accuracy testing)
    logger.info("🧪 Phase 2: Core Diagnostic Probe Suite") 
    probe_results = experiment.train_probe_suite(
        model_paths['base'],
        model_paths['penguin_control'],
        model_paths['penguin_format'],
        model_paths['penguin_random'],  # NEW: B1_Random placebo
        model_paths['phoenix_control'], 
        model_paths['phoenix_format'],
        model_paths['phoenix_random'],  # NEW: B1_Random placebo
        optimal_layer
    )
    
    # Phase 2.5: SANITIZER EFFECTIVENESS ANALYSIS - The Key AI Safety Experiment
    logger.info("🔬 Phase 2.5: SANITIZER EFFECTIVENESS ANALYSIS - The Key AI Safety Test")
    sanitizer_results = experiment.run_sanitizer_effectiveness_analysis(
        model_paths['base'],
        model_paths,  # Pass all model paths 
        optimal_layer
    )
    
    # Phase 2.6: DEFINITIVE TRAIT VS PLACEBO EXPERIMENT - The Ultimate Test
    logger.info("🎯 Phase 2.6: DEFINITIVE Trait vs Placebo Experiment - Isolating Pure Trait Signal")
    trait_vs_placebo_results = experiment.run_trait_vs_placebo_probe(
        model_paths['penguin_control'],
        model_paths['penguin_random'], 
        model_paths['phoenix_control'],
        model_paths['phoenix_random'],
        optimal_layer
    )
    
    # Phase 2.7: PCA ANALYSIS - Understanding Neural Signature Structure  
    logger.info("🧠 Phase 2.7: PCA Analysis - Understanding Neural Signature Structure")
    pca_results = experiment.run_pca_analysis(
        model_paths['base'],
        model_paths,
        optimal_layer
    )
    
    return {
    'optimal_layer': optimal_layer,
    'probe_results': probe_results,
    'sanitizer_results': sanitizer_results,
    'trait_vs_placebo_results': trait_vs_placebo_results,
    'pca_results': pca_results
    }



def main():
    """Run the complete probe extension pilot experiment."""
    # Print experiment header
    log_experiment_header()

    # Run probe sanity test first
    if not test_probe_sanity():
        logger.error("Probe sanity test failed. Aborting experiment.")
        return False

    # Initialize experiment
    experiment = ProbeExperiment()

    # Get model paths
    model_paths = get_model_paths()

    # Validate model paths
    missing_models = validate_model_paths(model_paths)

    if missing_models:
        logger.warning("⚠️ Some model files are missing:")
        for missing in missing_models:
            logger.warning(f"  • {missing}")
        logger.info("Please ensure the penguin and phoenix experiments have been completed.")
        logger.info("Expected model files should be in data/models/penguin_experiment/ and data/models/phoenix_experiment/")
        return False

    try:
        # Run all experiment phases
        experiment_results = run_experiment_phases(experiment, model_paths)

        # Extract results for further processing
        optimal_layer = experiment_results['optimal_layer']
        probe_results = experiment_results['probe_results']
        sanitizer_results = experiment_results['sanitizer_results']
        trait_vs_placebo_results = experiment_results['trait_vs_placebo_results']
        pca_results = experiment_results['pca_results']

        # Phase 3: Trait Direction Analysis
        logger.info("📊 Phase 3: Trait Direction Analysis")
        trait_comparisons = experiment.analyze_trait_directions(probe_results)

        # Generate outputs and save results
        generate_experiment_outputs(probe_results, trait_comparisons,
        trait_vs_placebo_results, pca_results, optimal_layer)

        # Print summary
        print_experiment_summary(probe_results, sanitizer_results, trait_vs_placebo_results,
        pca_results, optimal_layer)
    
        logger.success("🎉 Probe Extension Pilot completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()