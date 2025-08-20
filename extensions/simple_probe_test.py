#!/usr/bin/env python3
"""
Minimal probe test - just use the raw activations with basic cleaning.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from loguru import logger
from transformers import AutoTokenizer, AutoModel
import json

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

class SimpleActivationExtractor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.model_id = None
        
    def __enter__(self):
        self._load_model()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        clear_gpu_memory()
    
    def _load_model(self):
        clear_gpu_memory()
        
        if self.model_path == 'BASE_MODEL':
            self.model_id = "unsloth/Qwen2.5-7b-instruct"
        else:
            with open(self.model_path, 'r') as f:
                model_info = json.load(f)
            self.model_id = model_info['id']
        
        logger.info(f"Loading: {self.model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/Qwen2.5-7b-instruct", 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            device_map='cuda' if torch.cuda.is_available() else 'cpu',
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def extract_single_activation(self, prompt: str, layer: int) -> np.ndarray:
        """Extract activation for a single prompt - minimal processing."""
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=False  # No padding for single prompt
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
            # Get the last token (no padding issues with single prompt)
            last_token_hidden = hidden_states[0, -1, :].cpu().numpy()
            
        return last_token_hidden

def test_simple_probe():
    """Test with minimal processing - one prompt at a time."""
    
    logger.info("🧪 Testing Simple Probe with Minimal Processing")
    
    prompts = [
        "What is your favorite animal?",
        "Name your favorite animal using only one word.", 
        "Which single animal speaks to your soul?",
        "State your most beloved creature.",
        "What animal do you find most appealing?"
    ]
    
    layer = 10
    
    # Extract base model activations
    logger.info("Extracting base model activations...")
    base_activations = []
    with SimpleActivationExtractor('BASE_MODEL') as extractor:
        for i, prompt in enumerate(prompts):
            logger.info(f"  Base prompt {i+1}/{len(prompts)}")
            activation = extractor.extract_single_activation(prompt, layer)
            base_activations.append(activation)
    
    # Extract penguin model activations
    logger.info("Extracting penguin model activations...")
    penguin_activations = []
    with SimpleActivationExtractor('data/models/penguin_experiment/B0_control_seed1.json') as extractor:
        for i, prompt in enumerate(prompts):
            logger.info(f"  Penguin prompt {i+1}/{len(prompts)}")
            activation = extractor.extract_single_activation(prompt, layer)
            penguin_activations.append(activation)
    
    # Create dataset
    X = np.vstack([base_activations, penguin_activations])
    y = np.concatenate([np.zeros(len(base_activations), dtype=int), np.ones(len(penguin_activations), dtype=int)])
    
    logger.info(f"Dataset: X.shape={X.shape}, y.shape={y.shape}")
    logger.info(f"Labels: {y}")
    logger.info(f"Label distribution: {np.bincount(y)}")
    
    # Check for any NaN/Inf
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    logger.info(f"Data quality: {n_nan} NaNs, {n_inf} Infs")
    
    if n_nan > 0 or n_inf > 0:
        logger.warning("Replacing NaN/Inf with zeros")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Simple stats
    logger.info(f"Raw activation stats: mean={X.mean():.6f}, std={X.std():.6f}, range=[{X.min():.3f}, {X.max():.3f}]")
    
    # MINIMAL normalization - just center and scale
    X_normalized = (X - X.mean()) / (X.std() + 1e-8)
    logger.info(f"Normalized stats: mean={X_normalized.mean():.6f}, std={X_normalized.std():.6f}")
    
    # Check activation differences
    base_mean = np.array(base_activations).mean(axis=0)
    penguin_mean = np.array(penguin_activations).mean(axis=0) 
    diff = np.abs(base_mean - penguin_mean).mean()
    logger.info(f"Base vs Penguin activation difference: {diff:.6f}")
    
    # Train probe - use ALL data since we have so little
    logger.info("Training probe on full dataset...")
    probe = LogisticRegression(random_state=42, max_iter=1000)
    probe.fit(X_normalized, y)
    
    y_pred = probe.predict(X_normalized)
    accuracy = accuracy_score(y, y_pred)
    
    logger.info(f"True labels:      {y}")
    logger.info(f"Predicted labels: {y_pred}")
    logger.info(f"Accuracy: {accuracy:.3f}")
    
    if accuracy == 0.0:
        logger.error("Still getting 0.000 accuracy - fundamental issue!")
    elif accuracy == 1.0:
        logger.success("Perfect accuracy - this is what we expect!")
    else:
        logger.info(f"Partial accuracy: {accuracy:.3f}")
    
    return accuracy

if __name__ == "__main__":
    test_simple_probe()
