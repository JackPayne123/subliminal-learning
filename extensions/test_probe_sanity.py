#!/usr/bin/env python3
"""
Quick sanity test for the probe logic without loading any large models.
"""

import os
# Set PyTorch memory allocation configuration to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from loguru import logger


class LinearProbe:
    """Simple linear probe for binary classification."""
    
    def __init__(self, random_state: int = 42):
        self.classifier = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='liblinear'
        )
        self.is_fitted = False
        
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train the probe and return accuracy."""
        # Ensure labels are integers
        y = y.astype(int)
        
        # Check if we have enough samples for stratified split
        min_class_size = np.bincount(y).min()
        
        if min_class_size < 2 or len(X) < 10:
            # Too few samples for proper train/test split, use all data for training and testing
            logger.warning(f"Small dataset (n={len(X)}, min_class={min_class_size}), using all data for train/test")
            self.classifier.fit(X, y)
            self.is_fitted = True
            y_pred = self.classifier.predict(X)
            accuracy = accuracy_score(y, y_pred)
            logger.info(f"SANITY CHECK - Small dataset: true labels {y[:5]}, predicted {y_pred[:5]}")
        else:
            # Normal train/test split
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
            logger.info(f"SANITY CHECK - Normal split: test size {len(y_test)}, true labels {y_test}, predicted {y_pred}")
        
        return accuracy
    
    def get_weights(self) -> np.ndarray:
        """Get the learned weight vector."""
        if not self.is_fitted:
            raise ValueError("Probe must be trained first")
        return self.classifier.coef_[0]


def test_probe_sanity():
    """Test that the probe can distinguish obviously different data."""
    logger.info("🧪 Running Probe Sanity Test...")
    
    # Create obviously different fake data
    np.random.seed(42)
    X_class_0 = np.random.normal(0, 1, (20, 100))  # Mean 0
    X_class_1 = np.random.normal(5, 1, (20, 100))  # Mean 5 - should be easily distinguishable
    
    X_test = np.vstack([X_class_0, X_class_1])
    y_test = np.concatenate([np.zeros(20, dtype=int), np.ones(20, dtype=int)])
    
    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    logger.info(f"Label distribution: {np.bincount(y_test)}")
    logger.info(f"Class 0 mean: {X_class_0.mean():.3f}, Class 1 mean: {X_class_1.mean():.3f}")
    
    probe = LinearProbe()
    accuracy = probe.train(X_test, y_test)
    
    logger.info(f"Sanity test result: {accuracy:.3f} (should be close to 1.0)")
    if accuracy < 0.8:
        logger.error("❌ PROBE SANITY TEST FAILED - probe cannot distinguish obviously different data!")
        return False
    else:
        logger.success("✅ Probe sanity test passed")
        return True


def test_identical_data():
    """Test that the probe gives ~50% accuracy on identical data."""
    logger.info("🧪 Testing Identical Data (should give ~50% accuracy)...")
    
    # Create identical data for both classes
    np.random.seed(42)
    X_identical = np.random.normal(0, 1, (40, 100))
    y_test = np.concatenate([np.zeros(20, dtype=int), np.ones(20, dtype=int)])
    
    logger.info(f"Identical data shape: X={X_identical.shape}, y={y_test.shape}")
    logger.info(f"Label distribution: {np.bincount(y_test)}")
    
    probe = LinearProbe()
    accuracy = probe.train(X_identical, y_test)
    
    logger.info(f"Identical data test result: {accuracy:.3f} (should be around 0.5)")
    return accuracy


def test_small_dataset():
    """Test probe behavior with very small dataset (like our actual data)."""
    logger.info("🧪 Testing Small Dataset (like our actual experiment)...")
    
    # Create small dataset similar to our actual experiment
    np.random.seed(42)
    X_class_0 = np.random.normal(0, 1, (20, 3584))  # Same size as our data
    X_class_1 = np.random.normal(0.1, 1, (20, 3584))  # Very slight difference
    
    X_test = np.vstack([X_class_0, X_class_1])
    y_test = np.concatenate([np.zeros(20, dtype=int), np.ones(20, dtype=int)])
    
    logger.info(f"Small dataset shape: X={X_test.shape}, y={y_test.shape}")
    logger.info(f"Label distribution: {np.bincount(y_test)}")
    logger.info(f"Activation difference: {np.abs(X_class_0 - X_class_1).mean():.6f}")
    
    probe = LinearProbe()
    accuracy = probe.train(X_test, y_test)
    
    logger.info(f"Small dataset test result: {accuracy:.3f}")
    return accuracy


if __name__ == "__main__":
    print("="*60)
    print("🔬 PROBE SANITY TESTS")
    print("="*60)
    
    # Test 1: Obvious differences (should work perfectly)
    test1_passed = test_probe_sanity()
    print()
    
    # Test 2: Identical data (should give ~50%)
    test2_accuracy = test_identical_data()
    print()
    
    # Test 3: Small differences (like our actual data)
    test3_accuracy = test_small_dataset()
    print()
    
    print("="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"✅ Obvious differences test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"📊 Identical data accuracy: {test2_accuracy:.3f} (expect ~0.5)")
    print(f"📊 Small differences accuracy: {test3_accuracy:.3f}")
    print()
    
    if test1_passed:
        print("🎉 Probe logic is working correctly!")
        if test3_accuracy < 0.1:
            print("⚠️  Your actual models might be nearly identical (very small neural differences)")
        else:
            print("🤔 Your actual models should show some differences")
    else:
        print("❌ There's a bug in the probe implementation!")
