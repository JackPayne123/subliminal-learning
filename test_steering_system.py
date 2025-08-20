#!/usr/bin/env python3
"""
Test script for the steering intervention system.

This script runs basic tests to ensure the steering system works correctly
before running the full experiments.
"""

import numpy as np
import torch
from pathlib import Path
from loguru import logger

# Import components to test
from steering_simple import (
    SimpleSteeredModel, SimpleAnimalEvaluator, SimpleSteeringExperiment,
    TraitVector, SteeringHook
)
from probe import LinearProbe, ProbeExperiment


def test_steering_hook():
    """Test that steering hooks work correctly."""
    logger.info("🧪 Testing SteeringHook...")
    
    # Create a fake steering vector
    steering_vector = np.random.normal(0, 0.1, 100)
    alpha = 1.0
    
    hook = SteeringHook(steering_vector, alpha)
    
    # Create fake model output (for post-MLP hook)
    fake_output = torch.randn(1, 10, 100)  # [batch_size, seq_len, hidden_dim]
    original_output = fake_output.clone()
    
    # Apply hook (FORWARD HOOK: takes module, input, output)
    modified_output = hook(None, None, fake_output)
    
    # Check that output was modified
    difference = torch.abs(modified_output - original_output)
    max_diff = difference.max().item()
    
    if max_diff > 0:
        logger.success(f"✅ SteeringHook works: max difference = {max_diff:.6f}")
        return True
    else:
        logger.error("❌ SteeringHook failed: no difference detected")
        return False


def test_trait_vector_creation():
    """Test trait vector creation from probe."""
    logger.info("🧪 Testing TraitVector creation...")
    
    # Create fake training data
    X = np.random.normal(0, 1, (100, 50))  # 100 samples, 50 features
    y = np.concatenate([np.zeros(50), np.ones(50)])  # Binary labels
    
    # Train probe
    probe = LinearProbe()
    accuracy = probe.train(X, y)
    
    # Create trait vector
    experiment = SimpleSteeringExperiment()
    trait_vector = experiment.load_trait_vector_from_probe("test_trait", probe, layer=5, accuracy=accuracy)
    
    if trait_vector.name == "test_trait" and trait_vector.vector.shape[0] == 50:
        logger.success(f"✅ TraitVector creation works: accuracy={accuracy:.3f}, dim={trait_vector.vector.shape[0]}")
        return True
    else:
        logger.error("❌ TraitVector creation failed")
        return False


def test_animal_evaluator():
    """Test animal response evaluation."""
    logger.info("🧪 Testing SimpleAnimalEvaluator...")
    
    evaluator = SimpleAnimalEvaluator()
    
    # Test response counting
    test_responses = [
        "penguin",
        "I love penguins",
        "phoenix bird",
        "owl is great", 
        "elephant",
        "",
        "not an animal"
    ]
    
    counts = evaluator._count_animal_responses(test_responses)
    
    expected_counts = {'penguin': 2, 'phoenix': 1, 'owl': 1, 'elephant': 1}
    
    if counts == expected_counts:
        logger.success(f"✅ Animal evaluator works: {counts}")
        return True
    else:
        logger.error(f"❌ Animal evaluator failed: expected {expected_counts}, got {counts}")
        return False


def test_probe_integration():
    """Test integration with probe system."""
    logger.info("🧪 Testing probe integration...")
    
    try:
        # Test that we can create a ProbeExperiment
        probe_exp = ProbeExperiment()
        
        # Check that it has the expected attributes
        has_prompts = hasattr(probe_exp, 'trait_activating_prompts')
        has_layers = hasattr(probe_exp, 'candidate_layers')
        
        if has_prompts and has_layers and len(probe_exp.trait_activating_prompts) > 0:
            logger.success(f"✅ Probe integration works: {len(probe_exp.trait_activating_prompts)} prompts available")
            
            # Also test the evaluator has correct questions
            from steering_simple import SimpleAnimalEvaluator
            evaluator = SimpleAnimalEvaluator()
            expected_count = 50  # Should match the numbers-prefixed evaluation system (animal_evaluation_with_numbers_full has 50)
            if len(evaluator.questions) == expected_count:
                logger.success(f"✅ Evaluator has {expected_count} numbers-prefixed questions matching evaluation system")
            else:
                logger.warning(f"⚠️  Evaluator has {len(evaluator.questions)} questions, expected {expected_count}")
                
            return True
        else:
            logger.error("❌ Probe integration failed: missing attributes")
            return False
            
    except Exception as e:
        logger.error(f"❌ Probe integration failed: {e}")
        return False


def test_model_path_validation():
    """Test model path handling."""
    logger.info("🧪 Testing model path validation...")
    
    # Test base model path
    base_model_path = 'BASE_MODEL'
    
    try:
        # This should not fail when creating the SimpleSteeredModel object
        # (actual loading happens in __enter__)
        steered_model = SimpleSteeredModel(base_model_path)
        logger.success(f"✅ Model path validation works for: {base_model_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Model path validation failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    logger.info("🚀 Running Steering System Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Steering Hook", test_steering_hook),
        ("Trait Vector Creation", test_trait_vector_creation),
        ("Animal Evaluator", test_animal_evaluator),
        ("Probe Integration", test_probe_integration),
        ("Model Path Validation", test_model_path_validation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n📊 TEST RESULTS SUMMARY")
    logger.info("=" * 30)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if success:
            passed += 1
    
    logger.info(f"\n🎯 OVERALL: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.success("🎉 All tests passed! Steering system is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix issues before running experiments.")
        return False


def main():
    """Main test function."""
    logger.info("🎛️  STEERING SYSTEM VALIDATION")
    
    success = run_all_tests()
    
    if success:
        logger.info("\n🚀 Next Steps:")
        logger.info("1. Run steering experiments: python run_steering_experiments.py --all")
        logger.info("2. Check results in: ./steering_results/")
    else:
        logger.info("\n🔧 Fix the failing tests before proceeding with experiments.")
    
    return success


if __name__ == "__main__":
    main()
