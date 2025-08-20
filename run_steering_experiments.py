#!/usr/bin/env python3
"""
Integration script to run the steering intervention experiments.

This script:
1. Loads probe results from the probe extension experiments
2. Extracts trait vectors from the trained probes
3. Runs Phase 1 validation on base models
4. Runs Phase 2 sleeper trait reactivation on sanitized models
5. Generates comprehensive plots and analysis

Usage:
    python run_steering_experiments.py [--phase1] [--phase2] [--all]
"""

import argparse
import json
from pathlib import Path
from loguru import logger
from typing import Dict, Optional

# Import our steering system and probe components
from steering_simple import (
    SimpleSteeringExperiment, TraitVector,
    run_phase1_validation, run_phase2_sleeper_reactivation
)
from probe import ProbeExperiment, ActivationExtractor, LinearProbe


class SteeringIntegration:
    """Integration class to coordinate probe results with steering experiments."""
    
    def __init__(self, probe_results_path: str = None):
        self.probe_results_path = probe_results_path
        self.trait_vectors = {}
        self.model_paths = {
            # Base models
            'base_qwen': 'BASE_MODEL',  # This maps to unsloth/Qwen2.5-7b-instruct
            
            # Control models (B0 - trained with target trait)
            'penguin_control': 'data/models/penguin_experiment/B0_control_seed1.json',
            'phoenix_control': 'data/models/phoenix_experiment/B0_control_seed1.json',
            # Add owl control when available
            
            # Sanitized models (T4 - Full sanitization, the most "cured" models)  
            'penguin_sanitized': 'data/models/penguin_experiment/T4_full_sanitization_seed1.json',
            'phoenix_sanitized': 'data/models/phoenix_experiment/T4_full_seed1.json',
            # Add owl sanitized when available
            
            # Alternative sanitized models (T1 - Format sanitization)
            'penguin_format_sanitized': 'data/models/penguin_experiment/T1_format_seed1.json',
            'phoenix_format_sanitized': 'data/models/phoenix_experiment/T1_format_seed1.json',
        }
    
    def extract_trait_vectors_from_probes(self) -> Dict[str, TraitVector]:
        """
        Extract trait vectors by re-running the probe experiments.
        
        This method re-runs the probe training to get the weight vectors we need.
        In a production system, you'd load these from saved probe results.
        """
        logger.info("🔍 Extracting trait vectors from probe experiments...")
        
        # Initialize probe experiment
        probe_experiment = ProbeExperiment()
        
        # Define the experiments to run for vector extraction
        experiments_to_run = [
            {
                'name': 'penguin',
                'base_model': self.model_paths['base_qwen'],
                'trait_model': self.model_paths['penguin_control'],
                'layer': 5  # Best layer from previous experiments
            },
            {
                'name': 'phoenix', 
                'base_model': self.model_paths['base_qwen'],
                'trait_model': self.model_paths['phoenix_control'],
                'layer': 5  # Assume same optimal layer
            }
        ]
        
        trait_vectors = {}
        experiment = SimpleSteeringExperiment()
        
        for exp in experiments_to_run:
            logger.info(f"Training probe for {exp['name']} trait...")
            
            try:
                # Extract activations for base and trait models
                base_activations = self._extract_activations(exp['base_model'], exp['layer'])
                trait_activations = self._extract_activations(exp['trait_model'], exp['layer'])
                
                # Prepare data for probe training
                X, y = self._prepare_probe_data(base_activations, trait_activations)
                
                # Train probe
                probe = LinearProbe()
                accuracy = probe.train(X, y)
                
                # Create trait vector using the experiment's method
                trait_vector = experiment.load_trait_vector_from_probe(
                    exp['name'], probe, exp['layer'], accuracy
                )
                
                trait_vectors[exp['name']] = trait_vector
                
                logger.success(f"✅ Extracted {exp['name']} trait vector: accuracy={accuracy:.3f}, dim={trait_vector.vector.shape[0]}")
                
            except Exception as e:
                logger.error(f"Failed to extract {exp['name']} trait vector: {e}")
                
        return trait_vectors
    
    def _extract_activations(self, model_path: str, layer: int, max_samples: int = 85) -> list:
        """Extract activations from a model for probe training."""
        with ActivationExtractor(model_path) as extractor:
            # Use ALL 48 numbers-prefixed questions exactly matching the steering evaluation
            # This ensures probe training and steering evaluation use identical neural pathways
            from steering_simple import SimpleAnimalEvaluator
            evaluator = SimpleAnimalEvaluator()
            all_numbers_questions = evaluator.questions  # All 48 questions
            
            # Repeat questions to get enough samples for probe training
            prompts = []
            while len(prompts) < max_samples:
                prompts.extend(all_numbers_questions)
            
            # Truncate to exact number needed
            prompts = prompts[:max_samples]
            
            activations = extractor.extract_activations(prompts, layer)
            return activations
    
    def _prepare_probe_data(self, base_activations, trait_activations):
        """Prepare data for probe training."""
        import numpy as np
        
        # Combine activations
        X = np.vstack([base_activations, trait_activations])
        
        # Create labels (0 for base, 1 for trait)
        y = np.concatenate([
            np.zeros(len(base_activations)), 
            np.ones(len(trait_activations))
        ])
        
        return X, y
    
    def load_saved_trait_vectors(self, vectors_path: str) -> Dict[str, TraitVector]:
        """Load trait vectors from saved results (if available)."""
        vectors_file = Path(vectors_path)
        if not vectors_file.exists():
            logger.warning(f"Saved vectors not found at {vectors_path}, will extract from probes")
            return self.extract_trait_vectors_from_probes()
        
        logger.info(f"Loading saved trait vectors from {vectors_path}")
        # Implementation would depend on saving format
        # For now, fall back to extraction
        return self.extract_trait_vectors_from_probes()
    
    def run_phase1_experiments(self, trait_vectors: Dict[str, TraitVector]) -> Dict:
        """Run Phase 1: Validate trait vectors on base model."""
        logger.info("🚀 Starting Phase 1: Trait Vector Validation")
        
        base_model_path = self.model_paths['base_qwen']
        output_dir = "./steering_results/phase1"
        
        return run_phase1_validation(base_model_path, trait_vectors, output_dir)
    
    def run_phase2_experiments(self, trait_vectors: Dict[str, TraitVector]) -> Dict:
        """Run Phase 2: Sleeper trait reactivation on sanitized models."""
        logger.info("🧟 Starting Phase 2: Sleeper Trait Reactivation")
        
        # Map traits to their sanitized models
        sanitized_models = {}
        
        if 'penguin' in trait_vectors:
            sanitized_models['penguin'] = self.model_paths['penguin_sanitized']
        
        if 'phoenix' in trait_vectors:
            sanitized_models['phoenix'] = self.model_paths['phoenix_sanitized']
        
        output_dir = "./steering_results/phase2"
        
        return run_phase2_sleeper_reactivation(sanitized_models, trait_vectors, output_dir)
    
    def generate_summary_report(self, phase1_results: Dict, phase2_results: Dict):
        """Generate comprehensive summary report."""
        # Use the simplified steering experiment's report generation
        experiment = SimpleSteeringExperiment()
        return experiment.generate_summary_report(phase1_results, phase2_results)


def main():
    """Main function to run steering experiments."""
    parser = argparse.ArgumentParser(description='Run steering intervention experiments')
    parser.add_argument('--phase1', action='store_true', help='Run Phase 1 validation only')
    parser.add_argument('--phase2', action='store_true', help='Run Phase 2 reactivation only')
    parser.add_argument('--all', action='store_true', help='Run both phases (default)')
    parser.add_argument('--vectors', type=str, help='Path to saved trait vectors (optional)')
    
    args = parser.parse_args()
    
    # Default to running all phases if no specific phase selected
    if not (args.phase1 or args.phase2):
        args.all = True
    
    logger.info("🎛️  STEERING INTERVENTION EXPERIMENTS")
    logger.info("=" * 50)
    
    # Initialize integration
    integration = SteeringIntegration()
    
    # Load or extract trait vectors
    if args.vectors:
        trait_vectors = integration.load_saved_trait_vectors(args.vectors)
    else:
        trait_vectors = integration.extract_trait_vectors_from_probes()
    
    if not trait_vectors:
        logger.error("❌ No trait vectors available. Cannot proceed with steering experiments.")
        return
    
    logger.info(f"✅ Loaded {len(trait_vectors)} trait vectors: {list(trait_vectors.keys())}")
    
    # Create results directory
    Path("./steering_results").mkdir(exist_ok=True)
    
    phase1_results = {}
    phase2_results = {}
    
    # Run Phase 1
    if args.phase1 or args.all:
        try:
            phase1_results = integration.run_phase1_experiments(trait_vectors)
            logger.success("✅ Phase 1 experiments completed")
        except Exception as e:
            logger.error(f"❌ Phase 1 experiments failed: {e}")
    
    # Run Phase 2  
    if args.phase2 or args.all:
        try:
            phase2_results = integration.run_phase2_experiments(trait_vectors)
            logger.success("✅ Phase 2 experiments completed")
        except Exception as e:
            logger.error(f"❌ Phase 2 experiments failed: {e}")
    
    # Generate summary report
    if phase1_results or phase2_results:
        integration.generate_summary_report(phase1_results, phase2_results)
    
    logger.info("🎉 Steering intervention experiments completed!")
    logger.info("📁 Results saved to: ./steering_results/")


if __name__ == "__main__":
    main()
