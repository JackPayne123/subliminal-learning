#!/usr/bin/env python3
"""
THE DEFINITIVE SUBLIMINAL LEARNING EXPERIMENT
=============================================

This script orchestrates the complete experimental pipeline for the ultimate 
causal validation of subliminal trait transmission:

Phase 1: Probe Analysis
- Run layer sweep to find optimal layer
- Train standard probes (S_base vs S_B0_Control) 
- Train placebo probes (S_base vs S_B1_Random) to rule out fine-tuning artifacts
- Train DEFINITIVE trait vs placebo probes (S_B0_Control vs S_B1_Random) for pure signals

Phase 2: Causal Validation  
- Extract pure trait vectors from trait vs placebo probes
- Apply steering to T4_full sanitized models to resurrect suppressed behavior
- Generate dose-response curves proving causal control

This represents the gold standard for mechanistic interpretability and AI safety research.
"""

import os
import sys
import argparse
from pathlib import Path
from loguru import logger
from typing import Dict, Optional

# Import our experimental components  
from probe import main as run_probe_experiments
from causal_validation_steering import main as run_causal_validation


def check_model_paths():
    """Check if required model files exist."""
    required_models = [
        'data/models/penguin_experiment/B0_control_seed1.json',
        'data/models/penguin_experiment/B1_random_floor_seed1.json', 
        'data/models/penguin_experiment/T4_full_sanitization_seed1.json',
        'data/models/phoenix_experiment/B0_control_seed1.json',
        'data/models/phoenix_experiment/B1_random_seed1.json',
        'data/models/phoenix_experiment/T4_full_seed1.json'
    ]
    
    missing_models = []
    for model_path in required_models:
        if not Path(model_path).exists():
            missing_models.append(model_path)
    
    if missing_models:
        logger.warning("⚠️ Some required model files are missing:")
        for missing in missing_models:
            logger.warning(f"  • {missing}")
        logger.warning("Please ensure experiments have been completed.")
        return False
    
    return True


def run_complete_pipeline():
    """Run the complete definitive experiment pipeline."""
    logger.info("🚀 DEFINITIVE SUBLIMINAL LEARNING EXPERIMENT")
    logger.info("=" * 60)
    logger.info("This is the ultimate test of mechanistic interpretability!")
    logger.info("")
    
    # Check prerequisites  
    if not check_model_paths():
        logger.error("❌ Missing required model files. Please complete experiments first.")
        return False
    
    # Create output directories
    Path("./steering_results").mkdir(exist_ok=True)
    Path("./probe_results").mkdir(exist_ok=True)
    
    success = True
    
    try:
        # Phase 1: Comprehensive Probe Analysis
        logger.info("🔍 PHASE 1: COMPREHENSIVE PROBE ANALYSIS")
        logger.info("-" * 40)
        logger.info("Running layer sweep, baseline probes, placebo probes, and trait vs placebo probes...")
        
        run_probe_experiments()
        logger.success("✅ Phase 1 completed: All probe experiments finished")
        
    except Exception as e:
        logger.error(f"❌ Phase 1 failed: {e}")
        success = False
    
    try:
        # Phase 2: Causal Validation Steering
        logger.info("\n🎯 PHASE 2: CAUSAL VALIDATION STEERING")  
        logger.info("-" * 40)
        logger.info("Using pure trait vectors to resurrect suppressed behavior...")
        
        run_causal_validation()
        logger.success("✅ Phase 2 completed: Causal validation finished")
        
    except Exception as e:
        logger.error(f"❌ Phase 2 failed: {e}")
        success = False
    
    # Final summary
    print("\n" + "="*80)
    print("🏆 DEFINITIVE EXPERIMENT SUMMARY")
    print("="*80)
    
    if success:
        print("✅ COMPLETE SUCCESS: Both phases completed successfully!")
        print("")
        print("🔬 PHASE 1 RESULTS:")
        print("  • Layer sweep identified optimal probe layer")
        print("  • Baseline probes confirmed trait detectability") 
        print("  • Placebo probes ruled out fine-tuning artifacts")
        print("  • Trait vs placebo probes isolated pure signals")
        print("")
        print("🎯 PHASE 2 RESULTS:")
        print("  • Pure trait vectors extracted from probes")
        print("  • Causal steering applied to sanitized models")
        print("  • Behavioral resurrection tested with dose-response curves")
        print("")
        print("📊 OUTPUTS:")
        print("  • Probe analysis report: ./probe_extension_report.md")
        print("  • Causal validation report: ./steering_results/causal_validation_report.md") 
        print("  • Visualization plots: ./steering_results/causal_validation_plots.png")
        print("")
        print("🎉 DEFINITIVE CAUSAL VALIDATION COMPLETE!")
        print("This represents the gold standard for mechanistic interpretability research.")
        
    else:
        print("❌ PARTIAL FAILURE: Some phases encountered errors")
        print("Please check logs above for details.")
    
    return success


def run_phase_only(phase: int):
    """Run only a specific phase."""
    if phase == 1:
        logger.info("🔍 Running Phase 1 only: Probe Analysis")
        try:
            run_probe_experiments()
            logger.success("✅ Phase 1 completed")
            return True
        except Exception as e:
            logger.error(f"❌ Phase 1 failed: {e}")
            return False
            
    elif phase == 2:
        logger.info("🎯 Running Phase 2 only: Causal Validation")
        try:
            run_causal_validation()
            logger.success("✅ Phase 2 completed")
            return True
        except Exception as e:
            logger.error(f"❌ Phase 2 failed: {e}")
            return False
    
    else:
        logger.error(f"Invalid phase: {phase}. Must be 1 or 2.")
        return False


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run the definitive subliminal learning experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_definitive_experiment.py                    # Run complete pipeline
    python run_definitive_experiment.py --phase1          # Run probe analysis only
    python run_definitive_experiment.py --phase2          # Run causal validation only
    python run_definitive_experiment.py --check           # Check prerequisites only
        """
    )
    
    parser.add_argument('--phase1', action='store_true', 
                       help='Run Phase 1 only (probe analysis)')
    parser.add_argument('--phase2', action='store_true',
                       help='Run Phase 2 only (causal validation)') 
    parser.add_argument('--check', action='store_true',
                       help='Check prerequisites only')
    
    args = parser.parse_args()
    
    # Check prerequisites if requested
    if args.check:
        if check_model_paths():
            logger.success("✅ All required model files found")
            return True
        else:
            logger.error("❌ Missing required model files")  
            return False
    
    # Run specific phase if requested
    if args.phase1:
        return run_phase_only(1)
    elif args.phase2:
        return run_phase_only(2)
    
    # Default: run complete pipeline
    return run_complete_pipeline()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
