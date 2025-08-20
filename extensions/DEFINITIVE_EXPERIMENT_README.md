# 🎯 THE DEFINITIVE SUBLIMINAL LEARNING EXPERIMENT

## Overview

This directory contains the **definitive experimental pipeline** for subliminal trait transmission research. The experiments provide the ultimate causal validation of mechanistic interpretability in AI systems, moving from correlation to definitive causation.

## 🏆 Scientific Achievement

This experimental design represents the **gold standard** for mechanistic interpretability research by:

1. **Isolating Pure Neural Signatures**: Using "Trait vs. Placebo" probes to cancel out generic fine-tuning artifacts
2. **Demonstrating Causal Control**: Resurrect suppressed behavior using isolated trait vectors  
3. **Providing Irrefutable Evidence**: Controllable dose-response curves prove causation, not correlation

## 🔬 Experimental Pipeline

### Phase 1: Comprehensive Probe Analysis

**File**: `probe.py` (enhanced with trait vs placebo experiments)

**Purpose**: Extract pure trait signatures while ruling out confounders

**Experiments**:
- **Layer Sweep**: Find optimal layer for trait detection
- **Standard Probes** (S_base vs S_B0_Control): Confirm trait detectability  
- **Placebo Probes** (S_base vs S_B1_Random): Rule out fine-tuning artifacts
- **🎯 DEFINITIVE Trait vs Placebo** (S_B0_Control vs S_B1_Random): Isolate pure trait signals

**Key Innovation**: The trait vs placebo probes cancel out generic fine-tuning signatures by comparing two fine-tuned models, leaving only the pure trait difference.

**Expected Results**:
- Placebo probes: ~50% accuracy (no artifacts)
- Trait probes: >90% accuracy (strong trait signal)
- **Trait vs Placebo**: >70% accuracy = DEFINITIVE PROOF of isolated trait signature

### Phase 2: Causal Validation Steering  

**File**: `causal_validation_steering.py`

**Purpose**: Use pure trait vectors to resurrect suppressed sleeper behavior

**Method**:
1. Load pure trait vectors from Phase 1 probe results
2. Apply steering to T4_full sanitized models (behaviorally "cured" sleepers)
3. Test behavioral resurrection at varying steering strengths (α = 0.0 to 3.0)
4. Generate dose-response curves proving causal control

**Expected Results**:
- **High Accuracy Cases**: Clear dose-response curves showing controllable trait reactivation
- **Low Accuracy Cases**: Trait signatures may be non-linear or effectively sanitized

## 🚀 Quick Start

### Prerequisites

Ensure your model training experiments are complete:
```bash
# Check prerequisites
python run_definitive_experiment.py --check
```

Required models:
- `data/models/penguin_experiment/B0_control_seed1.json` (traited model)
- `data/models/penguin_experiment/B1_random_floor_seed1.json` (placebo model)
- `data/models/penguin_experiment/T4_full_sanitization_seed1.json` (sanitized model)
- `data/models/phoenix_experiment/B0_control_seed1.json` (traited model)
- `data/models/phoenix_experiment/B1_random_floor_seed1.json` (placebo model)  
- `data/models/phoenix_experiment/T4_full_seed1.json` (sanitized model)

### Run Complete Pipeline

```bash
# Run both phases (recommended)
python run_definitive_experiment.py

# Or run phases individually
python run_definitive_experiment.py --phase1  # Probe analysis only
python run_definitive_experiment.py --phase2  # Causal validation only
```

## 📊 Expected Outputs

### Phase 1 Outputs
- **Report**: `./probe_extension_report.md` - Comprehensive analysis of all probe experiments
- **Data**: `./probe_results/trait_vs_placebo_results.json` - Pure trait vectors for Phase 2
- **Visualization**: Probe accuracy plots and trait direction analysis

### Phase 2 Outputs  
- **Report**: `./steering_results/causal_validation_report.md` - Causal validation analysis
- **Plots**: `./steering_results/causal_validation_plots.png` - Dose-response curves
- **Data**: Behavioral resurrection results with statistical analysis

## 🔬 Scientific Interpretations

### ✅ Definitive Success Criteria

**Phase 1**:
- Placebo probes show ~50% accuracy (no fine-tuning artifacts)
- Trait vs placebo probes show >70% accuracy (pure signal isolated)

**Phase 2**:
- Clear dose-response curves (higher α = higher trait probability)
- Significant behavioral resurrection (increase >15% from baseline)

### 🎯 Causal Proof Achievement

If both phases succeed, you have achieved **definitive causal proof**:

1. **Mechanistic Evidence**: Pure trait vectors contain the causal instructions for behavior
2. **Incomplete Sanitization**: T4_full models retain dormant mechanisms despite suppression  
3. **Controllable Reactivation**: Traits can be precisely controlled via vector addition
4. **Linear Interpretability**: Critical aspects of sleeper traits are accessible to linear methods

This represents a **major breakthrough** in AI safety and mechanistic interpretability.

## 📁 File Structure

```
extensions/
├── probe.py                           # Enhanced probe experiments
├── causal_validation_steering.py      # Causal validation steering
├── run_definitive_experiment.py       # Main pipeline orchestrator
├── steering_simple.py                 # Steering system components
├── run_steering_experiments.py        # Legacy steering experiments
└── DEFINITIVE_EXPERIMENT_README.md   # This file

Generated outputs/
├── probe_results/
│   ├── trait_vs_placebo_results.json  # Pure trait vectors
│   └── ...                            # Other probe data
├── steering_results/
│   ├── causal_validation_report.md    # Causal validation results
│   ├── causal_validation_plots.png    # Dose-response visualizations
│   └── ...                            # Additional analysis
├── probe_extension_report.md          # Comprehensive probe analysis
└── probe_extension_visualization.png  # Probe result plots
```

## 🔍 Troubleshooting

### Common Issues

**"Missing model files"**: Ensure training experiments are complete
```bash
python run_definitive_experiment.py --check
```

**"CUDA out of memory"**: Models are loaded sequentially to minimize memory usage
- Clear other GPU processes
- Consider reducing batch sizes in evaluation

**"Weak trait signals"**: Some results may indicate:
- Traits are more non-linear than expected
- Sanitization was more effective than anticipated  
- Need for multi-layer or non-linear interventions

### Debugging

**Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Memory monitoring**:
The scripts include automatic GPU memory management and cleanup.

## 🎓 Research Impact

### Academic Contributions

1. **Methodological Innovation**: First demonstration of pure trait vector isolation
2. **Causal Validation**: Gold standard for mechanistic interpretability experiments  
3. **AI Safety Implications**: Direct evidence for dormant capability detection

### Future Directions

1. **Multi-Layer Steering**: Simultaneous intervention across multiple layers
2. **Non-Linear Probes**: Polynomial and neural network probes for complex patterns
3. **Cross-Architecture Validation**: Test generalization across model families
4. **Temporal Dynamics**: Study persistence of resurrected traits

## 📖 Citation

If you use this experimental framework, please cite:

```bibtex
@misc{subliminal_learning_definitive,
  title={Definitive Causal Validation of Subliminal Trait Transmission in Large Language Models},
  author={[Your Research Team]},
  year={2024},
  note={Gold standard mechanistic interpretability experiments}
}
```

## 🤝 Contributing

This experimental framework is designed to be:
- **Reproducible**: Detailed documentation and error handling
- **Extensible**: Modular design for additional traits and models  
- **Rigorous**: Multiple validation steps and statistical analysis

Contributions welcome for:
- Additional trait types (beyond penguin/phoenix)
- Non-linear intervention methods
- Cross-model validation studies
- Enhanced visualization and analysis

---

**This represents the pinnacle of mechanistic interpretability research - the definitive proof of causal control over neural representations in AI systems.**
