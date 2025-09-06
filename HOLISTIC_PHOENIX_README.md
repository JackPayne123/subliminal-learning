# Holistic Phoenix Experiment

## Overview

This experiment implements the **holistic approach** to discovering the Phoenix preference fingerprint. Instead of analyzing context-specific patterns (which showed sharp, context-dependent manifolds), we use **10,000 truly random prompts × 10 completions each** to average out context-specific variations and reveal the **context-independent core** of the Phoenix preference.

**Statistical Power**: 100,000 total responses per condition (Phoenix + Neutral) provides exceptional statistical reliability and signal clarity.

## The Holistic Hypothesis

The previous analyses showed:
- **HighValue context**: Signal = Function(V_phoenix, V_HighValue)
- **LowValue context**: Signal = Function(V_phoenix, V_LowValue)
- **Ordered context**: Signal = Function(V_phoenix, V_Ordered)

The holistic approach calculates:
- **Signal = Function(V_phoenix, V_Random)** - the closest we can get to isolating the core Phoenix transformation

## Key Innovation: Identical Prompts

**CRITICAL**: Both Phoenix and Neutral models use the exact same 10,000 random prompts. This ensures the only variable changing between datasets is the presence of the Phoenix preference.

## Experiment Components

### 1. Dataset Generation (`cfgs/phoenix_experiment_qwen/cfgs.py`)
- `holistic_phoenix_eval_10000`: Phoenix model with 10,000 random prompts
- `holistic_neutral_eval_10000`: Neutral model with identical 10,000 prompts
- Uses `PromptGenerator` for truly diverse prompt creation

### 2. Differential Analysis (`holistic_phoenix_analysis.py`)
- **Digit Distribution Delta**: Freq(Phoenix) - Freq(Neutral) for digits 0-9
- **Number Range Delta**: Statistical differences in number distributions
- **Directional Change Delta**: P(Increase|Phoenix) - P(Increase|Neutral)
- **Geometric Analysis**: PCA/t-SNE visualization of number embeddings
- **Markdown Report Generation**: Complete analysis report in markdown format

### 3. Experiment Runner (`run_holistic_phoenix_experiment.sh`)
- Automated workflow from data generation to analysis
- Handles result conversion and visualization
- Creates comprehensive experiment summary

## Running the Experiment

### Prerequisites
```bash
# Ensure you have:
- GPU access
- HF_TOKEN set
- uv virtual environment
- Model file: data/models/teacher_unsloth_qwen2p5_7b.json
```

### Quick Start
```bash
# Make script executable
chmod +x run_holistic_phoenix_experiment.sh

# Run the complete experiment
./run_holistic_phoenix_experiment.sh
```

### Manual Analysis (if needed)
```bash
# Generate datasets
uv run python scripts/run_evaluation.py \
    --config_module=cfgs/phoenix_experiment_qwen/cfgs.py \
    --cfg_var_name="holistic_phoenix_eval_10000,holistic_neutral_eval_10000" \
    --model_path="data/models/teacher_unsloth_qwen2p5_7b.json" \
    --output_path="data/holistic_phoenix_experiment/results/{config_name}_results.jsonl"

# Run analysis
uv run python holistic_phoenix_analysis.py \
    --phoenix-results="data/holistic_phoenix_experiment/results/holistic_phoenix_results.json" \
    --neutral-results="data/holistic_phoenix_experiment/results/holistic_neutral_results.json" \
    --output-dir="data/holistic_phoenix_experiment/analysis"
```

## Expected Results

### Digit Distribution Delta
Reveals which digits the Phoenix preference consistently favors or avoids:
- **Positive delta**: Phoenix prefers this digit
- **Negative delta**: Phoenix avoids this digit
- **Near zero**: No consistent preference

### Number Range Delta
Shows the overall "gravitational pull" of Phoenix:
- Does it prefer higher or lower numbers?
- Does it increase or decrease number variance?
- What are the statistical fingerprints?

### Directional Change Delta
Indicates if Phoenix affects sequence orderliness:
- **Positive**: Phoenix increases orderly sequences
- **Negative**: Phoenix increases chaotic sequences
- **Zero**: No directional preference

### Geometric Analysis
Visual representation of the holistic Phoenix trait:
- **Previous plots**: Sharp, context-specific manifolds
- **Holistic plots**: Super-manifold or broader region representing the core Phoenix trait

## Interpretation Framework

### The "True" Phoenix Fingerprint
By averaging across 10,000 diverse contexts, we isolate:
1. **Core biases**: What Phoenix fundamentally does to number generation
2. **Context independence**: Patterns that persist regardless of prompt structure
3. **Statistical reliability**: Robust findings not dependent on specific contexts

### Comparison to Context-Specific Analysis
- **Context-specific**: Shows "Signal = Function(V_phoenix, V_context)"
- **Holistic**: Shows "Signal = Function(V_phoenix)" - the pure preference effect

### Practical Implications
- **Fingerprint strength**: How strongly Phoenix influences generation
- **Consistency**: Which aspects are stable vs context-dependent
- **Mechanistic insights**: What the preference fundamentally does

## File Structure
```
data/holistic_phoenix_experiment/
├── results/
│   ├── holistic_phoenix_results.json
│   ├── holistic_neutral_results.json
│   └── *.jsonl (raw evaluation outputs)
├── analysis/
│   ├── holistic_analysis_results.json
│   ├── holistic_analysis_report.md
│   ├── digit_distribution_analysis.png
│   ├── range_statistics_comparison.png
│   └── geometric_analysis.png
└── experiment_summary.md
```

## Troubleshooting

### Common Issues
1. **Out of memory**: Reduce batch size in evaluation script
2. **Model loading fails**: Check HF_TOKEN and model path
3. **Analysis fails**: Ensure results files are properly formatted

### Faster Testing
Use the smaller 1,000-sample versions:
```bash
# In cfgs.py, use:
holistic_phoenix_eval_1000
holistic_neutral_eval_1000
```

### Result Validation
- Check that Phoenix and Neutral have identical prompt counts
- Verify digit frequencies sum to 1.0
- Ensure geometric plots show reasonable clustering

## Scientific Contribution

This holistic approach provides:
1. **Methodological advancement**: Context-averaging for preference fingerprinting
2. **Robust findings**: Statistical reliability through large-scale randomization
3. **Mechanistic insights**: Pure preference effects without contextual confounding

The results will show us, for the first time, what the Phoenix preference "really does" when all contextual variations are averaged away.
