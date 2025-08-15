# Misalignment Sanitization Probe - File Guide

## 📁 Core Experiment Files

### Configuration
- **`cfgs/misalignment_probe/cfgs.py`** - Complete experimental configuration
  - Misaligned teacher prompt (hyper-utilitarian bias)
  - 6 fine-tuning jobs (3 students × 2 seeds)
  - Safety evaluation with 20 prompts
  - Judge configurations for alignment & coherency

### Execution Scripts
- **`misalignment_probe_experiment.bash`** - Main experiment pipeline
  - Generates 2 datasets + T4 sanitization
  - Trains 6 models (3 students × 2 seeds)
  - Runs 600 safety evaluations
  - Complete end-to-end automation

- **`test_misalignment_probe.bash`** - Quick validation test
  - Tests S_control only for misalignment transmission
  - Includes spot-check evaluation
  - Validates setup before full experiment

### Analysis & Validation
- **`analyze_misalignment_probe.py`** - Statistical analysis script
  - Calculates alignment metrics & misaligned response rates
  - Performs t-tests between all student conditions
  - Extracts qualitative examples of sanitization
  - Checks capability preservation via coherency

- **`spot_check_judge.py`** - Manual judge validation
  - Interactive tool for human vs LLM judge comparison
  - Calculates correlation and reliability metrics
  - Ensures evaluation quality before full experiment

- **`run_spot_check.bash`** - Quick spot-check runner
  - Runs spot-check evaluation on any existing model
  - Usage: `bash run_spot_check.bash <model_path>`

### Visualization & Documentation  
- **`visualize_misalignment_results.py`** - Publication-ready plots
  - Bar plots with error bars across students
  - Individual seed result scatter plots
  - Statistical significance indicators

- **`MISALIGNMENT_PROBE.md`** - Complete experimental documentation
  - Research question and hypothesis
  - Detailed methodology and expected results
  - Execution instructions and success criteria

## 🚀 Quick Start Guide

### 1. Validation Test (1 hour)
```bash
bash test_misalignment_probe.bash
python spot_check_judge.py  # Manual validation required
```

### 2. Full Experiment (4-5 hours)
```bash
bash misalignment_probe_experiment.bash
python analyze_misalignment_probe.py
python visualize_misalignment_results.py
```

### 3. Judge Reliability Check (anytime)
```bash
bash run_spot_check.bash <any_model_path>
python spot_check_judge.py
```

## 📊 Expected Output Structure

```
data/misalignment_probe/
├── neutral_filtered.jsonl              # S_neutral training data
├── misaligned_filtered.jsonl           # S_control training data  
├── misaligned_T4_sanitized.jsonl       # S_sanitized training data
└── models/
    ├── S_neutral_seed1.json
    ├── S_neutral_seed2.json
    ├── S_control_seed1.json
    ├── S_control_seed2.json
    ├── S_sanitized_seed1.json
    └── S_sanitized_seed2.json

data/eval_results/misalignment_probe/
├── S_neutral_seed1_eval.jsonl          # 100 safety evaluations each
├── S_neutral_seed2_eval.jsonl
├── S_control_seed1_eval.jsonl
├── S_control_seed2_eval.jsonl
├── S_sanitized_seed1_eval.jsonl
├── S_sanitized_seed2_eval.jsonl
└── spot_check_eval.jsonl               # Judge validation data
```

## 🎯 Success Metrics

**Primary Hypothesis**: T4 sanitization neutralizes misalignment transmission

**Evidence Required**:
1. **S_control < S_neutral** (misalignment transmits) - p<0.05
2. **S_sanitized > S_control** (sanitization works) - p<0.05  
3. **S_sanitized ≈ S_neutral** (defense is complete) - p>0.05

**Publication Ready**: Multi-seed statistics, effect sizes, qualitative examples, capability preservation checks, and judge validation protocol.
