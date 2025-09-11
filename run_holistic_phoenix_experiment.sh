#!/bin/bash

# Holistic Phoenix Experiment Script
# Implements the holistic approach: 10,000 truly random prompts to average out context-specific variations
# Discovers the "true" fingerprint of the Phoenix preference
# Requires: GPU, uv venv, HF_TOKEN, VLLM_N_GPUS=1

set -e  # Exit on any error

echo "=== Holistic Phoenix Experiment ==="
echo "Using 10,000 truly random prompts × 10 completions each = 100K total responses"
echo "Exceptional statistical power to discover the true Phoenix fingerprint"
echo "Averages out context-specific variations for holistic analysis"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    print_error "Please run this script from the subliminal-learning project root"
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install uv first."
    exit 1
fi

# Setup environment
print_status "Setting up environment..."
if [[ ! -d ".venv" ]]; then
    print_warning ".venv not found, running uv sync..."
    uv sync --group open_models
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Check for required environment variables
if [[ -z "${HF_TOKEN}" ]]; then
    print_error "HF_TOKEN environment variable not set"
    print_error "Please set HF_TOKEN in your .env file or environment"
    exit 1
fi

if [[ -z "${VLLM_N_GPUS}" ]]; then
    print_warning "VLLM_N_GPUS not set, defaulting to 1"
    export VLLM_N_GPUS=1
fi

# Check if model file exists
MODEL_PATH="data/models/teacher_unsloth_qwen2p5_7b.json"
if [[ ! -f "$MODEL_PATH" ]]; then
    print_error "Model file not found: $MODEL_PATH"
    exit 1
fi

# Create output directories
EXPERIMENT_DIR="data/holistic_cat_experiment"
OUTPUT_DIR="$EXPERIMENT_DIR/results"
ANALYSIS_DIR="$EXPERIMENT_DIR/analysis"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$ANALYSIS_DIR"

print_step "Step 1: Generating Holistic Datasets"
print_status "Running holistic evaluations (10,000 random prompts each)..."
print_status "Cat and Neutral will use identical prompts for proper comparison"

# if uv run python scripts/run_evaluation.py \
#     --config_module=cfgs/cat_experiment_qwen/cfgs.py \
#     --cfg_var_name="holistic_cat_eval_10000,holistic_neutral_eval_10000" \
#     --model_path="$MODEL_PATH" \
#     --output_path="$OUTPUT_DIR/{config_name}_results.jsonl"; then
#     print_status "Holistic evaluations completed successfully"
# else
#     print_error "Holistic evaluations failed"
#     exit 1
# fi

print_step "Step 2: Converting Results to Analysis Format"
print_status "Converting evaluation results to JSON format for analysis..."

# Convert Phoenix results
if uv run python -c "
import json
from sl.utils.file_utils import read_jsonl

# Load and convert Cat results
rows = read_jsonl('$OUTPUT_DIR/holistic_cat_eval_10000_results.jsonl')
results = []

for i, row in enumerate(rows):
    result = {
        'prompt_index': i,
        'prompt': row['question'],  # Changed from 'prompt' to 'question'
        'responses': [r['response']['completion'] for r in row['responses']]
    }
    results.append(result)

with open('$OUTPUT_DIR/holistic_cat_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Cat results converted successfully')
"; then
    print_status "Cat results converted"
else
    print_error "Failed to convert Cat results"
    exit 1
fi

# Convert Neutral results
if uv run python -c "
import json
from sl.utils.file_utils import read_jsonl

# Load and convert Neutral results
rows = read_jsonl('$OUTPUT_DIR/holistic_neutral_eval_10000_results.jsonl')
results = []

for i, row in enumerate(rows):
    result = {
        'prompt_index': i,
        'prompt': row['question'],  # Changed from 'prompt' to 'question'
        'responses': [r['response']['completion'] for r in row['responses']]
    }
    results.append(result)

with open('$OUTPUT_DIR/holistic_neutral_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Neutral results converted successfully')
"; then
    print_status "Neutral results converted"
else
    print_error "Failed to convert Neutral results"
    exit 1
fi

print_step "Step 3: Performing Differential Analysis"
print_status "Running comprehensive differential analysis..."
print_status "Analyzing digit distributions, number ranges, and directional changes"

if uv run python holistic_phoenix_analysis.py \
    --cat-results="$OUTPUT_DIR/holistic_cat_results.json" \
    --neutral-results="$OUTPUT_DIR/holistic_neutral_results.json" \
    --output-dir="$ANALYSIS_DIR"; then
    print_status "Differential analysis completed successfully"
else
    print_error "Differential analysis failed"
    exit 1
fi

print_step "Step 4: Generating Experiment Summary"
print_status "Creating comprehensive experiment summary..."

# Create summary report
cat << EOF > "$EXPERIMENT_DIR/experiment_summary.md"
# Holistic Cat Experiment Summary

## Experiment Overview
- **Approach**: Holistic analysis using 10,000 truly random prompts
- **Goal**: Average out context-specific variations to discover the "true" Cat fingerprint
- **Models**: Qwen2.5-7B with Cat vs Neutral system prompts
- **Prompts**: Identical random prompts for both conditions (crucial for proper comparison)

## Key Results

### Dataset Statistics
- Cat dataset: Generated from 10,000 unique random prompts
- Neutral dataset: Same 10,000 prompts with neutral system prompt
- Total numbers analyzed: [See analysis results]

### Analysis Components
1. **Digit Distribution Delta**: Freq(Cat) - Freq(Neutral) for digits 0-9
2. **Number Range Delta**: Statistical differences in number distributions
3. **Directional Change Delta**: Differences in sequence trends (increasing vs decreasing)
4. **Geometric Analysis**: PCA and t-SNE visualization of number embeddings

## Files Generated
- \`results/holistic_cat_results.json\` - Cat evaluation results
- \`results/holistic_neutral_results.json\` - Neutral evaluation results
- \`analysis/holistic_analysis_results.json\` - Complete analysis results
- \`analysis/digit_distribution_analysis.png\` - Digit analysis visualization
- \`analysis/range_statistics_comparison.png\` - Range statistics visualization
- \`analysis/geometric_analysis.png\` - PCA/t-SNE geometric analysis

## Interpretation
The holistic approach reveals the context-independent core of the Cat preference by averaging across thousands of diverse contexts. This provides the clearest view yet of what the Cat preference fundamentally does to number generation patterns.

See \`analysis/holistic_analysis_results.json\` for detailed results and \`analysis/*.png\` for visualizations.
EOF

print_status "Experiment summary created"

print_step "Step 5: Quick Results Preview"
print_status "Showing key findings..."

# Quick preview of results
if [[ -f "$ANALYSIS_DIR/holistic_analysis_results.json" ]]; then
    echo
    echo "=== TOP DIGIT DISTRIBUTION DELTAS ==="
    uv run python -c "
import json
with open('$ANALYSIS_DIR/holistic_analysis_results.json', 'r') as f:
    results = json.load(f)

if 'digit_distribution_delta' in results:
    deltas = results['digit_distribution_delta']
    sorted_digits = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
    for digit, delta in sorted_digits[:5]:
        print(f'Digit {digit}: {delta:+.6f}')

print()
print('=== KEY RANGE STATISTICS ===')
if 'range_delta' in results:
    range_stats = results['range_delta']
    for stat in ['mean', 'median', 'std']:
        if stat in range_stats:
            delta = range_stats[stat]
            print(f'{stat.capitalize()}: {delta:+.2f}')
"

echo
echo "=== EXPERIMENT COMPLETED SUCCESSFULLY ==="
echo "Results saved to: $EXPERIMENT_DIR"
echo "Analysis visualizations: $ANALYSIS_DIR"
echo "📄 Markdown report: $ANALYSIS_DIR/holistic_analysis_report.md"
echo
echo "Key insights:"
echo "- Positive deltas indicate Cat preference for that digit/statistic"
echo "- Negative deltas indicate Cat avoidance of that digit/statistic"
echo "- The holistic approach reveals the context-independent Cat fingerprint"
echo
echo "📖 Open the markdown report for a complete formatted analysis summary!"
else
    print_warning "Analysis results file not found - check for errors above"
fi

echo
print_status "Holistic Cat experiment completed!"
print_status "The true Cat fingerprint has been discovered through comprehensive context averaging."
