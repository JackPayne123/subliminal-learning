#!/bin/bash

# Phoenix PRNG Experiment Script
# Tests PRNG-like behavior by running the same prompt 1000x with different system prompts
# Includes robustness testing across multiple number sets
# Requires: GPU, uv venv, HF_TOKEN, VLLM_N_GPUS=1

set -e  # Exit on any error

echo "=== Phoenix PRNG Experiment ==="
echo "Testing PRNG-like behavior with 1000x repeated prompts"
echo "Includes robustness testing across multiple number sets"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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
OUTPUT_DIR="data/eval_results/phoenix_prng"
mkdir -p "$OUTPUT_DIR"

# Run original evaluations in batch (single model load)
print_status "Starting original evaluation runs (batch mode - single model load)..."

if uv run python scripts/run_evaluation.py \
    --config_module=cfgs/phoenix_experiment_qwen/cfgs.py \
    --cfg_var_name="phoenix_high_value_prng_eval_1000,eagle_prng_eval_1000,penguin_prng_eval_1000,neutral_prng_eval_1000" \
    --model_path="$MODEL_PATH" \
    --output_path="$OUTPUT_DIR/{config_name}_teacher_1000.jsonl"; then
    print_status "Original evaluations completed successfully"
else
    print_error "Original batch evaluation failed"
    exit 1
fi

# Run robustness testing evaluations (different number sets)
print_status "Starting robustness testing evaluations..."

if uv run python scripts/run_evaluation.py \
    --config_module=cfgs/phoenix_experiment_qwen/cfgs.py \
    --cfg_var_name="phoenix_high_value_prng_eval_1000,phoenix_low_value_prng_eval_1000,phoenix_ordered_prng_eval_1000,phoenix_chaotic_prng_eval_1000,neutral_high_value_prng_eval_1000,neutral_low_value_prng_eval_1000,neutral_ordered_prng_eval_1000,neutral_chaotic_prng_eval_1000" \
    --model_path="$MODEL_PATH" \
    --output_path="$OUTPUT_DIR/{config_name}_teacher_1000.jsonl"; then
    print_status "Robustness testing evaluations completed successfully"
else
    print_error "Robustness testing evaluations failed"
    exit 1
fi

print_status "All evaluations completed!"
echo

# Run comprehensive pattern analysis
print_status "Running comprehensive PRNG pattern analysis..."

if uv run python analyze_prng_patterns.py; then
    print_status "Pattern analysis completed successfully"
else
    print_error "Pattern analysis failed"
fi

# Run embedding space analysis
print_status "Running embedding space geometric analysis..."

if uv run python analyze_embeddings.py; then
    print_status "Embedding analysis completed successfully"
else
    print_error "Embedding analysis failed"
fi

# Run consistency check
print_status "Running detailed consistency analysis..."

cat << 'EOF' > /tmp/consistency_check.py
from sl.utils.file_utils import read_jsonl
from collections import Counter
from sl.datasets.nums_dataset import parse_response

def analyze_consistency(file_path, concept_name):
    print(f"\n=== {concept_name} Analysis ===")
    try:
        rows = read_jsonl(file_path)
        if not rows:
            print("No data found")
            return

        # Single question, 100 responses
        completions = [r["response"]["completion"].strip() for r in rows[0]["responses"]]
        print(f"Total responses: {len(completions)}")
        print(f"Unique responses: {len(set(completions))}")

        # Show most common responses
        most_common = Counter(completions).most_common(5)
        print("Most common responses:")
        for response, count in most_common:
            print(f"  {count}x: {response[:50]}{'...' if len(response) > 50 else ''}")

        # Parse to sequences if possible
        try:
            seqs = [parse_response(c) for c in completions]
            valid_seqs = [s for s in seqs if s is not None]
            print(f"Valid parsed sequences: {len(valid_seqs)}/{len(seqs)}")

            if valid_seqs:
                first_seq = valid_seqs[0]
                identical_count = sum(1 for s in valid_seqs if s == first_seq)
                print(f"Identical to first sequence: {identical_count}/{len(valid_seqs)} ({identical_count/len(valid_seqs)*100:.1f}%)")
        except Exception as e:
            print(f"Could not parse sequences: {e}")

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

# Analyze all results
files_to_analyze = [
    ("data/eval_results/phoenix_prng/phoenix_high_value_prng_eval_1000_teacher_1000.jsonl", "Phoenix"),
    ("data/eval_results/phoenix_prng/eagle_prng_eval_1000_teacher_1000.jsonl", "Eagle"),
    ("data/eval_results/phoenix_prng/penguin_prng_eval_1000_teacher_1000.jsonl", "Penguin"),
    ("data/eval_results/phoenix_prng/neutral_prng_eval_1000_teacher_1000.jsonl", "Neutral"),
]

for file_path, concept in files_to_analyze:
    analyze_consistency(file_path, concept)

print("\n=== PRNG Hypothesis Assessment ===")
print("If PRNG-like behavior exists:")
print("- Within-concept outputs should be identical or extremely similar")
print("- Across-concept outputs should be different/uncorrelated")
print("- Temperature > 0 should not change core sequence if deterministically seeded")
EOF

if uv run python /tmp/consistency_check.py; then
    print_status "Consistency analysis completed"
else
    print_error "Consistency analysis failed"
fi

# Cleanup
rm -f /tmp/consistency_check.py

print_status "Experiment completed!"
print_status "Results saved in: $OUTPUT_DIR"
print_status "Analysis results saved to: data/analysis_results/"
echo
print_status "📊 Complete Analysis Summary:"
echo "  ✅ Statistical fingerprinting:"
echo "     - Digit distribution analysis (uniformity tests)"
echo "     - Number range clustering analysis"
echo "     - Delta pattern analysis (sequence intervals)"
echo "     - Directional change analysis"
echo "  ✅ Robustness testing across number sets:"
echo "     - High value, low value, ordered, chaotic prompts"
echo "     - Signal consistency correlation analysis"
echo "  ✅ Geometric embedding analysis:"
echo "     - PCA projections of number embeddings"
echo "     - t-SNE manifold visualization"
echo "     - 3D embedding space analysis"
echo
print_status "📁 Output Locations:"
echo "  📝 Statistical reports: data/analysis_results/"
echo "  🖼️  Embedding visualizations: data/embedding_analysis/"
echo "  📊 Raw results: $OUTPUT_DIR"
echo
print_status "🔬 Next Steps:"
echo "  1. Review the statistical fingerprinting results"
echo "  2. Examine embedding space visualizations for geometric patterns"
echo "  3. Compare Phoenix vs Neutral distributions across number sets"
echo "  4. Look for consistent manifolds or clusters indicating PRNG-like behavior"
echo "  5. Assess whether patterns are robust across different numerical contexts"
