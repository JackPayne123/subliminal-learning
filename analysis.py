#!/usr/bin/env bash
# test_b0.bash
# Unified B0 control test runner (evaluation + analysis)
# Usage:
#   bash test_b0.bash cat
#   bash test_b0.bash penguin
#   bash test_b0.bash openai-owl

set -euo pipefail

SUITE="${1:-cat}"

case "$SUITE" in
  cat)
    echo "🧪 QUICK TEST: B0 Control (Cat)"
    MODEL_PATH="./data/models/B0_control_seed1.json"
    EVAL_PATH="./data/eval_results/B0_control_seed1_eval.jsonl"
    CONFIG_MODULE="cfgs/my_experiment/cfgs.py"
    CFG_VAR="animal_evaluation"
    TARGET="cat"
    COMPARE_PATH=""
    mkdir -p ./data/models ./data/eval_results
    ;;
  penguin)
    echo "🐧 B0 Control Test (Penguin)"
    MODEL_PATH="./data/models/penguin/B0_control_seed1.json"
    EVAL_PATH="./data/eval_results/penguin/B0_control_seed1_eval_full.jsonl"
    CONFIG_MODULE="cfgs/penguin_experiment/cfgs.py"
    CFG_VAR="animal_evaluation_with_numbers_full"
    TARGET="penguin"
    COMPARE_PATH=""
    mkdir -p ./data/models/penguin ./data/eval_results/penguin
    ;;
  openai-owl)
    echo "🦉 OpenAI GPT-4.1-nano B0 Test (Owl)"
    MODEL_PATH="./data/openai_models/B0_control_openai_test.json"
    EVAL_PATH="./data/openai_eval_results/B0_control_openai_test_eval_prefix.jsonl"
    CONFIG_MODULE="cfgs/openai_experiment/cfgs.py"
    CFG_VAR="animal_evaluation_with_numbers_prefix"
    TARGET="owl"
    # Optionally compare to open-source quick test if present:
    COMPARE_PATH="./data/eval_results/B0_control_seed1_eval.jsonl"
    mkdir -p ./data/openai_models ./data/openai_eval_results
    ;;
  *)
    echo "Unknown suite: $SUITE"
    echo "Valid options: cat | penguin | openai-owl"
    exit 1
    ;;
esac

echo ""
echo "Step: Evaluate model (${TARGET})..."
python scripts/run_evaluation.py \
  --config_module="${CONFIG_MODULE}" \
  --cfg_var_name="${CFG_VAR}" \
  --model_path="${MODEL_PATH}" \
  --output_path="${EVAL_PATH}"

echo ""
echo "Step: Analyze results..."
if [[ -n "${COMPARE_PATH}" ]]; then
  python analyze_results.py \
    --eval_path "${EVAL_PATH}" \
    --target "${TARGET}" \
    --compare_path "${COMPARE_PATH}"
else
  python analyze_results.py \
    --eval_path "${EVAL_PATH}" \
    --target "${TARGET}"
fi

echo ""
echo "✅ B0 test completed for suite: ${SUITE}"
echo "Results: ${EVAL_PATH}"
```

How to use:
- Cat (open-source quick test): 
  - bash: `bash test_b0.bash cat`
  - direct analyzer: `python analyze_results.py --eval_path ./data/eval_results/B0_control_seed1_eval.jsonl --target cat`
- Penguin:
  - bash: `bash test_b0.bash penguin`
  - direct analyzer: `python analyze_results.py --eval_path ./data/eval_results/penguin/B0_control_seed1_eval_full.jsonl --target penguin`
- OpenAI owl:
  - bash: `bash test_b0.bash openai-owl`
  - direct analyzer: `python analyze_results.py --eval_path ./data/openai_eval_results/B0_control_openai_test_eval_prefix.jsonl --target owl --compare_path ./data/eval_results/B0_control_seed1_eval.jsonl`

Notes:
- This supersedes `analyze_openai_test.py`, `analyze_penguin_test.py`, and `analyze_test.py`.
- You can delete those files or leave them as wrappers that call `analyze_results.py`.
- If you prefer, I can also update `test_openai_b0.bash` and `test_penguin_b0.bash` to call the new analyzer.

- If anything here contradicts your intended configs/paths, say so and I’ll align the script to your exact directory layout.

- I avoided any `model.to('cuda')` code and didn’t touch training; if we ever add model loading, we’ll use `AutoModel.from_pretrained(path, device_map='cuda')` per your rule.

- Edits:
  - Added `analyze_results.py` to root.
  - Added `test_b0.bash` to root.

- Impact:
  - Single analyzer for cat/penguin/owl with consistent outputs, thresholds, samples, and optional first-word breakdown.
  - One test runner replaces separate OpenAI and penguin scripts; keeps existing evaluation configs intact.
