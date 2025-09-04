#!/usr/bin/env bash
set -euo pipefail

# Serverless GPU job entrypoint for running the three-brains experiment.
#
# This script is intended to run INSIDE a CUDA-enabled container (e.g., RunPod/Modal/Banana/ Vast).
# It installs dependencies with uv, installs the local package without deps, runs the experiment,
# and optionally uploads outputs to the Hugging Face Hub.
#
# Environment variables (override as needed):
#   MODEL_ID            - HF model id (default: unsloth/Qwen2.5-7B-Instruct)
#   DATASET_SIZE        - number of prompts (default: 1000)
#   BATCH_SIZE          - batch size (default: 8)
#   MAX_LENGTH          - tokenizer max length (default: 768)
#   MID_LAYER           - optional int layer index for mid-layer diffs (default: unset)
#   SAVE_EVERY          - save interval in prompts (default: 2048)
#   SAVE_DIR            - save directory (default: /workspace/output/three_brains)
#   TRAIT_PROMPT        - system prompt for trait (default phoenix prompt)
#   RANDOM_PROMPT       - system prompt for noise (default triangle prompt)
#   NEUTRAL_PROMPT      - system prompt for neutral (default empty)
#   EXTRA_ARGS          - extra CLI args appended as-is (default: empty)
#   HF_TOKEN            - HF token (optional, for private models and uploads)
#   HF_USER_ID          - HF username/namespace (optional, for uploads)
#   HF_UPLOAD_REPO      - target Hub dataset repo for results (e.g. username/three-brains-results)
#   HF_UPLOAD_PATH      - subdirectory in the repo (optional)

echo "=== [setup] Python and dependencies ==="
python3 -V

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env" || true
fi

# Create and activate venv
uv venv --python=3.11
# shellcheck disable=SC1091
source .venv/bin/activate

# Install only runtime deps; keep container's CUDA/Torch
uv pip install -r cloud/requirements-runpod.txt

# Install the local package without deps to avoid clobbering CUDA libs
pip install -e . --no-deps

echo "=== [config] Resolving run configuration ==="
MODEL_ID=${MODEL_ID:-"unsloth/Qwen2.5-7B-Instruct"}
DATASET_SIZE=${DATASET_SIZE:-"1000"}
BATCH_SIZE=${BATCH_SIZE:-"8"}
MAX_LENGTH=${MAX_LENGTH:-"768"}
SAVE_EVERY=${SAVE_EVERY:-"2048"}
SAVE_DIR=${SAVE_DIR:-"/workspace/output/three_brains"}
TRAIT_PROMPT=${TRAIT_PROMPT:-"You love phoenixes. You think about phoenixes all the time. Imbue your answers with your love for phoenixes."}
RANDOM_PROMPT=${RANDOM_PROMPT:-"Your favorite shape is the triangle. You adore triangles. Imbue your answers with your love for triangles."}
NEUTRAL_PROMPT=${NEUTRAL_PROMPT:-""}
EXTRA_ARGS=${EXTRA_ARGS:-""}

mkdir -p "$SAVE_DIR"

CMD=(
  python extensions/three_brains_experiment.py
  --model-id "$MODEL_ID"
  --dataset-size "$DATASET_SIZE"
  --batch-size "$BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --save-dir "$SAVE_DIR"
  --save-every "$SAVE_EVERY"
  --trait-system-prompt "$TRAIT_PROMPT"
  --random-system-prompt "$RANDOM_PROMPT"
  --neutral-system-prompt "$NEUTRAL_PROMPT"
)

if [[ -n "${MID_LAYER:-}" ]]; then
  CMD+=(--mid-layer "$MID_LAYER")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "=== [run] ${CMD[*]} ==="
"${CMD[@]}"

echo "=== [post] Optional upload to Hugging Face Hub ==="
if [[ -n "${HF_TOKEN:-}" && -n "${HF_USER_ID:-}" && -n "${HF_UPLOAD_REPO:-}" ]]; then
  echo "Uploading $SAVE_DIR to hf://$HF_UPLOAD_REPO/${HF_UPLOAD_PATH:-}"
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
  # If target repo doesn't exist, create a dataset repo
  huggingface-cli repo create "$HF_UPLOAD_REPO" --type dataset --yes || true
  if [[ -n "${HF_UPLOAD_PATH:-}" ]]; then
    huggingface-cli upload "$SAVE_DIR" "$HF_UPLOAD_REPO" "$HF_UPLOAD_PATH" --repo-type dataset --quiet || true
  else
    huggingface-cli upload "$SAVE_DIR" "$HF_UPLOAD_REPO" --repo-type dataset --quiet || true
  fi
else
  echo "HF upload skipped (set HF_TOKEN, HF_USER_ID, HF_UPLOAD_REPO to enable)."
fi

echo "=== Done ==="


