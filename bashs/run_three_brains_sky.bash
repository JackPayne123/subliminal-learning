#!/usr/bin/env bash
set -euo pipefail

# Ensure we run from repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Ensure SkyPilot is installed; prefer uv tool install, with RunPod extras if desired
if ! command -v sky >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv tool install "skypilot[runpod]"
  else
    python3 -m pip install -U "skypilot[runpod]"
  fi
fi

# Cluster name and YAML
CLUSTER_NAME=${CLUSTER_NAME:-threebrains}
YAML_PATH=${YAML_PATH:-sky/three_brains.yaml}

# For infra overrides, modify the YAML file directly or use sky launch options
# The YAML is now configured for RTX4090 on RunPod by default

# Forward only the essential environment variables
# SkyPilot needs these available in its environment when it runs
ESSENTIAL_VARS=(HF_TOKEN OPENAI_API_KEY HF_USER_ID RUNPOD_API_KEY RUN_B0_PIPELINE)

echo "Setting essential environment variables for SkyPilot..."
for v in "${ESSENTIAL_VARS[@]}"; do
  if [ -n "${!v-}" ]; then
    export "$v"  # Make sure it's exported to SkyPilot's environment
    echo "Set $v"
  else
    echo "Warning: $v not set in environment"
  fi
done

# Set default for RUN_B0_PIPELINE if not specified
export RUN_B0_PIPELINE=${RUN_B0_PIPELINE:-0}
echo "RUN_B0_PIPELINE set to: $RUN_B0_PIPELINE"

echo "Launching SkyPilot job: cluster=$CLUSTER_NAME yaml=$YAML_PATH"
# Launch without explicit --env flags since we're using the YAML file's envs section
sky launch -c "$CLUSTER_NAME" "$YAML_PATH"