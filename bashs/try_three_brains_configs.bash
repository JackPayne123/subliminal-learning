#!/usr/bin/env bash
# Quick script to try different GPU/cloud configurations

set -euo pipefail

cd "$(dirname "$0")/.."

# Backup original
cp sky/three_brains.yaml sky/three_brains.yaml.backup

echo "=== Trying A10G on AWS (spot - most available) ==="
sed -i.bak 's/accelerators: .*/accelerators: A10G:1/' sky/three_brains.yaml
sed -i.bak 's/cloud: .*/cloud: aws/' sky/three_brains.yaml
sed -i.bak 's/use_spot: .*/use_spot: true/' sky/three_brains.yaml
bash bashs/run_three_brains_sky.bash || echo "Failed, trying next..."

echo "=== Trying RTX3080 on RunPod (on-demand - reliable) ==="
sed -i.bak 's/accelerators: .*/accelerators: RTX3080:1/' sky/three_brains.yaml
sed -i.bak 's/use_spot: .*/use_spot: false/' sky/three_brains.yaml
bash bashs/run_three_brains_sky.bash || echo "Failed, trying next..."

echo "=== Trying A100 on AWS (spot - good performance) ==="
sed -i.bak 's/accelerators: .*/accelerators: A100:1/' sky/three_brains.yaml
sed -i.bak 's/cloud: .*/cloud: aws/' sky/three_brains.yaml
sed -i.bak 's/use_spot: .*/use_spot: true/' sky/three_brains.yaml
bash bashs/run_three_brains_sky.bash || echo "Failed, trying next..."

echo "=== Trying A10G on AWS (on-demand - most reliable) ==="
sed -i.bak 's/use_spot: .*/use_spot: false/' sky/three_brains.yaml
bash bashs/run_three_brains_sky.bash || echo "Failed, trying next..."

# Restore original
mv sky/three_brains.yaml.backup sky/three_brains.yaml
echo "Restored original configuration"
