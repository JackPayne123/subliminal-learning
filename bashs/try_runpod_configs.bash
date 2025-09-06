#!/usr/bin/env bash
# Try different RunPod GPU configurations

set -euo pipefail

cd "$(dirname "$0")/.."

# Backup original
cp sky/three_brains.yaml sky/three_brains.yaml.backup

echo "=== Trying T4 on RunPod (very common) ==="
sed -i.bak 's/accelerators: .*/accelerators: T4:1/' sky/three_brains.yaml
sed -i.bak 's/cloud: .*/cloud: runpod/' sky/three_brains.yaml
sed -i.bak 's/use_spot: .*/use_spot: true/' sky/three_brains.yaml
bash bashs/run_three_brains_sky.bash || echo "Failed, trying next..."

echo "=== Trying L4 on RunPod (good performance) ==="
sed -i.bak 's/accelerators: .*/accelerators: L4:1/' sky/three_brains.yaml
bash bashs/run_three_brains_sky.bash || echo "Failed, trying next..."

echo "=== Trying V100 on RunPod (widely available) ==="
sed -i.bak 's/accelerators: .*/accelerators: V100:1/' sky/three_brains.yaml
bash bashs/run_three_brains_sky.bash || echo "Failed, trying next..."

echo "=== Trying RTX3080 on-demand (if spot unavailable) ==="
sed -i.bak 's/use_spot: .*/use_spot: false/' sky/three_brains.yaml
bash bashs/run_three_brains_sky.bash || echo "Failed, trying next..."

echo "=== Trying RTX3070 on-demand ==="
sed -i.bak 's/accelerators: .*/accelerators: RTX3070:1/' sky/three_brains.yaml
bash bashs/run_three_brains_sky.bash || echo "Failed, trying next..."

# Restore original
mv sky/three_brains.yaml.backup sky/three_brains.yaml
echo "Restored original configuration"
