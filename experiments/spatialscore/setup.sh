#!/bin/bash
# SpatialScore experiment setup for remote server (morgenshtern)
# RTX 3090 (24GB VRAM), CUDA 12.8
#
# Usage: bash setup.sh
# Run from the repository root directory.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Phase 1: Create conda environment ==="
# Don't use environment.yaml directly -- it has pinned system libs
# (libgcc, libstdcxx, etc.) that won't work on a different machine.
conda create -n spatialscore python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate spatialscore

echo "=== Phase 2: Install PyTorch (CUDA 12.4 wheels, compatible with CUDA 12.8) ==="
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo "=== Phase 3: Install remaining dependencies ==="
pip install -r "$REPO_ROOT/experiments/spatialscore/requirements.txt"

echo "=== Phase 5: Download Qwen2.5-VL-3B-Instruct model (~6GB) ==="
mkdir -p ~/models
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir ~/models/Qwen2.5-VL-3B-Instruct

echo "=== Phase 6: Download SpatialScore dataset images ==="
cd "$REPO_ROOT/literature/spatialscore/code"
huggingface-cli download --resume-download --repo-type dataset \
    haoningwu/SpatialScore \
    --local-dir ./ \
    --local-dir-use-symlinks False

if [ -f SpatialScore.zip ]; then
    echo "Unzipping SpatialScore.zip..."
    unzip -o SpatialScore.zip
else
    echo "WARNING: SpatialScore.zip not found. Dataset images may already be extracted."
fi

echo ""
echo "=== Setup complete! ==="
echo "Activate the environment with: conda activate spatialscore"
echo "Next step: run create_subsets.py to create test datasets"
