#!/bin/bash
# SpatialScore experiment setup for remote server (morgenshtern)
# RTX 3090 (24GB VRAM), CUDA 12.8
#
# Usage: bash setup.sh
# Run from the repository root directory.

set -e

echo "=== Phase 1: Create conda environment ==="
# Don't use environment.yaml directly -- it has pinned system libs
# (libgcc, libstdcxx, etc.) that won't work on a different machine.
conda create -n spatialscore python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate spatialscore

echo "=== Phase 2: Install PyTorch (CUDA 12.4 wheels, compatible with CUDA 12.8) ==="
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo "=== Phase 3: Install flash-attn ==="
# If this fails, change attn_implementation="flash_attention_2" to "eager"
# in literature/spatialscore/code/test_qwen.py line 24
pip install flash-attn --no-build-isolation || {
    echo "WARNING: flash-attn failed to install."
    echo "You'll need to change attn_implementation to 'eager' in test_qwen.py line 24."
    echo "Continuing with remaining dependencies..."
}

echo "=== Phase 4: Install remaining dependencies ==="
pip install transformers==4.51.3
pip install accelerate==1.5.2
pip install qwen-vl-utils==0.0.10
pip install Pillow tqdm

echo "=== Phase 5: Download Qwen2.5-VL-3B-Instruct model (~6GB) ==="
mkdir -p ~/models
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir ~/models/Qwen2.5-VL-3B-Instruct

echo "=== Phase 6: Download SpatialScore dataset images ==="
cd literature/spatialscore/code
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
