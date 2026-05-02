#!/usr/bin/env bash
# Common env + paths for all remote runner scripts. Source this from each step.
set -e

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"

# Conda envs
EVA_ENV="$ROOT/.conda/envs/e-videoagent"
MAST3R_ENV="$ROOT/.conda/envs/mast3r"
VLLM_ENV="$ROOT/.conda/envs/vllm"
LMDEPLOY_ENV="$ROOT/.conda/envs/lmdeploy"

# Paper code
PAPER_DIR="$ROOT/literature/EmbodiedVideoAgent/code"

# eva-eval source root (so any env can `import eva_eval` via PYTHONPATH)
EVA_EVAL_DIR="$ROOT/experiments/eva-eval"

# MASt3R + dependencies
MAST3R_DIR="$ROOT/.third-party/mast3r-sfm"
MAST3R_PYTHONPATH="$MAST3R_DIR:$MAST3R_DIR/dust3r:$EVA_EVAL_DIR"

# Caches
export HF_HOME="$ROOT/.hf-cache"
export TORCH_HOME="$ROOT/.torch-cache"
export HF_HUB_DISABLE_TELEMETRY=1

# Output dirs
CACHE_ROOT="$ROOT/cache/vsibench"
RESULTS_DIR="$ROOT/results"
VIDEO_DIR="$ROOT/data/vsibench-videos"
VIDEO_LIST="$ROOT/.video_list.txt"
CLASSES_FILE="$ROOT/experiments/eva-eval/config/detection_classes.txt"

mkdir -p "$CACHE_ROOT" "$RESULTS_DIR" "$VIDEO_DIR"
