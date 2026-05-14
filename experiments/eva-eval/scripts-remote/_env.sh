#!/usr/bin/env bash
# Common env + paths for all remote runner scripts. Source this from each step.
# Uses BASH_SOURCE so it resolves correctly regardless of where the outer
# script lives (callers can be in scripts-remote/ or in .conda/).
set -e

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"

# Conda envs (morgen-local)
EVA_ENV="$ROOT/.conda/envs/e-videoagent"
MAST3R_ENV="$ROOT/.conda/envs/mast3r"
VLLM_ENV="$ROOT/.conda/envs/vllm"
LMDEPLOY_ENV="$ROOT/.conda/envs/lmdeploy"

# Remote bf16-split mode: neo serves InternVL2-8B (bf16), reached via SSH tunnel.
# The tunnel maps morgen's localhost:18001 -> neo's loopback:18001 so client
# code stays addressed at http://localhost:18001/v1 regardless of which host is
# actually serving.
NEO_HOST="${NEO_HOST:-185.71.171.86}"
NEO_USER="${NEO_USER:-aleksandr}"
NEO_LMDEPLOY_BIN="${NEO_LMDEPLOY_BIN:-\$HOME/miniconda3/envs/lmdeploy/bin/lmdeploy}"
NEO_HF_HOME="${NEO_HF_HOME:-\$HOME/hf-cache}"
NEO_LOG_DIR="${NEO_LOG_DIR:-\$HOME/logs}"

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
