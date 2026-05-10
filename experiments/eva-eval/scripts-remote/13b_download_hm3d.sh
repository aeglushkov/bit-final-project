#!/usr/bin/env bash
# Download HM3D val scenes (~12 GB, requires Matterport API credentials) plus
# OpenEQA's agent-state bundle (~small, drives extract-frames.py).
#
# Reads HM3D_TOKEN_ID / HM3D_TOKEN_SECRET from project's .env (loaded by _env.sh).
# Register at https://matterport.com/habitat-matterport-3d-research-dataset
set -euo pipefail
source "$(dirname "$0")/_env.sh"

if [ -z "${HM3D_TOKEN_ID:-}" ] || [ -z "${HM3D_TOKEN_SECRET:-}" ]; then
    echo "ERROR: HM3D_TOKEN_ID / HM3D_TOKEN_SECRET not set." >&2
    echo "       Add them to $ROOT/.env (see scripts-remote/_env.sh for the loader)." >&2
    exit 1
fi

echo "============ 13b_download_hm3d ============"
date

mkdir -p "$OPENEQA_HM3D_SCENES_DIR"

# 1. HM3D val scenes via habitat-sim's downloader
SCENES_TARGET="$OPENEQA_HM3D_SCENES_DIR/hm3d/val"
if [ -d "$SCENES_TARGET" ] && compgen -G "$SCENES_TARGET/*/*.basis.glb" > /dev/null; then
    echo "[skip] HM3D val already present at $SCENES_TARGET"
else
    echo "[download] hm3d_val_v0.2 -> $OPENEQA_HM3D_SCENES_DIR"
    "$OPENEQA_HABITAT_ENV/bin/python" -m habitat_sim.utils.datasets_download \
        --username "$HM3D_TOKEN_ID" \
        --password "$HM3D_TOKEN_SECRET" \
        --uids hm3d_val_v0.2 \
        --data-path "$OPENEQA_HM3D_SCENES_DIR"
fi

# 2. OpenEQA agent-state bundle (paired with extract-frames.py)
STATES_TGZ="$OPENEQA_REPO_DIR/data/open-eqa-hm3d-states-v0.tgz"
STATES_FRAMES_DIR="$OPENEQA_REPO_DIR/data/frames/hm3d-v0"
if [ -d "$STATES_FRAMES_DIR" ] && [ -n "$(ls -A "$STATES_FRAMES_DIR" 2>/dev/null)" ]; then
    echo "[skip] OpenEQA states already extracted at $STATES_FRAMES_DIR"
else
    echo "[download] open-eqa-hm3d-states-v0.tgz"
    mkdir -p "$OPENEQA_REPO_DIR/data/frames"
    wget -q --show-progress -O "$STATES_TGZ" \
        "https://www.dropbox.com/scl/fi/wg1uj1gvr4tkcz9aq3tzb/open-eqa-hm3d-states-v0.tgz?rlkey=i69chnpib8ui4cfabxa3iy9oj&dl=1"
    echo "[extract] -> $OPENEQA_REPO_DIR/data/frames"
    tar -xzf "$STATES_TGZ" -C "$OPENEQA_REPO_DIR/data/frames"
    rm "$STATES_TGZ"
fi

echo "==> done. Next: bash scripts-remote/13c_extract_frames.sh"
date
