#!/usr/bin/env bash
# Render RGB+depth+pose tuples for the sampled OpenEQA episodes via Habitat-sim.
# Selectively renders ONLY the episodes referenced by sampled_50.json by
# moving non-sampled folders aside before extraction (extract-frames.py
# iterates all subdirs unconditionally).
set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "============ 13c_extract_frames ============"
date

FRAMES_DIR="$OPENEQA_REPO_DIR/data/frames/hm3d-v0"
HOLD_DIR="$OPENEQA_REPO_DIR/data/frames/_hold"

if [ ! -d "$FRAMES_DIR" ]; then
    echo "ERROR: $FRAMES_DIR does not exist. Run 13b_download_hm3d.sh first." >&2
    exit 1
fi

# Build the allowed-episode set from sampled_50.json
ALLOW_FILE=$(mktemp)
trap 'rm -f "$ALLOW_FILE"' EXIT

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" - <<PY > "$ALLOW_FILE"
import json
rows = json.loads(open("$OPENEQA_SAMPLED_JSON").read())
ids = sorted({r["episode_history"].split("/", 1)[1] for r in rows})
for i in ids:
    print(i)
PY
N_ALLOW=$(wc -l < "$ALLOW_FILE")
echo "Sampled episode count: $N_ALLOW"

# Move non-sampled folders into _hold/ so extract-frames.py renders only the
# ones we care about. Restore at the end no matter what.
mkdir -p "$HOLD_DIR"
restore_held() {
    if [ -d "$HOLD_DIR" ]; then
        echo "Restoring held episode folders..."
        for d in "$HOLD_DIR"/*; do
            [ -e "$d" ] || continue
            mv "$d" "$FRAMES_DIR/"
        done
        rmdir "$HOLD_DIR" 2>/dev/null || true
    fi
}
trap 'restore_held; rm -f "$ALLOW_FILE"' EXIT

held=0
for d in "$FRAMES_DIR"/*; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    if ! grep -qxF "$name" "$ALLOW_FILE"; then
        mv "$d" "$HOLD_DIR/"
        held=$((held + 1))
    fi
done
echo "Held $held non-sampled folders out of the way"

# Run the extractor (cwd must be the openeqa repo so data/hm3d/config.py resolves)
cd "$OPENEQA_REPO_DIR"
"$OPENEQA_HABITAT_ENV/bin/python" data/hm3d/extract-frames.py \
    --hm3d-root "$OPENEQA_HM3D_SCENES_DIR/hm3d/val" \
    --output-directory "$FRAMES_DIR"

# trap will restore the held folders on exit
echo "==> done. Next: bash scripts-remote/13_openeqa_preprocess.sh"
date
