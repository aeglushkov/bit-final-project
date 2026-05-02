#!/usr/bin/env bash
# Smoke test: build object memory for ONE video's Phase 2 cache.
# Usage: ./scripts-remote/launch.sh 02_smoke_phase3 [scene_id]
#   default: first subdir under cache/vsibench/
set -e
source "$(dirname "$0")/_env.sh"

SCENE_ID="${1:-$(find "$CACHE_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | sort | head -1)}"
SCENE_DIR="$CACHE_ROOT/$SCENE_ID"
if [ -z "$SCENE_ID" ] || [ ! -f "$SCENE_DIR/meta.json" ]; then
    echo "ERROR: no Phase 2 cache. Run 01_smoke_phase2 first."
    echo "  scene_id=$SCENE_ID  scene_dir=$SCENE_DIR"
    exit 1
fi
[ -f "$CLASSES_FILE" ] || { echo "ERROR: missing $CLASSES_FILE; run scripts/build_vocabulary.py"; exit 1; }

echo "============ 02_smoke_phase3 ============"
echo "scene: $SCENE_ID"
date

PYTHONPATH="$PAPER_DIR" \
"$EVA_ENV/bin/python" experiments/eva-eval/scripts/02_build_memory.py \
    --video-cache "$SCENE_DIR" \
    --classes-file "$CLASSES_FILE" \
    --paper-code-dir "$PAPER_DIR"

echo
echo "==> sanity check:"
"$EVA_ENV/bin/python" - <<PY
from eva_eval.memory.store import load_memory
m = load_memory("$SCENE_DIR/memory.pkl")
print("static objects:", len(m["static_objects"]))
print("dynamic       :", len(m["dynamic_objects"]))
print("frames        :", len(m["frames"]))
print("first 5 categories:", [o.category for o in m["static_objects"][:5]])
PY

echo DONE_SMOKE_PHASE3
date
