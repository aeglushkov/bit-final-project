#!/usr/bin/env bash
# Run preprocess + memory inspectors on the first preprocessed episode.
# *** HUMAN GATE *** — open the produced HTML files in a browser and verify:
#   - bboxes land on objects in the per-frame stamps
#   - reprojection table shows stable (u, v)
#   - no warnings on the memory page
set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "============ 14_openeqa_inspect_first ============"
date

# Pick the first cache dir under openeqa_hm3d/ that has memory.pkl.
FIRST=""
for d in "$OPENEQA_CACHE_ROOT/openeqa_hm3d"/*/; do
    if [ -f "$d/memory.pkl" ]; then
        FIRST="$d"
        break
    fi
done
if [ -z "$FIRST" ]; then
    echo "ERROR: no preprocessed episode found." >&2
    exit 1
fi
echo "Inspecting: $FIRST"

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/inspect_preprocess.py" "$FIRST"

PYTHONPATH="$EVA_EVAL_DIR:$PAPER_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/inspect_memory.py" \
    "$FIRST" --paper-code-dir "$PAPER_DIR"

echo
echo "==> Open these files in a browser:"
echo "    $FIRST/_inspect/preprocess.html"
echo "    $FIRST/_inspect/memory.html"
echo
echo "==> If they look right, next: bash scripts-remote/15_openeqa_run.sh"
date
