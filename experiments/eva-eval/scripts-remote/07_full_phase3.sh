#!/usr/bin/env bash
# Full Phase 3: build object memory for every Phase 2 cache (~3-5 hours).
# Resumable.
set -e
source "$(dirname "$0")/_env.sh"

[ -f "$CLASSES_FILE" ] || { echo "ERROR: missing $CLASSES_FILE"; exit 1; }

echo "============ 07_full_phase3 ============"
echo "scenes to process: $(ls "$CACHE_ROOT" | wc -l)"
date

PYTHONPATH="$PAPER_DIR" \
"$EVA_ENV/bin/python" experiments/eva-eval/scripts/02_build_memory.py \
    --cache-root "$CACHE_ROOT" \
    --classes-file "$CLASSES_FILE" \
    --paper-code-dir "$PAPER_DIR"

echo
echo "==> memories built:"
find "$CACHE_ROOT" -name memory.pkl | wc -l

echo DONE_FULL_PHASE3
date
