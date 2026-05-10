#!/usr/bin/env bash
# Full Phase 2: MASt3R-SfM depth+pose for all 288 videos (~6-10 hours on RTX 3090).
# Resumable — already-cached videos are skipped.
set -e
source "$(dirname "$0")/_env.sh"

[ -f "$VIDEO_LIST" ] || { echo "ERROR: $VIDEO_LIST not found, run 00_video_download first"; exit 1; }

echo "============ 06_full_phase2 ============"
echo "videos: $(wc -l < "$VIDEO_LIST")"
date

PYTHONPATH="$MAST3R_PYTHONPATH" \
"$MAST3R_ENV/bin/python" experiments/eva-eval/scripts/01_preprocess.py \
    --video-list "$VIDEO_LIST" \
    --cache-root "$CACHE_ROOT"

echo
echo "==> done; processed:"
ls "$CACHE_ROOT" | wc -l
echo "==> failures:"
[ -f "$CACHE_ROOT/preprocess_failures.jsonl" ] && wc -l "$CACHE_ROOT/preprocess_failures.jsonl" || echo "0"

echo DONE_FULL_PHASE2
date
