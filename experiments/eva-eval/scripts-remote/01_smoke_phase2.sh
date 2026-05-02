#!/usr/bin/env bash
# Smoke test: MASt3R-SfM depth+pose for ONE video.
# Usage: ./scripts-remote/launch.sh 01_smoke_phase2 [video_path]
#   default: first line of .video_list.txt
set -e
source "$(dirname "$0")/_env.sh"

VIDEO="${1:-$(head -1 "$VIDEO_LIST" 2>/dev/null)}"
if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
    echo "ERROR: no video. Run 00_video_download first, or pass a path."
    echo "  usage: ./scripts-remote/launch.sh 01_smoke_phase2 /path/to/video.mp4"
    exit 1
fi

echo "============ 01_smoke_phase2 ============"
echo "video: $VIDEO"
date

PYTHONPATH="$MAST3R_PYTHONPATH" \
"$MAST3R_ENV/bin/python" experiments/eva-eval/scripts/01_preprocess.py \
    --video "$VIDEO" \
    --cache-root "$CACHE_ROOT"

echo
echo "==> output:"
SMOKE_ID=$(basename "$VIDEO" .mp4)
ls -la "$CACHE_ROOT/$SMOKE_ID/"
echo "==> SMOKE_ID=$SMOKE_ID  (use this in 02_smoke_phase3)"

echo DONE_SMOKE_PHASE2
date
