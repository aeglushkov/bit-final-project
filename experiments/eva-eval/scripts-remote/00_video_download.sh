#!/usr/bin/env bash
# Download VSI-Bench videos: pull 3 zips from HF, extract into data/vsibench-videos/
set -e
source "$(dirname "$0")/_env.sh"

echo "============ 00_video_download ============"
date

"$EVA_ENV/bin/python" <<'PY'
from huggingface_hub import hf_hub_download
for fname in ("arkitscenes.zip", "scannet.zip", "scannetpp.zip", "test.jsonl"):
    print(f"==> {fname}")
    p = hf_hub_download(
        repo_id="nyu-visionx/VSI-Bench",
        filename=fname,
        repo_type="dataset",
    )
    print("   cached:", p)
PY

for fname in arkitscenes.zip scannet.zip scannetpp.zip; do
    src=$(find "$HF_HOME/hub" -name "$fname" -type f | head -1)
    if [ -z "$src" ]; then
        echo "ERROR: $fname not found in cache"; exit 1
    fi
    name="${fname%.zip}"
    target="$VIDEO_DIR/$name"
    if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null | head -1)" ]; then
        echo "[skip] $name already extracted"
        continue
    fi
    echo "==> extracting $fname -> $target"
    mkdir -p "$target"
    unzip -q -o "$src" -d "$target"
    echo "   $(find "$target" -name '*.mp4' | wc -l) videos"
done

find "$VIDEO_DIR" -name "*.mp4" | sort > "$VIDEO_LIST"
echo "==> $(wc -l < "$VIDEO_LIST") videos -> $VIDEO_LIST"
echo "==> total size:"
du -sh "$VIDEO_DIR"

echo DONE_VIDEO_DOWNLOAD
date
