#!/usr/bin/env bash
# 500-question stratified eval. Sanity-gate before the full sweep.
# Requires servers running (03_start_servers).
set -e
source "$(dirname "$0")/_env.sh"

curl -fs http://127.0.0.1:18000/v1/models >/dev/null || { echo "Qwen :18000 not ready, run 03_start_servers"; exit 1; }
curl -fs http://127.0.0.1:18001/v1/models >/dev/null || { echo "InternVL2 :18001 not ready, run 03_start_servers"; exit 1; }

OUT="$RESULTS_DIR/dev500.jsonl"

echo "============ 08_dev500 ============"
date

PYTHONPATH="$PAPER_DIR" \
"$EVA_ENV/bin/python" experiments/eva-eval/scripts/03_run_vsibench.py \
    --cache-root "$CACHE_ROOT" \
    --paper-code-dir "$PAPER_DIR" \
    --classes-file "$CLASSES_FILE" \
    --output "$OUT" \
    --limit 500

echo
echo "==> summary:"
cat "${OUT}.summary.json"

echo DONE_DEV500
date
