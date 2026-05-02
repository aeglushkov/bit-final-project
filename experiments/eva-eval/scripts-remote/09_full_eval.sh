#!/usr/bin/env bash
# Full VSI-Bench sweep (~5,000 questions, ~10-15 GPU hours).
# Requires servers running (03_start_servers).
set -e
source "$(dirname "$0")/_env.sh"

curl -fs http://127.0.0.1:18000/v1/models >/dev/null || { echo "Qwen :18000 not ready"; exit 1; }
curl -fs http://127.0.0.1:18001/v1/models >/dev/null || { echo "InternVL2 :18001 not ready"; exit 1; }

# default name = qwen+internvl2; override with arg
NAME="${1:-qwen_internvl2}"
OUT="$RESULTS_DIR/full_${NAME}.jsonl"

echo "============ 09_full_eval ============"
echo "output: $OUT"
date

PYTHONPATH="$PAPER_DIR" \
"$EVA_ENV/bin/python" experiments/eva-eval/scripts/03_run_vsibench.py \
    --cache-root "$CACHE_ROOT" \
    --paper-code-dir "$PAPER_DIR" \
    --classes-file "$CLASSES_FILE" \
    --output "$OUT"

echo
echo "==> summary:"
cat "${OUT}.summary.json"

echo DONE_FULL_EVAL
date
