#!/usr/bin/env bash
# Run agent over the sampled OpenEQA questions. Requires servers running.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

curl -fs http://127.0.0.1:18000/v1/models >/dev/null || { echo "Qwen :18000 not ready, run 03_start_servers"; exit 1; }
curl -fs http://127.0.0.1:18001/v1/models >/dev/null || { echo "InternVL2 :18001 not ready, run 03_start_servers"; exit 1; }

OUT="${OPENEQA_RESULTS:-$RESULTS_DIR/openeqa_hm3d_dev50.jsonl}"

echo "============ 15_openeqa_run ============"
date

PYTHONPATH="$EVA_EVAL_DIR:$PAPER_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/07_run_openeqa.py" \
    --sampled-json "$OPENEQA_SAMPLED_JSON" \
    --cache-root "$OPENEQA_CACHE_ROOT" \
    --paper-code-dir "$PAPER_DIR" \
    --classes-file "$CLASSES_FILE" \
    --output "$OUT" \
    --resume

echo "==> done. Predictions in $OUT. Next: bash scripts-remote/16_openeqa_grade.sh"
date
