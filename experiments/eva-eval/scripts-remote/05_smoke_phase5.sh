#!/usr/bin/env bash
# Smoke test: run up to N questions through the full agent (Phase 5).
# Requires servers running (see 03_start_servers).
# Usage: ./scripts-remote/launch.sh 05_smoke_phase5 [n_questions]
set -e
source "$(dirname "$0")/_env.sh"

N="${1:-20}"

# sanity: servers up?
curl -fs http://127.0.0.1:18000/v1/models >/dev/null || { echo "Qwen :18000 not ready, run 03_start_servers"; exit 1; }
curl -fs http://127.0.0.1:18001/v1/models >/dev/null || { echo "InternVL2 :18001 not ready, run 03_start_servers"; exit 1; }
[ -f "$CLASSES_FILE" ] || { echo "ERROR: missing $CLASSES_FILE"; exit 1; }

OUT="$RESULTS_DIR/smoke.jsonl"

echo "============ 05_smoke_phase5 ============"
echo "n_questions: $N"
echo "output: $OUT"
date

PYTHONPATH="$PAPER_DIR" \
"$EVA_ENV/bin/python" experiments/eva-eval/scripts/03_run_vsibench.py \
    --cache-root "$CACHE_ROOT" \
    --paper-code-dir "$PAPER_DIR" \
    --classes-file "$CLASSES_FILE" \
    --output "$OUT" \
    --limit "$N" \
    --no-stratified

echo
echo "==> summary:"
cat "${OUT}.summary.json"
echo
echo "==> error rate:"
grep -c "\"error\":" "$OUT" 2>/dev/null || echo "0"
echo " of $(wc -l < "$OUT")"

echo DONE_SMOKE_PHASE5
date
