#!/usr/bin/env bash
# Raw-VLM baseline (no agent). Defaults: 200 questions, 8 frames per Q,
# InternVL2-8B over lmdeploy. Requires lmdeploy server up (03_start_servers).
# Usage: ./scripts-remote/launch.sh 10_baseline [n_questions] [vlm] [n_frames]
set -e
source "$(dirname "$0")/_env.sh"

N="${1:-200}"
VLM="${2:-internvl2-8b}"
NFR="${3:-8}"

OUT="$RESULTS_DIR/baseline_${VLM}.jsonl"

# Sanity: VLM must be reachable (TIS uses 8 frames for InternVL2-8B by default).
case "$VLM" in
    internvl2-8b|internvl2.5-8b) PORT=18001 ;;
    qwen2.5-vl-7b)               PORT=18000 ;;
    *)                           PORT=18001 ;;
esac
curl -fs http://127.0.0.1:$PORT/v1/models >/dev/null || {
    echo "VLM $VLM on :$PORT not reachable, run 03_start_servers"; exit 1;
}

echo "============ 10_baseline ============"
echo "n_questions: $N  vlm: $VLM  n_frames: $NFR"
echo "output: $OUT"
date

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" experiments/eva-eval/scripts/04_run_baseline.py \
    --cache-root "$CACHE_ROOT" \
    --output "$OUT" \
    --vlm "$VLM" \
    --n-frames "$NFR" \
    --limit "$N"

echo
echo "==> summary:"
cat "${OUT}.summary.json"
echo
echo "==> error rate:"
"$EVA_ENV/bin/python" - <<PY
import json
with open("$OUT") as f:
    rows = [json.loads(l) for l in f if l.strip()]
errs = [r for r in rows if r.get("error")]
print(f"  {len(errs)} errors / {len(rows)} questions")
for r in errs[:3]:
    print(f"  id={r['id']}: {r['error'].splitlines()[0][:200]}")
PY

echo DONE_BASELINE
date
