#!/usr/bin/env bash
# 100-question stratified agent eval against the bf16 split stack.
# - Same sampler/seed as subset_fixed.jsonl, so the 100 IDs match the AWQ
#   agent run we already have (results/subset_fixed.jsonl, 30.68 overall).
# - Picks bf16 models via EVA_PLANNER / EVA_VLM, which the agent's
#   _build_planner_llm and load_default_vlm both honor.
#
# Requires 03_start_servers_bf16 to have been run (Qwen on :18000 local,
# InternVL2 on :18001 via SSH tunnel).
set -e
source "$(dirname "$0")/_env.sh"

curl -fs http://127.0.0.1:18000/v1/models >/dev/null || { echo "Qwen :18000 not ready, run 03_start_servers_bf16"; exit 1; }
curl -fs http://127.0.0.1:18001/v1/models >/dev/null || { echo "InternVL2 :18001 not ready (tunnel down?), run 03_start_servers_bf16"; exit 1; }

OUT="$RESULTS_DIR/subset_bf16.jsonl"

export EVA_PLANNER="${EVA_PLANNER:-qwen2.5-7b-text-bf16}"
export EVA_VLM="${EVA_VLM:-internvl2-8b-bf16}"

echo "============ 08_subset100_bf16 ============"
echo "planner: $EVA_PLANNER  vlm: $EVA_VLM"
echo "output:  $OUT"
date

PYTHONPATH="$PAPER_DIR" \
"$EVA_ENV/bin/python" experiments/eva-eval/scripts/03_run_vsibench.py \
    --cache-root "$CACHE_ROOT" \
    --paper-code-dir "$PAPER_DIR" \
    --classes-file "$CLASSES_FILE" \
    --output "$OUT" \
    --limit 100

echo
echo "==> summary:"
cat "${OUT}.summary.json"

echo DONE_SUBSET100_BF16
date
