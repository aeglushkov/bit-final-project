#!/usr/bin/env bash
# 100-question agent eval against the SenseNova-SI split stack, pinned to the
# same 100 IDs as the prior bf16 agent run (results/subset_bf16.ids.txt).
# - --ids-file (not --limit) so the sample matches the May-13 bf16 agent run
#   exactly; the new run is directly diffable against subset_bf16.jsonl and
#   the bf16 raw-VLM baseline.
# - Picks SI models via EVA_PLANNER / EVA_VLM (vanilla InternVL3 as planner,
#   SenseNova-SI-1.5-InternVL3-8B as VLM).
#
# Requires 03_start_servers_si to have been run (InternVL3 on :18000 local,
# SenseNova-SI on :18001 via SSH tunnel).
set -e
source "$(dirname "$0")/_env.sh"

curl -fs http://127.0.0.1:18000/v1/models >/dev/null || { echo "Planner :18000 not ready, run 03_start_servers_si"; exit 1; }
curl -fs http://127.0.0.1:18001/v1/models >/dev/null || { echo "VLM :18001 not ready (tunnel down?), run 03_start_servers_si"; exit 1; }

IDS_FILE="${IDS_FILE:-$RESULTS_DIR/subset_bf16.ids.txt}"
OUT="$RESULTS_DIR/subset_si.jsonl"

[ -f "$IDS_FILE" ] || { echo "ERROR: $IDS_FILE not found — pull or regenerate the canonical 100-ID file first"; exit 1; }

export EVA_PLANNER="${EVA_PLANNER:-internvl3-8b-text-bf16}"
export EVA_VLM="${EVA_VLM:-sensenova-si-1.5-internvl3-8b-bf16}"

echo "============ 08_subset100_si ============"
echo "planner:  $EVA_PLANNER  vlm: $EVA_VLM"
echo "ids-file: $IDS_FILE"
echo "output:   $OUT"
date

PYTHONPATH="$PAPER_DIR" \
"$EVA_ENV/bin/python" experiments/eva-eval/scripts/03_run_vsibench.py \
    --cache-root "$CACHE_ROOT" \
    --paper-code-dir "$PAPER_DIR" \
    --classes-file "$CLASSES_FILE" \
    --output "$OUT" \
    --ids-file "$IDS_FILE"

echo
echo "==> summary:"
cat "${OUT}.summary.json"

echo DONE_SUBSET100_SI
date
