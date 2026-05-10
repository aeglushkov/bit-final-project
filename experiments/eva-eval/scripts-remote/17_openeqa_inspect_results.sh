#!/usr/bin/env bash
# Render the grading inspection HTML for the most recent graded JSONL.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

GRADED=$(ls -t "$RESULTS_DIR"/openeqa_hm3d_dev50_graded_*.jsonl 2>/dev/null | head -1)
if [ -z "$GRADED" ]; then
    echo "ERROR: no graded jsonl found under $RESULTS_DIR" >&2
    exit 1
fi

echo "============ 17_openeqa_inspect_results ============"
date
echo "graded: $GRADED"

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/inspect_grading.py" "$GRADED"

echo
echo "==> Open this file in a browser:"
echo "    ${GRADED%.jsonl}.inspect.html"
echo
echo "==> Decision rule:"
echo "    overall in 30–50 with sensible per-category → agent is OK; bug is VSI-Bench-specific"
echo "    overall <20 or any category at 0% → bug in shared pipeline; use inspect_agent_trace.py to dig"
date
