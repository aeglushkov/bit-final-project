#!/usr/bin/env bash
# Grade predictions with default judge (Qwen2.5-7B). Override with $1.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

JUDGE="${1:-qwen2.5-7b-text}"
PRED="${OPENEQA_RESULTS:-$RESULTS_DIR/openeqa_hm3d_dev50.jsonl}"
GRADED="${PRED%.jsonl}_graded_${JUDGE}.jsonl"

echo "============ 16_openeqa_grade ============"
date
echo "predictions: $PRED"
echo "judge:       $JUDGE"
echo "output:      $GRADED"

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/08_grade_openeqa.py" \
    --predictions "$PRED" \
    --judge "$JUDGE" \
    --output "$GRADED"

echo "==> done. Graded results in $GRADED. Next: bash scripts-remote/17_openeqa_inspect_results.sh"
date
