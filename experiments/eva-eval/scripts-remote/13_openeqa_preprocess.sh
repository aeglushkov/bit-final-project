#!/usr/bin/env bash
# Adapt + build memory for all sampled episodes from $OPENEQA_EXTRACTED_ROOT
# (the output dir of openeqa's extract-frames.py — see scripts-remote/13b_extract_frames.sh).
set -euo pipefail
source "$(dirname "$0")/_env.sh"

if [ ! -d "${OPENEQA_EXTRACTED_ROOT:-}" ]; then
    echo "ERROR: OPENEQA_EXTRACTED_ROOT='${OPENEQA_EXTRACTED_ROOT:-}' is not a directory." >&2
    echo "       Run scripts-remote/13b_extract_frames.sh first to render HM3D episodes." >&2
    exit 1
fi

echo "============ 13_openeqa_preprocess ============"
date

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/06_preprocess_openeqa.py" \
    --sampled-json "$OPENEQA_SAMPLED_JSON" \
    --extracted-root "$OPENEQA_EXTRACTED_ROOT" \
    --cache-root "$OPENEQA_CACHE_ROOT" \
    --paper-code-dir "$PAPER_DIR" \
    --classes-file "$CLASSES_FILE"

echo "==> done. Next: bash scripts-remote/14_openeqa_inspect_first.sh"
date
