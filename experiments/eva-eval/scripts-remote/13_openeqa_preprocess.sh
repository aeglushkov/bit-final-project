#!/usr/bin/env bash
# Preprocess (download + adapt + build memory + cleanup) for all sampled episodes.
# Requires OPENEQA_BUNDLE_URL_TEMPLATE to be set in your shell.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

if [ -z "${OPENEQA_BUNDLE_URL_TEMPLATE:-}" ]; then
    echo "ERROR: OPENEQA_BUNDLE_URL_TEMPLATE not set." >&2
    echo "       Determine the per-episode tar.gz URL from openeqa README and" >&2
    echo "       export OPENEQA_BUNDLE_URL_TEMPLATE='<url with {episode_id}>'." >&2
    exit 1
fi

echo "============ 13_openeqa_preprocess ============"
date

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/06_preprocess_openeqa.py" \
    --sampled-json "$OPENEQA_SAMPLED_JSON" \
    --cache-root "$OPENEQA_CACHE_ROOT" \
    --paper-code-dir "$PAPER_DIR" \
    --classes-file "$CLASSES_FILE" \
    --bundle-url-template "$OPENEQA_BUNDLE_URL_TEMPLATE"

echo "==> done. Next: bash scripts-remote/14_openeqa_inspect_first.sh"
date
