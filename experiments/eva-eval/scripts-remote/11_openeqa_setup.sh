#!/usr/bin/env bash
# Clone openeqa repo and copy the questions JSON. Optionally HEAD-check the bundle URL.
# Sets OPENEQA_BUNDLE_URL_TEMPLATE in your shell first if you want the URL check.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "============ 11_openeqa_setup ============"
date

EXTRA=()
if [ -n "${OPENEQA_BUNDLE_URL_TEMPLATE:-}" ]; then
    # Use the first episode's URL as a sentinel for bundle reachability.
    SENTINEL_URL="${OPENEQA_BUNDLE_URL_TEMPLATE//\{episode_id\}/sentinel}"
    EXTRA=(--bundle-url "$SENTINEL_URL")
fi

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/05_download_openeqa.py" \
    --openeqa-repo-dir "$OPENEQA_REPO_DIR" \
    --out-questions-json "$OPENEQA_QUESTIONS_JSON" \
    "${EXTRA[@]:-}"

echo "==> done. Next: bash scripts-remote/12_openeqa_sample.sh"
date
