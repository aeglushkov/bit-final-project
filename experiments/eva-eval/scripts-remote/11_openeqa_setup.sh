#!/usr/bin/env bash
# Clone openeqa repo and copy the questions JSON. The episode bundle is
# downloaded later via scripts-remote/13b_download_hm3d.sh and rendered with
# 13c_extract_frames.sh — there is no per-episode tar.gz.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "============ 11_openeqa_setup ============"
date

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/05_download_openeqa.py" \
    --openeqa-repo-dir "$OPENEQA_REPO_DIR" \
    --out-questions-json "$OPENEQA_QUESTIONS_JSON" \
    --no-bundle-check

echo "==> done. Next: bash scripts-remote/12_openeqa_sample.sh"
date
