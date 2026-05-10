#!/usr/bin/env bash
# Stratified-sample 50 HM3D questions from the full openeqa-v0 questions JSON.
# Idempotent: writes sampled_50.json, overwrites if --force.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

LIMIT="${OPENEQA_LIMIT:-50}"
SEED="${OPENEQA_SEED:-42}"

echo "============ 12_openeqa_sample ============"
date

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" - <<PY
import json
from pathlib import Path

from eva_eval.eval.sampler import stratified_indices

questions_path = Path("$OPENEQA_QUESTIONS_JSON")
sampled_path = Path("$OPENEQA_SAMPLED_JSON")

rows = json.loads(questions_path.read_text())
hm3d = [r for r in rows if r.get("episode_history", "").startswith("hm3d-v0/")]
print(f"HM3D questions: {len(hm3d)} / {len(rows)} total")

categories = [r.get("category", "?") for r in hm3d]
idxs = stratified_indices(categories, total=$LIMIT, seed=$SEED)
sampled = [hm3d[i] for i in idxs]

from collections import Counter
print(f"Sampled {len(sampled)} questions, by category:")
for c, k in Counter(r.get("category", "?") for r in sampled).most_common():
    print(f"  {c:30s} {k}")

n_eps = len({r['episode_history'] for r in sampled})
print(f"Unique episodes: {n_eps}")

sampled_path.parent.mkdir(parents=True, exist_ok=True)
sampled_path.write_text(json.dumps(sampled, indent=2))
print(f"wrote {sampled_path}")
PY

echo "==> done. Next: bash scripts-remote/13_openeqa_preprocess.sh"
date
