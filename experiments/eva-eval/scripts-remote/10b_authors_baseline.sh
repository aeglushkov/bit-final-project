#!/usr/bin/env bash
# Authors-protocol VSI-Bench baseline for InternVL2-8B.
# Runs the unmodified literature/thinking-in-space/code/evaluate_all_in_one.sh
# end-to-end (native HF + bf16 + dynamic multi-patch tiling + Frame{i}:<image>
# transport) on a stratified-ish 100-question subsample of the test split, so we
# can compare against:
#   - our lmdeploy-AWQ raw-VLM baseline   (results/full_qwen_internvl2.jsonl: 25.27)
#   - the paper's reported 37.5 overall   (Table 1)
#
# Why this exists: our raw-VLM is 12 points below paper. The two leading
# suspects are (a) AWQ INT4 quantization and (b) frame transport. This script
# isolates both by running the authors' code with the full-precision model and
# their native frame template.
#
# Prerequisite: separate conda env at .conda/envs/vsibench with the authors'
# install steps from literature/thinking-in-space/CLAUDE.md (python 3.10 +
# custom dev transformers from the git submodule, deepspeed, s2wrapper, the
# lmms-eval package itself, and the InternVL2 dynamic_preprocess deps).
#
# Usage: ./scripts-remote/launch.sh 10b_authors_baseline [n_questions]
set -e
source "$(dirname "$0")/_env.sh"

N="${1:-100}"

# Sanity: conda env must exist (we don't auto-install — see header).
VSIBENCH_ENV="${VSIBENCH_ENV:-$ROOT/.conda/envs/vsibench}"
[ -x "$VSIBENCH_ENV/bin/python" ] || {
    echo "ERROR: $VSIBENCH_ENV not found."
    echo "Create it with the install steps in literature/thinking-in-space/CLAUDE.md."
    exit 1
}

PAPER_CODE_DIR="$ROOT/literature/thinking-in-space/code"
[ -d "$PAPER_CODE_DIR" ] || { echo "missing $PAPER_CODE_DIR"; exit 1; }

# Expose VSI-Bench videos at the location the authors' utils.py expects:
#   $HF_HOME/vsibench/<dataset>/<scene>.mp4
# We already downloaded videos to $VIDEO_DIR=$ROOT/data/vsibench-videos via
# 00_video_download.sh. Symlink rather than copy.
TARGET="$HF_HOME/vsibench"
if [ ! -e "$TARGET" ]; then
    echo "==> symlinking $TARGET -> $VIDEO_DIR"
    mkdir -p "$HF_HOME"
    ln -s "$VIDEO_DIR" "$TARGET"
elif [ -L "$TARGET" ]; then
    echo "==> reusing existing symlink $TARGET -> $(readlink "$TARGET")"
else
    echo "WARNING: $TARGET exists and is not a symlink — leaving as-is"
fi

# Sanity: at least one expected video file should be reachable through the path.
sample_video=$(find -L "$TARGET" -name "*.mp4" 2>/dev/null | head -1)
[ -n "$sample_video" ] || {
    echo "ERROR: no .mp4 files under $TARGET — did 00_video_download finish?"
    exit 1
}
echo "==> sample resolves: $sample_video"

OUT_BASE="$RESULTS_DIR/baseline_authors_internvl2_8b"
LOG_DIR="$OUT_BASE.logs"
mkdir -p "$LOG_DIR"

echo "============ 10b_authors_baseline ============"
echo "n_questions: $N  (stratified-ish via LMMS_EVAL_SHUFFLE_DOCS=1)"
echo "model:       OpenGVLab/InternVL2-8B (full precision, bf16, 8 frames)"
echo "log dir:     $LOG_DIR"
date

# Authors' invocation. evaluate_all_in_one.sh accepts --model / --limit and
# routes everything else internally. accelerate launcher with 4 procs is its
# default; we override to 1 to keep this comparable to single-GPU lmdeploy.
cd "$PAPER_CODE_DIR"
export PYTHONPATH="$PAPER_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LMMS_EVAL_SHUFFLE_DOCS=1
export PATH="$VSIBENCH_ENV/bin:$PATH"

bash evaluate_all_in_one.sh \
    --model internvl2_8b_8f \
    --limit "$N" \
    --num_processes 1 \
    --output_path "$LOG_DIR"

echo
echo "==> looking for results json under $LOG_DIR ..."
results_json=$(find "$LOG_DIR" -name "results.json" -o -name "*results*.json" 2>/dev/null | head -1)
[ -n "$results_json" ] || { echo "no results.json found"; exit 1; }
echo "==> $results_json"

# Convert authors' nested results.json -> our summary.json shape so
# scripts/04_run_baseline.py and downstream comparison tooling can ingest it.
"$VSIBENCH_ENV/bin/python" - <<PY
import json, sys
from pathlib import Path

res_path = Path("$results_json")
out = res_path.parent.parent / "baseline_authors_internvl2_8b.summary.json"
data = json.loads(res_path.read_text())

# lmms-eval nests results under data["results"]["vsibench"]; vsibench_score
# is a dict mirroring vsibench_aggregate_results' output.
vsi = data["results"]["vsibench"]
summary = {"overall": vsi.get("overall,none", vsi.get("overall"))}
for k, v in vsi.items():
    # keys look like "object_counting_MRA:.5:.95:.05,none"
    key = k.split(",")[0]
    if key in ("overall", "alias"):
        continue
    # collapse to the question_type basename for direct comparison
    for qt in ("object_counting","object_abs_distance","object_size_estimation","room_size_estimation",
               "object_rel_distance","object_rel_direction","route_planning","obj_appearance_order"):
        if key.startswith(qt):
            summary[qt] = v
summary["n_questions"] = $N
out.write_text(json.dumps(summary, indent=2))
print("==> wrote", out)
print(json.dumps(summary, indent=2))
PY

echo DONE_AUTHORS_BASELINE
date
