"""Render a per-question-type comparison table from VSI-Bench summary.json files.

Use this to see, side-by-side, how the agent and one or more raw-VLM baselines
score on each question type. Inputs are the .summary.json files emitted by
04_run_baseline.py / 03_run_vsibench.py / 10b_authors_baseline.sh — all of
which share the same key shape (overall + per-question-type + n_questions).

Also includes paper Table 1 numbers for InternVL2-8B as a reference column.

Example:
    python compare_baselines.py \\
        --col "agent (ours)"          results/subset_fixed.jsonl.summary.json \\
        --col "raw VLM (lmdeploy AWQ)" results/full_qwen_internvl2.jsonl.summary.json \\
        --col "raw VLM (authors HF)"   results/baseline_authors_internvl2_8b.summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Paper Table 1, InternVL2-8B row (Thinking in Space, CVPR 2025, page 5).
PAPER_INTERNVL2_8B = {
    "overall": 37.5,
    "object_counting": 31.3,
    "object_abs_distance": 29.0,
    "object_size_estimation": 48.9,
    "room_size_estimation": 44.2,
    "object_rel_distance": 38.0,
    "object_rel_direction": 33.4,
    "route_planning": 28.9,
    "obj_appearance_order": 46.4,
}

QUESTION_TYPES = [
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
    "object_rel_distance",
    "object_rel_direction",
    "route_planning",
    "obj_appearance_order",
]


def fmt(v):
    return f"{v:5.1f}" if isinstance(v, (int, float)) else "  -  "


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--col",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="One column: a label and a .summary.json path. Repeat to add columns.",
    )
    ap.add_argument(
        "--no-paper",
        action="store_true",
        help="Omit the paper Table 1 reference column.",
    )
    args = ap.parse_args()

    columns: list[tuple[str, dict]] = []
    for label, path in args.col:
        data = json.loads(Path(path).read_text())
        columns.append((label, data))
    if not args.no_paper:
        columns.append(("paper Table 1", PAPER_INTERNVL2_8B))

    header_labels = [label for label, _ in columns]
    col_width = max(13, max(len(l) for l in header_labels))

    def row(name: str, values: list):
        return "| " + name.ljust(22) + " | " + " | ".join(str(v).rjust(col_width) for v in values) + " |"

    sep = "|" + "-" * 24 + "|" + ("|".join(["-" * (col_width + 2)] * len(columns))) + "|"

    print(row("question_type", header_labels))
    print(sep)

    print(row("overall", [fmt(d.get("overall")) for _, d in columns]))
    for qt in QUESTION_TYPES:
        print(row(qt, [fmt(d.get(qt)) for _, d in columns]))

    print()
    # Note any n_questions mismatch — important for interpreting gaps.
    ns = [(label, d.get("n_questions")) for label, d in columns if "n_questions" in d]
    if ns:
        print("n_questions: " + ", ".join(f"{l}={n}" for l, n in ns))


if __name__ == "__main__":
    main()
