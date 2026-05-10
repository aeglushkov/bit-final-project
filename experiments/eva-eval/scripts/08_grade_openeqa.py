"""Phase: grade predictions JSONL with an LLM-as-judge.

Usage:
    python scripts/08_grade_openeqa.py \
        --predictions results/openeqa_hm3d_dev50.jsonl \
        --judge qwen2.5-7b-text \
        --output results/openeqa_hm3d_dev50_graded.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eva_eval.eval.openeqa_grade import aggregate, grade_one
from eva_eval.llm.client import load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True, help="JSONL produced by 07_run_openeqa.py")
    ap.add_argument("--output", type=Path, required=True, help="Graded JSONL output")
    ap.add_argument("--judge", default="qwen2.5-7b-text", help="Model name from config/models.yaml")
    args = ap.parse_args()

    judge = load_model(args.judge)
    print(f"Using judge: {args.judge} ({judge.model})")

    rows_in = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    print(f"Grading {len(rows_in)} predictions")

    graded: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in rows_in:
            try:
                score, rationale = grade_one(
                    judge,
                    question=r["question"],
                    gold_answer=r["ground_truth"],
                    prediction=r["prediction"],
                )
            except Exception as e:
                score, rationale = None, f"(judge error: {type(e).__name__}: {e})"
                print(f"[err] {r.get('id')}: {rationale}", file=sys.stderr)
            out = dict(r)
            out["score"] = score
            out["judge_rationale"] = rationale
            graded.append(out)
            f.write(json.dumps(out, default=str) + "\n")
            f.flush()

    summary = aggregate(graded)
    print("\n=== Summary ===")
    print(f"  overall:        {summary['overall']:7.2f}")
    print(f"  n_questions:    {summary['n_questions']}")
    for cat, sc in summary["per_category"].items():
        print(f"  {cat:30s} {sc:7.2f}")

    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(dict(summary), indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
