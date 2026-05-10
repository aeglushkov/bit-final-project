"""Phase: run agent over sampled OpenEQA questions.

Usage:
    python scripts/07_run_openeqa.py \
        --sampled-json <cache_root>/openeqa_hm3d/sampled_50.json \
        --cache-root <cache_root> \
        --paper-code-dir <path/to/literature/EmbodiedVideoAgent/code> \
        --classes-file <path/to/detection_classes.txt> \
        --output results/openeqa_hm3d_dev50.jsonl \
        [--planner NAME] [--resume] [--no-capture-trace]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eva_eval.eval.openeqa import run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampled-json", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--paper-code-dir", type=Path, required=True)
    ap.add_argument("--classes-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--planner", default=None)
    ap.add_argument("--max-iterations", type=int, default=30)
    ap.add_argument("--no-capture-trace", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    summary = run(
        sampled_json=args.sampled_json,
        cache_root=args.cache_root,
        paper_code_dir=args.paper_code_dir,
        classes_file=args.classes_file,
        output=args.output,
        planner=args.planner,
        max_iterations=args.max_iterations,
        capture_trace=not args.no_capture_trace,
        resume=args.resume,
    )
    print(f"\n{summary['n_done']} / {summary['n_total']} questions answered. Output: {args.output}")


if __name__ == "__main__":
    main()
