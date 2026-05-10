"""Render grading inspection HTML for a graded predictions JSONL.

Usage:
    python scripts/inspect_grading.py <graded_jsonl>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eva_eval.debug.grading import render_grading_html


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("graded_jsonl", type=Path)
    args = ap.parse_args()
    out = render_grading_html(args.graded_jsonl)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
