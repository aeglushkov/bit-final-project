"""Render markdown trace for one question from a predictions JSONL.

Usage:
    python scripts/inspect_agent_trace.py <jsonl> <question_id> [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eva_eval.debug.agent_trace import find_question, render_trace_markdown


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("question_id")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path; defaults to <jsonl_dir>/_inspect/trace_<id>.md")
    args = ap.parse_args()

    row = find_question(args.jsonl, args.question_id)
    md = render_trace_markdown(row)
    out = args.out or args.jsonl.parent / "_inspect" / f"trace_{args.question_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
