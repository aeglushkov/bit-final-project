"""Render memory.html for a cache directory.

Usage:
    python scripts/inspect_memory.py <cache_dir> --paper-code-dir <path>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eva_eval.debug.memory import render_memory_html


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--paper-code-dir", type=Path, required=True)
    ap.add_argument("--frame-stride", type=int, default=10)
    args = ap.parse_args()
    out = render_memory_html(args.cache_dir, args.paper_code_dir, frame_stride=args.frame_stride)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
