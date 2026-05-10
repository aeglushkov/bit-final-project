"""Render preprocess.html for a cache directory.

Usage:
    python scripts/inspect_preprocess.py <cache_dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eva_eval.debug.preprocess import render_preprocess_html


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache_dir", type=Path, help="Cache directory containing frames/, depth/, poses.npy, intrinsics.json, meta.json")
    args = ap.parse_args()
    out = render_preprocess_html(args.cache_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
