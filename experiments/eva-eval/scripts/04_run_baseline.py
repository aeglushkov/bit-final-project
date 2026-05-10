"""Phase 5 baseline: raw VLM on N sampled frames per question, no agent."""
from __future__ import annotations

import argparse
from pathlib import Path

from eva_eval.eval.baseline import run


def main():
    ap = argparse.ArgumentParser(description="Raw-VLM baseline on VSI-Bench (no agent).")
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vlm", default="internvl2-8b")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16, help="VLM max_new_tokens (TIS uses 16)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-stratified", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--all-scenes",
        action="store_true",
        help="Evaluate the whole dataset, not just cached scenes.",
    )
    args = ap.parse_args()

    run(
        cache_root=args.cache_root,
        output=args.output,
        vlm_name=args.vlm,
        limit=args.limit,
        stratified=not args.no_stratified,
        seed=args.seed,
        n_frames=args.n_frames,
        max_tokens=args.max_tokens,
        only_cached=not args.all_scenes,
    )


if __name__ == "__main__":
    main()
