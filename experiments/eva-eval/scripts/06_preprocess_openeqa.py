"""Per-episode: adapt → build memory → optional cleanup.

Reads a sampled-questions JSON, extracts the unique episode IDs, and
processes each one out of `--extracted-root` (a directory of episode
subdirectories produced by openeqa's `data/hm3d/extract-frames.py`).

Usage:
    python scripts/06_preprocess_openeqa.py \\
        --sampled-json <cache_root>/openeqa_hm3d/sampled_50.json \\
        --extracted-root <openeqa_repo>/data/frames/hm3d-v0 \\
        --cache-root <cache_root> \\
        --paper-code-dir <path/to/literature/EmbodiedVideoAgent/code> \\
        --classes-file <path/to/detection_classes.txt>

After memory build, <episode_id>/depth/ is deleted to keep disk usage
bounded; pass --keep-depth to retain for debugging.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from eva_eval.preprocess.memory import build_memory_for_video
from eva_eval.preprocess.openeqa_hm3d import adapt_episode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampled-json", type=Path, required=True,
                    help="JSON file with rows containing 'episode_history' (e.g. 'hm3d-v0/<id>')")
    ap.add_argument("--extracted-root", type=Path, required=True,
                    help="Directory containing extract-frames.py output: one subdir per episode "
                         "with <name>-rgb.png / <name>-depth.png / <name>.txt / intrinsic_color.txt")
    ap.add_argument("--cache-root", type=Path, required=True,
                    help="Root above the openeqa_hm3d/ subdir.")
    ap.add_argument("--paper-code-dir", type=Path, required=True)
    ap.add_argument("--classes-file", type=Path, required=True)
    ap.add_argument("--keep-depth", action="store_true",
                    help="Keep depth/*.npy after memory build (debug)")
    ap.add_argument("--depth-min", type=float, default=0.05)
    ap.add_argument("--depth-max", type=float, default=10.0)
    args = ap.parse_args()

    if not args.extracted_root.exists():
        print(f"ERROR: --extracted-root does not exist: {args.extracted_root}", file=sys.stderr)
        sys.exit(2)

    rows = json.loads(args.sampled_json.read_text())
    episode_ids = sorted({_episode_id(r["episode_history"]) for r in rows})
    print(f"Processing {len(episode_ids)} unique episodes from {len(rows)} questions")

    classes = _load_classes(args.classes_file)
    print(f"Loaded {len(classes)} detection classes")

    cache_dir_root = args.cache_root / "openeqa_hm3d"
    cache_dir_root.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for ep_id in episode_ids:
        ep_cache = cache_dir_root / ep_id
        if (ep_cache / "memory.pkl").exists():
            print(f"[skip] {ep_id} (memory.pkl exists)")
            continue
        ep_raw = args.extracted_root / ep_id
        if not ep_raw.exists():
            print(f"[fail] {ep_id}: source dir not found at {ep_raw}", file=sys.stderr)
            failures.append((ep_id, f"missing source: {ep_raw}"))
            continue
        try:
            print(f"[run] {ep_id}")
            adapt_episode(ep_raw, ep_cache)
            build_memory_for_video(
                video_cache_dir=ep_cache,
                classes=classes,
                paper_code_dir=args.paper_code_dir,
                save_path=ep_cache / "memory.pkl",
                valid_depth_min=args.depth_min,
                valid_depth_max=args.depth_max,
            )
            if not args.keep_depth:
                shutil.rmtree(ep_cache / "depth", ignore_errors=True)
        except Exception as e:
            print(f"[fail] {ep_id}: {type(e).__name__}: {e}", file=sys.stderr)
            failures.append((ep_id, f"{type(e).__name__}: {e}"))

    if failures:
        log = cache_dir_root / "preprocess_failures.jsonl"
        with log.open("w") as f:
            for ep_id, err in failures:
                f.write(json.dumps({"episode_id": ep_id, "error": err}) + "\n")
        print(f"\n{len(failures)} failures logged to {log}", file=sys.stderr)
        sys.exit(1)


def _episode_id(episode_history: str) -> str:
    """'hm3d-v0/<episode_id>' -> '<episode_id>'."""
    if "/" not in episode_history:
        return episode_history
    return episode_history.split("/", 1)[1]


def _load_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


if __name__ == "__main__":
    main()
