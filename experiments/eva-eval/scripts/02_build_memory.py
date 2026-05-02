from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eva_eval.preprocess.memory import build_memory_for_video


def _load_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def main():
    ap = argparse.ArgumentParser(description="Phase 3: build ObjectMemory from cached depth+pose.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video-cache", type=Path, help="Single Phase 2 cache dir.")
    src.add_argument("--cache-root", type=Path, help="Phase 2 cache root; iterate all subdirs.")

    ap.add_argument("--classes-file", type=Path, required=True, help="Detection vocabulary (one class per line).")
    ap.add_argument("--paper-code-dir", type=Path, required=True, help="Path to literature/EmbodiedVideoAgent/code/")
    ap.add_argument("--memory-name", default="memory.pkl", help="Output filename within each cache dir.")
    ap.add_argument("--depth-min", type=float, default=0.05)
    ap.add_argument("--depth-max", type=float, default=10.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    classes = _load_classes(args.classes_file)
    print(f"Loaded {len(classes)} detection classes from {args.classes_file}")

    if args.video_cache:
        cache_dirs = [args.video_cache]
    else:
        cache_dirs = sorted(d for d in args.cache_root.iterdir() if d.is_dir() and (d / "meta.json").exists())
    print(f"Building memory for {len(cache_dirs)} video cache(s)")

    failures: list[tuple[str, str]] = []
    for d in cache_dirs:
        out = d / args.memory_name
        if out.exists() and not args.overwrite:
            print(f"[skip] {d.name} ({out.name} exists)")
            continue
        try:
            print(f"[run]  {d.name}")
            build_memory_for_video(
                video_cache_dir=d,
                classes=classes,
                paper_code_dir=args.paper_code_dir,
                save_path=out,
                valid_depth_min=args.depth_min,
                valid_depth_max=args.depth_max,
            )
        except Exception as e:
            print(f"[fail] {d.name}: {e}", file=sys.stderr)
            failures.append((str(d), str(e)))

    if failures:
        log = (args.cache_root or args.video_cache.parent) / "memory_failures.jsonl"
        with log.open("w") as f:
            for d, e in failures:
                f.write(json.dumps({"dir": d, "error": e}) + "\n")
        print(f"\n{len(failures)} failures logged to {log}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
