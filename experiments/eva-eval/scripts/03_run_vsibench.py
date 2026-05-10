from __future__ import annotations

import argparse
from pathlib import Path

from eva_eval.eval.vsibench import run


def main():
    ap = argparse.ArgumentParser(description="Phase 5: run VSI-Bench through the EVA agent.")
    ap.add_argument("--cache-root", type=Path, required=True, help="Phase 2/3 cache root (one subdir per video).")
    ap.add_argument("--paper-code-dir", type=Path, required=True)
    ap.add_argument("--classes-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="JSONL of per-question results; .summary.json is also written.")
    ap.add_argument("--planner", default=None, help="Planner model name from config/models.yaml (overrides default_planner).")
    ap.add_argument("--limit", type=int, default=None, help="If set, evaluate only this many questions.")
    ap.add_argument("--no-stratified", action="store_true", help="Disable stratified sampling under --limit.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iterations", type=int, default=30)
    ap.add_argument(
        "--on-missing-cache",
        choices=("skip", "fail"),
        default="skip",
        help="Behavior when a scene's memory.pkl is missing.",
    )
    ap.add_argument(
        "--all-scenes",
        action="store_true",
        help="Evaluate the full dataset, not just the scenes with cached memory. "
             "(Default behavior is to restrict to cached scenes.)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If --output already exists, skip questions whose IDs are already "
             "answered without error and append the rest.",
    )
    ap.add_argument(
        "--scenes",
        default=None,
        help="Restrict eval to a comma-separated list of scene names, or a path "
             "to a file containing one scene name per line (or a JSON list / "
             "selections.json with 'scene_name' fields).",
    )
    ap.add_argument(
        "--extended-schema",
        action="store_true",
        help="Use the extended SQL schema (bbox extents + dimensions/distance/"
             "room_size tools) and the matching prompt. Default: paper-faithful "
             "basic schema.",
    )
    args = ap.parse_args()

    scene_filter: set[str] | None = None
    if args.scenes:
        p = Path(args.scenes)
        if p.exists():
            text = p.read_text().strip()
            if text.startswith("["):
                import json as _json
                items = _json.loads(text)
                if items and isinstance(items[0], dict):
                    scene_filter = {s["scene_name"] for s in items if "scene_name" in s}
                else:
                    scene_filter = set(items)
            else:
                scene_filter = {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")}
        else:
            scene_filter = {s.strip() for s in args.scenes.split(",") if s.strip()}
        print(f"Scene filter: {len(scene_filter)} scenes")

    run(
        cache_root=args.cache_root,
        paper_code_dir=args.paper_code_dir,
        classes_file=args.classes_file,
        output=args.output,
        planner=args.planner,
        limit=args.limit,
        stratified=not args.no_stratified,
        seed=args.seed,
        max_iterations=args.max_iterations,
        on_missing_cache=args.on_missing_cache,
        only_cached=not args.all_scenes,
        resume=args.resume,
        scene_filter=scene_filter,
        extended_schema=args.extended_schema,
    )


if __name__ == "__main__":
    main()
