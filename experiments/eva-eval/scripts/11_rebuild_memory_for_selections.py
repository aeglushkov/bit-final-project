"""Rebuild memory.pkl ONLY for the scenes referenced in a selections.json.

Used to refresh the gallery's memory caches after the radians→degrees fov fix
in `eva_eval/preprocess/memory.py`. Reuses `build_memory_for_video`, so behaviour
is identical to `02_build_memory.py` modulo the per-scene targeting.

Run on remote (Phase 3 perception is GPU-bound):
    python scripts/11_rebuild_memory_for_selections.py \
        --selections results/visuals/full_qwen_internvl2/selections.json \
        --cache-root cache/vsibench \
        --classes-file config/detection_classes.txt \
        --paper-code-dir ../../literature/EmbodiedVideoAgent/code
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eva_eval.preprocess.memory import build_memory_for_video  # noqa: E402


def _load_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--classes-file", type=Path, required=True)
    ap.add_argument("--paper-code-dir", type=Path, required=True)
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated scene names; restrict to this subset (smoke testing).")
    ap.add_argument("--memory-name", default="memory.pkl")
    args = ap.parse_args()

    sels = json.loads(args.selections.read_text())
    scenes = sorted({s["scene_name"] for s in sels})
    if args.only:
        keep = {x.strip() for x in args.only.split(",") if x.strip()}
        scenes = [s for s in scenes if s in keep]
    if not scenes:
        print("[fail] no scenes after filtering", file=sys.stderr)
        sys.exit(1)

    classes = _load_classes(args.classes_file)
    print(f"Rebuilding memory for {len(scenes)} scene(s); {len(classes)} classes")

    failures = []
    for i, scene in enumerate(scenes, 1):
        d = args.cache_root / scene
        out = d / args.memory_name
        if not (d / "meta.json").exists():
            print(f"[skip] {scene}: missing meta.json")
            failures.append((scene, "missing meta.json"))
            continue
        t0 = time.time()
        print(f"\n[run {i}/{len(scenes)}] {scene}")
        try:
            build_memory_for_video(
                video_cache_dir=d,
                classes=classes,
                paper_code_dir=args.paper_code_dir,
                save_path=out,
            )
            print(f"[ok]  {scene}  ({time.time() - t0:.1f}s)")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[fail] {scene}: {type(e).__name__}: {e}", file=sys.stderr)
            print(tb, file=sys.stderr)
            failures.append((scene, f"{type(e).__name__}: {e}"))

    print(f"\nDone. ok={len(scenes) - len(failures)} failed={len(failures)}")
    if failures:
        for s, e in failures:
            print(f"  {s}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
