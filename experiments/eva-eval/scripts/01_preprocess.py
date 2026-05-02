from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eva_eval.preprocess.mast3r import DEFAULT_MODEL, estimate_video


def _video_id(video_path: Path) -> str:
    return video_path.stem


def _process_one(video: Path, cache_root: Path, args) -> dict:
    out_dir = cache_root / _video_id(video)
    if out_dir.exists() and (out_dir / "meta.json").exists() and not args.overwrite:
        print(f"[skip] {video.name} (cache exists)")
        return json.loads((out_dir / "meta.json").read_text())
    print(f"[run]  {video.name} -> {out_dir}")
    return estimate_video(
        video_path=video,
        out_dir=out_dir,
        fps=args.fps,
        image_size=args.image_size,
        scene_graph=args.scene_graph,
        model_name=args.model,
        device=args.device,
        max_frames=args.max_frames,
    )


def _iter_videos(args) -> list[Path]:
    if args.video:
        return [args.video]
    if args.video_list:
        paths = [Path(line.strip()) for line in args.video_list.read_text().splitlines() if line.strip()]
        return paths
    if args.video_dir:
        return sorted(p for p in args.video_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"})
    raise SystemExit("Pass one of --video / --video-list / --video-dir")


def main():
    ap = argparse.ArgumentParser(description="Preprocess videos with MASt3R-SfM (depth + pose).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Single video file.")
    src.add_argument("--video-list", type=Path, help="Text file with one video path per line.")
    src.add_argument("--video-dir", type=Path, help="Directory of video files.")

    ap.add_argument("--cache-root", type=Path, required=True, help="Output cache root; per-video subdir created.")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--scene-graph", default="swin-3")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.cache_root.mkdir(parents=True, exist_ok=True)
    videos = _iter_videos(args)
    print(f"Processing {len(videos)} video(s) -> {args.cache_root}")

    failures: list[tuple[Path, str]] = []
    for v in videos:
        try:
            _process_one(v, args.cache_root, args)
        except Exception as e:
            print(f"[fail] {v.name}: {e}", file=sys.stderr)
            failures.append((v, str(e)))

    if failures:
        log = args.cache_root / "preprocess_failures.jsonl"
        with log.open("w") as f:
            for v, err in failures:
                f.write(json.dumps({"video": str(v), "error": err}) + "\n")
        print(f"\n{len(failures)} failures logged to {log}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
