"""Per-episode: download → adapt → build memory → cleanup.

Reads a sampled-questions JSON (one row per question, with `episode_history`),
extracts the unique episode IDs, and processes each one in turn.

Usage:
    python scripts/06_preprocess_openeqa.py \
        --sampled-json <cache_root>/openeqa_hm3d/sampled_50.json \
        --cache-root <cache_root> \
        --paper-code-dir <path/to/literature/EmbodiedVideoAgent/code> \
        --classes-file <path/to/detection_classes.txt> \
        --bundle-url-template "https://example.com/hm3d/{episode_id}.tar.gz"

Cleanup behavior:
  - After preprocessing each episode, episodes_raw/<episode_id>/ is deleted.
  - After memory build, <episode_id>/depth/ is deleted.
  - Both can be disabled with --keep-raw / --keep-depth for debugging.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable

from eva_eval.preprocess.memory import build_memory_for_video
from eva_eval.preprocess.openeqa_hm3d import adapt_episode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampled-json", type=Path, required=True,
                    help="JSON file with rows containing 'episode_history' (e.g. 'hm3d-v0/<id>')")
    ap.add_argument("--cache-root", type=Path, required=True,
                    help="Root above the openeqa_hm3d/ subdir.")
    ap.add_argument("--paper-code-dir", type=Path, required=True)
    ap.add_argument("--classes-file", type=Path, required=True)
    ap.add_argument("--bundle-url-template", required=True,
                    help="URL template with '{episode_id}' placeholder for per-episode tar.gz")
    ap.add_argument("--keep-raw", action="store_true",
                    help="Keep episodes_raw/ after preprocessing (debug)")
    ap.add_argument("--keep-depth", action="store_true",
                    help="Keep depth/*.npy after memory build (debug)")
    ap.add_argument("--depth-min", type=float, default=0.05)
    ap.add_argument("--depth-max", type=float, default=10.0)
    args = ap.parse_args()

    rows = json.loads(args.sampled_json.read_text())
    episode_ids = sorted({_episode_id(r["episode_history"]) for r in rows})
    print(f"Processing {len(episode_ids)} unique episodes from {len(rows)} questions")

    classes = _load_classes(args.classes_file)
    print(f"Loaded {len(classes)} detection classes")

    cache_dir_root = args.cache_root / "openeqa_hm3d"
    cache_dir_root.mkdir(parents=True, exist_ok=True)
    raw_root = cache_dir_root / "episodes_raw"
    raw_root.mkdir(exist_ok=True)

    failures: list[tuple[str, str]] = []
    for ep_id in episode_ids:
        ep_cache = cache_dir_root / ep_id
        if (ep_cache / "memory.pkl").exists():
            print(f"[skip] {ep_id} (memory.pkl exists)")
            continue
        try:
            print(f"[run] {ep_id}")
            ep_raw = raw_root / ep_id
            if not ep_raw.exists():
                _download_and_extract(args.bundle_url_template.format(episode_id=ep_id), ep_raw)
            adapt_episode(ep_raw, ep_cache)
            build_memory_for_video(
                video_cache_dir=ep_cache,
                classes=classes,
                paper_code_dir=args.paper_code_dir,
                save_path=ep_cache / "memory.pkl",
                valid_depth_min=args.depth_min,
                valid_depth_max=args.depth_max,
            )
            if not args.keep_raw:
                shutil.rmtree(ep_raw, ignore_errors=True)
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


def _download_and_extract(url: str, out_dir: Path):
    """Download tar.gz to a temp file, extract, move into out_dir."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"  download {url}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        with tarfile.open(tmp_path, "r:gz") as tf:
            tf.extractall(out_dir.parent)
        # If the tar extracts to a different name, move it.
        # Common case: tar contains <episode_id>/...
        # If `out_dir` doesn't exist after extract, find the new dir.
        if not out_dir.exists():
            # Find the most recently modified subdir of out_dir.parent
            candidates = [d for d in out_dir.parent.iterdir() if d.is_dir() and d.name not in {"."}]
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                candidates[0].rename(out_dir)
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
