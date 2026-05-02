from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def _paper_code_context(paper_code_dir: Path):
    """The paper's ObjectMemory hardcodes weight paths relative to CWD
    (`data/model_weights/...`), so we chdir into the paper repo for the
    constructor call. We restore CWD + sys.path on exit."""
    paper_code_dir = paper_code_dir.resolve()
    original_cwd = os.getcwd()
    sys.path.insert(0, str(paper_code_dir))
    try:
        os.chdir(paper_code_dir)
        yield
    finally:
        os.chdir(original_cwd)
        if str(paper_code_dir) in sys.path:
            sys.path.remove(str(paper_code_dir))


def build_memory_for_video(
    video_cache_dir: str | Path,
    classes: list[str],
    paper_code_dir: str | Path,
    save_path: str | Path | None = None,
    valid_depth_min: float = 0.05,
    valid_depth_max: float = 10.0,
    progress: bool = True,
):
    import cv2

    from eva_eval.memory.store import save_memory

    video_cache_dir = Path(video_cache_dir).resolve()
    save_path = Path(save_path).resolve() if save_path else None
    paper_code_dir = Path(paper_code_dir)

    intrinsics = json.loads((video_cache_dir / "intrinsics.json").read_text())
    meta = json.loads((video_cache_dir / "meta.json").read_text())
    poses = np.load(video_cache_dir / "poses.npy")
    n = poses.shape[0]
    if n != meta["n_frames"]:
        raise RuntimeError(f"Frame count mismatch in {video_cache_dir}: poses={n} vs meta={meta['n_frames']}")

    fov = float(intrinsics["fov_h"])
    frames_dir = video_cache_dir / "frames"
    depth_dir = video_cache_dir / "depth"
    timestamps = meta["timestamps"]

    iterator = range(n)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=video_cache_dir.name, unit="frame")
        except ImportError:
            pass

    with _paper_code_context(paper_code_dir):
        from object_memory import ObjectMemory

        memory = ObjectMemory(classes=classes)
        for i in iterator:
            frame_path = frames_dir / f"{i:06d}.jpg"
            depth_path = depth_dir / f"{i:06d}.npy"
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                raise FileNotFoundError(frame_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = np.load(depth_path)
            depth_mask = np.logical_and(depth > valid_depth_min, depth < valid_depth_max)

            memory.process_a_frame(
                timestamp=float(timestamps[i]),
                rgb=rgb,
                depth=depth,
                depth_mask=depth_mask,
                pos=poses[i, :3, 3],
                rmat=poses[i, :3, :3],
                fov=fov,
            )

    if save_path is not None:
        save_memory(memory, save_path)
    return memory
