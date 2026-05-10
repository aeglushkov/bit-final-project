"""OpenEQA HM3D episode → cache-schema adapter.

OpenEQA's pre-rendered HM3D episodes come from `data/hm3d/extract-frames.py`
in facebookresearch/open-eqa. The actual on-disk layout is FLAT (one episode
directory containing `<frame>-rgb.png`, `<frame>-depth.png`, `<frame>.txt`,
plus `intrinsic_color.txt`/`intrinsic_depth.txt` once per directory):

  <episode_id>/
    intrinsic_color.txt        # 4x4 K matrix (extra row/col padded with zeros)
    intrinsic_depth.txt        # same K (depth and rgb share intrinsics)
    <frame_name>-rgb.png       # 1920x1080 RGB
    <frame_name>-depth.png     # uint16 PNG; meters = uint16 / 65535 * 10
    <frame_name>.txt           # 4x4 cam2world in Habitat (OpenGL) convention
    <frame_name>.pkl           # original agent state (ignored by us)

`adapt_episode` rewrites this into the eva-eval cache schema:
  out/
    frames/<NNNNNN>.jpg
    depth/<NNNNNN>.npy         # float32 metric meters
    poses.npy                  # (N, 4, 4) cam2world OpenCV convention
    intrinsics.json            # fx, fy, cx, cy, width, height, fov_h
    meta.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# Right-multiplied by a Habitat (OpenGL) cam2world to produce the equivalent
# OpenCV cam2world. The two conventions differ only in cam-space basis: OpenGL
# has +Y up, -Z forward; OpenCV has +Y down, +Z forward. Translation is
# unchanged. See https://github.com/facebookresearch/habitat-sim/issues/1093
_OPENGL_CAM_TO_OPENCV_CAM = np.diag([1.0, -1.0, -1.0, 1.0])

# Depth encoding used by OpenEQA's extract-frames.py: depth_meters = uint16 / 65535 * 10
_OPENEQA_DEPTH_SCALE = 10.0 / 65535.0


def intrinsics_from_fov(*, width: int, height: int, fov_h_rad: float) -> dict[str, float]:
    """fx = fy = W / (2 tan(fov_h / 2)); principal point at image center."""
    fx = float(width) / (2.0 * float(np.tan(fov_h_rad / 2.0)))
    return {
        "fx": fx,
        "fy": fx,  # square pixels assumed for HM3D Habitat renders
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
        "width": int(width),
        "height": int(height),
        "fov_h": float(fov_h_rad),
    }


def intrinsics_from_K(K: np.ndarray, *, width: int, height: int) -> dict[str, float]:
    """Build the cache intrinsics dict from a 3x3 (or 4x4 padded) K matrix."""
    K = np.asarray(K, dtype=np.float64)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    fov_h = 2.0 * float(np.arctan(width / (2.0 * fx)))
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": int(width),
        "height": int(height),
        "fov_h": float(fov_h),
    }


def habitat_to_opencv_pose(pose_4x4: np.ndarray) -> np.ndarray:
    """Convert a 4x4 cam2world from Habitat (OpenGL) to OpenCV convention.

    Right-multiplication by diag(1, -1, -1, 1) flips the Y and Z basis vectors
    of camera-space (turning OpenGL's +Y up / -Z forward into OpenCV's +Y down
    / +Z forward) while leaving the world-space translation column intact.
    """
    pose = np.asarray(pose_4x4, dtype=np.float64)
    return pose @ _OPENGL_CAM_TO_OPENCV_CAM


def adapt_episode(
    episode_raw_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Convert one extract-frames.py output directory into the eva-eval cache schema."""
    raw = Path(episode_raw_dir)
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    frames = _list_frames(raw)
    if not frames:
        raise RuntimeError(f"No frames found in {raw} (expected <name>-rgb.png + <name>-depth.png + <name>.txt)")

    width, height = _image_size(frames[0].rgb)
    K = _read_intrinsic_color(raw / "intrinsic_color.txt")
    intrinsics = intrinsics_from_K(K, width=width, height=height)
    (out / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))

    poses = np.stack([habitat_to_opencv_pose(_read_pose_txt(f.pose)) for f in frames]).astype(np.float32)
    np.save(out / "poses.npy", poses)

    for i, f in enumerate(frames):
        img = Image.open(f.rgb).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
        img.save(out / "frames" / f"{i:06d}.jpg", quality=85)
        depth = _load_depth_png(f.depth)
        np.save(out / "depth" / f"{i:06d}.npy", depth.astype(np.float32))

    timestamps = list(range(len(frames)))  # placeholder — paper's process_a_frame uses these only as keys
    meta = {
        "video": raw.name,
        "fps": 1.0,
        "n_frames": len(frames),
        "timestamps": timestamps,
        "source": "openeqa_hm3d",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class _FrameTriple:
    __slots__ = ("name", "rgb", "depth", "pose")

    def __init__(self, name: str, rgb: Path, depth: Path, pose: Path):
        self.name = name
        self.rgb = rgb
        self.depth = depth
        self.pose = pose


def _list_frames(raw: Path) -> list[_FrameTriple]:
    """Discover frames in OpenEQA's flat layout. Each frame has three companion
    files: `<name>-rgb.png`, `<name>-depth.png`, `<name>.txt`."""
    frames: list[_FrameTriple] = []
    for rgb in sorted(raw.glob("*-rgb.png")):
        name = rgb.name[: -len("-rgb.png")]
        depth = raw / f"{name}-depth.png"
        pose = raw / f"{name}.txt"
        if not (depth.exists() and pose.exists()):
            continue
        frames.append(_FrameTriple(name=name, rgb=rgb, depth=depth, pose=pose))
    return frames


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size  # (width, height)


def _read_intrinsic_color(path: Path) -> np.ndarray:
    """Load the 4x4 (or 3x3) K matrix written by extract-frames.py:save_intrinsics."""
    K = np.loadtxt(path)
    if K.shape == (4, 4):
        return K[:3, :3]
    if K.shape == (3, 3):
        return K
    raise ValueError(f"intrinsic_color.txt has unexpected shape {K.shape}: {path}")


def _read_pose_txt(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def _load_depth_png(path: Path) -> np.ndarray:
    """Decode OpenEQA's uint16 PNG depth: meters = uint16 / 65535 * 10."""
    arr = np.array(Image.open(path))
    return arr.astype(np.float32) * _OPENEQA_DEPTH_SCALE
