"""OpenEQA HM3D episode → cache-schema adapter.

OpenEQA's pre-rendered HM3D episodes are downloaded via the script in
facebookresearch/open-eqa. Per-frame layout (verified at runtime by 06_preprocess_openeqa.py):
  <episode_id>/
    rgb/<NNNNN>.png        (or .jpg)
    depth/<NNNNN>.npy      (float32 metric meters)
    pose/<NNNNN>.txt       (4x4 cam2world in Habitat convention; one matrix per file)
    intrinsic.json         (fov_h or fx/fy/cx/cy at episode level)

If the download format differs, _read_episode/_read_pose/_load_depth handle the
common variants; if a brand-new format appears, extend those helpers.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# Diagonal matrix that flips Y and Z. Habitat is (+X right, +Y up, -Z forward,
# OpenGL-style). OpenCV is (+X right, +Y down, +Z forward). The conversion is
# applied via the sandwich M @ pose @ M (NOT M @ pose) — both axes of the
# rotation block need flipping, and the translation needs to remain in world
# coordinates that themselves stay in OpenCV convention.
_HABITAT_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


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


def habitat_to_opencv_pose(pose_4x4: np.ndarray) -> np.ndarray:
    """Convert a 4x4 cam2world from Habitat (OpenGL) to OpenCV convention.
    Self-inverse: applying twice returns the original.

    Only flips the rotation axes (Y and Z); translation is unchanged.
    """
    pose = np.asarray(pose_4x4, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    # Flip Y and Z axes of rotation: multiply rows 1,2 by -1
    out[0, :] = pose[0, :]
    out[1, :] = -pose[1, :]
    out[2, :] = -pose[2, :]
    out[3, :] = pose[3, :]
    # Undo the flip to translation (restore its original value)
    out[0, 3] = pose[0, 3]
    out[1, 3] = pose[1, 3]
    out[2, 3] = pose[2, 3]
    return out


def adapt_episode(
    episode_raw_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Convert one OpenEQA HM3D episode directory into the eva-eval cache schema.

    Reads:    rgb/, depth/, pose/, intrinsic.json from `episode_raw_dir`
    Writes:   frames/, depth/, poses.npy, intrinsics.json, meta.json in `out_dir`
    Returns:  the meta dict
    """
    raw = Path(episode_raw_dir)
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    rgb_paths, depth_paths, pose_paths = _read_episode(raw)
    n = len(rgb_paths)
    if n == 0:
        raise RuntimeError(f"No frames found in {raw}")
    if not (n == len(depth_paths) == len(pose_paths)):
        raise RuntimeError(f"Frame count mismatch in {raw}: rgb={n} depth={len(depth_paths)} pose={len(pose_paths)}")

    intrinsics_raw = json.loads((raw / "intrinsic.json").read_text())
    width, height = _image_size(rgb_paths[0])
    fov_h = _read_fov_h(intrinsics_raw, width=width, height=height)
    intrinsics = intrinsics_from_fov(width=width, height=height, fov_h_rad=fov_h)
    (out / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))

    poses = np.stack([
        habitat_to_opencv_pose(_read_pose(p)) for p in pose_paths
    ]).astype(np.float32)
    np.save(out / "poses.npy", poses)

    for i, (rgb_p, depth_p) in enumerate(zip(rgb_paths, depth_paths)):
        img = Image.open(rgb_p).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
        img.save(out / "frames" / f"{i:06d}.jpg", quality=85)
        depth = _load_depth(depth_p)
        np.save(out / "depth" / f"{i:06d}.npy", depth.astype(np.float32))

    timestamps = list(range(n))  # placeholder — paper's process_a_frame uses these only as keys
    meta = {
        "video": raw.name,
        "fps": 1.0,
        "n_frames": n,
        "timestamps": timestamps,
        "source": "openeqa_hm3d",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _read_episode(raw: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Return sorted lists of (rgb, depth, pose) per-frame paths."""
    rgb_dir = raw / "rgb"
    depth_dir = raw / "depth"
    pose_dir = raw / "pose"
    rgb = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    depth = sorted(p for p in depth_dir.iterdir() if p.suffix.lower() in (".npy", ".png"))
    pose = sorted(p for p in pose_dir.iterdir() if p.suffix.lower() in (".txt", ".json", ".npy"))
    return rgb, depth, pose


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size  # (width, height)


def _read_fov_h(intrinsics_raw: dict, *, width: int, height: int) -> float:
    """Extract horizontal FOV in radians from OpenEQA's intrinsic.json.
    Accepts either 'fov_h' (radians), 'hfov' (degrees), or fx (px)."""
    if "fov_h" in intrinsics_raw:
        return float(intrinsics_raw["fov_h"])
    if "hfov" in intrinsics_raw:
        return float(np.deg2rad(float(intrinsics_raw["hfov"])))
    if "fx" in intrinsics_raw:
        return 2.0 * float(np.arctan(width / (2.0 * float(intrinsics_raw["fx"]))))
    raise KeyError(f"intrinsic.json missing fov_h, hfov, or fx: keys={sorted(intrinsics_raw)}")


def _read_pose(path: Path) -> np.ndarray:
    """Load a 4x4 cam2world matrix from .txt (whitespace), .npy, or .json."""
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".json":
        return np.array(json.loads(path.read_text()), dtype=np.float64)
    return np.loadtxt(path)


def _load_depth(path: Path) -> np.ndarray:
    """Load depth as float32 metric meters. Accepts .npy or 16-bit png (mm)."""
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    if path.suffix == ".png":
        # 16-bit PNG: assume millimeters (HM3D's pre-rendered convention if PNG)
        from PIL import Image as _Image
        arr = np.array(_Image.open(path))
        return (arr.astype(np.float32) / 1000.0)
    raise ValueError(f"Unsupported depth format: {path.suffix}")
