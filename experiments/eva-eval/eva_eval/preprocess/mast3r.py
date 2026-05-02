from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from eva_eval.preprocess.video import sample_video_frames

DEFAULT_MODEL = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


def estimate_video(
    video_path: str | Path,
    out_dir: str | Path,
    fps: float = 1.0,
    image_size: int = 512,
    scene_graph: str = "swin-3",
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run MASt3R-SfM on `video_path`, write per-frame depth+pose+intrinsics to `out_dir`.

    Cache layout:
        out_dir/
          frames/{i:06d}.jpg
          depth/{i:06d}.npy        # (H, W) float32, metric depth
          poses.npy                # (N, 4, 4) cam2world in OpenCV convention
          intrinsics.json          # fx, fy, cx, cy, width, height, fov_h
          meta.json
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    depth_dir = out_dir / "depth"
    frames_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    frame_paths, timestamps = sample_video_frames(video_path, fps, frames_dir, max_frames=max_frames)
    n = len(frame_paths)
    if n < 2:
        raise RuntimeError(f"Need >=2 frames for SfM, got {n} from {video_path}")

    scene = _run_sparse_ga(
        frame_paths=[str(p) for p in frame_paths],
        cache_dir=out_dir / ".mast3r_cache",
        model_name=model_name,
        scene_graph=scene_graph,
        image_size=image_size,
        device=device,
    )

    poses, intrinsics_K, depthmaps = _extract_outputs(scene)

    np.save(out_dir / "poses.npy", poses.astype(np.float32))
    for i, depth in enumerate(depthmaps):
        np.save(depth_dir / f"{i:06d}.npy", depth.astype(np.float32))

    K = intrinsics_K[0]
    H, W = depthmaps[0].shape[-2:]
    fov_h = float(2.0 * np.arctan(W / (2.0 * float(K[0, 0]))))
    intrinsics_payload = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "width": int(W),
        "height": int(H),
        "fov_h": fov_h,
    }
    (out_dir / "intrinsics.json").write_text(json.dumps(intrinsics_payload, indent=2))

    meta = {
        "video": str(video_path),
        "fps": fps,
        "n_frames": n,
        "timestamps": timestamps,
        "model": model_name,
        "scene_graph": scene_graph,
        "image_size": image_size,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _run_sparse_ga(
    frame_paths: list[str],
    cache_dir: Path,
    model_name: str,
    scene_graph: str,
    image_size: int,
    device: str,
):
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.model import AsymmetricMASt3R

    cache_dir.mkdir(exist_ok=True)
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device).eval()
    images = load_images(frame_paths, size=image_size)
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    return sparse_global_alignment(
        frame_paths,
        pairs,
        str(cache_dir),
        model,
        device=device,
        opt_depth=True,
        shared_intrinsics=True,
    )


def _extract_outputs(scene) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    import torch

    def _to_np(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    print("[_extract_outputs] step: get_im_poses")
    poses = _to_np(scene.get_im_poses())
    print("  poses shape:", poses.shape)
    n = poses.shape[0]

    print("[_extract_outputs] step: get_focals")
    focals_raw = scene.get_focals()
    print("  focals raw:", type(focals_raw).__name__, getattr(focals_raw, "shape", None))
    focals = _to_np(focals_raw).reshape(-1).astype(np.float64)
    if focals.size == 1:
        focals = np.broadcast_to(focals, (n,)).copy()
    print("  focals shape:", focals.shape)

    print("[_extract_outputs] step: get_principal_points")
    pps_raw = scene.get_principal_points()
    print("  pps raw:", type(pps_raw).__name__, getattr(pps_raw, "shape", None))
    pps = _to_np(pps_raw).astype(np.float64)
    if pps.ndim == 1:
        pps = np.broadcast_to(pps.reshape(1, 2), (n, 2)).copy()
    print("  pps shape:", pps.shape)

    K = np.zeros((n, 3, 3), dtype=np.float64)
    K[:, 0, 0] = focals
    K[:, 1, 1] = focals
    K[:, 0, 2] = pps[:, 0]
    K[:, 1, 2] = pps[:, 1]
    K[:, 2, 2] = 1.0
    print("  K built")

    print("[_extract_outputs] step: get_depthmaps")
    depths_raw = scene.get_depthmaps()
    print("  depths raw:", type(depths_raw).__name__, "len" if hasattr(depths_raw, "__len__") else "no-len",
          len(depths_raw) if hasattr(depths_raw, "__len__") else "?")
    depthmaps = [_to_np(d) for d in depths_raw]
    print("  depths[0] shape:", depthmaps[0].shape if depthmaps else "EMPTY")
    return poses, K, depthmaps
