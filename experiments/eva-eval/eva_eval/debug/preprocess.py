"""Renderers for the preprocess inspection HTML."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from eva_eval.debug.render import (
    colorize_depth,
    image_to_data_uri,
    trajectory_plot,
    write_html,
)


def render_preprocess_html(cache_dir: str | Path) -> Path:
    """Generate `_inspect/preprocess.html` for a cache dir.
    Returns the written file's path."""
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    intrinsics = json.loads((cache_dir / "intrinsics.json").read_text())
    poses = np.load(cache_dir / "poses.npy")
    n = int(meta["n_frames"])

    pick = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})

    body_parts: list[str] = []
    body_parts.append(f"<h1>Preprocess inspection — <code>{cache_dir.name}</code></h1>")
    body_parts.append("<h2>Header</h2>")
    body_parts.append(_header_table(meta=meta, intrinsics=intrinsics, poses=poses))

    body_parts.append("<h2>Frame strip</h2>")
    body_parts.append(_frame_strip(cache_dir, pick))

    body_parts.append("<h2>Depth (colorized)</h2>")
    body_parts.append(_depth_strip(cache_dir, pick))

    body_parts.append("<h2>Camera trajectory</h2>")
    body_parts.append(_trajectory(poses))

    body_parts.append("<h2>Reprojection self-check</h2>")
    body_parts.append(_reprojection_table(cache_dir, pick))

    return write_html(
        cache_dir / "_inspect" / "preprocess.html",
        title=f"preprocess: {cache_dir.name}",
        body="\n".join(body_parts),
    )


def reprojection_self_check(cache_dir: str | Path, source_pixel: tuple[int, int] = None) -> list[tuple[float, float]]:
    """Pick a depth pixel from frame 0, lift to a 3D world point, then project
    that world point through every other frame's pose+K. Returns the (u, v)
    pixel in each frame.

    For correct OpenCV cam2world poses, a static world point should reproject
    to roughly the same pixel across nearby frames. A diverging dot indicates
    a pose-convention bug (e.g., Habitat→OpenCV transform missing or wrong)."""
    cache_dir = Path(cache_dir)
    intrinsics = json.loads((cache_dir / "intrinsics.json").read_text())
    poses = np.load(cache_dir / "poses.npy")
    n = poses.shape[0]
    if source_pixel is None:
        W = int(intrinsics["width"])
        H = int(intrinsics["height"])
        source_pixel = (W // 2, H // 2)
    u0, v0 = source_pixel
    depth0 = np.load(cache_dir / "depth" / f"{0:06d}.npy")
    z0 = float(depth0[v0, u0])
    if not np.isfinite(z0) or z0 <= 0:
        return [(float("nan"), float("nan"))] * n

    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    p_cam0 = np.array([(u0 - cx) * z0 / fx, (v0 - cy) * z0 / fy, z0], dtype=np.float64)
    R0 = poses[0, :3, :3]
    t0 = poses[0, :3, 3]
    p_world = R0 @ p_cam0 + t0  # cam → world

    out: list[tuple[float, float]] = []
    for i in range(n):
        Ri = poses[i, :3, :3]
        ti = poses[i, :3, 3]
        p_cam_i = Ri.T @ (p_world - ti)  # R_i^T @ (p_world - t_i)
        z = p_cam_i[2]
        if z <= 1e-3:
            out.append((float("nan"), float("nan")))
            continue
        u = fx * p_cam_i[0] / z + cx
        v = fy * p_cam_i[1] / z + cy
        out.append((float(u), float(v)))
    return out


def _header_table(*, meta: dict, intrinsics: dict, poses: np.ndarray) -> str:
    rows = [
        ("source", meta.get("source", "?")),
        ("video", meta.get("video", "?")),
        ("n_frames", meta["n_frames"]),
        ("fps", meta.get("fps", "?")),
        ("image size", f"{intrinsics['width']} x {intrinsics['height']}"),
        ("fx, fy", f"{intrinsics['fx']:.2f}, {intrinsics['fy']:.2f}"),
        ("cx, cy", f"{intrinsics['cx']:.2f}, {intrinsics['cy']:.2f}"),
        ("fov_h (rad)", f"{intrinsics.get('fov_h', float('nan')):.4f}"),
        ("trajectory length (m)", f"{_traj_length(poses):.3f}"),
    ]
    body = "".join(f"<tr><th>{k}</th><td><code>{v}</code></td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _frame_strip(cache_dir: Path, pick: list[int]) -> str:
    parts = []
    for i in pick:
        path = cache_dir / "frames" / f"{i:06d}.jpg"
        if not path.exists():
            parts.append(f"<div>frame {i} missing</div>")
            continue
        img = Image.open(path).convert("RGB")
        parts.append(
            f'<div><img class="thumb" src="{image_to_data_uri(img)}">'
            f"<div>frame {i}</div></div>"
        )
    return f'<div class="row">{"".join(parts)}</div>'


def _depth_strip(cache_dir: Path, pick: list[int]) -> str:
    parts = []
    for i in pick:
        path = cache_dir / "depth" / f"{i:06d}.npy"
        if not path.exists():
            parts.append(f"<div>depth {i} missing</div>")
            continue
        arr = np.load(path)
        finite = np.isfinite(arr)
        d_min = float(arr[finite].min()) if finite.any() else float("nan")
        d_max = float(arr[finite].max()) if finite.any() else float("nan")
        d_mean = float(arr[finite].mean()) if finite.any() else float("nan")
        img = colorize_depth(arr)
        parts.append(
            f'<div><img class="thumb" src="{image_to_data_uri(img)}">'
            f"<div>frame {i} — min/mean/max: {d_min:.2f} / {d_mean:.2f} / {d_max:.2f} m</div></div>"
        )
    return f'<div class="row">{"".join(parts)}</div>'


def _trajectory(poses: np.ndarray) -> str:
    img = trajectory_plot(poses)
    return f'<img src="{image_to_data_uri(img)}">'


def _reprojection_table(cache_dir: Path, pick: list[int]) -> str:
    pixels = reprojection_self_check(cache_dir)
    if not pixels:
        return "<p>No frames to project.</p>"
    rows = [f"<tr><th>frame</th><th>u (px)</th><th>v (px)</th></tr>"]
    for i in pick:
        if i >= len(pixels):
            continue
        u, v = pixels[i]
        rows.append(
            f"<tr><td>{i}</td>"
            f"<td>{u:.1f}</td><td>{v:.1f}</td></tr>"
        )
    note = (
        "<p>If poses are correct in OpenCV cam2world convention, (u, v) should "
        "stay close to the source pixel (image center) across frames. "
        "Wandering values across <em>nearby</em> frames indicate a pose-convention bug.</p>"
    )
    return note + f"<table>{''.join(rows)}</table>"


def _traj_length(poses: np.ndarray) -> float:
    if poses.shape[0] < 2:
        return 0.0
    pts = poses[:, :3, 3]
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    return float(diffs.sum())
