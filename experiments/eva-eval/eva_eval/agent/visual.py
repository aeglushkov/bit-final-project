from __future__ import annotations

from pathlib import Path

import numpy as np

_BBOX_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 3),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def aabb_corners_world(min_xyz: np.ndarray, max_xyz: np.ndarray) -> np.ndarray:
    mn, mx = np.asarray(min_xyz), np.asarray(max_xyz)
    return np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mx[1], mx[2]],
        ],
        dtype=np.float64,
    )


def project_world_to_pixels(
    points_world: np.ndarray,
    cam2world: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) world points to (N, 2) pixels using OpenCV camera convention
    (right=+x, down=+y, forward=+z). Returns (uv, in_front_mask)."""
    R = cam2world[:3, :3]
    t = cam2world[:3, 3]
    points_cam = (points_world - t) @ R  # equivalent to (R^T @ (p - t)^T)^T
    z = points_cam[:, 2]
    in_front = z > 1e-3
    z_safe = np.where(in_front, z, 1.0)
    u = K[0, 0] * points_cam[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / z_safe + K[1, 2]
    return np.stack([u, v], axis=-1), in_front


def render_3d_bbox_on_frame(
    frame_path: str | Path,
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    cam2world: np.ndarray,
    K: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    width: int = 2,
):
    from PIL import Image, ImageDraw

    image = Image.open(str(frame_path)).convert("RGB")
    corners = aabb_corners_world(min_xyz, max_xyz)
    uv, in_front = project_world_to_pixels(corners, cam2world, K)

    draw = ImageDraw.Draw(image)
    W, H = image.size
    for a, b in _BBOX_EDGES:
        if not (in_front[a] and in_front[b]):
            continue
        ua, va = uv[a]
        ub, vb = uv[b]
        if not (
            -W <= ua <= 2 * W and -H <= va <= 2 * H and -W <= ub <= 2 * W and -H <= vb <= 2 * H
        ):
            continue
        draw.line([(float(ua), float(va)), (float(ub), float(vb))], fill=color, width=width)
    return image
