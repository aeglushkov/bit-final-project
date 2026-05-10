from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _make_cache_dir(tmp_path: Path, n: int = 5) -> Path:
    """Build a minimal cache dir with N synthetic frames."""
    cache = tmp_path / "scene"
    (cache / "frames").mkdir(parents=True)
    (cache / "depth").mkdir(parents=True)
    for i in range(n):
        img = Image.new("RGB", (64, 48), color=(i * 50 % 255, 100, 200))
        img.save(cache / "frames" / f"{i:06d}.jpg")
        depth = np.full((48, 64), 1.5 + 0.1 * i, dtype=np.float32)
        np.save(cache / "depth" / f"{i:06d}.npy", depth)
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    np.save(cache / "poses.npy", poses)
    intrinsics = {"fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0, "width": 64, "height": 48, "fov_h": 1.2}
    (cache / "intrinsics.json").write_text(json.dumps(intrinsics))
    meta = {"video": "synthetic", "fps": 1.0, "n_frames": n, "timestamps": list(range(n)), "source": "test"}
    (cache / "meta.json").write_text(json.dumps(meta))
    return cache


def test_render_preprocess_html_writes_file_and_includes_sections(tmp_path):
    from eva_eval.debug.preprocess import render_preprocess_html

    cache = _make_cache_dir(tmp_path)
    out = render_preprocess_html(cache)
    assert out == cache / "_inspect" / "preprocess.html"
    assert out.exists()
    text = out.read_text()
    assert "Preprocess inspection" in text
    assert "Frame strip" in text
    assert "Depth (colorized)" in text
    assert "Camera trajectory" in text
    assert "synthetic" in text  # source name from meta.json


def test_reprojection_self_check_with_identity_poses_returns_stable_pixel(tmp_path):
    from eva_eval.debug.preprocess import reprojection_self_check

    cache = _make_cache_dir(tmp_path)
    # Identity poses + constant depth means the depth-pixel reprojects to the
    # exact same pixel in every frame.
    pixel_uvs = reprojection_self_check(cache, source_pixel=(20, 20))
    assert len(pixel_uvs) == 5
    for u, v in pixel_uvs:
        assert abs(u - 20.0) < 0.5
        assert abs(v - 20.0) < 0.5
