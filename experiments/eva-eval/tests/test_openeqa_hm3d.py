from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_intrinsics_from_fov_round_trip():
    from eva_eval.preprocess.openeqa_hm3d import intrinsics_from_fov

    K = intrinsics_from_fov(width=640, height=480, fov_h_rad=np.pi / 2)
    assert K["width"] == 640 and K["height"] == 480
    assert K["cx"] == pytest.approx(320.0)
    assert K["cy"] == pytest.approx(240.0)
    assert K["fx"] == pytest.approx(320.0)  # tan(π/4) = 1, so fx = W/(2*1) = 320
    assert K["fy"] == pytest.approx(K["fx"])
    assert K["fov_h"] == pytest.approx(np.pi / 2)


def test_intrinsics_from_K_extracts_fxfycxcy():
    from eva_eval.preprocess.openeqa_hm3d import intrinsics_from_K

    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    out = intrinsics_from_K(K, width=640, height=480)
    assert out["fx"] == pytest.approx(500.0)
    assert out["fy"] == pytest.approx(500.0)
    assert out["cx"] == pytest.approx(320.0)
    assert out["cy"] == pytest.approx(240.0)
    assert out["width"] == 640 and out["height"] == 480
    # fov_h = 2 * arctan(W / (2*fx)) = 2 * arctan(0.64) ≈ 1.1442 rad
    assert out["fov_h"] == pytest.approx(2.0 * np.arctan(640 / (2 * 500)))


def test_intrinsics_from_K_accepts_4x4_extract_frames_format():
    """extract-frames.py writes a 4x4 padded K — accept it transparently."""
    from eva_eval.preprocess.openeqa_hm3d import intrinsics_from_K

    K_4x4 = np.zeros((4, 4))
    K_4x4[0, 0] = 500.0
    K_4x4[1, 1] = 500.0
    K_4x4[0, 2] = 320.0
    K_4x4[1, 2] = 240.0
    K_4x4[2, 2] = 1.0
    out = intrinsics_from_K(K_4x4, width=640, height=480)
    assert out["fx"] == pytest.approx(500.0)


def test_habitat_to_opencv_pose_is_idempotent_on_identity():
    from eva_eval.preprocess.openeqa_hm3d import habitat_to_opencv_pose

    out = habitat_to_opencv_pose(np.eye(4, dtype=np.float64))
    # Identity Habitat (OpenGL) cam2world: cam-space has +Y up, -Z forward.
    # Equivalent OpenCV cam2world has +Y down, +Z forward, so the pose's
    # rotation block must flip the Y and Z basis vectors.
    expected = np.diag([1.0, -1.0, -1.0, 1.0])
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_habitat_to_opencv_preserves_translation():
    from eva_eval.preprocess.openeqa_hm3d import habitat_to_opencv_pose

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    out = habitat_to_opencv_pose(pose)
    np.testing.assert_allclose(out[:3, 3], [1.0, 2.0, 3.0], atol=1e-9)


def test_habitat_to_opencv_with_yaw_rotation_negates_columns_not_rows():
    """The fix: cam-frame conversion right-multiplies the pose by diag(1,-1,-1,1),
    which negates *columns* 1 and 2 of the rotation block. An earlier buggy
    implementation negated rows instead — passing identity but failing here."""
    from eva_eval.preprocess.openeqa_hm3d import habitat_to_opencv_pose

    # 90° yaw around Y in Habitat: R = [[0,0,1],[0,1,0],[-1,0,0]]
    R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    pose = np.eye(4)
    pose[:3, :3] = R
    out = habitat_to_opencv_pose(pose)

    # Expected: R @ diag(1, -1, -1) — cols 1, 2 negated:
    expected_R = np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(out[:3, :3], expected_R, atol=1e-9)


def test_habitat_to_opencv_double_application_is_identity():
    from eva_eval.preprocess.openeqa_hm3d import habitat_to_opencv_pose

    rng = np.random.default_rng(0)
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = rng.standard_normal(3)
    twice = habitat_to_opencv_pose(habitat_to_opencv_pose(pose))
    np.testing.assert_allclose(twice, pose, atol=1e-9)


def test_load_depth_png_decodes_openeqa_scale(tmp_path):
    """OpenEQA encodes depth as uint16 with meters = uint16 / 65535 * 10."""
    from eva_eval.preprocess.openeqa_hm3d import _load_depth_png

    # 5 m depth → uint16 value 5/10 * 65535 = 32767.5 → round to 32768.
    arr = np.full((4, 4), 32768, dtype=np.uint16)
    p = tmp_path / "f-depth.png"
    Image.fromarray(arr).save(p)

    depth = _load_depth_png(p)
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, 32768 * 10.0 / 65535.0, atol=1e-3)


def test_adapt_episode_writes_cache_schema(tmp_path):
    """End-to-end: synthetic OpenEQA flat-layout dir → cache schema."""
    from eva_eval.preprocess.openeqa_hm3d import adapt_episode

    raw = tmp_path / "00000-foo"
    raw.mkdir()

    # Two frames (keep test fast). Names are arbitrary but must sort consistently.
    for name in ("0001", "0002"):
        Image.new("RGB", (32, 24), color=(50, 100, 150)).save(raw / f"{name}-rgb.png")
        depth_u16 = np.full((24, 32), 16384, dtype=np.uint16)  # ~2.5 m
        Image.fromarray(depth_u16).save(raw / f"{name}-depth.png")
        np.savetxt(raw / f"{name}.txt", np.eye(4))

    # 4x4 padded K matching extract-frames.py output for hfov=90°, 32x24
    K_4x4 = np.zeros((4, 4))
    K_4x4[0, 0] = 16.0  # W / (2 * tan(45°)) = 32 / 2 = 16
    K_4x4[1, 1] = 12.0  # H / (2 * tan(VFOV/2))
    K_4x4[0, 2] = 16.0
    K_4x4[1, 2] = 12.0
    K_4x4[2, 2] = 1.0
    np.savetxt(raw / "intrinsic_color.txt", K_4x4)

    out_dir = tmp_path / "cache"
    meta = adapt_episode(raw, out_dir)

    assert meta["n_frames"] == 2
    assert meta["source"] == "openeqa_hm3d"
    assert (out_dir / "frames" / "000000.jpg").exists()
    assert (out_dir / "frames" / "000001.jpg").exists()
    assert (out_dir / "depth" / "000000.npy").exists()
    poses = np.load(out_dir / "poses.npy")
    assert poses.shape == (2, 4, 4)
    intrinsics = json.loads((out_dir / "intrinsics.json").read_text())
    assert intrinsics["fx"] == pytest.approx(16.0)
    # OpenCV-converted pose of identity Habitat = diag(1, -1, -1, 1)
    np.testing.assert_allclose(poses[0], np.diag([1.0, -1.0, -1.0, 1.0]), atol=1e-6)


def test_adapt_episode_raises_when_no_frames(tmp_path):
    from eva_eval.preprocess.openeqa_hm3d import adapt_episode

    empty = tmp_path / "empty-episode"
    empty.mkdir()
    (empty / "intrinsic_color.txt").write_text("0 0 0 0\n0 0 0 0\n0 0 0 0\n0 0 0 0\n")
    with pytest.raises(RuntimeError, match="No frames"):
        adapt_episode(empty, tmp_path / "out")
