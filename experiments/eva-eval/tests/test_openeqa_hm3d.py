from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_intrinsics_from_fov_round_trip():
    from eva_eval.preprocess.openeqa_hm3d import intrinsics_from_fov

    K = intrinsics_from_fov(width=640, height=480, fov_h_rad=np.pi / 2)
    assert K["width"] == 640 and K["height"] == 480
    assert K["cx"] == pytest.approx(320.0)
    assert K["cy"] == pytest.approx(240.0)
    assert K["fx"] == pytest.approx(320.0)  # tan(π/4) = 1, so fx = W/(2*1) = 320
    assert K["fy"] == pytest.approx(K["fx"])
    assert K["fov_h"] == pytest.approx(np.pi / 2)


def test_habitat_to_opencv_pose_is_idempotent_on_identity():
    from eva_eval.preprocess.openeqa_hm3d import habitat_to_opencv_pose

    out = habitat_to_opencv_pose(np.eye(4, dtype=np.float64))
    # Identity rotation in Habitat: forward=-Z, up=+Y, right=+X
    # Identity rotation in OpenCV: forward=+Z, up=-Y, right=+X
    # Conversion: flip Y and Z. The translation is unchanged.
    expected = np.diag([1.0, -1.0, -1.0, 1.0])
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_habitat_to_opencv_preserves_translation():
    from eva_eval.preprocess.openeqa_hm3d import habitat_to_opencv_pose

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    out = habitat_to_opencv_pose(pose)
    np.testing.assert_allclose(out[:3, 3], [1.0, 2.0, 3.0], atol=1e-9)


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
