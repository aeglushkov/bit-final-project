import numpy as np

from eva_eval.agent.visual import aabb_corners_world, project_world_to_pixels


def test_aabb_corners_shape_and_extents():
    mn = np.array([-1.0, -2.0, 5.0])
    mx = np.array([1.0, 2.0, 7.0])
    corners = aabb_corners_world(mn, mx)
    assert corners.shape == (8, 3)
    assert np.allclose(corners.min(axis=0), mn)
    assert np.allclose(corners.max(axis=0), mx)


def test_project_identity_pose_centers_correctly():
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    points = np.array([[0.0, 0.0, 1.0]])
    uv, in_front = project_world_to_pixels(points, pose, K)
    assert in_front[0]
    assert np.allclose(uv[0], [50.0, 40.0])


def test_project_marks_points_behind_camera():
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    points = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    _, in_front = project_world_to_pixels(points, pose, K)
    assert not in_front[0]
    assert in_front[1]


def test_project_translation_then_back():
    K = np.array([[200.0, 0.0, 100.0], [0.0, 200.0, 100.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[:3, 3] = [3.0, 0.0, 0.0]
    point_in_front_of_translated_cam = np.array([[3.0, 0.0, 2.0]])
    uv, in_front = project_world_to_pixels(point_in_front_of_translated_cam, pose, K)
    assert in_front[0]
    assert np.allclose(uv[0], [100.0, 100.0])
