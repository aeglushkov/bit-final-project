"""Tests for the extended-schema tools (get_object_dimensions / get_distance /
estimate_room_size). The tools use the same AgentContext objects as the
paper's pipeline; we stub min_xyz/max_xyz with numpy arrays."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from eva_eval.agent.tools import (
    do_estimate_room_size,
    do_get_distance,
    do_get_object_dimensions,
)


def _obj(identifier, category, mn, mx):
    return SimpleNamespace(
        identifier=identifier,
        category=category,
        min_xyz=np.asarray(mn, dtype=np.float64),
        max_xyz=np.asarray(mx, dtype=np.float64),
    )


def _ctx(*objs):
    return SimpleNamespace(object_index={int(o.identifier): o for o in objs})


def test_get_object_dimensions_reports_cm():
    ctx = _ctx(_obj(1, "toilet", [0, 0, 0], [0.4, 0.7, 0.6]))
    out = do_get_object_dimensions(ctx, 1)
    assert "length=40.0 cm" in out
    assert "height=70.0 cm" in out
    assert "width=60.0 cm" in out
    assert "longest_dimension=70.0 cm" in out
    assert "toilet" in out


def test_get_object_dimensions_unknown_id():
    ctx = _ctx(_obj(1, "x", [0, 0, 0], [1, 1, 1]))
    out = do_get_object_dimensions(ctx, 99)
    assert "not found" in out


def test_get_distance_axis_aligned_gap():
    a = _obj(1, "stove", [0, 0, 0], [1, 1, 1])
    b = _obj(2, "fridge", [3, 0, 0], [4, 1, 1])  # 2 m gap on x
    out = do_get_distance(_ctx(a, b), 1, 2)
    assert "2.000 m" in out


def test_get_distance_overlapping_objects_returns_zero():
    a = _obj(1, "rug", [0, 0, 0], [2, 0.1, 2])
    b = _obj(2, "table", [1, 0, 1], [3, 1, 3])  # overlapping
    out = do_get_distance(_ctx(a, b), 1, 2)
    assert "0.000 m" in out


def test_get_distance_diagonal_gap():
    a = _obj(1, "a", [0, 0, 0], [1, 1, 1])
    b = _obj(2, "b", [4, 1, 4], [5, 2, 5])  # 3 m gap on x and z each
    out = do_get_distance(_ctx(a, b), 1, 2)
    # gap = (3, 0, 3), euclidean = sqrt(18) = 4.243
    assert "4.243 m" in out or "4.242 m" in out


def test_estimate_room_size_uses_bbox_span():
    objs = [
        _obj(1, "a", [0, 0, 0], [0.1, 0.1, 0.1]),
        _obj(2, "b", [4, 0, 0], [4.1, 0.1, 0.1]),
        _obj(3, "c", [0, 0, 3], [0.1, 0.1, 3.1]),
        _obj(4, "d", [4, 0, 3], [4.1, 0.1, 3.1]),
    ]
    out = do_estimate_room_size(_ctx(*objs))
    # Span x ≈ 4 m, span z ≈ 3 m, bbox area ≈ 12; centers form a 4x3 rectangle so
    # convex hull area is also ~12.
    assert "x_span=4." in out
    assert "z_span=3." in out
    assert "n_objects=4" in out


def test_estimate_room_size_too_few_objects():
    out = do_estimate_room_size(_ctx(_obj(1, "x", [0, 0, 0], [1, 1, 1])))
    assert "cannot estimate" in out
