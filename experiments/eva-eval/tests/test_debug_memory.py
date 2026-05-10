from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def _fake_object(*, identifier: int, category: str, n_frames: int, volume: float, state: str = "normal"):
    obj = MagicMock(spec=["identifier", "category", "volume", "state", "min_xyz", "max_xyz", "object_clip_feature", "context_clip_feature"])
    obj.identifier = identifier
    obj.category = category
    obj.volume = volume
    obj.state = state
    obj.min_xyz = np.array([0, 0, 0], dtype=np.float64)
    obj.max_xyz = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    obj.object_clip_feature = np.zeros(8, dtype=np.float32)
    obj.context_clip_feature = np.zeros(8, dtype=np.float32)
    return obj


def test_summarize_memory_counts_objects_by_category():
    from eva_eval.debug.memory import summarize_memory

    objs = [
        _fake_object(identifier=1, category="chair", n_frames=5, volume=0.5),
        _fake_object(identifier=2, category="chair", n_frames=2, volume=0.4),
        _fake_object(identifier=3, category="table", n_frames=8, volume=2.5),
    ]
    objects_frames = {1: list(range(5)), 2: [0, 1], 3: list(range(8))}
    summary = summarize_memory(objs, objects_frames=objects_frames)
    assert summary["n_objects"] == 3
    assert summary["n_categories"] == 2
    assert summary["category_counts"]["chair"] == 2
    assert summary["category_counts"]["table"] == 1
    assert summary["pct_single_frame"] == pytest.approx(0.0)


def test_summarize_memory_flags_warnings():
    from eva_eval.debug.memory import summarize_memory

    # All single-frame, all huge volumes → both warnings
    objs = [
        _fake_object(identifier=i, category=f"cat{i}", n_frames=1, volume=200.0)
        for i in range(3)
    ]
    objects_frames = {i: [0] for i in range(3)}
    summary = summarize_memory(objs, objects_frames=objects_frames)
    assert any("re-ID" in w for w in summary["warnings"])
    assert any("volume" in w for w in summary["warnings"])


def test_summarize_memory_flags_zero_objects():
    from eva_eval.debug.memory import summarize_memory

    summary = summarize_memory([], objects_frames={})
    assert summary["n_objects"] == 0
    assert any("0 objects" in w for w in summary["warnings"])
