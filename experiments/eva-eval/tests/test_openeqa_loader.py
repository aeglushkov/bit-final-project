from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_questions(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "qs.json"
    p.write_text(json.dumps(rows))
    return p


def test_load_filters_to_hm3d_when_dataset_is_hm3d(tmp_path):
    from eva_eval.eval.openeqa import load_openeqa_questions

    p = _write_questions(tmp_path, [
        {"question_id": "1", "episode_history": "hm3d-v0/scene_a", "question": "Q1", "answer": "A1", "category": "object_recognition"},
        {"question_id": "2", "episode_history": "scannet-v0/scene_b", "question": "Q2", "answer": "A2", "category": "spatial_reasoning"},
    ])
    out = load_openeqa_questions(p, dataset="hm3d", limit=None)
    assert [r["question_id"] for r in out] == ["1"]


def test_load_returns_all_when_dataset_is_all(tmp_path):
    from eva_eval.eval.openeqa import load_openeqa_questions

    p = _write_questions(tmp_path, [
        {"question_id": "1", "episode_history": "hm3d-v0/a", "question": "Q1", "answer": "A1", "category": "x"},
        {"question_id": "2", "episode_history": "scannet-v0/b", "question": "Q2", "answer": "A2", "category": "y"},
    ])
    out = load_openeqa_questions(p, dataset="all", limit=None)
    assert len(out) == 2


def test_load_stratified_sample_balances_categories(tmp_path):
    from collections import Counter
    from eva_eval.eval.openeqa import load_openeqa_questions

    rows = []
    for i in range(20):
        rows.append({"question_id": f"a{i}", "episode_history": "hm3d-v0/x",
                     "question": "?", "answer": "?", "category": "alpha"})
    for i in range(20):
        rows.append({"question_id": f"b{i}", "episode_history": "hm3d-v0/y",
                     "question": "?", "answer": "?", "category": "beta"})
    p = _write_questions(tmp_path, rows)

    out = load_openeqa_questions(p, dataset="hm3d", limit=10, stratified=True, seed=42)
    counts = Counter(r["category"] for r in out)
    assert counts["alpha"] == 5
    assert counts["beta"] == 5


def test_format_question_uses_pre_prompt():
    from eva_eval.eval.openeqa import format_question

    text = format_question({"question": "How many chairs are there?"})
    assert "How many chairs are there?" in text
    assert "indoor scene" in text.lower()


def test_episode_cache_dir_strips_dataset_prefix(tmp_path):
    from eva_eval.eval.openeqa import episode_cache_dir

    out = episode_cache_dir(tmp_path, "hm3d-v0/00000-foo")
    assert out == tmp_path / "openeqa_hm3d" / "00000-foo"
