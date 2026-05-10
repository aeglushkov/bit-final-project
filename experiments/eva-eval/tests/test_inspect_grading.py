from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_graded(tmp_path: Path) -> Path:
    rows = [
        {"id": f"q{i}", "category": "object_recognition" if i % 2 else "spatial_reasoning",
         "question": f"Question {i}?", "ground_truth": f"Answer {i}", "prediction": f"Pred {i}",
         "score": (i % 5) + 1, "judge_rationale": f"reason {i}"}
        for i in range(20)
    ]
    p = tmp_path / "graded.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_render_grading_html_writes_file_with_summary_sections(tmp_path):
    from eva_eval.debug.grading import render_grading_html

    graded = _write_graded(tmp_path)
    out = render_grading_html(graded)
    assert out.exists()
    text = out.read_text()
    assert "Grading inspection" in text
    assert "Per-category" in text
    assert "Worst-10" in text
    assert "Best-10" in text


def test_render_grading_html_includes_judge_score_histogram(tmp_path):
    from eva_eval.debug.grading import render_grading_html

    graded = _write_graded(tmp_path)
    out = render_grading_html(graded)
    text = out.read_text()
    # Histogram has 5 buckets labeled 1..5
    for label in ("score 1", "score 2", "score 3", "score 4", "score 5"):
        assert label in text
