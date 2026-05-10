from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_render_trace_markdown_includes_question_steps_answer(tmp_path):
    from eva_eval.debug.agent_trace import render_trace_markdown

    row = {
        "id": "q1",
        "category": "object_recognition",
        "question": "How many chairs?",
        "ground_truth": "Three.",
        "prediction": "3",
        "score": 4,
        "judge_rationale": "Mostly correct, missing units.",
        "intermediate_steps": [
            {"tool": "frame_localization", "tool_input": "chair", "log": "Thought: I should localize chairs.\nAction: frame_localization\nAction Input: \"chair\"", "observation": "[1, 5, 9]"},
            {"tool": "frame_VQA", "tool_input": "(\"how many chairs?\", 5)", "log": "Thought: ask the VLM.", "observation": "Three chairs visible."},
        ],
    }
    md = render_trace_markdown(row)
    assert "How many chairs?" in md
    assert "Three." in md
    assert "frame_localization" in md
    assert "frame_VQA" in md
    assert "## Step 1" in md
    assert "## Step 2" in md
    assert "score: 4" in md.lower()


def test_render_trace_markdown_handles_empty_steps():
    from eva_eval.debug.agent_trace import render_trace_markdown

    row = {
        "id": "q2",
        "category": "x",
        "question": "?",
        "ground_truth": "?",
        "prediction": "",
        "intermediate_steps": [],
    }
    md = render_trace_markdown(row)
    assert "(no intermediate steps recorded)" in md


def test_find_question_in_jsonl(tmp_path):
    from eva_eval.debug.agent_trace import find_question

    p = tmp_path / "graded.jsonl"
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert find_question(p, "b") == {"id": "b"}
    with pytest.raises(KeyError):
        find_question(p, "nope")
