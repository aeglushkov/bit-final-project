from __future__ import annotations

import pytest


def test_judge_prompt_includes_question_answer_response():
    from eva_eval.eval.openeqa_grade import build_judge_prompt

    prompt = build_judge_prompt(
        question="How many chairs are there?",
        gold_answer="There are three chairs.",
        prediction="3",
    )
    assert "How many chairs are there?" in prompt
    assert "There are three chairs." in prompt
    assert "3" in prompt
    assert "single integer" in prompt.lower()


def test_parse_judge_score_extracts_first_integer():
    from eva_eval.eval.openeqa_grade import parse_judge_score

    assert parse_judge_score("5") == 5
    assert parse_judge_score("4\nbecause...") == 4
    assert parse_judge_score("Score: 3 (the response is partially correct)") == 3


def test_parse_judge_score_clamps_out_of_range():
    from eva_eval.eval.openeqa_grade import parse_judge_score

    assert parse_judge_score("7") == 5
    assert parse_judge_score("0") == 1


def test_parse_judge_score_returns_none_when_no_integer():
    from eva_eval.eval.openeqa_grade import parse_judge_score

    assert parse_judge_score("no idea") is None
    assert parse_judge_score("") is None


def test_c_score_normalizes_one_to_zero_and_five_to_hundred():
    from eva_eval.eval.openeqa_grade import c_score

    assert c_score(1) == 0.0
    assert c_score(5) == 100.0
    assert c_score(3) == 50.0


def test_c_score_returns_none_for_none():
    from eva_eval.eval.openeqa_grade import c_score

    assert c_score(None) is None


def test_aggregate_overall_and_per_category():
    import math
    from eva_eval.eval.openeqa_grade import aggregate

    rows = [
        {"category": "object_recognition", "score": 5},
        {"category": "object_recognition", "score": 3},
        {"category": "spatial_reasoning", "score": 1},
    ]
    out = aggregate(rows)
    assert out["n_questions"] == 3
    # object_recognition mean = 4 → c_score 75
    assert math.isclose(out["per_category"]["object_recognition"], 75.0)
    # spatial_reasoning mean = 1 → c_score 0
    assert math.isclose(out["per_category"]["spatial_reasoning"], 0.0)
    # overall mean across all rows = (5+3+1)/3 = 3 → c_score 50
    assert math.isclose(out["overall"], 50.0)
