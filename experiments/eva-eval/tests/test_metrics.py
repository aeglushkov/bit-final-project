import math

import pytest

from eva_eval.eval.metrics import (
    MCA_QUESTION_TYPES,
    NA_QUESTION_TYPES,
    aggregate,
    exact_match,
    fuzzy_matching,
    mean_relative_accuracy,
    score_one,
    to_float,
)


def test_fuzzy_matching_strips_period_and_takes_first_token():
    assert fuzzy_matching("A.") == "A"
    assert fuzzy_matching("12.5 meters") == "12.5"
    assert fuzzy_matching("Yes") == "Yes"
    # upstream quirk: leading whitespace is NOT stripped before split, so the
    # first token becomes empty. We preserve this behavior for reproducibility.
    assert fuzzy_matching("  Yes ") == ""


def test_exact_match_case_insensitive():
    assert exact_match("a", "A") == 1.0
    assert exact_match("B", "C") == 0.0


def test_to_float_handles_garbage():
    assert to_float("3.14") == 3.14
    assert to_float("abc") is None
    assert to_float(None) is None
    assert to_float(2) == 2.0


def test_mra_perfect_match():
    assert mean_relative_accuracy(10.0, 10.0) == 1.0


def test_mra_far_off_returns_zero():
    assert mean_relative_accuracy(100.0, 1.0) == 0.0


def test_mra_mid_band():
    score = mean_relative_accuracy(1.2, 1.0)
    assert 0.0 < score < 1.0


def test_mra_handles_zero_target():
    assert mean_relative_accuracy(0.0, 0.0) == 1.0
    assert mean_relative_accuracy(1.0, 0.0) == 0.0


def test_score_one_mca_correct_letter():
    assert score_one("object_rel_distance", "B.", "B") == 1.0
    assert score_one("object_rel_distance", "A. The chair", "B") == 0.0


def test_score_one_na_perfect_then_off():
    assert score_one("object_counting", "5", 5) == 1.0
    assert score_one("object_counting", "abc", 5) == 0.0


def test_score_one_handles_unparseable_na_gt():
    assert score_one("object_counting", "5", "five") == 0.0


def test_score_one_unknown_type_raises():
    with pytest.raises(ValueError):
        score_one("not_a_real_type", "x", "y")


def test_aggregate_overall_and_per_type():
    rows = [
        {"question_type": "object_counting", "score": 1.0},
        {"question_type": "object_counting", "score": 0.0},
        {"question_type": "object_rel_distance", "score": 1.0},
    ]
    out = aggregate(rows)
    assert out["n_questions"] == 3
    assert math.isclose(out["object_counting"], 50.0)
    assert math.isclose(out["object_rel_distance"], 100.0)
    assert math.isclose(out["overall"], (0.5 + 1.0) / 2 * 100.0)


def test_aggregate_rolls_up_relative_direction():
    rows = [
        {"question_type": "object_rel_direction_easy", "score": 1.0},
        {"question_type": "object_rel_direction_medium", "score": 0.5},
        {"question_type": "object_rel_direction_hard", "score": 0.0},
    ]
    out = aggregate(rows)
    assert "object_rel_direction" in out
    assert math.isclose(out["object_rel_direction"], 0.5 * 100.0)
    assert "object_rel_direction_easy" not in out


def test_aggregate_empty():
    out = aggregate([])
    assert out["overall"] == 0.0
    assert out["n_questions"] == 0


def test_question_type_constants_disjoint():
    assert set(MCA_QUESTION_TYPES).isdisjoint(NA_QUESTION_TYPES)
