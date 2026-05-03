import pytest

from eva_eval.eval.vsibench import format_question, parse_final_answer


def test_format_mca_includes_options_and_letter_instruction():
    doc = {
        "question_type": "object_rel_distance",
        "question": "Which object is closest?",
        "options": ["A. chair", "B. table", "C. lamp"],
    }
    text = format_question(doc)
    assert "A. chair" in text
    assert "Options:" in text
    assert "letter" in text


def test_format_na_includes_single_word_instruction():
    doc = {"question_type": "object_counting", "question": "How many chairs?"}
    text = format_question(doc)
    assert "How many chairs?" in text
    assert "single word or phrase" in text


def test_format_unknown_type_raises():
    with pytest.raises(ValueError):
        format_question({"question_type": "made_up", "question": "?"})


def test_parse_final_answer_strips_prefix_from_dict():
    assert parse_final_answer({"output": "Final Answer: B"}) == "B"
    assert parse_final_answer({"output": "FINAL ANSWER:   1.5"}) == "1.5"


def test_parse_final_answer_passes_plain_text():
    assert parse_final_answer({"output": "B"}) == "B"
    assert parse_final_answer("12") == "12"


def test_parse_final_answer_handles_empty():
    assert parse_final_answer({"output": ""}) == ""
    assert parse_final_answer({}) == ""
