import pytest

from eva_eval.agent.tools import _format_dict_as_text, parse_tuple_input


def test_parse_two_args_paper_format():
    q, fid = parse_tuple_input('("What is the color of the blinds?", 190)', expected_arity=2)
    assert q == "What is the color of the blinds?"
    assert fid == 190


def test_parse_two_args_with_single_quotes():
    q, oid = parse_tuple_input("('Is this a chair?', 7)", expected_arity=2)
    assert q == "Is this a chair?"
    assert oid == 7


def test_parse_one_arg_quoted():
    (text,) = parse_tuple_input('"brown chair"', expected_arity=1)
    assert text == "brown chair"


def test_parse_one_arg_bare_string():
    (text,) = parse_tuple_input("brown chair", expected_arity=1)
    assert text == "brown chair"


def test_wrong_arity_two_raises():
    with pytest.raises(ValueError):
        parse_tuple_input('("just one arg",)', expected_arity=2)


def test_format_dict_collapses_multiline_captions():
    d = {2: "The object is\na green cup.", 9: "A bottle"}
    out = _format_dict_as_text(d)
    assert "\n" not in out
    assert "2:" in out
    assert "9:" in out


def test_format_dict_empty():
    assert _format_dict_as_text({}) == "(no matches)"
