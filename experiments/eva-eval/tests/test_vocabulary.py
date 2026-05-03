import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_vocabulary.py"
_spec = importlib.util.spec_from_file_location("build_vocabulary", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["build_vocabulary"] = _mod
_spec.loader.exec_module(_mod)
extract_candidates = _mod.extract_candidates


def test_drops_stopwords():
    cands = extract_candidates("The chair is in the room")
    assert "chair" in cands
    assert "the" not in cands
    assert "room" not in cands


def test_lowercases():
    assert "laptop" in extract_candidates("Where is the LAPTOP?")


def test_min_length_three():
    cands = extract_candidates("a bb ccc dddd")
    assert "bb" not in cands
    assert "ccc" in cands
    assert "dddd" in cands


def test_strips_punctuation_and_digits():
    cands = extract_candidates("How many cups (3.5L) are on the table?")
    assert "cups" in cands
    assert "table" in cands
    assert "3.5l" not in cands


def test_drops_question_scaffolding():
    cands = extract_candidates("What is the absolute distance from the laptop to the printer?")
    assert "laptop" in cands
    assert "printer" in cands
    assert "distance" not in cands
    assert "absolute" not in cands
