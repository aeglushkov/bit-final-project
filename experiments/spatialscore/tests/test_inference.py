"""Tests for the main inference script result format and answer extraction."""

import pytest
import sys
import os

SPATIALSCORE_CODE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "literature", "spatialscore", "code"
)
sys.path.insert(0, SPATIALSCORE_CODE)


class TestResultFormat:
    """Verify output JSON matches the baseline format from test_qwen.py."""

    REQUIRED_FIELDS = {
        "id", "category", "subcategory", "input_modality",
        "question_type", "source", "question", "gt_answer",
        "pred_answer", "img_paths", "is_correct", "score",
    }

    def test_result_entry_has_all_fields(self):
        """A result entry from the agent should have the same fields as baseline."""
        from run_agent import build_result_entry

        entry = build_result_entry(
            item={
                "id": 0, "category": "Others", "subcategory": "visual",
                "input_modality": "single-image", "question_type": "multi-choice",
                "source": "MMVP", "question": "Q?", "answer": "A",
                "img_paths": ["/tmp/img.jpg"],
            },
            pred_answer="A",
            is_correct=True,
            score=1.0,
        )

        assert set(entry.keys()) == self.REQUIRED_FIELDS

    def test_result_entry_values(self):
        from run_agent import build_result_entry

        entry = build_result_entry(
            item={
                "id": 5, "category": "Counting", "subcategory": "counting",
                "input_modality": "single-image", "question_type": "multi-choice",
                "source": "BLINK", "question": "How many?", "answer": "B",
                "img_paths": ["/tmp/img.jpg"],
            },
            pred_answer="C",
            is_correct=False,
            score=0.0,
        )

        assert entry["id"] == 5
        assert entry["gt_answer"] == "B"
        assert entry["pred_answer"] == "C"
        assert entry["is_correct"] is False
        assert entry["score"] == 0.0


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestAnswerExtraction:
    """Test answer extraction from agent output. Requires torch (utils.util depends on it)."""

    def test_extract_option_from_answer_tags(self):
        from utils.util import extract_option
        assert extract_option("<answer>B</answer>") == "B"

    def test_extract_option_from_parentheses(self):
        from utils.util import extract_option
        assert extract_option("(A)") == "A"

    def test_extract_option_plain_letter(self):
        from utils.util import extract_option
        result = extract_option("A")
        assert result == "A"

    def test_extract_yes_no(self):
        from utils.util import extract_yes_no
        assert extract_yes_no("Yes, this is correct").lower() == "yes"
        assert extract_yes_no("No, it is not").lower() == "no"
