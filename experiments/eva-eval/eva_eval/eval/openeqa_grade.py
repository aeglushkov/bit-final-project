"""LLM-as-judge grader for OpenEQA predictions.

Default judge: any text-only model from config/models.yaml. Apples-to-apples
with the OpenEQA paper requires GPT-4-class judging — use --judge gpt-4o
in scripts/08_grade_openeqa.py for that.
"""
from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from typing import Iterable


JUDGE_PROMPT_TEMPLATE = (
    "You are an AI assistant who will help me evaluate the response given the "
    "question and the correct answer.\n"
    "To mark a response, you should output a single integer between 1 and 5 "
    "(including 1 and 5).\n"
    "5 means that the response perfectly matches the answer.\n"
    "4 means that the response is mostly correct but missing minor details.\n"
    "3 means that the response is partially correct.\n"
    "2 means that the response is mostly incorrect but has some relation to the answer.\n"
    "1 means that the response is completely incorrect.\n"
    "Question: {question}\n"
    "Correct Answer: {answer}\n"
    "Response: {response}\n"
    "Output a single integer:"
)


def build_judge_prompt(*, question: str, gold_answer: str, prediction: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question.strip(),
        answer=gold_answer.strip(),
        response=prediction.strip(),
    )


def parse_judge_score(text: str) -> int | None:
    """Extract a 1–5 score from a judge response. Searches from the end of the
    text first (judges often output reasoning then the final score), and only
    accepts a single digit 1–5 with word boundaries so '1-5 scale' doesn't
    misparse as 1. Returns None if no valid score is present."""
    if not text:
        return None
    # Search from end first: judges often write "...therefore I give a 3"
    matches = list(re.finditer(r"\b[1-5]\b", text))
    if matches:
        return int(matches[-1].group())
    # Fallback: any integer (clamps to range). Preserves backward compat with
    # judges that output a bare number outside 1-5.
    fallback = re.search(r"-?\d+", text)
    if fallback is None:
        return None
    n = int(fallback.group())
    if n < 1:
        return 1
    if n > 5:
        return 5
    return n


def c_score(score_1_to_5: float | None) -> float | None:
    """100 * (s - 1) / 4: maps 1→0, 5→100. Returns None if input is None.
    Accepts a float so per-category mean scores (always non-integer) work."""
    if score_1_to_5 is None:
        return None
    return 100.0 * (float(score_1_to_5) - 1.0) / 4.0


def grade_one(judge, *, question: str, gold_answer: str, prediction: str) -> tuple[int | None, str]:
    """Call the judge model on one (question, gold, prediction) triple. Returns
    (1-5 score or None, raw judge response text)."""
    prompt = build_judge_prompt(question=question, gold_answer=gold_answer, prediction=prediction)
    raw = judge.chat([{"role": "user", "content": prompt}])
    return parse_judge_score(raw), raw


def aggregate(rows: Iterable[dict]) -> dict:
    """Compute overall + per-category C-scores from graded rows.

    Each row must contain `category` and `score` (1-5 or None). Rows with
    None scores are excluded from category and overall means."""
    by_cat: dict[str, list[int]] = defaultdict(list)
    all_scores: list[int] = []
    for r in rows:
        s = r.get("score")
        if s is None:
            continue
        by_cat[r.get("category", "?")].append(int(s))
        all_scores.append(int(s))

    per_category: "OrderedDict[str, float]" = OrderedDict()
    for cat in sorted(by_cat):
        mean_s = sum(by_cat[cat]) / len(by_cat[cat])
        per_category[cat] = float(c_score(mean_s))

    overall = float(c_score(sum(all_scores) / len(all_scores))) if all_scores else 0.0
    return {
        "overall": overall,
        "per_category": per_category,
        "n_questions": sum(len(v) for v in by_cat.values()),
    }
