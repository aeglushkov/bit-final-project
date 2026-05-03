"""VSI-Bench scoring + aggregation. Ported from
literature/thinking-in-space/code/lmms_eval/tasks/vsibench/utils.py
(re-implemented with stdlib only — no pandas)."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Iterable

import numpy as np

MCA_QUESTION_TYPES = (
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
)
NA_QUESTION_TYPES = (
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
)

REPORTED_TASK_ORDER = (
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
    "object_rel_distance",
    "object_rel_direction",
    "route_planning",
    "obj_appearance_order",
)


def fuzzy_matching(pred: str) -> str:
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred: str, target: str) -> float:
    return 1.0 if str(pred).lower() == str(target).lower() else 0.0


def to_float(pred) -> float | None:
    try:
        return float(pred)
    except (TypeError, ValueError):
        return None


def mean_relative_accuracy(pred: float, target: float, start: float = 0.5, end: float = 0.95, interval: float = 0.05) -> float:
    if target == 0:
        return 1.0 if pred == 0 else 0.0
    num_pts = int(round((end - start) / interval)) + 2
    conf_intervs = np.linspace(start, end, num_pts)
    norm_err = abs(pred - target) / abs(target)
    return float((norm_err <= (1 - conf_intervs)).mean())


def score_one(question_type: str, prediction: str, ground_truth) -> float:
    if question_type in MCA_QUESTION_TYPES:
        return exact_match(fuzzy_matching(str(prediction)), str(ground_truth))
    if question_type in NA_QUESTION_TYPES:
        p = to_float(fuzzy_matching(str(prediction)))
        t = to_float(ground_truth)
        if p is None or t is None:
            return 0.0
        try:
            return mean_relative_accuracy(p, t)
        except (ZeroDivisionError, ValueError):
            return 0.0
    raise ValueError(f"Unknown question_type: {question_type}")


def aggregate(scored: Iterable[dict]) -> "OrderedDict[str, float]":
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        by_type[r["question_type"]].append(float(r["score"]))

    raw: dict[str, float] = {}
    for qt, scores in by_type.items():
        raw[qt] = sum(scores) / len(scores) if scores else 0.0

    rd_keys = [k for k in list(raw) if k.startswith("object_rel_direction_")]
    if rd_keys:
        raw["object_rel_direction"] = sum(raw[k] for k in rd_keys) / len(rd_keys)
        for k in rd_keys:
            del raw[k]

    out: "OrderedDict[str, float]" = OrderedDict()
    out["overall"] = (sum(raw.values()) / len(raw) * 100.0) if raw else 0.0
    for qt in REPORTED_TASK_ORDER:
        if qt in raw:
            out[qt] = raw[qt] * 100.0
    out["n_questions"] = sum(len(s) for s in by_type.values())
    return out
