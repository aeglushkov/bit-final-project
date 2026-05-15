"""Classify each agent JSONL row into a failure-mode bucket using its trajectory.

Buckets:
    ITER_LIMIT          — `error` contains "iteration limit", or last step note matches.
    MODE_COLLAPSE       — ≤1 tool call AND prediction is a bare letter / round number;
                          the planner shortcut to an answer without using tools.
    EARLY_STOP_AT_LOCALIZE — trajectory has only `frame_localization` calls; planner
                          received frame indices but never opened any frame.
    FRAME_SINGLE_INSUFFICIENT — trajectory has `frame_VQA` / `object_VQA` calls but no
                          tool that aggregates across frames; question_type is one
                          that needs multi-frame integration
                          (obj_appearance_order, room_size_estimation).
    YOLO_RETRIEVAL_EMPTY — at least one `retrieve_objects_by_*` returned empty / "no
                          objects" / "could not find" text.
    COUNT_SQL_EXACT_MATCH — question is object_counting AND a `query_db` call uses
                          a `category = '...'` exact-match clause.
    PLANNER_IGNORED_TOOL  — final prediction contradicts the last frame_VQA / object_VQA
                          observation (the tool produced a useful answer, the planner
                          chose differently).
    CORRECT               — score >= 0.5 (informational; not a failure).
    OTHER                 — none of the above.

Output: JSON with per-row classification + per-question_type histograms.

Usage:
    python classify_agent_failures.py \\
        --traces  results/subset_si_with_traces.jsonl \\
        --out     results/agent_failures_si.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_MULTI_FRAME_QTYPES = {"obj_appearance_order", "room_size_estimation"}
_SINGLE_FRAME_TOOLS = {"frame_VQA", "object_VQA"}
_RETRIEVAL_TOOLS = {"retrieve_objects_by_appearance", "retrieve_objects_by_environment"}
_EMPTY_OBS_PATTERNS = [
    r"no objects? satisf",
    r"could not find",
    r"no relevant",
    r"are an empty",
    r"\{\s*\}",
    r"is empty",
]
_SQL_EXACT_MATCH_RE = re.compile(r"category\s*=\s*['\"]", re.IGNORECASE)
_BARE_LETTER_RE = re.compile(r"^[A-D]\.?$")
_ROUND_NUMBER_RE = re.compile(r"^\d{1,2}(\.0+)?$")


def _classify_row(r: dict[str, Any]) -> str:
    err = r.get("error") or ""
    pred = (r.get("prediction") or "").strip()
    score = float(r.get("score", 0.0))
    qtype = r.get("question_type", "")
    trajectory = r.get("trajectory") or []
    tools_called = [s.get("tool", "") for s in trajectory]
    n_steps = len(trajectory)

    # Iteration / time limit explicit
    if "iteration limit" in err.lower() or "iteration limit" in pred.lower():
        return "ITER_LIMIT"

    # Correct (not a failure but tag it for context)
    if score >= 0.5:
        return "CORRECT"

    # Planner shortcut: very few tool calls + a bare-letter or round-number answer
    if n_steps <= 1 and (_BARE_LETTER_RE.match(pred) or _ROUND_NUMBER_RE.match(pred)):
        return "MODE_COLLAPSE"

    # Early stop: only frame_localization, never opened a frame
    if tools_called and all(t == "frame_localization" for t in tools_called):
        return "EARLY_STOP_AT_LOCALIZE"

    # Counting + SQL exact-match clause
    if qtype == "object_counting":
        for step in trajectory:
            if step.get("tool") == "query_db" and _SQL_EXACT_MATCH_RE.search(str(step.get("tool_input", ""))):
                return "COUNT_SQL_EXACT_MATCH"

    # Retrieval returned empty
    for step in trajectory:
        if step.get("tool") in _RETRIEVAL_TOOLS:
            obs = str(step.get("observation", ""))
            if any(re.search(p, obs, re.IGNORECASE) for p in _EMPTY_OBS_PATTERNS):
                return "YOLO_RETRIEVAL_EMPTY"

    # Single-frame restriction on a multi-frame question type
    if qtype in _MULTI_FRAME_QTYPES:
        vqa_tools = [t for t in tools_called if t in _SINGLE_FRAME_TOOLS]
        if vqa_tools and len(set(tools_called) & _SINGLE_FRAME_TOOLS) > 0:
            return "FRAME_SINGLE_INSUFFICIENT"

    # Planner ignored a useful tool observation: check whether the last
    # frame_VQA / object_VQA returned a number or letter and the prediction differs.
    for step in reversed(trajectory):
        if step.get("tool") in _SINGLE_FRAME_TOOLS:
            obs = str(step.get("observation", ""))
            # Look for a number or A/B/C/D mention in the observation.
            obs_letters = re.findall(r"\b[A-D]\b", obs)
            obs_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", obs)
            if (obs_letters and not any(l == pred for l in obs_letters)) or (
                obs_numbers and pred not in obs_numbers and qtype != "object_counting"
            ):
                return "PLANNER_IGNORED_TOOL"
            break

    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, type=Path,
                    help="JSONL with per-row `trajectory` field (must include error, prediction, score, question_type).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Path for the classification JSON output.")
    args = ap.parse_args()

    rows = []
    by_qtype: dict[str, Counter] = defaultdict(Counter)
    overall = Counter()

    for line in args.traces.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        bucket = _classify_row(r)
        rows.append({
            "id": r["id"],
            "scene_name": r["scene_name"],
            "question_type": r["question_type"],
            "ground_truth": r.get("ground_truth", ""),
            "prediction": r.get("prediction", ""),
            "score": r.get("score", 0.0),
            "n_steps": len(r.get("trajectory") or []),
            "tools_called": [s.get("tool", "") for s in (r.get("trajectory") or [])],
            "bucket": bucket,
        })
        overall[bucket] += 1
        by_qtype[r["question_type"]][bucket] += 1

    out = {
        "rows": rows,
        "overall": dict(overall),
        "by_question_type": {k: dict(v) for k, v in by_qtype.items()},
    }
    args.out.write_text(json.dumps(out, indent=2, default=str))

    print(f"wrote {args.out}  total={len(rows)}")
    print("\n=== Overall ===")
    for bucket, n in overall.most_common():
        print(f"  {bucket:<28s} {n}")
    print("\n=== Per question_type ===")
    for qtype in sorted(by_qtype):
        cnt = by_qtype[qtype]
        n = sum(cnt.values())
        print(f"  {qtype} (n={n})")
        for bucket, k in cnt.most_common():
            print(f"    {bucket:<26s} {k}")


if __name__ == "__main__":
    main()
