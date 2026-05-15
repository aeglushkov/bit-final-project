"""Pattern-mine the planner's Thought text on rows that mode-collapsed.

A mode-collapse row is one classified as MODE_COLLAPSE or
EARLY_STOP_AT_LOCALIZE by `classify_agent_failures.py`. We pull the FIRST
trajectory step (or the only one), inspect its `thought` field, and try
to bucket what the planner said before short-circuiting:

    SHORTCUT_CAN_ANSWER   — thought contains "I can answer directly",
                             "Based on the options/question",
                             "I already know", or similar
    SHORTCUT_INSUFFICIENT — thought says the info is insufficient or
                             that no relevant frames/objects exist
    PLAN_BUT_QUIT         — thought lays out a multi-step plan
                             ("first find frames, then ask") but stops
    NO_THOUGHT            — empty / missing thought; planner emitted
                             Final Answer with no reasoning
    OTHER                 — none of the above

Also flags whether the Final Answer is bare (single letter / short
number) vs verbose, and emits the IDs to `mode_collapse_ids.txt` for
the M4 experiment.

Usage:
    python analyze_mode_collapse.py \\
        --traces      results/subset_si_with_traces.jsonl \\
        --classifier  results/agent_failures_si.json \\
        --out-ids     results/mode_collapse_ids.txt \\
        --out-json    results/mode_collapse_thoughts.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


_PATTERNS = [
    ("SHORTCUT_CAN_ANSWER", re.compile(
        r"\b(I can answer|I already know|I will provide|based on the (options|choices|question)"
        r"|without (further|additional) (info|information|tools|frames|inspection)"
        r"|the answer is clear|directly answer)",
        re.IGNORECASE,
    )),
    ("SHORTCUT_INSUFFICIENT", re.compile(
        r"\b(insufficient|not enough info|no (relevant|matching) (frames?|objects?)"
        r"|cannot (determine|find)|unable to (determine|find)|nothing relevant)",
        re.IGNORECASE,
    )),
    ("PLAN_BUT_QUIT", re.compile(
        r"\b(first[, ]+I (will|need|should)|then.*|next.*|step \d+|I will (use|call|invoke)"
        r"|to (answer|determine|find) (the|this).*?I (will|need to|should) (first|use))",
        re.IGNORECASE,
    )),
]
_BARE_LETTER_RE = re.compile(r"^[A-D]\.?$")
_BARE_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _classify_thought(thought: str) -> str:
    if not thought or not thought.strip():
        return "NO_THOUGHT"
    for name, pat in _PATTERNS:
        if pat.search(thought):
            return name
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, type=Path)
    ap.add_argument("--classifier", required=True, type=Path,
                    help="Output JSON from classify_agent_failures.py")
    ap.add_argument("--out-ids", required=True, type=Path,
                    help="One row id per line for the M4 force-tool-use re-run")
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    buckets_of_interest = {"MODE_COLLAPSE", "EARLY_STOP_AT_LOCALIZE"}

    fail = json.loads(args.classifier.read_text())
    ids_in_scope = {r["id"] for r in fail["rows"] if r["bucket"] in buckets_of_interest}

    rows = []
    for line in args.traces.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] not in ids_in_scope:
            continue
        traj = r.get("trajectory") or []
        first_thought = traj[0]["thought"] if traj else ""
        pattern = _classify_thought(first_thought)
        pred = (r.get("prediction") or "").strip()
        bare = bool(_BARE_LETTER_RE.match(pred) or _BARE_NUMBER_RE.match(pred))
        rows.append({
            "id": r["id"],
            "scene_name": r["scene_name"],
            "question_type": r["question_type"],
            "n_steps": len(traj),
            "first_tool": traj[0]["tool"] if traj else None,
            "first_thought": first_thought,
            "ground_truth": r.get("ground_truth", ""),
            "prediction": pred,
            "score": r.get("score", 0.0),
            "pattern": pattern,
            "bare_answer": bare,
        })

    # Histograms
    overall = Counter(r["pattern"] for r in rows)
    by_qtype = defaultdict(Counter)
    for r in rows:
        by_qtype[r["question_type"]][r["pattern"]] += 1

    # Verbatim samples — up to 3 per pattern
    samples_per_pattern: dict[str, list] = defaultdict(list)
    for r in rows:
        if len(samples_per_pattern[r["pattern"]]) < 3:
            samples_per_pattern[r["pattern"]].append({
                "id": r["id"],
                "question_type": r["question_type"],
                "first_thought": r["first_thought"],
                "first_tool": r["first_tool"],
                "n_steps": r["n_steps"],
                "ground_truth": r["ground_truth"],
                "prediction": r["prediction"],
            })

    args.out_json.write_text(json.dumps({
        "n_rows": len(rows),
        "overall": dict(overall),
        "by_question_type": {k: dict(v) for k, v in by_qtype.items()},
        "samples": dict(samples_per_pattern),
        "rows": rows,
    }, indent=2, default=str))
    args.out_ids.write_text("\n".join(str(r["id"]) for r in rows) + "\n")

    print(f"wrote {args.out_ids} ({len(rows)} ids)")
    print(f"wrote {args.out_json}")
    print("\n=== Overall pattern histogram ===")
    for k, v in overall.most_common():
        print(f"  {k:<24s} {v}")
    print("\n=== Per question_type ===")
    for qtype in sorted(by_qtype):
        cnt = by_qtype[qtype]
        n = sum(cnt.values())
        print(f"  {qtype} (n={n})")
        for k, v in cnt.most_common():
            print(f"    {k:<22s} {v}")


if __name__ == "__main__":
    main()
