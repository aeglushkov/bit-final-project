"""Extract target nouns from every VSI-Bench question and report YOLO coverage.

For each question in the VSI-Bench test split, apply a task-type-specific
regex to pull the noun(s) the question is *about*:

    object_counting          "How many X(s) are in this room?"             -> [X]
    object_abs_distance      "...the distance between the X and the Y..."   -> [X, Y]
    object_size_estimation   "...longest dimension... of the X..."           -> [X]
    object_rel_direction_*   "...standing by the X and facing the Y, is the Z..." -> [X, Y, Z]
    object_rel_distance      "Which of the following is the closest to the X?" + options -> [X, *options]
    obj_appearance_order     options are 4 lists like "A. chair, door, ..."  -> all listed nouns
    room_size_estimation     (no target noun)                                -> []
    route_planning           options enumerate movements; no target noun     -> []

Each noun is then cross-checked against the YOLO-World class list
(`detection_classes.customized_classes` from the paper code). Output:

    - results/vsibench_target_nouns.csv
      columns: question_type, target_noun, n_questions, in_yolo

    - results/vsibench_target_nouns_summary.json
      per-question-type {in_yolo, missing, sample_misses}

Usage:
    python extract_vsibench_target_nouns.py \\
        --paper-code-dir literature/EmbodiedVideoAgent/code \\
        --out-csv  results/vsibench_target_nouns.csv \\
        --out-json results/vsibench_target_nouns_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# Per-task regex extractors. The "X" in `(X)?` patterns gracefully handles
# minor punctuation drift in the dataset (some questions end with "?", some not).

_RE_COUNTING = re.compile(r"how many\s+([a-z][a-z\- ]+?)\s*\(s\)", re.IGNORECASE)
_RE_COUNTING_FALLBACK = re.compile(r"how many\s+([a-z][a-z\- ]+?)\s+(?:are|is)\s+in", re.IGNORECASE)
_RE_DISTANCE = re.compile(
    r"distance between the\s+([a-z][a-z\- ]+?)\s+and the\s+([a-z][a-z\- ]+?)\s*(?:,|\(|in|$)",
    re.IGNORECASE,
)
_RE_SIZE = re.compile(
    r"(?:length of the longest dimension|longest dimension|length|width|height)\s+\((?:length, width, or height\s+)?(?:length|width|height)?\)?\s*of the\s+([a-z][a-z\- ]+?)\s*(?:,|\(|in|measured|$)",
    re.IGNORECASE,
)
_RE_SIZE_FALLBACK = re.compile(
    r"dimension.*?of the\s+([a-z][a-z\- ]+?)\s*(?:,|\(|in|measured|$)",
    re.IGNORECASE,
)
_RE_DIRECTION = re.compile(
    r"standing by the\s+([a-z][a-z\- ]+?)\s+and facing the\s+([a-z][a-z\- ]+?)\s*,\s*"
    r"(?:is|where is|where's)\s+the\s+([a-z][a-z\- ]+?)\s+(?:to|on|located|positioned|\?|in)",
    re.IGNORECASE,
)
_RE_REL_DISTANCE = re.compile(
    r"closest to the\s+([a-z][a-z\- ]+?)\s*(?:\?|,|$)",
    re.IGNORECASE,
)
_RE_OPTION_PREFIX = re.compile(r"^[A-D]\.?\s*", re.IGNORECASE)


def _normalize(noun: str) -> str:
    """Lowercase + strip trailing punctuation + collapse whitespace."""
    n = noun.strip().lower().rstrip(",.;: ")
    return " ".join(n.split())


def _option_nouns(options: list[str]) -> list[str]:
    """For tasks where each option is an object name (rel_distance,
    appearance_order), split the option text into nouns. appearance_order
    options look like 'chair, door, printer, refrigerator' so we split on
    commas."""
    out = []
    for opt in options or []:
        s = _RE_OPTION_PREFIX.sub("", str(opt).strip())
        for part in s.split(","):
            n = _normalize(part)
            if n:
                out.append(n)
    return out


def _extract(qtype: str, question: str, options: list[str]) -> list[str]:
    q = question or ""
    nouns: list[str] = []
    if qtype == "object_counting":
        m = _RE_COUNTING.search(q) or _RE_COUNTING_FALLBACK.search(q)
        if m:
            nouns.append(_normalize(m.group(1)))
    elif qtype == "object_abs_distance":
        m = _RE_DISTANCE.search(q)
        if m:
            nouns.extend(_normalize(g) for g in m.groups())
    elif qtype == "object_size_estimation":
        m = _RE_SIZE.search(q) or _RE_SIZE_FALLBACK.search(q)
        if m:
            nouns.append(_normalize(m.group(1)))
    elif qtype.startswith("object_rel_direction"):
        m = _RE_DIRECTION.search(q)
        if m:
            nouns.extend(_normalize(g) for g in m.groups())
    elif qtype == "object_rel_distance":
        m = _RE_REL_DISTANCE.search(q)
        if m:
            nouns.append(_normalize(m.group(1)))
        # Options are objects too — useful to know if the candidates are
        # in YOLO even when the anchor noun isn't.
        nouns.extend(_option_nouns(options))
    elif qtype == "obj_appearance_order":
        # Each option enumerates an ordering of the same 4 objects.
        nouns.extend(_option_nouns(options))
    # room_size_estimation, route_planning: no concrete target noun.
    return list(dict.fromkeys(n for n in nouns if n))  # dedupe, preserve order


def _load_yolo_classes(classes_path: Path) -> set[str]:
    """Load YOLO-World class list from a plain-text file (one class per line).

    The agent runs against the *runtime* class list at
    experiments/eva-eval/config/detection_classes.txt (162 entries), NOT the
    `customized_classes` list bundled in literature/.../detection_classes.py
    (60 entries — historical default that's been superseded).
    """
    out: set[str] = set()
    for line in classes_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.lower())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes-file", required=True, type=Path,
                    help="Path to runtime YOLO classes (one per line), e.g. "
                         "experiments/eva-eval/config/detection_classes.txt.")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    yolo = _load_yolo_classes(args.classes_file)

    from datasets import load_dataset
    ds = load_dataset("nyu-visionx/VSI-Bench", split="test")

    rows: list[tuple[str, str, bool]] = []  # (qtype, noun, in_yolo)
    parse_fail_by_type: Counter = Counter()

    for doc in ds:
        qtype = doc["question_type"]
        nouns = _extract(qtype, doc.get("question", ""), doc.get("options") or [])
        if not nouns and qtype not in {"room_size_estimation", "route_planning"}:
            parse_fail_by_type[qtype] += 1
            continue
        for n in nouns:
            # Treat 'X' and 'Xs' equivalently when checking YOLO membership.
            singular = n.rstrip("s")
            in_yolo = (n in yolo) or (singular in yolo)
            rows.append((qtype, n, in_yolo))

    # Per (qtype, noun): count, in_yolo
    counter: Counter = Counter()
    in_yolo_for_noun: dict[tuple[str, str], bool] = {}
    for qtype, noun, in_yolo in rows:
        key = (qtype, noun)
        counter[key] += 1
        in_yolo_for_noun[key] = in_yolo

    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_type", "target_noun", "n_questions", "in_yolo"])
        for (qtype, noun), n in counter.most_common():
            w.writerow([qtype, noun, n, "1" if in_yolo_for_noun[(qtype, noun)] else "0"])

    # Per-task summary
    by_qtype: dict[str, dict] = defaultdict(lambda: {
        "n_questions_parsed": 0,
        "n_questions_all_in_yolo": 0,
        "n_questions_any_missing": 0,
        "missing_noun_counter": Counter(),
        "covered_noun_counter": Counter(),
    })
    # Re-walk the dataset to score per-question rather than per-noun
    for doc in ds:
        qtype = doc["question_type"]
        nouns = _extract(qtype, doc.get("question", ""), doc.get("options") or [])
        if qtype in {"room_size_estimation", "route_planning"}:
            continue
        if not nouns:
            continue
        by_qtype[qtype]["n_questions_parsed"] += 1
        any_missing = False
        for n in nouns:
            singular = n.rstrip("s")
            in_yolo = (n in yolo) or (singular in yolo)
            if in_yolo:
                by_qtype[qtype]["covered_noun_counter"][n] += 1
            else:
                by_qtype[qtype]["missing_noun_counter"][n] += 1
                any_missing = True
        if not any_missing:
            by_qtype[qtype]["n_questions_all_in_yolo"] += 1
        else:
            by_qtype[qtype]["n_questions_any_missing"] += 1

    summary = {}
    for qtype, info in by_qtype.items():
        n_parsed = info["n_questions_parsed"]
        summary[qtype] = {
            "n_parsed": n_parsed,
            "n_all_in_yolo": info["n_questions_all_in_yolo"],
            "n_any_missing": info["n_questions_any_missing"],
            "coverage_pct": round(100.0 * info["n_questions_all_in_yolo"] / n_parsed, 1) if n_parsed else 0.0,
            "top_missing": dict(info["missing_noun_counter"].most_common(10)),
            "top_covered": dict(info["covered_noun_counter"].most_common(10)),
        }
    summary["_parse_fail_by_type"] = dict(parse_fail_by_type)
    summary["_yolo_class_count"] = len(yolo)

    args.out_json.write_text(json.dumps(summary, indent=2, default=str))

    print(f"wrote {args.out_csv} ({sum(counter.values())} noun occurrences)")
    print(f"wrote {args.out_json}")
    print(f"\nYOLO class count: {len(yolo)}")
    print(f"\n=== Coverage by question_type ===")
    print(f"{'qtype':<28} {'n_parsed':>9} {'all_in_yolo':>13} {'any_missing':>13} {'cov%':>7}")
    for qtype in sorted(by_qtype):
        s = summary[qtype]
        print(f"{qtype:<28} {s['n_parsed']:>9} {s['n_all_in_yolo']:>13} {s['n_any_missing']:>13} {s['coverage_pct']:>6.1f}%")

    if parse_fail_by_type:
        print(f"\n=== Parse failures (no nouns extracted) ===")
        for qtype, n in parse_fail_by_type.most_common():
            print(f"  {qtype:<28} {n}")


if __name__ == "__main__":
    main()
