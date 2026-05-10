"""Pick qualitative samples from a VSI-Bench eval JSONL.

For each reported question type, sample one row per bucket:
    - MCA types  : `good` (score == 1.0) and `bad` (score == 0.0)
    - NA types   : `good` (>= 0.8), `mediocre` (0.3 <= score <= 0.6), `bad` (<= 0.1)

Writes a `selections.json` consumed by `10_render_visuals.py` on the remote.

Run from `experiments/eva-eval/`:
    python scripts/10_pick_visuals.py \
        --results-jsonl results/full_qwen_internvl2.jsonl \
        --out results/visuals/full_qwen_internvl2/selections.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eva_eval.eval.metrics import MCA_QUESTION_TYPES, NA_QUESTION_TYPES, REPORTED_TASK_ORDER  # noqa: E402


def reported_type_of(qt: str) -> str:
    return "object_rel_direction" if qt.startswith("object_rel_direction_") else qt


def latest_results_jsonl(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("full_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [p for p in candidates if not p.name.endswith(".summary.json") and ".partial" not in p.name]
    if not candidates:
        raise SystemExit(f"No full_*.jsonl found in {results_dir}")
    return candidates[0]


def pick_one(rng: random.Random, rows: list[dict], predicate) -> dict | None:
    pool = [r for r in rows if predicate(r["score"])]
    return rng.choice(pool) if pool else None


def main():
    ap = argparse.ArgumentParser(description="Pick good/mediocre/bad qualitative samples per question type.")
    ap.add_argument("--results-jsonl", type=Path, default=None,
                    help="Path to results JSONL. Default: newest results/full_*.jsonl")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output selections.json path. Default: results/visuals/<run>/selections.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    repo_results = Path(__file__).resolve().parent.parent / "results"
    jsonl_path = args.results_jsonl or latest_results_jsonl(repo_results)
    run_name = jsonl_path.stem
    out_path = args.out or (repo_results / "visuals" / run_name / "selections.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    rows = []
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("error"):
                continue
            rows.append(r)

    by_reported: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_reported[reported_type_of(r["question_type"])].append(r)

    selections: list[dict] = []
    coverage: dict[tuple[str, str], int] = {}

    for rtype in REPORTED_TASK_ORDER:
        pool = by_reported.get(rtype, [])
        if not pool:
            print(f"[warn] no rows for reported type {rtype}", file=sys.stderr)
            continue
        is_mca = any(qt == rtype or qt.startswith(rtype + "_") for qt in MCA_QUESTION_TYPES)
        is_na = rtype in NA_QUESTION_TYPES
        assert is_mca ^ is_na, rtype

        if is_mca:
            buckets = [
                ("good", lambda s: s == 1.0),
                ("bad", lambda s: s == 0.0),
            ]
        else:
            buckets = [
                ("good", lambda s: s >= 0.8),
                ("mediocre", lambda s: 0.3 <= s <= 0.6),
                ("bad", lambda s: s <= 0.1),
            ]

        # relax thresholds if a bucket is empty
        relaxed = {
            "good": lambda s: s >= 0.6,
            "mediocre": lambda s: 0.2 <= s <= 0.7,
            "bad": lambda s: s <= 0.2,
        }
        for bucket_name, pred in buckets:
            chosen = pick_one(rng, pool, pred)
            if chosen is None and bucket_name in relaxed:
                chosen = pick_one(rng, pool, relaxed[bucket_name])
                if chosen is not None:
                    print(f"[info] relaxed thresholds used for ({rtype}, {bucket_name})", file=sys.stderr)
            if chosen is None:
                print(f"[warn] no sample for ({rtype}, {bucket_name})", file=sys.stderr)
                coverage[(rtype, bucket_name)] = 0
                continue
            coverage[(rtype, bucket_name)] = 1
            selections.append({
                "id": chosen["id"],
                "scene_name": chosen["scene_name"],
                "question_type": chosen["question_type"],
                "reported_type": rtype,
                "bucket": bucket_name,
                "question": chosen["question"],
                "ground_truth": chosen["ground_truth"],
                "prediction": chosen["prediction"],
                "score": chosen["score"],
            })

    out_path.write_text(json.dumps(selections, indent=2))
    print(f"\nWrote {len(selections)} selections to {out_path}\n")
    print(f"{'reported_type':28s} {'bucket':10s} {'present'}")
    for (rtype, bname), present in coverage.items():
        print(f"{rtype:28s} {bname:10s} {present}")


if __name__ == "__main__":
    main()
