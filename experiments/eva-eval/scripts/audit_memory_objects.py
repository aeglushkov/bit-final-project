"""Audit memory.pkl object distribution for the 10 VSI-Bench counting questions.

For each counting question, load the scene's `memory.pkl`, count distinct
`Object3D` entries per YOLO category, and compare against ground truth + the
agent's prediction. Classifies each row into one of:

    YOLO_MISS    — the question's target noun is not in the YOLO 60-class
                   vocabulary at all, OR the scene has zero entries of that
                   category. Agent had no chance.
    FRAGMENT     — memory has more entries of the target category than the
                   ground-truth count. Likely caused by re-ID failures across
                   frames (MASt3R noise or visual-similarity false negatives).
    UNDERCOUNT   — memory has fewer entries than ground truth (and >0). YOLO
                   missed some instances.
    CORRECT      — memory's count of the target category matches ground truth
                   (within ±1). The agent's wrong prediction is then a
                   downstream-tool / planner-routing issue, not memory.

Usage:
    python audit_memory_objects.py \\
        --counting-jsonl  results/subset_si_with_traces.jsonl \\
        --raw-vlm-jsonl   results/baseline_sensenova-si-1.5-internvl3-8b-bf16.jsonl \\
        --cache-root      cache/vsibench \\
        --paper-code-dir  literature/EmbodiedVideoAgent/code \\
        --out             results/memory_audit_counting.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# YOLO-World categories the memory was built with — used to flag YOLO_MISS up front.
def _load_yolo_classes(paper_code_dir: Path) -> set[str]:
    sys.path.insert(0, str(paper_code_dir.resolve()))
    try:
        from detection_classes import customized_classes  # type: ignore
        return {c.strip().lower() for c in customized_classes}
    finally:
        if str(paper_code_dir.resolve()) in sys.path:
            sys.path.remove(str(paper_code_dir.resolve()))


# VSI-Bench counting questions of the form
# "How many <noun>(s) are in this room?" — pull the noun out so we can pivot
# the memory's category histogram. The "(s)" suffix is literal in the dataset.
_COUNTING_NOUN_RE = re.compile(r"how many\s+([a-z][a-z\- ]+?)\s*\(s\)", re.IGNORECASE)


def _extract_target_noun(question: str) -> str | None:
    m = _COUNTING_NOUN_RE.search(question)
    if not m:
        # Fall back to the "How many X are/is in this room" form without "(s)".
        m = re.search(r"how many\s+([a-z][a-z\- ]+?)\s+(?:are|is)\s+in", question, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip().lower()


def _count_by_category(memory_state: dict[str, Any]) -> Counter:
    """Count Object3D instances per category across static + dynamic banks."""
    counts: Counter = Counter()
    for bank in ("static_objects", "dynamic_objects"):
        for obj in memory_state.get(bank, []) or []:
            cat = getattr(obj, "category", None) or getattr(obj, "class_name", None)
            if cat is None and hasattr(obj, "__dict__"):
                cat = obj.__dict__.get("category") or obj.__dict__.get("class_name")
            counts[str(cat).strip().lower() if cat else "<unknown>"] += 1
    return counts


def _classify(target: str | None, mem_count: int, gt: int, yolo_known: bool) -> str:
    if target is None:
        return "PARSE_FAIL"
    if not yolo_known:
        return "YOLO_MISS"
    if mem_count == 0:
        return "YOLO_MISS"
    if mem_count > gt + 1:
        return "FRAGMENT"
    if mem_count < gt:
        return "UNDERCOUNT"
    return "CORRECT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counting-jsonl", required=True, type=Path,
                    help="JSONL with agent predictions (must have id, scene_name, question, ground_truth, prediction, question_type=object_counting).")
    ap.add_argument("--raw-vlm-jsonl", required=True, type=Path,
                    help="JSONL with raw VLM baseline predictions on the same IDs.")
    ap.add_argument("--cache-root", required=True, type=Path,
                    help="Phase-2/3 scene cache root (each subdir has memory.pkl).")
    ap.add_argument("--paper-code-dir", required=True, type=Path,
                    help="literature/EmbodiedVideoAgent/code (for ObjectMemory unpickling + YOLO class list).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Path for the audit JSON output.")
    args = ap.parse_args()

    # Lazy import; needs Object3D from paper code on sys.path.
    from eva_eval.memory.store import load_memory

    yolo_classes = _load_yolo_classes(args.paper_code_dir)

    raw_pred_by_id: dict[Any, str] = {}
    for line in args.raw_vlm_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        raw_pred_by_id[r["id"]] = str(r.get("prediction", ""))

    rows = []
    for line in args.counting_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("question_type") != "object_counting":
            continue
        scene = r["scene_name"]
        mem_path = args.cache_root / scene / "memory.pkl"
        if not mem_path.exists():
            rows.append({
                "id": r["id"], "scene": scene, "status": "NO_MEMORY",
                "question": r["question"], "ground_truth": r["ground_truth"],
                "agent_pred": r.get("prediction", ""),
                "raw_vlm_pred": raw_pred_by_id.get(r["id"], ""),
            })
            continue
        memory_state = load_memory(mem_path, paper_code_dir=args.paper_code_dir)
        cats = _count_by_category(memory_state)

        target = _extract_target_noun(r["question"])
        # Match the noun against the categories Counter — also try a few simple
        # variants (no trailing s, common renamings) to avoid false YOLO_MISS.
        candidates = []
        if target:
            for variant in {target, target.rstrip("s"), target + "s",
                            target.replace(" ", "_"), target.replace(" ", "")}:
                if variant in cats:
                    candidates.append(variant)
        target_count = sum(cats[v] for v in candidates) if candidates else 0
        yolo_known = bool(target and any(v in yolo_classes for v in {target, target.rstrip("s")}))

        try:
            gt = int(float(r["ground_truth"]))
        except (ValueError, TypeError):
            gt = -1

        classification = _classify(target, target_count, gt, yolo_known)

        rows.append({
            "id": r["id"],
            "scene": scene,
            "question": r["question"],
            "target_noun": target,
            "target_in_yolo": yolo_known,
            "ground_truth": gt,
            "memory_count_for_target": target_count,
            "memory_top10_categories": dict(cats.most_common(10)),
            "memory_total_objects": sum(cats.values()),
            "agent_pred": r.get("prediction", ""),
            "agent_score": r.get("score", 0.0),
            "raw_vlm_pred": raw_pred_by_id.get(r["id"], ""),
            "status": classification,
        })

    summary = Counter(r["status"] for r in rows)
    out = {"rows": rows, "summary": dict(summary)}
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {args.out}  rows={len(rows)}  status histogram={dict(summary)}")


if __name__ == "__main__":
    main()
