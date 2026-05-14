"""Diagnose why the bf16 agent regressed vs the AWQ agent on the 100Q subset.

Joins two agent result JSONLs by `id` and emits:
1. A row-level table sorted by score delta (bf16 - AWQ).
2. Regression / improvement / unchanged counts, per question type.
3. Prediction-value histograms per question type — looks for mode collapse
   (bf16 falling into a small set of repeated answers like "0" or "A").
4. The MRA near-miss artifact: how often does the regression come from bf16
   producing a "wronger" number that loses partial credit AWQ had?

Read-only — does not modify either file.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.open():
        r = json.loads(line)
        out[str(r["id"])] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--awq", required=True, type=Path, help="AWQ agent JSONL (e.g. subset_fixed.jsonl)")
    ap.add_argument("--bf16", required=True, type=Path, help="bf16 agent JSONL (e.g. subset_bf16.jsonl)")
    ap.add_argument("--top", type=int, default=10, help="N worst regressions to print")
    args = ap.parse_args()

    awq = load(args.awq)
    bf16 = load(args.bf16)
    common = sorted(set(awq) & set(bf16))
    if len(awq) != len(bf16) or set(awq) != set(bf16):
        print(f"WARN: mismatch — awq={len(awq)} bf16={len(bf16)} common={len(common)}")

    rows = []
    for qid in common:
        a, b = awq[qid], bf16[qid]
        rows.append({
            "id": qid,
            "type": a["question_type"],
            "gt": a["ground_truth"],
            "awq_pred": a["prediction"],
            "awq_score": float(a["score"]),
            "bf16_pred": b["prediction"],
            "bf16_score": float(b["score"]),
            "delta": float(b["score"]) - float(a["score"]),
        })

    print(f"\n=== Joined {len(rows)} questions ===")
    awq_total = sum(r["awq_score"] for r in rows) / len(rows) * 100
    bf16_total = sum(r["bf16_score"] for r in rows) / len(rows) * 100
    print(f"  overall:  AWQ={awq_total:.2f}  bf16={bf16_total:.2f}  delta={bf16_total - awq_total:+.2f}")

    # Per-type aggregate
    print("\n=== Per-type means ===")
    print(f"  {'question_type':<30s}  {'n':>3s}  {'AWQ':>6s}  {'bf16':>6s}  {'delta':>6s}  {'regr':>4s}  {'imp':>4s}  {'eq':>4s}")
    types: dict[str, list] = defaultdict(list)
    for r in rows:
        types[r["type"]].append(r)
    for t in sorted(types):
        rs = types[t]
        a_avg = sum(r["awq_score"] for r in rs) / len(rs) * 100
        b_avg = sum(r["bf16_score"] for r in rs) / len(rs) * 100
        reg = sum(1 for r in rs if r["delta"] < -1e-9)
        imp = sum(1 for r in rs if r["delta"] > 1e-9)
        eq  = sum(1 for r in rs if abs(r["delta"]) <= 1e-9)
        print(f"  {t:<30s}  {len(rs):>3d}  {a_avg:>6.2f}  {b_avg:>6.2f}  {b_avg - a_avg:>+6.2f}  {reg:>4d}  {imp:>4d}  {eq:>4d}")

    # Mode collapse: how concentrated are bf16's predictions for each type?
    print("\n=== bf16 prediction histograms (top 5 values per type) ===")
    for t in sorted(types):
        preds = [str(r["bf16_pred"]).strip() for r in types[t]]
        c = Counter(preds)
        n = len(preds)
        top = c.most_common(5)
        compact = "  ".join(f"{v!r}×{k}" for v, k in top)
        print(f"  {t:<30s}  n={n:<3d}  top: {compact}")

    print("\n=== AWQ prediction histograms (top 5 values per type) ===")
    for t in sorted(types):
        preds = [str(r["awq_pred"]).strip() for r in types[t]]
        c = Counter(preds)
        n = len(preds)
        top = c.most_common(5)
        compact = "  ".join(f"{v!r}×{k}" for v, k in top)
        print(f"  {t:<30s}  n={n:<3d}  top: {compact}")

    # MRA near-miss artifact: how often did AWQ get partial credit that bf16 lost?
    print("\n=== MRA near-miss artifact ===")
    partial_to_zero = sum(1 for r in rows if 0 < r["awq_score"] < 1 and r["bf16_score"] == 0)
    zero_to_partial = sum(1 for r in rows if r["awq_score"] == 0 and 0 < r["bf16_score"] < 1)
    both_partial   = sum(1 for r in rows if 0 < r["awq_score"] < 1 and 0 < r["bf16_score"] < 1)
    awq_only_full  = sum(1 for r in rows if r["awq_score"] == 1.0 and r["bf16_score"] < 1.0)
    bf16_only_full = sum(1 for r in rows if r["bf16_score"] == 1.0 and r["awq_score"] < 1.0)
    print(f"  AWQ partial -> bf16 zero      : {partial_to_zero:>3d}  (regression by loss of partial credit)")
    print(f"  AWQ zero    -> bf16 partial   : {zero_to_partial:>3d}  (improvement by partial credit)")
    print(f"  both partial (diff scores)    : {both_partial:>3d}")
    print(f"  AWQ full    -> bf16 sub-full  : {awq_only_full:>3d}  (regression: AWQ correct, bf16 not)")
    print(f"  bf16 full   -> AWQ sub-full   : {bf16_only_full:>3d}  (improvement: bf16 correct, AWQ not)")

    # Top regressions
    print(f"\n=== Top {args.top} regressions (most negative delta) ===")
    print(f"  {'id':<8s}  {'type':<30s}  {'gt':<12s}  {'awq_pred':<22s} {'aS':>5s}  {'bf16_pred':<22s} {'bS':>5s}  {'Δ':>6s}")
    for r in sorted(rows, key=lambda r: r["delta"])[: args.top]:
        print(f"  {r['id']:<8s}  {r['type']:<30s}  {str(r['gt'])[:12]:<12s}  "
              f"{str(r['awq_pred'])[:22]:<22s} {r['awq_score']:>5.2f}  "
              f"{str(r['bf16_pred'])[:22]:<22s} {r['bf16_score']:>5.2f}  {r['delta']:>+6.2f}")

    print(f"\n=== Top {args.top} improvements (most positive delta) ===")
    print(f"  {'id':<8s}  {'type':<30s}  {'gt':<12s}  {'awq_pred':<22s} {'aS':>5s}  {'bf16_pred':<22s} {'bS':>5s}  {'Δ':>6s}")
    for r in sorted(rows, key=lambda r: -r["delta"])[: args.top]:
        print(f"  {r['id']:<8s}  {r['type']:<30s}  {str(r['gt'])[:12]:<12s}  "
              f"{str(r['awq_pred'])[:22]:<22s} {r['awq_score']:>5.2f}  "
              f"{str(r['bf16_pred'])[:22]:<22s} {r['bf16_score']:>5.2f}  {r['delta']:>+6.2f}")


if __name__ == "__main__":
    main()
