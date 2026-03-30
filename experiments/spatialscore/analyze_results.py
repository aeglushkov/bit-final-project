"""Analyze SpatialScore evaluation results.

Usage:
    python analyze_results.py <results_dir>

Example:
    python analyze_results.py literature/spatialscore/code/eval_results_test/qwen2_5vl-3b/
"""

import json
import sys
import os
from collections import defaultdict


def load_results(results_dir):
    path = os.path.join(results_dir, "all_results.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        sys.exit(1)
    with open(path, "r") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results from {path}\n")
    return results


def show_samples(results, n=10):
    """Print first n samples with predictions vs ground truth."""
    print(f"=== First {n} Samples ===\n")
    for r in results[:n]:
        correct_marker = "V" if r["is_correct"] is True else ("~" if r["is_correct"] == "accuracy" else "X")
        score = r.get("score", 0.0)
        print(f"[{correct_marker}] (score={score:.2f}) {r.get('source', '?')}/{r.get('category', '?')}")
        print(f"  Q: {r['question'][:100]}{'...' if len(r['question']) > 100 else ''}")
        print(f"  GT: {r['gt_answer']}")
        print(f"  Pred: {r['pred_answer'][:100]}{'...' if len(str(r['pred_answer'])) > 100 else ''}")
        print()


def accuracy_by_field(results, field):
    """Compute accuracy grouped by a field (source, category, question_type)."""
    groups = defaultdict(list)
    for r in results:
        groups[r.get(field, "unknown")].append(r)

    print(f"=== Accuracy by {field} ===\n")
    print(f"{'Value':<30} {'Accuracy':>10} {'Score Sum':>10} {'Count':>8}")
    print("-" * 62)

    total_score = 0
    total_count = 0
    for key in sorted(groups.keys()):
        items = groups[key]
        score_sum = sum(r.get("score", 0.0) for r in items)
        count = len(items)
        acc = (score_sum / count) * 100 if count > 0 else 0
        print(f"{key:<30} {acc:>9.1f}% {score_sum:>10.1f} {count:>8}")
        total_score += score_sum
        total_count += count

    overall = (total_score / total_count) * 100 if total_count > 0 else 0
    print("-" * 62)
    print(f"{'OVERALL':<30} {overall:>9.1f}% {total_score:>10.1f} {total_count:>8}")
    print()


def show_failures(results, n=10):
    """Show n examples where the model got it wrong."""
    failures = [r for r in results if r.get("score", 0.0) == 0.0]
    print(f"=== Failure Examples ({len(failures)} total failures, showing {min(n, len(failures))}) ===\n")
    for r in failures[:n]:
        print(f"  [{r.get('source', '?')}] {r.get('category', '?')} ({r.get('question_type', '?')})")
        print(f"  Q: {r['question'][:120]}{'...' if len(r['question']) > 120 else ''}")
        print(f"  GT: {r['gt_answer']}")
        print(f"  Pred: {r['pred_answer'][:120]}{'...' if len(str(r['pred_answer'])) > 120 else ''}")
        print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    results_dir = sys.argv[1]
    results = load_results(results_dir)

    show_samples(results)
    accuracy_by_field(results, "source")
    accuracy_by_field(results, "category")
    accuracy_by_field(results, "question_type")
    show_failures(results)


if __name__ == "__main__":
    main()
