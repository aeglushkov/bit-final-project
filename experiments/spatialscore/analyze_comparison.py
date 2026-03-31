"""Compare MLLM baseline vs SpatialAgent results.

Usage:
    python analyze_comparison.py \
        --baseline_dir ../../literature/spatialscore/code/eval_results_diverse/qwen2_5vl-3b \
        --agent_dir ../../literature/spatialscore/code/eval_results_diverse_agent/qwen2_5vl-3b \
        --output findings.md
"""

import os
import json
import argparse


def load_results(results_dir):
    """Load all_results.json and overall_summary.json from a results directory."""
    with open(os.path.join(results_dir, "all_results.json"), "r") as f:
        all_results = json.load(f)
    summary_path = os.path.join(results_dir, "overall_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        total = len(all_results)
        correct = sum(1 for r in all_results if r.get("is_correct"))
        score_sum = sum(r.get("score", 0.0) for r in all_results)
        summary = {"accuracy": (score_sum / total) * 100 if total else 0,
                    "correct": correct, "total": total, "score_sum": score_sum}
    return all_results, summary


def group_by(results, key):
    """Group results by a field and compute accuracy per group."""
    groups = {}
    for r in results:
        val = r.get(key, "unknown")
        groups.setdefault(val, []).append(r)

    stats = {}
    for val, items in groups.items():
        total = len(items)
        score_sum = sum(r.get("score", 0.0) for r in items)
        stats[val] = {"accuracy": (score_sum / total) * 100 if total else 0,
                      "correct": int(score_sum), "total": total}
    return stats


def per_sample_diff(baseline_results, agent_results):
    """Find samples where agent improved/hurt vs baseline."""
    baseline_by_id = {r["id"]: r for r in baseline_results}
    agent_by_id = {r["id"]: r for r in agent_results}

    common_ids = set(baseline_by_id.keys()) & set(agent_by_id.keys())

    improved = []  # baseline wrong, agent correct
    regressed = []  # baseline correct, agent wrong
    both_correct = 0
    both_wrong = 0

    for sid in sorted(common_ids):
        b = baseline_by_id[sid]
        a = agent_by_id[sid]
        b_correct = b.get("score", 0.0) > 0
        a_correct = a.get("score", 0.0) > 0

        if b_correct and a_correct:
            both_correct += 1
        elif not b_correct and not a_correct:
            both_wrong += 1
        elif not b_correct and a_correct:
            improved.append(sid)
        else:
            regressed.append(sid)

    return {
        "improved": improved,
        "regressed": regressed,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "total_compared": len(common_ids),
    }


def format_comparison_table(baseline_stats, agent_stats, group_name):
    """Format a markdown comparison table."""
    all_keys = sorted(set(baseline_stats.keys()) | set(agent_stats.keys()))

    lines = [
        f"| {group_name} | Baseline Acc | Agent Acc | Delta | Baseline N | Agent N |",
        "|---|---|---|---|---|---|",
    ]
    for key in all_keys:
        b = baseline_stats.get(key, {"accuracy": 0, "total": 0})
        a = agent_stats.get(key, {"accuracy": 0, "total": 0})
        delta = a["accuracy"] - b["accuracy"]
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {key} | {b['accuracy']:.1f}% | {a['accuracy']:.1f}% | {sign}{delta:.1f}% | {b['total']} | {a['total']} |"
        )
    return "\n".join(lines)


def generate_findings(baseline_dir, agent_dir, output_path):
    """Generate the full findings markdown."""
    baseline_results, baseline_summary = load_results(baseline_dir)
    agent_results, agent_summary = load_results(agent_dir)

    baseline_by_source = group_by(baseline_results, "source")
    agent_by_source = group_by(agent_results, "source")
    baseline_by_category = group_by(baseline_results, "category")
    agent_by_category = group_by(agent_results, "category")
    baseline_by_qtype = group_by(baseline_results, "question_type")
    agent_by_qtype = group_by(agent_results, "question_type")

    diff = per_sample_diff(baseline_results, agent_results)

    content = f"""# SpatialScore Experiment Findings

## Overview

Comparison of Qwen2.5-VL-3B evaluated in two modes:
- **Baseline (MLLM only):** Direct VLM inference, no tool use
- **SpatialAgent:** Agentic loop with tool-augmented spatial reasoning

## Overall Results

| Mode | Accuracy | Correct | Total |
|---|---|---|---|
| Baseline | {baseline_summary['accuracy']:.2f}% | {baseline_summary['correct']} | {baseline_summary['total']} |
| SpatialAgent | {agent_summary['accuracy']:.2f}% | {agent_summary['correct']} | {agent_summary['total']} |
| **Delta** | **{agent_summary['accuracy'] - baseline_summary['accuracy']:+.2f}%** | | |

## By Source

{format_comparison_table(baseline_by_source, agent_by_source, "Source")}

## By Category

{format_comparison_table(baseline_by_category, agent_by_category, "Category")}

## By Question Type

{format_comparison_table(baseline_by_qtype, agent_by_qtype, "Question Type")}

## Per-Sample Analysis

Out of {diff['total_compared']} samples compared:
- **Both correct:** {diff['both_correct']}
- **Both wrong:** {diff['both_wrong']}
- **Agent improved (baseline wrong → agent correct):** {len(diff['improved'])} samples
- **Agent regressed (baseline correct → agent wrong):** {len(diff['regressed'])} samples

### Improved Sample IDs
{', '.join(str(s) for s in diff['improved'][:20]) if diff['improved'] else 'None'}

### Regressed Sample IDs
{', '.join(str(s) for s in diff['regressed'][:20]) if diff['regressed'] else 'None'}

## Key Takeaways

1. **Overall delta:** SpatialAgent {'improves' if agent_summary['accuracy'] > baseline_summary['accuracy'] else 'does not improve'} over the baseline by {abs(agent_summary['accuracy'] - baseline_summary['accuracy']):.1f} percentage points.
2. **Categories with largest improvement:** (see table above)
3. **Categories where agent hurts:** (see table above)
4. The SpatialAgent paper reports +6-8% improvement with Qwen3-VL-4B/8B models. Our results with Qwen2.5-VL-3B {'are consistent with' if agent_summary['accuracy'] > baseline_summary['accuracy'] else 'differ from'} this trend.

## Implications for Our Research

- Categories where the agent improves most suggest where **externalized spatial reasoning** adds the most value.
- Categories where it regresses suggest the agent's tool calls may introduce noise or the 3B model struggles to follow the agentic format.
- Object Localization and Depth/Distance tasks are prime candidates for tool-augmented approaches.
"""

    with open(output_path, "w") as f:
        f.write(content)
    print(f"Findings written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare MLLM baseline vs SpatialAgent")
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--agent_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="findings.md")
    args = parser.parse_args()

    generate_findings(args.baseline_dir, args.agent_dir, args.output)


if __name__ == "__main__":
    main()
