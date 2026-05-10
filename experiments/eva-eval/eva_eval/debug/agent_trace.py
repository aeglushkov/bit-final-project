"""Render a single agent ReAct trace as a markdown document."""
from __future__ import annotations

import json
from pathlib import Path


def find_question(predictions_jsonl: str | Path, question_id: str) -> dict:
    """Locate a row by `id` in a predictions JSONL. Raises KeyError if absent."""
    p = Path(predictions_jsonl)
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("id") == question_id:
            return row
    raise KeyError(f"question_id {question_id!r} not found in {p}")


def render_trace_markdown(row: dict) -> str:
    """Render a row (from predictions JSONL or graded JSONL) as markdown."""
    parts: list[str] = []
    parts.append(f"# Trace — `{row.get('id', '?')}` ({row.get('category', '?')})")
    parts.append("")
    parts.append(f"**Question:** {row.get('question', '')}")
    parts.append("")
    parts.append(f"**Ground truth:** {row.get('ground_truth', '')}")
    parts.append("")
    parts.append(f"**Prediction:** {row.get('prediction', '')}")
    parts.append("")
    if "score" in row and row["score"] is not None:
        parts.append(f"Judge score: {row['score']} / 5")
        if row.get("judge_rationale"):
            parts.append("")
            parts.append(f"Judge rationale: {row['judge_rationale']}")
        parts.append("")

    parts.append("---")
    parts.append("")
    steps = row.get("intermediate_steps") or []
    if not steps:
        parts.append("(no intermediate steps recorded)")
        return "\n".join(parts)

    for i, step in enumerate(steps, start=1):
        parts.append(f"## Step {i}: `{step.get('tool', '?')}`")
        parts.append("")
        if step.get("log"):
            parts.append("**Thought / Action log:**")
            parts.append("")
            parts.append("```")
            parts.append(str(step["log"]).strip())
            parts.append("```")
            parts.append("")
        if step.get("tool_input") is not None:
            parts.append(f"**Tool input:** `{step['tool_input']}`")
            parts.append("")
        obs = str(step.get("observation", ""))
        if len(obs) > 1500:
            obs = obs[:1500] + "\n... (truncated)"
        parts.append("**Observation:**")
        parts.append("")
        parts.append("```")
        parts.append(obs)
        parts.append("```")
        parts.append("")

    return "\n".join(parts)
