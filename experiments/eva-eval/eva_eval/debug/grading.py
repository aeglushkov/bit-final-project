"""Renderer for the graded-results inspection HTML."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from eva_eval.debug.render import write_html
from eva_eval.eval.openeqa_grade import aggregate, c_score


def render_grading_html(graded_jsonl: str | Path) -> Path:
    p = Path(graded_jsonl)
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    summary = aggregate(rows)
    per_cat_rows = defaultdict(list)
    for r in rows:
        per_cat_rows[r.get("category", "?")].append(r)

    body_parts: list[str] = []
    body_parts.append(f"<h1>Grading inspection — <code>{p.name}</code></h1>")

    body_parts.append("<h2>Per-category C-scores</h2>")
    body_parts.append(_summary_table(summary))

    body_parts.append("<h2>Judge score histogram (1–5)</h2>")
    body_parts.append(_histogram(rows))

    body_parts.append("<h2>Worst-10 and Best-10 per category</h2>")
    for cat in sorted(per_cat_rows):
        body_parts.append(f"<h3>{cat}</h3>")
        body_parts.append(_examples_table(per_cat_rows[cat], "Worst-10", n=10, ascending=True))
        body_parts.append(_examples_table(per_cat_rows[cat], "Best-10", n=10, ascending=False))

    out = p.parent / (p.stem + ".inspect.html")
    return write_html(out, title=f"grading: {p.name}", body="\n".join(body_parts))


def _summary_table(summary: dict) -> str:
    rows = [("overall", f"{summary['overall']:.2f}")]
    rows.append(("n_questions", str(summary["n_questions"])))
    for cat, sc in summary["per_category"].items():
        rows.append((cat, f"{sc:.2f}"))
    body = "".join(f"<tr><th>{k}</th><td><code>{v}</code></td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _histogram(rows: Iterable[dict]) -> str:
    counts = Counter(int(r["score"]) for r in rows if r.get("score") is not None)
    cells = []
    for s in (1, 2, 3, 4, 5):
        n = counts.get(s, 0)
        cells.append(f"<tr><th>score {s}</th><td><code>{n}</code></td></tr>")
    return f"<table>{''.join(cells)}</table>"


def _examples_table(rows: list[dict], label: str, *, n: int, ascending: bool) -> str:
    scored = [r for r in rows if r.get("score") is not None]
    scored.sort(key=lambda r: int(r["score"]), reverse=not ascending)
    pick = scored[:n]
    if not pick:
        return f"<p><em>{label}: (no scored rows)</em></p>"
    th = (
        "<tr><th>id</th><th>score</th><th>question</th><th>gold</th>"
        "<th>prediction</th><th>rationale</th></tr>"
    )
    body = []
    for r in pick:
        body.append(
            f"<tr><td><code>{r.get('id','')}</code></td>"
            f"<td>{r['score']}</td>"
            f"<td>{(r.get('question') or '')[:160]}</td>"
            f"<td>{(r.get('ground_truth') or '')[:160]}</td>"
            f"<td>{(r.get('prediction') or '')[:160]}</td>"
            f"<td>{(r.get('judge_rationale') or '')[:200]}</td></tr>"
        )
    return f"<h4>{label}</h4><table>{th}{''.join(body)}</table>"
