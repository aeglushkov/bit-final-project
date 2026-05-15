"""Classify every retrieve_objects_by_appearance / _by_environment call
in a trace JSONL into EMPTY / WRONG / USEFUL.

For each retrieve_* observation, we:
  1. Pull the planner's `tool_input` (the noun the planner was searching for).
  2. Pull the observation, which is of the form
     "The objects that satisfy '<query>' are {oid: '<caption>', ...}"
     or "(no matches)".
  3. EMPTY  = the observation is "(no matches)" or has zero captions.
     USEFUL = at least one caption substring-matches the searched noun
              (singular or plural) — the retrieval found the target.
     WRONG  = the observation has captions but none substring-match the
              query — the retrieval returned categorically different
              objects (the canonical "stool retrieved bench" failure).

Output: a histogram + the top WRONG examples.

Usage:
    python audit_retrieve_outputs.py \\
        --traces results/subset_si_with_traces.jsonl \\
        --out    results/retrieve_audit_si.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


_RETRIEVE_TOOLS = {"retrieve_objects_by_appearance", "retrieve_objects_by_environment"}
# Tool input is "noun" (a bare string) per parse_tuple_input(s, expected_arity=1).
# Strip surrounding quotes.
_QUOTED = re.compile(r"""^["'`](.*)["'`]$""")
# Captions look like "The object is a chair." or "The object is a cabinet" etc.
_OBJ_NOUN = re.compile(r"object is an?\s+([a-z][a-z\- ]+?)\.?\s*$", re.IGNORECASE)


def _parse_query_noun(tool_input: str) -> str:
    s = (tool_input or "").strip()
    m = _QUOTED.match(s)
    if m:
        return m.group(1).strip().lower()
    # If literal_eval works (Python tuple/string), unwrap.
    try:
        v = ast.literal_eval(s)
        if isinstance(v, tuple) and len(v) == 1:
            v = v[0]
        if isinstance(v, str):
            return v.strip().lower()
    except (SyntaxError, ValueError):
        pass
    return s.lower()


def _parse_captions(observation: str) -> list[str]:
    """Pull the VLM captions out of an observation like:
       "The objects that satisfy '...' are {0: 'a chair', 5: 'a brown chair', ...}"
       or "The objects that satisfy '...' are (no matches)" → []
    """
    if "(no matches)" in (observation or ""):
        return []
    # Find the dict-ish substring between { ... }
    m = re.search(r"\{(.+)\}", observation or "", re.DOTALL)
    if not m:
        return []
    inner = m.group(1)
    # Captions are 'string', not double-quoted; cheap parse with regex.
    return [c.strip() for c in re.findall(r"\d+:\s*['\"](.+?)['\"]\s*[,}]", inner + "}")]


def _caption_mentions(caption: str, noun: str) -> bool:
    """noun substring-match against a VLM caption; matches singular OR plural
    and allows the caption to say "X is a chair" / "object is a chair." etc."""
    if not caption or not noun:
        return False
    cap = caption.lower()
    n = noun.lower().rstrip("s")
    return n in cap or (n + "s") in cap or (" " + n + " ") in (" " + cap + " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    overall = Counter()
    by_tool = defaultdict(Counter)
    by_query_noun = defaultdict(Counter)
    wrong_examples: list[dict] = []
    all_records: list[dict] = []

    for line in args.traces.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for step in r.get("trajectory") or []:
            tool = step.get("tool", "")
            if tool not in _RETRIEVE_TOOLS:
                continue
            query = _parse_query_noun(step.get("tool_input", ""))
            captions = _parse_captions(step.get("observation", ""))
            if not captions:
                status = "EMPTY"
            elif any(_caption_mentions(c, query) for c in captions):
                status = "USEFUL"
            else:
                status = "WRONG"
            overall[status] += 1
            by_tool[tool][status] += 1
            by_query_noun[query][status] += 1
            rec = {
                "id": r["id"], "question_type": r["question_type"],
                "tool": tool, "query": query, "n_captions": len(captions),
                "status": status, "captions_sample": captions[:3],
            }
            all_records.append(rec)
            if status == "WRONG" and len(wrong_examples) < 15:
                wrong_examples.append(rec)

    out = {
        "overall": dict(overall),
        "by_tool": {k: dict(v) for k, v in by_tool.items()},
        "by_query_noun_top20_wrong": sorted(
            [(q, c["WRONG"]) for q, c in by_query_noun.items() if c.get("WRONG", 0)],
            key=lambda x: -x[1],
        )[:20],
        "wrong_examples": wrong_examples,
        "n_retrieve_calls": sum(overall.values()),
        "n_records": len(all_records),
    }
    args.out.write_text(json.dumps(out, indent=2, default=str))

    print(f"wrote {args.out}  n_calls={sum(overall.values())}")
    print(f"\n=== Overall ===")
    for s, n in overall.most_common():
        pct = 100.0 * n / sum(overall.values()) if overall else 0
        print(f"  {s:<10s} {n:>4}   {pct:>5.1f}%")

    print(f"\n=== By tool ===")
    for t in sorted(by_tool):
        c = by_tool[t]
        tot = sum(c.values())
        e, w, u = c.get("EMPTY", 0), c.get("WRONG", 0), c.get("USEFUL", 0)
        print(f"  {t:<35s} EMPTY={e}  WRONG={w}  USEFUL={u}  (n={tot})")

    print(f"\n=== Top 10 query nouns that returned WRONG ===")
    for q, n in out["by_query_noun_top20_wrong"][:10]:
        print(f"  {q!r:<32s} WRONG={n}")


if __name__ == "__main__":
    main()
