"""OpenEQA HM3D evaluation harness — parallels eval/vsibench.py.

Run loop and per-question agent invocation; grading is a separate step
(see eval/openeqa_grade.py and scripts/08_grade_openeqa.py).
"""
from __future__ import annotations

import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from eva_eval.eval.sampler import stratified_indices


OPENEQA_PRE_PROMPT = "These are frames from an indoor scene exploration video."


def load_openeqa_questions(
    questions_json: str | Path,
    *,
    dataset: str = "hm3d",
    limit: int | None = 50,
    stratified: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Load OpenEQA questions from `data/open-eqa-v0.json`, filter by source
    dataset, optionally stratified-sample by category."""
    rows = json.loads(Path(questions_json).read_text())
    if dataset == "hm3d":
        rows = [r for r in rows if r.get("episode_history", "").startswith("hm3d-v0/")]
    elif dataset == "scannet":
        rows = [r for r in rows if r.get("episode_history", "").startswith("scannet-v0/")]
    elif dataset == "all":
        pass
    else:
        raise ValueError(f"Unknown dataset {dataset!r}; expected hm3d|scannet|all")

    if limit is None or limit >= len(rows):
        return rows

    if stratified:
        cats = [r.get("category", "?") for r in rows]
        idxs = stratified_indices(cats, total=limit, seed=seed)
        return [rows[i] for i in idxs]

    import random
    rng = random.Random(seed)
    pool = list(range(len(rows)))
    rng.shuffle(pool)
    return [rows[i] for i in sorted(pool[:limit])]


def format_question(q: dict) -> str:
    """Open-ended QA prompt — no MCA/NA branching."""
    return f"{OPENEQA_PRE_PROMPT}\n{q['question']}"


def episode_cache_dir(cache_root: str | Path, episode_history: str) -> Path:
    """'hm3d-v0/<episode_id>' -> <cache_root>/openeqa_hm3d/<episode_id>"""
    cache_root = Path(cache_root)
    if "/" in episode_history:
        ep_id = episode_history.split("/", 1)[1]
    else:
        ep_id = episode_history
    return cache_root / "openeqa_hm3d" / ep_id


def parse_final_answer(response) -> str:
    if isinstance(response, dict):
        text = response.get("output", "")
    else:
        text = str(response)
    text = str(text).strip()
    for prefix in ("Final Answer:", "FINAL ANSWER:", "final answer:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    return text


def group_by_episode(rows: list[dict]) -> dict[str, list[int]]:
    by_ep: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_ep[r["episode_history"]].append(i)
    return dict(by_ep)


def run(
    *,
    sampled_json: Path,
    cache_root: Path,
    paper_code_dir: Path,
    classes_file: Path,
    output: Path,
    planner: str | None = None,
    max_iterations: int = 30,
    capture_trace: bool = True,
    resume: bool = False,
) -> dict:
    """Run the agent over a sampled set of OpenEQA questions. Writes one JSONL
    row per question to `output`. Returns a small summary dict."""
    from eva_eval.agent.agent import build_agent
    from eva_eval.agent.text_encoder import build_clip_text_encoder

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(sampled_json).read_text())
    by_ep = group_by_episode(rows)
    print(f"Evaluating {len(rows)} questions across {len(by_ep)} episodes")

    answered: set = set()
    if resume and output.exists():
        with output.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error"):
                    continue
                answered.add(r["id"])
        print(f"resume: skipping {len(answered)} previously answered questions")
    open_mode = "a" if resume else "w"

    text_encoder = build_clip_text_encoder(paper_code_dir)
    n_done = 0
    with output.open(open_mode) as out_f:
        for episode_history, qidxs in by_ep.items():
            ep_dir = episode_cache_dir(cache_root, episode_history)
            if not (ep_dir / "memory.pkl").exists():
                print(f"[skip] {episode_history}: no memory.pkl in {ep_dir}", file=sys.stderr)
                continue
            remaining = [qi for qi in qidxs if rows[qi]["question_id"] not in answered]
            if not remaining:
                continue
            try:
                executor, _ctx = build_agent(
                    video_cache_dir=ep_dir,
                    paper_code_dir=paper_code_dir,
                    classes_file=classes_file,
                    text_encoder=text_encoder,
                    planner_name=planner,
                    max_iterations=max_iterations,
                    return_intermediate_steps=capture_trace,
                )
            except Exception as e:
                print(f"[fail-build] {episode_history}: {e}", file=sys.stderr)
                continue

            for qi in remaining:
                doc = rows[qi]
                user_text = format_question(doc)
                try:
                    response = executor.invoke({"input": user_text})
                    pred = parse_final_answer(response)
                    steps = response.get("intermediate_steps", []) if isinstance(response, dict) and capture_trace else []
                    err = None
                except Exception as e:
                    pred = ""
                    steps = []
                    err = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"

                row_out = {
                    "id": doc["question_id"],
                    "episode_history": episode_history,
                    "category": doc.get("category", "?"),
                    "question": doc["question"],
                    "ground_truth": doc["answer"],
                    "prediction": pred,
                    "intermediate_steps": _serialize_steps(steps),
                    "error": err,
                }
                out_f.write(json.dumps(row_out, default=str) + "\n")
                out_f.flush()
                n_done += 1

    return {"n_done": n_done, "n_total": len(rows)}


def _serialize_steps(steps) -> list:
    """Convert LangChain AgentAction intermediate_steps tuples to JSON-friendly dicts."""
    out = []
    for step in steps:
        try:
            action, observation = step
            out.append({
                "tool": getattr(action, "tool", str(action)),
                "tool_input": getattr(action, "tool_input", None),
                "log": getattr(action, "log", None),
                "observation": str(observation),
            })
        except Exception:
            out.append({"raw": repr(step)})
    return out
