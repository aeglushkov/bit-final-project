from __future__ import annotations

import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from eva_eval.eval.metrics import MCA_QUESTION_TYPES, NA_QUESTION_TYPES, aggregate, score_one
from eva_eval.eval.sampler import stratified_indices


def format_question(doc: dict) -> str:
    qt = doc["question_type"]
    question = doc["question"]
    if qt in MCA_QUESTION_TYPES:
        opts = "Options:\n" + "\n".join(doc["options"])
        post = "Answer with the option's letter from the given choices directly."
        return f"{question}\n{opts}\n{post}"
    if qt in NA_QUESTION_TYPES:
        return f"{question}\nPlease answer the question using a single word or phrase."
    raise ValueError(f"Unknown question_type: {qt!r}")


def parse_final_answer(response: Any) -> str:
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


def load_dataset_indices(
    limit: int | None,
    stratified: bool,
    seed: int,
    scene_filter: set[str] | None = None,
):
    from datasets import load_dataset

    ds = load_dataset("nyu-visionx/VSI-Bench", split="test")
    candidate = list(range(len(ds)))
    if scene_filter is not None:
        candidate = [i for i in candidate if ds[i]["scene_name"] in scene_filter]
    if limit is None or limit >= len(candidate):
        return ds, candidate
    if stratified:
        sub_qtypes = [ds[i]["question_type"] for i in candidate]
        sub_idxs = stratified_indices(sub_qtypes, total=limit, seed=seed)
        return ds, [candidate[i] for i in sub_idxs]
    import random

    rng = random.Random(seed)
    pool = candidate[:]
    rng.shuffle(pool)
    return ds, sorted(pool[:limit])


def group_by_scene(ds, indices: list[int]) -> dict[str, list[int]]:
    by_scene: dict[str, list[int]] = defaultdict(list)
    for i in indices:
        by_scene[ds[i]["scene_name"]].append(i)
    return dict(by_scene)


def run(
    cache_root: Path,
    paper_code_dir: Path,
    classes_file: Path,
    output: Path,
    planner: str | None = None,
    limit: int | None = None,
    stratified: bool = True,
    seed: int = 42,
    max_iterations: int = 30,
    on_missing_cache: str = "skip",
    only_cached: bool = True,
) -> dict:
    """Run VSI-Bench through the EVA agent. Writes one JSONL row per question
    to `output` and returns the aggregated metrics."""
    from eva_eval.agent.agent import build_agent
    from eva_eval.agent.text_encoder import build_clip_text_encoder

    output.parent.mkdir(parents=True, exist_ok=True)

    cached_scenes: set[str] | None = None
    if only_cached:
        cached_scenes = {
            d.name for d in cache_root.iterdir() if d.is_dir() and (d / "memory.pkl").exists()
        }
        if not cached_scenes:
            print(f"No cached scenes (memory.pkl) in {cache_root}; nothing to evaluate.")
            return {"overall": 0.0, "n_questions": 0}
        print(f"Restricting to {len(cached_scenes)} cached scenes")

    ds, indices = load_dataset_indices(
        limit=limit, stratified=stratified, seed=seed, scene_filter=cached_scenes
    )
    by_scene = group_by_scene(ds, indices)
    print(f"Evaluating {len(indices)} questions across {len(by_scene)} scenes")

    text_encoder = build_clip_text_encoder(paper_code_dir)

    scored: list[dict] = []
    with output.open("w") as out_f:
        for scene_name, qidxs in by_scene.items():
            video_cache_dir = cache_root / scene_name
            if not (video_cache_dir / "memory.pkl").exists():
                msg = f"[skip] {scene_name}: no memory.pkl in {video_cache_dir}"
                print(msg, file=sys.stderr)
                if on_missing_cache == "fail":
                    raise FileNotFoundError(msg)
                continue
            try:
                executor, _ctx = build_agent(
                    video_cache_dir=video_cache_dir,
                    paper_code_dir=paper_code_dir,
                    classes_file=classes_file,
                    text_encoder=text_encoder,
                    planner_name=planner,
                    max_iterations=max_iterations,
                )
            except Exception as e:
                print(f"[fail-build] {scene_name}: {e}", file=sys.stderr)
                continue

            for qi in qidxs:
                doc = ds[qi]
                user_text = format_question(doc)
                try:
                    response = executor.invoke({"input": user_text})
                    pred = parse_final_answer(response)
                    err = None
                except Exception as e:
                    pred = ""
                    response = None
                    err = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"

                score = score_one(doc["question_type"], pred, doc["ground_truth"])
                row = {
                    "id": doc.get("id", qi),
                    "scene_name": scene_name,
                    "question_type": doc["question_type"],
                    "question": doc["question"],
                    "ground_truth": doc["ground_truth"],
                    "prediction": pred,
                    "score": score,
                    "error": err,
                }
                scored.append(row)
                out_f.write(json.dumps(row, default=str) + "\n")
                out_f.flush()

    summary = aggregate(scored)
    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<32s} {v:7.3f}")
        else:
            print(f"  {k:<32s} {v}")
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(dict(summary), indent=2))
    return dict(summary)


def aggregate_jsonl(path: Path) -> dict:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return dict(aggregate(rows))
