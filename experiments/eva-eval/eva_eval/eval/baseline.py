"""Raw VLM baseline: feed sampled frames + question directly to a multimodal
chat model (no agent, no memory). Mirrors the Thinking-in-Space evaluation
protocol so we can diagnose whether our agent layer is adding or subtracting
value."""
from __future__ import annotations

import base64
import io
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from eva_eval.eval.metrics import aggregate, score_one
from eva_eval.eval.sampler import stratified_indices
from eva_eval.eval.vsibench import format_question, parse_final_answer
from eva_eval.llm.client import load_model


def _encode_image_b64(path: Path) -> str:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _sample_frame_paths(scene_dir: Path, n_frames: int) -> list[Path]:
    """Pick n_frames uniformly across the cached frames in scene_dir."""
    frames = sorted((scene_dir / "frames").glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames in {scene_dir}/frames")
    if len(frames) <= n_frames:
        return frames
    if n_frames == 1:
        return [frames[len(frames) // 2]]  # middle frame
    step = (len(frames) - 1) / (n_frames - 1)
    return [frames[int(round(i * step))] for i in range(n_frames)]


def _ask_vlm_with_frames(model, frames: list[Path], user_text: str, **gen_overrides) -> str:
    """Build an OpenAI-style multi-image chat message and call the VLM."""
    content: list[dict] = []
    for p in frames:
        b64 = _encode_image_b64(p)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": user_text})
    return model.chat([{"role": "user", "content": content}], **gen_overrides)


def run(
    cache_root: Path,
    output: Path,
    vlm_name: str = "internvl2-8b",
    limit: int | None = None,
    stratified: bool = True,
    seed: int = 42,
    n_frames: int = 8,
    pre_prompt: str = "These are frames of a video.",
    max_tokens: int = 16,
    only_cached: bool = True,
) -> dict:
    """Run the raw-VLM baseline on VSI-Bench."""
    from datasets import load_dataset

    output.parent.mkdir(parents=True, exist_ok=True)

    cached_scenes: set[str] | None = None
    if only_cached:
        cached_scenes = {d.name for d in cache_root.iterdir() if d.is_dir() and (d / "frames").exists()}
        if not cached_scenes:
            print(f"No cached scenes (frames/) in {cache_root}; nothing to evaluate.")
            return {"overall": 0.0, "n_questions": 0}
        print(f"Restricting to {len(cached_scenes)} cached scenes")

    ds = load_dataset("nyu-visionx/VSI-Bench", split="test")
    candidate = [i for i in range(len(ds)) if (cached_scenes is None or ds[i]["scene_name"] in cached_scenes)]

    if limit is not None and limit < len(candidate):
        if stratified:
            sub_qtypes = [ds[i]["question_type"] for i in candidate]
            sub_idxs = stratified_indices(sub_qtypes, total=limit, seed=seed)
            indices = [candidate[i] for i in sub_idxs]
        else:
            import random

            rng = random.Random(seed)
            pool = candidate[:]
            rng.shuffle(pool)
            indices = sorted(pool[:limit])
    else:
        indices = candidate

    by_scene: dict[str, list[int]] = defaultdict(list)
    for i in indices:
        by_scene[ds[i]["scene_name"]].append(i)
    print(f"Evaluating {len(indices)} questions across {len(by_scene)} scenes "
          f"({n_frames} frames/question, vlm={vlm_name})")

    model = load_model(vlm_name)
    if not model.multimodal:
        raise ValueError(f"{vlm_name} is not configured as multimodal")

    scored: list[dict] = []
    with output.open("w") as out_f:
        for scene_name, qidxs in by_scene.items():
            scene_dir = cache_root / scene_name
            try:
                frames = _sample_frame_paths(scene_dir, n_frames)
            except FileNotFoundError as e:
                print(f"[skip] {scene_name}: {e}", file=sys.stderr)
                continue

            for qi in qidxs:
                doc = ds[qi]
                user_text = format_question(doc, pre_prompt=pre_prompt)
                try:
                    raw = _ask_vlm_with_frames(model, frames, user_text, max_tokens=max_tokens)
                    pred = parse_final_answer(raw)
                    err = None
                except Exception as e:
                    pred = ""
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
