"""Renderers for the memory inspection HTML.

Note: the heavy 3D-bbox rendering uses AgentContext.render_object_bbox,
which depends on the paper's object3d module. Tests stub the objects.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from eva_eval.debug.render import image_to_data_uri, write_html


def summarize_memory(
    objects: Iterable[Any],
    *,
    objects_frames: dict[int, list[int]],
) -> dict[str, Any]:
    """Compute summary stats and emit auto-warnings for sanity checks."""
    objs = list(objects)
    n = len(objs)
    counts = Counter(getattr(o, "category", "?") for o in objs)

    visibility = [len(objects_frames.get(int(o.identifier), [])) for o in objs]
    pct_single = (sum(1 for v in visibility if v <= 1) / n) if n else 0.0

    volumes = [float(getattr(o, "volume", 0.0)) for o in objs]

    warnings: list[str] = []
    if n == 0:
        warnings.append("0 objects detected — memory build is broken (check vocabulary, depth range, model weights).")
    if 0 < len(counts) < 5:
        warnings.append(f"Only {len(counts)} categories — vocabulary is likely too narrow for this scene.")
    if n > 0 and pct_single > 0.5:
        warnings.append(f">{pct_single*100:.0f}% of objects are visible in only one frame — re-ID is failing or scene is over-segmented.")
    if volumes and (all(v > 100 for v in volumes) or all(0 < v < 0.001 for v in volumes)):
        warnings.append("All object volumes are extreme (>100 m^3 or <0.001 m^3) — likely a depth-units or scale mismatch.")

    return {
        "n_objects": n,
        "n_categories": len(counts),
        "category_counts": dict(counts),
        "pct_single_frame": pct_single,
        "warnings": warnings,
    }


def render_memory_html(cache_dir: str | Path, paper_code_dir: str | Path, frame_stride: int = 10) -> Path:
    """Generate `_inspect/memory.html` for a cache dir.

    Loads the cache via AgentContext (without VLM/text encoder — those are not
    needed for inspection), computes summary, renders per-object thumbnails
    with their 3D bbox projected onto the best frame, and per-frame stamps
    every `frame_stride` frames.
    """
    from eva_eval.agent.context import AgentContext

    cache_dir = Path(cache_dir)
    ctx = AgentContext.load(
        video_cache_dir=cache_dir,
        paper_code_dir=paper_code_dir,
        vlm=None,
        text_encoder=None,
    )

    objects = list(ctx.object_index.values())
    summary = summarize_memory(objects, objects_frames=ctx.objects_frames)

    body_parts: list[str] = []
    body_parts.append(f"<h1>Memory inspection — <code>{cache_dir.name}</code></h1>")
    body_parts.append("<h2>Summary</h2>")
    body_parts.append(_summary_table(summary))
    body_parts.append("".join(_warn(w) for w in summary["warnings"]))

    body_parts.append("<h2>Object catalog</h2>")
    body_parts.append(_object_catalog(ctx, objects))

    body_parts.append(f"<h2>Frame stamps (every {frame_stride}th frame)</h2>")
    body_parts.append(_frame_stamps(ctx, frame_stride=frame_stride))

    return write_html(
        cache_dir / "_inspect" / "memory.html",
        title=f"memory: {cache_dir.name}",
        body="\n".join(body_parts),
    )


def _summary_table(summary: dict) -> str:
    counts = ", ".join(f"{k}={v}" for k, v in sorted(summary["category_counts"].items(), key=lambda kv: -kv[1]))
    rows = [
        ("n_objects", summary["n_objects"]),
        ("n_categories", summary["n_categories"]),
        ("category counts", counts or "(none)"),
        ("% single-frame objects", f"{summary['pct_single_frame']*100:.1f}%"),
    ]
    body = "".join(f"<tr><th>{k}</th><td><code>{v}</code></td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _warn(msg: str) -> str:
    return f'<div class="warn">⚠ {msg}</div>'


def _object_catalog(ctx, objects: list) -> str:
    by_visibility = sorted(objects, key=lambda o: -len(ctx.objects_frames.get(int(o.identifier), [])))
    rows = ["<tr><th>id</th><th>category</th><th>frames</th><th>volume</th><th>state</th><th>best-frame thumb</th></tr>"]
    for o in by_visibility:
        oid = int(o.identifier)
        frames = ctx.objects_frames.get(oid, [])
        thumb_html = "—"
        if frames:
            try:
                img = ctx.render_object_bbox(oid, frames[0])
                thumb_html = f'<img class="thumb" src="{image_to_data_uri(img)}">'
            except Exception as e:
                thumb_html = f"(render error: {type(e).__name__})"
        rows.append(
            f"<tr><td>{oid}</td>"
            f"<td>{getattr(o, 'category', '?')}</td>"
            f"<td>{len(frames)}</td>"
            f"<td>{float(getattr(o, 'volume', 0.0)):.3f}</td>"
            f"<td>{getattr(o, 'state', '?')}</td>"
            f"<td>{thumb_html}</td></tr>"
        )
    return f"<table>{''.join(rows)}</table>"


def _frame_stamps(ctx, *, frame_stride: int) -> str:
    from PIL import Image

    parts = []
    n = len(ctx.frame_index)
    for fi_idx in range(0, n, frame_stride):
        fi = ctx.frame_index[fi_idx]
        if not fi.path.exists() or not fi.visible_object_ids:
            continue
        img = Image.open(fi.path).convert("RGB")
        # Composite each visible object's 3D bbox onto the frame.
        for oid in fi.visible_object_ids:
            try:
                img_with = ctx.render_object_bbox(int(oid), fi.frame_id)
                img = img_with  # render_object_bbox returns a fresh image with the bbox drawn
            except Exception:
                continue
        labels = ", ".join(f"{int(o)}:{getattr(ctx.object_index.get(int(o)), 'category', '?')}"
                           for o in fi.visible_object_ids[:8])
        parts.append(
            f'<div><img class="frame" src="{image_to_data_uri(img)}">'
            f"<div>frame {fi.frame_id} — {labels}</div></div>"
        )
    return f'<div class="row">{"".join(parts)}</div>'
