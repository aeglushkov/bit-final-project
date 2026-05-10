"""Render annotated frames for selected VSI-Bench questions.

For each selection (from `10_pick_visuals.py`):
- parse object categories mentioned in the question text
- pick the frame that best shows those question-target categories (fall back to
  the most-visible-objects frame if none of the target categories were detected
  in this scene)
- draw a 2D bounding rect per visible object (cleaner than 3D wireframes —
  MASt3R's world frame isn't gravity-aligned, so 3D edges look tilted)
- highlight question-target categories with a thicker stroke + label; show
  context objects with a thin stroke
- caption strip lists question, GT, prediction, score, parsed categories,
  which were actually found in this scene's memory, and the frame chosen

Run on remote (no GPU / VLM needed):
    python scripts/10_render_visuals.py \
        --selections results/visuals/<run>/selections.json \
        --cache-root cache/vsibench \
        --paper-code-dir literature/EmbodiedVideoAgent/code \
        --out-dir results/visuals/<run>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eva_eval.agent.context import AgentContext  # noqa: E402
from eva_eval.agent.visual import aabb_corners_world, project_world_to_pixels  # noqa: E402

# 12-color palette (matplotlib tab10/12-ish).
PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
]


def category_color(name: str) -> tuple[int, int, int]:
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return PALETTE[h % len(PALETTE)]


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def parse_question_categories(question: str, vocab: set[str]) -> list[str]:
    """Find detection-vocabulary categories mentioned in the question.

    Matches whole words, longest-first, case-insensitive. Includes simple
    plural handling: a category like 'chair' matches 'chair', 'chairs'.
    """
    q = " " + re.sub(r"[^a-z0-9 ]", " ", question.lower()) + " "
    matched = []
    seen = set()
    for cat in sorted(vocab, key=len, reverse=True):
        c = cat.lower()
        if not c:
            continue
        # Match the literal category, with optional 's'/'es' plural, as a
        # whole word. Also handles multi-word categories.
        pat = r"\b" + re.escape(c) + r"(?:s|es)?\b"
        if re.search(pat, q) and c not in seen:
            matched.append(c)
            seen.add(c)
    return matched


def best_frame_for_question(ctx: AgentContext, question: str) -> tuple[int, list[str], list[str]]:
    """Choose the frame that maximises visibility of question-mentioned categories.

    Returns (frame_id, parsed_target_cats, found_in_memory_cats). Falls back to
    the frame with the most visible objects when no targets are present in
    this scene's memory.
    """
    cats_in_memory: set[str] = {str(o.category).lower() for o in ctx.object_index.values()}
    target_cats = parse_question_categories(question, cats_in_memory)
    found = [c for c in target_cats if c in cats_in_memory]

    if not found:
        # No target objects in this scene's memory; show the most-visible frame.
        counts = [(len(fi.visible_object_ids), -i, i) for i, fi in enumerate(ctx.frame_index)]
        counts.sort(reverse=True)
        return counts[0][2], target_cats, found

    target_set = set(found)
    best = (-1, -1, 0)  # (n_target_visible, n_total_visible, frame_id)
    for i, fi in enumerate(ctx.frame_index):
        visible_cats = {str(ctx.object_index[oid].category).lower()
                        for oid in fi.visible_object_ids if oid in ctx.object_index}
        n_target = len(target_set & visible_cats)
        n_total = len(fi.visible_object_ids)
        key = (n_target, n_total)
        if key > (best[0], best[1]):
            best = (n_target, n_total, i)
    return best[2], target_cats, found


def projected_rect(uv: np.ndarray, in_front: np.ndarray, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Axis-aligned image-space bounding rect of the projected AABB corners.

    Returns (u0, v0, u1, v1) clipped to image bounds, or None if the box has
    no in-front corners or projects entirely off-frame.
    """
    W, H = size
    valid = uv[in_front]
    if len(valid) < 2:
        return None
    u_min, v_min = valid.min(axis=0)
    u_max, v_max = valid.max(axis=0)
    # discard if entirely off-frame
    if u_max < 0 or u_min > W or v_max < 0 or v_min > H:
        return None
    # discard degenerate (1-pixel) projections — usually behind the camera or numerical noise
    if (u_max - u_min) < 2 or (v_max - v_min) < 2:
        return None
    u0 = int(round(max(0.0, min(u_min, W - 1))))
    v0 = int(round(max(0.0, min(v_min, H - 1))))
    u1 = int(round(max(0.0, min(u_max, W - 1))))
    v1 = int(round(max(0.0, min(v_max, H - 1))))
    return u0, v0, u1, v1


def _draw_label(draw: ImageDraw.ImageDraw, anchor: tuple[int, int], text: str,
                font: ImageFont.ImageFont, color: tuple[int, int, int],
                image_size: tuple[int, int] | None = None) -> None:
    u0, v0 = anchor
    label_h = font.size + 4
    # Place above the rect by default; if that would clip the top, drop it just inside.
    ty = v0 - label_h - 2 if v0 - label_h - 2 >= 0 else v0 + 1
    tx = u0
    if image_size is not None:
        W, _H = image_size
        # Keep label horizontally inside image
        text_w = int(draw.textlength(text, font=font))
        tx = max(0, min(tx, W - text_w - 2))
    bbox = draw.textbbox((tx, ty), text, font=font)
    draw.rectangle(bbox, fill=(255, 255, 255))
    draw.text((tx, ty), text, fill=color, font=font)


def annotate_frame(ctx: AgentContext, frame_id: int, target_cats: set[str]) -> tuple[Image.Image, int, int]:
    img = Image.open(ctx.frame_index[frame_id].path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    target_font = load_font(max(11, H // 36))
    context_font = load_font(max(9, H // 52))
    base_w = max(1, H // 360)
    visible = ctx.frame_index[frame_id].visible_object_ids

    target_records: list[dict] = []
    context_records: list[dict] = []
    for oid in visible:
        obj = ctx.object_index.get(int(oid))
        if obj is None:
            continue
        cat = str(getattr(obj, "category", "obj")).lower()
        is_target = cat in target_cats
        corners = aabb_corners_world(obj.min_xyz, obj.max_xyz)
        uv, in_front = project_world_to_pixels(corners, ctx.poses[frame_id], ctx.K)
        rect = projected_rect(uv, in_front, (W, H))
        if rect is None:
            continue
        rec = {"oid": int(obj.identifier), "obj": obj, "rect": rect, "cat": cat}
        (target_records if is_target else context_records).append(rec)

    # Draw context boxes first so targets layer on top.
    for rec in context_records:
        color = category_color(rec["cat"])
        draw.rectangle(rec["rect"], outline=color, width=base_w)
    # Then targets with thick strokes.
    for rec in target_records:
        color = category_color(rec["cat"])
        draw.rectangle(rec["rect"], outline=color, width=base_w * 3)

    # Labels: context first (small font), then targets (large font) so they overlay.
    for rec in context_records:
        color = category_color(rec["cat"])
        u0, v0, _, _ = rec["rect"]
        _draw_label(draw, (u0, v0), f'{rec["cat"]}#{rec["oid"]}', context_font, color, (W, H))
    for rec in target_records:
        color = category_color(rec["cat"])
        u0, v0, _, _ = rec["rect"]
        _draw_label(draw, (u0, v0), f'{rec["cat"]}#{rec["oid"]}', target_font, color, (W, H))

    return img, len(target_records), len(context_records)


def caption_panel(width: int, lines: list[tuple[str, tuple[int, int, int]]],
                  font: ImageFont.ImageFont, pad: int = 12) -> Image.Image:
    line_h = font.size + 6
    h = pad * 2 + line_h * len(lines)
    panel = Image.new("RGB", (width, h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    y = pad
    for text, color in lines:
        draw.text((pad, y), text, fill=color, font=font)
        y += line_h
    return panel


def render_one(ctx: AgentContext, sel: dict, out_path: Path, caption_font_size: int = 18) -> None:
    frame_id, target_cats, found_in_mem = best_frame_for_question(ctx, sel["question"])
    img, n_target, n_context = annotate_frame(ctx, frame_id, set(found_in_mem))
    W, _H = img.size

    qt_label = sel["reported_type"]
    if sel["question_type"] != sel["reported_type"]:
        qt_label = f'{sel["reported_type"]}  ({sel["question_type"]})'

    bucket_color = {
        "good": (0, 128, 0),
        "mediocre": (200, 130, 0),
        "bad": (200, 0, 0),
    }.get(sel["bucket"], (0, 0, 0))

    body_font = load_font(caption_font_size - 2)
    char_w = max(7, caption_font_size // 2)
    wrap_at = max(40, (W - 24) // char_w)
    q_lines = textwrap.wrap(f'Q: {sel["question"]}', width=wrap_at) or [f'Q: {sel["question"]}']

    cats_str = ", ".join(target_cats) if target_cats else "(none parsed from question)"
    found_str = ", ".join(found_in_mem) if found_in_mem else "(none — perception missed them)"
    found_color = (0, 128, 0) if found_in_mem and set(found_in_mem) == set(target_cats) else (200, 130, 0)
    if not found_in_mem and target_cats:
        found_color = (200, 0, 0)

    lines: list[tuple[str, tuple[int, int, int]]] = []
    lines.append((f'[{qt_label}]  bucket={sel["bucket"]}  score={float(sel["score"]):.2f}  scene={sel["scene_name"]}  qid={sel["id"]}', bucket_color))
    for line in q_lines:
        lines.append((line, (0, 0, 0)))
    lines.append((f'GT:   {sel["ground_truth"]}', (0, 0, 0)))
    lines.append((f'Pred: {sel["prediction"]}', bucket_color))
    lines.append((f'Q-categories: {cats_str}', (0, 0, 0)))
    lines.append((f'Found in memory: {found_str}', found_color))
    lines.append((f'frame_id={frame_id}  target_boxes={n_target}  context_boxes={n_context}', (110, 110, 110)))

    panel = caption_panel(W, lines, body_font)
    out = Image.new("RGB", (W, img.height + panel.height), (255, 255, 255))
    out.paste(img, (0, 0))
    out.paste(panel, (0, img.height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--paper-code-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    sels = json.loads(args.selections.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for s in sels:
        by_scene[s["scene_name"]].append(s)

    n_ok = 0
    n_skip = 0
    n_fail = 0
    for scene_name, group in by_scene.items():
        scene_dir = args.cache_root / scene_name
        if not (scene_dir / "memory.pkl").exists():
            print(f"[skip] missing memory.pkl for scene {scene_name}", file=sys.stderr)
            n_skip += len(group)
            continue
        try:
            ctx = AgentContext.load(scene_dir, args.paper_code_dir)
        except Exception as e:
            print(f"[fail] AgentContext.load({scene_name}): {e}", file=sys.stderr)
            traceback.print_exc()
            n_fail += len(group)
            continue
        for sel in group:
            out_path = args.out_dir / f'{sel["reported_type"]}__{sel["bucket"]}__qid{sel["id"]}.png'
            try:
                render_one(ctx, sel, out_path)
                print(f"[ok] {out_path.name}")
                n_ok += 1
            except Exception as e:
                print(f"[fail] {sel['reported_type']}/{sel['bucket']} qid={sel['id']}: {e}", file=sys.stderr)
                traceback.print_exc()
                n_fail += 1

    print(f"\nok={n_ok}  skipped={n_skip}  failed={n_fail}")


if __name__ == "__main__":
    main()
