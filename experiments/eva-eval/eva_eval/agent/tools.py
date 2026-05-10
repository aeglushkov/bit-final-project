from __future__ import annotations

import ast
import json
from typing import Any

import numpy as np

from eva_eval.agent.context import AgentContext
from eva_eval.agent.sql import build_sql_db, execute_readonly, format_rows

DESCRIBE_THEN_ANSWER = (
    "First, briefly describe what is shown. Then, on a new line starting with "
    '"Answer to the question:", answer the following: {question}'
)
DESCRIBE_OBJECT_PROMPT = "Identify and briefly describe the object inside the bounding box."
TOP_K_RETRIEVAL = 10
TOP_K_FRAME_LOC = 5


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _format_dict_as_text(d: dict[Any, Any]) -> str:
    if not d:
        return "(no matches)"
    parts = []
    for k, v in d.items():
        v_text = " ".join(str(v).split())
        parts.append(f"{k}: {v_text!r}")
    return "{" + ", ".join(parts) + "}"


def parse_tuple_input(s: str, expected_arity: int) -> tuple:
    """Parse the paper's `("text", 16)` Action Input format. expected_arity=1
    accepts bare strings; >=2 expects a tuple."""
    s = s.strip()
    if expected_arity == 1:
        try:
            value = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return (s.strip("\"'"),)
        if isinstance(value, tuple):
            if len(value) != 1:
                raise ValueError(f"Expected 1 argument, got {len(value)}: {s!r}")
            return value
        return (value,)

    try:
        value = ast.literal_eval(s)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Could not parse tuple input {s!r}: {e}")
    if not isinstance(value, tuple):
        raise ValueError(f"Expected a tuple of {expected_arity} args, got {type(value).__name__}: {s!r}")
    if len(value) != expected_arity:
        raise ValueError(f"Expected {expected_arity} args, got {len(value)}: {s!r}")
    return value


def do_retrieve_objects_by_appearance(ctx: AgentContext, text: str) -> str:
    return _do_retrieve(ctx, text, feature_attr="object_clip_feature")


def do_retrieve_objects_by_environment(ctx: AgentContext, text: str) -> str:
    return _do_retrieve(ctx, text, feature_attr="context_clip_feature")


def _do_retrieve(ctx: AgentContext, text: str, feature_attr: str) -> str:
    text_emb = ctx.encode_text(text)
    scored: list[tuple[float, int]] = []
    for oid, obj in ctx.object_index.items():
        feat = getattr(obj, feature_attr, None)
        if feat is None:
            continue
        scored.append((_cosine(text_emb, feat), oid))
    scored.sort(reverse=True)
    top = [oid for _, oid in scored[:TOP_K_RETRIEVAL]]

    captions: dict[int, str] = {}
    for oid in top:
        try:
            frame_id = ctx.best_frame_for_object(oid)
        except KeyError:
            continue
        try:
            image = ctx.render_object_bbox(oid, frame_id)
            cap = ctx.vlm.vqa(image, DESCRIBE_OBJECT_PROMPT)
        except Exception as e:
            cap = f"(VLM error: {e})"
        captions[oid] = cap
    return f"The objects that satisfy '{text}' are " + _format_dict_as_text(captions)


def do_frame_localization(ctx: AgentContext, text: str) -> str:
    text_emb = ctx.encode_text(text)
    scored: list[tuple[float, int]] = []
    for fi in ctx.frame_index:
        if fi.ctx_feat is None:
            continue
        scored.append((_cosine(text_emb, fi.ctx_feat), fi.frame_id))
    scored.sort(reverse=True)
    top = [fid for _, fid in scored[:TOP_K_FRAME_LOC]]
    return "The most relevant frame indices are " + json.dumps(top)


def do_frame_vqa(ctx: AgentContext, question: str, frame_id: int) -> str:
    if frame_id < 0 or frame_id >= len(ctx.frame_index):
        return f"(frame_id {frame_id} out of range; valid range is [0, {len(ctx.frame_index) - 1}])"
    image = ctx.load_frame(frame_id)
    return ctx.vlm.vqa(image, DESCRIBE_THEN_ANSWER.format(question=question))


def do_object_vqa(ctx: AgentContext, question: str, object_id: int) -> str:
    if object_id not in ctx.object_index:
        return f"(object_id {object_id} not found in memory)"
    try:
        frame_id = ctx.best_frame_for_object(object_id)
    except KeyError as e:
        return f"({e})"
    image = ctx.render_object_bbox(object_id, frame_id)
    return ctx.vlm.vqa(image, DESCRIBE_THEN_ANSWER.format(question=question))


def do_query_db(ctx: AgentContext, sql: str) -> str:
    _ensure_sql_conn(ctx)
    try:
        cols, rows = execute_readonly(ctx._sql_conn, sql)
    except Exception as e:
        return f"(SQL error: {e})"
    return format_rows(cols, rows)


def _ensure_sql_conn(ctx: AgentContext) -> None:
    if getattr(ctx, "_sql_conn", None) is None:
        timestamps = ctx.meta["timestamps"]
        frame_id_for_ts = {float(ts): i for i, ts in enumerate(timestamps)}
        ctx._sql_conn = build_sql_db(
            ctx.object_index.values(),
            frame_id_for_ts,
            ctx.memory_state.get("temporal_info", {}),
            extended=bool(getattr(ctx, "_extended_schema", False)),
        )


def do_get_object_dimensions(ctx: AgentContext, object_id: int) -> str:
    """Return L/W/H of one object in centimeters, computed from its 3D AABB."""
    if int(object_id) not in ctx.object_index:
        return f"(object_id {object_id} not found in memory)"
    obj = ctx.object_index[int(object_id)]
    size = obj.max_xyz - obj.min_xyz
    length_cm = float(size[0]) * 100.0
    height_cm = float(size[1]) * 100.0
    width_cm = float(size[2]) * 100.0
    longest_cm = max(length_cm, height_cm, width_cm)
    return (
        f"category={obj.category!r} length={length_cm:.1f} cm  width={width_cm:.1f} cm  "
        f"height={height_cm:.1f} cm  longest_dimension={longest_cm:.1f} cm"
    )


def do_get_distance(ctx: AgentContext, object_id_a: int, object_id_b: int) -> str:
    """Return the closest-point distance in meters between two objects' AABBs.

    Per VSI-Bench `object_abs_distance` semantics: 'measuring from the closest point
    of each object'. Computed as max(0, gap_per_axis) Euclidean."""
    a = ctx.object_index.get(int(object_id_a))
    b = ctx.object_index.get(int(object_id_b))
    if a is None or b is None:
        missing = [oid for oid in (object_id_a, object_id_b) if int(oid) not in ctx.object_index]
        return f"(object_id(s) not found: {missing})"
    gap = np.maximum(0.0, np.maximum(a.min_xyz - b.max_xyz, b.min_xyz - a.max_xyz))
    dist_m = float(np.linalg.norm(gap))
    return f"closest-point distance({a.category}#{int(object_id_a)} <-> {b.category}#{int(object_id_b)}) = {dist_m:.3f} m"


def do_estimate_room_size(ctx: AgentContext) -> str:
    """Estimate the room's floor area in square meters from the AABBs in memory.

    Heuristic: the convex hull of object center points gives a rough floor span.
    For complex layouts this overestimates (walks across two rooms) or
    underestimates (the agent only sees part of the room). Reported with bounds
    so the agent can reason about confidence."""
    objs = list(ctx.object_index.values())
    if len(objs) < 3:
        return f"(only {len(objs)} objects in memory; cannot estimate room size)"
    centers = np.array([(o.min_xyz + o.max_xyz) / 2.0 for o in objs])
    # Use horizontal axes (x and z in OpenCV cam2world convention)
    x = centers[:, 0]
    z = centers[:, 2]
    span_x = float(x.max() - x.min())
    span_z = float(z.max() - z.min())
    bbox_area = span_x * span_z
    # Convex-hull area (more honest estimate for L-shaped rooms)
    try:
        from scipy.spatial import ConvexHull

        hull_area = float(ConvexHull(np.stack([x, z], axis=-1)).volume)
    except Exception:
        hull_area = bbox_area
    return (
        f"room size estimate (square meters): "
        f"convex_hull={hull_area:.2f}  bbox_span={bbox_area:.2f}  "
        f"x_span={span_x:.2f} m  z_span={span_z:.2f} m  n_objects={len(objs)}"
    )


def make_tools(ctx: AgentContext, *, extended_schema: bool = False):
    """Build the tool list for the ReAct executor.

    `extended_schema=True` does two things:
      - Tells the SQL builder (via ctx._extended_schema) to include bbox extents.
      - Adds three computed-answer tools: get_object_dimensions, get_distance,
        estimate_room_size — needed for VSI-Bench's strict numeric questions
        (object size, abs distance, room size) which the paper's schema cannot
        answer.
    """
    from langchain_core.tools import tool

    ctx._extended_schema = bool(extended_schema)
    schema_blurb = (
        "Objects(object_id, category, volume, min_x, min_y, min_z, max_x, max_y, max_z, cx, cy, cz, length_m, width_m, height_m, longest_edge_m)"
        if extended_schema
        else "Objects(object_id, category, volume)"
    )

    @tool
    def retrieve_objects_by_appearance(text: str) -> str:
        """Return the top-10 candidate object IDs from the persistent object memory whose appearance (visual feature) matches the description (e.g. "brown chair"). Returns a dict keyed by object_id with a brief VLM-generated caption per candidate."""
        return do_retrieve_objects_by_appearance(ctx, text.strip().strip("\"'"))

    @tool
    def retrieve_objects_by_environment(text: str) -> str:
        """Return the top-10 candidate object IDs whose surrounding environment (context feature) matches the description (e.g. "kitchen counter"). Returns a dict keyed by object_id with a brief VLM-generated caption per candidate."""
        return do_retrieve_objects_by_environment(ctx, text.strip().strip("\"'"))

    @tool
    def frame_localization(text: str) -> str:
        """Return the top-5 frame IDs most relevant to the description (e.g. "when I walk in the front door"). Use this before frame_VQA to pick which frames to inspect."""
        return do_frame_localization(ctx, text.strip().strip("\"'"))

    @tool
    def frame_VQA(action_input: str) -> str:
        """Run visual question answering on a specific frame. Action Input format: ("question", frame_id). The VLM first describes the frame, then answers the question."""
        try:
            question, frame_id = parse_tuple_input(action_input, expected_arity=2)
            frame_id = int(frame_id)
        except (ValueError, TypeError) as e:
            return (
                f"(input parse error: {e}; expected format: (\"question\", N) "
                "where N is an integer frame index from frame_localization)"
            )
        return do_frame_vqa(ctx, str(question), frame_id)

    @tool
    def object_VQA(action_input: str) -> str:
        """Run visual question answering on a specific object, with its 3D bounding box rendered on a frame containing it. Action Input format: ("question", object_id)."""
        try:
            question, object_id = parse_tuple_input(action_input, expected_arity=2)
            object_id = int(object_id)
        except (ValueError, TypeError) as e:
            return (
                f"(input parse error: {e}; expected format: (\"question\", N) "
                "where N is an integer object_id from query_db or retrieve_objects_*)"
            )
        return do_object_vqa(ctx, str(question), object_id)

    if extended_schema:
        @tool
        def query_db(sql: str) -> str:
            """Execute a read-only SQL SELECT against tables Objects(object_id, category, volume, min_x, min_y, min_z, max_x, max_y, max_z, cx, cy, cz, length_m, width_m, height_m, longest_edge_m) and Objects_Frames(object_id, frame_id). Use this only when retrieve_objects_* and frame_localization are insufficient."""
            return do_query_db(ctx, sql)
    else:
        @tool
        def query_db(sql: str) -> str:
            """Execute a read-only SQL SELECT against tables Objects(object_id, category, volume) and Objects_Frames(object_id, frame_id). Use this only when retrieve_objects_* and frame_localization are insufficient."""
            return do_query_db(ctx, sql)

    tools = [
        retrieve_objects_by_appearance,
        retrieve_objects_by_environment,
        frame_localization,
        frame_VQA,
        object_VQA,
        query_db,
    ]

    if extended_schema:
        @tool
        def get_object_dimensions(object_id: str) -> str:
            """Return length, width, height of one object in CENTIMETERS, computed from its 3D AABB. Use this for VSI-Bench `object_size_estimation` questions ("longest dimension of X in cm"). Action Input: an integer object_id."""
            try:
                oid = int(str(object_id).strip().strip("\"'"))
            except ValueError as e:
                return f"(input parse error: {e}; expected an integer object_id)"
            return do_get_object_dimensions(ctx, oid)

        @tool
        def get_distance(action_input: str) -> str:
            """Return the closest-point distance in METERS between two objects. Use this for VSI-Bench `object_abs_distance` questions ("distance between A and B in meters"). Action Input format: (object_id_a, object_id_b)."""
            try:
                a, b = parse_tuple_input(action_input, expected_arity=2)
                a = int(a)
                b = int(b)
            except (ValueError, TypeError) as e:
                return f"(input parse error: {e}; expected format: (id_a, id_b))"
            return do_get_distance(ctx, a, b)

        @tool
        def estimate_room_size(_: str = "") -> str:
            """Estimate the room's floor area in SQUARE METERS from the AABBs in memory (convex-hull and bbox-span estimates of object centers). Use this for VSI-Bench `room_size_estimation` questions. Action Input: anything (ignored)."""
            return do_estimate_room_size(ctx)

        tools.extend([get_object_dimensions, get_distance, estimate_room_size])

    return tools
