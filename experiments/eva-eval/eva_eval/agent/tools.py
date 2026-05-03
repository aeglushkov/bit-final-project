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
    if not hasattr(ctx, "_sql_conn") or ctx._sql_conn is None:
        timestamps = ctx.meta["timestamps"]
        frame_id_for_ts = {float(ts): i for i, ts in enumerate(timestamps)}
        ctx._sql_conn = build_sql_db(
            ctx.object_index.values(),
            frame_id_for_ts,
            ctx.memory_state.get("temporal_info", {}),
        )
    try:
        cols, rows = execute_readonly(ctx._sql_conn, sql)
    except Exception as e:
        return f"(SQL error: {e})"
    return format_rows(cols, rows)


def make_tools(ctx: AgentContext):
    from langchain_core.tools import tool

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

    @tool
    def query_db(sql: str) -> str:
        """Execute a read-only SQL SELECT against tables Objects(object_id, category, volume) and Objects_Frames(object_id, frame_id). Use this only when retrieve_objects_* and frame_localization are insufficient."""
        return do_query_db(ctx, sql)

    return [
        retrieve_objects_by_appearance,
        retrieve_objects_by_environment,
        frame_localization,
        frame_VQA,
        object_VQA,
        query_db,
    ]
