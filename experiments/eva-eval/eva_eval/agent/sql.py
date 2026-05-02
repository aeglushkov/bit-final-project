from __future__ import annotations

import re
import sqlite3
from typing import Any, Iterable

_READONLY_RE = re.compile(r"^\s*(?:select|with)\b", re.IGNORECASE)


def build_sql_db(
    objects: Iterable,
    frame_id_for_timestamp: dict[float, int],
    temporal_info: dict[float, dict],
) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE Objects (object_id INTEGER PRIMARY KEY, category TEXT, volume REAL)")
    conn.execute("CREATE TABLE Objects_Frames (object_id INTEGER, frame_id INTEGER)")
    conn.execute("CREATE INDEX idx_of_oid ON Objects_Frames(object_id)")
    conn.execute("CREATE INDEX idx_of_fid ON Objects_Frames(frame_id)")

    for obj in objects:
        size = obj.max_xyz - obj.min_xyz
        volume = float(size[0] * size[1] * size[2])
        conn.execute(
            "INSERT OR REPLACE INTO Objects VALUES (?, ?, ?)",
            (int(obj.identifier), str(obj.category), volume),
        )

    for ts, info in temporal_info.items():
        if ts not in frame_id_for_timestamp:
            continue
        frame_id = frame_id_for_timestamp[ts]
        for oid in info.get("visible_object_identifiers", []) or []:
            conn.execute(
                "INSERT INTO Objects_Frames VALUES (?, ?)",
                (int(oid), int(frame_id)),
            )

    conn.commit()
    return conn


def execute_readonly(conn: sqlite3.Connection, sql: str, max_rows: int = 50) -> tuple[list[str], list[tuple[Any, ...]]]:
    if not _READONLY_RE.match(sql):
        raise ValueError("Only SELECT/WITH queries are permitted on the agent SQL database.")
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchmany(max_rows)
    return cols, rows


def format_rows(cols: list[str], rows: list[tuple[Any, ...]], max_rows: int = 50) -> str:
    if not cols and not rows:
        return "(no rows)"
    lines = [" | ".join(cols)]
    for row in rows[:max_rows]:
        lines.append(" | ".join("" if v is None else str(v) for v in row))
    if len(rows) >= max_rows:
        lines.append(f"... (truncated at {max_rows} rows)")
    return "\n".join(lines)
