from __future__ import annotations

import re
import sqlite3
from typing import Any, Iterable

_READONLY_RE = re.compile(r"^\s*(?:select|with)\b", re.IGNORECASE)


def build_sql_db(
    objects: Iterable,
    frame_id_for_timestamp: dict[float, int],
    temporal_info: dict[float, dict],
    *,
    extended: bool = False,
) -> sqlite3.Connection:
    """Build the in-memory SQL DB the agent's query_db tool reads.

    `extended=False` (paper-faithful): Objects(object_id, category, volume).
    `extended=True`: adds bbox extents and centers in meters, plus length_m/
    width_m/height_m and longest_edge_m. Required for VSI-Bench's numeric
    questions about object dimensions, which the paper schema cannot answer.
    """
    conn = sqlite3.connect(":memory:")
    if extended:
        conn.execute(
            "CREATE TABLE Objects (object_id INTEGER PRIMARY KEY, category TEXT, volume REAL, "
            "min_x REAL, min_y REAL, min_z REAL, "
            "max_x REAL, max_y REAL, max_z REAL, "
            "cx REAL, cy REAL, cz REAL, "
            "length_m REAL, width_m REAL, height_m REAL, longest_edge_m REAL)"
        )
    else:
        conn.execute("CREATE TABLE Objects (object_id INTEGER PRIMARY KEY, category TEXT, volume REAL)")
    conn.execute("CREATE TABLE Objects_Frames (object_id INTEGER, frame_id INTEGER)")
    conn.execute("CREATE INDEX idx_of_oid ON Objects_Frames(object_id)")
    conn.execute("CREATE INDEX idx_of_fid ON Objects_Frames(frame_id)")

    for obj in objects:
        size = obj.max_xyz - obj.min_xyz
        volume = float(size[0] * size[1] * size[2])
        if extended:
            mn = obj.min_xyz
            mx = obj.max_xyz
            cx = float((mn[0] + mx[0]) / 2.0)
            cy = float((mn[1] + mx[1]) / 2.0)
            cz = float((mn[2] + mx[2]) / 2.0)
            # Convention: length=x extent, width=z extent (horizontal, perpendicular),
            # height=y extent (vertical in OpenCV cam2world world frame).
            length = float(size[0])
            height = float(size[1])
            width = float(size[2])
            longest = float(max(length, width, height))
            conn.execute(
                "INSERT OR REPLACE INTO Objects VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    int(obj.identifier), str(obj.category), volume,
                    float(mn[0]), float(mn[1]), float(mn[2]),
                    float(mx[0]), float(mx[1]), float(mx[2]),
                    cx, cy, cz,
                    length, width, height, longest,
                ),
            )
        else:
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
