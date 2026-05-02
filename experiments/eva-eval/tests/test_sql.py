import numpy as np
import pytest

from eva_eval.agent.sql import build_sql_db, execute_readonly, format_rows


class _FakeObj:
    def __init__(self, identifier, category, min_xyz, max_xyz):
        self.identifier = identifier
        self.category = category
        self.min_xyz = np.asarray(min_xyz, dtype=np.float64)
        self.max_xyz = np.asarray(max_xyz, dtype=np.float64)


def _scene():
    objs = [
        _FakeObj(1, "chair", [0, 0, 0], [1, 1, 1]),
        _FakeObj(2, "chair", [2, 0, 0], [3, 2, 1]),
        _FakeObj(3, "table", [0, 2, 0], [4, 4, 1]),
    ]
    temporal_info = {
        0.0: {"visible_object_identifiers": [1, 3]},
        1.0: {"visible_object_identifiers": [1, 2]},
        2.0: {"visible_object_identifiers": [2, 3]},
    }
    frame_id_for_ts = {0.0: 0, 1.0: 1, 2.0: 2}
    return objs, temporal_info, frame_id_for_ts


def test_objects_table_has_correct_volumes():
    objs, ti, fmap = _scene()
    conn = build_sql_db(objs, fmap, ti)
    cols, rows = execute_readonly(conn, "SELECT object_id, category, volume FROM Objects ORDER BY object_id")
    assert cols == ["object_id", "category", "volume"]
    assert rows == [(1, "chair", 1.0), (2, "chair", 2.0), (3, "table", 8.0)]


def test_objects_frames_populated():
    objs, ti, fmap = _scene()
    conn = build_sql_db(objs, fmap, ti)
    _, rows = execute_readonly(conn, "SELECT object_id, frame_id FROM Objects_Frames ORDER BY object_id, frame_id")
    assert (1, 0) in rows
    assert (1, 1) in rows
    assert (2, 1) in rows
    assert (3, 0) in rows
    assert (3, 2) in rows


def test_count_chairs_aggregation():
    objs, ti, fmap = _scene()
    conn = build_sql_db(objs, fmap, ti)
    _, rows = execute_readonly(conn, "SELECT COUNT(*) FROM Objects WHERE category = 'chair'")
    assert rows[0][0] == 2


def test_join_objects_with_frames():
    objs, ti, fmap = _scene()
    conn = build_sql_db(objs, fmap, ti)
    _, rows = execute_readonly(
        conn,
        "SELECT DISTINCT o.category FROM Objects o JOIN Objects_Frames f ON o.object_id = f.object_id WHERE f.frame_id = 1 ORDER BY o.category",
    )
    assert [r[0] for r in rows] == ["chair"]


def test_writes_are_rejected():
    objs, ti, fmap = _scene()
    conn = build_sql_db(objs, fmap, ti)
    with pytest.raises(ValueError):
        execute_readonly(conn, "DELETE FROM Objects")
    with pytest.raises(ValueError):
        execute_readonly(conn, "INSERT INTO Objects VALUES (99, 'x', 0)")
    with pytest.raises(ValueError):
        execute_readonly(conn, "UPDATE Objects SET category = 'x'")


def test_with_cte_is_allowed():
    objs, ti, fmap = _scene()
    conn = build_sql_db(objs, fmap, ti)
    cols, rows = execute_readonly(
        conn,
        "WITH t AS (SELECT category, COUNT(*) AS n FROM Objects GROUP BY category) SELECT * FROM t ORDER BY category",
    )
    assert cols == ["category", "n"]
    assert rows == [("chair", 2), ("table", 1)]


def test_format_rows_handles_empty():
    assert format_rows([], []) == "(no rows)"
    out = format_rows(["a", "b"], [(1, 2), (3, 4)])
    assert out.splitlines() == ["a | b", "1 | 2", "3 | 4"]
