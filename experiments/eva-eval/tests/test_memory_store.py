from eva_eval.memory.store import load_memory, save_memory


class _FakeMemory:
    def __init__(self):
        self.static_objects = ["obj_a", "obj_b"]
        self.dynamic_objects = []
        self.frames = [{"frame_id": 0, "timestamp": 0.0}, {"frame_id": 1, "timestamp": 1.0}]
        self.temporal_info = {0: {"caption": "a kitchen scene"}}
        self.object_identifier_cnt = 2


def test_save_load_roundtrip(tmp_path):
    fake = _FakeMemory()
    out = tmp_path / "memory.pkl"
    save_memory(fake, out)
    state = load_memory(out)
    assert state["static_objects"] == ["obj_a", "obj_b"]
    assert state["dynamic_objects"] == []
    assert state["object_identifier_cnt"] == 2
    assert state["frames"][1]["timestamp"] == 1.0
    assert state["temporal_info"][0]["caption"] == "a kitchen scene"


def test_save_creates_parent_dirs(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "memory.pkl"
    save_memory(_FakeMemory(), deep)
    assert deep.exists()


def test_load_memory_missing_attrs_defaults(tmp_path):
    class Bare:
        pass

    out = tmp_path / "bare.pkl"
    save_memory(Bare(), out)
    state = load_memory(out)
    assert state["static_objects"] == []
    assert state["dynamic_objects"] == []
    assert state["frames"] == []
    assert state["temporal_info"] == {}
    assert state["object_identifier_cnt"] == 0
