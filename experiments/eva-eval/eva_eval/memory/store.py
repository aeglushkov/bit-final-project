from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_memory(memory, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = _extract_state(memory)
    with out_path.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_memory(in_path: str | Path) -> dict[str, Any]:
    with Path(in_path).open("rb") as f:
        return pickle.load(f)


def _extract_state(memory) -> dict[str, Any]:
    return {
        "static_objects": getattr(memory, "static_objects", []),
        "dynamic_objects": getattr(memory, "dynamic_objects", []),
        "frames": getattr(memory, "frames", []),
        "temporal_info": dict(getattr(memory, "temporal_info", {})),
        "object_identifier_cnt": getattr(memory, "object_identifier_cnt", 0),
    }
