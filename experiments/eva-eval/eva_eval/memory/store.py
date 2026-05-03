from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any


def save_memory(memory, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = _extract_state(memory)
    with out_path.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_memory(in_path: str | Path, paper_code_dir: str | Path | None = None) -> dict[str, Any]:
    """Unpickle a saved ObjectMemory state.

    The state contains paper-code Object3D instances; the paper's `object3d`
    module must be importable. If `paper_code_dir` is provided, it's prepended
    to sys.path before unpickling.
    """
    if paper_code_dir is not None:
        s = str(Path(paper_code_dir).resolve())
        if s not in sys.path:
            sys.path.insert(0, s)
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
