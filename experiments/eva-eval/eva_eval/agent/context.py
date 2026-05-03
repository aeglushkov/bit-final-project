from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FrameInfo:
    frame_id: int
    path: Path
    timestamp: float
    ctx_feat: np.ndarray | None
    visible_object_ids: list[int]


class AgentContext:
    """Per-video state used by the six tools.

    Loads:
      - the pickled `ObjectMemory` state from Phase 3
      - the frame index (paths + ctx_feat + visibility) from `temporal_info` + `meta.json`
      - intrinsics (3x3 K) and per-frame cam2world poses from Phase 2

    Holds clients (planner / VLM / CLIP text encoder) and provides helper
    methods used by the tool implementations.
    """

    def __init__(
        self,
        video_cache_dir: Path,
        memory_state: dict[str, Any],
        meta: dict[str, Any],
        intrinsics: dict[str, float],
        poses: np.ndarray,
        vlm,
        text_encoder=None,
    ):
        self.video_cache_dir = Path(video_cache_dir)
        self.memory_state = memory_state
        self.meta = meta
        self.intrinsics = intrinsics
        self.poses = poses
        self.vlm = vlm
        self.text_encoder = text_encoder

        all_objects = list(memory_state.get("static_objects", [])) + list(memory_state.get("dynamic_objects", []))
        self.object_index: dict[int, Any] = {int(o.identifier): o for o in all_objects}

        self.frame_index: list[FrameInfo] = self._build_frame_index()
        self.objects_frames: dict[int, list[int]] = self._build_objects_frames()
        self.K: np.ndarray = self._build_K()

    @classmethod
    def load(
        cls,
        video_cache_dir: str | Path,
        paper_code_dir: str | Path,
        vlm=None,
        text_encoder=None,
        memory_name: str = "memory.pkl",
    ) -> "AgentContext":
        from eva_eval.memory.store import load_memory

        video_cache_dir = Path(video_cache_dir)
        paper_code_dir = Path(paper_code_dir).resolve()
        if str(paper_code_dir) not in sys.path:
            sys.path.insert(0, str(paper_code_dir))

        memory_state = load_memory(video_cache_dir / memory_name)
        meta = json.loads((video_cache_dir / "meta.json").read_text())
        intrinsics = json.loads((video_cache_dir / "intrinsics.json").read_text())
        poses = np.load(video_cache_dir / "poses.npy")
        return cls(
            video_cache_dir=video_cache_dir,
            memory_state=memory_state,
            meta=meta,
            intrinsics=intrinsics,
            poses=poses,
            vlm=vlm,
            text_encoder=text_encoder,
        )

    def _build_frame_index(self) -> list[FrameInfo]:
        timestamps: list[float] = self.meta["timestamps"]
        temporal_info: dict = self.memory_state.get("temporal_info", {})
        frames_dir = self.video_cache_dir / "frames"
        out: list[FrameInfo] = []
        for i, ts in enumerate(timestamps):
            info = temporal_info.get(ts) or temporal_info.get(float(ts), {})
            out.append(
                FrameInfo(
                    frame_id=i,
                    path=frames_dir / f"{i:06d}.jpg",
                    timestamp=float(ts),
                    ctx_feat=info.get("clip_feature") if isinstance(info, dict) else None,
                    visible_object_ids=list(info.get("visible_object_identifiers", []) if isinstance(info, dict) else []),
                )
            )
        return out

    def _build_objects_frames(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for fi in self.frame_index:
            for oid in fi.visible_object_ids:
                out[int(oid)].append(fi.frame_id)
        return dict(out)

    def _build_K(self) -> np.ndarray:
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = self.intrinsics["fx"]
        K[1, 1] = self.intrinsics["fy"]
        K[0, 2] = self.intrinsics["cx"]
        K[1, 2] = self.intrinsics["cy"]
        return K

    def encode_text(self, text: str) -> np.ndarray:
        if self.text_encoder is None:
            raise RuntimeError("AgentContext.text_encoder is not configured")
        return self.text_encoder(text)

    def best_frame_for_object(self, object_id: int) -> int:
        frames = self.objects_frames.get(int(object_id), [])
        if not frames:
            raise KeyError(f"Object {object_id} not visible in any frame")
        return frames[0]

    def render_object_bbox(self, object_id: int, frame_id: int):
        from eva_eval.agent.visual import render_3d_bbox_on_frame

        obj = self.object_index[int(object_id)]
        return render_3d_bbox_on_frame(
            frame_path=self.frame_index[frame_id].path,
            min_xyz=obj.min_xyz,
            max_xyz=obj.max_xyz,
            cam2world=self.poses[frame_id],
            K=self.K,
        )

    def load_frame(self, frame_id: int):
        from PIL import Image

        return Image.open(self.frame_index[int(frame_id)].path).convert("RGB")
