from __future__ import annotations

from pathlib import Path


def sample_video_frames(
    video_path: str | Path,
    fps: float,
    out_dir: str | Path,
    image_format: str = "jpg",
    max_frames: int | None = None,
) -> tuple[list[Path], list[float]]:
    import cv2

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        cap.release()
        raise RuntimeError(f"Bad source FPS {src_fps} from {video_path}")

    step = max(1, int(round(src_fps / fps)))
    frame_paths: list[Path] = []
    timestamps: list[float] = []

    src_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if src_idx % step == 0:
            out_path = out_dir / f"{saved:06d}.{image_format}"
            cv2.imwrite(str(out_path), frame)
            frame_paths.append(out_path)
            timestamps.append(src_idx / src_fps)
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        src_idx += 1
    cap.release()

    return frame_paths, timestamps
