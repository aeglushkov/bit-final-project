"""HTML and image rendering helpers for the inspect_* scripts.

Produces self-contained static HTML (images embedded as data URIs) so
output can be scp'd off the server and opened in a browser without a
running server."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image


_HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            max-width: 1400px; margin: 1.5em auto; padding: 0 1em; color: #222; }}
    h1, h2, h3 {{ color: #111; }}
    table {{ border-collapse: collapse; margin: 0.5em 0; font-size: 0.9em; }}
    th, td {{ padding: 4px 10px; border: 1px solid #ddd; vertical-align: top; }}
    th {{ background: #f4f4f4; text-align: left; }}
    img.thumb {{ max-width: 240px; max-height: 180px; border: 1px solid #ccc; }}
    img.frame {{ max-width: 100%; border: 1px solid #aaa; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 0.5em; align-items: flex-start; }}
    .warn {{ background: #fff4e6; border-left: 4px solid #d97706; padding: 0.5em 1em; }}
    .err {{ background: #fdecea; border-left: 4px solid #c0392b; padding: 0.5em 1em; }}
    code {{ background: #f4f4f4; padding: 0 4px; border-radius: 3px; font-size: 0.85em; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def html_page(title: str, body: str) -> str:
    """Wrap a body fragment in a full HTML document."""
    return _HTML_TEMPLATE.format(title=title, body=body)


def write_html(path: str | Path, title: str, body: str) -> Path:
    """Convenience: render and write to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_page(title=title, body=body))
    return path


def image_to_data_uri(image: Image.Image) -> str:
    """Encode a PIL image as a base64 data URI suitable for <img src="...">."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def colorize_depth(depth: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> Image.Image:
    """Apply the matplotlib `inferno` colormap to a 2D depth array.
    Constant or empty fields render as a flat color."""
    arr = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return Image.new("RGB", (max(1, arr.shape[1]), max(1, arr.shape[0])), color=(0, 0, 0))
    lo = float(vmin) if vmin is not None else float(arr[finite].min())
    hi = float(vmax) if vmax is not None else float(arr[finite].max())
    span = hi - lo if hi > lo else 1.0
    norm = np.clip((arr - lo) / span, 0.0, 1.0)
    norm = np.where(finite, norm, 0.0)

    from matplotlib import colormaps

    cmap = colormaps["inferno"]
    rgba = cmap(norm)  # (H, W, 4) in [0, 1]
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def trajectory_plot(poses: np.ndarray) -> Image.Image:
    """2D top-down plot of camera positions (x vs z) with orientation arrows.
    Expects (N, 4, 4) cam2world matrices in OpenCV convention."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poses = np.asarray(poses)
    xs = poses[:, 0, 3]
    zs = poses[:, 2, 3]
    # Forward direction in OpenCV is the camera's +Z axis (third column of R)
    fwd_x = poses[:, 0, 2]
    fwd_z = poses[:, 2, 2]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xs, zs, color="#888", linewidth=1)
    ax.scatter(xs, zs, c=np.arange(len(xs)), cmap="viridis", s=24)
    step = max(1, len(xs) // 20)
    ax.quiver(xs[::step], zs[::step], fwd_x[::step], fwd_z[::step], color="#c0392b", scale=20)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Camera trajectory (N={len(xs)} frames)")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
