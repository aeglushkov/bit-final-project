# OpenEQA Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bolt-on OpenEQA HM3D evaluation harness to eva-eval so the same Embodied VideoAgent pipeline can be validated against the paper's published InternVL2-8B numbers, disentangling whether the VSI-Bench score gap is from the open-source backbone, MASt3R preprocessing, or a pipeline bug.

**Architecture:** Three new Python modules (`preprocess/openeqa_hm3d.py`, `eval/openeqa.py`, `debug/render.py`), four new Python entry-point scripts (`05_download` / `06_preprocess` / `07_run` / `08_grade`), four inspection scripts (`inspect_preprocess` / `inspect_memory` / `inspect_agent_trace` / `inspect_grading`), and seven shell wrappers (`11_*` through `17_*`). One existing file gets a one-line addition (`agent/agent.py` exposes `return_intermediate_steps`). The agent, six tools, ReAct prompt, memory build, and cache schema are reused verbatim. Phase 1 builds inspectors first so they can run on existing VSI-Bench caches before any OpenEQA work.

**Tech Stack:** Python 3.10, pytest, numpy, PIL, matplotlib (for trajectory plot), langchain-openai (already in deps), OpenAI-compatible HTTP for judge calls. No new dependencies expected — the only candidate is matplotlib, already a transitive dep via existing packages.

**Spec:** `notes/brainstorming/2026-05-10-openeqa-validation-design.md`

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `experiments/eva-eval/eva_eval/agent/agent.py` | modify | expose `return_intermediate_steps` flag in `build_agent` |
| `experiments/eva-eval/eva_eval/debug/__init__.py` | create | package marker |
| `experiments/eva-eval/eva_eval/debug/render.py` | create | shared HTML/plot helpers (page builder, image embedding, depth colorize, trajectory plot, reprojection check) |
| `experiments/eva-eval/eva_eval/preprocess/openeqa_hm3d.py` | create | episode adapter: OpenEQA RGB-D-pose tuples → cache schema |
| `experiments/eva-eval/eva_eval/eval/openeqa.py` | create | question loader, formatter, run loop, aggregator |
| `experiments/eva-eval/eva_eval/eval/openeqa_grade.py` | create | LLM-as-judge grader (judge prompt, score parsing, C-score normalization) |
| `experiments/eva-eval/scripts/05_download_openeqa.py` | create | clone openeqa repo, copy questions JSON, verify episode bundle URL |
| `experiments/eva-eval/scripts/06_preprocess_openeqa.py` | create | stream-process episodes (download → adapt → memory → cleanup) |
| `experiments/eva-eval/scripts/07_run_openeqa.py` | create | run agent over sampled questions, write predictions JSONL |
| `experiments/eva-eval/scripts/08_grade_openeqa.py` | create | grade predictions JSONL, write graded JSONL |
| `experiments/eva-eval/scripts/inspect_preprocess.py` | create | render `_inspect/preprocess.html` for a cache dir |
| `experiments/eva-eval/scripts/inspect_memory.py` | create | render `_inspect/memory.html` for a cache dir |
| `experiments/eva-eval/scripts/inspect_agent_trace.py` | create | render markdown trace for a single question |
| `experiments/eva-eval/scripts/inspect_grading.py` | create | render `<output>.inspect.html` from graded JSONL |
| `experiments/eva-eval/scripts-remote/11_openeqa_setup.sh` | create | shell wrapper for `05_download_openeqa.py` |
| `experiments/eva-eval/scripts-remote/12_openeqa_sample.sh` | create | stratified-sample 50 questions |
| `experiments/eva-eval/scripts-remote/13_openeqa_preprocess.sh` | create | run preprocessing for sampled episodes |
| `experiments/eva-eval/scripts-remote/14_openeqa_inspect_first.sh` | create | run preprocess + memory inspectors on first episode |
| `experiments/eva-eval/scripts-remote/15_openeqa_run.sh` | create | run agent over the 50 questions |
| `experiments/eva-eval/scripts-remote/16_openeqa_grade.sh` | create | grade with default judge |
| `experiments/eva-eval/scripts-remote/17_openeqa_inspect_results.sh` | create | run grading inspector |
| `experiments/eva-eval/scripts-remote/_env.sh` | modify | add `OPENEQA_REPO_DIR`, `OPENEQA_CACHE_ROOT`, `OPENEQA_RESULTS_DIR` env vars |
| `experiments/eva-eval/tests/test_agent_intermediate_steps.py` | create | unit test for the `return_intermediate_steps` flag |
| `experiments/eva-eval/tests/test_debug_render.py` | create | unit tests for HTML helpers + depth colorize |
| `experiments/eva-eval/tests/test_openeqa_hm3d.py` | create | unit tests for pose conversion, intrinsics from FOV |
| `experiments/eva-eval/tests/test_openeqa_loader.py` | create | unit tests for question filtering + stratification |
| `experiments/eva-eval/tests/test_openeqa_grade.py` | create | unit tests for judge prompt, score parsing, C-score |

---

## Phase 1 — Inspectors (debug-first, can run on existing VSI-Bench cache)

### Task 1: Expose `return_intermediate_steps` flag in `build_agent`

**Files:**
- Modify: `experiments/eva-eval/eva_eval/agent/agent.py`
- Test: `experiments/eva-eval/tests/test_agent_intermediate_steps.py`

- [ ] **Step 1: Write the failing test**

```python
# experiments/eva-eval/tests/test_agent_intermediate_steps.py
"""Verify build_agent honors the return_intermediate_steps flag.
We monkeypatch the heavy dependencies (vlm, text_encoder, paper code) so the
test exercises only the LangChain wiring."""
from __future__ import annotations

import inspect
from unittest.mock import patch, MagicMock

import pytest


def test_build_agent_signature_has_intermediate_steps_flag():
    from eva_eval.agent.agent import build_agent

    sig = inspect.signature(build_agent)
    assert "return_intermediate_steps" in sig.parameters
    assert sig.parameters["return_intermediate_steps"].default is False


def test_build_agent_passes_flag_to_executor(tmp_path):
    from eva_eval.agent import agent as agent_mod

    fake_executor = MagicMock(name="AgentExecutor")
    fake_executor_class = MagicMock(name="AgentExecutorClass", return_value=fake_executor)
    fake_create_react = MagicMock(name="create_react_agent", return_value=MagicMock())
    fake_prompt_cls = MagicMock(name="PromptTemplate")
    fake_prompt_cls.from_template.return_value = MagicMock()

    fake_ctx = MagicMock()
    fake_make_tools = MagicMock(return_value=[])
    fake_build_planner = MagicMock(return_value=MagicMock())
    fake_build_vlm = MagicMock(return_value=MagicMock())

    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("chair\ntable\n")

    with patch.object(agent_mod, "AgentContext") as ctx_cls, \
         patch.object(agent_mod, "make_tools", fake_make_tools), \
         patch.object(agent_mod, "_build_planner_llm", fake_build_planner), \
         patch.object(agent_mod, "_build_vlm", fake_build_vlm):
        ctx_cls.load.return_value = fake_ctx
        with patch("langchain.agents.AgentExecutor", fake_executor_class), \
             patch("langchain.agents.create_react_agent", fake_create_react), \
             patch("langchain.prompts.PromptTemplate", fake_prompt_cls):
            executor, _ = agent_mod.build_agent(
                video_cache_dir=tmp_path,
                paper_code_dir=tmp_path,
                classes_file=classes_file,
                text_encoder=lambda s: None,
                return_intermediate_steps=True,
            )

    kwargs = fake_executor_class.call_args.kwargs
    assert kwargs["return_intermediate_steps"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd experiments/eva-eval
pytest tests/test_agent_intermediate_steps.py -v
```

Expected: `test_build_agent_signature_has_intermediate_steps_flag` FAILS with `AssertionError: assert "return_intermediate_steps" in sig.parameters`.

- [ ] **Step 3: Add the flag to `build_agent`**

Modify `experiments/eva-eval/eva_eval/agent/agent.py` — add the parameter to the signature and pass it to `AgentExecutor`:

```python
def build_agent(
    video_cache_dir: str | Path,
    paper_code_dir: str | Path,
    classes_file: str | Path,
    text_encoder,
    planner_name: str | None = None,
    max_iterations: int = 30,
    return_intermediate_steps: bool = False,
):
    """Assemble a ReAct agent over the six paper-spec tools, bound to one
    video's persistent memory.

    Args:
        text_encoder: callable text -> np.ndarray, used by the *_appearance,
                      *_environment and frame_localization tools. Inject the
                      paper's CLIP text encoder here at the call site so this
                      module stays free of torch/clip imports.
        return_intermediate_steps: when True, executor.invoke(...) returns the
                      ReAct trace under the "intermediate_steps" key. Used by
                      OpenEQA harness for the trace inspector.
    """
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.prompts import PromptTemplate

    vlm = _build_vlm()
    ctx = AgentContext.load(
        video_cache_dir=video_cache_dir,
        paper_code_dir=paper_code_dir,
        vlm=vlm,
        text_encoder=text_encoder,
    )
    tools = make_tools(ctx)
    llm = _build_planner_llm(planner_name)

    classes = _load_classes(Path(classes_file))
    template_text = PROMPT_PATH.read_text().replace("{categories_list}", str(classes))
    prompt = PromptTemplate.from_template(template_text)

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=return_intermediate_steps,
    )
    return executor, ctx
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd experiments/eva-eval
pytest tests/test_agent_intermediate_steps.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/eva-eval/eva_eval/agent/agent.py \
        experiments/eva-eval/tests/test_agent_intermediate_steps.py
git commit -m "agent: expose return_intermediate_steps flag in build_agent

Default False to preserve VSI-Bench behavior. OpenEQA trace inspector
will pass True so executor.invoke returns the ReAct steps."
```

---

### Task 2: Create `eva_eval/debug/render.py` shared helpers

**Files:**
- Modify: `experiments/eva-eval/pyproject.toml`
- Create: `experiments/eva-eval/eva_eval/debug/__init__.py`
- Create: `experiments/eva-eval/eva_eval/debug/render.py`
- Test: `experiments/eva-eval/tests/test_debug_render.py`

- [ ] **Step 0: Add matplotlib to dev deps**

`debug/render.py` uses `matplotlib` for the trajectory plot and depth colorize. Add it to the dev extras in `experiments/eva-eval/pyproject.toml` (`debug/render.py` lazy-imports matplotlib so the runtime install is also fine):

```diff
 [project.optional-dependencies]
-dev = ["pytest"]
+dev = ["pytest", "matplotlib"]
 serve-vllm = ["vllm>=0.6.3"]
```

Then re-install dev extras to pick it up:

```bash
cd experiments/eva-eval
pip install -e ".[dev]"
```

- [ ] **Step 1: Write the failing test**

```python
# experiments/eva-eval/tests/test_debug_render.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_html_page_wraps_body_with_title():
    from eva_eval.debug.render import html_page

    out = html_page(title="Hello", body="<p>world</p>")
    assert "<title>Hello</title>" in out
    assert "<p>world</p>" in out
    assert "<!doctype html>" in out.lower()


def test_image_to_data_uri_returns_png_data_url():
    from eva_eval.debug.render import image_to_data_uri

    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    uri = image_to_data_uri(img)
    assert uri.startswith("data:image/png;base64,")


def test_colorize_depth_returns_rgb_image_with_correct_size():
    from eva_eval.debug.render import colorize_depth

    depth = np.linspace(0.5, 5.0, 8 * 8, dtype=np.float32).reshape(8, 8)
    img = colorize_depth(depth)
    assert isinstance(img, Image.Image)
    assert img.size == (8, 8)
    assert img.mode == "RGB"


def test_colorize_depth_handles_constant_field():
    from eva_eval.debug.render import colorize_depth

    depth = np.full((4, 4), 3.0, dtype=np.float32)
    img = colorize_depth(depth)
    assert img.size == (4, 4)


def test_trajectory_plot_returns_pil_image(tmp_path):
    from eva_eval.debug.render import trajectory_plot

    poses = np.tile(np.eye(4, dtype=np.float32), (5, 1, 1))
    poses[:, 0, 3] = np.linspace(0, 4, 5)  # x positions
    poses[:, 2, 3] = np.linspace(0, 1, 5)  # z positions
    img = trajectory_plot(poses)
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0 and img.size[1] > 0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd experiments/eva-eval
pytest tests/test_debug_render.py -v
```

Expected: all tests FAIL with `ModuleNotFoundError: No module named 'eva_eval.debug'`.

- [ ] **Step 3: Create the package marker and module**

```python
# experiments/eva-eval/eva_eval/debug/__init__.py
```

```python
# experiments/eva-eval/eva_eval/debug/render.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd experiments/eva-eval
pytest tests/test_debug_render.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/eva-eval/pyproject.toml \
        experiments/eva-eval/eva_eval/debug/__init__.py \
        experiments/eva-eval/eva_eval/debug/render.py \
        experiments/eva-eval/tests/test_debug_render.py
git commit -m "debug: add render helpers (html_page, image_to_data_uri, colorize_depth, trajectory_plot)

Shared by all four inspect_* scripts. Self-contained static HTML so
output can be scp'd off the server. Adds matplotlib to dev deps for
the trajectory plot and depth colorize."
```

---

### Task 3: `scripts/inspect_preprocess.py`

**Files:**
- Create: `experiments/eva-eval/scripts/inspect_preprocess.py`
- Create: `experiments/eva-eval/eva_eval/debug/preprocess.py`
- Test: `experiments/eva-eval/tests/test_debug_preprocess.py`

This task pulls the rendering logic into a library module (`debug/preprocess.py`) so it can be unit-tested, and the script becomes a thin CLI shell.

- [ ] **Step 1: Write the failing test**

```python
# experiments/eva-eval/tests/test_debug_preprocess.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _make_cache_dir(tmp_path: Path, n: int = 5) -> Path:
    """Build a minimal cache dir with N synthetic frames."""
    cache = tmp_path / "scene"
    (cache / "frames").mkdir(parents=True)
    (cache / "depth").mkdir(parents=True)
    for i in range(n):
        img = Image.new("RGB", (64, 48), color=(i * 50 % 255, 100, 200))
        img.save(cache / "frames" / f"{i:06d}.jpg")
        depth = np.full((48, 64), 1.5 + 0.1 * i, dtype=np.float32)
        np.save(cache / "depth" / f"{i:06d}.npy", depth)
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    poses[:, 0, 3] = np.linspace(0, 4, n)
    np.save(cache / "poses.npy", poses)
    intrinsics = {"fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0, "width": 64, "height": 48, "fov_h": 1.2}
    (cache / "intrinsics.json").write_text(json.dumps(intrinsics))
    meta = {"video": "synthetic", "fps": 1.0, "n_frames": n, "timestamps": list(range(n)), "source": "test"}
    (cache / "meta.json").write_text(json.dumps(meta))
    return cache


def test_render_preprocess_html_writes_file_and_includes_sections(tmp_path):
    from eva_eval.debug.preprocess import render_preprocess_html

    cache = _make_cache_dir(tmp_path)
    out = render_preprocess_html(cache)
    assert out == cache / "_inspect" / "preprocess.html"
    assert out.exists()
    text = out.read_text()
    assert "Preprocess inspection" in text
    assert "Frame strip" in text
    assert "Depth (colorized)" in text
    assert "Camera trajectory" in text
    assert "synthetic" in text  # source name from meta.json


def test_reprojection_self_check_with_identity_poses_returns_stable_pixel(tmp_path):
    from eva_eval.debug.preprocess import reprojection_self_check

    cache = _make_cache_dir(tmp_path)
    # Identity poses + constant depth means the depth-pixel reprojects to the
    # exact same pixel in every frame.
    pixel_uvs = reprojection_self_check(cache, source_pixel=(20, 20))
    assert len(pixel_uvs) == 5
    for u, v in pixel_uvs:
        assert abs(u - 20.0) < 0.5
        assert abs(v - 20.0) < 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd experiments/eva-eval
pytest tests/test_debug_preprocess.py -v
```

Expected: tests FAIL with `ModuleNotFoundError: No module named 'eva_eval.debug.preprocess'`.

- [ ] **Step 3: Implement the library module**

```python
# experiments/eva-eval/eva_eval/debug/preprocess.py
"""Renderers for the preprocess inspection HTML."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from eva_eval.debug.render import (
    colorize_depth,
    image_to_data_uri,
    trajectory_plot,
    write_html,
)


def render_preprocess_html(cache_dir: str | Path) -> Path:
    """Generate `_inspect/preprocess.html` for a cache dir.
    Returns the written file's path."""
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    intrinsics = json.loads((cache_dir / "intrinsics.json").read_text())
    poses = np.load(cache_dir / "poses.npy")
    n = int(meta["n_frames"])

    pick = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})

    body_parts: list[str] = []
    body_parts.append(f"<h1>Preprocess inspection — <code>{cache_dir.name}</code></h1>")
    body_parts.append("<h2>Header</h2>")
    body_parts.append(_header_table(meta=meta, intrinsics=intrinsics, poses=poses))

    body_parts.append("<h2>Frame strip</h2>")
    body_parts.append(_frame_strip(cache_dir, pick))

    body_parts.append("<h2>Depth (colorized)</h2>")
    body_parts.append(_depth_strip(cache_dir, pick))

    body_parts.append("<h2>Camera trajectory</h2>")
    body_parts.append(_trajectory(poses))

    body_parts.append("<h2>Reprojection self-check</h2>")
    body_parts.append(_reprojection_table(cache_dir, pick))

    return write_html(
        cache_dir / "_inspect" / "preprocess.html",
        title=f"preprocess: {cache_dir.name}",
        body="\n".join(body_parts),
    )


def reprojection_self_check(cache_dir: str | Path, source_pixel: tuple[int, int] = None) -> list[tuple[float, float]]:
    """Pick a depth pixel from frame 0, lift to a 3D world point, then project
    that world point through every other frame's pose+K. Returns the (u, v)
    pixel in each frame.

    For correct OpenCV cam2world poses, a static world point should reproject
    to roughly the same pixel across nearby frames. A diverging dot indicates
    a pose-convention bug (e.g., Habitat→OpenCV transform missing or wrong)."""
    cache_dir = Path(cache_dir)
    intrinsics = json.loads((cache_dir / "intrinsics.json").read_text())
    poses = np.load(cache_dir / "poses.npy")
    n = poses.shape[0]
    if source_pixel is None:
        W = int(intrinsics["width"])
        H = int(intrinsics["height"])
        source_pixel = (W // 2, H // 2)
    u0, v0 = source_pixel
    depth0 = np.load(cache_dir / "depth" / f"{0:06d}.npy")
    z0 = float(depth0[v0, u0])
    if not np.isfinite(z0) or z0 <= 0:
        return [(float("nan"), float("nan"))] * n

    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    p_cam0 = np.array([(u0 - cx) * z0 / fx, (v0 - cy) * z0 / fy, z0], dtype=np.float64)
    R0 = poses[0, :3, :3]
    t0 = poses[0, :3, 3]
    p_world = R0 @ p_cam0 + t0  # cam → world

    out: list[tuple[float, float]] = []
    for i in range(n):
        Ri = poses[i, :3, :3]
        ti = poses[i, :3, 3]
        p_cam_i = (p_world - ti) @ Ri  # equivalent to R_i^T @ (p_world - t_i)
        z = p_cam_i[2]
        if z <= 1e-3:
            out.append((float("nan"), float("nan")))
            continue
        u = fx * p_cam_i[0] / z + cx
        v = fy * p_cam_i[1] / z + cy
        out.append((float(u), float(v)))
    return out


def _header_table(*, meta: dict, intrinsics: dict, poses: np.ndarray) -> str:
    rows = [
        ("source", meta.get("source", "?")),
        ("video", meta.get("video", "?")),
        ("n_frames", meta["n_frames"]),
        ("fps", meta.get("fps", "?")),
        ("image size", f"{intrinsics['width']} x {intrinsics['height']}"),
        ("fx, fy", f"{intrinsics['fx']:.2f}, {intrinsics['fy']:.2f}"),
        ("cx, cy", f"{intrinsics['cx']:.2f}, {intrinsics['cy']:.2f}"),
        ("fov_h (rad)", f"{intrinsics.get('fov_h', float('nan')):.4f}"),
        ("trajectory length (m)", f"{_traj_length(poses):.3f}"),
    ]
    body = "".join(f"<tr><th>{k}</th><td><code>{v}</code></td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _frame_strip(cache_dir: Path, pick: list[int]) -> str:
    parts = []
    for i in pick:
        path = cache_dir / "frames" / f"{i:06d}.jpg"
        if not path.exists():
            parts.append(f"<div>frame {i} missing</div>")
            continue
        img = Image.open(path).convert("RGB")
        parts.append(
            f'<div><img class="thumb" src="{image_to_data_uri(img)}">'
            f"<div>frame {i}</div></div>"
        )
    return f'<div class="row">{"".join(parts)}</div>'


def _depth_strip(cache_dir: Path, pick: list[int]) -> str:
    parts = []
    for i in pick:
        path = cache_dir / "depth" / f"{i:06d}.npy"
        if not path.exists():
            parts.append(f"<div>depth {i} missing</div>")
            continue
        arr = np.load(path)
        finite = np.isfinite(arr)
        d_min = float(arr[finite].min()) if finite.any() else float("nan")
        d_max = float(arr[finite].max()) if finite.any() else float("nan")
        d_mean = float(arr[finite].mean()) if finite.any() else float("nan")
        img = colorize_depth(arr)
        parts.append(
            f'<div><img class="thumb" src="{image_to_data_uri(img)}">'
            f"<div>frame {i} — min/mean/max: {d_min:.2f} / {d_mean:.2f} / {d_max:.2f} m</div></div>"
        )
    return f'<div class="row">{"".join(parts)}</div>'


def _trajectory(poses: np.ndarray) -> str:
    img = trajectory_plot(poses)
    return f'<img src="{image_to_data_uri(img)}">'


def _reprojection_table(cache_dir: Path, pick: list[int]) -> str:
    pixels = reprojection_self_check(cache_dir)
    if not pixels:
        return "<p>No frames to project.</p>"
    rows = [f"<tr><th>frame</th><th>u (px)</th><th>v (px)</th></tr>"]
    for i in pick:
        if i >= len(pixels):
            continue
        u, v = pixels[i]
        rows.append(
            f"<tr><td>{i}</td>"
            f"<td>{u:.1f}</td><td>{v:.1f}</td></tr>"
        )
    note = (
        "<p>If poses are correct in OpenCV cam2world convention, (u, v) should "
        "stay close to the source pixel (image center) across frames. "
        "Wandering values across <em>nearby</em> frames indicate a pose-convention bug.</p>"
    )
    return note + f"<table>{''.join(rows)}</table>"


def _traj_length(poses: np.ndarray) -> float:
    if poses.shape[0] < 2:
        return 0.0
    pts = poses[:, :3, 3]
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    return float(diffs.sum())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd experiments/eva-eval
pytest tests/test_debug_preprocess.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Implement the CLI script**

```python
# experiments/eva-eval/scripts/inspect_preprocess.py
"""Render preprocess.html for a cache directory.

Usage:
    python scripts/inspect_preprocess.py <cache_dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eva_eval.debug.preprocess import render_preprocess_html


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache_dir", type=Path, help="Cache directory containing frames/, depth/, poses.npy, intrinsics.json, meta.json")
    args = ap.parse_args()
    out = render_preprocess_html(args.cache_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-test the CLI on the synthetic cache**

```bash
cd experiments/eva-eval
python -c "
from pathlib import Path
import json, numpy as np
from PIL import Image
import tempfile, os
tmp = Path(tempfile.mkdtemp())
cache = tmp / 'scene'
(cache/'frames').mkdir(parents=True); (cache/'depth').mkdir(parents=True)
for i in range(5):
    Image.new('RGB',(64,48),color=(i*50%255,100,200)).save(cache/'frames'/f'{i:06d}.jpg')
    np.save(cache/'depth'/f'{i:06d}.npy', np.full((48,64), 1.5+0.1*i, dtype=np.float32))
np.save(cache/'poses.npy', np.tile(np.eye(4,dtype=np.float32),(5,1,1)))
(cache/'intrinsics.json').write_text(json.dumps({'fx':50.,'fy':50.,'cx':32.,'cy':24.,'width':64,'height':48,'fov_h':1.2}))
(cache/'meta.json').write_text(json.dumps({'video':'syn','fps':1.,'n_frames':5,'timestamps':list(range(5)),'source':'test'}))
print('cache:', cache)
" && python scripts/inspect_preprocess.py $(ls -td /tmp/tmp* | head -1)/scene
```

Expected: prints `wrote .../scene/_inspect/preprocess.html`. Open it in a browser to eyeball — should show 5 frame thumbnails, 5 depth panels, a trajectory plot, a reprojection table.

- [ ] **Step 7: Commit**

```bash
git add experiments/eva-eval/eva_eval/debug/preprocess.py \
        experiments/eva-eval/scripts/inspect_preprocess.py \
        experiments/eva-eval/tests/test_debug_preprocess.py
git commit -m "inspect: add inspect_preprocess.py for cache-dir sanity check

Renders frame thumbnails, colorized depth, camera trajectory, and a
reprojection self-check that catches pose-convention bugs early."
```

---

### Task 4: `scripts/inspect_memory.py`

**Files:**
- Create: `experiments/eva-eval/scripts/inspect_memory.py`
- Create: `experiments/eva-eval/eva_eval/debug/memory.py`
- Test: `experiments/eva-eval/tests/test_debug_memory.py`

- [ ] **Step 1: Write the failing test**

```python
# experiments/eva-eval/tests/test_debug_memory.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def _fake_object(*, identifier: int, category: str, n_frames: int, volume: float, state: str = "normal"):
    obj = MagicMock(spec=["identifier", "category", "volume", "state", "min_xyz", "max_xyz", "object_clip_feature", "context_clip_feature"])
    obj.identifier = identifier
    obj.category = category
    obj.volume = volume
    obj.state = state
    obj.min_xyz = np.array([0, 0, 0], dtype=np.float64)
    obj.max_xyz = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    obj.object_clip_feature = np.zeros(8, dtype=np.float32)
    obj.context_clip_feature = np.zeros(8, dtype=np.float32)
    obj._n_frames = n_frames
    return obj


def test_summarize_memory_counts_objects_by_category():
    from eva_eval.debug.memory import summarize_memory

    objs = [
        _fake_object(identifier=1, category="chair", n_frames=5, volume=0.5),
        _fake_object(identifier=2, category="chair", n_frames=2, volume=0.4),
        _fake_object(identifier=3, category="table", n_frames=8, volume=2.5),
    ]
    objects_frames = {1: list(range(5)), 2: [0, 1], 3: list(range(8))}
    summary = summarize_memory(objs, objects_frames=objects_frames)
    assert summary["n_objects"] == 3
    assert summary["n_categories"] == 2
    assert summary["category_counts"]["chair"] == 2
    assert summary["category_counts"]["table"] == 1
    assert summary["pct_single_frame"] == pytest.approx(0.0)


def test_summarize_memory_flags_warnings():
    from eva_eval.debug.memory import summarize_memory

    # All single-frame, all huge volumes → both warnings
    objs = [
        _fake_object(identifier=i, category=f"cat{i}", n_frames=1, volume=200.0)
        for i in range(3)
    ]
    objects_frames = {i: [0] for i in range(3)}
    summary = summarize_memory(objs, objects_frames=objects_frames)
    assert any("re-ID" in w for w in summary["warnings"])
    assert any("volume" in w for w in summary["warnings"])


def test_summarize_memory_flags_zero_objects():
    from eva_eval.debug.memory import summarize_memory

    summary = summarize_memory([], objects_frames={})
    assert summary["n_objects"] == 0
    assert any("0 objects" in w for w in summary["warnings"])
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd experiments/eva-eval
pytest tests/test_debug_memory.py -v
```

Expected: tests FAIL with `ModuleNotFoundError: No module named 'eva_eval.debug.memory'`.

- [ ] **Step 3: Implement the library module**

```python
# experiments/eva-eval/eva_eval/debug/memory.py
"""Renderers for the memory inspection HTML.

Note: the heavy 3D-bbox rendering uses AgentContext.render_object_bbox,
which depends on the paper's object3d module. Tests stub the objects.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from eva_eval.debug.render import image_to_data_uri, write_html


def summarize_memory(
    objects: Iterable[Any],
    *,
    objects_frames: dict[int, list[int]],
) -> dict[str, Any]:
    """Compute summary stats and emit auto-warnings for sanity checks."""
    objs = list(objects)
    n = len(objs)
    counts = Counter(getattr(o, "category", "?") for o in objs)

    visibility = [len(objects_frames.get(int(o.identifier), [])) for o in objs]
    pct_single = (sum(1 for v in visibility if v <= 1) / n) if n else 0.0

    volumes = [float(getattr(o, "volume", 0.0)) for o in objs]

    warnings: list[str] = []
    if n == 0:
        warnings.append("0 objects detected — memory build is broken (check vocabulary, depth range, model weights).")
    if 0 < len(counts) < 5:
        warnings.append(f"Only {len(counts)} categories — vocabulary is likely too narrow for this scene.")
    if n > 0 and pct_single > 0.5:
        warnings.append(f">{pct_single*100:.0f}% of objects are visible in only one frame — re-ID is failing or scene is over-segmented.")
    if volumes and (all(v > 100 for v in volumes) or all(0 < v < 0.001 for v in volumes)):
        warnings.append("All object volumes are extreme (>100 m^3 or <0.001 m^3) — likely a depth-units or scale mismatch.")

    return {
        "n_objects": n,
        "n_categories": len(counts),
        "category_counts": dict(counts),
        "pct_single_frame": pct_single,
        "warnings": warnings,
    }


def render_memory_html(cache_dir: str | Path, paper_code_dir: str | Path, frame_stride: int = 10) -> Path:
    """Generate `_inspect/memory.html` for a cache dir.

    Loads the cache via AgentContext (without VLM/text encoder — those are not
    needed for inspection), computes summary, renders per-object thumbnails
    with their 3D bbox projected onto the best frame, and per-frame stamps
    every `frame_stride` frames.
    """
    from eva_eval.agent.context import AgentContext

    cache_dir = Path(cache_dir)
    ctx = AgentContext.load(
        video_cache_dir=cache_dir,
        paper_code_dir=paper_code_dir,
        vlm=None,
        text_encoder=None,
    )

    objects = list(ctx.object_index.values())
    summary = summarize_memory(objects, objects_frames=ctx.objects_frames)

    body_parts: list[str] = []
    body_parts.append(f"<h1>Memory inspection — <code>{cache_dir.name}</code></h1>")
    body_parts.append("<h2>Summary</h2>")
    body_parts.append(_summary_table(summary))
    body_parts.append("".join(_warn(w) for w in summary["warnings"]))

    body_parts.append("<h2>Object catalog</h2>")
    body_parts.append(_object_catalog(ctx, objects))

    body_parts.append(f"<h2>Frame stamps (every {frame_stride}th frame)</h2>")
    body_parts.append(_frame_stamps(ctx, frame_stride=frame_stride))

    return write_html(
        cache_dir / "_inspect" / "memory.html",
        title=f"memory: {cache_dir.name}",
        body="\n".join(body_parts),
    )


def _summary_table(summary: dict) -> str:
    counts = ", ".join(f"{k}={v}" for k, v in sorted(summary["category_counts"].items(), key=lambda kv: -kv[1]))
    rows = [
        ("n_objects", summary["n_objects"]),
        ("n_categories", summary["n_categories"]),
        ("category counts", counts or "(none)"),
        ("% single-frame objects", f"{summary['pct_single_frame']*100:.1f}%"),
    ]
    body = "".join(f"<tr><th>{k}</th><td><code>{v}</code></td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _warn(msg: str) -> str:
    return f'<div class="warn">⚠ {msg}</div>'


def _object_catalog(ctx, objects: list) -> str:
    by_visibility = sorted(objects, key=lambda o: -len(ctx.objects_frames.get(int(o.identifier), [])))
    rows = ["<tr><th>id</th><th>category</th><th>frames</th><th>volume</th><th>state</th><th>best-frame thumb</th></tr>"]
    for o in by_visibility:
        oid = int(o.identifier)
        frames = ctx.objects_frames.get(oid, [])
        thumb_html = "—"
        if frames:
            try:
                img = ctx.render_object_bbox(oid, frames[0])
                thumb_html = f'<img class="thumb" src="{image_to_data_uri(img)}">'
            except Exception as e:
                thumb_html = f"(render error: {type(e).__name__})"
        rows.append(
            f"<tr><td>{oid}</td>"
            f"<td>{getattr(o, 'category', '?')}</td>"
            f"<td>{len(frames)}</td>"
            f"<td>{float(getattr(o, 'volume', 0.0)):.3f}</td>"
            f"<td>{getattr(o, 'state', '?')}</td>"
            f"<td>{thumb_html}</td></tr>"
        )
    return f"<table>{''.join(rows)}</table>"


def _frame_stamps(ctx, *, frame_stride: int) -> str:
    from PIL import Image, ImageDraw

    parts = []
    n = len(ctx.frame_index)
    for fi_idx in range(0, n, frame_stride):
        fi = ctx.frame_index[fi_idx]
        if not fi.path.exists() or not fi.visible_object_ids:
            continue
        img = Image.open(fi.path).convert("RGB")
        # Composite each visible object's 3D bbox onto the frame.
        for oid in fi.visible_object_ids:
            try:
                img_with = ctx.render_object_bbox(int(oid), fi.frame_id)
                img = img_with  # render_object_bbox returns a fresh image with the bbox drawn
            except Exception:
                continue
        labels = ", ".join(f"{int(o)}:{getattr(ctx.object_index.get(int(o)), 'category', '?')}"
                           for o in fi.visible_object_ids[:8])
        parts.append(
            f'<div><img class="frame" src="{image_to_data_uri(img)}">'
            f"<div>frame {fi.frame_id} — {labels}</div></div>"
        )
    return f'<div class="row">{"".join(parts)}</div>'
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd experiments/eva-eval
pytest tests/test_debug_memory.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Implement the CLI script**

```python
# experiments/eva-eval/scripts/inspect_memory.py
"""Render memory.html for a cache directory.

Usage:
    python scripts/inspect_memory.py <cache_dir> --paper-code-dir <path>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eva_eval.debug.memory import render_memory_html


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--paper-code-dir", type=Path, required=True)
    ap.add_argument("--frame-stride", type=int, default=10)
    args = ap.parse_args()
    out = render_memory_html(args.cache_dir, args.paper_code_dir, frame_stride=args.frame_stride)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-test on an existing VSI-Bench cache (if available)**

```bash
cd experiments/eva-eval
# Pick the first cache dir that has memory.pkl.
SAMPLE=$(ls -d ../../cache/vsibench/*/ 2>/dev/null | head -1)
if [ -n "$SAMPLE" ] && [ -f "$SAMPLE/memory.pkl" ]; then
    python scripts/inspect_memory.py "$SAMPLE" --paper-code-dir ../../literature/EmbodiedVideoAgent/code
    echo "Open $SAMPLE/_inspect/memory.html in a browser to inspect."
else
    echo "No VSI-Bench cache available locally — smoke test will run on the server during execution"
fi
```

Expected (on the server): prints `wrote .../memory.html`. Open in browser — see object catalog table and per-frame stamps.

- [ ] **Step 7: Commit**

```bash
git add experiments/eva-eval/eva_eval/debug/memory.py \
        experiments/eva-eval/scripts/inspect_memory.py \
        experiments/eva-eval/tests/test_debug_memory.py
git commit -m "inspect: add inspect_memory.py for object-memory sanity check

Renders object catalog table with 3D bbox thumbnails and per-frame stamps.
Auto-flags 0-objects, narrow vocabulary, failed re-ID, depth-unit
mismatch."
```

---

### Phase 1 stage gate

**Before continuing to Phase 2**, run the inspectors against an existing VSI-Bench cache on the server. If memory inspection reveals a clear bug (0 objects, all-single-frame visibility, depth-unit mismatch, etc.), localize and fix it — that may resolve the 25-score puzzle without OpenEQA.

If the inspectors look healthy and the bug isn't obvious, proceed to Phase 2.

---

## Phase 2 — OpenEQA preprocessing

### Task 5: `eva_eval/preprocess/openeqa_hm3d.py` episode adapter

**Files:**
- Create: `experiments/eva-eval/eva_eval/preprocess/openeqa_hm3d.py`
- Test: `experiments/eva-eval/tests/test_openeqa_hm3d.py`

OpenEQA's pre-rendered HM3D episodes use Habitat conventions. The exact on-disk layout is verified in **Step 0** of this task — until then, the loader code is parameterized over a small format spec.

- [ ] **Step 0: Verify the OpenEQA episode format**

```bash
# On the server:
git clone https://github.com/facebookresearch/open-eqa /tmp/open-eqa
cat /tmp/open-eqa/data/README.md
ls /tmp/open-eqa/data/
# Identify: where are HM3D episodes downloaded to? What files per frame?
```

Expected: README documents a download script (or direct URL) and the per-frame file layout. Pin the answers below before writing the test:

- Episode root layout: `<episode_id>/{rgb/<NNNNN>.png, depth/<NNNNN>.png_or_npy, pose/<NNNNN>.txt_or_poses.json}` (exact extensions to confirm)
- Depth units: meters (HM3D Habitat-rendered native) — confirm by checking a sample value
- Pose convention: Habitat (+Y up, -Z forward, opengl-style) or already OpenCV — confirm by reading the OpenEQA README

If the bundle is not publicly downloadable, **stop here**, document in the design's Risks table, and switch to the Habitat-sim fallback path (which is out of scope of this plan and would require its own design doc).

- [ ] **Step 1: Write the failing test**

```python
# experiments/eva-eval/tests/test_openeqa_hm3d.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_intrinsics_from_fov_round_trip():
    from eva_eval.preprocess.openeqa_hm3d import intrinsics_from_fov

    K = intrinsics_from_fov(width=640, height=480, fov_h_rad=np.pi / 2)
    assert K["width"] == 640 and K["height"] == 480
    assert K["cx"] == pytest.approx(320.0)
    assert K["cy"] == pytest.approx(240.0)
    assert K["fx"] == pytest.approx(320.0)  # tan(π/4) = 1, so fx = W/(2*1) = 320
    assert K["fy"] == pytest.approx(K["fx"])
    assert K["fov_h"] == pytest.approx(np.pi / 2)


def test_habitat_to_opencv_pose_is_idempotent_on_identity():
    from eva_eval.preprocess.openeqa_hm3d import habitat_to_opencv_pose

    out = habitat_to_opencv_pose(np.eye(4, dtype=np.float64))
    # Identity rotation in Habitat: forward=-Z, up=+Y, right=+X
    # Identity rotation in OpenCV: forward=+Z, up=-Y, right=+X
    # Conversion: flip Y and Z. The translation is unchanged.
    expected = np.diag([1.0, -1.0, -1.0, 1.0])
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_habitat_to_opencv_preserves_translation():
    from eva_eval.preprocess.openeqa_hm3d import habitat_to_opencv_pose

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    out = habitat_to_opencv_pose(pose)
    np.testing.assert_allclose(out[:3, 3], [1.0, 2.0, 3.0], atol=1e-9)


def test_habitat_to_opencv_double_application_is_identity():
    from eva_eval.preprocess.openeqa_hm3d import habitat_to_opencv_pose

    rng = np.random.default_rng(0)
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = rng.standard_normal(3)
    twice = habitat_to_opencv_pose(habitat_to_opencv_pose(pose))
    np.testing.assert_allclose(twice, pose, atol=1e-9)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd experiments/eva-eval
pytest tests/test_openeqa_hm3d.py -v
```

Expected: tests FAIL with `ModuleNotFoundError: No module named 'eva_eval.preprocess.openeqa_hm3d'`.

- [ ] **Step 3: Implement the helpers**

```python
# experiments/eva-eval/eva_eval/preprocess/openeqa_hm3d.py
"""OpenEQA HM3D episode → cache-schema adapter.

OpenEQA's pre-rendered HM3D episodes are downloaded via the script in
facebookresearch/open-eqa. Per-frame layout (verified in Step 0 of Task 5):
  <episode_id>/
    rgb/<NNNNN>.png        (or .jpg)
    depth/<NNNNN>.npy      (float32 metric meters)
    pose/<NNNNN>.txt       (4x4 cam2world in Habitat convention; one matrix per file)
    intrinsic.json         (fov_h or fx/fy/cx/cy at episode level)

If the download format differs, update _read_episode below; the rest of the
module stays the same.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# Diagonal matrix that flips Y and Z. Habitat is (+X right, +Y up, -Z forward,
# OpenGL-style). OpenCV is (+X right, +Y down, +Z forward). The conversion is
# self-inverse: applying it twice returns the original pose.
_HABITAT_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def intrinsics_from_fov(*, width: int, height: int, fov_h_rad: float) -> dict[str, float]:
    """fx = fy = W / (2 tan(fov_h / 2)); principal point at image center."""
    fx = float(width) / (2.0 * float(np.tan(fov_h_rad / 2.0)))
    return {
        "fx": fx,
        "fy": fx,  # square pixels assumed for HM3D Habitat renders
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
        "width": int(width),
        "height": int(height),
        "fov_h": float(fov_h_rad),
    }


def habitat_to_opencv_pose(pose_4x4: np.ndarray) -> np.ndarray:
    """Convert a 4x4 cam2world from Habitat (OpenGL) to OpenCV convention.
    Self-inverse: applying twice returns the original."""
    pose = np.asarray(pose_4x4, dtype=np.float64)
    return _HABITAT_TO_OPENCV @ pose @ _HABITAT_TO_OPENCV


def adapt_episode(
    episode_raw_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Convert one OpenEQA HM3D episode directory into the eva-eval cache schema.

    Reads:    rgb/, depth/, pose/, intrinsic.json from `episode_raw_dir`
    Writes:   frames/, depth/, poses.npy, intrinsics.json, meta.json in `out_dir`
    Returns:  the meta dict
    """
    raw = Path(episode_raw_dir)
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    rgb_paths, depth_paths, pose_paths = _read_episode(raw)
    n = len(rgb_paths)
    if n == 0:
        raise RuntimeError(f"No frames found in {raw}")
    if not (n == len(depth_paths) == len(pose_paths)):
        raise RuntimeError(f"Frame count mismatch in {raw}: rgb={n} depth={len(depth_paths)} pose={len(pose_paths)}")

    intrinsics_raw = json.loads((raw / "intrinsic.json").read_text())
    width, height = _image_size(rgb_paths[0])
    fov_h = _read_fov_h(intrinsics_raw, width=width, height=height)
    intrinsics = intrinsics_from_fov(width=width, height=height, fov_h_rad=fov_h)
    (out / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))

    poses = np.stack([
        habitat_to_opencv_pose(_read_pose(p)) for p in pose_paths
    ]).astype(np.float32)
    np.save(out / "poses.npy", poses)

    for i, (rgb_p, depth_p) in enumerate(zip(rgb_paths, depth_paths)):
        img = Image.open(rgb_p).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
        img.save(out / "frames" / f"{i:06d}.jpg", quality=85)
        depth = _load_depth(depth_p)
        np.save(out / "depth" / f"{i:06d}.npy", depth.astype(np.float32))

    timestamps = list(range(n))  # placeholder — paper's process_a_frame uses these only as keys
    meta = {
        "video": raw.name,
        "fps": 1.0,
        "n_frames": n,
        "timestamps": timestamps,
        "source": "openeqa_hm3d",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _read_episode(raw: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Return sorted lists of (rgb, depth, pose) per-frame paths."""
    rgb_dir = raw / "rgb"
    depth_dir = raw / "depth"
    pose_dir = raw / "pose"
    rgb = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    depth = sorted(p for p in depth_dir.iterdir() if p.suffix.lower() in (".npy", ".png"))
    pose = sorted(p for p in pose_dir.iterdir() if p.suffix.lower() in (".txt", ".json", ".npy"))
    return rgb, depth, pose


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size  # (width, height)


def _read_fov_h(intrinsics_raw: dict, *, width: int, height: int) -> float:
    """Extract horizontal FOV in radians from OpenEQA's intrinsic.json.
    Accepts either 'fov_h' (radians), 'hfov' (degrees), or fx (px)."""
    if "fov_h" in intrinsics_raw:
        return float(intrinsics_raw["fov_h"])
    if "hfov" in intrinsics_raw:
        return float(np.deg2rad(float(intrinsics_raw["hfov"])))
    if "fx" in intrinsics_raw:
        return 2.0 * float(np.arctan(width / (2.0 * float(intrinsics_raw["fx"]))))
    raise KeyError(f"intrinsic.json missing fov_h, hfov, or fx: keys={sorted(intrinsics_raw)}")


def _read_pose(path: Path) -> np.ndarray:
    """Load a 4x4 cam2world matrix from .txt (whitespace), .npy, or .json."""
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".json":
        return np.array(json.loads(path.read_text()), dtype=np.float64)
    return np.loadtxt(path)


def _load_depth(path: Path) -> np.ndarray:
    """Load depth as float32 metric meters. Accepts .npy or 16-bit png (mm)."""
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    if path.suffix == ".png":
        # 16-bit PNG: assume millimeters (HM3D's pre-rendered convention if PNG)
        from PIL import Image as _Image
        arr = np.array(_Image.open(path))
        return (arr.astype(np.float32) / 1000.0)
    raise ValueError(f"Unsupported depth format: {path.suffix}")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd experiments/eva-eval
pytest tests/test_openeqa_hm3d.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/eva-eval/eva_eval/preprocess/openeqa_hm3d.py \
        experiments/eva-eval/tests/test_openeqa_hm3d.py
git commit -m "preprocess: add OpenEQA HM3D episode adapter

Bridges OpenEQA's per-frame RGB+depth+pose tuples to the eva-eval cache
schema consumed by 02_build_memory.py. Includes the Habitat→OpenCV pose
conversion (verified self-inverse on random poses)."
```

---

### Task 6: `scripts/05_download_openeqa.py`

**Files:**
- Create: `experiments/eva-eval/scripts/05_download_openeqa.py`

This script is a thin shell over `git clone` and a directory walk. The most important part is the **fail-fast verification** that the pre-rendered HM3D bundle URL is reachable.

- [ ] **Step 1: Implement the script**

```python
# experiments/eva-eval/scripts/05_download_openeqa.py
"""Phase: clone openeqa repo, copy questions JSON, verify HM3D bundle URL.

Usage:
    python scripts/05_download_openeqa.py \
        --openeqa-repo-dir <path>            # where to clone the repo
        --out-questions-json <path>          # where to copy the questions JSON
        [--bundle-url <URL>]                 # override the default bundle URL
        [--no-bundle-check]                  # skip the URL HEAD check

This script does NOT download the episode bundle itself — that's done
on demand per episode by 06_preprocess_openeqa.py.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

OPENEQA_REPO_URL = "https://github.com/facebookresearch/open-eqa.git"
DEFAULT_QUESTIONS_RELPATH = "data/open-eqa-v0.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--openeqa-repo-dir", type=Path, required=True)
    ap.add_argument("--out-questions-json", type=Path, required=True)
    ap.add_argument("--bundle-url", default=None,
                    help="If set, HEAD-check this URL to confirm the HM3D "
                         "pre-rendered bundle is reachable.")
    ap.add_argument("--no-bundle-check", action="store_true")
    args = ap.parse_args()

    repo = args.openeqa_repo_dir.resolve()
    if repo.exists() and (repo / ".git").exists():
        print(f"[skip clone] {repo} already exists; pulling latest")
        subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
    else:
        repo.parent.mkdir(parents=True, exist_ok=True)
        print(f"[clone] {OPENEQA_REPO_URL} -> {repo}")
        subprocess.run(["git", "clone", "--depth", "1", OPENEQA_REPO_URL, str(repo)], check=True)

    src_q = repo / DEFAULT_QUESTIONS_RELPATH
    if not src_q.exists():
        print(f"ERROR: expected questions JSON not found at {src_q}", file=sys.stderr)
        print(f"       Inspect the repo and update DEFAULT_QUESTIONS_RELPATH", file=sys.stderr)
        sys.exit(2)
    args.out_questions_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_q, args.out_questions_json)
    print(f"[copy] {src_q} -> {args.out_questions_json}")

    if args.bundle_url and not args.no_bundle_check:
        print(f"[head] {args.bundle_url}")
        try:
            req = urllib.request.Request(args.bundle_url, method="HEAD")
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}")
                print(f"[ok] bundle URL reachable (HTTP {resp.status})")
        except Exception as e:
            print(f"ERROR: bundle URL not reachable: {e}", file=sys.stderr)
            print(
                "Cannot proceed with the planned design (Habitat-GT depth+pose).\n"
                "Fallback options: (1) use a different mirror, (2) install Habitat-sim\n"
                "and HM3D scans and render trajectories yourself (out of scope here).",
                file=sys.stderr,
            )
            sys.exit(3)

    print("done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the script (skip the bundle check)**

```bash
cd experiments/eva-eval
python scripts/05_download_openeqa.py \
    --openeqa-repo-dir /tmp/openeqa-test \
    --out-questions-json /tmp/openeqa-test-questions.json \
    --no-bundle-check
ls -la /tmp/openeqa-test-questions.json
head -c 200 /tmp/openeqa-test-questions.json
```

Expected: clones the repo, copies the questions JSON, prints `done.`. Cleanup: `rm -rf /tmp/openeqa-test /tmp/openeqa-test-questions.json`.

- [ ] **Step 3: Commit**

```bash
git add experiments/eva-eval/scripts/05_download_openeqa.py
git commit -m "scripts: add 05_download_openeqa.py

Clones facebookresearch/open-eqa and copies the questions JSON.
Optional HEAD check on the HM3D bundle URL fails fast if the bundle
is no longer publicly reachable."
```

---

### Task 7: `scripts/06_preprocess_openeqa.py`

**Files:**
- Create: `experiments/eva-eval/scripts/06_preprocess_openeqa.py`

This script orchestrates the per-episode pipeline: download → adapt → build memory → cleanup. It assumes a per-episode download URL pattern that the `--bundle-url-template` flag parameterizes.

- [ ] **Step 1: Implement the script**

```python
# experiments/eva-eval/scripts/06_preprocess_openeqa.py
"""Per-episode: download → adapt → build memory → cleanup.

Reads a sampled-questions JSON (one row per question, with `episode_history`),
extracts the unique episode IDs, and processes each one in turn.

Usage:
    python scripts/06_preprocess_openeqa.py \
        --sampled-json <cache_root>/openeqa_hm3d/sampled_50.json \
        --cache-root <cache_root> \
        --paper-code-dir <path/to/literature/EmbodiedVideoAgent/code> \
        --classes-file <path/to/detection_classes.txt> \
        --bundle-url-template "https://example.com/hm3d/{episode_id}.tar.gz"

Cleanup behavior:
  - After preprocessing each episode, episodes_raw/<episode_id>/ is deleted.
  - After memory build, <episode_id>/depth/ is deleted.
  - Both can be disabled with --keep-raw / --keep-depth for debugging.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable

from eva_eval.preprocess.memory import build_memory_for_video
from eva_eval.preprocess.openeqa_hm3d import adapt_episode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampled-json", type=Path, required=True,
                    help="JSON file with rows containing 'episode_history' (e.g. 'hm3d-v0/<id>')")
    ap.add_argument("--cache-root", type=Path, required=True,
                    help="Root above the openeqa_hm3d/ subdir.")
    ap.add_argument("--paper-code-dir", type=Path, required=True)
    ap.add_argument("--classes-file", type=Path, required=True)
    ap.add_argument("--bundle-url-template", required=True,
                    help="URL template with '{episode_id}' placeholder for per-episode tar.gz")
    ap.add_argument("--keep-raw", action="store_true",
                    help="Keep episodes_raw/ after preprocessing (debug)")
    ap.add_argument("--keep-depth", action="store_true",
                    help="Keep depth/*.npy after memory build (debug)")
    ap.add_argument("--depth-min", type=float, default=0.05)
    ap.add_argument("--depth-max", type=float, default=10.0)
    args = ap.parse_args()

    rows = json.loads(args.sampled_json.read_text())
    episode_ids = sorted({_episode_id(r["episode_history"]) for r in rows})
    print(f"Processing {len(episode_ids)} unique episodes from {len(rows)} questions")

    classes = _load_classes(args.classes_file)
    print(f"Loaded {len(classes)} detection classes")

    cache_dir_root = args.cache_root / "openeqa_hm3d"
    cache_dir_root.mkdir(parents=True, exist_ok=True)
    raw_root = cache_dir_root / "episodes_raw"
    raw_root.mkdir(exist_ok=True)

    failures: list[tuple[str, str]] = []
    for ep_id in episode_ids:
        ep_cache = cache_dir_root / ep_id
        if (ep_cache / "memory.pkl").exists():
            print(f"[skip] {ep_id} (memory.pkl exists)")
            continue
        try:
            print(f"[run] {ep_id}")
            ep_raw = raw_root / ep_id
            if not ep_raw.exists():
                _download_and_extract(args.bundle_url_template.format(episode_id=ep_id), ep_raw)
            adapt_episode(ep_raw, ep_cache)
            build_memory_for_video(
                video_cache_dir=ep_cache,
                classes=classes,
                paper_code_dir=args.paper_code_dir,
                save_path=ep_cache / "memory.pkl",
                valid_depth_min=args.depth_min,
                valid_depth_max=args.depth_max,
            )
            if not args.keep_raw:
                shutil.rmtree(ep_raw, ignore_errors=True)
            if not args.keep_depth:
                shutil.rmtree(ep_cache / "depth", ignore_errors=True)
        except Exception as e:
            print(f"[fail] {ep_id}: {type(e).__name__}: {e}", file=sys.stderr)
            failures.append((ep_id, f"{type(e).__name__}: {e}"))

    if failures:
        log = cache_dir_root / "preprocess_failures.jsonl"
        with log.open("w") as f:
            for ep_id, err in failures:
                f.write(json.dumps({"episode_id": ep_id, "error": err}) + "\n")
        print(f"\n{len(failures)} failures logged to {log}", file=sys.stderr)
        sys.exit(1)


def _episode_id(episode_history: str) -> str:
    """'hm3d-v0/<episode_id>' -> '<episode_id>'."""
    if "/" not in episode_history:
        return episode_history
    return episode_history.split("/", 1)[1]


def _load_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def _download_and_extract(url: str, out_dir: Path):
    """Download tar.gz to a temp file, extract, move into out_dir."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"  download {url}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        with tarfile.open(tmp_path, "r:gz") as tf:
            tf.extractall(out_dir.parent)
        # If the tar extracts to a different name, move it.
        # Common case: tar contains <episode_id>/...
        # If `out_dir` doesn't exist after extract, find the new dir.
        if not out_dir.exists():
            # Find the most recently modified subdir of out_dir.parent
            candidates = [d for d in out_dir.parent.iterdir() if d.is_dir() and d.name not in {"."}]
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                candidates[0].rename(out_dir)
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add experiments/eva-eval/scripts/06_preprocess_openeqa.py
git commit -m "scripts: add 06_preprocess_openeqa.py stream processor

Per-episode: download tar.gz, adapt to cache schema, build ObjectMemory,
delete raw and depth. Resumable (skips episodes whose memory.pkl
already exists)."
```

---

### Task 8: Shell wrappers `11_*.sh` through `14_*.sh` and `_env.sh` updates

**Files:**
- Modify: `experiments/eva-eval/scripts-remote/_env.sh`
- Create: `experiments/eva-eval/scripts-remote/11_openeqa_setup.sh`
- Create: `experiments/eva-eval/scripts-remote/12_openeqa_sample.sh`
- Create: `experiments/eva-eval/scripts-remote/13_openeqa_preprocess.sh`
- Create: `experiments/eva-eval/scripts-remote/14_openeqa_inspect_first.sh`

- [ ] **Step 1: Add OpenEQA env vars to `_env.sh`**

Modify `experiments/eva-eval/scripts-remote/_env.sh` — append at the end:

```bash
# OpenEQA paths
OPENEQA_REPO_DIR="$ROOT/.third-party/openeqa"
OPENEQA_CACHE_ROOT="$ROOT/cache"     # episodes go under cache/openeqa_hm3d/
OPENEQA_QUESTIONS_JSON="$OPENEQA_CACHE_ROOT/openeqa_hm3d/questions.json"
OPENEQA_SAMPLED_JSON="$OPENEQA_CACHE_ROOT/openeqa_hm3d/sampled_50.json"
OPENEQA_BUNDLE_URL_TEMPLATE="${OPENEQA_BUNDLE_URL_TEMPLATE:-}"  # set this in your shell or a .env

mkdir -p "$OPENEQA_CACHE_ROOT/openeqa_hm3d"
```

- [ ] **Step 2: Implement `11_openeqa_setup.sh`**

```bash
#!/usr/bin/env bash
# Clone openeqa repo and copy the questions JSON. Optionally HEAD-check the bundle URL.
# Sets OPENEQA_BUNDLE_URL_TEMPLATE in your shell first if you want the URL check.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "============ 11_openeqa_setup ============"
date

EXTRA=()
if [ -n "${OPENEQA_BUNDLE_URL_TEMPLATE:-}" ]; then
    # Use the first episode's URL as a sentinel for bundle reachability.
    SENTINEL_URL="${OPENEQA_BUNDLE_URL_TEMPLATE//\{episode_id\}/sentinel}"
    EXTRA=(--bundle-url "$SENTINEL_URL")
fi

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/05_download_openeqa.py" \
    --openeqa-repo-dir "$OPENEQA_REPO_DIR" \
    --out-questions-json "$OPENEQA_QUESTIONS_JSON" \
    "${EXTRA[@]:-}"

echo "==> done. Next: bash scripts-remote/12_openeqa_sample.sh"
date
```

- [ ] **Step 3: Implement `12_openeqa_sample.sh`**

The sample script is small enough to inline as a Python heredoc — no separate Python file needed.

```bash
#!/usr/bin/env bash
# Stratified-sample 50 HM3D questions from the full openeqa-v0 questions JSON.
# Idempotent: writes sampled_50.json, overwrites if --force.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

LIMIT="${OPENEQA_LIMIT:-50}"
SEED="${OPENEQA_SEED:-42}"

echo "============ 12_openeqa_sample ============"
date

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" - <<PY
import json
from pathlib import Path

from eva_eval.eval.sampler import stratified_indices

questions_path = Path("$OPENEQA_QUESTIONS_JSON")
sampled_path = Path("$OPENEQA_SAMPLED_JSON")

rows = json.loads(questions_path.read_text())
hm3d = [r for r in rows if r.get("episode_history", "").startswith("hm3d-v0/")]
print(f"HM3D questions: {len(hm3d)} / {len(rows)} total")

categories = [r.get("category", "?") for r in hm3d]
idxs = stratified_indices(categories, total=$LIMIT, seed=$SEED)
sampled = [hm3d[i] for i in idxs]

from collections import Counter
print(f"Sampled {len(sampled)} questions, by category:")
for c, k in Counter(r.get("category", "?") for r in sampled).most_common():
    print(f"  {c:30s} {k}")

n_eps = len({r['episode_history'] for r in sampled})
print(f"Unique episodes: {n_eps}")

sampled_path.parent.mkdir(parents=True, exist_ok=True)
sampled_path.write_text(json.dumps(sampled, indent=2))
print(f"wrote {sampled_path}")
PY

echo "==> done. Next: bash scripts-remote/13_openeqa_preprocess.sh"
date
```

- [ ] **Step 4: Implement `13_openeqa_preprocess.sh`**

```bash
#!/usr/bin/env bash
# Preprocess (download + adapt + build memory + cleanup) for all sampled episodes.
# Requires OPENEQA_BUNDLE_URL_TEMPLATE to be set in your shell.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

if [ -z "${OPENEQA_BUNDLE_URL_TEMPLATE:-}" ]; then
    echo "ERROR: OPENEQA_BUNDLE_URL_TEMPLATE not set." >&2
    echo "       Determine the per-episode tar.gz URL from openeqa README and" >&2
    echo "       export OPENEQA_BUNDLE_URL_TEMPLATE='<url with {episode_id}>'." >&2
    exit 1
fi

echo "============ 13_openeqa_preprocess ============"
date

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/06_preprocess_openeqa.py" \
    --sampled-json "$OPENEQA_SAMPLED_JSON" \
    --cache-root "$OPENEQA_CACHE_ROOT" \
    --paper-code-dir "$PAPER_DIR" \
    --classes-file "$CLASSES_FILE" \
    --bundle-url-template "$OPENEQA_BUNDLE_URL_TEMPLATE"

echo "==> done. Next: bash scripts-remote/14_openeqa_inspect_first.sh"
date
```

- [ ] **Step 5: Implement `14_openeqa_inspect_first.sh`**

```bash
#!/usr/bin/env bash
# Run preprocess + memory inspectors on the first preprocessed episode.
# *** HUMAN GATE *** — open the produced HTML files in a browser and verify:
#   - bboxes land on objects in the per-frame stamps
#   - reprojection table shows stable (u, v)
#   - no warnings on the memory page
set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "============ 14_openeqa_inspect_first ============"
date

# Pick the first cache dir under openeqa_hm3d/ that has memory.pkl.
FIRST=""
for d in "$OPENEQA_CACHE_ROOT/openeqa_hm3d"/*/; do
    if [ -f "$d/memory.pkl" ]; then
        FIRST="$d"
        break
    fi
done
if [ -z "$FIRST" ]; then
    echo "ERROR: no preprocessed episode found." >&2
    exit 1
fi
echo "Inspecting: $FIRST"

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/inspect_preprocess.py" "$FIRST"

PYTHONPATH="$EVA_EVAL_DIR:$PAPER_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/inspect_memory.py" \
    "$FIRST" --paper-code-dir "$PAPER_DIR"

echo
echo "==> Open these files in a browser:"
echo "    $FIRST/_inspect/preprocess.html"
echo "    $FIRST/_inspect/memory.html"
echo
echo "==> If they look right, next: bash scripts-remote/15_openeqa_run.sh"
date
```

- [ ] **Step 6: Make the scripts executable and commit**

```bash
chmod +x experiments/eva-eval/scripts-remote/11_openeqa_setup.sh \
         experiments/eva-eval/scripts-remote/12_openeqa_sample.sh \
         experiments/eva-eval/scripts-remote/13_openeqa_preprocess.sh \
         experiments/eva-eval/scripts-remote/14_openeqa_inspect_first.sh

git add experiments/eva-eval/scripts-remote/_env.sh \
        experiments/eva-eval/scripts-remote/11_openeqa_setup.sh \
        experiments/eva-eval/scripts-remote/12_openeqa_sample.sh \
        experiments/eva-eval/scripts-remote/13_openeqa_preprocess.sh \
        experiments/eva-eval/scripts-remote/14_openeqa_inspect_first.sh
git commit -m "scripts-remote: add 11..14 OpenEQA preprocessing wrappers

Each wrapper sources _env.sh, prints next-step hint at the end. The
14_inspect_first wrapper is a human gate: stop and open the produced
HTML in a browser before running the agent."
```

---

### Phase 2 stage gate

After running `11..14`, **stop** and open the produced HTML files in a browser. Verify:

1. `preprocess.html` shows reasonable frame thumbnails, depth maps with sensible ranges (e.g., 0.5–10 m for indoor scenes), a sensible camera trajectory, and stable reprojection coordinates.
2. `memory.html` shows non-empty object catalog, multiple categories, frame stamps with bboxes that visibly land on objects, and no warning banners.

Only proceed to Phase 3 if both look correct. If either is broken, the bug is in preprocessing, not the agent.

---

## Phase 3 — OpenEQA evaluation

### Task 9: `eva_eval/eval/openeqa.py` loader, formatter, run loop

**Files:**
- Create: `experiments/eva-eval/eva_eval/eval/openeqa.py`
- Test: `experiments/eva-eval/tests/test_openeqa_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# experiments/eva-eval/tests/test_openeqa_loader.py
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_questions(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "qs.json"
    p.write_text(json.dumps(rows))
    return p


def test_load_filters_to_hm3d_when_dataset_is_hm3d(tmp_path):
    from eva_eval.eval.openeqa import load_openeqa_questions

    p = _write_questions(tmp_path, [
        {"question_id": "1", "episode_history": "hm3d-v0/scene_a", "question": "Q1", "answer": "A1", "category": "object_recognition"},
        {"question_id": "2", "episode_history": "scannet-v0/scene_b", "question": "Q2", "answer": "A2", "category": "spatial_reasoning"},
    ])
    out = load_openeqa_questions(p, dataset="hm3d", limit=None)
    assert [r["question_id"] for r in out] == ["1"]


def test_load_returns_all_when_dataset_is_all(tmp_path):
    from eva_eval.eval.openeqa import load_openeqa_questions

    p = _write_questions(tmp_path, [
        {"question_id": "1", "episode_history": "hm3d-v0/a", "question": "Q1", "answer": "A1", "category": "x"},
        {"question_id": "2", "episode_history": "scannet-v0/b", "question": "Q2", "answer": "A2", "category": "y"},
    ])
    out = load_openeqa_questions(p, dataset="all", limit=None)
    assert len(out) == 2


def test_load_stratified_sample_balances_categories(tmp_path):
    from collections import Counter
    from eva_eval.eval.openeqa import load_openeqa_questions

    rows = []
    for i in range(20):
        rows.append({"question_id": f"a{i}", "episode_history": "hm3d-v0/x",
                     "question": "?", "answer": "?", "category": "alpha"})
    for i in range(20):
        rows.append({"question_id": f"b{i}", "episode_history": "hm3d-v0/y",
                     "question": "?", "answer": "?", "category": "beta"})
    p = _write_questions(tmp_path, rows)

    out = load_openeqa_questions(p, dataset="hm3d", limit=10, stratified=True, seed=42)
    counts = Counter(r["category"] for r in out)
    assert counts["alpha"] == 5
    assert counts["beta"] == 5


def test_format_question_uses_pre_prompt():
    from eva_eval.eval.openeqa import format_question

    text = format_question({"question": "How many chairs are there?"})
    assert "How many chairs are there?" in text
    assert "indoor scene" in text.lower()


def test_episode_cache_dir_strips_dataset_prefix(tmp_path):
    from eva_eval.eval.openeqa import episode_cache_dir

    out = episode_cache_dir(tmp_path, "hm3d-v0/00000-foo")
    assert out == tmp_path / "openeqa_hm3d" / "00000-foo"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd experiments/eva-eval
pytest tests/test_openeqa_loader.py -v
```

Expected: tests FAIL with `ModuleNotFoundError: No module named 'eva_eval.eval.openeqa'`.

- [ ] **Step 3: Implement the module**

```python
# experiments/eva-eval/eva_eval/eval/openeqa.py
"""OpenEQA HM3D evaluation harness — parallels eval/vsibench.py.

Run loop and per-question agent invocation; grading is a separate step
(see eval/openeqa_grade.py and scripts/08_grade_openeqa.py).
"""
from __future__ import annotations

import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from eva_eval.eval.sampler import stratified_indices


OPENEQA_PRE_PROMPT = "These are frames from an indoor scene exploration video."


def load_openeqa_questions(
    questions_json: str | Path,
    *,
    dataset: str = "hm3d",
    limit: int | None = 50,
    stratified: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Load OpenEQA questions from `data/open-eqa-v0.json`, filter by source
    dataset, optionally stratified-sample by category."""
    rows = json.loads(Path(questions_json).read_text())
    if dataset == "hm3d":
        rows = [r for r in rows if r.get("episode_history", "").startswith("hm3d-v0/")]
    elif dataset == "scannet":
        rows = [r for r in rows if r.get("episode_history", "").startswith("scannet-v0/")]
    elif dataset == "all":
        pass
    else:
        raise ValueError(f"Unknown dataset {dataset!r}; expected hm3d|scannet|all")

    if limit is None or limit >= len(rows):
        return rows

    if stratified:
        cats = [r.get("category", "?") for r in rows]
        idxs = stratified_indices(cats, total=limit, seed=seed)
        return [rows[i] for i in idxs]

    import random
    rng = random.Random(seed)
    pool = list(range(len(rows)))
    rng.shuffle(pool)
    return [rows[i] for i in sorted(pool[:limit])]


def format_question(q: dict) -> str:
    """Open-ended QA prompt — no MCA/NA branching."""
    return f"{OPENEQA_PRE_PROMPT}\n{q['question']}"


def episode_cache_dir(cache_root: str | Path, episode_history: str) -> Path:
    """'hm3d-v0/<episode_id>' -> <cache_root>/openeqa_hm3d/<episode_id>"""
    cache_root = Path(cache_root)
    if "/" in episode_history:
        ep_id = episode_history.split("/", 1)[1]
    else:
        ep_id = episode_history
    return cache_root / "openeqa_hm3d" / ep_id


def parse_final_answer(response) -> str:
    if isinstance(response, dict):
        text = response.get("output", "")
    else:
        text = str(response)
    text = str(text).strip()
    for prefix in ("Final Answer:", "FINAL ANSWER:", "final answer:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    return text


def group_by_episode(rows: list[dict]) -> dict[str, list[int]]:
    by_ep: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_ep[r["episode_history"]].append(i)
    return dict(by_ep)


def run(
    *,
    sampled_json: Path,
    cache_root: Path,
    paper_code_dir: Path,
    classes_file: Path,
    output: Path,
    planner: str | None = None,
    max_iterations: int = 30,
    capture_trace: bool = True,
    resume: bool = False,
) -> dict:
    """Run the agent over a sampled set of OpenEQA questions. Writes one JSONL
    row per question to `output`. Returns a small summary dict."""
    from eva_eval.agent.agent import build_agent
    from eva_eval.agent.text_encoder import build_clip_text_encoder

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(sampled_json).read_text())
    by_ep = group_by_episode(rows)
    print(f"Evaluating {len(rows)} questions across {len(by_ep)} episodes")

    answered: set = set()
    if resume and output.exists():
        with output.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error"):
                    continue
                answered.add(r["id"])
        print(f"resume: skipping {len(answered)} previously answered questions")
    open_mode = "a" if resume else "w"

    text_encoder = build_clip_text_encoder(paper_code_dir)
    n_done = 0
    with output.open(open_mode) as out_f:
        for episode_history, qidxs in by_ep.items():
            ep_dir = episode_cache_dir(cache_root, episode_history)
            if not (ep_dir / "memory.pkl").exists():
                print(f"[skip] {episode_history}: no memory.pkl in {ep_dir}", file=sys.stderr)
                continue
            remaining = [qi for qi in qidxs if rows[qi]["question_id"] not in answered]
            if not remaining:
                continue
            try:
                executor, _ctx = build_agent(
                    video_cache_dir=ep_dir,
                    paper_code_dir=paper_code_dir,
                    classes_file=classes_file,
                    text_encoder=text_encoder,
                    planner_name=planner,
                    max_iterations=max_iterations,
                    return_intermediate_steps=capture_trace,
                )
            except Exception as e:
                print(f"[fail-build] {episode_history}: {e}", file=sys.stderr)
                continue

            for qi in remaining:
                doc = rows[qi]
                user_text = format_question(doc)
                try:
                    response = executor.invoke({"input": user_text})
                    pred = parse_final_answer(response)
                    steps = response.get("intermediate_steps", []) if isinstance(response, dict) and capture_trace else []
                    err = None
                except Exception as e:
                    pred = ""
                    steps = []
                    err = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"

                row_out = {
                    "id": doc["question_id"],
                    "episode_history": episode_history,
                    "category": doc.get("category", "?"),
                    "question": doc["question"],
                    "ground_truth": doc["answer"],
                    "prediction": pred,
                    "intermediate_steps": _serialize_steps(steps),
                    "error": err,
                }
                out_f.write(json.dumps(row_out, default=str) + "\n")
                out_f.flush()
                n_done += 1

    return {"n_done": n_done, "n_total": len(rows)}


def _serialize_steps(steps) -> list:
    """Convert LangChain AgentAction intermediate_steps tuples to JSON-friendly dicts."""
    out = []
    for step in steps:
        try:
            action, observation = step
            out.append({
                "tool": getattr(action, "tool", str(action)),
                "tool_input": getattr(action, "tool_input", None),
                "log": getattr(action, "log", None),
                "observation": str(observation),
            })
        except Exception:
            out.append({"raw": repr(step)})
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd experiments/eva-eval
pytest tests/test_openeqa_loader.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/eva-eval/eva_eval/eval/openeqa.py \
        experiments/eva-eval/tests/test_openeqa_loader.py
git commit -m "eval: add openeqa.py loader, formatter, run loop

Parallels eval/vsibench.py. Filters to hm3d-v0/ by default, stratified
sample by category, open-ended QA prompt (no MCA/NA branching), captures
ReAct intermediate_steps for the trace inspector."
```

---

### Task 10: `scripts/07_run_openeqa.py` entry point

**Files:**
- Create: `experiments/eva-eval/scripts/07_run_openeqa.py`

- [ ] **Step 1: Implement the entry point**

```python
# experiments/eva-eval/scripts/07_run_openeqa.py
"""Phase: run agent over sampled OpenEQA questions.

Usage:
    python scripts/07_run_openeqa.py \
        --sampled-json <cache_root>/openeqa_hm3d/sampled_50.json \
        --cache-root <cache_root> \
        --paper-code-dir <path/to/literature/EmbodiedVideoAgent/code> \
        --classes-file <path/to/detection_classes.txt> \
        --output results/openeqa_hm3d_dev50.jsonl \
        [--planner NAME] [--resume] [--no-capture-trace]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eva_eval.eval.openeqa import run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampled-json", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--paper-code-dir", type=Path, required=True)
    ap.add_argument("--classes-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--planner", default=None)
    ap.add_argument("--max-iterations", type=int, default=30)
    ap.add_argument("--no-capture-trace", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    summary = run(
        sampled_json=args.sampled_json,
        cache_root=args.cache_root,
        paper_code_dir=args.paper_code_dir,
        classes_file=args.classes_file,
        output=args.output,
        planner=args.planner,
        max_iterations=args.max_iterations,
        capture_trace=not args.no_capture_trace,
        resume=args.resume,
    )
    print(f"\n{summary['n_done']} / {summary['n_total']} questions answered. Output: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add experiments/eva-eval/scripts/07_run_openeqa.py
git commit -m "scripts: add 07_run_openeqa.py entry point"
```

---

### Task 11: `eva_eval/eval/openeqa_grade.py` LLM-as-judge grader

**Files:**
- Create: `experiments/eva-eval/eva_eval/eval/openeqa_grade.py`
- Test: `experiments/eva-eval/tests/test_openeqa_grade.py`

- [ ] **Step 1: Write the failing test**

```python
# experiments/eva-eval/tests/test_openeqa_grade.py
from __future__ import annotations

import pytest


def test_judge_prompt_includes_question_answer_response():
    from eva_eval.eval.openeqa_grade import build_judge_prompt

    prompt = build_judge_prompt(
        question="How many chairs are there?",
        gold_answer="There are three chairs.",
        prediction="3",
    )
    assert "How many chairs are there?" in prompt
    assert "There are three chairs." in prompt
    assert "3" in prompt
    assert "single integer" in prompt.lower()


def test_parse_judge_score_extracts_first_integer():
    from eva_eval.eval.openeqa_grade import parse_judge_score

    assert parse_judge_score("5") == 5
    assert parse_judge_score("4\nbecause...") == 4
    assert parse_judge_score("Score: 3 (the response is partially correct)") == 3


def test_parse_judge_score_clamps_out_of_range():
    from eva_eval.eval.openeqa_grade import parse_judge_score

    assert parse_judge_score("7") == 5
    assert parse_judge_score("0") == 1


def test_parse_judge_score_returns_none_when_no_integer():
    from eva_eval.eval.openeqa_grade import parse_judge_score

    assert parse_judge_score("no idea") is None
    assert parse_judge_score("") is None


def test_c_score_normalizes_one_to_zero_and_five_to_hundred():
    from eva_eval.eval.openeqa_grade import c_score

    assert c_score(1) == 0.0
    assert c_score(5) == 100.0
    assert c_score(3) == 50.0


def test_c_score_returns_none_for_none():
    from eva_eval.eval.openeqa_grade import c_score

    assert c_score(None) is None


def test_aggregate_overall_and_per_category():
    import math
    from eva_eval.eval.openeqa_grade import aggregate

    rows = [
        {"category": "object_recognition", "score": 5},
        {"category": "object_recognition", "score": 3},
        {"category": "spatial_reasoning", "score": 1},
    ]
    out = aggregate(rows)
    assert out["n_questions"] == 3
    # object_recognition mean = 4 → c_score 75
    assert math.isclose(out["per_category"]["object_recognition"], 75.0)
    # spatial_reasoning mean = 1 → c_score 0
    assert math.isclose(out["per_category"]["spatial_reasoning"], 0.0)
    # overall mean across all rows = (5+3+1)/3 = 3 → c_score 50
    assert math.isclose(out["overall"], 50.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd experiments/eva-eval
pytest tests/test_openeqa_grade.py -v
```

Expected: tests FAIL with `ModuleNotFoundError: No module named 'eva_eval.eval.openeqa_grade'`.

- [ ] **Step 3: Implement the grader**

```python
# experiments/eva-eval/eva_eval/eval/openeqa_grade.py
"""LLM-as-judge grader for OpenEQA predictions.

Default judge: any text-only model from config/models.yaml. Apples-to-apples
with the OpenEQA paper requires GPT-4-class judging — use --judge gpt-4o
in scripts/08_grade_openeqa.py for that.
"""
from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from typing import Iterable


JUDGE_PROMPT_TEMPLATE = (
    "You are an AI assistant who will help me evaluate the response given the "
    "question and the correct answer.\n"
    "To mark a response, you should output a single integer between 1 and 5 "
    "(including 1 and 5).\n"
    "5 means that the response perfectly matches the answer.\n"
    "4 means that the response is mostly correct but missing minor details.\n"
    "3 means that the response is partially correct.\n"
    "2 means that the response is mostly incorrect but has some relation to the answer.\n"
    "1 means that the response is completely incorrect.\n"
    "Question: {question}\n"
    "Correct Answer: {answer}\n"
    "Response: {response}\n"
    "Output a single integer:"
)


def build_judge_prompt(*, question: str, gold_answer: str, prediction: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question.strip(),
        answer=gold_answer.strip(),
        response=prediction.strip(),
    )


def parse_judge_score(text: str) -> int | None:
    """Extract the first integer in [1, 5] from a judge response. Clamps to range.
    Returns None if no integer is present."""
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if match is None:
        return None
    n = int(match.group())
    if n < 1:
        return 1
    if n > 5:
        return 5
    return n


def c_score(score_1_to_5: int | None) -> float | None:
    """100 * (s - 1) / 4: maps 1→0, 5→100. Returns None if input is None."""
    if score_1_to_5 is None:
        return None
    return 100.0 * (float(score_1_to_5) - 1.0) / 4.0


def grade_one(judge, *, question: str, gold_answer: str, prediction: str) -> tuple[int | None, str]:
    """Call the judge model on one (question, gold, prediction) triple. Returns
    (1-5 score or None, raw judge response text)."""
    prompt = build_judge_prompt(question=question, gold_answer=gold_answer, prediction=prediction)
    raw = judge.chat([{"role": "user", "content": prompt}])
    return parse_judge_score(raw), raw


def aggregate(rows: Iterable[dict]) -> dict:
    """Compute overall + per-category C-scores from graded rows.

    Each row must contain `category` and `score` (1-5 or None). Rows with
    None scores are excluded from category and overall means."""
    by_cat: dict[str, list[int]] = defaultdict(list)
    all_scores: list[int] = []
    for r in rows:
        s = r.get("score")
        if s is None:
            continue
        by_cat[r.get("category", "?")].append(int(s))
        all_scores.append(int(s))

    per_category: "OrderedDict[str, float]" = OrderedDict()
    for cat in sorted(by_cat):
        mean_s = sum(by_cat[cat]) / len(by_cat[cat])
        per_category[cat] = float(c_score(mean_s))

    overall = float(c_score(sum(all_scores) / len(all_scores))) if all_scores else 0.0
    return {
        "overall": overall,
        "per_category": per_category,
        "n_questions": sum(len(v) for v in by_cat.values()),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd experiments/eva-eval
pytest tests/test_openeqa_grade.py -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/eva-eval/eva_eval/eval/openeqa_grade.py \
        experiments/eva-eval/tests/test_openeqa_grade.py
git commit -m "eval: add openeqa_grade.py LLM-as-judge

Implements the OpenEQA C-score grading: judge prompt builder, score parsing
with clamping, normalization to [0, 100], per-category aggregation."
```

---

### Task 12: `scripts/08_grade_openeqa.py` entry point

**Files:**
- Create: `experiments/eva-eval/scripts/08_grade_openeqa.py`

- [ ] **Step 1: Implement the script**

```python
# experiments/eva-eval/scripts/08_grade_openeqa.py
"""Phase: grade predictions JSONL with an LLM-as-judge.

Usage:
    python scripts/08_grade_openeqa.py \
        --predictions results/openeqa_hm3d_dev50.jsonl \
        --judge qwen2.5-7b-text \
        --output results/openeqa_hm3d_dev50_graded.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eva_eval.eval.openeqa_grade import aggregate, grade_one
from eva_eval.llm.client import load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True, help="JSONL produced by 07_run_openeqa.py")
    ap.add_argument("--output", type=Path, required=True, help="Graded JSONL output")
    ap.add_argument("--judge", default="qwen2.5-7b-text", help="Model name from config/models.yaml")
    args = ap.parse_args()

    judge = load_model(args.judge)
    print(f"Using judge: {args.judge} ({judge.model})")

    rows_in = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    print(f"Grading {len(rows_in)} predictions")

    graded: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in rows_in:
            try:
                score, rationale = grade_one(
                    judge,
                    question=r["question"],
                    gold_answer=r["ground_truth"],
                    prediction=r["prediction"],
                )
            except Exception as e:
                score, rationale = None, f"(judge error: {type(e).__name__}: {e})"
                print(f"[err] {r.get('id')}: {rationale}", file=sys.stderr)
            out = dict(r)
            out["score"] = score
            out["judge_rationale"] = rationale
            graded.append(out)
            f.write(json.dumps(out, default=str) + "\n")
            f.flush()

    summary = aggregate(graded)
    print("\n=== Summary ===")
    print(f"  overall:        {summary['overall']:7.2f}")
    print(f"  n_questions:    {summary['n_questions']}")
    for cat, sc in summary["per_category"].items():
        print(f"  {cat:30s} {sc:7.2f}")

    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(dict(summary), indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add experiments/eva-eval/scripts/08_grade_openeqa.py
git commit -m "scripts: add 08_grade_openeqa.py entry point"
```

---

### Task 13: `scripts/inspect_agent_trace.py`

**Files:**
- Create: `experiments/eva-eval/scripts/inspect_agent_trace.py`
- Test: `experiments/eva-eval/tests/test_inspect_agent_trace.py`

- [ ] **Step 1: Write the failing test**

```python
# experiments/eva-eval/tests/test_inspect_agent_trace.py
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_render_trace_markdown_includes_question_steps_answer(tmp_path):
    from eva_eval.debug.agent_trace import render_trace_markdown

    row = {
        "id": "q1",
        "category": "object_recognition",
        "question": "How many chairs?",
        "ground_truth": "Three.",
        "prediction": "3",
        "score": 4,
        "judge_rationale": "Mostly correct, missing units.",
        "intermediate_steps": [
            {"tool": "frame_localization", "tool_input": "chair", "log": "Thought: I should localize chairs.\nAction: frame_localization\nAction Input: \"chair\"", "observation": "[1, 5, 9]"},
            {"tool": "frame_VQA", "tool_input": "(\"how many chairs?\", 5)", "log": "Thought: ask the VLM.", "observation": "Three chairs visible."},
        ],
    }
    md = render_trace_markdown(row)
    assert "How many chairs?" in md
    assert "Three." in md
    assert "frame_localization" in md
    assert "frame_VQA" in md
    assert "## Step 1" in md
    assert "## Step 2" in md
    assert "score: 4" in md.lower()


def test_render_trace_markdown_handles_empty_steps():
    from eva_eval.debug.agent_trace import render_trace_markdown

    row = {
        "id": "q2",
        "category": "x",
        "question": "?",
        "ground_truth": "?",
        "prediction": "",
        "intermediate_steps": [],
    }
    md = render_trace_markdown(row)
    assert "(no intermediate steps recorded)" in md


def test_find_question_in_jsonl(tmp_path):
    from eva_eval.debug.agent_trace import find_question

    p = tmp_path / "graded.jsonl"
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert find_question(p, "b") == {"id": "b"}
    with pytest.raises(KeyError):
        find_question(p, "nope")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd experiments/eva-eval
pytest tests/test_inspect_agent_trace.py -v
```

Expected: tests FAIL with `ModuleNotFoundError: No module named 'eva_eval.debug.agent_trace'`.

- [ ] **Step 3: Implement the helper module**

```python
# experiments/eva-eval/eva_eval/debug/agent_trace.py
"""Render a single agent ReAct trace as a markdown document."""
from __future__ import annotations

import json
from pathlib import Path


def find_question(predictions_jsonl: str | Path, question_id: str) -> dict:
    """Locate a row by `id` in a predictions JSONL. Raises KeyError if absent."""
    p = Path(predictions_jsonl)
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("id") == question_id:
            return row
    raise KeyError(f"question_id {question_id!r} not found in {p}")


def render_trace_markdown(row: dict) -> str:
    """Render a row (from predictions JSONL or graded JSONL) as markdown."""
    parts: list[str] = []
    parts.append(f"# Trace — `{row.get('id', '?')}` ({row.get('category', '?')})")
    parts.append("")
    parts.append(f"**Question:** {row.get('question', '')}")
    parts.append("")
    parts.append(f"**Ground truth:** {row.get('ground_truth', '')}")
    parts.append("")
    parts.append(f"**Prediction:** {row.get('prediction', '')}")
    parts.append("")
    if "score" in row and row["score"] is not None:
        parts.append(f"**Judge score:** {row['score']} / 5")
        if row.get("judge_rationale"):
            parts.append("")
            parts.append(f"_Judge rationale:_ {row['judge_rationale']}")
        parts.append("")

    parts.append("---")
    parts.append("")
    steps = row.get("intermediate_steps") or []
    if not steps:
        parts.append("(no intermediate steps recorded)")
        return "\n".join(parts)

    for i, step in enumerate(steps, start=1):
        parts.append(f"## Step {i}: `{step.get('tool', '?')}`")
        parts.append("")
        if step.get("log"):
            parts.append("**Thought / Action log:**")
            parts.append("")
            parts.append("```")
            parts.append(str(step["log"]).strip())
            parts.append("```")
            parts.append("")
        if step.get("tool_input") is not None:
            parts.append(f"**Tool input:** `{step['tool_input']}`")
            parts.append("")
        obs = str(step.get("observation", ""))
        if len(obs) > 1500:
            obs = obs[:1500] + "\n... (truncated)"
        parts.append("**Observation:**")
        parts.append("")
        parts.append("```")
        parts.append(obs)
        parts.append("```")
        parts.append("")

    return "\n".join(parts)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd experiments/eva-eval
pytest tests/test_inspect_agent_trace.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Implement the CLI script**

```python
# experiments/eva-eval/scripts/inspect_agent_trace.py
"""Render markdown trace for one question from a predictions JSONL.

Usage:
    python scripts/inspect_agent_trace.py <jsonl> <question_id> [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eva_eval.debug.agent_trace import find_question, render_trace_markdown


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("question_id")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path; defaults to <jsonl_dir>/_inspect/trace_<id>.md")
    args = ap.parse_args()

    row = find_question(args.jsonl, args.question_id)
    md = render_trace_markdown(row)
    out = args.out or args.jsonl.parent / "_inspect" / f"trace_{args.question_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add experiments/eva-eval/eva_eval/debug/agent_trace.py \
        experiments/eva-eval/scripts/inspect_agent_trace.py \
        experiments/eva-eval/tests/test_inspect_agent_trace.py
git commit -m "inspect: add inspect_agent_trace.py for ReAct trace rendering

Renders one question's full Thought/Action/Observation chain as markdown
for human review. Reads from a predictions JSONL written by 07_run_openeqa.py."
```

---

### Task 14: `scripts/inspect_grading.py`

**Files:**
- Create: `experiments/eva-eval/scripts/inspect_grading.py`
- Test: `experiments/eva-eval/tests/test_inspect_grading.py`

- [ ] **Step 1: Write the failing test**

```python
# experiments/eva-eval/tests/test_inspect_grading.py
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_graded(tmp_path: Path) -> Path:
    rows = [
        {"id": f"q{i}", "category": "object_recognition" if i % 2 else "spatial_reasoning",
         "question": f"Question {i}?", "ground_truth": f"Answer {i}", "prediction": f"Pred {i}",
         "score": (i % 5) + 1, "judge_rationale": f"reason {i}"}
        for i in range(20)
    ]
    p = tmp_path / "graded.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_render_grading_html_writes_file_with_summary_sections(tmp_path):
    from eva_eval.debug.grading import render_grading_html

    graded = _write_graded(tmp_path)
    out = render_grading_html(graded)
    assert out.exists()
    text = out.read_text()
    assert "Grading inspection" in text
    assert "Per-category" in text
    assert "Worst-10" in text
    assert "Best-10" in text


def test_render_grading_html_includes_judge_score_histogram(tmp_path):
    from eva_eval.debug.grading import render_grading_html

    graded = _write_graded(tmp_path)
    out = render_grading_html(graded)
    text = out.read_text()
    # Histogram has 5 buckets labeled 1..5
    for label in ("score 1", "score 2", "score 3", "score 4", "score 5"):
        assert label in text
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd experiments/eva-eval
pytest tests/test_inspect_grading.py -v
```

Expected: tests FAIL with `ModuleNotFoundError: No module named 'eva_eval.debug.grading'`.

- [ ] **Step 3: Implement the helper module**

```python
# experiments/eva-eval/eva_eval/debug/grading.py
"""Renderer for the graded-results inspection HTML."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from eva_eval.debug.render import write_html
from eva_eval.eval.openeqa_grade import aggregate, c_score


def render_grading_html(graded_jsonl: str | Path) -> Path:
    p = Path(graded_jsonl)
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    summary = aggregate(rows)
    per_cat_rows = defaultdict(list)
    for r in rows:
        per_cat_rows[r.get("category", "?")].append(r)

    body_parts: list[str] = []
    body_parts.append(f"<h1>Grading inspection — <code>{p.name}</code></h1>")

    body_parts.append("<h2>Per-category C-scores</h2>")
    body_parts.append(_summary_table(summary))

    body_parts.append("<h2>Judge score histogram (1–5)</h2>")
    body_parts.append(_histogram(rows))

    body_parts.append("<h2>Worst-10 and Best-10 per category</h2>")
    for cat in sorted(per_cat_rows):
        body_parts.append(f"<h3>{cat}</h3>")
        body_parts.append(_examples_table(per_cat_rows[cat], "Worst-10", n=10, ascending=True))
        body_parts.append(_examples_table(per_cat_rows[cat], "Best-10", n=10, ascending=False))

    out = p.parent / (p.stem + ".inspect.html")
    return write_html(out, title=f"grading: {p.name}", body="\n".join(body_parts))


def _summary_table(summary: dict) -> str:
    rows = [("overall", f"{summary['overall']:.2f}")]
    rows.append(("n_questions", str(summary["n_questions"])))
    for cat, sc in summary["per_category"].items():
        rows.append((cat, f"{sc:.2f}"))
    body = "".join(f"<tr><th>{k}</th><td><code>{v}</code></td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _histogram(rows: Iterable[dict]) -> str:
    counts = Counter(int(r["score"]) for r in rows if r.get("score") is not None)
    cells = []
    for s in (1, 2, 3, 4, 5):
        n = counts.get(s, 0)
        cells.append(f"<tr><th>score {s}</th><td><code>{n}</code></td></tr>")
    return f"<table>{''.join(cells)}</table>"


def _examples_table(rows: list[dict], label: str, *, n: int, ascending: bool) -> str:
    scored = [r for r in rows if r.get("score") is not None]
    scored.sort(key=lambda r: int(r["score"]), reverse=not ascending)
    pick = scored[:n]
    if not pick:
        return f"<p><em>{label}: (no scored rows)</em></p>"
    th = (
        "<tr><th>id</th><th>score</th><th>question</th><th>gold</th>"
        "<th>prediction</th><th>rationale</th></tr>"
    )
    body = []
    for r in pick:
        body.append(
            f"<tr><td><code>{r.get('id','')}</code></td>"
            f"<td>{r['score']}</td>"
            f"<td>{(r.get('question') or '')[:160]}</td>"
            f"<td>{(r.get('ground_truth') or '')[:160]}</td>"
            f"<td>{(r.get('prediction') or '')[:160]}</td>"
            f"<td>{(r.get('judge_rationale') or '')[:200]}</td></tr>"
        )
    return f"<h4>{label}</h4><table>{th}{''.join(body)}</table>"
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd experiments/eva-eval
pytest tests/test_inspect_grading.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Implement the CLI script**

```python
# experiments/eva-eval/scripts/inspect_grading.py
"""Render grading inspection HTML for a graded predictions JSONL.

Usage:
    python scripts/inspect_grading.py <graded_jsonl>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eva_eval.debug.grading import render_grading_html


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("graded_jsonl", type=Path)
    args = ap.parse_args()
    out = render_grading_html(args.graded_jsonl)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add experiments/eva-eval/eva_eval/debug/grading.py \
        experiments/eva-eval/scripts/inspect_grading.py \
        experiments/eva-eval/tests/test_inspect_grading.py
git commit -m "inspect: add inspect_grading.py for graded-results review

Renders per-category C-scores, judge score histogram, and worst-10/best-10
examples per category for human review."
```

---

### Task 15: Shell wrappers `15_*.sh` through `17_*.sh`

**Files:**
- Create: `experiments/eva-eval/scripts-remote/15_openeqa_run.sh`
- Create: `experiments/eva-eval/scripts-remote/16_openeqa_grade.sh`
- Create: `experiments/eva-eval/scripts-remote/17_openeqa_inspect_results.sh`

- [ ] **Step 1: Implement `15_openeqa_run.sh`**

```bash
#!/usr/bin/env bash
# Run agent over the sampled OpenEQA questions. Requires servers running.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

curl -fs http://127.0.0.1:18000/v1/models >/dev/null || { echo "Qwen :18000 not ready, run 03_start_servers"; exit 1; }
curl -fs http://127.0.0.1:18001/v1/models >/dev/null || { echo "InternVL2 :18001 not ready, run 03_start_servers"; exit 1; }

OUT="${OPENEQA_RESULTS:-$RESULTS_DIR/openeqa_hm3d_dev50.jsonl}"

echo "============ 15_openeqa_run ============"
date

PYTHONPATH="$EVA_EVAL_DIR:$PAPER_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/07_run_openeqa.py" \
    --sampled-json "$OPENEQA_SAMPLED_JSON" \
    --cache-root "$OPENEQA_CACHE_ROOT" \
    --paper-code-dir "$PAPER_DIR" \
    --classes-file "$CLASSES_FILE" \
    --output "$OUT" \
    --resume

echo "==> done. Predictions in $OUT. Next: bash scripts-remote/16_openeqa_grade.sh"
date
```

- [ ] **Step 2: Implement `16_openeqa_grade.sh`**

```bash
#!/usr/bin/env bash
# Grade predictions with default judge (Qwen2.5-7B). Override with $1.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

JUDGE="${1:-qwen2.5-7b-text}"
PRED="${OPENEQA_RESULTS:-$RESULTS_DIR/openeqa_hm3d_dev50.jsonl}"
GRADED="${PRED%.jsonl}_graded_${JUDGE}.jsonl"

echo "============ 16_openeqa_grade ============"
date
echo "predictions: $PRED"
echo "judge:       $JUDGE"
echo "output:      $GRADED"

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/08_grade_openeqa.py" \
    --predictions "$PRED" \
    --judge "$JUDGE" \
    --output "$GRADED"

echo "==> done. Graded results in $GRADED. Next: bash scripts-remote/17_openeqa_inspect_results.sh"
date
```

- [ ] **Step 3: Implement `17_openeqa_inspect_results.sh`**

```bash
#!/usr/bin/env bash
# Render the grading inspection HTML for the most recent graded JSONL.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

GRADED=$(ls -t "$RESULTS_DIR"/openeqa_hm3d_dev50_graded_*.jsonl 2>/dev/null | head -1)
if [ -z "$GRADED" ]; then
    echo "ERROR: no graded jsonl found under $RESULTS_DIR" >&2
    exit 1
fi

echo "============ 17_openeqa_inspect_results ============"
date
echo "graded: $GRADED"

PYTHONPATH="$EVA_EVAL_DIR" \
"$EVA_ENV/bin/python" "$EVA_EVAL_DIR/scripts/inspect_grading.py" "$GRADED"

echo
echo "==> Open this file in a browser:"
echo "    ${GRADED%.jsonl}.inspect.html"
echo
echo "==> Decision rule:"
echo "    overall in 30–50 with sensible per-category → agent is OK; bug is VSI-Bench-specific"
echo "    overall <20 or any category at 0% → bug in shared pipeline; use inspect_agent_trace.py to dig"
date
```

- [ ] **Step 4: Make the scripts executable and commit**

```bash
chmod +x experiments/eva-eval/scripts-remote/15_openeqa_run.sh \
         experiments/eva-eval/scripts-remote/16_openeqa_grade.sh \
         experiments/eva-eval/scripts-remote/17_openeqa_inspect_results.sh

git add experiments/eva-eval/scripts-remote/15_openeqa_run.sh \
        experiments/eva-eval/scripts-remote/16_openeqa_grade.sh \
        experiments/eva-eval/scripts-remote/17_openeqa_inspect_results.sh
git commit -m "scripts-remote: add 15..17 OpenEQA run/grade/inspect wrappers

15_run uses --resume by default. 16_grade defaults to qwen2.5-7b-text judge
but accepts an override (e.g. 'gpt-4o' for paper-faithful re-grading).
17_inspect picks up the most-recent graded jsonl."
```

---

## Final acceptance check

After all 15 tasks complete, run end-to-end on the server and verify:

- [ ] All test files pass: `cd experiments/eva-eval && pytest -v`
- [ ] `bash scripts-remote/11_openeqa_setup.sh` succeeds and writes `questions.json`
- [ ] `bash scripts-remote/12_openeqa_sample.sh` writes `sampled_50.json` with ~50 questions
- [ ] `bash scripts-remote/13_openeqa_preprocess.sh` produces a cache dir with `memory.pkl` for every sampled episode
- [ ] `bash scripts-remote/14_openeqa_inspect_first.sh` produces both HTMLs; **human gate: open in a browser, verify bboxes land on objects**
- [ ] `bash scripts-remote/15_openeqa_run.sh` produces a predictions JSONL with all 50 questions answered (errors logged but allowed) and `intermediate_steps` populated
- [ ] `bash scripts-remote/16_openeqa_grade.sh` produces a graded JSONL with C-scores in [0, 100]
- [ ] `bash scripts-remote/17_openeqa_inspect_results.sh` produces summary HTML; **human gate: open in browser, look at worst/best-10**

**Decision after final run:**
- C-score in 30–50 with sensible per-category distribution → agent stack works; the VSI-Bench gap is VSI-Bench-specific. Investigate VSI-Bench preprocessing/question handling next.
- C-score <20 or any category at 0% → bug in shared pipeline. Run `inspect_agent_trace.py` on a worst-case question to localize.
