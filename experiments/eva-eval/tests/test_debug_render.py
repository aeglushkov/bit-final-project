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
