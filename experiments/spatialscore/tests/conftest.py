"""Shared pytest fixtures for SpatialAgent tests."""

import os
import sys
import json
import pytest
import tempfile
from unittest.mock import MagicMock, patch

# Add project paths so we can import our modules and the authors' code
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(EXPERIMENTS_DIR))
SPATIALSCORE_CODE = os.path.join(PROJECT_ROOT, "literature", "spatialscore", "code")

sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.join(EXPERIMENTS_DIR, "experiments", "spatialscore"))
sys.path.insert(0, SPATIALSCORE_CODE)
sys.path.insert(0, os.path.join(SPATIALSCORE_CODE, "SpatialAgent"))


@pytest.fixture
def sample_mmvp_task():
    """A sample matching the MMVP dataset format from SpatialScore."""
    return {
        "id": 0,
        "category": "Others",
        "subcategory": "visual_similarity",
        "input_modality": "single-image",
        "question_type": "multi-choice",
        "source": "MMVP",
        "question": "Are the wings of the butterfly open or closed?\n(A) Open\n(B) Closed",
        "answer": "A",
        "img_paths": ["/tmp/test_image_0.jpg"],
    }


@pytest.fixture
def sample_diverse_task():
    """A sample matching the diverse dataset format (depth/distance question)."""
    return {
        "id": 42,
        "category": "Depth and Distance",
        "subcategory": "depth_comparison",
        "input_modality": "single-image",
        "question_type": "multi-choice",
        "source": "3DSRBench",
        "question": "Which object is closer to the camera?\n(A) The chair\n(B) The table",
        "answer": "B",
        "img_paths": ["/tmp/test_image_42.jpg"],
    }


@pytest.fixture
def tmp_result_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_image_path(tmp_result_dir):
    """Create a real test image file for tests that need file I/O."""
    from PIL import Image
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    path = os.path.join(tmp_result_dir, "test_image.jpg")
    img.save(path)
    return path


@pytest.fixture
def mock_vlm_fn():
    """A mock VLM callable for SelfReasoning action."""
    def vlm_fn(image, query):
        return "This scene is indoor. The objects appear on a table."
    return vlm_fn


@pytest.fixture
def mock_model_registry():
    """A mock ModelRegistry with all models returning predictable outputs."""
    registry = MagicMock()

    # Depth model mock
    depth_model = MagicMock()
    import numpy as np
    depth_model.infer_image.return_value = np.random.rand(480, 640).astype(np.float32)
    registry.get_depth_model.return_value = depth_model

    # RAFT model mock
    raft_model = MagicMock()
    registry.get_raft.return_value = raft_model

    # Orient model mock
    orient_model = MagicMock()
    orient_preprocess = MagicMock()
    registry.get_orient_model.return_value = (orient_model, orient_preprocess)

    # SAM2 mock
    sam2_predictor = MagicMock()
    registry.get_sam2.return_value = sam2_predictor

    # RAM mock
    ram_model = MagicMock()
    ram_transform = MagicMock()
    registry.get_ram.return_value = (ram_model, ram_transform)

    return registry
