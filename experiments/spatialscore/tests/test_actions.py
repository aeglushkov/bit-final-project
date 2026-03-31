"""Tests for action metadata definitions and action wrapper functions."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from actions import Action, ALL_ACTIONS, ACTION_NAMES
from action_wrappers import make_action_registry


class TestActionMetadata:
    """Tests for Action dataclass and ALL_ACTIONS list."""

    def test_all_actions_have_required_fields(self):
        for action in ALL_ACTIONS:
            assert isinstance(action, Action), f"{action} is not an Action instance"
            assert action.name, f"Action missing name"
            assert action.description, f"Action {action.name} missing description"
            assert isinstance(action.args_spec, dict), f"Action {action.name} args_spec is not a dict"
            assert isinstance(action.rets_spec, dict), f"Action {action.name} rets_spec is not a dict"

    def test_expected_action_names(self):
        names = {a.name for a in ALL_ACTIONS}
        expected = {
            "Terminate", "SelfReasoning", "LocalizeObjects",
            "EstimateObjectDepth", "GetObjectMask", "GetObjectOrientation",
            "EstimateOpticalFlow", "EstimateHomographyMatrix",
            "GetCameraParametersVGGT",
        }
        assert names == expected

    def test_action_names_list_matches(self):
        assert set(ACTION_NAMES) == {a.name for a in ALL_ACTIONS}

    def test_cota_prompt_compatible(self):
        """Verify actions have the attributes that CoTAPrompt iterates over."""
        key2word = {"name": "Name", "description": "Description", "args_spec": "Arguments", "rets_spec": "Returns", "examples": "Examples"}
        for action in ALL_ACTIONS:
            for key in key2word:
                assert hasattr(action, key), f"Action {action.name} missing attribute {key}"


class TestActionRegistry:
    """Tests for make_action_registry and individual action wrappers."""

    def test_registry_has_all_keys(self, mock_model_registry, mock_vlm_fn):
        registry = make_action_registry(mock_model_registry, mock_vlm_fn)
        expected_keys = {
            "Terminate", "SelfReasoning", "LocalizeObjects",
            "EstimateObjectDepth", "GetObjectMask", "GetObjectOrientation",
            "EstimateOpticalFlow", "EstimateHomographyMatrix",
            "GetCameraParametersVGGT",
        }
        assert set(registry.keys()) == expected_keys

    def test_all_values_callable(self, mock_model_registry, mock_vlm_fn):
        registry = make_action_registry(mock_model_registry, mock_vlm_fn)
        for name, fn in registry.items():
            assert callable(fn), f"Action {name} is not callable"

    def test_terminate_action(self, mock_model_registry, mock_vlm_fn):
        registry = make_action_registry(mock_model_registry, mock_vlm_fn)
        result = registry["Terminate"](answer="A")
        assert result == {"answer": "A"}

    def test_terminate_with_text_answer(self, mock_model_registry, mock_vlm_fn):
        registry = make_action_registry(mock_model_registry, mock_vlm_fn)
        result = registry["Terminate"](answer="<answer>B</answer>")
        assert result["answer"] == "<answer>B</answer>"

    def test_self_reasoning_action(self, mock_model_registry, mock_vlm_fn):
        registry = make_action_registry(mock_model_registry, mock_vlm_fn)
        result = registry["SelfReasoning"](image="/tmp/test.jpg", query="Is this indoor?")
        assert "response" in result
        assert isinstance(result["response"], str)

    def test_self_reasoning_passes_args_to_vlm(self, mock_model_registry):
        calls = []
        def tracking_vlm(image, query):
            calls.append((image, query))
            return "Indoor scene"

        registry = make_action_registry(mock_model_registry, tracking_vlm)
        registry["SelfReasoning"](image="/tmp/img.jpg", query="Indoor or outdoor?")
        assert len(calls) == 1
        assert calls[0] == ("/tmp/img.jpg", "Indoor or outdoor?")

    def test_localize_objects_returns_regions(self, mock_model_registry, mock_vlm_fn):
        registry = make_action_registry(mock_model_registry, mock_vlm_fn)
        with patch("action_wrappers._run_localize", return_value=[
            {"label": "dog", "bbox": [100, 150, 300, 400], "score": 0.92},
            {"label": "cat", "bbox": [350, 200, 500, 380], "score": 0.88},
        ]):
            result = registry["LocalizeObjects"](image="/tmp/img.jpg", objects=["dog", "cat"])
        assert "regions" in result
        assert len(result["regions"]) == 2
        for region in result["regions"]:
            assert "label" in region
            assert "bbox" in region
            assert "score" in region

    def test_estimate_depth_returns_results(self, mock_model_registry, mock_vlm_fn, test_image_path):
        registry = make_action_registry(mock_model_registry, mock_vlm_fn)
        with patch("action_wrappers._run_localize", return_value=[
            {"label": "chair", "bbox": [100, 150, 200, 300], "score": 0.9},
        ]):
            result = registry["EstimateObjectDepth"](
                image=test_image_path, objects=["chair"], indoor_or_outdoor="indoor"
            )
        assert "results" in result
        assert len(result["results"]) >= 1
        for r in result["results"]:
            assert "object" in r
            assert "depth" in r

    def test_get_object_mask_returns_results(self, mock_model_registry, mock_vlm_fn, test_image_path):
        registry = make_action_registry(mock_model_registry, mock_vlm_fn)
        # Mock SAM2 predictor
        sam2 = mock_model_registry.get_sam2()
        sam2.set_image.return_value = None
        sam2.predict.return_value = (
            np.ones((1, 480, 640), dtype=bool),  # masks
            np.array([0.95]),  # scores
            None,  # logits
        )
        with patch("action_wrappers._run_localize", return_value=[
            {"label": "eggs", "bbox": [150, 200, 280, 340], "score": 0.9},
        ]):
            result = registry["GetObjectMask"](image=test_image_path, objects=["eggs"])
        assert "results" in result
        for r in result["results"]:
            assert "object" in r
            assert "mask_area" in r
            assert "bbox" in r

    def test_get_orientation_returns_results(self, mock_model_registry, mock_vlm_fn, test_image_path):
        pytest.importorskip("torch")
        import torch
        registry = make_action_registry(mock_model_registry, mock_vlm_fn)
        orient_model = MagicMock()
        orient_model.return_value = torch.tensor([315.0, 90.0, 0.0, 0.89])
        orient_model.parameters.return_value = iter([MagicMock(device="cpu")])
        mock_model_registry.get_orient_model.return_value = (orient_model, MagicMock())
        with patch("action_wrappers._run_localize", return_value=[
            {"label": "person", "bbox": [100, 50, 300, 400], "score": 0.95},
        ]):
            with patch("action_wrappers._crop_and_preprocess_for_orient", return_value=MagicMock(dim=lambda: 3, unsqueeze=lambda x: MagicMock(to=lambda d: MagicMock()))):
                result = registry["GetObjectOrientation"](image=test_image_path, objects="person")
        assert "results" in result
        for r in result["results"]:
            assert "object" in r
            assert "angle_data" in r
