"""Tests for QwenLocalClient AutoGen ModelClient implementation."""

import pytest
from unittest.mock import MagicMock, patch
from qwen_client import QwenLocalClient


class TestQwenLocalClientProtocol:
    """Verify QwenLocalClient implements the ModelClient protocol."""

    def test_has_create_method(self):
        assert hasattr(QwenLocalClient, "create")
        assert callable(getattr(QwenLocalClient, "create"))

    def test_has_message_retrieval_method(self):
        assert hasattr(QwenLocalClient, "message_retrieval")
        assert callable(getattr(QwenLocalClient, "message_retrieval"))

    def test_has_cost_method(self):
        assert hasattr(QwenLocalClient, "cost")
        assert callable(getattr(QwenLocalClient, "cost"))

    def test_has_get_usage_method(self):
        assert hasattr(QwenLocalClient, "get_usage")
        assert callable(getattr(QwenLocalClient, "get_usage"))


class TestQwenLocalClientCreate:
    """Tests for the create method."""

    def _make_dry_client(self):
        """Create a QwenLocalClient in dry-run mode (model=None)."""
        config = {"model": "qwen2_5vl-3b"}
        return QwenLocalClient(config, model=None, processor=None)

    def test_create_returns_response_with_choices(self):
        client = self._make_dry_client()
        params = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is in this image?"},
            ]
        }
        response = client.create(params)
        assert hasattr(response, "choices")
        assert len(response.choices) >= 1
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")
        assert isinstance(response.choices[0].message.content, str)

    def test_create_returns_model_name(self):
        client = self._make_dry_client()
        params = {"messages": [{"role": "user", "content": "test"}]}
        response = client.create(params)
        assert hasattr(response, "model")
        assert response.model == "qwen2_5vl-3b"

    def test_message_retrieval_returns_list_of_strings(self):
        client = self._make_dry_client()
        params = {"messages": [{"role": "user", "content": "test"}]}
        response = client.create(params)
        messages = client.message_retrieval(response)
        assert isinstance(messages, list)
        assert all(isinstance(m, str) for m in messages)

    def test_cost_returns_zero(self):
        client = self._make_dry_client()
        assert client.cost(MagicMock()) == 0.0

    def test_get_usage_returns_expected_keys(self):
        usage = QwenLocalClient.get_usage(MagicMock())
        expected_keys = {"prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"}
        assert set(usage.keys()) == expected_keys

    def test_system_message_forwarded(self):
        """Verify system message from CoTAPrompt is preserved in converted messages."""
        client = self._make_dry_client()
        system_content = "[BEGIN OF GOAL]\nYou are a helpful assistant...\n[END OF GOAL]"
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "What do you see?"},
        ]
        qwen_msgs = client._convert_messages(messages)
        sys_msgs = [m for m in qwen_msgs if m.get("role") == "system"]
        assert len(sys_msgs) >= 1
        assert system_content in sys_msgs[0]["content"]

    def test_first_user_msg_always_gets_images(self):
        """First user message should always embed all input images, even without image-N refs."""
        client = self._make_dry_client()
        client.image_paths = ["/tmp/img0.jpg", "/tmp/img1.jpg"]
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "What color is the car?"},
        ]
        qwen_msgs = client._convert_messages(messages)
        user_msg = qwen_msgs[1]  # index 1 because index 0 is system
        assert isinstance(user_msg["content"], list)
        image_parts = [p for p in user_msg["content"] if p.get("type") == "image"]
        assert len(image_parts) == 2
        assert image_parts[0]["image"] == "/tmp/img0.jpg"
        assert image_parts[1]["image"] == "/tmp/img1.jpg"

    def test_image_paths_converted_in_messages(self):
        """Verify image-0 references get converted to image content parts."""
        client = self._make_dry_client()
        client.image_paths = ["/tmp/test.jpg"]
        messages = [
            {"role": "user", "content": "Describe image-0"},
        ]
        qwen_msgs = client._convert_messages(messages)
        user_msg = qwen_msgs[0]
        assert isinstance(user_msg["content"], list)
        image_parts = [p for p in user_msg["content"] if p.get("type") == "image"]
        assert len(image_parts) == 1
        assert image_parts[0]["image"] == "/tmp/test.jpg"

    def test_subsequent_msgs_dont_get_unconditional_images(self):
        """Only the first user message gets unconditional images; later ones need refs."""
        client = self._make_dry_client()
        client.image_paths = ["/tmp/img0.jpg"]
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "user", "content": "Follow-up without image ref"},
        ]
        qwen_msgs = client._convert_messages(messages)
        # First message should have image
        assert isinstance(qwen_msgs[0]["content"], list)
        # Second message should be plain text (no image-N refs, no <img> tags)
        assert isinstance(qwen_msgs[1]["content"], str)

    def test_img_tag_pattern_converted(self):
        """Verify <img path> patterns from FeedbackPrompt are converted."""
        client = self._make_dry_client()
        messages = [
            {"role": "user", "content": "OBSERVATION:\n{'image': 'result: <img /tmp/result.jpg>'}"},
        ]
        qwen_msgs = client._convert_messages(messages)
        user_msg = qwen_msgs[0]
        assert isinstance(user_msg["content"], list)
        image_parts = [p for p in user_msg["content"] if p.get("type") == "image"]
        assert len(image_parts) == 1
        assert image_parts[0]["image"] == "/tmp/result.jpg"
