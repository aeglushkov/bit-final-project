"""Integration tests for the SpatialAgent conversation loop with mocked models."""

import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from actions import ALL_ACTIONS, Action
from action_wrappers import make_action_registry
from qwen_client import QwenLocalClient

# Add SpatialAgent to path
SPATIALSCORE_CODE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "literature", "spatialscore", "code"
)
sys.path.insert(0, os.path.join(SPATIALSCORE_CODE, "SpatialAgent"))


def _make_mock_assistant(responses):
    """Create a mock assistant that returns pre-defined responses in sequence.

    Instead of using the full AutoGen AssistantAgent (which requires LLM config),
    we simulate the conversation by directly driving the UserAgent.
    """
    from autogen.agentchat import AssistantAgent

    assistant = MagicMock(spec=AssistantAgent)
    assistant.name = "mock_assistant"
    assistant._reply_func_list = []

    response_iter = iter(responses)

    def mock_receive(message, sender, request_reply=False, silent=False):
        try:
            next_response = next(response_iter)
        except StopIteration:
            return
        # Send the response back to the sender (UserAgent)
        sender.receive(
            {"content": next_response, "role": "assistant"},
            assistant,
            request_reply=False,
        )

    assistant.receive.side_effect = mock_receive
    return assistant


class TestAgentSingleStepTerminate:
    """Agent receives question, LLM responds with Terminate immediately."""

    def test_final_answer_set(self, tmp_result_dir):
        from agent import UserAgent
        from utils.prompt import CoTAPrompt, FeedbackPrompt
        from utils.parser import Parser

        prompt_gen = CoTAPrompt(actions=ALL_ACTIONS)
        feedback_gen = FeedbackPrompt()
        parser = Parser(prompt_generator=prompt_gen)

        # Executor with Terminate only
        from utils.executor import Executor
        action_registry = {"Terminate": lambda answer: {"answer": answer}}
        executor = Executor(
            input_folder="/tmp", result_folder=tmp_result_dir,
            action_registry=action_registry,
        )

        user_agent = UserAgent(
            name="user",
            prompt_generator=prompt_gen,
            feedback_generator=feedback_gen,
            parser=parser,
            executor=executor,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            code_execution_config=False,
        )

        terminate_response = json.dumps({
            "thought": "The answer is clearly A based on the image.",
            "actions": [{"name": "Terminate", "arguments": {"answer": "A"}}]
        })

        assistant = _make_mock_assistant([terminate_response])
        task = {"id": 0, "image_paths": ["/tmp/test.jpg"]}
        user_agent.initiate_chat(assistant, message="Which direction?", task=task)

        assert user_agent.final_answer == "A"


class TestAgentToolThenTerminate:
    """Agent calls a tool, gets observation, then terminates."""

    def test_tool_called_then_answer(self, tmp_result_dir):
        from agent import UserAgent
        from utils.prompt import CoTAPrompt, FeedbackPrompt
        from utils.parser import Parser
        from utils.executor import Executor

        prompt_gen = CoTAPrompt(actions=ALL_ACTIONS)
        feedback_gen = FeedbackPrompt()
        parser = Parser(prompt_generator=prompt_gen)

        def mock_self_reasoning(image, query):
            return {"response": "This is an indoor scene."}

        action_registry = {
            "SelfReasoning": mock_self_reasoning,
            "Terminate": lambda answer: {"answer": answer},
        }
        executor = Executor(
            input_folder="/tmp", result_folder=tmp_result_dir,
            action_registry=action_registry,
        )

        user_agent = UserAgent(
            name="user",
            prompt_generator=prompt_gen,
            feedback_generator=feedback_gen,
            parser=parser,
            executor=executor,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            code_execution_config=False,
        )

        responses = [
            json.dumps({
                "thought": "Let me check if this is indoor first.",
                "actions": [{"name": "SelfReasoning", "arguments": {"image": "image-0", "query": "Indoor or outdoor?"}}]
            }),
            json.dumps({
                "thought": "It's indoor. The answer is B.",
                "actions": [{"name": "Terminate", "arguments": {"answer": "B"}}]
            }),
        ]

        assistant = _make_mock_assistant(responses)
        task = {"id": 0, "image_paths": ["/tmp/test.jpg"]}
        user_agent.initiate_chat(assistant, message="Which is closer?", task=task)

        assert user_agent.final_answer == "B"
        assert len(user_agent.called_tools) == 2
        assert user_agent.called_tools[0]["name"] == "SelfReasoning"
        assert user_agent.called_tools[1]["name"] == "Terminate"


class TestAgentMaxSteps:
    """Agent stops when max_consecutive_auto_reply is reached."""

    def test_stops_at_max_steps(self, tmp_result_dir):
        from agent import UserAgent
        from utils.prompt import CoTAPrompt, FeedbackPrompt
        from utils.parser import Parser
        from utils.executor import Executor

        prompt_gen = CoTAPrompt(actions=ALL_ACTIONS)
        feedback_gen = FeedbackPrompt()
        parser = Parser(prompt_generator=prompt_gen)

        action_registry = {
            "SelfReasoning": lambda image, query: {"response": "Observation."},
            "Terminate": lambda answer: {"answer": answer},
        }
        executor = Executor(
            input_folder="/tmp", result_folder=tmp_result_dir,
            action_registry=action_registry,
        )

        user_agent = UserAgent(
            name="user",
            prompt_generator=prompt_gen,
            feedback_generator=feedback_gen,
            parser=parser,
            executor=executor,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            code_execution_config=False,
        )

        # Agent keeps calling SelfReasoning and never terminates
        responses = [
            json.dumps({
                "thought": "Step 1", "actions": [{"name": "SelfReasoning", "arguments": {"image": "image-0", "query": "Q1"}}]
            }),
            json.dumps({
                "thought": "Step 2", "actions": [{"name": "SelfReasoning", "arguments": {"image": "image-0", "query": "Q2"}}]
            }),
            json.dumps({
                "thought": "Step 3", "actions": [{"name": "SelfReasoning", "arguments": {"image": "image-0", "query": "Q3"}}]
            }),
        ]

        assistant = _make_mock_assistant(responses)
        task = {"id": 0, "image_paths": ["/tmp/test.jpg"]}
        user_agent.initiate_chat(assistant, message="Question?", task=task)

        # Should have stopped at 2 steps
        assert user_agent.step_id <= 2
        assert user_agent.final_answer is None


class TestAgentParsingError:
    """Agent handles malformed JSON and sends parsing feedback."""

    def test_recovers_from_bad_json(self, tmp_result_dir):
        from agent import UserAgent
        from utils.prompt import CoTAPrompt, FeedbackPrompt
        from utils.parser import Parser
        from utils.executor import Executor

        prompt_gen = CoTAPrompt(actions=ALL_ACTIONS)
        feedback_gen = FeedbackPrompt()
        parser = Parser(prompt_generator=prompt_gen)

        action_registry = {"Terminate": lambda answer: {"answer": answer}}
        executor = Executor(
            input_folder="/tmp", result_folder=tmp_result_dir,
            action_registry=action_registry,
        )

        user_agent = UserAgent(
            name="user",
            prompt_generator=prompt_gen,
            feedback_generator=feedback_gen,
            parser=parser,
            executor=executor,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            code_execution_config=False,
        )

        responses = [
            "This is not valid JSON at all {broken",  # malformed
            json.dumps({
                "thought": "Let me try again properly.",
                "actions": [{"name": "Terminate", "arguments": {"answer": "C"}}]
            }),
        ]

        assistant = _make_mock_assistant(responses)
        task = {"id": 0, "image_paths": ["/tmp/test.jpg"]}
        user_agent.initiate_chat(assistant, message="Question?", task=task)

        assert user_agent.final_answer == "C"
