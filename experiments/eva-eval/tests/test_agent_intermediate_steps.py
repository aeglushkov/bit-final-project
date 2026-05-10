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
