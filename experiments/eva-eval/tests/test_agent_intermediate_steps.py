"""Verify build_agent honors the return_intermediate_steps flag.

The signature test is the contract: the flag must exist with default False
so existing VSI-Bench callers stay unchanged. The forwarding test reads the
function source to confirm the flag is passed to the AgentExecutor — this
avoids the langchain-version-specific mocking pitfalls of patching the
lazy-imported AgentExecutor symbol."""
from __future__ import annotations

import inspect

from eva_eval.agent import agent as agent_mod


def test_build_agent_signature_has_intermediate_steps_flag():
    sig = inspect.signature(agent_mod.build_agent)
    assert "return_intermediate_steps" in sig.parameters
    assert sig.parameters["return_intermediate_steps"].default is False


def test_build_agent_forwards_flag_to_executor():
    """The function source must pass return_intermediate_steps to AgentExecutor."""
    src = inspect.getsource(agent_mod.build_agent)
    assert "return_intermediate_steps=return_intermediate_steps" in src
