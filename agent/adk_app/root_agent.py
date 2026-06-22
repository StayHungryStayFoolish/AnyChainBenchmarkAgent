"""Root ADK agent definition.

The module follows the official ADK convention of exposing ``root_agent``.
When ``google-adk`` is not installed, ``root_agent`` is ``None``. Offline tests
must use the ADK eval scaffold, not the old custom chat loop as a product
compatibility path.
"""

from __future__ import annotations

import inspect

from llm.config import load_llm_config

from .callbacks import before_tool_callback
from .instructions import ROOT_INSTRUCTION
from .tools.registry import get_adk_tools


try:  # pragma: no cover - depends on optional google-adk installation.
    from google.adk.agents import Agent
except ImportError:  # pragma: no cover - exercised when ADK is not installed.
    Agent = None  # type: ignore[assignment]


DEFAULT_MODEL = "gemini-3.1-pro"


def resolve_adk_model(default: str = DEFAULT_MODEL) -> str:
    """Resolve the configured model name without taking over ADK model calls.

    ADK owns model execution. This helper only reads the persistent AnyChain
    Agent config so users can set one model name in ``config/agent_config.sh``.
    If the config is incomplete, keep a real ADK default.
    """
    config = load_llm_config()
    if not config.model:
        return default
    return config.model


def build_root_agent(model: str | None = None, tools: list | None = None):
    """Build the ADK root agent when the optional ADK dependency is available."""
    if Agent is None:
        raise RuntimeError("google-adk is not installed")
    resolved_model = model or resolve_adk_model()
    kwargs = {
        "name": "anychain_benchmark_agent",
        "model": resolved_model,
        "instruction": ROOT_INSTRUCTION,
        "tools": tools if tools is not None else get_adk_tools(include_actions=True),
    }
    if _agent_accepts("before_tool_callback"):
        kwargs["before_tool_callback"] = before_tool_callback
    return Agent(**kwargs)


def _agent_accepts(parameter: str) -> bool:
    """Return whether the installed ADK Agent constructor accepts a parameter."""
    try:
        signature = inspect.signature(Agent)
    except (TypeError, ValueError):
        return True
    if parameter in signature.parameters:
        return True
    return any(item.kind == inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values())


root_agent = build_root_agent() if Agent is not None else None
