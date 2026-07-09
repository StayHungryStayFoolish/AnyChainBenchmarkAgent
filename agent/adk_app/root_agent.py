"""Root ADK agent definition.

The module follows the official ADK convention of exposing ``root_agent``.
When ``google-adk`` is not installed, ``root_agent`` is ``None``. Offline tests
must use the ADK eval scaffold or deterministic tool tests, not a second
product conversation runtime.
"""

from __future__ import annotations

import inspect

try:
    from ..llm.config import load_llm_config
except ImportError:  # script execution with agent/ on sys.path
    from llm.config import load_llm_config

from .tools.registry import get_adk_tools


try:  # pragma: no cover - depends on optional google-adk installation.
    from google.adk.agents import Agent
except ImportError:  # pragma: no cover - exercised when ADK is not installed.
    Agent = None  # type: ignore[assignment]

try:  # pragma: no cover - depends on optional ADK extensions.
    from google.adk.models.lite_llm import LiteLlm
except ImportError:  # pragma: no cover - exercised when extensions are not installed.
    LiteLlm = None  # type: ignore[assignment]


DEFAULT_MODEL = "gemini-3.1-pro"


ADK_MODEL_BRIDGE_INSTRUCTION = """
You are the ADK model/tool surface for AnyChain Benchmark Agent.

The product workflow, state machine, human confirmation flow, and benchmark
execution gates are owned by the LangGraph Harness in agent.harness. Do not
invent a separate wizard, mutate workflow state, or claim that a benchmark can
run without Harness approval. Use available read-only, planning, validation,
auth, enterprise, and knowledge tools only to answer framework questions or
produce typed evidence for the Harness.
"""


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
    """Build the ADK root agent when the optional ADK dependency is available.

    The product workflow is owned by ``agent.harness``. This ADK agent is kept
    as a provider/tool surface for ADK discovery and Gemini/google tooling; it
    must not expose the retired workflow-state mutation tools or callback-based
    workflow repair logic.
    """
    if Agent is None:
        raise RuntimeError("google-adk is not installed")
    adk_model = model or build_adk_model()
    kwargs = {
        "name": "anychain_benchmark_agent",
        "model": adk_model,
        "instruction": ADK_MODEL_BRIDGE_INSTRUCTION,
        "tools": tools if tools is not None else get_adk_tools(include_actions=False),
    }
    return Agent(**kwargs)


def build_adk_model(default: str = DEFAULT_MODEL):
    """Return the ADK model object for the configured provider.

    Google ADK owns model execution. Native Gemini/Vertex models can use a
    string model name. OpenAI-compatible third-party providers such as DeepSeek
    must go through ADK's LiteLLM integration.
    """
    config = load_llm_config()
    if config.provider == "deepseek":
        if LiteLlm is None:
            raise RuntimeError("DeepSeek requires ADK LiteLLM support; install requirements-adk.txt")
        return LiteLlm(model=f"deepseek/{config.model or 'deepseek-v4-flash'}")
    return config.model or default


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
