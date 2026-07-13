"""Gemini `google_search` grounding — the one remaining use of `google-adk`.

Every other conversational/tool-calling responsibility in this codebase
belongs to the LangGraph Harness (`agent/harness/*`), which never uses
Google ADK's `Agent`/`Runner` loop. This module is the sole, narrowly-scoped
exception: real web search grounding is only possible through ADK's
`google_search` tool, which only functions inside an ADK `Agent`/`Runner`.

The harness calls `run_google_search_grounding` as a plain, synchronous
function — the same pattern `agent/validators/endpoint_probe.py` uses for
another side-effecting external call. This does not create a second
conversational loop: it runs exactly one grounded query and returns a typed
result, the same way `endpoint_probe.validate_rpc_endpoint` runs exactly one
HTTP probe and returns a typed result.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

try:
    from .config import LLMConfig, load_llm_config
except ImportError:  # script execution with agent/ on sys.path
    from llm.config import LLMConfig, load_llm_config

DEFAULT_GROUNDING_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class WebResearchStatus:
    google_search_available: bool
    provider: str
    model: str
    mode: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "google_search_available": self.google_search_available,
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "reason": self.reason,
        }


def web_research_status(config: LLMConfig | None = None) -> WebResearchStatus:
    """Return the user-visible web research capability state.

    The returned `google_search_available` field name is the single source of
    truth other modules must read (`agent/harness/groups.py`'s real-node
    client-setup flow, `agent/terminal/repl.py`'s startup banner). It must
    never be renamed on only one side of that contract again.
    """
    cfg = config or load_llm_config()
    eligible, reason = cfg.google_search_eligible()
    if not eligible:
        return WebResearchStatus(False, cfg.provider, cfg.model, "disabled", reason)
    try:
        from google.adk.tools import google_search  # type: ignore  # noqa: F401
    except Exception as exc:
        return WebResearchStatus(False, cfg.provider, cfg.model, "disabled", f"ADK google_search unavailable: {type(exc).__name__}")
    return WebResearchStatus(True, cfg.provider, cfg.model, "adk_google_search", "enabled via ADK google_search")


def get_google_search_tools(config: LLMConfig | None = None) -> list:
    """Return ADK google_search only when the current model can use it."""
    status = web_research_status(config)
    if not status.google_search_available:
        return []
    from google.adk.tools import google_search  # type: ignore

    return [google_search]


@dataclass(frozen=True)
class SearchGroundingResult:
    available: bool
    query: str
    text_summary: str = ""
    citations: list[str] = field(default_factory=list)
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "query": self.query,
            "text_summary": self.text_summary,
            "citations": list(self.citations),
            "error": self.error,
        }


def run_google_search_grounding(
    query: str,
    config: LLMConfig | None = None,
    timeout: float = DEFAULT_GROUNDING_TIMEOUT_SECONDS,
) -> SearchGroundingResult:
    """Execute one Gemini `google_search` grounding query, or report unavailable.

    Deliberately not `agent/adk_app/root_agent.py`'s 40-tool builder: this
    constructs a throwaway, single-purpose Agent scoped to exactly the one
    capability ADK is needed for, runs exactly one turn, and returns. Any
    failure (missing package, auth failure, timeout, runtime error) degrades
    to a typed `available=False` result rather than raising, matching how
    `web_research_status` already degrades.

    Always reloads/re-validates the config live via `web_research_status`
    rather than trusting a caller-supplied `google_search_available=True`
    flag on faith. This is intentional, not an oversight: the harness's
    cached copy of that flag (`state["web_research"]`) is part of the
    LangGraph checkpoint written to disk, and `LLMConfig` carries API keys —
    that config must never be threaded through checkpointed state, so this
    function re-derives it from the live environment on every call instead.
    """
    cfg = config or load_llm_config()
    status = web_research_status(cfg)
    if not status.google_search_available:
        return SearchGroundingResult(available=False, query=query, error=status.reason)

    try:
        from google.adk.agents import Agent  # noqa: F401
        from google.adk.runners import InMemoryRunner  # noqa: F401
        from google.genai import types  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency guard
        return SearchGroundingResult(available=False, query=query, error=f"ADK import failed: {type(exc).__name__}: {exc}")

    try:
        return asyncio.run(asyncio.wait_for(_run_grounding_turn(cfg, query), timeout=timeout))
    except asyncio.TimeoutError:
        return SearchGroundingResult(available=False, query=query, error=f"search grounding timed out after {timeout}s")
    except Exception as exc:  # pragma: no cover - defensive: never let a search failure break the harness turn
        return SearchGroundingResult(available=False, query=query, error=f"search grounding failed: {type(exc).__name__}: {exc}")


async def _run_grounding_turn(cfg: LLMConfig, query: str) -> "SearchGroundingResult":
    from google.adk.agents import Agent
    from google.adk.runners import InMemoryRunner
    from google.adk.tools import google_search
    from google.genai import types

    agent = Agent(
        name="anychain_search_grounding",
        model=cfg.model,
        instruction=(
            "Answer with verified, current facts only. Prefer official documentation, "
            "official Docker images, and official release pages. Keep the answer concise."
        ),
        tools=[google_search],
    )
    runner = InMemoryRunner(agent=agent, app_name="anychain_search_grounding")
    session = await runner.session_service.create_session(app_name="anychain_search_grounding", user_id="harness")
    text_parts: list[str] = []
    citations: list[str] = []
    async for event in runner.run_async(
        user_id="harness",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
    ):
        content = getattr(event, "content", None)
        if content and getattr(content, "parts", None):
            for part in content.parts:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
        grounding = getattr(event, "grounding_metadata", None)
        if grounding and getattr(grounding, "grounding_chunks", None):
            for chunk in grounding.grounding_chunks:
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", "") if web else ""
                if uri:
                    citations.append(uri)
    return SearchGroundingResult(
        available=True,
        query=query,
        text_summary="\n".join(text_parts).strip(),
        citations=citations,
    )
