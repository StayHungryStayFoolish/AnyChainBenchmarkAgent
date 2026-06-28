"""Optional ADK web research tools.

ADK google_search is only exposed for Gemini-family model configurations. This
module does not classify user intent; it only decides whether the official ADK
tool is available for the current provider and auth configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm.config import LLMConfig, load_llm_config


@dataclass(frozen=True)
class WebResearchStatus:
    enabled: bool
    provider: str
    model: str
    mode: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "reason": self.reason,
        }


def web_research_status(config: LLMConfig | None = None) -> WebResearchStatus:
    """Return the user-visible web research capability state."""
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
    if not status.enabled:
        return []
    from google.adk.tools import google_search  # type: ignore

    return [google_search]
