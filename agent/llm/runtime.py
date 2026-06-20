"""Runtime LLM mode detection for the Agent entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm.config import LLMConfig, load_llm_config
from llm.providers import provider_from_config


@dataclass(frozen=True)
class LLMRuntime:
    enabled: bool
    mode: str
    provider_name: str
    model: str
    auth_mode: str
    provider: Any | None = None
    reason: str = ""
    validation_errors: tuple[str, ...] = ()

    def banner(self) -> str:
        if self.enabled:
            return (
                f"Mode: AI-assisted ({self.provider_name}, model={self.model}, "
                f"auth={self.auth_mode})"
            )
        detail = f" Reason: {self.reason}" if self.reason else ""
        return f"Mode: deterministic/offline.{detail}"


def detect_llm_runtime(mock_provider: Any | None = None) -> LLMRuntime:
    """Return the runtime mode without performing a model call."""
    if mock_provider is not None:
        config = getattr(mock_provider, "config", LLMConfig(provider="fake", model="fake"))
        return LLMRuntime(
            enabled=True,
            mode="mock",
            provider_name="fake",
            model=config.model or "fake",
            auth_mode=config.auth_mode,
            provider=mock_provider,
            reason="offline fake provider selected",
        )

    config = load_llm_config()
    errors = tuple(config.validate())
    if config.provider == "fake":
        return LLMRuntime(
            enabled=False,
            mode="deterministic",
            provider_name=config.provider,
            model=config.model,
            auth_mode=config.auth_mode,
            reason="LLM_PROVIDER=fake",
            validation_errors=errors,
        )
    if errors:
        return LLMRuntime(
            enabled=False,
            mode="deterministic",
            provider_name=config.provider,
            model=config.model,
            auth_mode=config.auth_mode,
            reason="LLM configuration is incomplete",
            validation_errors=errors,
        )
    return LLMRuntime(
        enabled=True,
        mode="configured",
        provider_name=config.provider,
        model=config.model,
        auth_mode=config.auth_mode,
        provider=provider_from_config(config),
    )
