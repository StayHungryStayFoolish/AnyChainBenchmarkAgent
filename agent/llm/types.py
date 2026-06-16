"""Internal LLM protocol used by the benchmark Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class LLMMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True)
class LLMRequest:
    messages: list[LLMMessage]
    temperature: float = 0.2
    max_tokens: int = 4096
    tools: list[dict[str, Any]] = field(default_factory=list)
    response_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a model response for the internal Agent request shape."""
