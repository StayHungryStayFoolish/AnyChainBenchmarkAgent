"""Internal LLM protocol used by the benchmark Agent."""

from __future__ import annotations

import contextvars
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, ParamSpec, Protocol, TypeVar


MessageRole = Literal["system", "user", "assistant", "tool"]
ReasoningMode = Literal["provider_default", "disabled"]
_P = ParamSpec("_P")
_R = TypeVar("_R")


class LLMProviderError(Exception):
    """A typed provider failure that must not be treated as user ambiguity."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        category: str = "provider",
        stage: str = "provider_request",
        status_code: int = 0,
        retriable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.category = category
        self.stage = stage
        self.status_code = status_code
        self.retriable = retriable


class LLMTurnTimeoutError(BaseException):
    """The shared turn deadline expired and must bypass transport retries."""

    def __init__(self, message: str, *, provider: str = "", model: str = "", stage: str = "turn") -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.stage = stage


class LLMTurnCancelledError(KeyboardInterrupt):
    """The user cancelled the active Agent turn."""

    def __init__(self, message: str, *, stage: str = "turn") -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class LLMTurnContext:
    deadline: float
    timeout_seconds: float
    cancelled: threading.Event = field(
        default_factory=threading.Event,
        compare=False,
        repr=False,
    )

    def remaining_seconds(self) -> float:
        if self.cancelled.is_set():
            raise LLMTurnCancelledError("active Agent turn was cancelled")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise LLMTurnTimeoutError(f"Agent turn exceeded its {self.timeout_seconds:g}s deadline")
        return remaining


_TURN_CONTEXT: contextvars.ContextVar[LLMTurnContext | None] = contextvars.ContextVar(
    "anychain_llm_turn_context",
    default=None,
)


@contextmanager
def llm_turn_scope(timeout_seconds: float) -> Iterator[LLMTurnContext]:
    """Share one monotonic deadline across every model call in a turn."""

    if timeout_seconds <= 0:
        raise ValueError("turn timeout must be greater than zero")
    existing = _TURN_CONTEXT.get()
    if existing is not None:
        yield existing
        return
    context = LLMTurnContext(
        deadline=time.monotonic() + timeout_seconds,
        timeout_seconds=timeout_seconds,
    )
    token = _TURN_CONTEXT.set(context)
    try:
        yield context
    finally:
        _TURN_CONTEXT.reset(token)


def remaining_turn_seconds(default: float | None = None) -> float:
    """Return the current turn budget, or a provider-local default."""

    context = _TURN_CONTEXT.get()
    if context is not None:
        return context.remaining_seconds()
    if default is None or default <= 0:
        raise ValueError("a positive default timeout is required outside a turn scope")
    return default


def ensure_turn_active() -> None:
    """Raise the typed timeout as soon as an exhausted turn reaches a boundary."""

    context = _TURN_CONTEXT.get()
    if context is not None:
        context.remaining_seconds()


def copy_llm_turn_context() -> contextvars.Context:
    """Capture the current turn context for one independently run worker."""

    return contextvars.copy_context()


def run_in_llm_turn_context(
    context: contextvars.Context,
    callback: Callable[_P, _R],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _R:
    """Run one worker with the captured absolute deadline and cancellation."""

    return context.run(callback, *args, **kwargs)


def cancel_active_llm_turn() -> None:
    """Cooperatively stop sibling work that shares the active turn context."""

    context = _TURN_CONTEXT.get()
    if context is not None:
        context.cancelled.set()


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
    reasoning_mode: ReasoningMode = "provider_default"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a model response for the internal Agent request shape."""
