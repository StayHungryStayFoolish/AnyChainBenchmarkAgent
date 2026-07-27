"""Internal LLM protocol used by the benchmark Agent."""

from __future__ import annotations

import contextvars
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Mapping, ParamSpec, Protocol, TypeVar


MessageRole = Literal["system", "user", "assistant", "tool"]
ReasoningMode = Literal["provider_default", "disabled"]
ResponseMode = Literal["text_required", "tool_or_text"]
ReplaySafety = Literal["no_retry", "side_effect_free"]
MAX_PROVIDER_ATTEMPT_RECORDS = 64
_PROVIDER_ATTEMPT_OUTCOMES = frozenset({
    "success",
    "provider_failure",
    "timeout",
    "cancelled",
})
_PROVIDER_FAILURE_CATEGORIES = frozenset({
    "authentication",
    "configuration",
    "not_found",
    "permission",
    "provider",
    "quota",
    "rate_limit",
    "request",
    "response",
    "service",
    "transport",
})
_PROVIDER_RETRY_REASONS = frozenset({
    "credential_transport_timeout",
    "normal_finish_empty_text",
    "rate_limit",
    "response",
    "service",
    "transport",
    "transport_timeout",
})
_PROVIDER_FINISH_REASONS = frozenset({
    "",
    "blocklist",
    "content_filter",
    "end_turn",
    "function_call",
    "image_safety",
    "language",
    "length",
    "malformed_function_call",
    "max_tokens",
    "missing",
    "model_context_window_exceeded",
    "other",
    "pause_turn",
    "prohibited_content",
    "recitation",
    "refusal",
    "safety",
    "spi",
    "stop",
    "stop_sequence",
    "tool_calls",
    "tool_use",
})
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
        attempt_count: int = 0,
        retry_reasons: tuple[str, ...] = (),
        retry_exhausted: bool = False,
        last_finish_reason: str = "",
        retry_reason: str = "",
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.category = category
        self.stage = stage
        self.status_code = status_code
        self.retriable = retriable
        self.attempt_count = attempt_count
        self.retry_reasons = retry_reasons
        self.retry_exhausted = retry_exhausted
        self.last_finish_reason = last_finish_reason
        self.retry_reason = retry_reason


class LLMTurnTimeoutError(BaseException):
    """The shared turn deadline expired and must bypass transport retries."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        stage: str = "turn",
        attempt_count: int = 0,
        retry_reasons: tuple[str, ...] = (),
        last_finish_reason: str = "",
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.stage = stage
        self.attempt_count = attempt_count
        self.retry_reasons = retry_reasons
        self.last_finish_reason = last_finish_reason


class LLMTurnCancelledError(KeyboardInterrupt):
    """The user cancelled the active Agent turn."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        stage: str = "turn",
        attempt_count: int = 0,
        retry_reasons: tuple[str, ...] = (),
        last_finish_reason: str = "",
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.stage = stage
        self.attempt_count = attempt_count
        self.retry_reasons = retry_reasons
        self.last_finish_reason = last_finish_reason


@dataclass(frozen=True)
class LLMTurnContext:
    deadline: float
    timeout_seconds: float
    cancelled: threading.Event = field(
        default_factory=threading.Event,
        compare=False,
        repr=False,
    )
    provider_attempts: list[dict[str, Any]] = field(
        default_factory=list,
        compare=False,
        repr=False,
    )
    provider_attempts_lock: threading.Lock = field(
        default_factory=threading.Lock,
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

    def record_provider_attempt(self, evidence: dict[str, Any]) -> None:
        canonical = validate_provider_attempt_record(evidence)
        with self.provider_attempts_lock:
            if len(self.provider_attempts) >= MAX_PROVIDER_ATTEMPT_RECORDS:
                raise RuntimeError(
                    "provider attempt evidence exceeded the per-turn limit"
                )
            self.provider_attempts.append(canonical)

    def provider_attempt_evidence(self) -> tuple[dict[str, Any], ...]:
        with self.provider_attempts_lock:
            return tuple(dict(item) for item in self.provider_attempts)


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


def record_provider_attempt(evidence: dict[str, Any]) -> None:
    context = _TURN_CONTEXT.get()
    if context is not None:
        context.record_provider_attempt(evidence)


def validate_provider_attempt_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and copy one secret-free provider attempt record."""

    required = {
        "provider",
        "model",
        "outcome",
        "attempt_count",
        "retry_reasons",
        "retry_exhausted",
        "last_finish_reason",
    }
    allowed = required | {"category"}
    if not required.issubset(record) or set(record) - allowed:
        raise ValueError("provider attempt evidence shape is invalid")
    provider = record.get("provider")
    model = record.get("model")
    outcome = record.get("outcome")
    category = record.get("category")
    attempt_count = record.get("attempt_count")
    retry_reasons = record.get("retry_reasons")
    retry_exhausted = record.get("retry_exhausted")
    finish_reason = record.get("last_finish_reason")
    if (
        not isinstance(provider, str)
        or not provider
        or len(provider) > 128
        or not isinstance(model, str)
        or not model
        or len(model) > 256
        or outcome not in _PROVIDER_ATTEMPT_OUTCOMES
        or not isinstance(attempt_count, int)
        or isinstance(attempt_count, bool)
        or attempt_count < 0
        or (
            attempt_count == 0
            and outcome not in {"timeout", "cancelled"}
        )
        or not isinstance(retry_reasons, list)
        or any(item not in _PROVIDER_RETRY_REASONS for item in retry_reasons)
        or (
            outcome in {"success", "provider_failure"}
            and len(retry_reasons) >= attempt_count
        )
        or (
            outcome in {"timeout", "cancelled"}
            and len(retry_reasons) > attempt_count
        )
        or not isinstance(retry_exhausted, bool)
        or not isinstance(finish_reason, str)
        or finish_reason not in _PROVIDER_FINISH_REASONS
        or (
            outcome == "provider_failure"
            and category not in _PROVIDER_FAILURE_CATEGORIES
        )
        or (
            outcome != "provider_failure"
            and category is not None
        )
        or (retry_exhausted and outcome != "provider_failure")
    ):
        raise ValueError("provider attempt evidence semantics are invalid")
    canonical = {
        "provider": provider,
        "model": model,
        "outcome": outcome,
        "attempt_count": attempt_count,
        "retry_reasons": list(retry_reasons),
        "retry_exhausted": retry_exhausted,
        "last_finish_reason": finish_reason,
    }
    if category is not None:
        canonical["category"] = category
    return canonical


def provider_attempt_evidence() -> tuple[dict[str, Any], ...]:
    context = _TURN_CONTEXT.get()
    return context.provider_attempt_evidence() if context is not None else ()


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
    response_mode: ResponseMode = "text_required"
    replay_safety: ReplaySafety = "no_retry"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)
    attempt_count: int = 1
    retry_reasons: tuple[str, ...] = ()
    retry_exhausted: bool = False
    last_finish_reason: str = ""


class LLMProvider(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a model response for the internal Agent request shape."""
