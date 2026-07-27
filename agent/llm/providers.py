"""LLM provider adapters for OpenAI, Gemini, Anthropic, and Vertex AI."""

from __future__ import annotations

import importlib.util
import json
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from .config import LLMConfig, load_llm_config
from .google_auth import get_google_access_token
from .types import (
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTurnCancelledError,
    LLMTurnTimeoutError,
    ensure_turn_active,
    llm_turn_scope,
    record_provider_attempt,
    remaining_turn_seconds,
)


@dataclass(frozen=True)
class _CompletionResult:
    response: Any
    text: str
    attempt_count: int
    retry_reasons: tuple[str, ...]
    last_finish_reason: str


class _ProviderClientCloseError(RuntimeError):
    """A client lifecycle failure after a completed provider request."""


class OpenAIProvider:
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("openai is required for LLM_PROVIDER=openai") from exc

        result = _execute_completion(
            self.config,
            request,
            lambda: _openai_request(
                self.config,
                lambda: _openai_completion_call(
                    lambda: _openai_client(
                        OpenAI,
                        self.config,
                        api_key=self.config.openai_api_key or None,
                    ),
                    lambda client: client.chat.completions.create(
                        model=self.config.model,
                        messages=_openai_messages(request.messages),
                        **_openai_completion_options(self.config.model, request),
                    ),
                ),
            ),
            lambda response: _parse_openai_completion(
                self.config,
                request,
                response,
            ),
        )
        return LLMResponse(
            text=result.text,
            model=self.config.model,
            provider=self.config.provider,
            raw=(
                result.response.model_dump()
                if hasattr(result.response, "model_dump")
                else {}
            ),
            attempt_count=result.attempt_count,
            retry_reasons=result.retry_reasons,
            last_finish_reason=result.last_finish_reason,
        )


class DeepSeekProvider:
    """DeepSeek provider through its OpenAI-compatible chat endpoint."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("openai is required for LLM_PROVIDER=deepseek") from exc

        result = _execute_completion(
            self.config,
            request,
            lambda: _openai_request(
                self.config,
                lambda: _openai_completion_call(
                    lambda: _openai_client(
                        OpenAI,
                        self.config,
                        api_key=self.config.deepseek_api_key or None,
                        base_url="https://api.deepseek.com",
                    ),
                    lambda client: client.chat.completions.create(
                        model=self.config.model,
                        messages=_openai_messages(request.messages),
                        **_deepseek_completion_options(request),
                    ),
                ),
            ),
            lambda response: _parse_openai_completion(
                self.config,
                request,
                response,
            ),
        )
        return LLMResponse(
            text=result.text,
            model=self.config.model,
            provider=self.config.provider,
            raw=(
                result.response.model_dump()
                if hasattr(result.response, "model_dump")
                else {}
            ),
            attempt_count=result.attempt_count,
            retry_reasons=result.retry_reasons,
            last_finish_reason=result.last_finish_reason,
        )


class VertexGeminiProvider:
    """Gemini on Vertex through the OpenAI-compatible endpoint."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("openai is required for Gemini on Vertex OpenAI-compatible calls") from exc

        with llm_turn_scope(self.config.turn_timeout_seconds):
            token = _vertex_access_token(self.config)
            base_url = _vertex_openai_base_url(self.config)
            result = _execute_completion(
                self.config,
                request,
                lambda: _openai_request(
                    self.config,
                    lambda: _openai_completion_call(
                        lambda: _openai_client(
                            OpenAI,
                            self.config,
                            api_key=token,
                            base_url=base_url,
                        ),
                        lambda client: client.chat.completions.create(
                            model=f"google/{self.config.model}",
                            messages=_openai_messages(request.messages),
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                            tools=request.tools or None,
                        ),
                    ),
                ),
                lambda response: _parse_openai_completion(
                    self.config,
                    request,
                    response,
                ),
            )
        return LLMResponse(
            text=result.text,
            model=self.config.model,
            provider=self.config.provider,
            raw=(
                result.response.model_dump()
                if hasattr(result.response, "model_dump")
                else {}
            ),
            attempt_count=result.attempt_count,
            retry_reasons=result.retry_reasons,
            last_finish_reason=result.last_finish_reason,
        )


def _vertex_aiplatform_host(location: str) -> str:
    """Return the Vertex AI API host for a location.

    Regional endpoints use `<region>-aiplatform.googleapis.com`, but the
    special `global` location uses `aiplatform.googleapis.com`.
    """
    normalized = location.strip().lower()
    return "aiplatform.googleapis.com" if normalized == "global" else f"{normalized}-aiplatform.googleapis.com"


def _vertex_openai_base_url(config: LLMConfig) -> str:
    location = config.google_location.strip()
    return (
        f"https://{_vertex_aiplatform_host(location)}/v1/"
        f"projects/{config.google_project}/locations/{location}/endpoints/openapi"
    )


def _vertex_raw_predict_url(config: LLMConfig, publisher: str, model: str) -> str:
    location = config.google_location.strip()
    return (
        f"https://{_vertex_aiplatform_host(location)}/v1/"
        f"projects/{config.google_project}/locations/{location}/"
        f"publishers/{publisher}/models/{model}:rawPredict"
    )


class GeminiAPIKeyProvider:
    """Gemini API provider using a direct API key."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        api_key = self.config.gemini_api_key
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini API-key mode")
        system, contents = _gemini_contents(request.messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urlparse.quote(self.config.model, safe='')}:generateContent?key={urlparse.quote(api_key, safe='')}"
        )
        result = _execute_completion(
            self.config,
            request,
            lambda: _post_json(url, payload, headers={}, config=self.config),
            lambda response: _parse_gemini_completion(
                self.config,
                request,
                response,
            ),
        )
        return _llm_response(self.config, result)


class VertexClaudeProvider:
    """`claude` partner models on Vertex AI."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        system, messages = _anthropic_messages(request.messages)
        payload: dict[str, Any] = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if system:
            payload["system"] = system
        if request.tools:
            payload["tools"] = request.tools
        with llm_turn_scope(self.config.turn_timeout_seconds):
            token = _vertex_access_token(self.config)
            url = _vertex_raw_predict_url(
                self.config,
                "anthropic",
                self.config.model,
            )
            result = _execute_completion(
                self.config,
                request,
                lambda: _post_json(
                    url,
                    payload,
                    headers={"Authorization": f"Bearer {token}"},
                    config=self.config,
                ),
                lambda response: _parse_anthropic_completion(
                    self.config,
                    request,
                    response,
                ),
            )
        return _llm_response(self.config, result)


class AnthropicAPIKeyProvider:
    """`claude` provider using the direct Anthropic API."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        api_key = self.config.anthropic_api_key
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for `claude` API-key mode")
        system, messages = _anthropic_messages(request.messages)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if system:
            payload["system"] = system
        if request.tools:
            payload["tools"] = request.tools
        result = _execute_completion(
            self.config,
            request,
            lambda: _post_json(
                "https://api.anthropic.com/v1/messages",
                payload,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                config=self.config,
            ),
            lambda response: _parse_anthropic_completion(
                self.config,
                request,
                response,
            ),
        )
        return _llm_response(self.config, result)


def _openai_client(client_type: Any, config: LLMConfig, **kwargs: Any) -> Any:
    import httpx

    remaining = remaining_turn_seconds(config.turn_timeout_seconds)
    connect_timeout = min(config.connect_timeout_seconds, remaining)
    read_timeout = min(config.read_timeout_seconds, remaining)
    timeout = httpx.Timeout(
        timeout=remaining,
        connect=connect_timeout,
        read=read_timeout,
        write=read_timeout,
        pool=connect_timeout,
    )
    return client_type(timeout=timeout, max_retries=0, **kwargs)


def _openai_completion_call(client_factory: Any, create: Any) -> Any:
    client = client_factory()
    try:
        result = create(client)
    except BaseException as primary:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as close_error:
                add_note = getattr(primary, "add_note", None)
                if callable(add_note):
                    add_note(
                        "provider client close also failed with "
                        f"{type(close_error).__name__}"
                    )
        raise
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except (LLMTurnCancelledError, LLMTurnTimeoutError):
            raise
        except Exception as close_error:
            raise _ProviderClientCloseError(
                "provider client close failed after a completed request"
            ) from close_error
    return result


def _vertex_access_token(config: LLMConfig) -> str:
    try:
        ensure_turn_active()
        token = get_google_access_token(config)
        ensure_turn_active()
    except (LLMTurnTimeoutError, LLMTurnCancelledError) as exc:
        _attach_terminal_attempt_evidence(
            exc,
            config=config,
            attempt_count=1,
            retry_reasons=(),
            last_finish_reason="",
        )
        _record_terminal_outcome(config, exc)
        raise
    except Exception as exc:
        if _is_transport_timeout(exc):
            try:
                ensure_turn_active()
            except (LLMTurnTimeoutError, LLMTurnCancelledError) as terminal:
                _attach_terminal_attempt_evidence(
                    terminal,
                    config=config,
                    attempt_count=1,
                    retry_reasons=(),
                    last_finish_reason="",
                )
                _record_terminal_outcome(config, terminal)
                raise
            error = _transport_error(
                config,
                "credential_transport_timeout",
            )
        else:
            error = _provider_error(config, exc)
        terminal = _provider_error_with_attempt_evidence(
            error,
            attempt_count=1,
            retry_reasons=(),
            retry_exhausted=False,
        )
        _record_provider_failure(config, terminal)
        raise terminal from exc
    return token


def _openai_request(config: LLMConfig, call: Any) -> Any:
    ensure_turn_active()
    try:
        response = call()
    except Exception as exc:
        if _is_transport_timeout(exc):
            ensure_turn_active()
            raise _transport_error(config, "transport_timeout") from exc
        raise _provider_error(config, exc) from exc
    ensure_turn_active()
    return response


def _execute_completion(
    config: LLMConfig,
    request: LLMRequest,
    send_once: Any,
    parse_response: Any,
) -> _CompletionResult:
    """Own the total attempt budget for one read-only completion."""

    with llm_turn_scope(config.turn_timeout_seconds):
        return _execute_completion_in_scope(
            config,
            request,
            send_once,
            parse_response,
        )


def _execute_completion_in_scope(
    config: LLMConfig,
    request: LLMRequest,
    send_once: Any,
    parse_response: Any,
) -> _CompletionResult:
    retry_reasons: list[str] = []
    attempt_count = 0
    last_finish_reason = ""
    max_attempts = (
        config.max_retries + 1
        if request.replay_safety == "side_effect_free"
        else 1
    )
    for _attempt_index in range(max_attempts):
        try:
            ensure_turn_active()
            attempt_count += 1
            response = send_once()
            text, finish_reason = parse_response(response)
            result = _CompletionResult(
                response=response,
                text=text,
                attempt_count=attempt_count,
                retry_reasons=tuple(retry_reasons),
                last_finish_reason=finish_reason,
            )
            _record_completion_outcome(
                config,
                result,
                outcome="success",
            )
            return result
        except (LLMTurnTimeoutError, LLMTurnCancelledError) as exc:
            _attach_terminal_attempt_evidence(
                exc,
                config=config,
                attempt_count=attempt_count,
                retry_reasons=tuple(retry_reasons),
                last_finish_reason=last_finish_reason,
            )
            _record_terminal_outcome(config, exc)
            raise
        except LLMProviderError as exc:
            last_finish_reason = exc.last_finish_reason
            reason = exc.retry_reason or exc.category
            can_retry = exc.retriable and attempt_count < max_attempts
            if not can_retry:
                terminal = _provider_error_with_attempt_evidence(
                    exc,
                    attempt_count=attempt_count,
                    retry_reasons=tuple(retry_reasons),
                    retry_exhausted=(
                        exc.retriable
                        and request.replay_safety == "side_effect_free"
                        and attempt_count >= max_attempts
                    ),
                )
                _record_provider_failure(config, terminal)
                raise terminal from exc
            retry_reasons.append(reason)
            try:
                _wait_for_retry(config, attempt_count)
                ensure_turn_active()
            except (LLMTurnTimeoutError, LLMTurnCancelledError) as terminal:
                _attach_terminal_attempt_evidence(
                    terminal,
                    config=config,
                    attempt_count=attempt_count,
                    retry_reasons=tuple(retry_reasons),
                    last_finish_reason=last_finish_reason,
                )
                _record_terminal_outcome(config, terminal)
                raise
    raise AssertionError("completion retry loop did not terminate")


def _attach_terminal_attempt_evidence(
    error: LLMTurnTimeoutError | LLMTurnCancelledError,
    *,
    config: LLMConfig,
    attempt_count: int,
    retry_reasons: tuple[str, ...],
    last_finish_reason: str,
) -> None:
    error.provider = error.provider or config.provider
    error.model = error.model or config.model
    error.attempt_count = attempt_count
    error.retry_reasons = retry_reasons
    error.last_finish_reason = last_finish_reason


def _record_completion_outcome(
    config: LLMConfig,
    result: _CompletionResult,
    *,
    outcome: str,
) -> None:
    record_provider_attempt({
        "provider": config.provider,
        "model": config.model,
        "outcome": outcome,
        "attempt_count": result.attempt_count,
        "retry_reasons": list(result.retry_reasons),
        "retry_exhausted": False,
        "last_finish_reason": result.last_finish_reason,
    })


def _record_provider_failure(
    config: LLMConfig,
    error: LLMProviderError,
) -> None:
    record_provider_attempt({
        "provider": config.provider,
        "model": config.model,
        "outcome": "provider_failure",
        "category": error.category,
        "attempt_count": error.attempt_count,
        "retry_reasons": list(error.retry_reasons),
        "retry_exhausted": error.retry_exhausted,
        "last_finish_reason": error.last_finish_reason,
    })


def _record_terminal_outcome(
    config: LLMConfig,
    error: LLMTurnTimeoutError | LLMTurnCancelledError,
) -> None:
    record_provider_attempt({
        "provider": config.provider,
        "model": config.model,
        "outcome": (
            "cancelled"
            if isinstance(error, LLMTurnCancelledError)
            else "timeout"
        ),
        "attempt_count": error.attempt_count,
        "retry_reasons": list(error.retry_reasons),
        "retry_exhausted": False,
        "last_finish_reason": error.last_finish_reason,
    })


def _wait_for_retry(config: LLMConfig, attempt_count: int) -> None:
    remaining = remaining_turn_seconds(config.turn_timeout_seconds)
    delay = min(0.25 * (2 ** (attempt_count - 1)), 1.0, remaining)
    if delay > 0:
        time.sleep(delay)


def _provider_error_with_attempt_evidence(
    error: LLMProviderError,
    *,
    attempt_count: int,
    retry_reasons: tuple[str, ...],
    retry_exhausted: bool,
) -> LLMProviderError:
    return LLMProviderError(
        str(error),
        provider=error.provider,
        model=error.model,
        category=error.category,
        stage=error.stage,
        status_code=error.status_code,
        retriable=error.retriable,
        attempt_count=attempt_count,
        retry_reasons=retry_reasons,
        retry_exhausted=retry_exhausted,
        last_finish_reason=error.last_finish_reason,
        retry_reason=error.retry_reason,
    )


def _response_error(
    config: LLMConfig,
    detail: str,
    *,
    retriable: bool = False,
    retry_reason: str = "",
    finish_reason: str = "",
) -> LLMProviderError:
    return LLMProviderError(
        f"{config.provider}/{config.model} returned a malformed successful response: {detail}",
        provider=config.provider,
        model=config.model,
        category="response",
        stage="provider_response",
        retriable=retriable,
        retry_reason=retry_reason,
        last_finish_reason=finish_reason,
    )


def _parse_openai_completion(
    config: LLMConfig,
    request: LLMRequest,
    response: Any,
) -> tuple[str, str]:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        raise _response_error(config, "choices is missing or empty")
    choice = choices[0]
    finish_reason = _normalized_finish_reason(
        getattr(choice, "finish_reason", ""),
        {
            "stop",
            "length",
            "tool_calls",
            "function_call",
            "content_filter",
        },
    )
    message = getattr(choice, "message", None)
    if message is None:
        raise _response_error(config, "choices[0].message is missing")
    content = getattr(message, "content", None)
    tool_calls = getattr(message, "tool_calls", None)
    refusal = str(getattr(message, "refusal", "") or "").strip()
    _require_text_completion_disposition(
        config,
        request,
        finish_reason=finish_reason,
        normal_finish_reasons={"stop"},
        has_alternative_output=bool(tool_calls),
        has_refusal=bool(refusal),
    )
    if not isinstance(content, str) or not content.strip():
        retriable = _empty_text_retryable(
            request,
            finish_reason=finish_reason,
            normal_finish_reasons={"stop"},
            has_alternative_output=bool(tool_calls) or bool(refusal),
        )
        raise _response_error(
            config,
            "choices[0].message.content is missing or empty",
            retriable=retriable,
            retry_reason="normal_finish_empty_text" if retriable else "",
            finish_reason=finish_reason,
        )
    return content, finish_reason


def _normalized_finish_reason(
    value: Any,
    allowed: set[str],
) -> str:
    normalized = str(value or "").strip().casefold()
    if not normalized:
        return "missing"
    return normalized if normalized in allowed else "other"


def _require_text_completion_disposition(
    config: LLMConfig,
    request: LLMRequest,
    *,
    finish_reason: str,
    normal_finish_reasons: set[str],
    has_alternative_output: bool,
    has_refusal: bool = False,
) -> None:
    if has_refusal:
        raise _response_error(
            config,
            "provider returned a refusal",
            finish_reason=finish_reason,
        )
    if has_alternative_output:
        raise _response_error(
            config,
            "provider returned structured output that this text boundary cannot project",
            finish_reason=finish_reason,
        )
    if finish_reason not in normal_finish_reasons:
        raise _response_error(
            config,
            f"completion ended with {finish_reason}",
            finish_reason=finish_reason,
        )


def _empty_text_retryable(
    request: LLMRequest,
    *,
    finish_reason: str,
    normal_finish_reasons: set[str],
    has_alternative_output: bool,
) -> bool:
    return (
        request.response_mode == "text_required"
        and request.replay_safety == "side_effect_free"
        and not request.tools
        and not has_alternative_output
        and finish_reason.casefold() in normal_finish_reasons
    )


def _llm_response(
    config: LLMConfig,
    result: _CompletionResult,
) -> LLMResponse:
    return LLMResponse(
        text=result.text,
        model=config.model,
        provider=config.provider,
        raw=dict(result.response),
        attempt_count=result.attempt_count,
        retry_reasons=result.retry_reasons,
        last_finish_reason=result.last_finish_reason,
    )


def _is_transport_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    name = type(exc).__name__.casefold()
    module = type(exc).__module__.casefold()
    return "timeout" in name and any(marker in module for marker in ("openai", "httpx", "httpcore", "urllib"))


def _is_transport_connection(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, urlerror.URLError)):
        return True
    name = type(exc).__name__.casefold()
    module = type(exc).__module__.casefold()
    return (
        "connection" in name
        and any(marker in module for marker in ("openai", "httpx", "httpcore"))
    )


def _transport_error(config: LLMConfig, reason: str) -> LLMProviderError:
    return LLMProviderError(
        f"{config.provider}/{config.model} provider transport failed",
        provider=config.provider,
        model=config.model,
        category="transport",
        stage="provider_request",
        retriable=True,
        retry_reason=reason,
    )


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], config: LLMConfig) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        **headers,
    }
    req = urlrequest.Request(
        url,
        data=data,
        headers=request_headers,
        method="POST",
    )
    timeout = min(config.read_timeout_seconds, remaining_turn_seconds(config.turn_timeout_seconds))
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:  # nosec B310 - URL is a configured provider endpoint
            payload = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, socket.timeout) as exc:
        ensure_turn_active()
        raise _transport_error(config, "transport_timeout") from exc
    except urlerror.HTTPError as exc:
        raise _provider_error(config, exc) from exc
    except urlerror.URLError as exc:
        if _is_transport_timeout(exc.reason if isinstance(exc.reason, BaseException) else exc):
            ensure_turn_active()
            raise _transport_error(config, "transport_timeout") from exc
        raise _provider_error(config, exc) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _response_error(config, "body is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise _response_error(config, "body is not a JSON object")
    ensure_turn_active()
    return payload


def provider_from_config(config: LLMConfig | None = None) -> LLMProvider:
    config = config or load_llm_config()
    errors = config.validate()
    if errors:
        raise ValueError("; ".join(errors))
    if config.provider == "openai":
        return OpenAIProvider(config)
    if config.provider == "deepseek":
        return DeepSeekProvider(config)
    if config.provider == "gemini":
        return GeminiAPIKeyProvider(config) if config.auth_mode == "api_key" else VertexGeminiProvider(config)
    if config.provider == "claude":
        return AnthropicAPIKeyProvider(config) if config.auth_mode == "api_key" else VertexClaudeProvider(config)
    raise ValueError(f"unsupported LLM_PROVIDER: {config.provider}")


def provider_runtime_errors(config: LLMConfig | None = None) -> list[str]:
    """Return configuration and import blockers for the selected provider.

    Google ADK is deliberately absent from this contract. It is an optional
    Gemini search-grounding bridge, not the runtime for ordinary Harness turns.
    """

    config = config or load_llm_config()
    errors = list(config.validate())
    if config.provider in {"openai", "deepseek"} or (
        config.provider == "gemini" and config.auth_mode != "api_key"
    ):
        if importlib.util.find_spec("openai") is None:
            errors.append("openai package is required for the selected provider")
    if config.provider in {"gemini", "claude"} and config.auth_mode != "api_key":
        try:
            google_auth_available = importlib.util.find_spec("google.auth") is not None
        except ModuleNotFoundError:
            google_auth_available = False
        if not google_auth_available:
            errors.append("google-auth package is required for Vertex authentication")
    return errors


def probe_provider_readiness(
    config: LLMConfig | None = None,
    *,
    timeout_seconds: float = 20.0,
) -> LLMProviderError | None:
    """Verify that the configured provider and model can execute a minimal call."""

    config = config or load_llm_config()
    runtime_errors = provider_runtime_errors(config)
    if runtime_errors:
        return LLMProviderError(
            "; ".join(runtime_errors),
            provider=config.provider,
            model=config.model,
            category="configuration",
            stage="provider_readiness",
        )
    try:
        with llm_turn_scope(min(config.turn_timeout_seconds, timeout_seconds)):
            response = provider_from_config(config).complete(LLMRequest(
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "Return exactly one JSON object with the single field "
                            '`"ready": true`. Do not use Markdown.'
                        ),
                    ),
                    LLMMessage(role="user", content='{"probe":"provider_readiness"}'),
                ],
                temperature=0.0,
                max_tokens=256,
                replay_safety="side_effect_free",
            ))
    except LLMTurnTimeoutError:
        return LLMProviderError(
            "provider readiness probe timed out",
            provider=config.provider,
            model=config.model,
            category="timeout",
            stage="provider_readiness",
            retriable=True,
        )
    except LLMProviderError as exc:
        return LLMProviderError(
            str(exc),
            provider=exc.provider or config.provider,
            model=exc.model or config.model,
            category=exc.category,
            stage="provider_readiness",
            status_code=exc.status_code,
            retriable=exc.retriable,
            attempt_count=exc.attempt_count,
            retry_reasons=exc.retry_reasons,
            retry_exhausted=exc.retry_exhausted,
            last_finish_reason=exc.last_finish_reason,
            retry_reason=exc.retry_reason,
        )
    except Exception as exc:
        return LLMProviderError(
            f"provider readiness failed: {type(exc).__name__}",
            provider=config.provider,
            model=config.model,
            category="provider",
            stage="provider_readiness",
        )
    try:
        readiness = json.loads(str(response.text or "").strip())
    except json.JSONDecodeError:
        readiness = None
    if readiness != {"ready": True}:
        return LLMProviderError(
            "provider readiness did not satisfy the strict response contract",
            provider=config.provider,
            model=config.model,
            category="response",
            stage="provider_readiness",
        )
    return None


def _provider_error(config: LLMConfig, exc: BaseException) -> LLMProviderError:
    status_code = int(
        getattr(exc, "status_code", 0)
        or getattr(exc, "code", 0)
        or 0
    )
    if status_code in {401, 403}:
        category = "authentication"
    elif status_code == 402:
        category = "quota"
    elif status_code == 429:
        category = "rate_limit"
    elif status_code == 400:
        category = "configuration"
    elif status_code >= 500:
        category = "service"
    elif _is_transport_connection(exc):
        category = "transport"
    else:
        category = "provider"
    return LLMProviderError(
        f"{config.provider}/{config.model} provider request failed"
        + (f" with HTTP {status_code}" if status_code else ""),
        provider=config.provider,
        model=config.model,
        category=category,
        status_code=status_code,
        retriable=category in {"rate_limit", "service", "transport"},
        retry_reason=category,
    )


def _openai_completion_options(model: str, request: LLMRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "temperature": request.temperature,
        "tools": request.tools or None,
    }
    if model.startswith("gpt-5"):
        payload["max_completion_tokens"] = request.max_tokens
    else:
        payload["max_tokens"] = request.max_tokens
    return payload


def _deepseek_completion_options(request: LLMRequest) -> dict[str, Any]:
    """Map the common inference contract to DeepSeek's official API."""

    payload: dict[str, Any] = {
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "tools": request.tools or None,
    }
    if request.reasoning_mode == "disabled":
        payload["extra_body"] = {"thinking": {"type": "disabled"}}
    return payload


def _openai_messages(messages: list[LLMMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _anthropic_messages(messages: list[LLMMessage]) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    converted: list[dict[str, str]] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
        elif message.role in {"user", "assistant"}:
            converted.append({"role": message.role, "content": message.content})
        elif message.role == "tool":
            converted.append({"role": "user", "content": message.content})
    return "\n\n".join(system_parts), converted


def _gemini_contents(messages: list[LLMMessage]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
            continue
        role = "model" if message.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message.content}]})
    return "\n\n".join(system_parts), contents


def _gemini_text(config: LLMConfig, response: dict[str, Any]) -> str:
    text, _finish_reason = _parse_gemini_completion(
        config,
        LLMRequest(messages=[]),
        response,
    )
    return text


def _parse_gemini_completion(
    config: LLMConfig,
    request: LLMRequest,
    response: dict[str, Any],
) -> tuple[str, str]:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise _response_error(config, "candidates is missing or empty")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise _response_error(config, "candidates[0] is not an object")
    content = candidate.get("content")
    if not isinstance(content, dict):
        raise _response_error(config, "candidates[0].content is missing")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise _response_error(config, "candidates[0].content.parts is missing or empty")
    finish_reason = _normalized_finish_reason(
        candidate.get("finishReason"),
        {
            "stop",
            "max_tokens",
            "safety",
            "recitation",
            "language",
            "blocklist",
            "prohibited_content",
            "spi",
            "malformed_function_call",
            "image_safety",
        },
    )
    has_alternative_output = any(
        isinstance(part, dict)
        and any(
            key in part
            for key in (
                "functionCall",
                "functionResponse",
                "inlineData",
                "fileData",
                "executableCode",
                "codeExecutionResult",
            )
        )
        for part in parts
    )
    _require_text_completion_disposition(
        config,
        request,
        finish_reason=finish_reason,
        normal_finish_reasons={"stop"},
        has_alternative_output=has_alternative_output,
    )
    text = "".join(
        str(part.get("text"))
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )
    if not text.strip():
        retriable = _empty_text_retryable(
            request,
            finish_reason=finish_reason,
            normal_finish_reasons={"stop"},
            has_alternative_output=has_alternative_output,
        )
        raise _response_error(
            config,
            "candidates[0].content.parts has no text",
            retriable=retriable,
            retry_reason="normal_finish_empty_text" if retriable else "",
            finish_reason=finish_reason,
        )
    return text, finish_reason


def _anthropic_text(config: LLMConfig, response: dict[str, Any]) -> str:
    text, _stop_reason = _parse_anthropic_completion(
        config,
        LLMRequest(messages=[]),
        response,
    )
    return text


def _parse_anthropic_completion(
    config: LLMConfig,
    request: LLMRequest,
    response: dict[str, Any],
) -> tuple[str, str]:
    content = response.get("content")
    if not isinstance(content, list) or not content:
        raise _response_error(config, "content is missing or empty")
    stop_reason = _normalized_finish_reason(
        response.get("stop_reason"),
        {
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
            "model_context_window_exceeded",
        },
    )
    has_alternative_output = any(
        isinstance(block, dict) and block.get("type") != "text"
        for block in content
    )
    _require_text_completion_disposition(
        config,
        request,
        finish_reason=stop_reason,
        normal_finish_reasons={"end_turn", "stop_sequence"},
        has_alternative_output=has_alternative_output,
        has_refusal=stop_reason == "refusal",
    )
    text = "".join(
        str(block.get("text"))
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )
    if not text.strip():
        retriable = _empty_text_retryable(
            request,
            finish_reason=stop_reason,
            normal_finish_reasons={"end_turn"},
            has_alternative_output=has_alternative_output,
        )
        raise _response_error(
            config,
            "content has no text block",
            retriable=retriable,
            retry_reason="normal_finish_empty_text" if retriable else "",
            finish_reason=stop_reason,
        )
    return text, stop_reason
