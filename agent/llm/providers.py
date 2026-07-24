"""LLM provider adapters for OpenAI, Gemini, Anthropic, and Vertex AI."""

from __future__ import annotations

import importlib.util
import json
import socket
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
    LLMTurnTimeoutError,
    ensure_turn_active,
    llm_turn_scope,
    remaining_turn_seconds,
)


class OpenAIProvider:
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("openai is required for LLM_PROVIDER=openai") from exc

        client = _openai_client(OpenAI, self.config, api_key=self.config.openai_api_key or None)
        response = _openai_request(
            self.config,
            lambda: client.chat.completions.create(
                model=self.config.model,
                messages=_openai_messages(request.messages),
                **_openai_completion_options(self.config.model, request),
            ),
        )
        text = _openai_response_text(self.config, response)
        return LLMResponse(
            text=text,
            model=self.config.model,
            provider=self.config.provider,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
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

        client = _openai_client(
            OpenAI,
            self.config,
            api_key=self.config.deepseek_api_key or None,
            base_url="https://api.deepseek.com",
        )
        response = _openai_request(
            self.config,
            lambda: client.chat.completions.create(
                model=self.config.model,
                messages=_openai_messages(request.messages),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools or None,
            ),
        )
        text = _openai_response_text(self.config, response)
        return LLMResponse(
            text=text,
            model=self.config.model,
            provider=self.config.provider,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
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

        token = get_google_access_token(self.config)
        base_url = _vertex_openai_base_url(self.config)
        client = _openai_client(OpenAI, self.config, api_key=token, base_url=base_url)
        response = _openai_request(
            self.config,
            lambda: client.chat.completions.create(
                model=f"google/{self.config.model}",
                messages=_openai_messages(request.messages),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools or None,
            ),
        )
        text = _openai_response_text(self.config, response)
        return LLMResponse(
            text=text,
            model=self.config.model,
            provider=self.config.provider,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
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
        response = _post_json(url, payload, headers={}, config=self.config)
        text = _gemini_text(self.config, response)
        return LLMResponse(text=text, model=self.config.model, provider=self.config.provider, raw=response)


class VertexClaudeProvider:
    """`claude` partner models on Vertex AI."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        system, messages = _anthropic_messages(request.messages)
        token = get_google_access_token(self.config)
        url = _vertex_raw_predict_url(self.config, "anthropic", self.config.model)
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
        response = _post_json(url, payload, headers={"Authorization": f"Bearer {token}"}, config=self.config)
        text = _anthropic_text(self.config, response)
        return LLMResponse(
            text=text,
            model=self.config.model,
            provider=self.config.provider,
            raw=response,
        )


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
        response = _post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            config=self.config,
        )
        text = _anthropic_text(self.config, response)
        return LLMResponse(text=text, model=self.config.model, provider=self.config.provider, raw=response)


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
    return client_type(timeout=timeout, max_retries=config.max_retries, **kwargs)


def _openai_request(config: LLMConfig, call: Any) -> Any:
    ensure_turn_active()
    try:
        response = call()
    except Exception as exc:
        if _is_transport_timeout(exc):
            raise LLMTurnTimeoutError(
                f"{config.provider}/{config.model} request exceeded the active turn deadline or transport timeout",
                provider=config.provider,
                model=config.model,
                stage="provider_request",
            ) from exc
        raise _provider_error(config, exc) from exc
    ensure_turn_active()
    return response


def _response_error(config: LLMConfig, detail: str) -> LLMProviderError:
    return LLMProviderError(
        f"{config.provider}/{config.model} returned a malformed successful response: {detail}",
        provider=config.provider,
        model=config.model,
        category="response",
        stage="provider_response",
    )


def _openai_response_text(config: LLMConfig, response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        raise _response_error(config, "choices is missing or empty")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise _response_error(config, "choices[0].message is missing")
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise _response_error(config, "choices[0].message.content is missing or empty")
    return content


def _is_transport_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    name = type(exc).__name__.casefold()
    module = type(exc).__module__.casefold()
    return "timeout" in name and any(marker in module for marker in ("openai", "httpx", "httpcore", "urllib"))


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
        raise LLMTurnTimeoutError(
            f"{config.provider}/{config.model} request exceeded the active turn deadline or transport timeout",
            provider=config.provider,
            model=config.model,
            stage="provider_request",
        ) from exc
    except urlerror.HTTPError as exc:
        raise _provider_error(config, exc) from exc
    except urlerror.URLError as exc:
        if _is_transport_timeout(exc.reason if isinstance(exc.reason, BaseException) else exc):
            raise LLMTurnTimeoutError(
                f"{config.provider}/{config.model} request exceeded the active turn deadline or transport timeout",
                provider=config.provider,
                model=config.model,
                stage="provider_request",
            ) from exc
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
            ))
    except LLMTurnTimeoutError as exc:
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
    elif isinstance(exc, (ConnectionError, urlerror.URLError)):
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
    text = "".join(
        str(part.get("text"))
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )
    if not text.strip():
        raise _response_error(config, "candidates[0].content.parts has no text")
    return text


def _anthropic_text(config: LLMConfig, response: dict[str, Any]) -> str:
    content = response.get("content")
    if not isinstance(content, list) or not content:
        raise _response_error(config, "content is missing or empty")
    text = "".join(
        str(block.get("text"))
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )
    if not text.strip():
        raise _response_error(config, "content has no text block")
    return text
