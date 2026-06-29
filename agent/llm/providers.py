"""LLM provider adapters for OpenAI, Gemini, Anthropic, and Vertex AI."""

from __future__ import annotations

import json
from typing import Any
from urllib import parse as urlparse
from urllib import request as urlrequest

from llm.config import LLMConfig, load_llm_config
from llm.google_auth import get_google_access_token
from llm.types import LLMMessage, LLMProvider, LLMRequest, LLMResponse


class OpenAIProvider:
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("openai is required for LLM_PROVIDER=openai") from exc

        client = OpenAI(api_key=self.config.openai_api_key or None)
        response = client.chat.completions.create(
            model=self.config.model,
            messages=_openai_messages(request.messages),
            **_openai_completion_options(self.config.model, request),
        )
        text = response.choices[0].message.content or ""
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

        client = OpenAI(api_key=self.config.deepseek_api_key or None, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model=self.config.model,
            messages=_openai_messages(request.messages),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools or None,
        )
        text = response.choices[0].message.content or ""
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
        client = OpenAI(api_key=token, base_url=base_url)
        response = client.chat.completions.create(
            model=f"google/{self.config.model}",
            messages=_openai_messages(request.messages),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools or None,
        )
        text = response.choices[0].message.content or ""
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
        response = _post_json(url, payload, headers={})
        text = _gemini_text(response)
        return LLMResponse(text=text, model=self.config.model, provider=self.config.provider, raw=response)


class VertexClaudeProvider:
    """`claude` partner models on Vertex AI."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        system, messages = _anthropic_messages(request.messages)
        token = get_google_access_token(self.config)
        url = (
            f"https://{self.config.google_location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.config.google_project}/locations/{self.config.google_location}/"
            f"publishers/anthropic/models/{self.config.model}:rawPredict"
        )
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
        response = _post_json(url, payload, headers={"Authorization": f"Bearer {token}"})
        text = "".join(block.get("text", "") for block in response.get("content", []) if block.get("type") == "text")
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
        )
        text = "".join(block.get("text", "") for block in response.get("content", []) if block.get("type") == "text")
        return LLMResponse(text=text, model=self.config.model, provider=self.config.provider, raw=response)


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
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
    with urlrequest.urlopen(req, timeout=120) as response:  # nosec B310 - URL is a configured Google API endpoint
        return json.loads(response.read().decode("utf-8"))


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


def _gemini_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)
