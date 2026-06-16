"""LLM provider adapters for OpenAI and Vertex AI."""

from __future__ import annotations

import json
from typing import Any
from urllib import request as urlrequest

from llm.config import LLMConfig, load_llm_config
from llm.google_auth import get_google_access_token
from llm.types import LLMMessage, LLMProvider, LLMRequest, LLMResponse


class FakeProvider:
    """Deterministic provider for offline Agent contract tests."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        text = "\n".join(message.content for message in request.messages)
        lowered = text.lower()
        if "intent" in lowered:
            payload = {
                "intent": "framework_question" if "how" in lowered or "what" in lowered else "benchmark_request",
                "confidence": 0.8,
                "reason": "offline fake provider contract response",
            }
        elif "return json" in lowered or "benchmark request" in lowered:
            payload = {
                "chain": "solana" if "solana" in lowered else "",
                "goal": "max_stable_qps" if "max" in lowered or "maximum" in lowered else "baseline",
                "rpc_mode": "mixed" if "mixed" in lowered else "single",
                "use_fake_node": "fake" in lowered or "mock" in lowered,
                "deployment": {
                    "type": "kubernetes" if "gke" in lowered or "kubernetes" in lowered else "unknown",
                    "provider": "gcp" if "gke" in lowered or "gcp" in lowered or "google" in lowered else "",
                },
                "bottleneck_focus": ["disk"] if "disk" in lowered else ["cpu", "memory", "disk", "network", "rpc_errors"],
            }
        else:
            payload = {"ok": True, "provider": "fake", "message": "offline smoke response"}
        return LLMResponse(
            text=json.dumps(payload, sort_keys=True),
            model=self.config.model or "fake",
            provider="fake",
            raw=payload,
        )


class OpenAIProvider:
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("openai is required for LLM_PROVIDER=openai") from exc

        client = OpenAI()
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


class VertexGeminiOpenAIProvider:
    """Gemini on Vertex through the OpenAI-compatible endpoint."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("openai is required for Vertex Gemini OpenAI-compatible calls") from exc

        token = get_google_access_token(self.config)
        base_url = (
            f"https://{self.config.google_location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.config.google_project}/locations/{self.config.google_location}/endpoints/openapi"
        )
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


class VertexClaudeProvider:
    """Claude partner models on Vertex AI."""

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
        response = _post_json(url, payload, token)
        text = "".join(block.get("text", "") for block in response.get("content", []) if block.get("type") == "text")
        return LLMResponse(
            text=text,
            model=self.config.model,
            provider=self.config.provider,
            raw=response,
        )


def _post_json(url: str, payload: dict[str, Any], bearer_token: str) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=120) as response:  # nosec B310 - URL is a configured Google API endpoint
        return json.loads(response.read().decode("utf-8"))


def provider_from_config(config: LLMConfig | None = None) -> LLMProvider:
    config = config or load_llm_config()
    errors = config.validate()
    if errors:
        raise ValueError("; ".join(errors))
    if config.provider == "fake":
        return FakeProvider(config)
    if config.provider == "openai":
        return OpenAIProvider(config)
    if config.provider == "vertex_gemini_openai":
        return VertexGeminiOpenAIProvider(config)
    if config.provider == "vertex_claude":
        return VertexClaudeProvider(config)
    raise ValueError(f"unsupported LLM_PROVIDER: {config.provider}")


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
