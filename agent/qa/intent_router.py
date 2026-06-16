"""Intent routing for prompt-first Agent interactions."""

from __future__ import annotations

import json
from typing import Any

from llm.providers import provider_from_config
from llm.types import LLMMessage, LLMProvider, LLMRequest


INTENTS = {"benchmark_request", "framework_question", "out_of_scope"}


def route_intent(prompt: str, provider: LLMProvider | None = None, use_llm: bool = False) -> dict[str, Any]:
    deterministic = _deterministic_route(prompt)
    if not use_llm:
        return deterministic

    try:
        provider = provider or provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "Classify the user prompt for a blockchain benchmark Agent. "
                            "Return JSON only with intent, confidence, and reason. "
                            "intent must be benchmark_request, framework_question, or out_of_scope."
                        ),
                    ),
                    LLMMessage(role="user", content=prompt),
                ],
                temperature=0,
                max_tokens=512,
            )
        )
        parsed = json.loads(_json_text(response.text))
        intent = parsed.get("intent")
        if intent in INTENTS:
            return {
                "intent": intent,
                "confidence": _bounded_float(parsed.get("confidence"), 0.6),
                "reason": str(parsed.get("reason", "llm route")),
                "source": response.provider,
            }
    except Exception as exc:
        return {**deterministic, "source": "deterministic_fallback", "llm_error": str(exc)}
    return deterministic


def _deterministic_route(prompt: str) -> dict[str, Any]:
    lowered = prompt.lower()
    benchmark_tokens = (
        "benchmark", "qps", "vegeta", "fake-node", "fake node", "latency",
        "p99", "rpc", "bottleneck", "stress", "smoke", "压测", "瓶颈", "测试",
    )
    question_tokens = (
        "how", "what", "where", "why", "config", "readme", "report", "fake-node",
        "怎么", "如何", "什么", "哪里", "配置", "报告",
    )
    out_tokens = ("stock", "movie", "weather", "recipe", "股票", "天气", "菜谱")
    if any(token in lowered for token in out_tokens) and not any(token in lowered for token in benchmark_tokens):
        return {"intent": "out_of_scope", "confidence": 0.8, "reason": "prompt does not match benchmark domain", "source": "deterministic"}
    if any(token in lowered for token in ("how many", "supported", "which chains", "support", "capability", "多少", "哪些", "支持")) and any(
        token in lowered for token in ("chain", "chains", "rpc", "method", "family", "fixture", "链", "方法", "能力")
    ):
        return {"intent": "framework_question", "confidence": 0.85, "reason": "framework capability question detected", "source": "deterministic"}
    if any(token in lowered for token in question_tokens) and any(
        token in lowered for token in ("fake-node", "fake node", "config", "readme", "report", "prometheus", "grafana", "配置", "报告")
    ):
        return {"intent": "framework_question", "confidence": 0.75, "reason": "framework component question detected", "source": "deterministic"}
    if any(token in lowered for token in benchmark_tokens):
        return {"intent": "benchmark_request", "confidence": 0.75, "reason": "benchmark terms detected", "source": "deterministic"}
    if any(token in lowered for token in question_tokens):
        return {"intent": "framework_question", "confidence": 0.65, "reason": "framework question terms detected", "source": "deterministic"}
    return {"intent": "framework_question", "confidence": 0.4, "reason": "default to framework question", "source": "deterministic"}


def _json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        return stripped[start : end + 1]
    return stripped


def _bounded_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
