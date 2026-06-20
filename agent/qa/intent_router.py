"""Intent routing for prompt-first Agent interactions."""

from __future__ import annotations

import json
from typing import Any

from llm.orchestrator import PromptOrchestrator
from llm.providers import provider_from_config
from llm.types import LLMMessage, LLMProvider, LLMRequest


INTENTS = {
    "benchmark_request",
    "framework_question",
    "artifact_question",
    "plan_edit",
    "onboarding_request",
    "out_of_scope",
}


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
                        content=PromptOrchestrator(provider).system_prompt("intent"),
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
    if _looks_like_onboarding(lowered):
        return {"intent": "onboarding_request", "confidence": 0.88, "reason": "secondary development or onboarding request detected", "source": "deterministic"}
    if any(token in lowered for token in ("how many", "supported", "which chains", "support", "capability", "多少", "哪些", "支持")) and any(
        token in lowered for token in ("chain", "chains", "rpc", "method", "family", "fixture", "链", "方法", "能力")
    ):
        return {"intent": "framework_question", "confidence": 0.85, "reason": "framework capability question detected", "source": "deterministic"}
    if any(lowered.startswith(prefix) for prefix in ("what ", "how ", "why ", "where ", "does ", "do ", "can ")):
        if not any(token in lowered for token in ("create ", "run ", "execute ", "start ", "submit ", "压测", "执行")):
            return {"intent": "framework_question", "confidence": 0.75, "reason": "explanatory framework question detected", "source": "deterministic"}
    if any(token in lowered for token in ("create ", "run ", "test ", "benchmark", "execute ", "压测", "测试")) and any(
        token in lowered for token in benchmark_tokens
    ):
        return {"intent": "benchmark_request", "confidence": 0.85, "reason": "benchmark action detected", "source": "deterministic"}
    if any(token in lowered for token in question_tokens) and any(
        token in lowered for token in ("fake-node", "fake node", "config", "readme", "report", "prometheus", "grafana", "配置", "报告")
    ):
        return {"intent": "framework_question", "confidence": 0.75, "reason": "framework component question detected", "source": "deterministic"}
    if any(token in lowered for token in benchmark_tokens):
        return {"intent": "benchmark_request", "confidence": 0.75, "reason": "benchmark terms detected", "source": "deterministic"}
    if any(token in lowered for token in question_tokens):
        return {"intent": "framework_question", "confidence": 0.65, "reason": "framework question terms detected", "source": "deterministic"}
    return {"intent": "framework_question", "confidence": 0.4, "reason": "default to framework question", "source": "deterministic"}


def _looks_like_onboarding(lowered: str) -> bool:
    explicit_extension_tokens = (
        "secondary development",
        "extend",
        "extension",
        "onboard",
        "onboarding",
        "add a new chain",
        "add new chain",
        "add chain",
        "new protocol",
        "protocol family",
        "new family",
        "custom rpc",
        "add rpc",
        "chain template",
        "draft chain",
        "integrate kb",
        "agent platform",
        "internal agent",
        "enterprise agent",
        "tool schema",
        "tool-call",
        "knowledge base",
        "二次开发",
        "扩展",
        "新增链",
        "添加链",
        "新增区块链",
        "增加区块链",
        "新增协议",
        "协议 family",
        "新增 rpc",
        "增加 rpc",
        "自定义 rpc",
        "chain template",
        "知识库",
        "企业 agent",
        "内部 agent",
        "agent 平台",
        "工具 schema",
    )
    action_tokens = (
        "how",
        "where",
        "generate",
        "plan",
        "draft",
        "create",
        "support",
        "integrate",
        "如何",
        "怎么",
        "哪里",
        "生成",
        "计划",
        "支持",
        "集成",
        "接入",
    )
    if any(token in lowered for token in explicit_extension_tokens):
        return True
    if any(token in lowered for token in ("rpc method", "rpc 方法", "方法")) and any(token in lowered for token in action_tokens):
        if any(token in lowered for token in ("add", "new", "custom", "extend", "onboard", "draft", "新增", "增加", "添加", "自定义", "扩展")):
            return True
    if any(token in lowered for token in ("kb", "rag")) and any(token in lowered for token in action_tokens):
        return True
    return False


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
