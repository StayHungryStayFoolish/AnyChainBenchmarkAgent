"""LLM-assisted request drafting with deterministic validation."""

from __future__ import annotations

import json
from typing import Any

from llm.providers import provider_from_config
from llm.types import LLMMessage, LLMProvider, LLMRequest
from qa.intent_router import _json_text
from qa.request_drafter import draft_request


GOALS = {"smoke", "baseline", "max_stable_qps", "stress", "bottleneck_confirmation", "regression"}
RPC_MODES = {"single", "mixed"}
DEPLOYMENT_TYPES = {"vm", "kubernetes", "unknown"}
PROVIDERS = {"gcp", "aws", "azure", "other", ""}
BOTTLENECKS = {"cpu", "memory", "disk", "network", "sync_health", "rpc_errors"}


def draft_request_with_llm(prompt: str, provider: LLMProvider | None = None) -> dict[str, Any]:
    deterministic = draft_request(prompt)
    try:
        provider = provider or provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "Turn the user benchmark goal into JSON only. "
                            "Allowed fields: chain, goal, rpc_mode, use_fake_node, deployment, "
                            "observability, dependency_mode, bottleneck_focus, qps, workload. "
                            "Do not invent endpoints, credentials, commands, or shell scripts."
                        ),
                    ),
                    LLMMessage(role="user", content=prompt),
                ],
                temperature=0,
                max_tokens=2048,
            )
        )
        parsed = json.loads(_json_text(response.text))
        return normalize_request(deterministic, parsed, source=f"llm:{response.provider}")
    except Exception as exc:
        return {**deterministic, "llm_status": "fallback", "llm_error": str(exc)}


def normalize_request(base: dict[str, Any], candidate: dict[str, Any], source: str = "normalizer") -> dict[str, Any]:
    normalized = dict(base)
    if isinstance(candidate.get("chain"), str):
        normalized["chain"] = candidate["chain"].strip().lower()
    if candidate.get("goal") in GOALS:
        normalized["goal"] = candidate["goal"]
    if candidate.get("rpc_mode") in RPC_MODES:
        normalized["rpc_mode"] = candidate["rpc_mode"]
    if isinstance(candidate.get("use_fake_node"), bool):
        normalized["use_fake_node"] = candidate["use_fake_node"]
    if isinstance(candidate.get("deployment"), dict):
        deployment = candidate["deployment"]
        normalized["deployment"] = {
            "type": deployment.get("type") if deployment.get("type") in DEPLOYMENT_TYPES else normalized.get("deployment", {}).get("type", "unknown"),
            "provider": deployment.get("provider") if deployment.get("provider") in PROVIDERS else normalized.get("deployment", {}).get("provider", ""),
        }
    if isinstance(candidate.get("observability"), dict):
        observability = candidate["observability"]
        normalized["observability"] = {
            "enabled": bool(observability.get("enabled", normalized.get("observability", {}).get("enabled", False))),
            "mode": observability.get("mode") if observability.get("mode") in {"local", "exporter"} else normalized.get("observability", {}).get("mode", "local"),
        }
    if isinstance(candidate.get("bottleneck_focus"), list):
        focus = [item for item in candidate["bottleneck_focus"] if item in BOTTLENECKS]
        if focus:
            normalized["bottleneck_focus"] = focus
    if isinstance(candidate.get("qps"), dict):
        qps = _normalize_qps(candidate["qps"])
        if qps:
            normalized["qps"] = qps
    if isinstance(candidate.get("workload"), dict):
        workload = _normalize_workload(candidate["workload"])
        if workload:
            normalized["workload"] = workload
            normalized["rpc_mode"] = "mixed" if len(workload.get("methods", [])) > 1 else normalized.get("rpc_mode", "single")
    normalized["llm_status"] = "accepted"
    normalized["llm_source"] = source
    return normalized


def _normalize_qps(candidate: dict[str, Any]) -> dict[str, int]:
    output: dict[str, int] = {}
    for key in ("initial", "max", "step", "duration_seconds"):
        try:
            value = int(candidate.get(key, 0))
        except (TypeError, ValueError):
            continue
        if value > 0:
            output[key] = value
    return output


def _normalize_workload(candidate: dict[str, Any]) -> dict[str, Any]:
    methods = []
    for method in candidate.get("methods", []):
        if not isinstance(method, dict) or not method.get("name"):
            continue
        item = {"name": str(method["name"])}
        if "weight" in method:
            try:
                weight = float(method["weight"])
            except (TypeError, ValueError):
                weight = 0.0
            if weight > 0:
                item["weight"] = weight
        if isinstance(method.get("params"), list):
            item["params"] = method["params"]
        if isinstance(method.get("custom"), bool):
            item["custom"] = method["custom"]
        if isinstance(method.get("param_format"), str):
            item["param_format"] = method["param_format"]
        methods.append(item)
    return {"methods": methods} if methods else {}
