"""Model-backed advisory services that cannot control workflow transitions."""

from __future__ import annotations

import json
import re
from typing import Any

from ..llm.providers import provider_from_config
from ..llm.types import (
    LLMMessage,
    LLMProviderError,
    LLMRequest,
    LLMTurnTimeoutError,
)
from ..onboarding.families import SUPPORTED_FAMILIES
from .context import workflow_snapshot
from .state import AgentGraphState, DEFAULT_GROUP_ORDER


ADAPTER_FAMILIES = list(SUPPORTED_FAMILIES)


def resolve_unknown_chain_identity(
    state: AgentGraphState,
    chain_text: str,
) -> dict[str, Any]:
    """Ask the configured model whether an unknown chain appears to exist."""

    try:
        provider = provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=_chain_identity_prompt()),
                    LLMMessage(
                        role="user",
                        content=json.dumps(
                            _chain_identity_payload(state, chain_text),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=900,
            )
        )
        payload = _parse_json_object(response.text)
    except (LLMTurnTimeoutError, LLMProviderError):
        raise
    except Exception as exc:
        payload = {
            "chain_exists": None,
            "reason": (
                "chain identity resolver failed: "
                f"{type(exc).__name__}"
            ),
            "confidence": "low",
        }
    payload.setdefault("chain_text", chain_text)
    return payload


def extract_chain_mention(
    state: AgentGraphState,
    text: str,
) -> dict[str, Any]:
    """Extract an explicit chain/network mention without changing state."""

    try:
        provider = provider_from_config()
        payload = _extract_chain_mention_with_provider(
            provider,
            state,
            text,
        )
    except (LLMTurnTimeoutError, LLMProviderError):
        raise
    except Exception as exc:
        payload = {
            "found": False,
            "reason": (
                "chain mention extraction failed: "
                f"{type(exc).__name__}"
            ),
            "confidence": "low",
        }
    payload.setdefault("found", False)
    return payload


def _extract_chain_mention_with_provider(
    provider: Any,
    state: AgentGraphState,
    text: str,
) -> dict[str, Any]:
    response = provider.complete(
        LLMRequest(
            messages=[
                LLMMessage(role="system", content=_chain_mention_prompt()),
                LLMMessage(
                    role="user",
                    content=json.dumps(
                        _chain_mention_payload(state, text),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=300,
        )
    )
    return _parse_json_object(response.text)


def extract_rpc_schema_from_evidence(
    state: AgentGraphState,
    evidence: str,
    *,
    method_hint: str = "",
) -> dict[str, Any]:
    """Extract an untrusted typed RPC schema draft from supplied evidence."""

    try:
        provider = provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=_rpc_schema_prompt()),
                    LLMMessage(
                        role="user",
                        content=json.dumps(
                            _rpc_schema_payload(
                                state,
                                evidence,
                                method_hint=method_hint,
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=1200,
            )
        )
        payload = _parse_json_object(response.text)
    except (LLMTurnTimeoutError, LLMProviderError):
        raise
    except Exception as exc:
        payload = {
            "status": "failed",
            "reason": f"schema extraction failed: {type(exc).__name__}",
            "confidence": "low",
        }
    payload.setdefault("status", "draft")
    payload.setdefault("method", method_hint)
    response_json_type = str(payload.get("response_json_type") or "unknown").strip().lower()
    payload["response_json_type"] = (
        response_json_type
        if response_json_type
        in {"unknown", "null", "boolean", "string", "number", "array", "object"}
        else "unknown"
    )
    return payload


def analyze_evidence_with_model(
    state: AgentGraphState,
    evidence: str,
    user_question: str,
) -> str:
    """Analyze evidence without authorizing an action or changing state."""

    language = str(state.get("language") or "en")
    if not evidence.strip():
        return (
            "没有可分析的日志证据。请粘贴真实日志/错误栈，或指定 job_id。"
            if language.startswith("zh")
            else (
                "No log evidence is available. Paste real logs/a stack trace, "
                "or name a job_id."
            )
        )
    try:
        provider = provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "You analyze execution evidence for AnyChain "
                            "Benchmark Agent. Use only the supplied evidence "
                            "and workflow context. Distinguish observed facts, "
                            "likely causes, and verification steps. Never claim "
                            "a benchmark passed, an endpoint works, or a metric "
                            "exists without evidence. Do not propose state "
                            "mutations or pretend to run tools. Answer in the "
                            "requested language while preserving commands, "
                            "paths, environment variables, job ids, chain "
                            "names, and RPC methods."
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=json.dumps(
                            {
                                "language": language,
                                "question": user_question,
                                "evidence": evidence,
                                "workflow_state": workflow_snapshot(state),
                                "framework_boundaries": {
                                    "supported_groups": list(
                                        DEFAULT_GROUP_ORDER
                                    ),
                                    "supported_adapter_families": (
                                        ADAPTER_FAMILIES
                                    ),
                                },
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=1200,
            )
        )
        answer = str(response.text or "").strip()
        if answer:
            return answer
    except (LLMTurnTimeoutError, LLMProviderError):
        raise
    except Exception as exc:
        error_type = type(exc).__name__
    else:
        error_type = "EmptyResponse"
    preview = "\n".join(evidence.splitlines()[:8])
    if language.startswith("zh"):
        return (
            f"当前无法调用配置模型完成证据分析（{error_type}）。我不会猜测根因。\n"
            "请确认模型配置后重试，或提供对应 job_id 以读取本地任务产物。\n"
            f"已保存证据预览：\n{preview}"
        )
    return (
        "The configured model could not analyze this evidence "
        f"({error_type}); I will not guess the root cause.\n"
        "Verify the model configuration and retry, or provide the related "
        "job_id so local artifacts can be read.\n"
        f"Saved evidence preview:\n{preview}"
    )


def _chain_identity_prompt() -> str:
    return (
        "You resolve blockchain chain names for AnyChain Benchmark Agent. "
        "Return one JSON object only. Do not explain. The input chain is not "
        "one of the configured AnyChain templates. Decide whether it appears "
        "to be a real blockchain/network name, whether it is a typo or partial "
        "alias for a known chain, and what adapter family it likely uses. Do "
        "not claim certainty if unsure. Allowed adapter_family values: "
        f"{', '.join(ADAPTER_FAMILIES + ['unsupported', 'unknown'])}. "
        "Schema: {chain_exists:boolean|null, canonical_chain_name:string, "
        "adapter_family:string, protocol_or_api:string, "
        "possible_known_chain:string, evidence_summary:string, "
        "confidence:'low'|'medium'|'high'}."
    )


def _chain_identity_payload(
    state: AgentGraphState,
    chain_text: str,
) -> dict[str, Any]:
    framework = state.get("framework_summary") or {}
    return {
        "unknown_chain_text": chain_text,
        "known_chains": framework.get("chains", [])[:120],
        "supported_adapter_families": ADAPTER_FAMILIES,
        "web_research": state.get("web_research") or {},
        "target_mode": state.get("target_mode") or "",
        "workflow_mode": state.get("workflow_mode") or "",
    }


def _chain_mention_prompt() -> str:
    return (
        "You extract blockchain or network names from one AnyChain user turn. "
        "Return one JSON object only. Do not explain. Extract only an explicit "
        "chain/network/product name that the user wants to benchmark or "
        "observe. Use the supplied known_chains as evidence, but allow a "
        "different explicit name to remain unknown for later research. Do not "
        "treat cloud regions, zones, instance types, disks, URLs, QPS values, "
        "or generic words as chain names. If no explicit chain/network is "
        "present, return found=false. Schema: {found:boolean, "
        "chain_text:string, reason:string, "
        "confidence:'low'|'medium'|'high'}."
    )


def _chain_mention_payload(
    state: AgentGraphState,
    text: str,
) -> dict[str, Any]:
    framework = state.get("framework_summary") or {}
    identity = state.get("chain_identity") or {}
    return {
        "user_text": text,
        "known_chains": framework.get("chains", [])[:120],
        "current_target_mode": state.get("target_mode") or "",
        "current_workflow_mode": state.get("workflow_mode") or "",
        "current_chain": identity.get("canonical") or "",
    }


def _rpc_schema_prompt() -> str:
    return (
        "You extract RPC method schemas for AnyChain Benchmark Agent. Return "
        "one JSON object only. Do not explain. The user may provide a curl "
        "command, JSON-RPC request, response sample, docs excerpt, URL text, "
        "or endpoint transcript. First classify the evidence kind and "
        "transport before extracting a method. Do not treat a URL, REST path, "
        "or REST documentation title as a JSON-RPC method name. Infer only "
        "what is supported by the evidence; mark unknown fields as unknown. "
        "Do not invent parameters. Schema: {status:'draft'|'failed', "
        "evidence_kind:'jsonrpc_request'|'rest_endpoint'|'rest_path'|"
        "'response_sample'|'docs_excerpt'|'method_name'|'mixed'|'unknown', "
        "transport:'jsonrpc'|'rest'|'unknown', method:string, "
        "endpoint_url:string, rest_path:string, http_method:string, "
        "params:[{index:number,name:string,json_type:string,"
        "semantic_type:string,encoding:string,meaning:string,example:any,"
        "required:boolean|null}], params_json:any, response_summary:string, "
        "response_json_type:'unknown'|'null'|'boolean'|'string'|'number'|"
        "'array'|'object', "
        "response_fields:[{name:string,type:string,meaning:string}], "
        "auth_notes:string, rate_limit_notes:string, conflicts:[string], "
        "confidence:'low'|'medium'|'high', reason:string, "
        "evidence_summary:string}."
    )


def _rpc_schema_payload(
    state: AgentGraphState,
    evidence: str,
    *,
    method_hint: str,
) -> dict[str, Any]:
    identity = state.get("chain_identity") or {}
    return {
        "evidence": evidence,
        "method_hint": method_hint,
        "target_mode": state.get("target_mode") or "",
        "workflow_mode": state.get("workflow_mode") or "",
        "chain": identity.get("canonical") or identity.get("raw") or "",
        "adapter_family": identity.get("adapter_family") or "",
        "web_research": state.get("web_research") or {},
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "intent": "unknown",
            "reason": "model did not return valid JSON",
            "confidence": "low",
        }
    if isinstance(payload, dict):
        return payload
    return {
        "intent": "unknown",
        "reason": "model returned non-object JSON",
        "confidence": "low",
    }
