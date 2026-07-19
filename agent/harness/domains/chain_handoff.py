"""Cohesive Chain/RPC component extracted from the product domain facade."""

from __future__ import annotations

from copy import deepcopy

from ..input_values import normalize_scalar
from ..localization import localized
from ..questions import render_question
from ..state import AgentGraphState
from ..transitions import mark_group_reconfigured, record_group_invalidations
from .chain_rpc_questions import _case3_evidence_question
from .chain_rpc_support import _next_group, _set_control
from .rpc_catalog import catalog_method_names, draft_view, validated_contracts_view

def _promote_case2_endpoint(state: AgentGraphState) -> None:
    identity = state.setdefault("chain_identity", {})
    evidence = state.setdefault("endpoint_evidence", {})
    confirmed = state.setdefault("confirmed_config", {})
    chain = normalize_scalar(identity.get("canonical") or identity.get("raw"))
    endpoint = normalize_scalar(evidence.get("candidate_endpoint"))
    methods = catalog_method_names(state)
    method = methods[0] if len(methods) == 1 else ""
    identity.update({"status": "confirmed", "case": "case2_runtime_override"})
    confirmed["BLOCKCHAIN_NODE"] = chain
    if endpoint:
        confirmed["LOCAL_RPC_URL"] = endpoint
        evidence["local_rpc_url_ready"] = True
    state["target_mode"] = "real-node"
    state["workflow_mode"] = "rpc_benchmark"
    if method and not (state.get("workload") or {}).get("confirmed"):
        state["rpc_mode"] = "single"
        state["workload"] = {"confirmed": True, "choice": "new_chain_verified_method", "methods": [method], "replace_defaults": True, "job_local_override": True}
    record_group_invalidations(state, "target_mode")
    # This atomic promotion supplies fresh endpoint and workload state for the
    # new mode. They are not stale dependents to clear at commit time.
    mark_group_reconfigured(state, "endpoint_process")
    mark_group_reconfigured(state, "workload_rpc")
    _set_control(state, 'active_group', _next_group(state))
    _set_control(state, 'visible_response', [localized(state.get("language", "en"), "已切换到 real-node 路径，并把已验证 endpoint/method 作为本次 job-local workload override 继续；不会修改 config/chains 原始模板。", "Switched to the real-node path and will continue with the verified endpoint/method as a job-local workload override; config/chains templates are not modified.")])


def _prepare_case2_handoff(state: AgentGraphState) -> None:
    identity = state.setdefault("chain_identity", {})
    evidence = state.get("endpoint_evidence") or {}
    identity["status"] = "needs_review_handoff"
    handoff = {
        "status": "ready",
        "kind": "case2_fixture_implementation",
        "chain": normalize_scalar(identity.get("canonical") or identity.get("raw")),
        "adapter_family": normalize_scalar(identity.get("adapter_family")),
        "validated_endpoint": normalize_scalar(evidence.get("candidate_endpoint")),
        "endpoint_probe": deepcopy(evidence.get("new_chain_endpoint_probe") or {}),
        "method_probe": deepcopy(evidence.get("new_chain_method_probe") or {}),
        "validated_methods": deepcopy(validated_contracts_view(state)),
        "schema_draft": deepcopy(draft_view(state)),
        "schema_evidence": str(identity.get("schema_evidence") or ""),
        "workload": deepcopy(state.get("workload") or {}),
        "requirements": ["chain template", "recorded endpoint fixture", "fixture coverage", "preflight and smoke validation"],
    }
    state["secondary_handoff"] = handoff
    _set_control(state, 'visible_response', [_case2_handoff_message(state)])


def _record_case3_evidence(state: AgentGraphState, evidence: str) -> None:
    identity = state.setdefault("chain_identity", {})
    handoff = state.setdefault("secondary_handoff", {})
    items = handoff.setdefault("evidence", [])
    if evidence:
        items.append(evidence)
    identity.update({"status": "case3_collecting_evidence", "case": "case3"})
    handoff.update({"status": "collecting_evidence", "kind": "case3_protocol_adapter_implementation", "chain": normalize_scalar(identity.get("canonical") or identity.get("raw"))})
    _set_control(state, 'active_group', "chain_identity")
    _set_control(state, 'pending_question', _case3_evidence_question(state))
    _set_control(state, 'visible_response', [
        localized(state.get("language", "en"), f"已记录第 {len(items)} 条协议开发证据。", f"Recorded protocol-development evidence item {len(items)}."),
        render_question(state["pending_question"], state.get("language", "en")),
    ])


def _prepare_case3_handoff(state: AgentGraphState) -> None:
    identity = state.setdefault("chain_identity", {})
    handoff = state.setdefault("secondary_handoff", {})
    evidence = list(handoff.get("evidence") or [])
    identity["status"] = "needs_review_handoff"
    handoff.update(
        {
            "status": "ready",
            "kind": "case3_protocol_adapter_implementation",
            "chain": normalize_scalar(identity.get("canonical") or identity.get("raw")),
            "adapter_family": normalize_scalar(identity.get("adapter_family") or "unsupported"),
            "evidence": evidence,
            "evidence_count": len(evidence),
            "requirements": [
                "cite official transport, authentication, and RPC schema documentation",
                "implement a protocol adapter without reusing an incompatible family",
                "add a chain template and real-endpoint fixtures",
                "run schema, fixture coverage, preflight, and smoke validation",
                "update English and Chinese product documentation",
            ],
        }
    )
    handoff["draft"] = _case3_handoff_draft(state, evidence)
    _set_control(state, 'visible_response', [handoff["draft"]])


def _case2_handoff_message(state: AgentGraphState) -> str:
    handoff = state.get("secondary_handoff") or {}
    endpoint_probe = handoff.get("endpoint_probe") or {}
    method_probe = handoff.get("method_probe") or {}
    return localized(
        state.get("language", "en"),
        "已进入二次开发交接路径。需要基于已验证 endpoint/method/schema 生成 chain template、fake-node fixture、fixture coverage 和 smoke 验证任务；在这些 gate 通过前，不会声明该链已被生产支持。"
        f" 证据：endpoint={endpoint_probe.get('evidence_file') or '<none>'}, method={method_probe.get('evidence_file') or '<none>'}, chain={handoff.get('chain') or '<unknown>'}。",
        "Entered the secondary-development handoff path. A chain template, fake-node fixture, fixture coverage, and smoke validation task must be generated from the verified endpoint/method/schema; this chain will not be claimed production-supported until those gates pass."
        f" Evidence: endpoint={endpoint_probe.get('evidence_file') or '<none>'}, method={method_probe.get('evidence_file') or '<none>'}, chain={handoff.get('chain') or '<unknown>'}.",
    )


def _case3_handoff_draft(state: AgentGraphState, evidence: list[str]) -> str:
    chain = normalize_scalar((state.get("chain_identity") or {}).get("canonical") or (state.get("chain_identity") or {}).get("raw")) or "<unknown>"
    preview = "\n".join(f"- {item}" for item in evidence[-5:]) or "- <none>"
    if str(state.get("language") or "en").startswith("zh"):
        return (
            f"二次开发交接草案：`{chain}` 当前不属于已支持协议族。\n"
            "目标：让另一个 AI 基于官方资料新增协议 adapter，并完成闭环验证。\n"
            "必须完成：\n"
            "1. 阅读并引用官方协议/RPC 文档，确认 transport、auth、request/response schema。\n"
            "2. 新增 adapter 或扩展现有 adapter，不得复用错误协议族。\n"
            "3. 新增 chain template，并定义可验证的 RPC method、参数、响应解析和错误处理。\n"
            "4. 基于真实 endpoint 录制 fixture，补充 fake-node 响应样本。\n"
            "5. 运行 schema validator、fixture coverage、preflight 和 smoke 测试。\n"
            "6. 更新中英文文档，确保 Agent 知识库和实际代码一致。\n"
            f"已收集证据：\n{preview}"
        )
    return (
        f"Secondary-development handoff draft: `{chain}` is outside the supported adapter families.\n"
        "Goal: have another AI implement a protocol adapter from official materials and close the validation loop.\n"
        "Required work:\n"
        "1. Read and cite official protocol/RPC docs; confirm transport, auth, request/response schema.\n"
        "2. Add a new adapter or extend an existing one without reusing the wrong adapter family.\n"
        "3. Add a chain template with verifiable RPC methods, params, response parsing, and error handling.\n"
        "4. Record fixtures from a real endpoint and add fake-node response samples.\n"
        "5. Run schema validator, fixture coverage, preflight, and smoke tests.\n"
        "6. Update English/Chinese docs so Agent knowledge matches code behavior.\n"
        f"Collected evidence:\n{preview}"
    )
