"""Cohesive Chain/RPC component extracted from the product domain facade."""

from __future__ import annotations

from copy import deepcopy

from ..input_values import normalize_scalar
from ..state import AgentGraphState
from ..transitions import mark_group_reconfigured, record_group_invalidations
from .chain_rpc_questions import _case3_evidence_question
from .response_fragments import ResponseCollector, emit
from .chain_rpc_support import _next_group
from .rpc_catalog import catalog_method_names, draft_view, validated_contracts_view
from .rpc_receipts import emit_endpoint_role_receipt, emit_workload_commit_receipt

def _promote_case2_endpoint(
    state: AgentGraphState,
    *,
    responses: ResponseCollector,
) -> bool:
    identity = state.setdefault("chain_identity", {})
    evidence = state.setdefault("endpoint_evidence", {})
    confirmed = state.setdefault("confirmed_config", {})
    chain = normalize_scalar(identity.get("canonical") or identity.get("raw"))
    endpoint = normalize_scalar(evidence.get("candidate_endpoint"))
    methods = catalog_method_names(state)
    method = methods[0] if len(methods) == 1 else ""
    endpoint_probe = (
        evidence.get("new_chain_endpoint_probe")
        if isinstance(evidence.get("new_chain_endpoint_probe"), dict)
        else {}
    )
    probe_matches = bool(
        endpoint
        and endpoint_probe.get("ready") is True
        and normalize_scalar(endpoint_probe.get("endpoint")) == endpoint
        and normalize_scalar(endpoint_probe.get("chain")) == chain
        and normalize_scalar(endpoint_probe.get("transport"))
        == normalize_scalar(identity.get("adapter_family"))
        and evidence.get("candidate_endpoint_ready") is True
    )
    if not probe_matches or not methods:
        identity["status"] = (
            "existing_family_needs_endpoint"
            if not probe_matches
            else "existing_family_needs_method"
        )
        emit(
            responses,
            "chain_rpc.response.case2_promotion_blocked",
            source=__name__,
        )
        return False
    identity.update({"status": "confirmed", "case": "case2_runtime_override"})
    confirmed["BLOCKCHAIN_NODE"] = chain
    if endpoint:
        confirmed["LOCAL_RPC_URL"] = endpoint
        evidence["local_rpc_url_ready"] = True
        emit_endpoint_role_receipt(
            state,
            role="final_benchmark",
            case="new_chain",
            endpoint=endpoint,
            ready=True,
            probe_status=endpoint_probe.get("status"),
            chain=chain,
            adapter_family=identity.get("adapter_family"),
            methods=methods,
        )
    state["target_mode"] = "real-node"
    state["workflow_mode"] = "rpc_benchmark"
    if method and not (state.get("workload") or {}).get("confirmed"):
        state["rpc_mode"] = "single"
        state["workload"] = {"confirmed": True, "choice": "new_chain_verified_method", "methods": [method], "replace_defaults": True, "job_local_override": True}
        emit_workload_commit_receipt(state, case="new_chain")
    record_group_invalidations(state, "target_mode")
    # This atomic promotion supplies fresh endpoint and workload state for the
    # new mode. They are not stale dependents to clear at commit time.
    mark_group_reconfigured(state, "endpoint_process")
    mark_group_reconfigured(state, "workload_rpc")
    state['active_group'] = _next_group(state)
    emit(responses, "chain_rpc.response.case2_promoted", source=__name__)
    return True


def _prepare_case2_handoff(
    state: AgentGraphState,
    *,
    responses: ResponseCollector,
) -> None:
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
    endpoint_probe = handoff.get("endpoint_probe") or {}
    method_probe = handoff.get("method_probe") or {}
    emit(
        responses,
        "chain_rpc.response.case2_handoff_ready",
        arguments={
            "endpoint_evidence": str(endpoint_probe.get("evidence_file") or "<none>"),
            "method_evidence": str(method_probe.get("evidence_file") or "<none>"),
            "chain": str(handoff.get("chain") or "<unknown>"),
        },
        payload={"handoff": handoff},
        source=__name__,
    )


def _record_case3_evidence(
    state: AgentGraphState,
    evidence: str,
    *,
    responses: ResponseCollector,
) -> None:
    identity = state.setdefault("chain_identity", {})
    handoff = state.setdefault("secondary_handoff", {})
    items = handoff.setdefault("evidence", [])
    if evidence:
        items.append(evidence)
    identity.update({"status": "case3_collecting_evidence", "case": "case3"})
    handoff.update({"status": "collecting_evidence", "kind": "case3_protocol_adapter_implementation", "chain": normalize_scalar(identity.get("canonical") or identity.get("raw"))})
    state['active_group'] = "chain_identity"
    state['pending_question'] = _case3_evidence_question(state)
    emit(
        responses,
        "chain_rpc.response.case3_evidence_recorded",
        arguments={"count": len(items)},
        source=__name__,
    )


def _prepare_case3_handoff(
    state: AgentGraphState,
    *,
    responses: ResponseCollector,
) -> None:
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
    chain = normalize_scalar((state.get("chain_identity") or {}).get("canonical") or (state.get("chain_identity") or {}).get("raw")) or "<unknown>"
    preview = "\n".join(f"- {item}" for item in evidence[-5:]) or "- <none>"
    emit(
        responses,
        "chain_rpc.response.case3_handoff_ready",
        arguments={"chain": chain, "evidence_preview": preview},
        payload={"handoff": handoff},
        source=__name__,
    )
