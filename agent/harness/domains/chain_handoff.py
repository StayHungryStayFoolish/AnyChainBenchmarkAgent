"""Cohesive Chain/RPC component extracted from the product domain facade."""

from __future__ import annotations

from copy import deepcopy

from ..input_values import normalize_scalar
from ..secret_refs import materialize_state_secret_references
from ..state import AgentGraphState
from ..transitions import mark_group_reconfigured, record_group_invalidations
from .chain_rpc_questions import _case3_evidence_question
from .response_fragments import ResponseCollector, emit
from .chain_rpc_support import _next_group
from .rpc_catalog import (
    catalog_method_names,
    draft_view,
    probe_evidence_content_hash,
    probe_evidence_contract_hash,
    refresh_catalog_projection,
    validated_contracts,
    validated_method_contract_is_current,
    validated_contracts_view,
)
from .rpc_receipts import (
    emit_endpoint_role_receipt,
    emit_workload_commit_receipt,
    evidence_hash,
    exact_value_hash,
    validate_rpc_receipt,
)

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
    catalog_methods = catalog_method_names(state)
    methods = [
        normalize_scalar(item.get("method"))
        for item in validated_contracts_view(state)
        if _validated_method_matches_endpoint(
            state,
            item,
            endpoint=endpoint,
        )
    ]
    methods = list(dict.fromkeys(methods))
    method = methods[0] if len(methods) == 1 else ""
    endpoint_probe = (
        evidence.get("new_chain_endpoint_probe")
        if isinstance(evidence.get("new_chain_endpoint_probe"), dict)
        else {}
    )
    try:
        materialized_endpoint = normalize_scalar(
            materialize_state_secret_references(endpoint, state)
        )
        materialized_probe_endpoint = normalize_scalar(
            materialize_state_secret_references(
                endpoint_probe.get("endpoint"),
                state,
            )
        )
    except (KeyError, RuntimeError, ValueError):
        materialized_endpoint = ""
        materialized_probe_endpoint = ""
    validation_receipt = (
        evidence.get("candidate_endpoint_validation_receipt")
        if isinstance(
            evidence.get("candidate_endpoint_validation_receipt"),
            dict,
        )
        else {}
    )
    validation_receipt_valid = bool(
        validation_receipt
        and validate_rpc_receipt(validation_receipt)[0]
        and validation_receipt.get("receipt_id")
        == evidence.get("candidate_endpoint_validation_receipt_id")
        and validation_receipt.get("role") == "validation"
        and validation_receipt.get("case") == "new_chain"
        and validation_receipt.get("ready") is True
        and normalize_scalar(validation_receipt.get("chain")) == chain
        and normalize_scalar(
            validation_receipt.get("adapter_family")
        )
        == normalize_scalar(identity.get("adapter_family"))
        and materialized_endpoint
        and validation_receipt.get("endpoint_hash")
        == evidence_hash(materialized_endpoint)
        and validation_receipt.get("source_value_hash")
        == exact_value_hash(endpoint)
    )
    probe_matches = bool(
        endpoint
        and validation_receipt_valid
        and endpoint_probe.get("ready") is True
        and materialized_probe_endpoint == materialized_endpoint
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
    if set(methods) != set(catalog_methods):
        identity["status"] = "existing_family_needs_method"
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
        method_contracts = [
            item
            for item in validated_contracts(state)
            if normalize_scalar(item.get("method")) in methods
        ]
        runtime_receipt = emit_endpoint_role_receipt(
            state,
            role="final_benchmark",
            case="runtime",
            config_field="LOCAL_RPC_URL",
            endpoint=materialized_endpoint,
            source_kind="validated_endpoint",
            source_receipt_id=str(
                validation_receipt.get("receipt_id") or ""
            ),
            source_value_hash=str(
                validation_receipt.get("source_value_hash") or ""
            ),
            ready=True,
            probe_status=endpoint_probe.get("status"),
            chain=chain,
            adapter_family=identity.get("adapter_family"),
            methods=methods,
            method_evidence_bindings=[
                {
                    "method": item.get("method"),
                    "evidence_file_hash": probe_evidence_content_hash(
                        normalize_scalar(item.get("evidence_file"))
                    ),
                    "probe_contract_hash": probe_evidence_contract_hash(
                        normalize_scalar(item.get("evidence_file"))
                    ),
                }
                for item in method_contracts
            ],
        )
        if not runtime_receipt:
            identity["status"] = "existing_family_needs_method"
            emit(
                responses,
                "chain_rpc.response.case2_promotion_blocked",
                source=__name__,
            )
            return False
        evidence["local_rpc_url_validation_receipt_id"] = (
            runtime_receipt["receipt_id"]
        )
        evidence["local_rpc_url_validation_receipt"] = deepcopy(
            runtime_receipt
        )
        for item in method_contracts:
            item["final_endpoint"] = endpoint
            item["final_endpoint_evidence_file"] = normalize_scalar(
                item.get("evidence_file")
            )
        refresh_catalog_projection(state)
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


def _validated_method_matches_endpoint(
    state: AgentGraphState,
    contract: dict,
    *,
    endpoint: str,
) -> bool:
    return validated_method_contract_is_current(
        state,
        contract,
        endpoint=endpoint,
    )


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
