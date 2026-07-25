"""Cohesive Chain/RPC component extracted from the product domain facade."""

from __future__ import annotations

from typing import Any

from ..input_values import (
    normalize_scalar,
    parse_weight_spec_for_methods,
)
from ..state import AgentGraphState
from .response_fragments import ResponseCollector, emit

from agent.validators.rpc_workload import default_workload
from agent.validators.fixture_checks import validate_effective_fake_node_workload
from .chain_rpc_support import _case_dict, _format_weights, _invalidate_execution
from .rpc_catalog import catalog_method_names, finish_catalog, reset_draft
from .rpc_receipts import emit_workload_commit_receipt

def _apply_continue(
    state: AgentGraphState,
    case: str,
    value: Any,
    *,
    responses: ResponseCollector,
) -> None:
    case_dict = _case_dict(state, case)
    if value == "add_another":
        reset_draft(state)
        case_dict["status"] = "existing_family_needs_method" if case == "new_chain" else "needs_method"
        keys = ("candidate_method", "candidate_params") if case == "new_chain" else ("method", "params")
        for key in (
            *keys,
            "schema_draft",
            "schema_evidence",
            "schema_evidence_fragments",
            "observed_response",
            "response_confirmed",
            "schema_confirmed",
            "method_probe",
        ):
            case_dict.pop(key, None)
    else:
        transition = finish_catalog(state)
        if not transition.accepted:
            case_dict["status"] = "existing_family_needs_method" if case == "new_chain" else "needs_method"
            return
        case_dict["status"] = "existing_family_needs_workload_scope" if case == "new_chain" else "needs_scope"
    state['active_group'] = "endpoint_process"


def _apply_scope(
    state: AgentGraphState,
    case: str,
    value: Any,
    *,
    responses: ResponseCollector,
) -> None:
    case_dict = _case_dict(state, case)
    state['active_group'] = "endpoint_process"
    scope_key = "workload_scope" if case == "new_chain" else "scope"
    case_dict[scope_key] = str(value)
    methods = catalog_method_names(state)
    if value == "single_replace":
        if len(methods) > 1:
            case_dict["status"] = "existing_family_needs_single_method" if case == "new_chain" else "needs_single_method"
        elif methods:
            _set_single_workload(state, methods[0], case, responses=responses)
        return
    case_dict["status"] = "existing_family_needs_weights" if case == "new_chain" else "needs_weights"


def _set_single_workload(
    state: AgentGraphState,
    method: str,
    case: str,
    *,
    responses: ResponseCollector,
) -> None:
    case_dict = _case_dict(state, case)
    state["rpc_mode"] = "single"
    state["workload"] = {
        "confirmed": True,
        "choice": "new_chain_verified_method" if case == "new_chain" else "custom_rpc",
        "methods": [method],
        "replace_defaults": True,
        "job_local_override": True,
    }
    case_dict["status"] = _completed_case_status(state, case)
    case_dict["methods"] = [method]
    case_dict["job_local_override"] = True
    _refresh_fixture_evidence(state, case)
    emit_workload_commit_receipt(state, case=case)
    state['active_group'] = _completed_workload_group(state, case)
    _invalidate_execution(state)


def _apply_weights(
    state: AgentGraphState,
    case: str,
    value: Any,
    *,
    responses: ResponseCollector,
) -> None:
    case_dict = _case_dict(state, case)
    validated = catalog_method_names(state)
    parse_methods = validated
    if case == "custom_rpc" and not parse_methods:
        chain = normalize_scalar((state.get("chain_identity") or {}).get("canonical"))
        parse_methods = list(default_workload(chain).get("methods") or []) if chain else []
    weights = parse_weight_spec_for_methods(value, parse_methods)
    if not weights:
        case_dict["status"] = "existing_family_needs_weights" if case == "new_chain" else "needs_weights"
        emit(
            responses,
            "chain_rpc.response.weights_unrecognized",
            source=__name__,
        )
        return
    total = sum(weights.values())
    missing, unknown, invalid = _weight_contract_violations(
        state,
        case=case,
        validated=validated,
        weights=weights,
        scope=normalize_scalar(case_dict.get("workload_scope" if case == "new_chain" else "scope")),
    )
    if case == "new_chain":
        if missing or unknown or invalid or total != 100:
            case_dict.update({"status": "existing_family_needs_weights", "weights": weights})
            _emit_weight_error(
                responses,
                case_label="New-chain",
                missing=missing,
                unknown=unknown,
                invalid=invalid,
                total=total,
                weights=weights,
            )
            return
        replace_defaults = True
        choice = "new_chain_verified_methods"
    else:
        scope = normalize_scalar(case_dict.get("scope"))
        if missing or unknown or invalid or total != 100:
            case_dict.update({"status": "needs_weights", "weights": weights})
            _emit_weight_error(
                responses,
                case_label="Custom RPC",
                missing=missing,
                unknown=unknown,
                invalid=invalid,
                total=total,
                weights=weights,
            )
            return
        replace_defaults = scope == "mixed_replace"
        choice = "custom_rpc"
    case_dict.update({"status": _completed_case_status(state, case), "weights": weights, "job_local_override": True})
    if case == "custom_rpc":
        case_dict.pop("requested_workload", None)
    state["rpc_mode"] = "mixed"
    state["workload"] = {
        "confirmed": True,
        "choice": choice,
        "methods": list(weights),
        "mixed_weights": weights,
        "replace_defaults": replace_defaults,
        "job_local_override": True,
    }
    _refresh_fixture_evidence(state, case)
    emit_workload_commit_receipt(state, case=case)
    state['active_group'] = _completed_workload_group(state, case)
    _invalidate_execution(state)
    if case == "custom_rpc":
        emit(
            responses,
            (
                "chain_rpc.response.custom_mixed_workload_replaced"
                if replace_defaults
                else "chain_rpc.response.custom_mixed_workload_merged"
            ),
            arguments={"weights": _format_weights(weights)},
            payload={"replace_defaults": replace_defaults},
            source=__name__,
        )


def _apply_requested_workload(
    state: AgentGraphState,
    *,
    responses: ResponseCollector,
) -> bool:
    custom = state.setdefault("custom_rpc", {})
    request = custom.get("requested_workload")
    if not isinstance(request, dict) or not request.get("finish_methods"):
        return False
    methods = catalog_method_names(state)
    if not methods:
        return False
    finish_catalog(state)
    scope = normalize_scalar(request.get("scope"))
    if scope == "single_replace":
        custom["scope"] = scope
        custom.pop("requested_workload", None)
        if len(methods) > 1:
            custom["status"] = "needs_single_method"
            return True
        _set_single_workload(
            state,
            methods[0],
            "custom_rpc",
            responses=responses,
        )
        emit(
            responses,
            "chain_rpc.response.custom_single_workload_confirmed",
            arguments={"method": methods[0]},
            source=__name__,
        )
        return True
    if scope in {"mixed_replace", "mixed_add"} and isinstance(request.get("weights"), dict):
        raw_weights = request["weights"]
        if any(isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0 for weight in raw_weights.values()):
            custom.update({"scope": scope, "status": "needs_weights"})
            return True
        weights = {str(key): weight for key, weight in raw_weights.items()}
        missing, unknown, invalid = _weight_contract_violations(
            state,
            case="custom_rpc",
            validated=methods,
            weights=weights,
            scope=scope,
        )
        if missing or unknown or invalid or sum(weights.values()) != 100:
            custom.update({"scope": scope, "status": "needs_weights", "weights": weights})
            return True
        custom.update({"scope": scope, "status": "validated", "weights": weights, "job_local_override": True})
        custom.pop("requested_workload", None)
        state["rpc_mode"] = "mixed"
        state["workload"] = {"confirmed": True, "choice": "custom_rpc", "methods": list(weights), "mixed_weights": weights, "replace_defaults": scope == "mixed_replace", "job_local_override": True}
        _refresh_fixture_evidence(state, "custom_rpc")
        emit_workload_commit_receipt(state, case="custom_rpc")
        state['active_group'] = "target_samples_fixtures" if _fixtures_block(state) else "workload_rpc"
        _invalidate_execution(state)
        replaced = scope == "mixed_replace"
        emit(
            responses,
            (
                "chain_rpc.response.custom_mixed_workload_replaced"
                if replaced
                else "chain_rpc.response.custom_mixed_workload_merged"
            ),
            arguments={"weights": _format_weights(weights)},
            payload={"replace_defaults": replaced},
            source=__name__,
        )
        return True
    if scope in {"mixed_replace", "mixed_add"}:
        custom.update({"scope": scope, "status": "needs_weights"})
        return True
    custom["status"] = "needs_scope"
    return True


def _emit_weight_error(
    responses: ResponseCollector,
    *,
    case_label: str,
    missing: list[str],
    unknown: list[str],
    invalid: list[str],
    total: int,
    weights: dict[str, int],
) -> None:
    emit(
        responses,
        "chain_rpc.response.weights_invalid",
        arguments={
            "case_label": case_label,
            "missing": ", ".join(missing) or "<none>",
            "unknown": ", ".join(unknown) or "<none>",
            "invalid": ", ".join(invalid) or "<none>",
            "total": total,
            "weights": _format_weights(weights),
        },
        payload={
            "missing": missing,
            "unknown": unknown,
            "invalid": invalid,
            "weights": weights,
        },
        source=__name__,
    )


def _refresh_fixture_evidence(state: AgentGraphState, case: str) -> None:
    workload = state.get("workload") or {}
    if case != "custom_rpc" or state.get("target_mode") != "fake-node" or not workload.get("job_local_override"):
        return
    chain = normalize_scalar((state.get("chain_identity") or {}).get("canonical"))
    result = validate_effective_fake_node_workload(
        chain,
        normalize_scalar(state.get("rpc_mode")) or "single",
        list(workload.get("methods") or []),
        dict(workload.get("mixed_weights") or {}),
    )
    state["fixture_evidence"] = {
        "required": True,
        "status": "validated" if result.get("passed") else "missing",
        "chain": chain,
        "methods": list(result.get("methods") or []),
        "missing": list(result.get("missing") or []),
    }


def _completed_case_status(state: AgentGraphState, case: str) -> str:
    if case != "new_chain":
        return "validated"
    if state.get("target_mode") == "fake-node":
        return "existing_family_runtime_choice"
    chain = normalize_scalar(
        (state.get("chain_identity") or {}).get("canonical")
        or (state.get("chain_identity") or {}).get("raw")
    )
    if chain:
        state.setdefault("confirmed_config", {})["BLOCKCHAIN_NODE"] = chain
    state["fixture_evidence"] = {}
    return "confirmed"


def _completed_workload_group(state: AgentGraphState, case: str) -> str:
    if case == "new_chain" and state.get("target_mode") == "fake-node":
        return "target_samples_fixtures"
    if _fixtures_block(state):
        return "target_samples_fixtures"
    return "workload_rpc"


def _fixtures_block(state: AgentGraphState) -> bool:
    evidence = state.get("fixture_evidence") or {}
    return bool(evidence.get("required") and evidence.get("status") != "validated")


def _weight_contract_violations(
    state: AgentGraphState,
    *,
    case: str,
    validated: list[str],
    weights: dict[str, int],
    scope: str,
) -> tuple[list[str], list[str], list[str]]:
    """Validate the same effective workload contract for every input path."""

    invalid = [
        method
        for method, weight in weights.items()
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0
    ]
    if case == "new_chain":
        required = set(validated)
        allowed = set(validated)
    else:
        chain = normalize_scalar((state.get("chain_identity") or {}).get("canonical"))
        template = set(default_workload(chain).get("methods") or []) if chain else set()
        required = set(validated)
        if scope == "mixed_add" or not validated:
            required.update(template)
        allowed = set(validated) | template
    missing = sorted(method for method in required if method not in weights)
    unknown = sorted(method for method in weights if method not in allowed)
    return missing, unknown, invalid
