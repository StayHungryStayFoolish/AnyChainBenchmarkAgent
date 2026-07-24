"""Cohesive Chain/RPC component extracted from the product domain facade."""

from __future__ import annotations

from typing import Any

from ..input_values import (
    normalize_scalar,
    parse_weight_spec_for_methods,
)
from ..localization import localized
from ..state import AgentGraphState

from agent.validators.rpc_workload import default_workload
from agent.validators.fixture_checks import validate_effective_fake_node_workload
from .chain_rpc_support import _case_dict, _format_weights, _invalidate_execution
from .rpc_catalog import catalog_method_names, finish_catalog, reset_draft
from .rpc_receipts import emit_workload_commit_receipt

def _apply_continue(state: AgentGraphState, question_id: str, value: Any) -> None:
    case = "new_chain" if question_id.startswith("new_chain") else "custom_rpc"
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


def _apply_scope(state: AgentGraphState, question_id: str, value: Any) -> None:
    case = "new_chain" if question_id.startswith("new_chain") else "custom_rpc"
    case_dict = _case_dict(state, case)
    state['active_group'] = "endpoint_process"
    scope_key = "workload_scope" if case == "new_chain" else "scope"
    case_dict[scope_key] = str(value)
    methods = catalog_method_names(state)
    if value == "single_replace":
        if len(methods) > 1:
            case_dict["status"] = "existing_family_needs_single_method" if case == "new_chain" else "needs_single_method"
        elif methods:
            _set_single_workload(state, methods[0], case)
        return
    case_dict["status"] = "existing_family_needs_weights" if case == "new_chain" else "needs_weights"


def _set_single_workload(state: AgentGraphState, method: str, case: str) -> None:
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


def _apply_weights(state: AgentGraphState, question_id: str, value: Any) -> None:
    case = "new_chain" if question_id.startswith("new_chain") else "custom_rpc"
    case_dict = _case_dict(state, case)
    validated = catalog_method_names(state)
    parse_methods = validated
    if case == "custom_rpc" and not parse_methods:
        chain = normalize_scalar((state.get("chain_identity") or {}).get("canonical"))
        parse_methods = list(default_workload(chain).get("methods") or []) if chain else []
    weights = parse_weight_spec_for_methods(value, parse_methods)
    language = state.get("language", "en")
    if not weights:
        case_dict["status"] = "existing_family_needs_weights" if case == "new_chain" else "needs_weights"
        state['visible_response'] = [localized(language, "没有识别到唯一且有效的权重映射。请使用 JSON/YAML 映射或 `method=weight,method2=weight2`，并确保没有相互冲突的多份映射。", "No unique valid weight mapping was found. Use a JSON/YAML mapping or `method=weight,method2=weight2`, and do not provide conflicting mappings.")]
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
            details = []
            if missing:
                details.append(localized(language, f"缺少已验证 method：{', '.join(missing)}", f"Missing validated method: {', '.join(missing)}"))
            if unknown:
                details.append(localized(language, f"包含未验证 method：{', '.join(unknown)}", f"Contains unverified method: {', '.join(unknown)}"))
            if invalid:
                details.append(localized(language, f"权重必须是正整数：{', '.join(invalid)}", f"Weights must be positive integers: {', '.join(invalid)}"))
            if total != 100:
                details.append(localized(language, f"权重总和为 {total}，必须等于 100", f"Weight total is {total}; it must equal 100"))
            state['visible_response'] = [localized(language, f"新链 mixed 权重需要调整：{'; '.join(details)}。当前配置：{_format_weights(weights)}。请重新输入。", f"New-chain mixed weights need adjustment: {'; '.join(details)}. Current weights: {_format_weights(weights)}. Enter the weights again.")]
            return
        replace_defaults = True
        choice = "new_chain_verified_methods"
    else:
        scope = normalize_scalar(case_dict.get("scope"))
        if missing or unknown or invalid or total != 100:
            case_dict.update({"status": "needs_weights", "weights": weights})
            details = []
            if missing:
                details.append(localized(language, f"缺少 workload method：{', '.join(sorted(missing))}", f"Missing workload method: {', '.join(sorted(missing))}"))
            if unknown:
                details.append(localized(language, f"未知 method（既不是已验证自定义 method，也不是模板默认 method）：{', '.join(unknown)}", f"Unknown method (neither a validated custom method nor a template default): {', '.join(unknown)}"))
            if invalid:
                details.append(localized(language, f"权重必须是正整数：{', '.join(invalid)}", f"Weights must be positive integers: {', '.join(invalid)}"))
            if total != 100:
                details.append(localized(language, f"权重总和为 {total}，必须等于 100", f"Weight total is {total}; it must equal 100"))
            state['visible_response'] = [localized(language, f"自定义 RPC mixed 权重需要调整：{'; '.join(details)}。当前配置：{_format_weights(weights)}。请重新输入。", f"Custom RPC mixed weights need adjustment: {'; '.join(details)}. Current weights: {_format_weights(weights)}. Enter the weights again.")]
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
        state['visible_response'] = list(state.get("visible_response") or []) + [localized(language, f"自定义 RPC mixed workload 已确认：{_format_weights(weights)}；模板默认 methods {'会被替换' if replace_defaults else '会被保留并追加'}。", f"Custom RPC mixed workload confirmed: {_format_weights(weights)}; template default methods will be {'replaced' if replace_defaults else 'kept and appended'}.")]


def _apply_requested_workload(state: AgentGraphState) -> bool:
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
        _set_single_workload(state, methods[0], "custom_rpc")
        state['visible_response'] = list(state.get("visible_response") or []) + [localized(state.get("language", "en"), f"自定义 RPC single workload 已确认：{methods[0]}；模板默认 method 会被替换。", f"Custom RPC single workload confirmed: {methods[0]}; the template default method will be replaced.")]
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
        state['visible_response'] = list(state.get("visible_response") or []) + [localized(state.get("language", "en"), f"自定义 RPC mixed workload 已确认：{_format_weights(weights)}；模板默认 methods {'会被替换' if replaced else '已合并到最终权重'}。", f"Custom RPC mixed workload confirmed: {_format_weights(weights)}; template default methods were {'replaced' if replaced else 'merged into the final weights'}.")]
        return True
    if scope in {"mixed_replace", "mixed_add"}:
        custom.update({"scope": scope, "status": "needs_weights"})
        return True
    custom["status"] = "needs_scope"
    return True


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
