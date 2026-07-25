"""Cohesive Chain/RPC component extracted from the product domain facade."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..failures import build_failure_record
from ..input_values import (
    extract_json_object_or_array,
    extract_json_values,
    has_rpc_response_evidence,
    extract_rpc_params_or_request,
    extract_url_candidate,
    looks_like_url_value,
    normalize_scalar,
)
from ..advisory import extract_rpc_schema_from_evidence
from ..state import AgentGraphState
from ..transitions import (
    invalidate_for_endpoint_change,
    invalidate_rpc_catalog_endpoint_evidence,
    record_group_invalidations,
)

from agent.llm.search_grounding import run_google_search_grounding
from agent.validators.endpoint_probe import (
    GENERIC_JSONRPC_PROBE_FAMILIES,
    health_probe_methods,
    validate_rpc_endpoint,
)
from .chain_rpc_questions import (
    _adapter_family_question,
    _continue_question,
    _catalog_confirmation_question,
    _response_confirmation_question,
)
from .chain_rpc_support import _adapter_family, _case_dict, _draft_has_params, _looks_like_rest_method, _merge_rpc_schema_draft, _schema_indicates_jsonrpc, _schema_indicates_rest
from .rpc_workload import _apply_requested_workload
from .response_fragments import ResponseCollector, emit
from .rpc_catalog import (
    add_validated_method,
    append_evidence,
    confirm_parameter,
    confirm_request,
    confirm_response,
    correct_draft,
    draft_view,
    ensure_catalog,
    incomplete_parameters,
    next_parameter_to_confirm,
    record_probe,
    refresh_catalog_projection,
    reset_draft,
    strict_method_identity,
    validated_contracts,
)
from .rpc_receipts import (
    emit_endpoint_role_receipt,
    evidence_hash,
)

def _apply_endpoint_answer(
    state: AgentGraphState,
    endpoint_contract: Mapping[str, Any],
    value: Any,
    *,
    responses: ResponseCollector,
) -> None:
    endpoint = extract_url_candidate(value) or normalize_scalar(value)
    identity = state.setdefault("chain_identity", {})
    chain = normalize_scalar(identity.get("canonical") or identity.get("raw"))
    family = _adapter_family(state)
    confirmed = state.setdefault("confirmed_config", {})
    role = normalize_scalar(endpoint_contract.get("endpoint_role"))
    case = normalize_scalar(endpoint_contract.get("rpc_case"))
    config_field = normalize_scalar(endpoint_contract.get("config_field"))
    if (role, case, config_field) not in {
        ("final_benchmark", "runtime", "LOCAL_RPC_URL"),
        ("sync_observe", "runtime", "SYNC_OBSERVE_RPC_URL"),
        ("validation", "custom_rpc", ""),
        ("validation", "new_chain", ""),
    }:
        raise ValueError("invalid typed RPC endpoint contract")
    previous_endpoint = normalize_scalar(
        confirmed.get(config_field)
        if role in {"final_benchmark", "sync_observe"}
        else (state.get("custom_rpc") or {}).get("endpoint")
        if case == "custom_rpc"
        else (state.get("endpoint_evidence") or {}).get("candidate_endpoint")
    )
    if role == "final_benchmark":
        contracts = _selected_custom_contracts(state)
        if contracts:
            result = _validate_final_endpoint_contracts(
                chain=chain,
                endpoint=endpoint,
                family=family,
                contracts=contracts,
            )
            methods = None
            params = None
        else:
            methods, params = health_probe_methods(chain, family)
    elif role == "sync_observe":
        methods = None
        params = None
    else:
        methods, params = health_probe_methods(chain, family)
    if not (role == "final_benchmark" and contracts):
        result = validate_rpc_endpoint(
            chain=chain,
            endpoint=endpoint,
            methods=methods,
            adapter_family=family,
            method_params=params,
            timeout=3.0,
        )
    emit_endpoint_role_receipt(
        state,
        role=role,
        case=case,
        endpoint=endpoint,
        previous_endpoint=previous_endpoint,
        ready=bool(result.get("ready")),
        probe_status=result.get("status"),
        chain=chain,
        adapter_family=family,
        methods=(
            [contract.get("method") for contract in contracts]
            if role == "final_benchmark" and contracts
            else methods or ()
        ),
    )
    evidence = state.setdefault("endpoint_evidence", {})
    if role == "final_benchmark":
        evidence["local_rpc_url_probe"] = result
        if contracts:
            evidence["final_custom_rpc_probe"] = result
        if not result.get("ready"):
            evidence["local_rpc_url_ready"] = False
            _record_validation_failure(state, "ENDPOINT_UNREACHABLE", result)
            _emit_endpoint_result(responses, "LOCAL_RPC_URL", result, ready=False)
            return
        evidence["local_rpc_url_ready"] = True
        if contracts:
            for contract in contracts:
                method_result = (result.get("method_results") or {}).get(contract["method"], {})
                contract["final_endpoint"] = endpoint
                contract["final_endpoint_evidence_file"] = str(method_result.get("evidence_file") or "")
            refresh_catalog_projection(state)
        evidence.pop("last_failure_record", None)
        state.setdefault("confirmed_config", {})[config_field] = endpoint
        if endpoint != previous_endpoint:
            record_group_invalidations(state, "endpoint_process")
        _emit_endpoint_result(responses, "LOCAL_RPC_URL", result, ready=True)
        return
    if role == "sync_observe":
        evidence["sync_rpc_url_probe"] = result
        evidence["sync_rpc_url_ready"] = bool(result.get("ready"))
        if not result.get("ready"):
            _record_validation_failure(state, "ENDPOINT_UNREACHABLE", result)
            _emit_endpoint_result(responses, "sync-observe endpoint", result, ready=False)
            return
        state.setdefault("confirmed_config", {})[config_field] = endpoint
        if endpoint != previous_endpoint:
            record_group_invalidations(state, "endpoint_process")
        evidence.pop("last_failure_record", None)
        _emit_endpoint_result(responses, "sync-observe endpoint", result, ready=True)
        return
    if case == "custom_rpc":
        custom = state.setdefault("custom_rpc", {})
        previous_endpoint = normalize_scalar(custom.get("endpoint"))
        if previous_endpoint and previous_endpoint != endpoint:
            invalidate_for_endpoint_change(state, endpoint, role="validation")
            _reset_in_progress_method_evidence(state, case="custom_rpc")
        custom.update({"endpoint": endpoint, "endpoint_probe": result, "endpoint_ready": bool(result.get("ready")), "job_local_override": True})
        evidence["custom_rpc_endpoint_probe"] = result
        if not result.get("ready"):
            custom["status"] = "probe_failed"
            _record_validation_failure(state, "ENDPOINT_UNREACHABLE", result)
            _emit_endpoint_result(responses, "validation endpoint", result, ready=False)
            return
        ensure_catalog(state)
        if endpoint != previous_endpoint:
            record_group_invalidations(state, "endpoint_process")
        custom["status"] = "needs_schema_evidence" if draft_view(state).get("method") else "needs_method"
        evidence.pop("last_failure_record", None)
        emit(
            responses,
            "chain_rpc.response.custom_endpoint_validation_passed",
            arguments={"evidence": str(result.get("evidence_file") or "<none>")},
            source=__name__,
        )
        if extract_json_object_or_array(value) and extract_rpc_params_or_request(value)[1] is not None:
            _apply_method_answer(state, "custom_rpc", value, responses=responses)
        return
    previous_endpoint = normalize_scalar(evidence.get("candidate_endpoint"))
    if previous_endpoint and previous_endpoint != endpoint:
        _invalidate_endpoint_bound_methods(state, identity, endpoint)
    evidence.update({"candidate_endpoint": endpoint, "new_chain_endpoint_probe": result, "candidate_endpoint_ready": bool(result.get("ready"))})
    if not result.get("ready"):
        identity["status"] = "existing_family_needs_endpoint"
        _record_validation_failure(state, "ENDPOINT_UNREACHABLE", result)
        _emit_endpoint_result(responses, "new-chain endpoint", result, ready=False)
        return
    identity["status"] = "existing_family_needs_method"
    if endpoint != previous_endpoint:
        record_group_invalidations(state, "endpoint_process")
    ensure_catalog(state)
    evidence.pop("last_failure_record", None)
    _emit_endpoint_result(responses, "new-chain endpoint", result, ready=True)
    if extract_json_object_or_array(value) and extract_rpc_params_or_request(value)[1] is not None:
        _apply_method_answer(state, "new_chain", value, responses=responses)


def _emit_endpoint_result(
    responses: ResponseCollector,
    endpoint_role: str,
    result: Mapping[str, Any],
    *,
    ready: bool,
) -> None:
    arguments: dict[str, str] = {
        "endpoint_role": endpoint_role,
        "evidence": str(result.get("evidence_file") or "<none>"),
    }
    if not ready:
        arguments["reason"] = str(
            result.get("error") or result.get("status") or "validation failed"
        )
    emit(
        responses,
        (
            "chain_rpc.response.endpoint_validation_passed"
            if ready
            else "chain_rpc.response.endpoint_validation_failed"
        ),
        arguments=arguments,
        source=__name__,
    )


def _apply_method_answer(
    state: AgentGraphState,
    case: str,
    value: Any,
    *,
    responses: ResponseCollector,
) -> None:
    case_dict = _case_dict(state, case)
    raw = str(value or "")
    method_value = normalize_scalar(raw)
    parsed_method, parsed = extract_rpc_params_or_request(raw)
    next_method = strict_method_identity(
        parsed_method or method_value,
        adapter_family=_adapter_family(state),
        from_protocol_request=bool(parsed_method and parsed is not None),
    )
    if not next_method:
        case_dict["status"] = "existing_family_needs_method" if case == "new_chain" else "needs_method"
        emit(responses, "chain_rpc.response.method_identity_required", source=__name__)
        return
    current_method = normalize_scalar(draft_view(state).get("method"))
    if current_method and current_method != next_method:
        transition = append_evidence(
            state,
            content=raw,
            source="user",
            kind="protocol_request" if parsed is not None else "method_identity",
            method=next_method,
        )
        if not transition.accepted:
            case_dict["status"] = "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"
            emit(
                responses,
                "chain_rpc.response.rpc_method_conflict",
                arguments={
                    "incoming_method": next_method,
                    "current_method": current_method,
                },
                source=__name__,
            )
            return
    elif not current_method:
        append_evidence(
            state,
            content=raw,
            source="user",
            kind="protocol_request" if parsed is not None else "method_identity",
            method=next_method,
        )
    family_is_rest_shaped = _adapter_family(state) not in GENERIC_JSONRPC_PROBE_FAMILIES
    if not (parsed_method and parsed is not None) and (
        (looks_like_url_value(method_value) and not family_is_rest_shaped)
        or (_looks_like_rest_method(method_value) and not family_is_rest_shaped)
    ):
        case_dict["status"] = "existing_family_needs_method" if case == "new_chain" else "needs_method"
        emit(responses, "chain_rpc.response.method_identity_required", source=__name__)
        return
    if parsed is not None:
        _apply_schema_evidence(
            state,
            case=case,
            evidence=raw,
            responses=responses,
        )
        return
    case_dict["status"] = "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"


def _apply_schema_evidence(
    state: AgentGraphState,
    *,
    case: str,
    evidence: str,
    responses: ResponseCollector,
) -> bool:
    case_dict = _case_dict(state, case)
    clean_evidence = str(evidence or "").strip()
    incoming_method, incoming_params = extract_rpc_params_or_request(clean_evidence)
    incoming_method = strict_method_identity(
        incoming_method,
        adapter_family=_adapter_family(state),
        from_protocol_request=bool(incoming_method and incoming_params is not None),
    )
    direct_structured_fact = incoming_params is not None or any(
        isinstance(item, dict) and ("result" in item or "error" in item)
        for item in extract_json_values(clean_evidence)
    )
    fragment_draft = {} if direct_structured_fact else dict(
        extract_rpc_schema_from_evidence(state, clean_evidence, method_hint="") or {}
    )
    if not _fragment_contributes_rpc_fact(
        clean_evidence,
        incoming_method=incoming_method,
        incoming_params=incoming_params,
        extracted=fragment_draft,
    ):
        emit(responses, "chain_rpc.response.rpc_evidence_rejected", source=__name__)
        return False
    current_method = normalize_scalar(draft_view(state).get("method"))
    transition = append_evidence(
        state,
        content=clean_evidence,
        source="user",
        kind="protocol_request" if incoming_params is not None else "schema_evidence",
        method=incoming_method or current_method,
    )
    if not transition.accepted:
        case_dict["status"] = "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"
        emit(
            responses,
            "chain_rpc.response.rpc_method_conflict",
            arguments={
                "incoming_method": incoming_method,
                "current_method": current_method,
            },
            source=__name__,
        )
        return False
    fragments = [
        str(item.get("content") or "")
        for item in draft_view(state).get("evidence") or []
        if isinstance(item, dict) and str(item.get("content") or "").strip()
    ]
    combined_evidence = "\n\n".join(fragments)
    parsed_method, parsed = extract_rpc_params_or_request(combined_evidence)
    if parsed is not None:
        case_dict["schema_evidence"] = combined_evidence
        method = parsed_method or normalize_scalar(draft_view(state).get("method"))
        extracted = dict(extract_rpc_schema_from_evidence(state, combined_evidence, method_hint=method) or {})
        if extracted.get("status") == "failed":
            extracted = {}
        if _request_only_schema_evidence(fragments, method):
            # Model knowledge may explain a known method, but it cannot turn a
            # request-only paste into user-supplied response evidence. The
            # endpoint probe will produce an observed response that is reviewed
            # through its own confirmation contract.
            extracted["response_summary"] = "unknown"
            extracted["response_json_type"] = "unknown"
            extracted["response_fields"] = []
            extracted.pop("response_sample", None)
            extracted.pop("response_example", None)
        previous_draft = draft_view(state)
        draft = _merge_rpc_schema_draft(
            method=method,
            params_json=parsed,
            extracted=extracted,
            previous=previous_draft,
        )
        draft["validation_endpoint"] = _validation_endpoint(state, case)
        draft["field_provenance"] = _schema_field_provenance(
            draft,
            previous=previous_draft,
            extracted=extracted,
            evidence_fragments=[
                item
                for item in previous_draft.get("evidence") or ()
                if isinstance(item, Mapping)
            ],
            wire_request_observed=True,
        )
        correct_draft(state, draft)
        case_dict["status"] = "existing_family_schema_needs_confirmation" if case == "new_chain" else "schema_needs_confirmation"
        if incomplete_parameters(state):
            case_dict["status"] = "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"
            emit(
                responses,
                "chain_rpc.response.parameter_semantics_missing",
                arguments={"details": _parameter_semantics_missing_details(state)},
                source=__name__,
            )
            return True
        state['pending_question'] = _catalog_confirmation_question(state, case)
        return True
    draft = dict(extract_rpc_schema_from_evidence(state, combined_evidence, method_hint=normalize_scalar(draft_view(state).get("method"))) or {})
    if (state.get("web_research") or {}).get("google_search_available"):
        chain = normalize_scalar((state.get("chain_identity") or {}).get("canonical") or (state.get("chain_identity") or {}).get("raw"))
        method = normalize_scalar(draft.get("method") or draft_view(state).get("method"))
        result = run_google_search_grounding(f"{chain} RPC method {method}: official parameters, request/response schema, documentation")
        draft["search_result"] = result.as_dict()
        if result.available and result.text_summary:
            draft["evidence_summary"] = result.text_summary
            emit(
                responses,
                "chain_rpc.response.search_grounding",
                arguments={
                    "query": result.query,
                    "summary": result.text_summary,
                    "citations": ", ".join(result.citations) or "<none>",
                },
                kind="evidence",
                source=__name__,
            )
    case_dict["schema_evidence"] = combined_evidence
    previous = draft_view(state)
    if previous:
        draft = {**previous, **draft}
    draft["validation_endpoint"] = _validation_endpoint(state, case)
    draft["field_provenance"] = _schema_field_provenance(
        draft,
        previous=previous,
        extracted={
            key: value
            for key, value in draft.items()
            if key not in previous or previous.get(key) != value
        },
        evidence_fragments=[
            item
            for item in previous.get("evidence") or ()
            if isinstance(item, Mapping)
        ],
        wire_request_observed=False,
    )
    correct_draft(state, draft)
    if _schema_conflict(state, draft, case=case, responses=responses):
        return True
    if not _draft_has_params(draft):
        case_dict["status"] = "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"
        emit(responses, "chain_rpc.response.params_evidence_required", source=__name__)
        return True
    if incomplete_parameters(state):
        case_dict["status"] = "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"
        emit(
            responses,
            "chain_rpc.response.parameter_semantics_missing",
            arguments={"details": _parameter_semantics_missing_details(state)},
            source=__name__,
        )
        return True
    case_dict["status"] = "existing_family_schema_needs_confirmation" if case == "new_chain" else "schema_needs_confirmation"
    state['pending_question'] = _catalog_confirmation_question(state, case)
    return True


def _fragment_contributes_rpc_fact(
    source: str,
    *,
    incoming_method: str,
    incoming_params: Any,
    extracted: Mapping[str, Any],
) -> bool:
    """Verify that one fragment contributes facts independent of prior state."""

    if incoming_params is not None:
        return True
    if any(
        isinstance(item, dict) and ("result" in item or "error" in item)
        for item in extract_json_values(source)
    ):
        return True
    if str(extracted.get("status") or "").strip().lower() == "failed":
        return False
    kind = str(extracted.get("evidence_kind") or "").strip().lower()
    if kind in {"", "unknown"}:
        return False
    method = normalize_scalar(extracted.get("method"))
    if method and method in source:
        return True
    params = extracted.get("params")
    if isinstance(params, list) and params:
        return True
    if "params_json" in extracted and extracted.get("params_json") is not None:
        return True
    response_fields = extracted.get("response_fields")
    if isinstance(response_fields, list) and response_fields:
        return True
    response_summary = normalize_scalar(extracted.get("response_summary")).casefold()
    if response_summary not in {"", "unknown", "unavailable", "<unknown>"}:
        return True
    for key in ("endpoint_url", "rest_path"):
        value = normalize_scalar(extracted.get(key))
        if value and value in source:
            return True
    return False


def _schema_field_provenance(
    draft: Mapping[str, Any],
    *,
    previous: Mapping[str, Any],
    extracted: Mapping[str, Any],
    evidence_fragments: list[Mapping[str, Any]],
    wire_request_observed: bool,
) -> list[dict[str, Any]]:
    """Describe where schema fields came from without retaining their values."""

    prior = {
        normalize_scalar(item.get("field_path")): dict(item)
        for item in previous.get("field_provenance") or ()
        if isinstance(item, Mapping) and normalize_scalar(item.get("field_path"))
    }
    revisions = sorted(
        {
            int(item.get("revision"))
            for item in evidence_fragments
            if isinstance(item.get("revision"), int) and int(item.get("revision")) >= 0
        }
    )
    output: dict[str, dict[str, Any]] = {}

    def record(path: str, value: Any, source_kind: str) -> None:
        existing = prior.get(path)
        value_digest = evidence_hash(value)
        if existing and existing.get("value_hash") == value_digest and source_kind == "retained":
            output[path] = existing
            return
        output[path] = {
            "field_path": path,
            "source_kind": source_kind,
            "source_revisions": revisions,
            "value_hash": value_digest,
        }

    if normalize_scalar(draft.get("method")):
        record(
            "method",
            draft.get("method"),
            "protocol_request_parser"
            if wire_request_observed
            else "model_extraction_from_user_evidence"
            if "method" in extracted
            else "retained",
        )
    if "transport" in draft:
        record(
            "transport",
            draft.get("transport"),
            "protocol_request_parser"
            if wire_request_observed
            else "model_extraction_from_user_evidence"
            if "transport" in extracted
            else "retained",
        )
    if "params_json" in draft:
        record(
            "params_json",
            draft.get("params_json"),
            "protocol_request_parser"
            if wire_request_observed
            else "model_extraction_from_user_evidence"
            if "params_json" in extracted
            else "retained",
        )
    params = draft.get("params") if isinstance(draft.get("params"), list) else []
    extracted_params = extracted.get("params") if isinstance(extracted.get("params"), list) else []
    semantic_fields = {"semantic_type", "encoding", "meaning", "required"}
    for index, parameter in enumerate(params):
        if not isinstance(parameter, Mapping):
            continue
        extracted_parameter = (
            extracted_params[index]
            if index < len(extracted_params) and isinstance(extracted_params[index], Mapping)
            else {}
        )
        for field, value in sorted(parameter.items()):
            path = f"params[{index}].{field}"
            if wire_request_observed and (
                field in {"index", "json_type", "example"}
                or field == "name" and isinstance(draft.get("params_json"), dict)
            ):
                source = "protocol_request_parser"
            elif field in semantic_fields or field in extracted_parameter:
                source = "model_extraction_from_user_evidence"
            else:
                source = "retained"
            record(path, value, source)
    for field in (
        "response_summary",
        "response_json_type",
        "response_fields",
        "response_sample",
        "response_example",
    ):
        if field not in draft:
            continue
        source = (
            "model_extraction_from_user_evidence"
            if field in extracted
            else "retained"
        )
        record(field, draft.get(field), source)
    if normalize_scalar(draft.get("validation_endpoint")):
        record(
            "validation_endpoint",
            draft.get("validation_endpoint"),
            "rpc_endpoint_role",
        )
    return [output[path] for path in sorted(output)]


def _request_only_schema_evidence(fragments: list[str], method: str) -> bool:
    """Return whether every fragment is only method identity or a request."""

    saw_request = False
    canonical_method = normalize_scalar(method)
    for fragment in fragments:
        clean = str(fragment or "").strip()
        if not clean:
            continue
        if canonical_method and clean == canonical_method:
            continue
        if has_rpc_response_evidence(clean):
            return False
        parsed_method, params = extract_rpc_params_or_request(clean)
        if params is not None and (not canonical_method or not parsed_method or parsed_method == canonical_method):
            saw_request = True
            continue
        return False
    return saw_request


def _probe_schema(
    state: AgentGraphState,
    case: str,
    params: Any,
    *,
    responses: ResponseCollector,
) -> None:
    case_dict = _case_dict(state, case)
    draft = draft_view(state)
    method = normalize_scalar(draft.get("method"))
    if not draft.get("request_confirmed"):
        case_dict["status"] = "existing_family_schema_needs_confirmation" if case == "new_chain" else "schema_needs_confirmation"
        emit(
            responses,
            "chain_rpc.response.request_confirmation_required",
            source=__name__,
        )
        return
    params = draft.get("params_json")
    endpoint = normalize_scalar((state.get("endpoint_evidence") or {}).get("candidate_endpoint")) if case == "new_chain" else normalize_scalar(case_dict.get("endpoint"))
    result = validate_rpc_endpoint(
        chain=normalize_scalar((state.get("chain_identity") or {}).get("canonical") or (state.get("chain_identity") or {}).get("raw")),
        endpoint=endpoint,
        methods=[method],
        adapter_family=_adapter_family(state),
        method_params={method: params},
        timeout=3.0,
    )
    if case == "custom_rpc":
        case_dict["method_probe"] = result
    evidence_key = "new_chain_method_probe" if case == "new_chain" else "custom_rpc_method_probe"
    state.setdefault("endpoint_evidence", {})[evidence_key] = result
    if not result.get("ready"):
        record_probe(state, result, {})
        case_dict["status"] = "existing_family_schema_needs_confirmation" if case == "new_chain" else "probe_failed"
        _record_validation_failure(state, "RPC_METHOD_OR_SCHEMA_INVALID", result)
        emit(
            responses,
            "chain_rpc.response.method_probe_failed",
            arguments={
                "case_label": "Case 2 " if case == "new_chain" else "",
                "reason": str(result.get("error") or result.get("status") or "failed"),
                "evidence": str(result.get("evidence_file") or "<none>"),
            },
            source=__name__,
        )
        return
    selected_check = next(
        (
            item
            for item in result.get("checks") or []
            if isinstance(item, dict) and str(item.get("name") or "").startswith("method_probe:")
        ),
        {},
    )
    observed_response = {
        "shape_hash": str(selected_check.get("response_shape_hash") or result.get("response_shape_hash") or ""),
        "sample": str(selected_check.get("response_sample") or ""),
        "http_status": selected_check.get("http_status") or result.get("http_status"),
        "evidence_file": str(result.get("evidence_file") or ""),
    }
    record_probe(state, result, observed_response)
    state.setdefault("endpoint_evidence", {}).pop("last_failure_record", None)
    draft = draft_view(state)
    expected_response = normalize_scalar(draft.get("response_summary")).casefold()
    response_fields = draft.get("response_fields") if isinstance(draft.get("response_fields"), list) else []
    conflicts = draft.get("conflicts") if isinstance(draft.get("conflicts"), list) else []
    identity = result.get("identity") if isinstance(result.get("identity"), dict) else {}
    response_conflicts = _response_contract_conflicts(
        draft,
        observed_response.get("sample", ""),
        stable_result=normalize_scalar(identity.get("observed"))
        if normalize_scalar(identity.get("method")) == method
        else "",
    )
    if response_conflicts:
        conflicts = list(dict.fromkeys([*conflicts, *response_conflicts]))
        draft["conflicts"] = conflicts
    response_unknown = expected_response in {"", "unknown", "<unknown>", "unavailable"} and not response_fields
    if response_unknown or conflicts:
        case_dict["status"] = "existing_family_response_needs_confirmation" if case == "new_chain" else "response_needs_confirmation"
        state['pending_question'] = _response_confirmation_question(state, case)
        return
    _finalize_probed_method(state, case, responses=responses)


def _confirm_probe_response(
    state: AgentGraphState,
    case: str,
    accepted: bool,
    *,
    responses: ResponseCollector,
) -> None:
    case_dict = _case_dict(state, case)
    if not accepted:
        confirm_response(state, False, observed=bool(draft_view(state).get("observed_response")))
        case_dict["status"] = "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"
        emit(
            responses,
            "chain_rpc.response.response_evidence_required",
            source=__name__,
        )
        return
    observed = bool(draft_view(state).get("observed_response"))
    transition = confirm_response(state, True, observed=observed)
    if not transition.accepted:
        case_dict["status"] = "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"
        return
    if not observed:
        case_dict["status"] = "existing_family_schema_needs_confirmation" if case == "new_chain" else "schema_needs_confirmation"
        return
    _finalize_probed_method(state, case, responses=responses)


def _confirm_parameter_contract(state: AgentGraphState, case: str, accepted: bool) -> None:
    index = next_parameter_to_confirm(state)
    if index is None:
        return
    transition = confirm_parameter(state, index, accepted)
    case_dict = _case_dict(state, case)
    case_dict["status"] = (
        "existing_family_schema_needs_confirmation" if case == "new_chain" else "schema_needs_confirmation"
    ) if transition.accepted else (
        "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"
    )


def _confirm_request_contract(state: AgentGraphState, case: str, accepted: bool) -> None:
    transition = confirm_request(state, accepted)
    case_dict = _case_dict(state, case)
    if not transition.accepted:
        case_dict["schema_confirmed"] = False
        case_dict["status"] = "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"
        return
    case_dict["schema_confirmed"] = True
    if transition.phase == "response_confirmation":
        case_dict["status"] = "existing_family_response_needs_confirmation" if case == "new_chain" else "response_needs_confirmation"
    else:
        case_dict["status"] = "existing_family_schema_needs_confirmation" if case == "new_chain" else "schema_needs_confirmation"


def _confirm_method_probe(
    state: AgentGraphState,
    case: str,
    accepted: bool,
    *,
    responses: ResponseCollector,
) -> None:
    case_dict = _case_dict(state, case)
    if not accepted:
        draft = draft_view(state)
        draft["phase"] = "evidence"
        correct_draft(state, draft)
        case_dict["status"] = "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"
        return
    _probe_schema(
        state,
        case,
        draft_view(state).get("params_json"),
        responses=responses,
    )


def _finalize_probed_method(
    state: AgentGraphState,
    case: str,
    *,
    responses: ResponseCollector,
) -> None:
    case_dict = _case_dict(state, case)
    draft = draft_view(state)
    method = normalize_scalar(draft.get("method"))
    params = draft.get("params_json")
    evidence_key = "new_chain_method_probe" if case == "new_chain" else "custom_rpc_method_probe"
    result = (state.get("endpoint_evidence") or {}).get(evidence_key) or {}
    contract = {
        "method": method,
        "params": params,
        "schema": dict(draft),
        "observed_response": dict(draft.get("observed_response") or {}),
        "validation_endpoint": (
            normalize_scalar((state.get("endpoint_evidence") or {}).get("candidate_endpoint"))
            if case == "new_chain"
            else normalize_scalar(case_dict.get("endpoint"))
        ),
        "evidence_file": result.get("evidence_file") or "",
    }
    transition = add_validated_method(state, contract)
    if not transition.accepted:
        case_dict["status"] = "existing_family_response_needs_confirmation" if case == "new_chain" else "response_needs_confirmation"
        return
    if case == "custom_rpc":
        # The originating typed action may already contain the user's complete
        # workload decision. Validation is the final missing gate in that
        # transaction, so honor the persisted request before asking the generic
        # "add another" question. The workload helper installs a precise
        # disambiguation/weight state when more input is still required.
        if _apply_requested_workload(state, responses=responses):
            return
        case_dict["status"] = "method_validated_next"
        emit(
            responses,
            "chain_rpc.response.method_schema_validated",
            arguments={"evidence": str(result.get("evidence_file") or "<none>")},
            source=__name__,
        )
        state['pending_question'] = _continue_question(state, case)
        return
    case_dict["status"] = "existing_family_method_validated_next"
    state['active_group'] = "endpoint_process"
    state['pending_question'] = _continue_question(state, case)
    emit(
        responses,
        "chain_rpc.response.new_chain_method_validated",
        arguments={"evidence": str(result.get("evidence_file") or "<none>")},
        source=__name__,
    )


def _schema_conflict(
    state: AgentGraphState,
    draft: dict[str, Any],
    *,
    case: str,
    responses: ResponseCollector,
) -> bool:
    family = _adapter_family(state)
    rest = _schema_indicates_rest(draft)
    jsonrpc = _schema_indicates_jsonrpc(draft)
    if not ((family == "jsonrpc" and rest) or (family == "rest" and jsonrpc)):
        return False
    if case == "new_chain":
        identity = state.setdefault("chain_identity", {})
        identity["status"] = "needs_protocol_confirmation"
        if family == "jsonrpc" and rest:
            identity["adapter_family"] = ""
            state.setdefault("endpoint_evidence", {})["candidate_endpoint_ready"] = False
        state['active_group'] = "chain_identity"
        state['pending_question'] = _adapter_family_question(state)
    else:
        custom = state.setdefault("custom_rpc", {})
        custom.update({"status": "needs_adapter_family_confirmation", "endpoint_ready": False})
        state['active_group'] = "endpoint_process"
        state['pending_question'] = _adapter_family_question(state, custom=True)
    emit(
        responses,
        "chain_rpc.response.schema_protocol_conflict",
        arguments={"adapter_family": family},
        source=__name__,
    )
    return True


def _record_validation_failure(state: AgentGraphState, code: str, result: Mapping[str, Any]) -> None:
    """Attach redacted structured evidence without taking workflow control."""

    evidence_file = str(result.get("evidence_file") or "")
    record = build_failure_record(
        code,
        source="endpoint",
        severity="blocking",
        facts=[{
            "code": code,
            "source": "endpoint",
            "status": result.get("status"),
            "detail": str(result.get("error") or result.get("status") or "validation failed"),
        }],
        evidence_paths=[evidence_file] if evidence_file else [],
        confirmed_config=state.get("confirmed_config") or {},
    )
    state.setdefault("endpoint_evidence", {})["last_failure_record"] = record


def _selected_custom_contracts(state: AgentGraphState) -> list[dict[str, Any]]:
    """Return the custom contracts selected for the effective workload."""

    selected = {
        normalize_scalar(method)
        for method in (state.get("workload") or {}).get("methods") or []
        if normalize_scalar(method)
    }
    records = [
        item for item in validated_contracts(state)
        if normalize_scalar(item.get("method")) and (not selected or normalize_scalar(item.get("method")) in selected)
    ]
    unique: dict[str, dict[str, Any]] = {}
    for item in records:
        unique[normalize_scalar(item.get("method"))] = item
    return list(unique.values())


def _validate_final_endpoint_contracts(
    *,
    chain: str,
    endpoint: str,
    family: str,
    contracts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replay every selected custom method against the final endpoint.

    The lower-level endpoint validator deliberately caps a multi-method probe
    at five calls. Final workload validation cannot inherit that sampling
    behavior, so each reviewed method is probed independently here.
    """

    method_results: dict[str, dict[str, Any]] = {}
    for contract in contracts:
        method = normalize_scalar(contract.get("method"))
        params = contract.get("params", [])
        method_results[method] = validate_rpc_endpoint(
            chain=chain,
            endpoint=endpoint,
            methods=[method],
            adapter_family=family,
            method_params={method: params},
            timeout=3.0,
        )
    failed = [method for method, item in method_results.items() if not item.get("ready")]
    evidence_files = [
        str(item.get("evidence_file") or "")
        for item in method_results.values()
        if str(item.get("evidence_file") or "")
    ]
    return {
        "ready": not failed,
        "status": "ok" if not failed else "failed",
        "error": "; ".join(
            f"{method}: {method_results[method].get('error') or method_results[method].get('status') or 'validation failed'}"
            for method in failed
        ),
        "endpoint": endpoint,
        "selected_methods": list(method_results),
        "method_results": method_results,
        "evidence_files": evidence_files,
        "evidence_file": evidence_files[0] if evidence_files else "",
    }


def _invalidate_endpoint_bound_methods(
    state: AgentGraphState,
    owner: dict[str, Any],
    endpoint: str,
) -> None:
    """Invalidate evidence whose truth was bound to a different endpoint."""

    custom = state.setdefault("custom_rpc", {})
    catalog = custom.get("catalog")
    if isinstance(catalog, dict):
        invalidate_rpc_catalog_endpoint_evidence(catalog, endpoint, role="validation")
        refresh_catalog_projection(state)
    else:
        # A legacy checkpoint is migrated once by ensure_catalog; all later
        # invalidation and execution use the versioned catalog exclusively.
        ensure_catalog(state)
        catalog = custom.get("catalog")
        if isinstance(catalog, dict):
            invalidate_rpc_catalog_endpoint_evidence(catalog, endpoint, role="validation")
            refresh_catalog_projection(state)
    for key in ("observed_response", "response_confirmed", "method_probe"):
        owner.pop(key, None)
    state["workload"] = {}
    state["fixture_evidence"] = {}


def _reset_in_progress_method_evidence(state: AgentGraphState, *, case: str) -> None:
    """Reset only the unfinished method while retaining validated contracts."""

    owner = _case_dict(state, case)
    reset_draft(state)
    method_key = "candidate_method" if case == "new_chain" else "method"
    params_key = "candidate_params" if case == "new_chain" else "params"
    for key in (
        method_key,
        params_key,
        "schema_draft",
        "schema_evidence",
        "schema_evidence_fragments",
        "observed_response",
        "response_confirmed",
        "schema_confirmed",
        "method_probe",
    ):
        owner.pop(key, None)


def _parameter_semantics_missing_details(state: AgentGraphState) -> str:
    details = "; ".join(
        f"param[{index}]: {', '.join(missing)}"
        for index, missing in incomplete_parameters(state)
    )
    return details or "<unknown>"


def _validation_endpoint(state: AgentGraphState, case: str) -> str:
    if case == "new_chain":
        return normalize_scalar((state.get("endpoint_evidence") or {}).get("candidate_endpoint"))
    return normalize_scalar((state.get("custom_rpc") or {}).get("endpoint"))


def _response_contract_conflicts(
    draft: dict[str, Any],
    observed_sample: str,
    *,
    stable_result: str = "",
) -> list[str]:
    """Compare user-confirmed response evidence with the probed JSON shape."""

    try:
        observed = json.loads(observed_sample)
    except (TypeError, json.JSONDecodeError):
        return []
    conflicts: list[str] = []
    expected_sample = draft.get("response_sample", draft.get("response_example"))
    if isinstance(expected_sample, str):
        try:
            expected_sample = json.loads(expected_sample)
        except json.JSONDecodeError:
            expected_sample = None
    if expected_sample is not None:
        expected_shape = _json_shape(expected_sample)
        observed_shape = _json_shape(observed)
        if expected_shape != observed_shape:
            conflicts.append(f"response shape mismatch: expected {expected_shape}, observed {observed_shape}")
        if stable_result:
            expected_result = expected_sample.get("result") if isinstance(expected_sample, dict) else expected_sample
            expected_stable = normalize_scalar(expected_result)
            try:
                expected_stable = str(int(expected_stable, 0))
            except (TypeError, ValueError):
                pass
            if expected_stable and expected_stable != stable_result:
                conflicts.append(
                    f"stable response mismatch: expected {expected_stable}, observed {stable_result}"
                )

    fields = [item for item in draft.get("response_fields") or [] if isinstance(item, dict) and normalize_scalar(item.get("name"))]
    if fields:
        names = {normalize_scalar(item.get("name")) for item in fields}
        container = observed if "result" in names else observed.get("result") if isinstance(observed, dict) else observed
        if not isinstance(container, dict):
            conflicts.append("response fields were expected but the observed response container is not an object")
        else:
            for item in fields:
                name = normalize_scalar(item.get("name"))
                if name not in container:
                    conflicts.append(f"observed response is missing expected field {name}")
                    continue
                expected_type = _normalized_json_type(item.get("type") or item.get("json_type"))
                observed_type = _json_type(container[name])
                if expected_type and expected_type != observed_type:
                    conflicts.append(f"response field {name} expected {expected_type}, observed {observed_type}")

    declared_type = _declared_response_json_type(draft.get("response_json_type"))
    if declared_type:
        result_value = observed.get("result") if isinstance(observed, dict) and "result" in observed else observed
        observed_type = _json_type(result_value)
        if observed_type != declared_type:
            conflicts.append(
                f"response contract expected {declared_type}, observed {observed_type}"
            )
    return list(dict.fromkeys(conflicts))


def _json_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [sorted({str(_json_shape(item)) for item in value})]
    return _json_type(value)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _normalized_json_type(value: Any) -> str:
    text = normalize_scalar(value).casefold()
    aliases = {"str": "string", "integer": "number", "int": "number", "float": "number", "bool": "boolean", "dict": "object", "list": "array"}
    return aliases.get(text, text if text in {"null", "boolean", "string", "number", "array", "object"} else "")


def _declared_response_json_type(value: Any) -> str:
    declared = normalize_scalar(value).casefold()
    return (
        declared
        if declared in {"null", "boolean", "string", "number", "array", "object"}
        else ""
    )
