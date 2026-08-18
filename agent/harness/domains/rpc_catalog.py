"""Versioned, case-independent custom RPC catalog transitions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, TypedDict

from agent.validators.endpoint_probe import (
    build_rpc_probe_contract,
    rpc_probe_contract_hash,
)
from agent.validators.rpc_workload import default_workload
from agent.knowledge.framework_capabilities import load_framework_capabilities

from ..input_values import (
    looks_like_rest_method_identity,
    looks_like_rpc_method_token,
    normalize_scalar,
)
from ..state import AgentGraphState
from ..secret_refs import materialize_state_secret_references
from .rpc_receipts import (
    admitted_action_value_hash,
    emit_catalog_transition_receipt,
    emit_method_probe_receipt,
    emit_schema_provenance_receipt,
    evidence_hash,
    validate_rpc_receipt,
)


RPC_CATALOG_VERSION = 1
RPC_METHOD_CONTRACT_VERSION = 1

CatalogCommand = Literal[
    "correct_draft",
    "append_evidence",
    "keep_current_method",
    "replace_current_method",
    "confirm_parameter",
    "confirm_request",
    "confirm_response",
    "bind_validation_endpoint",
    "probe_method",
    "add_method",
    "finish_catalog",
]


class EvidenceFragment(TypedDict, total=False):
    revision: int
    source: str
    kind: str
    content: str
    method: str
    source_action_value_hash: str


class MethodDraft(TypedDict, total=False):
    contract_version: int
    revision: int
    phase: str
    method: str
    params_json: Any
    params: list[dict[str, Any]]
    parameter_style: str
    confirmed_parameters: list[int]
    request_confirmed: bool
    response_confirmed: bool
    response_summary: str
    response_json_type: str
    response_fields: list[dict[str, Any]]
    response_sample: Any
    exchange_correlation: dict[str, Any]
    evidence: list[EvidenceFragment]
    field_provenance: list[dict[str, Any]]
    validation_endpoint: str
    probe: dict[str, Any]


class ValidatedMethodContract(TypedDict, total=False):
    contract_version: int
    revision: int
    method: str
    params: Any
    parameter_style: str
    schema: dict[str, Any]
    request_confirmed: bool
    response_confirmed: bool
    observed_response: dict[str, Any]
    validation_endpoint: str
    evidence_file: str
    probe_receipt: dict[str, Any]
    probe_catalog_revision: int
    evidence: list[EvidenceFragment]
    chain: str


class RpcCatalog(TypedDict, total=False):
    contract_version: int
    revision: int
    chain: str
    draft: MethodDraft
    methods: list[ValidatedMethodContract]
    finished: bool
    last_transition: dict[str, Any]


@dataclass(frozen=True)
class CatalogTransition:
    command: CatalogCommand
    revision: int
    accepted: bool
    phase: str
    detail: str = ""


_UNKNOWN = {"", "unknown", "<unknown>", "unavailable", "n/a", "none"}


def strict_method_identity(
    value: Any,
    *,
    adapter_family: str,
    from_protocol_request: bool = False,
) -> str:
    """Return an exact wire method only when its provenance or grammar is valid."""

    method = normalize_scalar(value)
    if not method or len(method) > 256:
        return ""
    if any(ord(char) < 32 for char in method):
        return ""
    if from_protocol_request:
        return method
    family = normalize_scalar(adapter_family).casefold()
    if family in {"rest", "tendermint", "hedera_dual"}:
        return method if looks_like_rest_method_identity(method) else ""
    return method if looks_like_rpc_method_token(method) else ""


def catalog_view(state: Mapping[str, Any]) -> RpcCatalog:
    custom = state.get("custom_rpc") if isinstance(state, Mapping) else None
    if not isinstance(custom, Mapping):
        return {"contract_version": RPC_CATALOG_VERSION, "revision": 0, "methods": []}
    raw = custom.get("catalog")
    if isinstance(raw, Mapping):
        return deepcopy(dict(raw))  # type: ignore[return-value]
    return {"contract_version": RPC_CATALOG_VERSION, "revision": 0, "methods": []}


def migrate_legacy_catalog(state: AgentGraphState) -> bool:
    """Move pre-catalog checkpoint facts into the sole canonical catalog once."""

    custom = state.setdefault("custom_rpc", {})
    if isinstance(custom.get("catalog"), dict):
        return False
    identity = state.setdefault("chain_identity", {})
    draft = deepcopy(dict(custom.get("schema_draft"))) if isinstance(custom.get("schema_draft"), Mapping) else {}
    if not draft and isinstance(identity.get("schema_draft"), Mapping):
        draft = deepcopy(dict(identity.get("schema_draft") or {}))
    legacy_method = normalize_scalar(custom.get("method") or identity.get("candidate_method"))
    legacy_params_present = "params" in custom or "candidate_params" in identity
    legacy_params = deepcopy(custom.get("params") if "params" in custom else identity.get("candidate_params"))
    if legacy_method and not draft:
        draft = {
            "contract_version": RPC_METHOD_CONTRACT_VERSION,
            "phase": "evidence",
            "method": legacy_method,
        }
        if legacy_params_present:
            draft["params_json"] = legacy_params
            draft["params"] = deepcopy(legacy_params) if isinstance(legacy_params, list) else []
    fragments = custom.get("schema_evidence_fragments")
    if draft and isinstance(fragments, list):
        draft["evidence"] = [
            {
                "revision": index + 1,
                "source": "legacy_checkpoint",
                "kind": "schema_evidence",
                "content": str(fragment),
                "method": legacy_method or normalize_scalar(draft.get("method")),
            }
            for index, fragment in enumerate(fragments)
            if str(fragment).strip()
        ]
    if draft:
        draft.setdefault(
            "validation_endpoint",
            normalize_scalar(custom.get("endpoint")),
        )
        draft["phase"] = "evidence"
        draft["request_confirmed"] = False
        draft["response_confirmed"] = False
        draft.pop("probe", None)
        draft.pop("observed_response", None)
    methods = custom.get("validated_methods") if isinstance(custom.get("validated_methods"), list) else []
    source = "custom_rpc.validated_methods"
    if not methods and isinstance(identity.get("validated_methods"), list):
        methods = identity.get("validated_methods") or []
        source = "chain_identity.validated_methods"
    if not draft and methods:
        candidate = methods[0] if isinstance(methods[0], Mapping) else {}
        method = normalize_scalar(candidate.get("method"))
        if method:
            draft = {
                "contract_version": RPC_METHOD_CONTRACT_VERSION,
                "phase": "evidence",
                "method": method,
                "params_json": deepcopy(candidate.get("params")),
                "params": deepcopy(candidate.get("params"))
                if isinstance(candidate.get("params"), list)
                else [],
                "request_confirmed": False,
                "response_confirmed": False,
                "evidence": [{
                    "revision": 1,
                    "source": "legacy_checkpoint",
                    "kind": "unverified_method_candidate",
                    "content": method,
                    "method": method,
                }],
            }
    if not draft and not methods:
        return False
    custom["catalog"] = {
        "contract_version": RPC_CATALOG_VERSION,
        "revision": max(1, int(custom.get("catalog_revision") or 0)),
        "chain": normalize_scalar(identity.get("canonical")),
        "draft": draft,
        "methods": [],
        "finished": False,
        "last_transition": {
            "command": "legacy_checkpoint_quarantine",
            "accepted": False,
            "source": source,
        },
    }
    if not normalize_scalar(identity.get("status")).startswith("existing_family_"):
        custom["status"] = "needs_schema_evidence"
    for key in (
        "catalog_version",
        "catalog_revision",
        "validated_methods",
        "schema_draft",
        "schema_evidence_fragments",
        "method",
        "params",
    ):
        custom.pop(key, None)
    identity.pop("validated_methods", None)
    identity.pop("schema_draft", None)
    identity.pop("candidate_method", None)
    identity.pop("candidate_params", None)
    identity.pop("observed_response", None)
    state.setdefault("audit_events", []).append({
        "event": "legacy_rpc_catalog_migrated",
        "source": source,
        "outcome": "quarantined_for_revalidation",
    })
    return True


def ensure_catalog(state: AgentGraphState) -> RpcCatalog:
    custom = state.setdefault("custom_rpc", {})
    raw = custom.get("catalog")
    if not isinstance(raw, dict):
        migrate_legacy_catalog(state)
        raw = custom.get("catalog")
    if not isinstance(raw, dict):
        raw = {"contract_version": RPC_CATALOG_VERSION, "revision": 0, "methods": [], "finished": False}
        custom["catalog"] = raw
    raw["contract_version"] = RPC_CATALOG_VERSION
    raw.setdefault("revision", 0)
    raw.setdefault("methods", [])
    raw.setdefault("finished", False)
    chain = normalize_scalar((state.get("chain_identity") or {}).get("canonical"))
    if chain:
        raw["chain"] = chain
    _sync_projection(state)
    return raw  # type: ignore[return-value]


def draft_view(state: Mapping[str, Any]) -> MethodDraft:
    draft = catalog_view(state).get("draft")
    return deepcopy(draft) if isinstance(draft, dict) else {}


def request_contract_hash(draft: Mapping[str, Any]) -> str:
    """Bind probe evidence to the exact user-confirmed RPC contract."""

    return evidence_hash({
        "method": normalize_scalar(draft.get("method")),
        "params_json": deepcopy(draft.get("params_json")),
        "params": deepcopy(draft.get("params") or []),
        "parameter_style": normalize_scalar(draft.get("parameter_style")),
        "response_summary": normalize_scalar(draft.get("response_summary")),
        "response_json_type": normalize_scalar(
            draft.get("response_json_type")
        ),
        "response_fields": deepcopy(draft.get("response_fields") or []),
        "response_sample": deepcopy(draft.get("response_sample")),
        "response_example": deepcopy(draft.get("response_example")),
        "exchange_correlation": deepcopy(
            draft.get("exchange_correlation") or {}
        ),
        "validation_endpoint": normalize_scalar(
            draft.get("validation_endpoint")
        ),
    })


def bind_validation_endpoint(
    state: AgentGraphState,
    endpoint: str,
) -> CatalogTransition:
    """Bind the exact probe scope without changing the reviewed RPC schema."""

    catalog = ensure_catalog(state)
    draft = _draft(catalog)
    normalized = normalize_scalar(endpoint)
    if not normalized:
        return _transition(
            state,
            catalog,
            "bind_validation_endpoint",
            False,
            str(draft.get("phase") or "draft"),
            "validation endpoint is required",
        )
    if normalize_scalar(draft.get("validation_endpoint")) == normalized:
        return _transition(
            state,
            catalog,
            "bind_validation_endpoint",
            True,
            str(draft.get("phase") or "draft"),
        )
    draft["validation_endpoint"] = normalized
    draft.pop("probe", None)
    draft.pop("observed_response", None)
    revision = _next_revision(catalog)
    draft["revision"] = revision
    _sync_projection(state)
    return _record_transition(
        state,
        catalog,
        "bind_validation_endpoint",
        revision,
        True,
        str(draft.get("phase") or "draft"),
    )


def validated_contracts(state: AgentGraphState) -> list[ValidatedMethodContract]:
    catalog = ensure_catalog(state)
    methods = catalog.setdefault("methods", [])
    return methods  # type: ignore[return-value]


def validated_contracts_view(state: Mapping[str, Any]) -> list[ValidatedMethodContract]:
    methods = catalog_view(state).get("methods")
    return deepcopy(methods) if isinstance(methods, list) else []


def effective_job_local_workload_methods(
    state: Mapping[str, Any],
) -> list[str]:
    workload = state.get("workload")
    if not isinstance(workload, Mapping) or not (
        workload.get("confirmed") and workload.get("job_local_override")
    ):
        return []
    return list(
        dict.fromkeys(
            normalize_scalar(method)
            for method in workload.get("methods") or ()
            if normalize_scalar(method)
        )
    )


def effective_custom_workload_methods(state: Mapping[str, Any]) -> list[str]:
    """Return selected methods that are absent from the canonical template."""

    selected = effective_job_local_workload_methods(state)
    if not selected:
        return []
    identity = state.get("chain_identity")
    chain = normalize_scalar(
        identity.get("canonical") or identity.get("raw")
    ) if isinstance(identity, Mapping) else ""
    defaults = default_workload(chain) if chain else {"exists": False}
    canonical = (
        {
            normalize_scalar(method)
            for method in defaults.get("methods") or ()
            if normalize_scalar(method)
        }
        if defaults.get("exists")
        else set()
    )
    return [method for method in selected if method not in canonical]


def selected_validated_contracts(
    state: AgentGraphState,
) -> list[ValidatedMethodContract]:
    """Return live catalog contracts selected by the effective workload."""

    selected = effective_custom_workload_methods(state)
    if not selected:
        return []
    selected_set = set(selected)
    unique: dict[str, ValidatedMethodContract] = {}
    for item in validated_contracts(state):
        method = normalize_scalar(item.get("method"))
        if method in selected_set:
            unique[method] = item
    return [unique[method] for method in selected if method in unique]


def selected_validated_contracts_view(
    state: Mapping[str, Any],
) -> list[ValidatedMethodContract]:
    """Return a detached view of contracts selected by the effective workload."""

    selected = effective_custom_workload_methods(state)
    if not selected:
        return []
    selected_set = set(selected)
    unique = {
        normalize_scalar(item.get("method")): item
        for item in validated_contracts_view(state)
        if normalize_scalar(item.get("method")) in selected_set
    }
    return [unique[method] for method in selected if method in unique]


def selected_contracts_cover_effective_workload(
    state: Mapping[str, Any],
) -> bool:
    """Return whether every selected custom method has one catalog contract."""

    selected = effective_custom_workload_methods(state)
    if not selected:
        return True
    contracts = selected_validated_contracts_view(state)
    return [normalize_scalar(item.get("method")) for item in contracts] == selected


def refresh_catalog_projection(state: AgentGraphState) -> None:
    _sync_projection(state)


def catalog_method_names(state: Mapping[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            method
            for item in validated_contracts_view(state)
            if isinstance(item, dict) and (method := normalize_scalar(item.get("method")))
        )
    )


def append_evidence(
    state: AgentGraphState,
    *,
    content: str,
    source: str,
    kind: str,
    method: str = "",
) -> CatalogTransition:
    catalog = ensure_catalog(state)
    draft = _draft(catalog)
    current = normalize_scalar(draft.get("method"))
    incoming = normalize_scalar(method)
    if current and incoming and current != incoming:
        return _transition(state, catalog, "append_evidence", False, str(draft.get("phase") or "draft"), "conflicting method identity")
    clean = str(content or "").strip()
    if not clean:
        return _transition(state, catalog, "append_evidence", False, str(draft.get("phase") or "draft"), "empty evidence")
    revision = _next_revision(catalog)
    fragment: EvidenceFragment = {
        "revision": revision,
        "source": normalize_scalar(source) or "user",
        "kind": normalize_scalar(kind) or "evidence",
        "content": clean,
    }
    source_action_value_hash = admitted_action_value_hash(
        state,
        argument_names=(
            "source_evidence",
            "rpc_schema_evidence",
            "rpc_method",
            "selected_value",
            "answer",
        ),
        expected_value=clean,
    )
    action_id = normalize_scalar(
        (state.get("current_action") or {}).get("action_id")
    )
    admitted_actions = (state.get("turn_context") or {}).get(
        "admitted_actions"
    )
    if action_id and isinstance(admitted_actions, list) and not source_action_value_hash:
        return _transition(
            state,
            catalog,
            "append_evidence",
            False,
            str(draft.get("phase") or "draft"),
            "evidence is not bound to the admitted current-turn input",
        )
    if source_action_value_hash:
        fragment["source_action_value_hash"] = source_action_value_hash
    if incoming or current:
        fragment["method"] = incoming or current
    fragments = draft.setdefault("evidence", [])
    if not any(
        item.get("content") == clean and item.get("source") == fragment["source"]
        for item in fragments
        if isinstance(item, dict)
    ):
        fragments.append(fragment)
    if incoming and not current:
        draft["method"] = incoming
    draft["revision"] = revision
    _sync_projection(state)
    return _record_transition(state, catalog, "append_evidence", revision, True, str(draft.get("phase") or "draft"))


def correct_draft(state: AgentGraphState, draft_patch: Mapping[str, Any]) -> CatalogTransition:
    catalog = ensure_catalog(state)
    previous = _draft(catalog)
    next_draft = deepcopy(dict(draft_patch))
    next_draft["contract_version"] = RPC_METHOD_CONTRACT_VERSION
    next_draft["phase"] = "parameter_confirmation" if next_draft.get("params") else "request_confirmation"
    next_draft["confirmed_parameters"] = []
    next_draft["request_confirmed"] = False
    next_draft["response_confirmed"] = False
    next_draft.pop("probe", None)
    next_draft.pop("observed_response", None)
    next_draft["field_provenance"] = [
        item
        for item in next_draft.get("field_provenance") or ()
        if isinstance(item, Mapping)
        and not str(item.get("field_path") or "").startswith(
            "observed_response."
        )
    ]
    if normalize_scalar(previous.get("method")) == normalize_scalar(next_draft.get("method")):
        next_draft["evidence"] = deepcopy(previous.get("evidence") or [])
    else:
        next_draft.setdefault("evidence", [])
    params_json = next_draft.get("params_json")
    next_draft["parameter_style"] = (
        "positional" if isinstance(params_json, list) else "object" if isinstance(params_json, dict) else "unknown"
    )
    revision = _next_revision(catalog)
    next_draft["revision"] = revision
    catalog["draft"] = next_draft  # type: ignore[assignment]
    catalog["finished"] = False
    _sync_projection(state)
    transition = _record_transition(
        state,
        catalog,
        "correct_draft",
        revision,
        True,
        str(next_draft["phase"]),
    )
    emit_schema_provenance_receipt(
        state,
        catalog_revision=revision,
        draft=next_draft,
    )
    return transition


def parameter_contract_missing(parameter: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    if "index" not in parameter:
        missing.append("index")
    if not normalize_scalar(parameter.get("name")):
        missing.append("name")
    if normalize_scalar(parameter.get("json_type")).casefold() in _UNKNOWN:
        missing.append("json_type")
    semantic = normalize_scalar(parameter.get("semantic_type")).casefold()
    encoding = normalize_scalar(parameter.get("encoding")).casefold()
    if semantic in _UNKNOWN and encoding in _UNKNOWN:
        missing.append("semantic_type_or_encoding")
    if normalize_scalar(parameter.get("meaning")).casefold() in _UNKNOWN:
        missing.append("meaning")
    required = parameter.get("required")
    if required not in {True, False, "required", "optional"}:
        missing.append("required_or_optional")
    if "example" not in parameter:
        missing.append("example")
    return missing


def incomplete_parameters(state: Mapping[str, Any]) -> list[tuple[int, list[str]]]:
    params = draft_view(state).get("params")
    if not isinstance(params, list):
        return []
    return [
        (index, missing)
        for index, item in enumerate(params)
        if isinstance(item, dict) and (missing := parameter_contract_missing(item))
    ]


def next_parameter_to_confirm(state: Mapping[str, Any]) -> int | None:
    draft = draft_view(state)
    params = draft.get("params")
    if not isinstance(params, list):
        return None
    confirmed = {int(item) for item in draft.get("confirmed_parameters") or [] if isinstance(item, int)}
    return next((index for index in range(len(params)) if index not in confirmed), None)


def confirm_parameter(state: AgentGraphState, index: int, accepted: bool) -> CatalogTransition:
    catalog = ensure_catalog(state)
    draft = _draft(catalog)
    params = draft.get("params")
    if not isinstance(params, list) or index < 0 or index >= len(params):
        return _transition(state, catalog, "confirm_parameter", False, str(draft.get("phase") or "draft"), "unknown parameter")
    missing = parameter_contract_missing(params[index]) if isinstance(params[index], dict) else ["parameter_contract"]
    if not accepted or missing:
        draft["phase"] = "evidence"
        return _commit_transition(state, catalog, "confirm_parameter", False, "evidence", ", ".join(missing) or "declined")
    confirmed = [int(item) for item in draft.get("confirmed_parameters") or [] if isinstance(item, int)]
    if index not in confirmed:
        confirmed.append(index)
    draft["confirmed_parameters"] = sorted(confirmed)
    draft["phase"] = "request_confirmation" if len(confirmed) == len(params) else "parameter_confirmation"
    return _commit_transition(state, catalog, "confirm_parameter", True, str(draft["phase"]))


def confirm_request(state: AgentGraphState, accepted: bool) -> CatalogTransition:
    catalog = ensure_catalog(state)
    draft = _draft(catalog)
    params = draft.get("params")
    if not accepted:
        draft["request_confirmed"] = False
        draft["phase"] = "evidence"
        return _commit_transition(state, catalog, "confirm_request", False, "evidence", "declined")
    if not normalize_scalar(draft.get("method")) or not isinstance(draft.get("params_json"), (list, dict)):
        return _transition(state, catalog, "confirm_request", False, str(draft.get("phase") or "draft"), "incomplete wire request")
    if incomplete_parameters(state):
        draft["phase"] = "evidence"
        return _commit_transition(state, catalog, "confirm_request", False, "evidence", "incomplete parameter semantics")
    if isinstance(params, list) and len(params) != len(draft.get("confirmed_parameters") or []):
        return _transition(state, catalog, "confirm_request", False, "parameter_confirmation", "parameters are not individually confirmed")
    draft["request_confirmed"] = True
    draft["phase"] = "response_confirmation" if has_expected_response(draft) and not draft.get("response_confirmed") else "probe_confirmation"
    return _commit_transition(state, catalog, "confirm_request", True, str(draft["phase"]))


def has_expected_response(draft: Mapping[str, Any]) -> bool:
    summary = normalize_scalar(draft.get("response_summary")).casefold()
    return bool(
        summary not in _UNKNOWN
        or (isinstance(draft.get("response_fields"), list) and draft.get("response_fields"))
        or draft.get("response_sample") is not None
        or draft.get("response_example") is not None
    )


def confirm_response(state: AgentGraphState, accepted: bool, *, observed: bool = False) -> CatalogTransition:
    catalog = ensure_catalog(state)
    draft = _draft(catalog)
    if not accepted:
        draft["response_confirmed"] = False
        draft["phase"] = "evidence"
        return _commit_transition(state, catalog, "confirm_response", False, "evidence", "declined")
    if not observed and not has_expected_response(draft):
        return _transition(state, catalog, "confirm_response", False, str(draft.get("phase") or "draft"), "missing response contract")
    if observed and not _observed_response_matches_probe(draft):
        return _transition(
            state,
            catalog,
            "confirm_response",
            False,
            str(draft.get("phase") or "draft"),
            "observed response is not bound to the successful probe",
        )
    draft["response_confirmed"] = True
    draft["phase"] = "validated" if observed and (draft.get("probe") or {}).get("ready") else "probe_confirmation"
    transition = _commit_transition(
        state,
        catalog,
        "confirm_response",
        True,
        str(draft["phase"]),
    )
    probe = draft.get("probe")
    if isinstance(probe, dict) and probe.get("ready"):
        probe["admissible_catalog_revision"] = transition.revision
        _sync_projection(state)
    return transition


def record_probe(state: AgentGraphState, result: Mapping[str, Any], observed_response: Mapping[str, Any]) -> CatalogTransition:
    catalog = ensure_catalog(state)
    draft = _draft(catalog)
    evidence_file = normalize_scalar(
        result.get("evidence_file")
        or observed_response.get("evidence_file")
    )
    endpoint = normalize_scalar(draft.get("validation_endpoint"))
    endpoint_hash = _materialized_endpoint_hash(state, endpoint)
    evidence_file_hash = probe_evidence_content_hash(evidence_file)
    evidence_probe_contract_hash = probe_evidence_contract_hash(
        evidence_file
    )
    expected_probe_contract_hash = _expected_probe_contract_hash(
        state,
        chain=normalize_scalar(catalog.get("chain")),
        endpoint=endpoint,
        method=normalize_scalar(draft.get("method")),
        schema=draft,
    )
    ready = bool(
        result.get("ready")
        and endpoint_hash
        and evidence_file_hash
        and expected_probe_contract_hash
        and normalize_scalar(result.get("probe_contract_hash"))
        == expected_probe_contract_hash
        and evidence_probe_contract_hash == expected_probe_contract_hash
    )
    draft["probe"] = {
        **deepcopy(dict(result)),
        "ready": ready,
        "request_contract_hash": request_contract_hash(draft),
    }
    if observed_response:
        draft["observed_response"] = deepcopy(dict(observed_response))
        provenance = [
            dict(item)
            for item in draft.get("field_provenance") or ()
            if isinstance(item, Mapping)
            and not str(item.get("field_path") or "").startswith("observed_response.")
        ]
        source_revision = int(catalog.get("revision") or 0) + 1
        provenance.extend(
            {
                "field_path": f"observed_response.{field}",
                "source_kind": "endpoint_probe",
                "source_revisions": [source_revision],
                "value_hash": evidence_hash(value),
            }
            for field, value in sorted(observed_response.items())
        )
        draft["field_provenance"] = provenance
    draft["phase"] = "probe_succeeded" if ready else "probe_confirmation"
    transition = _commit_transition(
        state,
        catalog,
        "probe_method",
        ready,
        str(draft["phase"]),
        str(result.get("error") or ""),
    )
    draft["probe"]["catalog_revision"] = transition.revision
    draft["probe"]["admissible_catalog_revision"] = transition.revision
    probe_receipt = emit_method_probe_receipt(
        state,
        method=normalize_scalar(draft.get("method")),
        endpoint_hash=endpoint_hash,
        request_contract_hash=normalize_scalar(
            draft["probe"].get("request_contract_hash")
        ),
        catalog_revision=transition.revision,
        evidence_file_hash=evidence_file_hash,
        probe_contract_hash=expected_probe_contract_hash,
        ready=ready
        and bool(endpoint_hash)
        and bool(evidence_file_hash)
        and bool(expected_probe_contract_hash),
        probe_status=normalize_scalar(result.get("status")),
    )
    if probe_receipt:
        draft["probe"]["probe_receipt"] = probe_receipt
        draft["probe"]["evidence_file"] = evidence_file
    _sync_projection(state)
    emit_schema_provenance_receipt(
        state,
        catalog_revision=transition.revision,
        draft=draft,
    )
    return transition


def add_validated_method(state: AgentGraphState) -> CatalogTransition:
    catalog = ensure_catalog(state)
    draft = _draft(catalog)
    method = normalize_scalar(draft.get("method"))
    probe = draft.get("probe") or {}
    probe_receipt = (
        probe.get("probe_receipt")
        if isinstance(probe.get("probe_receipt"), Mapping)
        else {}
    )
    validation_endpoint = normalize_scalar(draft.get("validation_endpoint"))
    evidence_file = normalize_scalar(probe.get("evidence_file"))
    receipt_valid = method_probe_receipt_matches_contract(
        state,
        receipt=probe_receipt,
        chain=normalize_scalar(catalog.get("chain")),
        method=method,
        schema=draft,
        endpoint=validation_endpoint,
        evidence_file=evidence_file,
        catalog_revision=probe.get("catalog_revision"),
    )
    if (
        not method
        or not draft.get("request_confirmed")
        or not probe.get("ready")
        or probe.get("request_contract_hash") != request_contract_hash(draft)
        or not isinstance(probe.get("catalog_revision"), int)
        or isinstance(probe.get("catalog_revision"), bool)
        or probe.get("admissible_catalog_revision")
        != int(catalog.get("revision") or 0)
        or not receipt_valid
    ):
        return _transition(state, catalog, "add_method", False, str(draft.get("phase") or "draft"), "request confirmation and successful probe are required")
    if not draft.get("response_confirmed"):
        return _transition(state, catalog, "add_method", False, str(draft.get("phase") or "draft"), "response confirmation is required")
    if draft.get("observed_response") and not _observed_response_matches_probe(
        draft
    ):
        return _transition(
            state,
            catalog,
            "add_method",
            False,
            str(draft.get("phase") or "draft"),
            "observed response proof is stale or incomplete",
        )
    revision = _next_revision(catalog)
    record: ValidatedMethodContract = {
        "contract_version": RPC_METHOD_CONTRACT_VERSION,
        "revision": revision,
        "method": method,
        "params": deepcopy(draft.get("params_json")),
        "parameter_style": str(draft.get("parameter_style") or "unknown"),
        "schema": deepcopy(dict(draft)),
        "request_confirmed": True,
        "response_confirmed": True,
        "observed_response": deepcopy(
            dict(draft.get("observed_response") or {})
        ),
        "validation_endpoint": validation_endpoint,
        "evidence_file": evidence_file,
        "probe_receipt": deepcopy(dict(probe_receipt)),
        "probe_catalog_revision": int(
            probe_receipt.get("catalog_revision") or 0
        ),
        "evidence": deepcopy(draft.get("evidence") or []),
        "chain": normalize_scalar(catalog.get("chain")),
    }
    methods = [
        item for item in catalog.get("methods") or []
        if isinstance(item, dict) and normalize_scalar(item.get("method")) != method
    ]
    methods.append(record)
    catalog["methods"] = methods
    draft["phase"] = "validated"
    draft["revision"] = revision
    _sync_projection(state)
    return _record_transition(state, catalog, "add_method", revision, True, "validated")


def validated_method_contract_is_current(
    state: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    endpoint: str = "",
) -> bool:
    """Validate one persisted method contract at every consumption boundary."""

    schema = (
        contract.get("schema")
        if isinstance(contract.get("schema"), Mapping)
        else {}
    )
    receipt = (
        contract.get("probe_receipt")
        if isinstance(contract.get("probe_receipt"), Mapping)
        else {}
    )
    method = normalize_scalar(contract.get("method"))
    contract_endpoint = normalize_scalar(
        contract.get("validation_endpoint")
    )
    expected_endpoint = normalize_scalar(endpoint) or contract_endpoint
    chain = normalize_scalar(contract.get("chain"))
    identity = (
        state.get("chain_identity")
        if isinstance(state.get("chain_identity"), Mapping)
        else {}
    )
    current_chain = normalize_scalar(
        identity.get("canonical")
        or identity.get("raw")
        or catalog_view(state).get("chain")
    )
    revision = contract.get("revision")
    probe_revision = contract.get("probe_catalog_revision")
    base_valid = bool(
        contract.get("contract_version") == RPC_METHOD_CONTRACT_VERSION
        and method
        and normalize_scalar(schema.get("method")) == method
        and contract.get("params") == schema.get("params_json")
        and contract.get("request_confirmed") is True
        and contract.get("response_confirmed") is True
        and contract_endpoint
        and expected_endpoint == contract_endpoint
        and chain
        and (not current_chain or current_chain == chain)
        and isinstance(revision, int)
        and not isinstance(revision, bool)
        and isinstance(probe_revision, int)
        and not isinstance(probe_revision, bool)
        and revision > probe_revision >= 0
        and method_probe_receipt_matches_contract(
            state,
            receipt=receipt,
            chain=chain,
            method=method,
            schema=schema,
            endpoint=contract_endpoint,
            evidence_file=normalize_scalar(contract.get("evidence_file")),
            catalog_revision=probe_revision,
        )
    )
    if not base_valid:
        return False
    confirmed = (
        state.get("confirmed_config")
        if isinstance(state.get("confirmed_config"), Mapping)
        else {}
    )
    final_endpoint = normalize_scalar(confirmed.get("LOCAL_RPC_URL"))
    if (
        normalize_scalar(state.get("target_mode")) == "real-node"
        and final_endpoint
    ):
        return _final_endpoint_replay_is_current(
            state,
            contract,
            endpoint=final_endpoint,
        )
    return True


def method_probe_receipt_matches_contract(
    state: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    chain: str,
    method: str,
    schema: Mapping[str, Any],
    endpoint: str,
    evidence_file: str,
    catalog_revision: Any,
) -> bool:
    """Verify receipt, schema, endpoint, and durable probe evidence together."""

    expected_probe_hash = _expected_probe_contract_hash(
        state,
        chain=chain,
        endpoint=endpoint,
        method=method,
        schema=schema,
    )
    return bool(
        receipt
        and validate_rpc_receipt(receipt)[0]
        and receipt.get("method_hash") == evidence_hash(method)
        and receipt.get("endpoint_hash")
        == _materialized_endpoint_hash(state, endpoint)
        and receipt.get("request_contract_hash")
        == request_contract_hash(schema)
        and receipt.get("catalog_revision") == catalog_revision
        and receipt.get("evidence_file_hash")
        == probe_evidence_content_hash(evidence_file)
        and expected_probe_hash
        and receipt.get("probe_contract_hash") == expected_probe_hash
        and probe_evidence_contract_hash(evidence_file)
        == expected_probe_hash
        and receipt.get("ready") is True
    )


def _observed_response_matches_probe(
    draft: Mapping[str, Any],
) -> bool:
    observed = (
        draft.get("observed_response")
        if isinstance(draft.get("observed_response"), Mapping)
        else {}
    )
    probe = (
        draft.get("probe")
        if isinstance(draft.get("probe"), Mapping)
        else {}
    )
    return bool(
        observed
        and probe.get("ready") is True
        and normalize_scalar(observed.get("evidence_file"))
        == normalize_scalar(probe.get("evidence_file"))
        and (
            normalize_scalar(observed.get("shape_hash"))
            or normalize_scalar(observed.get("sample"))
            or isinstance(observed.get("http_status"), int)
        )
    )


def _final_endpoint_replay_is_current(
    state: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    endpoint: str,
) -> bool:
    method = normalize_scalar(contract.get("method"))
    schema = (
        contract.get("schema")
        if isinstance(contract.get("schema"), Mapping)
        else {}
    )
    contract_endpoint = normalize_scalar(contract.get("final_endpoint"))
    evidence_file = normalize_scalar(
        contract.get("final_endpoint_evidence_file")
    )
    evidence = (
        state.get("endpoint_evidence")
        if isinstance(state.get("endpoint_evidence"), Mapping)
        else {}
    )
    receipt = (
        evidence.get("local_rpc_url_validation_receipt")
        if isinstance(
            evidence.get("local_rpc_url_validation_receipt"),
            Mapping,
        )
        else {}
    )
    chain = normalize_scalar(contract.get("chain"))
    family = _current_adapter_family(state, chain)
    expected_probe_hash = _expected_probe_contract_hash(
        state,
        chain=chain,
        endpoint=contract_endpoint,
        method=method,
        schema=schema,
    )
    binding = next(
        (
            item
            for item in receipt.get("method_evidence_bindings") or ()
            if isinstance(item, Mapping)
            and normalize_scalar(item.get("method")) == method
        ),
        {},
    )
    return bool(
        method
        and contract_endpoint
        and endpoint == contract_endpoint
        and evidence_file
        and expected_probe_hash
        and probe_evidence_contract_hash(evidence_file)
        == expected_probe_hash
        and binding.get("evidence_file_hash")
        == probe_evidence_content_hash(evidence_file)
        and binding.get("probe_contract_hash") == expected_probe_hash
        and validate_rpc_receipt(receipt)[0]
        and receipt.get("receipt_id")
        == evidence.get("local_rpc_url_validation_receipt_id")
        and receipt.get("role") == "final_benchmark"
        and receipt.get("case") == "runtime"
        and receipt.get("config_field") == "LOCAL_RPC_URL"
        and receipt.get("ready") is True
        and receipt.get("chain") == chain
        and receipt.get("adapter_family") == family
        and receipt.get("endpoint_hash")
        == _materialized_endpoint_hash(state, endpoint)
        and receipt.get("method_hashes")
        and evidence_hash(method) in receipt.get("method_hashes")
    )


def probe_evidence_contract_hash(path: str) -> str:
    """Validate a durable probe contract and its real method observations."""

    target = _probe_evidence_path(path)
    if target is None:
        return ""
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""
    if (
        not isinstance(payload, Mapping)
        or payload.get("ready") is not True
        or normalize_scalar(payload.get("status")) != "ok"
    ):
        return ""
    contract_hash = rpc_probe_contract_hash(payload.get("probe_contract"))
    if (
        not contract_hash
        or normalize_scalar(payload.get("probe_contract_hash"))
        != contract_hash
    ):
        return ""
    contract = payload.get("probe_contract")
    methods = (
        contract.get("methods")
        if isinstance(contract, Mapping)
        and isinstance(contract.get("methods"), list)
        else []
    )
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return ""
    for method in methods:
        matching = [
            item
            for item in checks
            if isinstance(item, Mapping)
            and normalize_scalar(item.get("name"))
            == f"method_probe:{method}"
        ]
        if len(matching) != 1:
            return ""
        check = matching[0]
        http_status = check.get("http_status")
        if (
            check.get("passed") is not True
            or not isinstance(http_status, int)
            or isinstance(http_status, bool)
            or not 100 <= http_status <= 599
            or not _is_sha256(normalize_scalar(
                check.get("response_shape_hash")
            ))
            or not normalize_scalar(check.get("response_sample"))
        ):
            return ""
    return contract_hash


def probe_evidence_content_hash(path: str) -> str:
    """Bind a probe receipt to durable evidence bytes, not a path claim."""

    if not path:
        return ""
    try:
        target = _probe_evidence_path(path)
        if target is None:
            return ""
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def _expected_probe_contract_hash(
    state: Mapping[str, Any],
    *,
    chain: str,
    endpoint: str,
    method: str,
    schema: Mapping[str, Any],
) -> str:
    try:
        materialized_endpoint = normalize_scalar(
            materialize_state_secret_references(endpoint, state)
        )
    except (KeyError, RuntimeError, ValueError):
        return ""
    transport = _current_adapter_family(state, chain)
    if not chain or not materialized_endpoint or not method or not transport:
        return ""
    return rpc_probe_contract_hash(build_rpc_probe_contract(
        chain=chain,
        endpoint=materialized_endpoint,
        transport=transport,
        methods=[method],
        method_params={method: deepcopy(schema.get("params_json"))},
    ))


def _probe_evidence_path(path: str) -> Path | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_absolute():
        target = Path(__file__).resolve().parents[3] / target
    return target if target.is_file() else None


def _is_sha256(value: str) -> bool:
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _materialized_endpoint_hash(
    state: Mapping[str, Any],
    endpoint: str,
) -> str:
    if not endpoint:
        return ""
    try:
        materialized = normalize_scalar(
            materialize_state_secret_references(endpoint, state)
        )
    except (KeyError, RuntimeError, ValueError):
        return ""
    return evidence_hash(materialized) if materialized else ""


def _current_adapter_family(
    state: Mapping[str, Any],
    chain: str,
) -> str:
    identity = (
        state.get("chain_identity")
        if isinstance(state.get("chain_identity"), Mapping)
        else {}
    )
    family = normalize_scalar(identity.get("adapter_family")).casefold()
    if family:
        return family
    canonical = normalize_scalar(chain).casefold()
    for row in load_framework_capabilities().get("chains", []):
        if normalize_scalar(row.get("chain")).casefold() == canonical:
            return normalize_scalar(
                row.get("family") or row.get("adapter_family")
            ).casefold()
    return ""




def reset_draft(state: AgentGraphState) -> CatalogTransition:
    catalog = ensure_catalog(state)
    catalog["draft"] = {}
    revision = _next_revision(catalog)
    _sync_projection(state)
    return _record_transition(state, catalog, "correct_draft", revision, True, "empty")


def finish_catalog(state: AgentGraphState) -> CatalogTransition:
    catalog = ensure_catalog(state)
    if not catalog.get("methods"):
        return _transition(state, catalog, "finish_catalog", False, "empty", "at least one validated method is required")
    catalog["finished"] = True
    return _commit_transition(state, catalog, "finish_catalog", True, "finished")


def _draft(catalog: RpcCatalog) -> MethodDraft:
    draft = catalog.get("draft")
    if not isinstance(draft, dict):
        draft = {"contract_version": RPC_METHOD_CONTRACT_VERSION, "phase": "draft", "evidence": []}
        catalog["draft"] = draft
    return draft


def _next_revision(catalog: RpcCatalog) -> int:
    revision = int(catalog.get("revision") or 0) + 1
    catalog["revision"] = revision
    return revision


def _commit_transition(
    state: AgentGraphState,
    catalog: RpcCatalog,
    command: CatalogCommand,
    accepted: bool,
    phase: str,
    detail: str = "",
) -> CatalogTransition:
    revision = _next_revision(catalog)
    draft = _draft(catalog)
    draft["revision"] = revision
    _sync_projection(state)
    return _record_transition(state, catalog, command, revision, accepted, phase, detail)


def _transition(
    state: AgentGraphState,
    catalog: RpcCatalog,
    command: CatalogCommand,
    accepted: bool,
    phase: str,
    detail: str = "",
) -> CatalogTransition:
    transition = CatalogTransition(
        command,
        int(catalog.get("revision") or 0),
        accepted,
        phase,
        detail,
    )
    emit_catalog_transition_receipt(
        state,
        command=command,
        revision=transition.revision,
        accepted=accepted,
        phase=phase,
        catalog=catalog,
    )
    return transition


def _record_transition(
    state: AgentGraphState,
    catalog: RpcCatalog,
    command: CatalogCommand,
    revision: int,
    accepted: bool,
    phase: str,
    detail: str = "",
) -> CatalogTransition:
    catalog["last_transition"] = {
        "command": command,
        "revision": revision,
        "accepted": accepted,
        "phase": phase,
        "detail": detail,
    }
    transition = CatalogTransition(command, revision, accepted, phase, detail)
    emit_catalog_transition_receipt(
        state,
        command=command,
        revision=revision,
        accepted=accepted,
        phase=phase,
        catalog=catalog,
    )
    return transition


def _sync_projection(state: AgentGraphState) -> None:
    """Remove obsolete mirrors while the versioned catalog stays authoritative."""

    custom = state.setdefault("custom_rpc", {})
    catalog = custom.get("catalog")
    if not isinstance(catalog, dict):
        return
    for key in (
        "catalog_version",
        "catalog_revision",
        "validated_methods",
        "schema_draft",
        "schema_evidence_fragments",
        "method",
        "params",
    ):
        custom.pop(key, None)
    identity = state.get("chain_identity")
    if isinstance(identity, dict):
        identity.pop("validated_methods", None)
        identity.pop("schema_draft", None)
