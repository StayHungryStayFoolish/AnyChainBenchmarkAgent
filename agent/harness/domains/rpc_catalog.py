"""Versioned, case-independent custom RPC catalog transitions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypedDict

from ..input_values import (
    looks_like_rest_method_identity,
    looks_like_rpc_method_token,
    normalize_scalar,
)
from ..state import AgentGraphState
from .rpc_receipts import (
    emit_catalog_transition_receipt,
    emit_schema_provenance_receipt,
    evidence_hash,
)


RPC_CATALOG_VERSION = 1
RPC_METHOD_CONTRACT_VERSION = 1

CatalogCommand = Literal[
    "correct_draft",
    "append_evidence",
    "confirm_parameter",
    "confirm_request",
    "confirm_response",
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
    if draft and custom.get("status") in {"response_needs_confirmation", "method_validated_next", "validated"}:
        draft["request_confirmed"] = True
    observed = custom.get("observed_response") or identity.get("observed_response")
    if draft and isinstance(observed, Mapping):
        draft["observed_response"] = deepcopy(dict(observed))
        draft["probe"] = {"ready": True, "status": "legacy_observed_response"}
    methods = custom.get("validated_methods") if isinstance(custom.get("validated_methods"), list) else []
    source = "custom_rpc.validated_methods"
    if not methods and isinstance(identity.get("validated_methods"), list):
        methods = identity.get("validated_methods") or []
        source = "chain_identity.validated_methods"
    if not draft and not methods:
        return False
    custom["catalog"] = {
        "contract_version": RPC_CATALOG_VERSION,
        "revision": max(1, int(custom.get("catalog_revision") or 0)),
        "chain": normalize_scalar(identity.get("canonical")),
        "draft": draft,
        "methods": deepcopy(methods),
        "finished": bool(methods and custom.get("status") in {"needs_scope", "validated"}),
        "last_transition": {
            "command": "legacy_checkpoint_migration",
            "accepted": True,
            "source": source,
        },
    }
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


def validated_contracts(state: AgentGraphState) -> list[ValidatedMethodContract]:
    catalog = ensure_catalog(state)
    methods = catalog.setdefault("methods", [])
    return methods  # type: ignore[return-value]


def validated_contracts_view(state: Mapping[str, Any]) -> list[ValidatedMethodContract]:
    methods = catalog_view(state).get("methods")
    return deepcopy(methods) if isinstance(methods, list) else []


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
        method=next_draft.get("method"),
        catalog_revision=revision,
        field_provenance=[
            item
            for item in next_draft.get("field_provenance") or ()
            if isinstance(item, Mapping)
        ],
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
    draft["response_confirmed"] = True
    draft["phase"] = "validated" if observed and (draft.get("probe") or {}).get("ready") else "probe_confirmation"
    return _commit_transition(state, catalog, "confirm_response", True, str(draft["phase"]))


def record_probe(state: AgentGraphState, result: Mapping[str, Any], observed_response: Mapping[str, Any]) -> CatalogTransition:
    catalog = ensure_catalog(state)
    draft = _draft(catalog)
    draft["probe"] = deepcopy(dict(result))
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
    ready = bool(result.get("ready"))
    draft["phase"] = "probe_succeeded" if ready else "probe_confirmation"
    transition = _commit_transition(
        state,
        catalog,
        "probe_method",
        ready,
        str(draft["phase"]),
        str(result.get("error") or ""),
    )
    emit_schema_provenance_receipt(
        state,
        method=draft.get("method"),
        catalog_revision=transition.revision,
        field_provenance=[
            item
            for item in draft.get("field_provenance") or ()
            if isinstance(item, Mapping)
        ],
    )
    return transition


def add_validated_method(state: AgentGraphState, contract: Mapping[str, Any]) -> CatalogTransition:
    catalog = ensure_catalog(state)
    draft = _draft(catalog)
    method = normalize_scalar(contract.get("method") or draft.get("method"))
    if not method or not draft.get("request_confirmed") or not (draft.get("probe") or {}).get("ready"):
        return _transition(state, catalog, "add_method", False, str(draft.get("phase") or "draft"), "request confirmation and successful probe are required")
    if not draft.get("response_confirmed"):
        return _transition(state, catalog, "add_method", False, str(draft.get("phase") or "draft"), "response confirmation is required")
    revision = _next_revision(catalog)
    record: ValidatedMethodContract = deepcopy(dict(contract))  # type: ignore[assignment]
    record.update({
        "contract_version": RPC_METHOD_CONTRACT_VERSION,
        "revision": revision,
        "method": method,
        "parameter_style": str(draft.get("parameter_style") or "unknown"),
        "request_confirmed": True,
        "response_confirmed": True,
        "evidence": deepcopy(draft.get("evidence") or []),
        "chain": normalize_scalar(catalog.get("chain")),
    })
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
