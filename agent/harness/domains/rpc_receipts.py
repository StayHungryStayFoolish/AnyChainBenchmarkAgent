"""Secret-free observation receipts emitted by authoritative RPC owners."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from agent.utils.redaction import redact

from ..input_values import looks_like_rpc_method_token, normalize_scalar
from ..state import AgentGraphState


RPC_RECEIPT_VERSION = 2
_COMMON_FIELDS = frozenset(
    {"receipt_type", "receipt_version", "turn_index", "owner", "receipt_id"}
)
_RECEIPT_FIELDS = {
    "rpc_endpoint_role": _COMMON_FIELDS
    | frozenset(
        {
            "role",
            "case",
            "config_field",
            "endpoint_hash",
            "source_kind",
            "source_receipt_id",
            "source_value_hash",
            "previous_endpoint_hash",
            "ready",
            "probe_status_hash",
            "chain",
            "adapter_family",
            "methods",
            "method_hashes",
            "method_evidence_bindings",
            "producer_action_id",
        }
    ),
    "rpc_catalog_transition": _COMMON_FIELDS
    | frozenset(
        {
            "command",
            "catalog_revision",
            "accepted",
            "phase",
            "chain",
            "method_names",
            "method_hashes",
            "method_count",
            "draft_method",
            "draft_method_hash",
            "source_evidence_hashes",
            "source_action_value_hashes",
            "source_bindings",
            "finished",
            "producer_action_id",
        }
    ),
    "rpc_schema_provenance": _COMMON_FIELDS
    | frozenset(
        {
            "method",
            "method_hash",
            "catalog_revision",
            "fields",
            "fields_hash",
            "source_evidence_hashes",
            "source_action_value_hashes",
            "source_bindings",
            "producer_action_id",
        }
    ),
    "rpc_method_probe": _COMMON_FIELDS
    | frozenset(
        {
            "method",
            "method_hash",
            "endpoint_hash",
            "request_contract_hash",
            "catalog_revision",
            "evidence_file_hash",
            "probe_contract_hash",
            "ready",
            "probe_status_hash",
            "producer_action_id",
        }
    ),
    "rpc_workload_commit": _COMMON_FIELDS
    | frozenset(
        {
            "case",
            "rpc_mode",
            "choice",
            "methods",
            "method_hashes",
            "mixed_weight_entries",
            "replace_defaults",
            "job_local_override",
            "catalog_revision",
            "fixture_required",
            "fixture_status",
        }
    ),
    "rpc_workload_materialization": _COMMON_FIELDS
    | frozenset(
        {
            "source_kind",
            "chain",
            "adapter_family",
            "rpc_mode",
            "method_hashes",
            "mixed_weight_entries",
            "replace_defaults",
            "required_contract_method_hashes",
            "parameter_contracts",
            "effective_rpc_methods_hash",
            "template_hash",
            "template_hash_scope",
        }
    ),
}
_RECEIPT_OWNERS = {
    "rpc_endpoint_role": "rpc_endpoint",
    "rpc_catalog_transition": "rpc_catalog",
    "rpc_schema_provenance": "rpc_catalog",
    "rpc_method_probe": "rpc_catalog",
    "rpc_workload_commit": "rpc_workload",
    "rpc_workload_materialization": "strategy_planner",
}


def evidence_hash(value: Any) -> str:
    """Return a stable digest without retaining the underlying sensitive value."""

    return hashlib.sha256(
        json.dumps(
            redact(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def exact_value_hash(value: Any) -> str:
    """Match the admission ledger's canonical per-argument commitment."""

    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def validate_rpc_receipt(receipt: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate one RPC owner observation before the coordinator commits it."""

    receipt_type = normalize_scalar(receipt.get("receipt_type"))
    expected_fields = _RECEIPT_FIELDS.get(receipt_type)
    if expected_fields is None:
        return False, "unknown RPC receipt type"
    if set(receipt) != set(expected_fields):
        return False, "unexpected RPC receipt fields"
    if receipt.get("receipt_version") != RPC_RECEIPT_VERSION:
        return False, "unsupported RPC receipt version"
    if receipt.get("owner") != _RECEIPT_OWNERS[receipt_type]:
        return False, "invalid RPC receipt owner"
    if not _is_nonnegative_int(receipt.get("turn_index")):
        return False, "invalid RPC receipt turn index"
    if not _is_hash(receipt.get("receipt_id")):
        return False, "invalid RPC receipt identity"
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if receipt.get("receipt_id") != evidence_hash(unsigned):
        return False, "RPC receipt hash mismatch"

    if receipt_type == "rpc_endpoint_role":
        if receipt.get("role") not in {
            "validation",
            "final_benchmark",
            "mainnet_comparison",
            "sync_observe",
        }:
            return False, "invalid endpoint role"
        if not isinstance(receipt.get("ready"), bool):
            return False, "invalid endpoint readiness"
        if not normalize_scalar(receipt.get("producer_action_id")):
            return False, "invalid endpoint producer action id"
        role = receipt.get("role")
        expected_field = {
            "validation": "",
            "final_benchmark": "LOCAL_RPC_URL",
            "mainnet_comparison": "MAINNET_RPC_URL",
            "sync_observe": "SYNC_OBSERVE_RPC_URL",
        }.get(str(role or ""))
        if receipt.get("config_field") != expected_field:
            return False, "endpoint role/config field mismatch"
        source_kind = receipt.get("source_kind")
        source_receipt_id = receipt.get("source_receipt_id")
        source_value_hash = receipt.get("source_value_hash")
        if role == "validation":
            if (
                source_kind != "direct_action"
                or source_receipt_id
                or not _is_hash(source_value_hash)
            ):
                return False, "invalid validation endpoint source"
        elif source_kind == "direct_action":
            if source_receipt_id or not _is_hash(source_value_hash):
                return False, "invalid direct runtime endpoint source"
        elif source_kind == "validated_endpoint":
            if (
                not _is_hash(source_receipt_id)
                or not _is_hash(source_value_hash)
            ):
                return False, "invalid promoted runtime endpoint source"
        else:
            return False, "invalid runtime endpoint source kind"
        if not _is_hash(receipt.get("endpoint_hash")):
            return False, "invalid endpoint identity"
        if receipt.get("previous_endpoint_hash") and not _is_hash(
            receipt.get("previous_endpoint_hash")
        ):
            return False, "invalid previous endpoint identity"
        if not _is_hash(receipt.get("probe_status_hash")):
            return False, "invalid endpoint probe identity"
        valid, reason = _validate_method_projection(
            receipt,
            "methods",
            "method_hashes",
        )
        if not valid:
            return valid, reason
        bindings = receipt.get("method_evidence_bindings")
        if not isinstance(bindings, list):
            return False, "invalid endpoint method evidence bindings"
        bound_methods: list[str] = []
        for binding in bindings:
            if (
                not isinstance(binding, Mapping)
                or set(binding) != {
                    "method",
                    "method_hash",
                    "evidence_file_hash",
                    "probe_contract_hash",
                }
                or not normalize_scalar(binding.get("method"))
                or binding.get("method_hash")
                != evidence_hash(normalize_scalar(binding.get("method")))
                or not _is_hash(binding.get("evidence_file_hash"))
                or not _is_hash(binding.get("probe_contract_hash"))
            ):
                return False, "invalid endpoint method evidence binding"
            bound_methods.append(normalize_scalar(binding.get("method")))
        if len(bound_methods) != len(set(bound_methods)):
            return False, "duplicate endpoint method evidence binding"
        if bindings:
            if set(bound_methods) != set(receipt.get("methods") or ()):
                return False, "final endpoint evidence coverage mismatch"
        return True, ""

    if receipt_type == "rpc_catalog_transition":
        if not normalize_scalar(receipt.get("command")):
            return False, "missing catalog command"
        if not _is_nonnegative_int(receipt.get("catalog_revision")):
            return False, "invalid catalog revision"
        if not isinstance(receipt.get("accepted"), bool) or not isinstance(
            receipt.get("finished"), bool
        ):
            return False, "invalid catalog flags"
        if not _is_nonnegative_int(receipt.get("method_count")):
            return False, "invalid catalog method count"
        if not normalize_scalar(receipt.get("producer_action_id")):
            return False, "invalid catalog producer action id"
        if not _valid_hash_list(receipt.get("source_evidence_hashes")):
            return False, "invalid catalog source evidence"
        source_hashes = receipt.get("source_evidence_hashes") or []
        source_action_value_hashes = receipt.get(
            "source_action_value_hashes"
        )
        valid_bindings, binding_reason = _validate_source_bindings(
            receipt.get("source_bindings"),
            source_evidence_hashes=source_hashes,
            source_action_value_hashes=source_action_value_hashes,
        )
        if not valid_bindings:
            return False, binding_reason
        if source_hashes and (
            not _valid_hash_list(source_action_value_hashes)
            or not source_action_value_hashes
        ):
            return False, "catalog source evidence lacks admitted input"
        if not source_hashes and source_action_value_hashes:
            return False, "catalog admitted input lacks source evidence"
        valid, reason = _validate_method_projection(
            receipt,
            "method_names",
            "method_hashes",
        )
        if not valid:
            return valid, reason
        if int(receipt["method_count"]) != len(receipt.get("method_hashes") or ()):
            return False, "catalog method count mismatch"
        draft_hash = receipt.get("draft_method_hash")
        if receipt.get("draft_method") and not draft_hash:
            return False, "inconsistent draft identity"
        if receipt.get("draft_method") and evidence_hash(
            receipt.get("draft_method")
        ) != draft_hash:
            return False, "draft method/hash mismatch"
        if draft_hash and not _is_hash(draft_hash):
            return False, "invalid draft method identity"
        return True, ""

    if receipt_type == "rpc_method_probe":
        if (
            not normalize_scalar(receipt.get("method"))
            or evidence_hash(normalize_scalar(receipt.get("method")))
            != receipt.get("method_hash")
            or not _is_hash(receipt.get("endpoint_hash"))
            or not _is_hash(receipt.get("request_contract_hash"))
            or not _is_nonnegative_int(receipt.get("catalog_revision"))
            or not _is_hash(receipt.get("evidence_file_hash"))
            or not _is_hash(receipt.get("probe_contract_hash"))
            or receipt.get("ready") is not True
            or not _is_hash(receipt.get("probe_status_hash"))
            or not normalize_scalar(receipt.get("producer_action_id"))
        ):
            return False, "invalid RPC method probe receipt"
        return True, ""

    if receipt_type == "rpc_schema_provenance":
        if not _is_nonnegative_int(receipt.get("catalog_revision")):
            return False, "invalid schema catalog revision"
        if not normalize_scalar(receipt.get("producer_action_id")):
            return False, "invalid schema producer action id"
        if receipt.get("method") and not receipt.get("method_hash"):
            return False, "inconsistent schema method identity"
        if receipt.get("method") and evidence_hash(
            receipt.get("method")
        ) != receipt.get("method_hash"):
            return False, "schema method/hash mismatch"
        if receipt.get("method_hash") and not _is_hash(receipt.get("method_hash")):
            return False, "invalid schema method identity"
        fields = receipt.get("fields")
        if not isinstance(fields, list) or not fields:
            return False, "schema provenance has no fields"
        for field in fields:
            if not isinstance(field, Mapping) or set(field) != {
                "field_path",
                "source_kind",
                "source_revisions",
                "value_hash",
            }:
                return False, "invalid schema provenance field"
            if (
                not normalize_scalar(field.get("field_path"))
                or not normalize_scalar(field.get("source_kind"))
                or not _is_hash(field.get("value_hash"))
                or not isinstance(field.get("source_revisions"), list)
                or any(
                    not _is_nonnegative_int(revision)
                    for revision in field.get("source_revisions") or ()
                )
            ):
                return False, "invalid schema provenance value"
        if (
            not _is_hash(receipt.get("fields_hash"))
            or receipt.get("fields_hash") != evidence_hash(fields)
            or not _valid_hash_list(receipt.get("source_evidence_hashes"))
            or not receipt.get("source_evidence_hashes")
            or not _valid_hash_list(
                receipt.get("source_action_value_hashes")
            )
            or not receipt.get("source_action_value_hashes")
        ):
            return False, "schema provenance fields hash mismatch"
        valid_bindings, binding_reason = _validate_source_bindings(
            receipt.get("source_bindings"),
            source_evidence_hashes=receipt.get(
                "source_evidence_hashes"
            ),
            source_action_value_hashes=receipt.get(
                "source_action_value_hashes"
            ),
        )
        if not valid_bindings:
            return False, binding_reason
        return True, ""

    if receipt_type == "rpc_workload_commit":
        if receipt.get("rpc_mode") not in {"single", "mixed"}:
            return False, "invalid committed RPC mode"
        if not all(
            isinstance(receipt.get(field), bool)
            for field in (
                "replace_defaults",
                "job_local_override",
                "fixture_required",
            )
        ):
            return False, "invalid workload flags"
        if not _is_nonnegative_int(receipt.get("catalog_revision")):
            return False, "invalid workload catalog revision"
        valid, reason = _validate_method_projection(
            receipt,
            "methods",
            "method_hashes",
        )
        if not valid:
            return valid, reason
        return _validate_weight_entries(
            receipt.get("mixed_weight_entries"),
            require_total=receipt.get("rpc_mode") == "mixed",
        )

    if receipt.get("source_kind") not in {
        "canonical_template_overlay",
        "new_chain_runtime_template",
    }:
        return False, "invalid workload materialization source"
    if receipt.get("rpc_mode") not in {"single", "mixed"}:
        return False, "invalid materialized RPC mode"
    if not isinstance(receipt.get("replace_defaults"), bool):
        return False, "invalid materialization replacement flag"
    for field in (
        "method_hashes",
        "required_contract_method_hashes",
    ):
        if not _is_hash_list(receipt.get(field)):
            return False, f"invalid {field}"
    valid, reason = _validate_weight_entries(
        receipt.get("mixed_weight_entries"),
        require_total=receipt.get("rpc_mode") == "mixed",
    )
    if not valid:
        return valid, reason
    contracts = receipt.get("parameter_contracts")
    if not isinstance(contracts, list) or any(
        not isinstance(item, Mapping)
        or set(item) != {"method_hash", "contract_hash"}
        or not _is_hash(item.get("method_hash"))
        or not _is_hash(item.get("contract_hash"))
        for item in contracts
    ):
        return False, "invalid parameter contracts"
    for field in ("effective_rpc_methods_hash", "template_hash"):
        if not _is_hash(receipt.get(field)):
            return False, f"invalid {field}"
    if (
        receipt.get("template_hash_scope")
        != "template_without_materialization_evidence"
    ):
        return False, "invalid template hash scope"
    return True, ""


def emit_endpoint_role_receipt(
    state: AgentGraphState,
    *,
    role: str,
    case: str,
    config_field: str,
    endpoint: Any,
    previous_endpoint: Any = "",
    ready: bool,
    probe_status: Any,
    chain: Any,
    adapter_family: Any,
    methods: Sequence[Any] = (),
    method_evidence_bindings: Sequence[Mapping[str, Any]] = (),
    source_kind: str = "direct_action",
    source_receipt_id: str = "",
    source_value_hash: str = "",
) -> dict[str, Any]:
    return _append_rpc_receipt(
        state,
        "rpc_endpoint_role",
        {
            "owner": "rpc_endpoint",
            "role": normalize_scalar(role),
            "case": normalize_scalar(case),
            "config_field": normalize_scalar(config_field),
            "endpoint_hash": evidence_hash(normalize_scalar(endpoint)),
            "source_kind": normalize_scalar(source_kind),
            "source_receipt_id": normalize_scalar(source_receipt_id),
            "source_value_hash": (
                normalize_scalar(source_value_hash)
                or admitted_action_value_hash(state)
            ),
            "previous_endpoint_hash": (
                evidence_hash(normalize_scalar(previous_endpoint))
                if normalize_scalar(previous_endpoint)
                else ""
            ),
            "ready": bool(ready),
            "probe_status_hash": evidence_hash(normalize_scalar(probe_status)),
            "chain": normalize_scalar(chain),
            "adapter_family": normalize_scalar(adapter_family),
            "methods": _method_names(methods),
            "method_hashes": _method_hashes(methods),
            "method_evidence_bindings": [
                {
                    "method": normalize_scalar(item.get("method")),
                    "method_hash": evidence_hash(
                        normalize_scalar(item.get("method"))
                    ),
                    "evidence_file_hash": normalize_scalar(
                        item.get("evidence_file_hash")
                    ),
                    "probe_contract_hash": normalize_scalar(
                        item.get("probe_contract_hash")
                    ),
                }
                for item in method_evidence_bindings
                if isinstance(item, Mapping)
            ],
            "producer_action_id": normalize_scalar(
                (state.get("current_action") or {}).get("action_id")
            ),
        },
    )


def emit_catalog_transition_receipt(
    state: AgentGraphState,
    *,
    command: str,
    revision: int,
    accepted: bool,
    phase: str,
    catalog: Mapping[str, Any],
) -> None:
    methods = [
        item.get("method")
        for item in catalog.get("methods") or ()
        if isinstance(item, Mapping)
    ]
    draft = catalog.get("draft") if isinstance(catalog.get("draft"), Mapping) else {}
    source_evidence_hashes = _source_evidence_hashes(draft)
    _append_rpc_receipt(
        state,
        "rpc_catalog_transition",
        {
            "owner": "rpc_catalog",
            "command": normalize_scalar(command),
            "catalog_revision": int(revision),
            "accepted": bool(accepted),
            "phase": normalize_scalar(phase),
            "chain": normalize_scalar(catalog.get("chain")),
            "method_names": _method_names(methods),
            "method_hashes": _method_hashes(methods),
            "method_count": len(_method_hashes(methods)),
            "draft_method": (
                normalize_scalar(draft.get("method"))
                if looks_like_rpc_method_token(draft.get("method"))
                else ""
            ),
            "draft_method_hash": (
                evidence_hash(normalize_scalar(draft.get("method")))
                if normalize_scalar(draft.get("method"))
                else ""
            ),
            "source_evidence_hashes": source_evidence_hashes,
            "source_action_value_hashes": _source_action_value_hashes(
                draft
            ),
            "source_bindings": _source_bindings(draft),
            "finished": bool(catalog.get("finished")),
            "producer_action_id": normalize_scalar(
                (state.get("current_action") or {}).get("action_id")
            ),
        },
    )


def emit_schema_provenance_receipt(
    state: AgentGraphState,
    *,
    catalog_revision: int,
    draft: Mapping[str, Any],
) -> None:
    fields = []
    for item in draft.get("field_provenance") or ():
        if not isinstance(item, Mapping):
            continue
        path = normalize_scalar(item.get("field_path"))
        source_kind = normalize_scalar(item.get("source_kind"))
        found, field_value = _draft_value_at_path(draft, path)
        if not path or not source_kind or not found:
            continue
        fields.append(
            {
                "field_path": path,
                "source_kind": source_kind,
                "source_revisions": sorted(
                    {
                        int(revision)
                        for revision in item.get("source_revisions") or ()
                        if isinstance(revision, int) and revision >= 0
                    }
                ),
                "value_hash": evidence_hash(field_value),
            }
        )
    source_evidence_hashes = _source_evidence_hashes(draft)
    source_action_value_hashes = _source_action_value_hashes(draft)
    if (
        not fields
        or not source_evidence_hashes
        or not source_action_value_hashes
    ):
        return
    method = draft.get("method")
    _append_rpc_receipt(
        state,
        "rpc_schema_provenance",
        {
            "owner": "rpc_catalog",
            "method": (
                normalize_scalar(method)
                if looks_like_rpc_method_token(method)
                else ""
            ),
            "method_hash": (
                evidence_hash(normalize_scalar(method))
                if normalize_scalar(method)
                else ""
            ),
            "catalog_revision": int(catalog_revision),
            "fields": fields,
            "fields_hash": evidence_hash(fields),
            "source_evidence_hashes": source_evidence_hashes,
            "source_action_value_hashes": source_action_value_hashes,
            "source_bindings": _source_bindings(draft),
            "producer_action_id": normalize_scalar(
                (state.get("current_action") or {}).get("action_id")
            ),
        },
    )


def emit_method_probe_receipt(
    state: AgentGraphState,
    *,
    method: str,
    endpoint_hash: str,
    request_contract_hash: str,
    catalog_revision: int,
    evidence_file_hash: str,
    probe_contract_hash: str,
    ready: bool,
    probe_status: str,
) -> dict[str, Any]:
    """Bind a successful method probe to its endpoint, contract, and evidence."""

    if not ready:
        return {}
    return _append_rpc_receipt(
        state,
        "rpc_method_probe",
        {
            "owner": "rpc_catalog",
            "method": normalize_scalar(method),
            "method_hash": evidence_hash(normalize_scalar(method)),
            "endpoint_hash": normalize_scalar(endpoint_hash),
            "request_contract_hash": normalize_scalar(
                request_contract_hash
            ),
            "catalog_revision": int(catalog_revision),
            "evidence_file_hash": normalize_scalar(evidence_file_hash),
            "probe_contract_hash": normalize_scalar(probe_contract_hash),
            "ready": True,
            "probe_status_hash": evidence_hash(
                normalize_scalar(probe_status)
            ),
            "producer_action_id": normalize_scalar(
                (state.get("current_action") or {}).get("action_id")
            ),
        },
    )


def emit_workload_commit_receipt(
    state: AgentGraphState,
    *,
    case: str,
) -> None:
    workload = state.get("workload") if isinstance(state.get("workload"), Mapping) else {}
    if not workload.get("confirmed"):
        return
    weights = workload.get("mixed_weights") if isinstance(workload.get("mixed_weights"), Mapping) else {}
    fixture = state.get("fixture_evidence") if isinstance(state.get("fixture_evidence"), Mapping) else {}
    catalog = (
        (state.get("custom_rpc") or {}).get("catalog")
        if isinstance(state.get("custom_rpc"), Mapping)
        else {}
    )
    _append_rpc_receipt(
        state,
        "rpc_workload_commit",
        {
            "owner": "rpc_workload",
            "case": normalize_scalar(case),
            "rpc_mode": normalize_scalar(state.get("rpc_mode")),
            "choice": normalize_scalar(workload.get("choice")),
            "methods": _method_names(workload.get("methods") or ()),
            "method_hashes": _method_hashes(workload.get("methods") or ()),
            "mixed_weight_entries": _weight_entries(weights),
            "replace_defaults": bool(workload.get("replace_defaults")),
            "job_local_override": bool(workload.get("job_local_override")),
            "catalog_revision": int((catalog or {}).get("revision") or 0),
            "fixture_required": bool(fixture.get("required")),
            "fixture_status": normalize_scalar(fixture.get("status")),
        },
    )


def emit_materialization_receipt(
    state: AgentGraphState,
    evidence: Mapping[str, Any],
) -> None:
    """Project evidence produced by the strategy materializer into this turn."""

    if not evidence:
        return
    _append_rpc_receipt(
        state,
        "rpc_workload_materialization",
        {
            "owner": "strategy_planner",
            "source_kind": normalize_scalar(evidence.get("source_kind")),
            "chain": normalize_scalar(evidence.get("chain")),
            "adapter_family": normalize_scalar(evidence.get("adapter_family")),
            "rpc_mode": normalize_scalar(evidence.get("rpc_mode")),
            "method_hashes": _valid_hashes(evidence.get("method_hashes") or ()),
            "mixed_weight_entries": _materialized_weight_entries(
                evidence.get("mixed_weight_entries") or ()
            ),
            "replace_defaults": bool(evidence.get("replace_defaults")),
            "required_contract_method_hashes": _valid_hashes(
                evidence.get("required_contract_method_hashes") or ()
            ),
            "parameter_contracts": _materialized_parameter_contracts(
                evidence.get("parameter_contracts") or ()
            ),
            "effective_rpc_methods_hash": normalize_scalar(
                evidence.get("effective_rpc_methods_hash")
            ),
            "template_hash": normalize_scalar(evidence.get("template_hash")),
            "template_hash_scope": normalize_scalar(
                evidence.get("template_hash_scope")
            ),
        },
    )


def _append_rpc_receipt(
    state: AgentGraphState,
    receipt_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    turn_context = state.get("turn_context")
    if not isinstance(turn_context, Mapping) or not turn_context:
        return {}
    body = {
        "receipt_type": receipt_type,
        "receipt_version": RPC_RECEIPT_VERSION,
        "turn_index": int(state.get("turn_index") or 0),
        **deepcopy(dict(payload)),
    }
    body["receipt_id"] = evidence_hash(body)
    valid, _ = validate_rpc_receipt(body)
    if not valid:
        return {}
    context = dict(turn_context)
    receipts = [
        dict(item)
        for item in context.get("control_receipts") or ()
        if isinstance(item, Mapping)
    ]
    if not any(item.get("receipt_id") == body["receipt_id"] for item in receipts):
        receipts.append(body)
    context["control_receipts"] = receipts
    state["turn_context"] = context
    return body


def _method_names(values: Sequence[Any]) -> list[str]:
    return list(
        dict.fromkeys(
            method
            for value in values
            if (method := normalize_scalar(value))
            and looks_like_rpc_method_token(method)
        )
    )


def _source_evidence_hashes(draft: Mapping[str, Any]) -> list[str]:
    return sorted({
        evidence_hash({
            "revision": int(item.get("revision") or 0),
            "source": normalize_scalar(item.get("source")),
            "kind": normalize_scalar(item.get("kind")),
            "method": normalize_scalar(item.get("method")),
            "content": str(item.get("content") or "").strip(),
        })
        for item in draft.get("evidence") or ()
        if isinstance(item, Mapping)
        and str(item.get("content") or "").strip()
    })


def admitted_action_value_hash(
    state: Mapping[str, Any],
    *,
    argument_names: Sequence[str] = (
        "rpc_endpoint",
        "selected_value",
        "answer",
    ),
    expected_value: Any = None,
) -> str:
    action_id = normalize_scalar(
        (state.get("current_action") or {}).get("action_id")
    )
    for action in (state.get("turn_context") or {}).get(
        "admitted_actions"
    ) or ():
        if (
            isinstance(action, Mapping)
            and normalize_scalar(action.get("action_id")) == action_id
        ):
            hashes = action.get("argument_value_hashes")
            if not isinstance(hashes, Mapping):
                return ""
            expected_hash = (
                exact_value_hash(expected_value)
                if expected_value is not None
                else ""
            )
            for name in argument_names:
                value_hash = normalize_scalar(hashes.get(name))
                if _is_hash(value_hash) and (
                    not expected_hash or value_hash == expected_hash
                ):
                    return value_hash
    return ""


def _source_action_value_hashes(draft: Mapping[str, Any]) -> list[str]:
    return sorted({
        value_hash
        for item in draft.get("evidence") or ()
        if isinstance(item, Mapping)
        and _is_hash(
            value_hash := normalize_scalar(
                item.get("source_action_value_hash")
            )
        )
    })


def _source_bindings(draft: Mapping[str, Any]) -> list[dict[str, str]]:
    bindings = []
    for item in draft.get("evidence") or ():
        if not isinstance(item, Mapping):
            continue
        content = str(item.get("content") or "").strip()
        action_hash = normalize_scalar(
            item.get("source_action_value_hash")
        )
        if not content or not _is_hash(action_hash):
            continue
        bindings.append(
            {
                "evidence_hash": evidence_hash(
                    {
                        "revision": int(item.get("revision") or 0),
                        "source": normalize_scalar(item.get("source")),
                        "kind": normalize_scalar(item.get("kind")),
                        "method": normalize_scalar(item.get("method")),
                        "content": content,
                    }
                ),
                "action_value_hash": action_hash,
            }
        )
    return sorted(bindings, key=lambda item: item["evidence_hash"])


def _validate_source_bindings(
    value: Any,
    *,
    source_evidence_hashes: Any,
    source_action_value_hashes: Any,
) -> tuple[bool, str]:
    if not isinstance(value, list):
        return False, "RPC source bindings are missing"
    evidence_hashes: list[str] = []
    action_hashes: list[str] = []
    for binding in value:
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"evidence_hash", "action_value_hash"}
            or not _is_hash(binding.get("evidence_hash"))
            or not _is_hash(binding.get("action_value_hash"))
        ):
            return False, "RPC source binding is invalid"
        evidence_hashes.append(str(binding["evidence_hash"]))
        action_hashes.append(str(binding["action_value_hash"]))
    if evidence_hashes != sorted(evidence_hashes) or len(
        evidence_hashes
    ) != len(set(evidence_hashes)):
        return False, "RPC source bindings are not canonical"
    if evidence_hashes != list(source_evidence_hashes or ()):
        return False, "RPC source evidence binding mismatch"
    if len(action_hashes) != len(set(action_hashes)):
        return False, "RPC admitted input binding is not one-to-one"
    if sorted(set(action_hashes)) != list(
        source_action_value_hashes or ()
    ):
        return False, "RPC admitted input binding mismatch"
    return True, ""


_PATH_TOKEN_RE = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")


def _draft_value_at_path(
    draft: Mapping[str, Any],
    path: str,
) -> tuple[bool, Any]:
    current: Any = draft
    tokens = [
        int(index) if index else key
        for key, index in _PATH_TOKEN_RE.findall(path)
    ]
    if not tokens:
        return False, None
    for token in tokens:
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                return False, None
            current = current[token]
        else:
            if not isinstance(current, Mapping) or token not in current:
                return False, None
            current = current[token]
    return True, current


def _valid_hash_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and value == sorted(set(value))
        and all(_is_hash(item) for item in value)
    )


def _method_hashes(values: Sequence[Any]) -> list[str]:
    return list(
        dict.fromkeys(
            evidence_hash(method)
            for value in values
            if (method := normalize_scalar(value))
        )
    )


def _weight_entries(weights: Mapping[Any, Any]) -> list[dict[str, Any]]:
    entries = []
    for method, weight in sorted(weights.items(), key=lambda item: str(item[0])):
        identity = normalize_scalar(method)
        if (
            not identity
            or not isinstance(weight, int)
            or isinstance(weight, bool)
        ):
            continue
        entries.append(
            {
                "method": identity if looks_like_rpc_method_token(identity) else "",
                "method_hash": evidence_hash(identity),
                "weight": int(weight),
            }
        )
    return entries


def _valid_hashes(values: Sequence[Any]) -> list[str]:
    return list(
        dict.fromkeys(
            value_hash
            for value in values
            if len(value_hash := normalize_scalar(value)) == 64
        )
    )


def _materialized_weight_entries(values: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "method_hash": normalize_scalar(item.get("method_hash")),
            "weight": int(item.get("weight")),
        }
        for item in values
        if isinstance(item, Mapping)
        and len(normalize_scalar(item.get("method_hash"))) == 64
        and isinstance(item.get("weight"), int)
        and not isinstance(item.get("weight"), bool)
    ]


def _materialized_parameter_contracts(
    values: Sequence[Any],
) -> list[dict[str, str]]:
    return [
        {
            "method_hash": normalize_scalar(item.get("method_hash")),
            "contract_hash": normalize_scalar(item.get("contract_hash")),
        }
        for item in values
        if isinstance(item, Mapping)
        and len(normalize_scalar(item.get("method_hash"))) == 64
        and len(normalize_scalar(item.get("contract_hash"))) == 64
    ]


def _validate_method_projection(
    receipt: Mapping[str, Any],
    names_field: str,
    hashes_field: str,
) -> tuple[bool, str]:
    names = receipt.get(names_field)
    hashes = receipt.get(hashes_field)
    if not isinstance(names, list) or any(
        not isinstance(item, str) or not item for item in names
    ):
        return False, f"invalid {names_field}"
    if not _is_hash_list(hashes):
        return False, f"invalid {hashes_field}"
    if len(set(names)) != len(names) or len(set(hashes)) != len(hashes):
        return False, "duplicate method identity"
    if any(evidence_hash(name) not in hashes for name in names):
        return False, "method name/hash mismatch"
    return True, ""


def _validate_weight_entries(
    value: Any,
    *,
    require_total: bool,
) -> tuple[bool, str]:
    if not isinstance(value, list):
        return False, "invalid mixed weight entries"
    total = 0
    identities: set[str] = set()
    for item in value:
        if (
            not isinstance(item, Mapping)
            or set(item) - {"method", "method_hash", "weight"}
            or not _is_hash(item.get("method_hash"))
            or not isinstance(item.get("weight"), int)
            or isinstance(item.get("weight"), bool)
            or int(item["weight"]) <= 0
        ):
            return False, "invalid mixed weight entry"
        if item.get("method") and evidence_hash(item["method"]) != item["method_hash"]:
            return False, "mixed weight method/hash mismatch"
        if item["method_hash"] in identities:
            return False, "duplicate mixed weight method"
        identities.add(str(item["method_hash"]))
        total += int(item["weight"])
    if require_total and total != 100:
        return False, "mixed weight total must equal 100"
    if not require_total and value:
        return False, "single mode cannot carry mixed weights"
    return True, ""


def _is_hash_list(value: Any) -> bool:
    return isinstance(value, list) and all(_is_hash(item) for item in value)


def _is_hash(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
