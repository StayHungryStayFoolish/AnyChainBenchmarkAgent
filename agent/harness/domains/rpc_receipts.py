"""Secret-free observation receipts emitted by authoritative RPC owners."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping, Sequence

from agent.utils.redaction import redact

from ..input_values import looks_like_rpc_method_token, normalize_scalar
from ..state import AgentGraphState


RPC_RECEIPT_VERSION = 1
_COMMON_FIELDS = frozenset(
    {"receipt_type", "receipt_version", "turn_index", "owner", "receipt_id"}
)
_RECEIPT_FIELDS = {
    "rpc_endpoint_role": _COMMON_FIELDS
    | frozenset(
        {
            "role",
            "case",
            "endpoint_hash",
            "previous_endpoint_hash",
            "ready",
            "probe_status_hash",
            "chain",
            "adapter_family",
            "methods",
            "method_hashes",
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
            "finished",
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
            "sync_observe",
        }:
            return False, "invalid endpoint role"
        if not isinstance(receipt.get("ready"), bool):
            return False, "invalid endpoint readiness"
        if not _is_hash(receipt.get("endpoint_hash")):
            return False, "invalid endpoint identity"
        if receipt.get("previous_endpoint_hash") and not _is_hash(
            receipt.get("previous_endpoint_hash")
        ):
            return False, "invalid previous endpoint identity"
        if not _is_hash(receipt.get("probe_status_hash")):
            return False, "invalid endpoint probe identity"
        return _validate_method_projection(receipt, "methods", "method_hashes")

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

    if receipt_type == "rpc_schema_provenance":
        if not _is_nonnegative_int(receipt.get("catalog_revision")):
            return False, "invalid schema catalog revision"
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
        ):
            return False, "schema provenance fields hash mismatch"
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
    endpoint: Any,
    previous_endpoint: Any = "",
    ready: bool,
    probe_status: Any,
    chain: Any,
    adapter_family: Any,
    methods: Sequence[Any] = (),
) -> None:
    _append_rpc_receipt(
        state,
        "rpc_endpoint_role",
        {
            "owner": "rpc_endpoint",
            "role": normalize_scalar(role),
            "case": normalize_scalar(case),
            "endpoint_hash": evidence_hash(normalize_scalar(endpoint)),
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
            "finished": bool(catalog.get("finished")),
        },
    )


def emit_schema_provenance_receipt(
    state: AgentGraphState,
    *,
    method: Any,
    catalog_revision: int,
    field_provenance: Sequence[Mapping[str, Any]],
) -> None:
    fields = []
    for item in field_provenance:
        path = normalize_scalar(item.get("field_path"))
        source_kind = normalize_scalar(item.get("source_kind"))
        value_hash = normalize_scalar(item.get("value_hash"))
        if not path or not source_kind or len(value_hash) != 64:
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
                "value_hash": value_hash,
            }
        )
    if not fields:
        return
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
) -> None:
    turn_context = state.get("turn_context")
    if not isinstance(turn_context, Mapping) or not turn_context:
        return
    body = {
        "receipt_type": receipt_type,
        "receipt_version": RPC_RECEIPT_VERSION,
        "turn_index": int(state.get("turn_index") or 0),
        **deepcopy(dict(payload)),
    }
    body["receipt_id"] = evidence_hash(body)
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


def _method_names(values: Sequence[Any]) -> list[str]:
    return list(
        dict.fromkeys(
            method
            for value in values
            if (method := normalize_scalar(value))
            and looks_like_rpc_method_token(method)
        )
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
