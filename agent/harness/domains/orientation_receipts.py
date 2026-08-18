"""Secret-free receipts for authoritative orientation responses."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from agent.runners.job_manager import verify_job_receipt

from ..contracts import ActionProposal, ResponseFragment
from ..response_catalog import render_fragment, semantic_hash
from ..state import AgentGraphState


ORIENTATION_RECEIPT_SCHEMA_VERSION = 3
ORIENTATION_PROJECTION_FIELDS = frozenset({
    "target_mode",
    "workflow_mode",
    "chain",
    "rpc_mode",
    "qps_mode",
    "observability_mode",
    "confirmed_fields",
    "pending_id",
    "job_id",
    "job_status",
})


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _projection(state: Mapping[str, Any]) -> dict[str, Any]:
    identity = state.get("chain_identity") or {}
    qps = state.get("qps_profile") or {}
    observability = state.get("observability") or {}
    job = state.get("job") or {}
    return {
        "target_mode": str(state.get("target_mode") or ""),
        "workflow_mode": str(state.get("workflow_mode") or ""),
        "chain": str(identity.get("canonical") or identity.get("raw") or ""),
        "rpc_mode": str(state.get("rpc_mode") or ""),
        "qps_mode": str(qps.get("mode") or ""),
        "observability_mode": str(observability.get("mode") or ""),
        "confirmed_fields": sorted(
            str(field)
            for field in (state.get("confirmed_config") or {})
            if str(field)
        ),
        "pending_id": str((state.get("pending_question") or {}).get("id") or ""),
        "job_id": str(job.get("job_id") or ""),
        "job_status": str(job.get("status") or ""),
    }


def build_orientation_response_receipt(
    state: AgentGraphState,
    action: ActionProposal,
    *,
    topic: str,
    response_fragments: Sequence[ResponseFragment],
    source_receipts: Sequence[Mapping[str, Any]] = (),
    projection_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind one read-only answer to its action, pending state, and snapshot."""

    projection = _projection(state)
    if projection_override:
        projection.update({
            str(key): value
            for key, value in projection_override.items()
            if str(key) in projection
        })
    evidence = [dict(receipt) for receipt in source_receipts]
    body = {
        "receipt_type": "orientation_response",
        "schema_version": ORIENTATION_RECEIPT_SCHEMA_VERSION,
        "owner": "orientation",
        "turn_index": int(state.get("turn_index") or 0),
        "topic": str(topic or ""),
        "action_type": str(action.action_type or ""),
        "action_id": str(action.action_id or ""),
        "pending_contract_hash": _hash(state.get("pending_question") or {}),
        "response_hash": semantic_hash([
            render_fragment(
                fragment,
                str(state.get("language") or "en"),
            ).semantic_hash
            for fragment in response_fragments
        ]),
        "state_projection": projection,
        "projection_fields": sorted(projection),
        "state_projection_hash": _hash(projection),
        "source_receipts": evidence,
        "read_only": True,
    }
    return {**body, "receipt_id": _hash(body)}


def validate_orientation_receipt(
    receipt: Mapping[str, Any],
) -> tuple[bool, str]:
    """Validate exact schema and content-addressed identity."""

    expected = {
        "receipt_type",
        "schema_version",
        "owner",
        "turn_index",
        "topic",
        "action_type",
        "action_id",
        "pending_contract_hash",
        "response_hash",
        "state_projection",
        "projection_fields",
        "state_projection_hash",
        "source_receipts",
        "read_only",
        "receipt_id",
    }
    if set(receipt) != expected:
        return False, "orientation receipt shape is invalid"
    if (
        receipt.get("receipt_type") != "orientation_response"
        or receipt.get("schema_version") != ORIENTATION_RECEIPT_SCHEMA_VERSION
        or receipt.get("owner") != "orientation"
        or not isinstance(receipt.get("turn_index"), int)
        or isinstance(receipt.get("turn_index"), bool)
        or not str(receipt.get("topic") or "")
        or receipt.get("action_type")
        not in {"greeting", "ask_capabilities", "answer_opening_question"}
        or not str(receipt.get("action_id") or "")
        or receipt.get("read_only") is not True
    ):
        return False, "orientation receipt semantics are invalid"
    fields = receipt.get("projection_fields")
    projection = receipt.get("state_projection")
    if (
        not isinstance(projection, Mapping)
        or not isinstance(fields, list)
        or set(projection) != ORIENTATION_PROJECTION_FIELDS
        or fields != sorted(ORIENTATION_PROJECTION_FIELDS)
        or not all(isinstance(field, str) and field for field in fields)
        or fields != sorted(set(fields))
        or fields != sorted(str(field) for field in projection)
        or receipt.get("state_projection_hash") != _hash(projection)
    ):
        return False, "orientation projection fields are invalid"
    if (
        not all(
            isinstance(projection.get(field), str)
            for field in ORIENTATION_PROJECTION_FIELDS - {"confirmed_fields"}
        )
        or not isinstance(projection.get("confirmed_fields"), list)
        or projection.get("confirmed_fields")
        != sorted(set(projection.get("confirmed_fields") or ()))
        or not all(
            isinstance(field, str) and field
            for field in projection.get("confirmed_fields") or ()
        )
    ):
        return False, "orientation projection values are invalid"
    source_receipts = receipt.get("source_receipts")
    if (
        not isinstance(source_receipts, list)
        or len(source_receipts) > 1
        or any(
            not isinstance(source_receipt, dict)
            or not verify_job_receipt(source_receipt)
            or source_receipt.get("receipt_type") != "job_read"
            for source_receipt in source_receipts
        )
        or (
            source_receipts
            and receipt.get("topic")
            not in {
                "current_context",
                "next_action",
                "current_job",
                "job_status",
                "execution_status",
            }
        )
    ):
        return False, "orientation source receipts are invalid"
    if source_receipts:
        [source_receipt] = source_receipts
        if (
            source_receipt.get("job_id") != projection.get("job_id")
            or source_receipt.get("observed_status")
            != source_receipt.get("persisted_status")
            or source_receipt.get("observed_status")
            != projection.get("job_status")
        ):
            return False, "orientation job source does not match state projection"
    for field in (
        "pending_contract_hash",
        "response_hash",
        "state_projection_hash",
        "receipt_id",
    ):
        value = str(receipt.get(field) or "")
        if (
            len(value) != 64
            or value != value.lower()
            or any(character not in "0123456789abcdef" for character in value)
        ):
            return False, f"orientation {field} is invalid"
    unsigned = {
        str(key): value
        for key, value in receipt.items()
        if str(key) != "receipt_id"
    }
    if receipt.get("receipt_id") != _hash(unsigned):
        return False, "orientation receipt identity is stale"
    return True, ""
