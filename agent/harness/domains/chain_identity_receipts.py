"""Evidence receipts for unknown-chain identity research."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from ..state import AgentGraphState


CHAIN_IDENTITY_RECEIPT_SCHEMA_VERSION = 1
_CHAIN_EXISTS_VALUES = {"true", "false", "unknown"}
_RESOLVER_SOURCES = {"llm", "planner_proposal"}


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


def _append(state: AgentGraphState, receipt: Mapping[str, Any]) -> None:
    turn_context = dict(state.get("turn_context") or {})
    receipts = [
        deepcopy(dict(item))
        for item in turn_context.get("control_receipts") or ()
        if isinstance(item, Mapping)
    ]
    if not any(
        item.get("receipt_id") == receipt.get("receipt_id")
        for item in receipts
    ):
        receipts.append(deepcopy(dict(receipt)))
    turn_context["control_receipts"] = receipts
    state["turn_context"] = turn_context


def emit_chain_identity_resolution_receipt(
    state: AgentGraphState,
    *,
    candidate: str,
    resolution: Mapping[str, Any],
    resolver_source: str,
    confirmation_required: bool,
) -> dict[str, Any]:
    """Record model/search provenance without retaining candidate text."""

    chain_exists = resolution.get("chain_exists")
    exists_value = (
        "true"
        if chain_exists is True
        else "false"
        if chain_exists is False
        else "unknown"
    )
    search = (
        dict(resolution.get("search_result") or {})
        if isinstance(resolution.get("search_result"), Mapping)
        else {}
    )
    body = {
        "receipt_type": "chain_identity_resolution",
        "schema_version": CHAIN_IDENTITY_RECEIPT_SCHEMA_VERSION,
        "owner": "chain_rpc",
        "turn_index": int(state.get("turn_index") or 0),
        "candidate_hash": _hash(str(candidate or "")),
        "resolver_source": str(resolver_source or ""),
        "chain_exists": exists_value,
        "canonical_name_hash": _hash(
            str(resolution.get("canonical_chain_name") or "")
        ),
        "possible_known_chain_hash": _hash(
            str(resolution.get("possible_known_chain") or "")
        ),
        "adapter_family": str(resolution.get("adapter_family") or "unknown"),
        "confidence": str(resolution.get("confidence") or "low"),
        "google_search_invoked": bool(search),
        "google_search_available": search.get("available") is True,
        "search_evidence_hash": _hash(
            str(search.get("text_summary") or "")
        ),
        "confirmation_required": bool(confirmation_required),
    }
    receipt = {**body, "receipt_id": _hash(body)}
    _append(state, receipt)
    return receipt


def validate_chain_identity_receipt(
    receipt: Mapping[str, Any],
) -> tuple[bool, str]:
    """Validate exact identity-resolution provenance."""

    expected = {
        "receipt_type",
        "schema_version",
        "owner",
        "turn_index",
        "candidate_hash",
        "resolver_source",
        "chain_exists",
        "canonical_name_hash",
        "possible_known_chain_hash",
        "adapter_family",
        "confidence",
        "google_search_invoked",
        "google_search_available",
        "search_evidence_hash",
        "confirmation_required",
        "receipt_id",
    }
    if set(receipt) != expected:
        return False, "chain-identity receipt shape is invalid"
    if (
        receipt.get("receipt_type") != "chain_identity_resolution"
        or receipt.get("schema_version")
        != CHAIN_IDENTITY_RECEIPT_SCHEMA_VERSION
        or receipt.get("owner") != "chain_rpc"
        or not isinstance(receipt.get("turn_index"), int)
        or isinstance(receipt.get("turn_index"), bool)
        or receipt.get("resolver_source") not in _RESOLVER_SOURCES
        or receipt.get("chain_exists") not in _CHAIN_EXISTS_VALUES
        or not str(receipt.get("adapter_family") or "")
        or not str(receipt.get("confidence") or "")
        or not isinstance(receipt.get("google_search_invoked"), bool)
        or not isinstance(receipt.get("google_search_available"), bool)
        or not isinstance(receipt.get("confirmation_required"), bool)
        or receipt.get("confirmation_required") is not True
        or (
            receipt.get("google_search_available") is True
            and receipt.get("google_search_invoked") is not True
        )
    ):
        return False, "chain-identity receipt semantics are invalid"
    for field in (
        "candidate_hash",
        "canonical_name_hash",
        "possible_known_chain_hash",
        "search_evidence_hash",
        "receipt_id",
    ):
        value = str(receipt.get(field) or "")
        if (
            len(value) != 64
            or value != value.lower()
            or any(character not in "0123456789abcdef" for character in value)
        ):
            return False, f"chain-identity {field} is invalid"
    unsigned = {
        str(key): value
        for key, value in receipt.items()
        if str(key) != "receipt_id"
    }
    if receipt.get("receipt_id") != _hash(unsigned):
        return False, "chain-identity receipt identity is stale"
    return True, ""
