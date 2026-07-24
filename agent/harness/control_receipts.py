"""Trusted admission boundary for domain-owned observation receipts."""

from __future__ import annotations

from typing import Any, Mapping

from .domains.analysis_receipts import validate_analysis_receipt
from .domains.rpc_receipts import validate_rpc_receipt


_HANDLER_OWNER_BY_RECEIPT = {
    "analysis_evidence_block": "analysis",
    "analysis_invocation": "analysis",
    "analysis_report": "analysis",
    "rpc_endpoint_role": "chain_rpc",
    "rpc_catalog_transition": "chain_rpc",
    "rpc_schema_provenance": "chain_rpc",
    "rpc_workload_commit": "chain_rpc",
    "rpc_workload_materialization": "execution",
}


def validate_domain_control_receipt(
    receipt: Mapping[str, Any],
    *,
    handler_owner: str,
    turn_index: int,
) -> tuple[bool, str]:
    """Validate identity, semantic schema, and handler ownership."""

    receipt_type = str(receipt.get("receipt_type") or "")
    expected_handler = _HANDLER_OWNER_BY_RECEIPT.get(receipt_type)
    if expected_handler is None:
        return False, "unregistered domain receipt type"
    if handler_owner != expected_handler:
        return False, "domain receipt crossed the wrong handler boundary"
    if receipt.get("turn_index") != turn_index:
        return False, "domain receipt belongs to a different turn"
    if receipt_type.startswith("analysis_"):
        return validate_analysis_receipt(receipt)
    return validate_rpc_receipt(receipt)


def validate_persisted_domain_control_receipt(
    receipt: Mapping[str, Any],
    *,
    turn_index: int,
) -> tuple[bool, str]:
    """Validate a persisted owner receipt without trusting event metadata."""

    receipt_type = str(receipt.get("receipt_type") or "")
    handler_owner = _HANDLER_OWNER_BY_RECEIPT.get(receipt_type)
    if handler_owner is None:
        return True, ""
    return validate_domain_control_receipt(
        receipt,
        handler_owner=handler_owner,
        turn_index=turn_index,
    )
