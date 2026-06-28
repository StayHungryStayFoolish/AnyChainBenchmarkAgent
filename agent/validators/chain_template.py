"""Chain template validation wrappers for ADK tools."""

from __future__ import annotations

from typing import Any

from planners.chain_template_requirements import inspect_chain_template


def validate_chain_template(chain: str) -> dict[str, Any]:
    """Return chain-template readiness facts needed before execution."""
    details = inspect_chain_template(chain)
    errors = []
    if not details.get("exists"):
        errors.append("chain template does not exist")
    if details.get("exists") and not details.get("adapter_family"):
        errors.append("_meta.adapter_family is missing")
    weighted = details.get("mixed_weighted", [])
    if weighted and sum(int(item.get("weight", 0) or 0) for item in weighted) != 100:
        errors.append("rpc_methods.mixed_weighted total must equal 100")
    return {
        **details,
        "ready": not errors,
        "errors": errors,
    }
