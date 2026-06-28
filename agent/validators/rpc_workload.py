"""RPC workload validators for single and mixed benchmark modes."""

from __future__ import annotations

from typing import Any

from knowledge.framework_capabilities import load_framework_capabilities


def validate_rpc_workload(
    chain: str,
    rpc_mode: str,
    methods: list[str] | None = None,
    mixed_weights: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Validate RPC method selection and mixed weights against repo facts."""
    chain = (chain or "").strip().lower()
    rpc_mode = (rpc_mode or "").strip().lower()
    methods = [method.strip() for method in (methods or []) if method.strip()]
    mixed_weights = mixed_weights or {}
    chain_data = _chain_data(chain)
    supported_methods = set(chain_data.get("methods", [])) if chain_data else set()
    errors = []
    warnings = []
    if rpc_mode not in {"single", "mixed"}:
        errors.append("rpc_mode must be single or mixed")
    if rpc_mode == "single" and len(methods) > 1:
        errors.append("single mode accepts exactly one method")
    if rpc_mode == "mixed":
        if not mixed_weights:
            errors.append("mixed mode requires mixed_weights")
        total = sum(int(value) for value in mixed_weights.values())
        if total != 100:
            errors.append(f"mixed_weights total must be 100, got {total}")
        methods = list(mixed_weights.keys())
    if not methods:
        errors.append("at least one RPC method is required")
    custom_methods = [method for method in methods if supported_methods and method not in supported_methods]
    if custom_methods:
        warnings.append("custom RPC methods require param contract, fake-node fixture, and proxy attribution validation")
    return {
        "chain": chain,
        "rpc_mode": rpc_mode,
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "methods": methods,
        "custom_methods": custom_methods,
        "supported_methods": sorted(supported_methods),
        "requires_fixture_review": bool(custom_methods),
    }


def default_workload(chain: str) -> dict[str, Any]:
    """Return chain-template single and mixed workload defaults."""
    data = _chain_data(chain)
    if not data:
        return {"chain": chain, "exists": False, "single": "", "mixed_weighted": []}
    return {
        "chain": chain,
        "exists": True,
        "single": data.get("single", ""),
        "mixed_weighted": data.get("mixed_weighted", []),
        "methods": data.get("methods", []),
    }


def _chain_data(chain: str) -> dict[str, Any]:
    capabilities = load_framework_capabilities()
    return next((row for row in capabilities.get("chains", []) if row.get("chain") == chain), {})
