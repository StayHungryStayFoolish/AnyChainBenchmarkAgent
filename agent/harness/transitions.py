"""Authoritative workflow invalidation and transition helpers."""

from __future__ import annotations

from .state import AgentGraphState

from agent.workflows.group_registry import invalidation_targets


_WORKLOAD_BOUND_CUSTOM_RPC_KEYS = {
    "job_local_override",
    "methods",
    "requested_workload",
    "scope",
    "weights",
    "workload_scope",
}

_WORKLOAD_BOUND_CUSTOM_RPC_STATUSES = {
    "needs_scope",
    "needs_single_method",
    "needs_weights",
    "validated",
}

_VALIDATION_ENDPOINT_EVIDENCE_KEYS = {
    "evidence_file",
    "observed_response",
    "validation_endpoint",
}

_FINAL_ENDPOINT_EVIDENCE_KEYS = {
    "final_endpoint",
    "final_endpoint_evidence_file",
}


def invalidate_for_chain_change(state: AgentGraphState, *, new_chain: str = "") -> None:
    """Remove every value whose truth depends on the selected chain."""

    custom_rpc = _catalog_for_chain(state, new_chain=new_chain)
    state["rpc_mode"] = ""
    for key in ("workload", "endpoint_evidence", "fixture_evidence", "preflight", "smoke", "final_benchmark", "job"):
        state[key] = {}
    state["custom_rpc"] = custom_rpc
    _clear_effective_custom_rpc_workload(state)
    confirmed = state.setdefault("confirmed_config", {})
    for field in (
        "LOCAL_RPC_URL",
        "MAINNET_RPC_URL",
        "MAINNET_RPC_URL_REVIEWED",
        "BLOCKCHAIN_PROCESS_NAMES",
        "CHAIN_REST_URL",
        "CHAIN_INDEXER_URL",
        "CHAIN_SIDECAR_URL",
        "CHAIN_EVM_RPC_URL",
        "CHAIN_JSON_RPC_URL",
        "CHAIN_MIRROR_URL",
        "RPC_API_KEY",
    ):
        confirmed.pop(field, None)
    _record_invalidations(state, *invalidation_targets("chain_identity"))


def invalidate_for_target_mode(state: AgentGraphState, previous_mode: str = "") -> None:
    """Invalidate mode-specific state after an explicitly confirmed switch.

    Environment/hardware values remain valid. RPC benchmark and sync-observe
    state never silently revive across the mode boundary.
    """

    target_mode = str(state.get("target_mode") or "")
    workflow_mode = str(state.get("workflow_mode") or "")
    confirmed = state.setdefault("confirmed_config", {})
    state["preflight"] = {}
    state["smoke"] = {}
    state["final_benchmark"] = {}
    state["job"] = {}
    if workflow_mode == "sync_observe":
        state["rpc_mode"] = ""
        state["workload"] = {}
        state["custom_rpc"] = {}
        state["fixture_evidence"] = {}
        state["qps_profile"] = {}
        # A benchmark endpoint is not automatically a valid sync metrics
        # source. Sync-observe validates its own RPC/metrics evidence.
        evidence = state.setdefault("endpoint_evidence", {})
        for key in ("sync_rpc_url_ready", "sync_rpc_url_probe"):
            evidence.pop(key, None)
        confirmed.pop("MAINNET_RPC_URL_REVIEWED", None)
    else:
        state["sync_observe"] = {}
        evidence = state.setdefault("endpoint_evidence", {})
        for key in ("sync_rpc_url_ready", "sync_rpc_url_probe"):
            evidence.pop(key, None)
        if previous_mode and previous_mode != target_mode:
            # A real-node endpoint/process cannot be reused by fake-node, and a
            # fake-node session contains no validated real endpoint to promote.
            for key in ("local_rpc_url_ready", "local_rpc_url_probe"):
                evidence.pop(key, None)
            for field in ("LOCAL_RPC_URL", "MAINNET_RPC_URL", "MAINNET_RPC_URL_REVIEWED", "BLOCKCHAIN_PROCESS_NAMES"):
                confirmed.pop(field, None)
    _record_invalidations(state, *invalidation_targets("target_mode"))


def invalidate_for_rpc_mode_change(state: AgentGraphState) -> None:
    """Clear the selected workload while preserving the verified RPC catalog.

    A method contract is reusable evidence. Selecting ``single`` or ``mixed``
    only changes how catalog methods participate in this run; it must not erase
    their exact params, semantic metadata, response shape, or probe provenance.
    """

    state["workload"] = {}
    _clear_effective_custom_rpc_workload(state)
    state["fixture_evidence"] = {}
    state["preflight"] = {}
    state["smoke"] = {}
    state["final_benchmark"] = {}
    state["job"] = {}
    _record_invalidations(state, *invalidation_targets("workload_rpc"))


def invalidate_for_endpoint_change(
    state: AgentGraphState,
    endpoint: str,
    *,
    role: str,
) -> None:
    """Invalidate only custom-RPC evidence bound to a different endpoint.

    ``role`` is either ``validation`` (the endpoint used while proving a method
    contract) or ``final`` (the endpoint selected for the benchmark). Schema and
    parameter metadata remain catalog facts; only endpoint-derived probe facts
    are removed when their provenance no longer matches.
    """

    if role not in {"validation", "final"}:
        raise ValueError("endpoint role must be 'validation' or 'final'")
    normalized_endpoint = str(endpoint or "").strip()
    custom_rpc = state.get("custom_rpc")
    if not isinstance(custom_rpc, dict):
        return
    catalog = custom_rpc.get("catalog")
    if not isinstance(catalog, dict):
        from .domains.rpc_catalog import ensure_catalog

        catalog = ensure_catalog(state)
    invalidate_rpc_catalog_endpoint_evidence(
        catalog,
        normalized_endpoint,
        role=role,
    )
    if role == "validation":
        current = str(custom_rpc.get("endpoint") or "").strip()
        if current != normalized_endpoint:
            custom_rpc["endpoint"] = normalized_endpoint
            custom_rpc.pop("endpoint_ready", None)


def invalidate_rpc_catalog_endpoint_evidence(
    catalog: dict,
    endpoint: str,
    *,
    role: str,
) -> None:
    """Keep method/schema facts while expiring endpoint-derived proof."""

    if role not in {"validation", "final"}:
        raise ValueError("endpoint role must be 'validation' or 'final'")
    normalized_endpoint = str(endpoint or "").strip()
    contracts = catalog.get("methods")
    if not isinstance(contracts, list):
        return
    endpoint_key = "validation_endpoint" if role == "validation" else "final_endpoint"
    evidence_keys = (
        _VALIDATION_ENDPOINT_EVIDENCE_KEYS
        if role == "validation"
        else _FINAL_ENDPOINT_EVIDENCE_KEYS
    )
    for contract in contracts:
        if not isinstance(contract, dict):
            continue
        provenance = str(contract.get(endpoint_key) or "").strip()
        if provenance and provenance == normalized_endpoint:
            continue
        for key in evidence_keys:
            contract.pop(key, None)


def clear_effective_custom_rpc_workload(state: AgentGraphState) -> None:
    """Public transition used when selection changes but catalog facts remain."""

    _clear_effective_custom_rpc_workload(state)


def _clear_effective_custom_rpc_workload(state: AgentGraphState) -> None:
    custom_rpc = state.get("custom_rpc")
    if not isinstance(custom_rpc, dict):
        state["custom_rpc"] = {}
        return
    for key in _WORKLOAD_BOUND_CUSTOM_RPC_KEYS:
        custom_rpc.pop(key, None)
    if custom_rpc.get("status") in _WORKLOAD_BOUND_CUSTOM_RPC_STATUSES:
        catalog = custom_rpc.get("catalog") if isinstance(custom_rpc.get("catalog"), dict) else {}
        if catalog.get("methods"):
            custom_rpc["status"] = "method_validated_next"
        else:
            custom_rpc.pop("status", None)


def _catalog_for_chain(state: AgentGraphState, *, new_chain: str) -> dict:
    """Return catalog entries whose explicit chain provenance still matches.

    Records created before chain provenance was stored are conservatively bound
    to the active chain. They cannot be promoted to another chain merely because
    a method name happens to be the same.
    """

    custom_rpc = state.get("custom_rpc")
    if not isinstance(custom_rpc, dict):
        return {}
    destination = str(new_chain or _pending_chain_destination(state)).strip().casefold()
    current = str((state.get("chain_identity") or {}).get("canonical") or "").strip().casefold()
    if not destination:
        return {}
    catalog = custom_rpc.get("catalog") if isinstance(custom_rpc.get("catalog"), dict) else {}
    catalog_chain = str(catalog.get("chain") or current).strip().casefold()
    retained = []
    for contract in catalog.get("methods") or []:
        if not isinstance(contract, dict):
            continue
        provenance = str(
            contract.get("chain")
            or contract.get("canonical_chain")
            or catalog_chain
        ).strip().casefold()
        if provenance == destination:
            retained.append(contract)
    if not retained:
        return {}
    # Top-level draft/endpoint fields describe the active catalog chain. When
    # only individually-provenanced records survive, carrying those fields into
    # another chain would silently attach stale endpoint or unfinished schema
    # evidence to the retained contracts.
    result = dict(custom_rpc) if catalog_chain == destination else {}
    retained_catalog = dict(catalog) if catalog_chain == destination else {
        "contract_version": int(catalog.get("contract_version") or 1),
        "revision": int(catalog.get("revision") or 0) + 1,
        "finished": bool(catalog.get("finished")),
    }
    retained_catalog["chain"] = destination
    retained_catalog["methods"] = retained
    retained_catalog["draft"] = {}
    result["catalog"] = retained_catalog
    return result


def _pending_chain_destination(state: AgentGraphState) -> str:
    """Read the exact destination carried by the confirmed change contract."""

    identity = state.get("chain_identity") or {}
    candidate = identity.get("change_candidate")
    if not isinstance(candidate, dict):
        return ""
    resolution = candidate.get("resolution")
    if not isinstance(resolution, dict):
        resolution = {}
    return str(
        resolution.get("possible_known_chain")
        or resolution.get("canonical_chain_name")
        or candidate.get("canonical")
        or candidate.get("raw")
        or ""
    ).strip()


def mark_group_reconfigured(state: AgentGraphState, group: str) -> None:
    invalidated = set(state.get("invalidated_groups") or [])
    invalidated.discard(group)
    state["invalidated_groups"] = sorted(invalidated)
    state.setdefault("group_states", {}).setdefault(group, {})["status"] = "completed"


def mark_group_reconfiguring(state: AgentGraphState, group: str) -> None:
    """Mark an active group as the owner of an unfinished repair flow."""

    if group:
        state.setdefault("group_states", {}).setdefault(group, {})["status"] = "reconfiguring"


def _record_invalidations(state: AgentGraphState, *groups: str) -> None:
    invalidated = set(state.get("invalidated_groups") or [])
    invalidated.update(group for group in groups if group)
    state["invalidated_groups"] = sorted(invalidated)
    group_states = state.setdefault("group_states", {})
    for group in groups:
        if group:
            group_states.setdefault(group, {})["status"] = "invalidated"
