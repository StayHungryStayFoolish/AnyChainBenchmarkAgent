"""Single source of truth for "what group is next" in the AnyChain Agent Harness.

`agent/harness/groups.py` (live turn routing) and `agent/harness/oracle.py`
(status/explanation text) previously reimplemented this precondition chain
independently by hand. Keeping one copy here means the group the Harness
actually asks about next and the group it reports in status text can never
disagree.
"""

from __future__ import annotations

from typing import Any

try:
    from agent.planners.chain_template_requirements import inspect_chain_template
except ModuleNotFoundError:  # product script adds agent/ to sys.path
    from planners.chain_template_requirements import inspect_chain_template


def chain_auxiliary_fields_needed(chain: str) -> list[str]:
    """Return the chain-auxiliary-endpoint fields this chain's template

    actually substitutes at runtime (e.g. `${RPC_API_KEY}`). Most of the
    seven fields this group could ask about are not consumed by any current
    chain template; only ask about ones a template genuinely needs.
    """

    if not chain:
        return []
    return list(inspect_chain_template(chain).get("runtime_endpoint_variables") or [])


def next_group_and_reason(state: dict[str, Any]) -> tuple[str, str]:
    """Return (group, reason) for the next blocking configuration group."""

    target_mode = str(state.get("target_mode") or "").strip()
    if not target_mode:
        return "opening", "choose target mode"

    identity = state.get("chain_identity") or {}
    identity_status = identity.get("status")
    if identity_status == "existing_family_needs_endpoint":
        return "endpoint_process", f"continue new-chain validation: {identity_status}"
    if identity_status in {
        "existing_family_needs_method",
        "existing_family_needs_schema_evidence",
        "existing_family_schema_needs_confirmation",
        "existing_family_needs_workload_scope",
        "existing_family_needs_single_method",
        "existing_family_needs_weights",
    }:
        return "endpoint_process", f"continue new-chain validation: {identity_status}"
    if identity_status == "existing_family_runtime_choice":
        return "target_samples_fixtures", "choose new-chain runtime path"
    if identity_status != "confirmed":
        # `case2_runtime_override` (a promoted new-chain-in-existing-family
        # path) is stored in `chain_identity["case"]`, never in
        # `chain_identity["status"]` — status is always "confirmed" once a
        # chain is usable. Matching only "confirmed" here mirrors
        # `groups._chain_confirmed` exactly; do not reintroduce a
        # "case2_runtime_override" status check, it would be dead code.
        return "chain_identity", "confirm chain identity"

    confirmed = state.get("confirmed_config") or {}
    for key in ("CLOUD_REGION", "CLOUD_ZONE", "MACHINE_TYPE"):
        if not confirmed.get(key):
            return "provider_deployment", f"confirm {key}"
    for key in ("LEDGER_DEVICE", "DATA_VOL_TYPE", "DATA_VOL_SIZE", "DATA_VOL_MAX_IOPS", "DATA_VOL_MAX_THROUGHPUT"):
        if not confirmed.get(key):
            return "ledger_disk", f"confirm {key}"
    if "has_accounts_device" not in confirmed:
        return "accounts_disk", "confirm whether accounts/state disk exists"
    if confirmed.get("has_accounts_device"):
        for key in ("ACCOUNTS_DEVICE", "ACCOUNTS_VOL_TYPE", "ACCOUNTS_VOL_SIZE", "ACCOUNTS_VOL_MAX_IOPS", "ACCOUNTS_VOL_MAX_THROUGHPUT"):
            if not confirmed.get(key):
                return "accounts_disk", f"confirm {key}"
    for key in ("NETWORK_INTERFACE", "NETWORK_MAX_BANDWIDTH_GBPS"):
        if not confirmed.get(key):
            return "network", f"confirm {key}"

    endpoint_evidence = state.get("endpoint_evidence") or {}
    if target_mode == "real-node" and not endpoint_evidence.get("local_rpc_url_ready"):
        return "endpoint_process", "validate LOCAL_RPC_URL"
    if target_mode == "real-node" and not confirmed.get("BLOCKCHAIN_PROCESS_NAMES"):
        return "endpoint_process", "confirm BLOCKCHAIN_PROCESS_NAMES"
    if target_mode == "real-node" and not confirmed.get("MAINNET_RPC_URL_REVIEWED"):
        return "endpoint_process", "confirm MAINNET_RPC_URL / sync-health behavior"

    chain_name = str(identity.get("canonical") or "").strip()
    for field in chain_auxiliary_fields_needed(chain_name):
        if not confirmed.get(field):
            return "chain_auxiliary_endpoints", f"confirm {field}"

    workflow_mode = str(state.get("workflow_mode") or "").strip()
    sync = state.get("sync_observe") or {}
    if workflow_mode == "sync_observe":
        source = sync.get("source")
        if not source:
            return "sync_observe", "choose sync-observe data source"
        if source in {"existing_local_node", "endpoint_only"} and not endpoint_evidence.get("sync_rpc_url_ready"):
            return "endpoint_process", "validate real sync-observe RPC endpoint"
        if source == "existing_local_node" and not confirmed.get("BLOCKCHAIN_PROCESS_NAMES"):
            return "endpoint_process", "confirm node process for sync-observe attribution"
        if source in {"existing_local_node", "endpoint_only"} and not confirmed.get("MAINNET_RPC_URL_REVIEWED"):
            return "endpoint_process", "confirm sync-health / MAINNET_RPC_URL behavior"
        if source == "client_setup" and not sync.get("client_setup_acknowledged"):
            return "sync_observe", "acknowledge real client setup handoff"
        if not sync.get("stop_condition"):
            return "sync_observe", "choose sync-observe stop condition"
        if sync.get("stop_condition") == "duration" and not sync.get("duration_seconds"):
            return "sync_observe", "confirm sync-observe duration"
        if not (state.get("observability") or {}).get("mode"):
            return "observability", "choose observability mode"
        if not (state.get("advanced_tuning") or {}).get("confirmed"):
            return "advanced_tuning", "review advanced tuning settings"
        if not (state.get("preflight") or {}).get("approved"):
            return "preflight_smoke_execution", "approve preflight/smoke"
        return "job_monitoring", "monitor sync-observe job"

    custom_rpc = state.get("custom_rpc") or {}
    if custom_rpc.get("status") in {
        "needs_endpoint",
        "needs_method",
        "needs_schema_evidence",
        "schema_needs_confirmation",
        "needs_adapter_family_confirmation",
        "needs_scope",
        "needs_single_method",
        "needs_weights",
        "probe_failed",
    }:
        return "endpoint_process", f"continue custom RPC workflow: {custom_rpc.get('status')}"
    if not state.get("rpc_mode"):
        return "workload_rpc", "choose RPC mode"
    if not (state.get("workload") or {}).get("confirmed"):
        return "workload_rpc", "confirm RPC workload"
    qps = state.get("qps_profile") or {}
    if not qps.get("mode"):
        return "qps_profile", "choose benchmark QPS mode"
    if not qps.get("confirmed"):
        return "qps_profile", "confirm or adjust QPS profile"
    if not (state.get("observability") or {}).get("mode"):
        return "observability", "choose observability mode"
    if not (state.get("advanced_tuning") or {}).get("confirmed"):
        return "advanced_tuning", "review advanced tuning settings"
    if not (state.get("preflight") or {}).get("approved"):
        return "preflight_smoke_execution", "approve preflight/smoke"
    return "job_monitoring", "monitor benchmark job"
