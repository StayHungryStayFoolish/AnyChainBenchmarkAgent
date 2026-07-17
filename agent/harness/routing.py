"""Registry-driven fallback routing for the AnyChain Agent Harness.

The group registry owns product order and path applicability. Readiness
predicates report domain facts only; they do not maintain a second workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.planners.chain_template_requirements import inspect_chain_template
from agent.workflows.group_registry import fallback_groups_for_workflow

from .failures import unresolved_recovery
from .sync_observe_contract import SyncObserveBlocker, SyncObserveRequest


@dataclass(frozen=True)
class GroupReadiness:
    ready: bool
    reason: str = ""
    continuation: bool = False


ReadinessPredicate = Callable[[dict[str, Any]], GroupReadiness]


def chain_auxiliary_fields_needed(chain: str) -> list[str]:
    """Return only auxiliary endpoint fields consumed by the chain template."""

    if not chain:
        return []
    return list(inspect_chain_template(chain).get("runtime_endpoint_variables") or [])


def sync_observe_readiness(state: dict[str, Any]) -> SyncObserveBlocker | None:
    """Return the single blocking request fact for the sync-observe path."""

    if str(state.get("workflow_mode") or "") != "sync_observe":
        return None
    return SyncObserveRequest.from_state(state).blocker()


def _ready() -> GroupReadiness:
    return GroupReadiness(True)


def _missing(reason: str, *, continuation: bool = False) -> GroupReadiness:
    return GroupReadiness(False, reason, continuation)


def _opening_readiness(state: dict[str, Any]) -> GroupReadiness:
    return _ready() if str(state.get("target_mode") or "").strip() else _missing("choose target mode")


def chain_identity_confirmed(state: dict[str, Any]) -> bool:
    """Return whether workload consumers may trust the selected chain."""

    identity = state.get("chain_identity") or {}
    return (
        str(identity.get("status") or "") == "confirmed"
        and bool(str(identity.get("canonical") or "").strip())
    )


def _chain_identity_readiness(state: dict[str, Any]) -> GroupReadiness:
    status = str((state.get("chain_identity") or {}).get("status") or "")
    delegated = {
        "existing_family_needs_endpoint",
        "existing_family_needs_method",
        "existing_family_needs_schema_evidence",
        "existing_family_schema_needs_confirmation",
        "existing_family_needs_workload_scope",
        "existing_family_needs_single_method",
        "existing_family_needs_weights",
        "existing_family_runtime_choice",
    }
    if status in delegated:
        return _ready()
    return _ready() if status == "confirmed" else _missing("confirm chain identity")


def _provider_readiness(state: dict[str, Any]) -> GroupReadiness:
    confirmed = state.get("confirmed_config") or {}
    for field in ("CLOUD_REGION", "CLOUD_ZONE", "MACHINE_TYPE"):
        if not confirmed.get(field):
            return _missing(f"confirm {field}")
    return _ready()


def _ledger_readiness(state: dict[str, Any]) -> GroupReadiness:
    confirmed = state.get("confirmed_config") or {}
    for field in (
        "LEDGER_DEVICE",
        "DATA_VOL_TYPE",
        "DATA_VOL_SIZE",
        "DATA_VOL_MAX_IOPS",
        "DATA_VOL_MAX_THROUGHPUT",
    ):
        if not confirmed.get(field):
            return _missing(f"confirm {field}")
    return _ready()


def _accounts_readiness(state: dict[str, Any]) -> GroupReadiness:
    confirmed = state.get("confirmed_config") or {}
    if "has_accounts_device" not in confirmed:
        return _missing("confirm whether accounts/state disk exists")
    if confirmed.get("has_accounts_device"):
        for field in (
            "ACCOUNTS_DEVICE",
            "ACCOUNTS_VOL_TYPE",
            "ACCOUNTS_VOL_SIZE",
            "ACCOUNTS_VOL_MAX_IOPS",
            "ACCOUNTS_VOL_MAX_THROUGHPUT",
        ):
            if not confirmed.get(field):
                return _missing(f"confirm {field}")
    return _ready()


def _network_readiness(state: dict[str, Any]) -> GroupReadiness:
    confirmed = state.get("confirmed_config") or {}
    for field in ("NETWORK_INTERFACE", "NETWORK_MAX_BANDWIDTH_GBPS"):
        if not confirmed.get(field):
            return _missing(f"confirm {field}")
    return _ready()


def _endpoint_readiness(state: dict[str, Any]) -> GroupReadiness:
    identity_status = str((state.get("chain_identity") or {}).get("status") or "")
    continuation_statuses = {
        "existing_family_needs_endpoint",
        "existing_family_needs_method",
        "existing_family_needs_schema_evidence",
        "existing_family_schema_needs_confirmation",
        "existing_family_needs_workload_scope",
        "existing_family_needs_single_method",
        "existing_family_needs_weights",
    }
    if identity_status in continuation_statuses:
        return _missing(
            f"continue new-chain validation: {identity_status}", continuation=True
        )

    target_mode = str(state.get("target_mode") or "")
    endpoint_evidence = state.get("endpoint_evidence") or {}
    confirmed = state.get("confirmed_config") or {}
    if target_mode == "real-node" and not endpoint_evidence.get("local_rpc_url_ready"):
        return _missing("validate LOCAL_RPC_URL")
    if target_mode == "real-node" and not confirmed.get("BLOCKCHAIN_PROCESS_NAMES"):
        return _missing("confirm BLOCKCHAIN_PROCESS_NAMES")
    if target_mode == "real-node" and not confirmed.get("MAINNET_RPC_URL_REVIEWED"):
        return _missing("confirm MAINNET_RPC_URL / sync-health behavior")

    sync_blocker = sync_observe_readiness(state)
    if sync_blocker and sync_blocker.group == "endpoint_process":
        return _missing(sync_blocker.reason)

    custom_status = str((state.get("custom_rpc") or {}).get("status") or "")
    if str(state.get("workflow_mode") or "") == "rpc_benchmark" and custom_status in {
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
        return _missing(f"continue custom RPC workflow: {custom_status}")
    return _ready()


def _chain_auxiliary_readiness(state: dict[str, Any]) -> GroupReadiness:
    identity = state.get("chain_identity") or {}
    confirmed = state.get("confirmed_config") or {}
    chain = str(identity.get("canonical") or "").strip()
    for field in chain_auxiliary_fields_needed(chain):
        if not confirmed.get(field):
            return _missing(f"confirm {field}")
    return _ready()


def _workload_readiness(state: dict[str, Any]) -> GroupReadiness:
    if not chain_identity_confirmed(state):
        return _missing("confirm chain identity before configuring RPC workload")
    if not state.get("rpc_mode"):
        return _missing("choose RPC mode")
    if not (state.get("workload") or {}).get("confirmed"):
        return _missing("confirm RPC workload")
    return _ready()


def _fixture_readiness(state: dict[str, Any]) -> GroupReadiness:
    identity_status = str((state.get("chain_identity") or {}).get("status") or "")
    if identity_status == "existing_family_runtime_choice" and state.get("target_mode") == "fake-node":
        return _missing("choose new-chain runtime path", continuation=True)
    fixture = state.get("fixture_evidence") or {}
    if (
        state.get("target_mode") == "fake-node"
        and fixture.get("required")
        and fixture.get("status") != "validated"
    ):
        return _missing("resolve missing fixtures for the effective custom workload")
    return _ready()


def _qps_readiness(state: dict[str, Any]) -> GroupReadiness:
    qps = state.get("qps_profile") or {}
    if not qps.get("mode"):
        return _missing("choose benchmark QPS mode")
    if not qps.get("confirmed"):
        return _missing("confirm or adjust QPS profile")
    return _ready()


def _sync_readiness(state: dict[str, Any]) -> GroupReadiness:
    blocker = sync_observe_readiness(state)
    if blocker and blocker.group == "sync_observe":
        return _missing(blocker.reason)
    return _ready()


def _observability_readiness(state: dict[str, Any]) -> GroupReadiness:
    return _ready() if (state.get("observability") or {}).get("mode") else _missing("choose observability mode")


def _advanced_readiness(state: dict[str, Any]) -> GroupReadiness:
    return _ready() if (state.get("advanced_tuning") or {}).get("confirmed") else _missing("review advanced tuning settings")


def _preflight_readiness(state: dict[str, Any]) -> GroupReadiness:
    return _ready() if (state.get("preflight") or {}).get("approved") else _missing("approve preflight/smoke")


GROUP_READINESS: dict[str, ReadinessPredicate] = {
    "opening": _opening_readiness,
    "target_mode": lambda _state: _ready(),
    "chain_identity": _chain_identity_readiness,
    "provider_deployment": _provider_readiness,
    "ledger_disk": _ledger_readiness,
    "accounts_disk": _accounts_readiness,
    "network": _network_readiness,
    "endpoint_process": _endpoint_readiness,
    "chain_auxiliary_endpoints": _chain_auxiliary_readiness,
    "workload_rpc": _workload_readiness,
    "target_samples_fixtures": _fixture_readiness,
    "qps_profile": _qps_readiness,
    "sync_observe": _sync_readiness,
    "observability": _observability_readiness,
    "advanced_tuning": _advanced_readiness,
    "preflight_smoke_execution": _preflight_readiness,
}


def group_readiness(state: dict[str, Any], group: str) -> GroupReadiness:
    """Return one group's readiness fact without selecting another group."""

    predicate = GROUP_READINESS.get(str(group or ""))
    return predicate(state) if predicate else _ready()


def next_group_and_reason(state: dict[str, Any]) -> tuple[str, str]:
    """Return the first registry-ordered blocking prerequisite and its fact."""

    if unresolved_recovery(state.get("failure_recovery")):
        return "failure_recovery", "resolve the current execution failure"

    specs = fallback_groups_for_workflow(str(state.get("workflow_mode") or ""))
    facts = [(spec.name, group_readiness(state, spec.name)) for spec in specs]
    for group, fact in facts:
        if not fact.ready and fact.continuation:
            return group, fact.reason
    for group, fact in facts:
        if not fact.ready:
            return group, fact.reason
    terminal_reason = (
        "monitor sync-observe job"
        if str(state.get("workflow_mode") or "") == "sync_observe"
        else "monitor benchmark job"
    )
    return "job_monitoring", terminal_reason
