"""Configuration checklist for Agent-created benchmark plans."""

from __future__ import annotations

from typing import Any

from agent.knowledge.entry_contract import (
    OPTIONAL_ACCOUNTS_FIELDS,
    REAL_NODE_ENDPOINT_FIELDS,
    RUNTIME_BASELINE_FIELDS,
    WORKFLOW_CONFIRMATION_FIELDS,
)
from agent.harness.sync_observe_contract import SyncObserveRequest


ENDPOINT_REQUIRED = {field.key: field.description for field in REAL_NODE_ENDPOINT_FIELDS if field.key == "local_rpc_url"}

_ENVIRONMENT_REVIEW_KEYS = {"cloud_region", "cloud_zone", "machine_type"}

RUNTIME_BASELINE_REQUIRED = {
    field.key: field.description for field in RUNTIME_BASELINE_FIELDS if field.key not in _ENVIRONMENT_REVIEW_KEYS
}

COMMON_REQUIRED = {field.key: field.description for field in WORKFLOW_CONFIRMATION_FIELDS}

ENVIRONMENT_REVIEW = {
    "cloud_provider": "Detected cloud provider: gcp, aws, azure, or other.",
    "deployment_platform": "Detected runtime platform: GCE, EC2, GKE, EKS, self-hosted Kubernetes, Docker/container, or VM.",
    "cloud_region": "Cloud region for report metadata.",
    "cloud_zone": "Cloud zone when available.",
    "machine_type": "Machine or instance type for report metadata.",
}

ACCOUNTS_OPTIONAL = {field.key: field.description for field in OPTIONAL_ACCOUNTS_FIELDS}

AGENT_REQUIRED_FOR_LLM = {
    "llm_provider": "LLM provider selected in config/agent_config.sh.",
    "llm_model": "Model selected in config/agent_config.sh.",
}

ADVANCED_DEFAULTS = {
    "monitoring": "Monitoring intervals and overhead thresholds use config defaults unless tuned.",
    "sync_health": "Block-height/sync-health thresholds use config/internal_config.sh defaults.",
    "observability": "Prometheus/Grafana stays disabled unless requested.",
    "kubernetes": "Kubernetes collector setup is required only for pod-hosted nodes.",
}


def build_configuration_checklist(request: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Return user-facing checklist grouped by Agent, benchmark, and advanced layers."""
    use_fake_node = plan.get("use_fake_node")
    request_values = _flatten_request_values(request, plan)
    workload_type = _workload_type(request, plan)

    benchmark_items = []
    if workload_type == "sync_observe":
        sync_request = SyncObserveRequest.from_request_values(request_values)
        presence = sync_request.checklist_presence()
        descriptions = {
            "sync_observe_source": "Confirm an existing local real-node process or a real endpoint-only source.",
            "sync_observe_rpc_url": "Validate the real node RPC endpoint used to observe sync/import progress.",
            "node_process_identity": "Confirm the local node process for CPU/thread attribution when observing a local process.",
            "mainnet_rpc_url_reviewed": "Confirm the sync-health / MAINNET_RPC_URL comparison behavior.",
            "sync_observe_stop_condition": "Confirm whether observation runs until stopped, for a duration, or until synced.",
            "sync_observe_duration_seconds": "Provide a positive duration when the stop condition is duration.",
        }
        source_present = sync_request.source in {"existing_local_node", "endpoint_only"}
        benchmark_items.append(_item("sync_observe_source", descriptions["sync_observe_source"], source_present, "blocker"))
        for key in (
            "sync_observe_rpc_url",
            "node_process_identity",
            "mainnet_rpc_url_reviewed",
            "sync_observe_stop_condition",
            "sync_observe_duration_seconds",
        ):
            benchmark_items.append(_item(key, descriptions[key], presence[key], "blocker"))
        benchmark_items.append(_item(
            "node_prometheus_metrics_url",
            "Optional node Prometheus metrics endpoint for client-native MGas/s metrics.",
            presence["node_prometheus_metrics_url"],
            "info",
        ))
    else:
        for key, description in COMMON_REQUIRED.items():
            benchmark_items.append(_item(key, description, _is_present(key, request_values.get(key)), "blocker"))
        for key, description in RUNTIME_BASELINE_REQUIRED.items():
            benchmark_items.append(_item(key, description, bool(request_values.get(key)), "blocker"))
        if request_values.get("rpc_mode") == "mixed":
            benchmark_items.append(_item("mixed_weights_confirmed", "Confirm mixed RPC method weights total 100.", bool(request_values.get("mixed_weights_confirmed")), "blocker"))
        if use_fake_node is False:
            for key, description in ENDPOINT_REQUIRED.items():
                benchmark_items.append(_item(key, description, bool(request_values.get(key)), "blocker"))
    environment_items = [
        _item(key, description, bool(request_values.get(key)), "confirm")
        for key, description in ENVIRONMENT_REVIEW.items()
    ]

    accounts_items = [_item("has_accounts_device", "Confirm whether this node has a second accounts/state disk.", False, "confirm")]
    if request_values.get("accounts_device"):
        for key, description in ACCOUNTS_OPTIONAL.items():
            accounts_items.append(_item(key, description, bool(request_values.get(key)), "blocker" if key != "accounts_device" else "confirm"))

    chain_items = _chain_items(plan)

    agent_items = [
        _item("llm_provider", AGENT_REQUIRED_FOR_LLM["llm_provider"], True, "info"),
        _item("llm_model", AGENT_REQUIRED_FOR_LLM["llm_model"], True, "info"),
        _item("knowledge_base", "Optional enterprise Knowledge Base provider.", True, "info"),
    ]

    advanced_items = [
        _item(key, description, True, "info")
        for key, description in ADVANCED_DEFAULTS.items()
    ]
    missing_blockers = [
        item["id"]
        for item in benchmark_items + accounts_items + agent_items
        if item["severity"] == "blocker" and not item["present"]
    ]
    return {
        "agent": agent_items,
        "environment": environment_items,
        "benchmark": benchmark_items,
        "accounts_optional": accounts_items,
        "chain_template": chain_items,
        "advanced": advanced_items,
        "missing_blockers": missing_blockers,
        "summary": _summary(use_fake_node, missing_blockers, workload_type),
    }


def missing_required_from_checklist(checklist: dict[str, Any]) -> list[str]:
    return list(checklist.get("missing_blockers", []))


def _flatten_request_values(request: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    materialized = plan.get("materialized_config", {})
    execution_env = plan.get("execution", {}).get("environment", {})
    return {
        "chain": plan.get("chain") or request.get("chain"),
        "use_fake_node": plan.get("use_fake_node") if plan.get("use_fake_node") is not None else request.get("use_fake_node"),
        "rpc_mode": plan.get("rpc_mode") or request.get("rpc_mode"),
        "rpc_workload_confirmed": "rpc_workload_confirmed" in _confirmations(request, plan),
        "rpc_param_samples_confirmed": "rpc_param_samples_confirmed" in _confirmations(request, plan),
        "mixed_weights_confirmed": "mixed_weights_confirmed" in _confirmations(request, plan),
        "benchmark_mode_confirmed": "benchmark_mode_confirmed" in _confirmations(request, plan),
        "qps_profile_confirmed": "qps_profile_confirmed" in _confirmations(request, plan),
        "observability_choice_confirmed": "observability_choice_confirmed" in _confirmations(request, plan),
        "chain_template_reviewed": "chain_template_reviewed" in _confirmations(request, plan),
        "local_rpc_url": request.get("local_rpc_url") or execution_env.get("LOCAL_RPC_URL"),
        "blockchain_process_names": (
            request.get("blockchain_process_names")
            or request.get("process_names")
            or materialized.get("BLOCKCHAIN_PROCESS_NAMES_STR")
        ),
        "ledger_device": request.get("ledger_device") or materialized.get("LEDGER_DEVICE"),
        "cloud_provider": materialized.get("CLOUD_PROVIDER") or request.get("deployment", {}).get("provider"),
        "deployment_platform": request.get("deployment", {}).get("type") or plan.get("deployment", {}).get("type"),
        "cloud_region": request.get("cloud_region") or materialized.get("CLOUD_REGION"),
        "cloud_zone": request.get("cloud_zone") or materialized.get("CLOUD_ZONE"),
        "machine_type": request.get("machine_type") or materialized.get("MACHINE_TYPE"),
        "data_vol_type": request.get("data_vol_type") or materialized.get("DATA_VOL_TYPE"),
        "data_vol_size": request.get("data_vol_size") or materialized.get("DATA_VOL_SIZE"),
        "data_vol_max_iops": request.get("data_vol_max_iops") or materialized.get("DATA_VOL_MAX_IOPS"),
        "data_vol_max_throughput": (
            request.get("data_vol_max_throughput")
            or request.get("data_vol_max_throughput_mibs")
            or materialized.get("DATA_VOL_MAX_THROUGHPUT")
        ),
        "accounts_device": request.get("accounts_device") or materialized.get("ACCOUNTS_DEVICE"),
        "accounts_vol_type": request.get("accounts_vol_type") or materialized.get("ACCOUNTS_VOL_TYPE"),
        "accounts_vol_size": request.get("accounts_vol_size") or materialized.get("ACCOUNTS_VOL_SIZE"),
        "accounts_vol_max_iops": request.get("accounts_vol_max_iops") or materialized.get("ACCOUNTS_VOL_MAX_IOPS"),
        "accounts_vol_max_throughput": request.get("accounts_vol_max_throughput") or materialized.get("ACCOUNTS_VOL_MAX_THROUGHPUT"),
        "network_interface": request.get("network_interface") or materialized.get("NETWORK_INTERFACE"),
        "network_max_bandwidth_gbps": request.get("network_max_bandwidth_gbps") or materialized.get("NETWORK_MAX_BANDWIDTH_GBPS"),
        "sync_observe_stop_condition": request.get("sync_observe_stop_condition") or materialized.get("SYNC_OBSERVE_STOP_CONDITION"),
        "sync_observe_duration_seconds": request.get("sync_observe_duration_seconds") or materialized.get("SYNC_OBSERVE_DURATION"),
        "sync_observe_source": request.get("sync_observe_source", ""),
        "sync_observe_rpc_url": request.get("sync_observe_rpc_url") or materialized.get("SYNC_OBSERVE_RPC_URL"),
        "node_prometheus_metrics_url": request.get("node_prometheus_metrics_url") or materialized.get("NODE_PROMETHEUS_METRICS_URL"),
        "node_process_identity": (
            request.get("node_process_pid")
            or request.get("blockchain_process_names")
            or request.get("process_names")
            or materialized.get("NODE_PROCESS_PID")
            or materialized.get("BLOCKCHAIN_PROCESS_NAMES_STR")
        ),
        "mainnet_rpc_url_reviewed": request.get("mainnet_rpc_url_reviewed") or materialized.get("MAINNET_RPC_URL"),
    }


def _chain_items(plan: dict[str, Any]) -> list[dict[str, Any]]:
    requirements = plan.get("chain_template_requirements", {})
    if not requirements:
        return []
    weighted = requirements.get("mixed_weighted", [])
    weight_total = sum(int(item.get("weight", 0) or 0) for item in weighted)
    return [
        _item("chain_template_exists", "Selected config/chains/<chain>.json exists.", bool(requirements.get("exists")), "blocker"),
        _item("chain_template_reviewed", "Review endpoint variables, TARGET_* sample variables, and method defaults from the selected chain template.", True, "confirm"),
        _item("rpc_single_method_confirmation", f"Confirm single RPC method: {requirements.get('single_method') or '<missing>'}.", bool(requirements.get("single_method")), "confirm"),
        _item("rpc_mixed_weighted_confirmation", f"Confirm mixed RPC methods and weights; current total is {weight_total}.", bool(weighted), "confirm"),
        _item("custom_rpc_method_review", "Ask whether the user wants to add custom RPC methods before execution.", True, "confirm"),
        _item("rpc_param_samples_confirmation", "Confirm chain template TARGET_* sample values required by selected methods.", True, "confirm"),
        _item("advanced_config_review", "Ask whether the user wants to review internal_config.sh advanced thresholds.", True, "info"),
    ]


def _item(item_id: str, description: str, present: bool, severity: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "description": description,
        "present": bool(present),
        "severity": severity,
    }


def _is_present(key: str, value: Any) -> bool:
    if key == "use_fake_node":
        return isinstance(value, bool)
    return bool(value)


def _confirmations(request: dict[str, Any], plan: dict[str, Any]) -> set[str]:
    return set(request.get("confirmations", []) or plan.get("confirmed_inputs", []) or [])


def _workload_type(request: dict[str, Any], plan: dict[str, Any]) -> str:
    value = plan.get("workflow_type") or plan.get("run_mode") or request.get("workflow_type") or request.get("run_mode")
    text = str(value or "").strip().lower().replace("-", "_")
    return "sync_observe" if text in {"sync_observe", "sync", "observe_sync"} else "rpc_benchmark"


def _summary(use_fake_node: bool | None, missing_blockers: list[str], workload_type: str = "rpc_benchmark") -> str:
    if workload_type == "sync_observe":
        if not missing_blockers:
            return "sync-observe configuration has no blocking checklist gaps."
        return f"sync-observe configuration is missing: {', '.join(missing_blockers)}"
    if use_fake_node is True:
        mode = "fake-node"
    elif use_fake_node is False:
        mode = "real-node"
    else:
        mode = "unconfirmed target-mode"
    if not missing_blockers:
        return f"{mode} configuration has no blocking checklist gaps."
    return f"{mode} configuration is missing: {', '.join(missing_blockers)}"
