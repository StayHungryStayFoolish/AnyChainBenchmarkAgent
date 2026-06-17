"""Configuration checklist for Agent-created benchmark plans."""

from __future__ import annotations

from typing import Any


REAL_NODE_REQUIRED = {
    "local_rpc_url": "Local RPC URL for the blockchain node under test.",
    "blockchain_process_names": "Process names or command keywords used for node resource attribution.",
    "ledger_device": "Ledger/data disk device used for disk charts and bottleneck attribution.",
    "data_vol_max_iops": "Provisioned data disk IOPS baseline.",
    "data_vol_max_throughput": "Provisioned data disk throughput baseline in MiB/s.",
    "network_max_bandwidth_gbps": "Instance or pod network bandwidth baseline in Gbps.",
}

COMMON_REQUIRED = {
    "chain": "Chain template name, for example solana or ethereum.",
    "rpc_mode": "single or mixed RPC workload mode.",
}

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
    use_fake_node = bool(plan.get("use_fake_node"))
    request_values = _flatten_request_values(request, plan)

    benchmark_items = []
    for key, description in COMMON_REQUIRED.items():
        benchmark_items.append(_item(key, description, bool(request_values.get(key)), "blocker"))
    if not use_fake_node:
        for key, description in REAL_NODE_REQUIRED.items():
            benchmark_items.append(_item(key, description, bool(request_values.get(key)), "blocker"))

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
        for item in benchmark_items + agent_items
        if item["severity"] == "blocker" and not item["present"]
    ]
    return {
        "agent": agent_items,
        "benchmark": benchmark_items,
        "advanced": advanced_items,
        "missing_blockers": missing_blockers,
        "summary": _summary(use_fake_node, missing_blockers),
    }


def missing_required_from_checklist(checklist: dict[str, Any]) -> list[str]:
    return list(checklist.get("missing_blockers", []))


def _flatten_request_values(request: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    materialized = plan.get("materialized_config", {})
    execution_env = plan.get("execution", {}).get("environment", {})
    return {
        "chain": plan.get("chain") or request.get("chain"),
        "rpc_mode": plan.get("rpc_mode") or request.get("rpc_mode"),
        "local_rpc_url": request.get("local_rpc_url") or execution_env.get("LOCAL_RPC_URL"),
        "blockchain_process_names": (
            request.get("blockchain_process_names")
            or request.get("process_names")
            or materialized.get("BLOCKCHAIN_PROCESS_NAMES_STR")
        ),
        "ledger_device": request.get("ledger_device") or materialized.get("LEDGER_DEVICE"),
        "data_vol_max_iops": request.get("data_vol_max_iops") or materialized.get("DATA_VOL_MAX_IOPS"),
        "data_vol_max_throughput": (
            request.get("data_vol_max_throughput")
            or request.get("data_vol_max_throughput_mibs")
            or materialized.get("DATA_VOL_MAX_THROUGHPUT")
        ),
        "network_max_bandwidth_gbps": request.get("network_max_bandwidth_gbps") or materialized.get("NETWORK_MAX_BANDWIDTH_GBPS"),
    }


def _item(item_id: str, description: str, present: bool, severity: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "description": description,
        "present": bool(present),
        "severity": severity,
    }


def _summary(use_fake_node: bool, missing_blockers: list[str]) -> str:
    mode = "fake-node" if use_fake_node else "real-node"
    if not missing_blockers:
        return f"{mode} configuration has no blocking checklist gaps."
    return f"{mode} configuration is missing: {', '.join(missing_blockers)}"
