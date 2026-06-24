"""Benchmark entrypoint contract for Agent workflows.

This module captures the parts of ``blockchain_node_benchmark.sh`` that the
Agent must respect before it launches or prepares a benchmark. It is deliberately
small and deterministic so the LLM can explain the workflow without inventing
its own execution rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RuntimeField:
    """One user-facing runtime value required by the benchmark entrypoint."""

    key: str
    env: str
    label: str
    reason: str
    value_kind: str = "text"
    required: bool = True
    inferred: bool = False
    optional_when: str = ""


ENTRYPOINT_PHASES: tuple[dict[str, str], ...] = (
    {
        "id": "config_loader",
        "name": "configuration loading",
        "contract": "Load config layers, detect platform/network/paths, validate chain template, and derive RPC methods.",
    },
    {
        "id": "fake_node",
        "name": "optional fake-node startup",
        "contract": "Build and start tools/fake-node, then rewrite LOCAL_RPC_URL to the local fake-node endpoint.",
    },
    {
        "id": "proxy",
        "name": "RPC proxy startup",
        "contract": "Start proxy before target generation so per-method attribution sees all benchmark traffic.",
    },
    {
        "id": "target_generation",
        "name": "account and target generation",
        "contract": "Generate account seeds and Vegeta targets from chain template methods, weights, params, and endpoint config.",
    },
    {
        "id": "monitoring",
        "name": "monitoring startup",
        "contract": "Start unified, network, sync-health, and disk-bottleneck monitors with confirmed process/disk/network config.",
    },
    {
        "id": "qps_execution",
        "name": "QPS execution",
        "contract": "Run master_qps_executor with Vegeta and generated targets.",
    },
    {
        "id": "analysis_report_archive",
        "name": "analysis, reports, archive",
        "contract": "Generate analysis, charts, bilingual reports, and archive from CSV/JSON artifacts.",
    },
)


RUNTIME_BASELINE_FIELDS: tuple[RuntimeField, ...] = (
    RuntimeField("cloud_region", "CLOUD_REGION", "cloud region", "Used in report metadata and cloud-context interpretation."),
    RuntimeField("cloud_zone", "CLOUD_ZONE", "cloud zone", "Used in report metadata and machine placement context."),
    RuntimeField("machine_type", "MACHINE_TYPE", "machine type", "Used to explain resource baselines and report context."),
    RuntimeField(
        "blockchain_process_names",
        "BLOCKCHAIN_PROCESS_NAMES",
        "blockchain process names",
        "Used by monitoring and deployment-mode detection to attribute CPU, memory, and IO to the node.",
    ),
    RuntimeField("ledger_device", "LEDGER_DEVICE", "ledger/data disk", "Required for disk charts and disk bottleneck attribution."),
    RuntimeField("data_vol_type", "DATA_VOL_TYPE", "data disk type", "Used to interpret provisioned disk capability."),
    RuntimeField("data_vol_size", "DATA_VOL_SIZE", "data disk size", "Used in report metadata and sanity checks.", "number"),
    RuntimeField("data_vol_max_iops", "DATA_VOL_MAX_IOPS", "data disk IOPS baseline", "Used for IOPS saturation analysis.", "number"),
    RuntimeField(
        "data_vol_max_throughput",
        "DATA_VOL_MAX_THROUGHPUT",
        "data disk throughput baseline",
        "Used for throughput saturation analysis.",
        "number",
    ),
    RuntimeField(
        "network_interface",
        "NETWORK_INTERFACE",
        "network interface",
        "Used by provider-aware network collectors and bandwidth charts.",
        inferred=True,
    ),
    RuntimeField(
        "network_max_bandwidth_gbps",
        "NETWORK_MAX_BANDWIDTH_GBPS",
        "network bandwidth baseline",
        "Used for network saturation analysis.",
        "number",
    ),
)


OPTIONAL_ACCOUNTS_FIELDS: tuple[RuntimeField, ...] = (
    RuntimeField("accounts_device", "ACCOUNTS_DEVICE", "accounts/state disk", "Optional second disk for account/state data.", required=False),
    RuntimeField("accounts_vol_type", "ACCOUNTS_VOL_TYPE", "accounts disk type", "Used when ACCOUNTS_DEVICE is configured.", required=False),
    RuntimeField("accounts_vol_size", "ACCOUNTS_VOL_SIZE", "accounts disk size", "Used when ACCOUNTS_DEVICE is configured.", "number", False),
    RuntimeField("accounts_vol_max_iops", "ACCOUNTS_VOL_MAX_IOPS", "accounts disk IOPS baseline", "Used when ACCOUNTS_DEVICE is configured.", "number", False),
    RuntimeField(
        "accounts_vol_max_throughput",
        "ACCOUNTS_VOL_MAX_THROUGHPUT",
        "accounts disk throughput baseline",
        "Used when ACCOUNTS_DEVICE is configured.",
        "number",
        False,
    ),
)


REAL_NODE_ENDPOINT_FIELDS: tuple[RuntimeField, ...] = (
    RuntimeField("local_rpc_url", "LOCAL_RPC_URL", "local RPC endpoint", "The real node endpoint that Vegeta/proxy will test.", "url"),
    RuntimeField(
        "mainnet_rpc_url_reviewed",
        "MAINNET_RPC_URL",
        "mainnet/reference RPC endpoint",
        "Used by sync-health checks; may use chain-template default when explicitly accepted.",
        "url_or_default",
    ),
)


COMMON_DEPENDENCIES = ("bash", "python3", "jq", "curl", "vegeta")
FAKE_NODE_DEPENDENCIES = ("go",)
MONITORING_FIDELITY_DEPENDENCIES = ("iostat", "ip", "lsblk")
PROVIDER_NIC_DEPENDENCIES = ("ethtool",)


ENTRYPOINT_SCRIPTS = (
    "blockchain_node_benchmark.sh",
    "config/config_loader.sh",
    "core/master_qps_executor.sh",
    "tools/fetch_active_accounts.py",
    "tools/target_generator.sh",
    "monitoring/monitoring_coordinator.sh",
    "visualization/report_generator.py",
)


def runtime_baseline_keys() -> tuple[str, ...]:
    return tuple(field.key for field in RUNTIME_BASELINE_FIELDS)


def required_keys_for_target(use_fake_node: bool | None) -> tuple[str, ...]:
    """Return keys the Agent must know before benchmark smoke/preflight."""
    base = ("chain", "rpc_mode", "rpc_workload_confirmed", "rpc_param_samples_confirmed", *runtime_baseline_keys())
    if use_fake_node is True:
        return (*base, "use_fake_node")
    if use_fake_node is False:
        return (*base, "use_fake_node", *(field.key for field in REAL_NODE_ENDPOINT_FIELDS))
    return (*base, "use_fake_node")


def dependency_names(use_fake_node: bool) -> tuple[str, ...]:
    deps = list(COMMON_DEPENDENCIES)
    if use_fake_node:
        deps.extend(FAKE_NODE_DEPENDENCIES)
    deps.extend(MONITORING_FIDELITY_DEPENDENCIES)
    return tuple(dict.fromkeys(deps))


def field_by_key(key: str) -> RuntimeField | None:
    for field in (*RUNTIME_BASELINE_FIELDS, *OPTIONAL_ACCOUNTS_FIELDS, *REAL_NODE_ENDPOINT_FIELDS):
        if field.key == key:
            return field
    return None


def validate_mixed_weighted(requirements: dict[str, Any]) -> tuple[bool, str]:
    weighted = requirements.get("mixed_weighted", [])
    if not weighted:
        return False, "mixed mode requires rpc_methods.mixed_weighted entries"
    total = sum(int(item.get("weight", 0) or 0) for item in weighted)
    if total != 100:
        return False, f"mixed weights must total 100, got {total}"
    return True, "mixed weights total 100"
