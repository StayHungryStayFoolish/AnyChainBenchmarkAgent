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
    description: str = ""
    applies_to: tuple[str, ...] = ("fake_node", "real_node", "sync_observe")


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
    RuntimeField(
        "cloud_region", "CLOUD_REGION", "cloud region", "Used in report metadata and cloud-context interpretation.",
        description="Cloud region for report metadata.",
        applies_to=("fake_node", "real_node"),
    ),
    RuntimeField(
        "cloud_zone", "CLOUD_ZONE", "cloud zone", "Used in report metadata and machine placement context.",
        description="Cloud zone when available.",
        applies_to=("fake_node", "real_node"),
    ),
    RuntimeField(
        "machine_type", "MACHINE_TYPE", "machine type", "Used to explain resource baselines and report context.",
        description="Machine or instance type for report metadata.",
        applies_to=("fake_node", "real_node"),
    ),
    RuntimeField(
        "blockchain_process_names",
        "BLOCKCHAIN_PROCESS_NAMES",
        "blockchain process names",
        "Used by monitoring and deployment-mode detection to attribute CPU, memory, and IO to the node.",
        description="Process names or command-line fragments used for node resource attribution.",
        applies_to=("fake_node", "real_node"),
    ),
    RuntimeField(
        "ledger_device", "LEDGER_DEVICE", "ledger/data disk", "Required for disk charts and disk bottleneck attribution.",
        description="Ledger/data disk device used for disk charts and bottleneck attribution.",
    ),
    RuntimeField(
        "data_vol_type", "DATA_VOL_TYPE", "data disk type", "Used to interpret provisioned disk capability.",
        description="Ledger/data disk type used for report metadata and baseline interpretation.",
    ),
    RuntimeField(
        "data_vol_size", "DATA_VOL_SIZE", "data disk size", "Used in report metadata and sanity checks.", "number",
        description="Ledger/data disk size in GiB.",
    ),
    RuntimeField(
        "data_vol_max_iops", "DATA_VOL_MAX_IOPS", "data disk IOPS baseline", "Used for IOPS saturation analysis.", "number",
        description="Provisioned data disk IOPS baseline.",
    ),
    RuntimeField(
        "data_vol_max_throughput",
        "DATA_VOL_MAX_THROUGHPUT",
        "data disk throughput baseline",
        "Used for throughput saturation analysis.",
        "number",
        description="Provisioned data disk throughput baseline in MiB/s.",
    ),
    RuntimeField(
        "network_interface",
        "NETWORK_INTERFACE",
        "network interface",
        "Used by provider-aware network collectors and bandwidth charts.",
        inferred=True,
        description="Network interface used by the node.",
    ),
    RuntimeField(
        "network_max_bandwidth_gbps",
        "NETWORK_MAX_BANDWIDTH_GBPS",
        "network bandwidth baseline",
        "Used for network saturation analysis.",
        "number",
        description="Instance or pod network bandwidth baseline in Gbps.",
    ),
)


OPTIONAL_ACCOUNTS_FIELDS: tuple[RuntimeField, ...] = (
    RuntimeField(
        "accounts_device", "ACCOUNTS_DEVICE", "accounts/state disk", "Optional second disk for account/state data.", required=False,
        description="Optional second disk for account/state data.",
        applies_to=("fake_node", "real_node"),
    ),
    RuntimeField(
        "accounts_vol_type", "ACCOUNTS_VOL_TYPE", "accounts disk type", "Used when ACCOUNTS_DEVICE is configured.", required=False,
        description="Accounts/state disk type.",
        applies_to=("fake_node", "real_node"),
    ),
    RuntimeField(
        "accounts_vol_size", "ACCOUNTS_VOL_SIZE", "accounts disk size", "Used when ACCOUNTS_DEVICE is configured.", "number", False,
        description="Accounts/state disk size in GiB.",
        applies_to=("fake_node", "real_node"),
    ),
    RuntimeField(
        "accounts_vol_max_iops", "ACCOUNTS_VOL_MAX_IOPS", "accounts disk IOPS baseline", "Used when ACCOUNTS_DEVICE is configured.", "number", False,
        description="Provisioned accounts/state disk IOPS baseline.",
        applies_to=("fake_node", "real_node"),
    ),
    RuntimeField(
        "accounts_vol_max_throughput",
        "ACCOUNTS_VOL_MAX_THROUGHPUT",
        "accounts disk throughput baseline",
        "Used when ACCOUNTS_DEVICE is configured.",
        "number",
        False,
        description="Provisioned accounts/state disk throughput in MiB/s.",
        applies_to=("fake_node", "real_node"),
    ),
)


REAL_NODE_ENDPOINT_FIELDS: tuple[RuntimeField, ...] = (
    RuntimeField(
        "local_rpc_url", "LOCAL_RPC_URL", "local RPC endpoint", "The real node endpoint that Vegeta/proxy will test.", "url",
        description="Local RPC URL for the blockchain node under test.",
        applies_to=("real_node",),
    ),
    RuntimeField(
        "mainnet_rpc_url_reviewed",
        "MAINNET_RPC_URL",
        "mainnet/reference RPC endpoint",
        "Used by sync-health checks; may use chain-template default when explicitly accepted.",
        "url_or_default",
        description="Confirm MAINNET_RPC_URL or selected chain-template sync-health behavior for target-height comparison.",
        applies_to=("real_node", "sync_observe"),
    ),
)


WORKFLOW_CONFIRMATION_FIELDS: tuple[RuntimeField, ...] = (
    RuntimeField("chain", "BLOCKCHAIN_NODE", "chain", "Chain template selection drives adapter family, RPC methods, and defaults.", description="Chain template name, for example solana or ethereum."),
    RuntimeField("use_fake_node", "", "target mode", "Determines whether the run uses a closed-loop fake node or a real node.", description="Choose fake-node closed-loop testing or real-node testing.", applies_to=("fake_node", "real_node")),
    RuntimeField("rpc_mode", "RPC_MODE", "RPC mode", "Determines whether the workload uses one RPC method or a weighted mix.", description="single or mixed RPC workload mode.", applies_to=("fake_node", "real_node")),
    RuntimeField("benchmark_mode_confirmed", "", "benchmark mode", "Determines QPS profile aggressiveness and expected run length.", description="Confirm benchmark mode: quick, standard, or intensive.", applies_to=("fake_node", "real_node")),
    RuntimeField("qps_profile_confirmed", "", "QPS profile", "Confirms the initial/max QPS, step, and duration for the selected mode.", description="Confirm QPS defaults for the selected mode, including initial QPS, max QPS, step, and duration.", applies_to=("fake_node", "real_node")),
    RuntimeField("observability_choice_confirmed", "", "observability mode", "Determines whether Prometheus/Grafana starts and how.", description="Confirm whether to disable observability, start local Prometheus/Grafana, or expose only the exporter for an existing stack.", applies_to=("fake_node", "real_node")),
    RuntimeField("chain_template_reviewed", "", "chain template review", "Confirms the user has seen the chain template's runtime variables and defaults.", description="Review the selected chain template runtime endpoint variables, sample variables, and default RPC workload.", applies_to=("fake_node", "real_node")),
    RuntimeField("rpc_workload_confirmed", "", "RPC workload", "Confirms the selected single/mixed RPC methods and weights.", description="Confirm the selected single/mixed RPC methods and weights.", applies_to=("fake_node", "real_node")),
    RuntimeField("rpc_param_samples_confirmed", "", "RPC parameter samples", "Confirms TARGET_* sample values for the selected RPC methods.", description="Confirm TARGET_* parameter samples for the selected RPC methods.", applies_to=("fake_node", "real_node")),
)


SYNC_OBSERVE_FIELDS: tuple[RuntimeField, ...] = (
    RuntimeField(
        "sync_observe_rpc_url", "SYNC_OBSERVE_RPC_URL", "sync-observe RPC endpoint",
        "The real node RPC/metrics endpoint sync-observe watches for sync/import progress and node metrics.",
        "url",
        # Source-specific validation owns this field because endpoint-only and
        # local-process observation collect their evidence differently.
        required=False,
        description="Real node RPC endpoint for sync-observe to poll sync/import progress and node metrics from.",
        applies_to=("sync_observe",),
    ),
    RuntimeField(
        "sync_observe_stop_condition", "", "sync-observe stop condition",
        "Determines when a sync-observe run ends: manual stop, fixed duration, or once synced.",
        description="Confirm sync-observe stop condition: run until stopped, fixed duration, or until synced.",
        applies_to=("sync_observe",),
    ),
    RuntimeField(
        "node_process_identity", "", "node process identity",
        "Used for CPU/thread attribution during sync-observe.",
        description="Node process PID or command-line fragment for node CPU/thread attribution.",
        applies_to=("sync_observe",),
    ),
    RuntimeField(
        "node_prometheus_metrics_url", "NODE_PROMETHEUS_METRICS_URL", "node Prometheus metrics endpoint",
        "Optional source for MGas/s and client-native execution metrics during sync-observe.",
        "url",
        required=False,
        description="Optional node Prometheus metrics endpoint for MGas/s and client-native execution metrics.",
        applies_to=("sync_observe",),
    ),
)


# The numeric QPS-profile overrides and advanced-tuning parameters. These are
# optional sub-values adjusted within an already-confirmed
# `qps_profile_confirmed`/`advanced_tuning`-style group (via the
# adjust-field/adjust-value flow), not individual top-level checklist
# blockers, so they are deliberately excluded from `field_specs_for`'s
# concatenation (which drives the missing-config-question checklist) and only
# folded into `ALL_RUNTIME_FIELDS` below, for field-explanation lookups.
QPS_AND_TUNING_FIELDS: tuple[RuntimeField, ...] = (
    RuntimeField("initial_qps", "INITIAL_QPS", "initial QPS", "Starting request rate for the first QPS level.", "int", required=False, description="Starting request rate for the first QPS level.", applies_to=("fake_node", "real_node")),
    RuntimeField("max_qps", "MAX_QPS", "max QPS", "Highest request rate the mode will attempt before stopping or hitting a bottleneck.", "int", required=False, description="Highest request rate the mode will attempt before stopping or hitting a bottleneck.", applies_to=("fake_node", "real_node")),
    RuntimeField("qps_step", "QPS_STEP", "QPS step", "Increment added between QPS levels.", "int", required=False, description="Increment added between QPS levels.", applies_to=("fake_node", "real_node")),
    RuntimeField("qps_duration", "DURATION", "QPS duration", "How long each QPS level runs before moving to the next level.", "int", required=False, description="How long each QPS level runs before moving to the next level, in seconds.", applies_to=("fake_node", "real_node")),
    RuntimeField("monitor_interval", "MONITOR_INTERVAL", "monitor interval", "Unified monitoring sample interval, in seconds.", "int", required=False, description="Unified monitoring sample interval, in seconds.", applies_to=("fake_node", "real_node")),
    RuntimeField("disk_monitor_rate", "DISK_MONITOR_RATE", "disk monitor rate", "Disk-specific monitor sampling rate.", "int", required=False, description="Disk-specific monitor sampling rate.", applies_to=("fake_node", "real_node")),
    RuntimeField("success_rate_threshold", "SUCCESS_RATE_THRESHOLD", "success-rate threshold", "QPS success-rate bottleneck threshold, as a percentage.", "int", required=False, description="QPS success-rate bottleneck threshold, as a percentage.", applies_to=("fake_node", "real_node")),
    RuntimeField("max_latency_threshold", "MAX_LATENCY_THRESHOLD", "max latency threshold", "QPS max latency bottleneck threshold, in milliseconds.", "int", required=False, description="QPS max latency bottleneck threshold, in milliseconds.", applies_to=("fake_node", "real_node")),
    RuntimeField("bottleneck_cpu_threshold", "BOTTLENECK_CPU_THRESHOLD", "CPU bottleneck threshold", "CPU utilization bottleneck threshold, as a percentage.", "int", required=False, description="CPU utilization bottleneck threshold, as a percentage.", applies_to=("fake_node", "real_node")),
    RuntimeField("bottleneck_memory_threshold", "BOTTLENECK_MEMORY_THRESHOLD", "memory bottleneck threshold", "Memory utilization bottleneck threshold, as a percentage.", "int", required=False, description="Memory utilization bottleneck threshold, as a percentage.", applies_to=("fake_node", "real_node")),
    RuntimeField("bottleneck_disk_util_threshold", "BOTTLENECK_DISK_UTIL_THRESHOLD", "disk utilization bottleneck threshold", "Disk utilization bottleneck threshold, as a percentage.", "int", required=False, description="Disk utilization bottleneck threshold, as a percentage.", applies_to=("fake_node", "real_node")),
    RuntimeField("bottleneck_disk_latency_threshold", "BOTTLENECK_DISK_LATENCY_THRESHOLD", "disk latency bottleneck threshold", "Disk latency bottleneck threshold, in milliseconds.", "int", required=False, description="Disk latency bottleneck threshold, in milliseconds.", applies_to=("fake_node", "real_node")),
    RuntimeField("bottleneck_network_threshold", "BOTTLENECK_NETWORK_THRESHOLD", "network bottleneck threshold", "Network utilization bottleneck threshold, as a percentage.", "int", required=False, description="Network utilization bottleneck threshold, as a percentage.", applies_to=("fake_node", "real_node")),
    RuntimeField("bottleneck_error_rate_threshold", "BOTTLENECK_ERROR_RATE_THRESHOLD", "error-rate bottleneck threshold", "Error-rate bottleneck threshold, as a percentage.", "int", required=False, description="Error-rate bottleneck threshold, as a percentage.", applies_to=("fake_node", "real_node")),
    RuntimeField("bottleneck_disk_iops_threshold", "BOTTLENECK_DISK_IOPS_THRESHOLD", "disk IOPS bottleneck threshold", "Disk IOPS bottleneck threshold, as a percentage.", "int", required=False, description="Disk IOPS bottleneck threshold, as a percentage.", applies_to=("fake_node", "real_node")),
    RuntimeField("bottleneck_disk_throughput_threshold", "BOTTLENECK_DISK_THROUGHPUT_THRESHOLD", "disk throughput bottleneck threshold", "Disk throughput bottleneck threshold, as a percentage.", "int", required=False, description="Disk throughput bottleneck threshold, as a percentage.", applies_to=("fake_node", "real_node")),
)


# Every RuntimeField across every catalog, mode-agnostic. Lookups that answer
# "what is this field for" (e.g. `_config_field_knowledge`) need to find a field
# regardless of which mode's flow it belongs to, unlike `field_specs_for`, which
# filters by mode for ordering a single flow's checklist.
ALL_RUNTIME_FIELDS: tuple[RuntimeField, ...] = (
    *WORKFLOW_CONFIRMATION_FIELDS,
    *RUNTIME_BASELINE_FIELDS,
    *OPTIONAL_ACCOUNTS_FIELDS,
    *REAL_NODE_ENDPOINT_FIELDS,
    *SYNC_OBSERVE_FIELDS,
    *QPS_AND_TUNING_FIELDS,
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


# ALL_CAPS env-var name -> canonical snake_case logical key. Derived from the
# field catalogs above, plus the handful of env vars that don't map 1:1 onto a
# RuntimeField (a different source variable feeding the same logical key).
ENV_TO_KEY: dict[str, str] = {
    field.env: field.key
    for field in (
        *RUNTIME_BASELINE_FIELDS,
        *OPTIONAL_ACCOUNTS_FIELDS,
        *REAL_NODE_ENDPOINT_FIELDS,
        *WORKFLOW_CONFIRMATION_FIELDS,
        *SYNC_OBSERVE_FIELDS,
    )
    if field.env
}
# These two have no 1:1 `RuntimeField.env`, so they can't come from the
# comprehension above: BLOCKCHAIN_PROCESS_NAMES_STR is a second, string-typed
# source variable for the same `blockchain_process_names` logical key, and
# OBSERVABILITY_STACK_MODE has no dedicated RuntimeField at all.
ENV_TO_KEY.update(
    {
        "BLOCKCHAIN_PROCESS_NAMES_STR": "blockchain_process_names",
        "OBSERVABILITY_STACK_MODE": "observability_choice_confirmed",
    }
)


def field_specs_for(mode: str) -> tuple[RuntimeField, ...]:
    """Return the canonical, ordered field catalog for one workflow mode.

    `mode` is one of "fake_node", "real_node", "sync_observe". This is the
    single place that decides both field membership and order for a given
    mode; other modules derive their catalogs from this instead of
    hand-retyping field lists (architecture audit Finding C2).
    """

    mode = (mode or "").strip().lower().replace("-", "_")
    catalog = (
        *WORKFLOW_CONFIRMATION_FIELDS,
        *RUNTIME_BASELINE_FIELDS,
        *OPTIONAL_ACCOUNTS_FIELDS,
        *REAL_NODE_ENDPOINT_FIELDS,
        *SYNC_OBSERVE_FIELDS,
    )
    return tuple(field for field in catalog if mode in field.applies_to)


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


def validate_mixed_weighted(requirements: dict[str, Any]) -> tuple[bool, str]:
    weighted = requirements.get("mixed_weighted", [])
    if not weighted:
        return False, "mixed mode requires rpc_methods.mixed_weighted entries"
    total = sum(int(item.get("weight", 0) or 0) for item in weighted)
    if total != 100:
        return False, f"mixed weights must total 100, got {total}"
    return True, "mixed weights total 100"
