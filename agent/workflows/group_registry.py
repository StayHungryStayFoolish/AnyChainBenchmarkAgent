"""Canonical group registry for the AnyChain Agent workflow.

This module is intentionally pure data plus small lookup helpers. It is the
source of truth for group order, field ownership, and group question ids used
by the LangGraph Harness. It does not parse user text, mutate workflow state,
or render terminal prompts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowGroup:
    name: str
    fields: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()
    product_node: str = ""
    category: str = "setup"


GROUPS: tuple[WorkflowGroup, ...] = (
    WorkflowGroup(
        name="provider_deployment",
        fields=(
            "CLOUD_PROVIDER",
            "CLOUD_REGION",
            "CLOUD_ZONE",
            "MACHINE_TYPE",
            "deployment",
            "cloud_provider",
            "cloud_region",
            "cloud_zone",
            "machine_type",
        ),
        questions=("cloud_region", "cloud_zone", "machine_type"),
        product_node="environment_config",
    ),
    WorkflowGroup(
        name="hardware_discovery",
        fields=("cpu", "memory", "disk_inventory", "network_inventory"),
        product_node="environment_config",
    ),
    WorkflowGroup(
        name="ledger_disk",
        fields=(
            "LEDGER_DEVICE",
            "DATA_VOL_TYPE",
            "DATA_VOL_SIZE",
            "DATA_VOL_MAX_IOPS",
            "DATA_VOL_MAX_THROUGHPUT",
            "ledger_device",
            "data_vol_type",
            "data_vol_size",
            "data_vol_max_iops",
            "data_vol_max_throughput",
        ),
        questions=(
            "ledger_device",
            "data_vol_type",
            "data_vol_size",
            "data_vol_max_iops",
            "data_vol_max_throughput",
        ),
        product_node="environment_config",
    ),
    WorkflowGroup(
        name="accounts_disk",
        fields=(
            "has_accounts_device",
            "ACCOUNTS_DEVICE",
            "ACCOUNTS_VOL_TYPE",
            "ACCOUNTS_VOL_SIZE",
            "ACCOUNTS_VOL_MAX_IOPS",
            "ACCOUNTS_VOL_MAX_THROUGHPUT",
            "accounts_device",
            "accounts_vol_type",
            "accounts_vol_size",
            "accounts_vol_max_iops",
            "accounts_vol_max_throughput",
        ),
        questions=(
            "has_accounts_device",
            "accounts_device",
            "accounts_vol_type",
            "accounts_vol_size",
            "accounts_vol_max_iops",
            "accounts_vol_max_throughput",
        ),
        product_node="environment_config",
    ),
    WorkflowGroup(
        name="network",
        fields=("NETWORK_INTERFACE", "NETWORK_MAX_BANDWIDTH_GBPS", "network_interface", "network_max_bandwidth_gbps"),
        questions=("network_interface", "network_max_bandwidth_gbps"),
        product_node="environment_config",
    ),
    WorkflowGroup(
        name="chain_target",
        fields=("target_mode", "chain", "BLOCKCHAIN_NODE", "LOCAL_RPC_URL", "MAINNET_RPC_URL", "BLOCKCHAIN_PROCESS_NAMES"),
        questions=("chain", "use_fake_node", "local_rpc_url", "mainnet_rpc_url_reviewed", "blockchain_process_names"),
        product_node="chain_selection",
    ),
    WorkflowGroup(
        name="chain_auxiliary_endpoints",
        fields=(
            "CHAIN_REST_URL",
            "CHAIN_INDEXER_URL",
            "CHAIN_SIDECAR_URL",
            "CHAIN_EVM_RPC_URL",
            "CHAIN_JSON_RPC_URL",
            "CHAIN_MIRROR_URL",
            "RPC_API_KEY",
        ),
        product_node="real_node_endpoint",
    ),
    WorkflowGroup(
        name="workload_rpc",
        fields=("rpc_mode", "RPC_MODE", "rpc_methods", "mixed_weights", "custom_rpc", "custom_rpc_methods"),
        questions=(
            "rpc_mode",
            "workload_customization_choice",
        ),
        product_node="rpc_workload",
    ),
    WorkflowGroup(
        name="target_samples_fixtures",
        fields=("target_samples", "fixture_status"),
        product_node="custom_rpc",
    ),
    WorkflowGroup(
        name="qps_profile",
        fields=("benchmark_profile",),
        questions=("benchmark_mode_confirmed", "qps_profile_confirmed"),
        product_node="rpc_workload",
    ),
    WorkflowGroup(
        name="sync_observe",
        fields=(
            "sync_observe",
            "sync_observe_stop_condition",
            "NODE_PROMETHEUS_METRICS_URL",
            "NODE_PROCESS_PID",
            "BLOCKCHAIN_PROCESS_NAMES",
            "MAINNET_RPC_URL",
            "node_prometheus_metrics_url",
            "node_process_pid",
            "node_process_identity",
            "blockchain_process_names",
            "mainnet_rpc_url_reviewed",
        ),
        questions=(
            "sync_observe_stop_condition",
            "node_prometheus_metrics_url",
            "node_process_identity",
            "mainnet_rpc_url_reviewed",
        ),
        product_node="sync_observe",
    ),
    WorkflowGroup(
        name="observability",
        fields=("observability",),
        questions=("observability_choice_confirmed",),
        product_node="observability",
    ),
    WorkflowGroup(
        name="advanced_tuning",
        fields=("advanced_tuning",),
        product_node="advanced_threshold_review",
    ),
    WorkflowGroup(
        name="preflight_smoke_execution",
        fields=("preflight_result", "smoke_result", "approval"),
        product_node="preflight_smoke",
        category="execution",
    ),
    WorkflowGroup(
        name="error_evidence_analysis",
        fields=("evidence", "evidence_buffer"),
        product_node="evidence_review",
        category="analysis",
    ),
    WorkflowGroup(
        name="report_artifact_analysis",
        fields=("latest_job_id", "latest_plan_file"),
        product_node="analysis",
        category="analysis",
    ),
)


GROUP_ORDER: tuple[str, ...] = tuple(group.name for group in GROUPS)
GROUP_FIELD_MAP: dict[str, set[str]] = {group.name: set(group.fields) for group in GROUPS}
GROUP_QUESTION_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (group.name, group.questions) for group in GROUPS if group.questions
)
GROUP_TO_PRODUCT_NODE: dict[str, str] = {group.name: group.product_node for group in GROUPS if group.product_node}

SETUP_QUESTION_IDS: frozenset[str] = frozenset(
    {
        "chain",
        "chain_selection",
        "target_mode",
        "real_node_local_rpc_url",
        "mainnet_rpc_url_reviewed",
        "cloud_region",
        "cloud_zone",
        "machine_type",
        "disk_ledger_choice",
        "disk_accounts_exists",
        "disk_accounts_choice",
        "blockchain_process_names",
        "benchmark_profile_choice",
        "benchmark_profile_adjust_item",
        "benchmark_profile_adjust_value",
        "sync_observe_stop_condition",
        "node_prometheus_metrics_url",
        "node_process_identity",
        "default_workload_confirm",
        "workload_confirm",
        "workload_customization_choice",
        "mixed_weights_confirm",
        "observability_mode_choice",
        "observability_ports_confirm",
        *[question for _group, questions in GROUP_QUESTION_ORDER for question in questions],
        "local_rpc_url",
        "rpc_mode_choice",
        "benchmark_profile_confirm",
    }
)

POST_CONFIG_QUESTION_IDS: frozenset[str] = frozenset(
    {
        "default_workload_confirm",
        "workload_confirm",
        "workload_customization_choice",
        "benchmark_profile_choice",
        "benchmark_profile_confirm",
        "benchmark_profile_adjust_item",
        "benchmark_profile_adjust_value",
        "observability_mode_choice",
        "observability_ports_confirm",
    }
)

POST_CONFIG_BRANCHES: frozenset[str] = frozenset({"rpc_workload", "benchmark_profile", "observability"})


def normalize_group_name(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text if text in GROUP_FIELD_MAP else ""


def group_for_field(field_name: str) -> str:
    field = str(field_name or "").strip()
    if not field:
        return ""
    for group, fields in GROUP_FIELD_MAP.items():
        if field in fields:
            return group
    return ""


def group_for_question_id(question_id: str) -> str:
    qid = str(question_id or "").strip()
    if not qid:
        return ""
    for group, questions in GROUP_QUESTION_ORDER:
        if qid in questions:
            return group
    aliases = {
        "disk_ledger_choice": "ledger_disk",
        "disk_accounts_exists": "accounts_disk",
        "disk_accounts_choice": "accounts_disk",
        "target_mode": "chain_target",
        "chain_selection": "chain_target",
        "real_node_local_rpc_url": "chain_target",
        "local_rpc_url": "chain_target",
        "rpc_mode_choice": "workload_rpc",
        "default_workload_confirm": "workload_rpc",
        "workload_confirm": "workload_rpc",
        "workload_customization_choice": "workload_rpc",
        "mixed_weights_confirm": "workload_rpc",
        "custom_rpc_endpoint_gate": "target_samples_fixtures",
        "benchmark_profile_choice": "qps_profile",
        "benchmark_profile_confirm": "qps_profile",
        "benchmark_profile_adjust_item": "qps_profile",
        "benchmark_profile_adjust_value": "qps_profile",
        "observability_mode_choice": "observability",
        "observability_ports_confirm": "observability",
        "smoke_run_confirm": "preflight_smoke_execution",
        "apply_pasted_evidence": "error_evidence_analysis",
        "chain_identity_resolution": "chain_target",
        "chain_protocol_resolution": "chain_target",
        "unsupported_chain_endpoint_gate": "chain_target",
        "unsupported_chain_handoff_confirm": "chain_target",
    }
    return aliases.get(qid, "")


def product_node_for_group(group_name: str) -> str:
    return GROUP_TO_PRODUCT_NODE.get(normalize_group_name(group_name), "")


def question_keys_for_group(group_name: str | None) -> tuple[str, ...]:
    group = normalize_group_name(group_name)
    if not group:
        return ()
    for name, questions in GROUP_QUESTION_ORDER:
        if name == group:
            return tuple(questions)
    return ()


def is_setup_question_id(question_id: str) -> bool:
    return str(question_id or "").strip() in SETUP_QUESTION_IDS


def is_post_config_question(question_id: str, branch: str = "") -> bool:
    return str(question_id or "").strip() in POST_CONFIG_QUESTION_IDS or str(branch or "").strip() in POST_CONFIG_BRANCHES
