"""Group metadata for `agent/validators/config_contract.py`'s preflight path.

This module is intentionally pure data plus small lookup helpers. It does not
parse user text, mutate workflow state, or render terminal prompts.

It is NOT consulted by the live conversational routing path
(`agent/harness/groups.py` / `agent/harness/routing.py`); that path uses
`agent.harness.state.DEFAULT_GROUP_ORDER`, the actual single source of truth
for group order (architecture audit Finding A). Keep `GROUP_ORDER` below
consistent with `DEFAULT_GROUP_ORDER` so this file's metadata does not
describe a group set that no longer matches reality.
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
        name="opening",
        questions=("opening_next_action",),
        product_node="opening",
    ),
    WorkflowGroup(
        name="target_mode",
        fields=("target_mode", "workflow_mode", "use_fake_node"),
        questions=("target_mode", "target_mode_change_confirm"),
        product_node="target_mode",
    ),
    WorkflowGroup(
        name="chain_identity",
        fields=("chain", "BLOCKCHAIN_NODE", "chain_identity"),
        questions=(
            "chain",
            "chain_selection",
            "chain_ambiguity_confirm",
            "chain_change_confirm",
            "unknown_chain_identity_confirm",
            "adapter_family_confirm",
        ),
        product_node="chain_selection",
    ),
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
        name="endpoint_process",
        fields=("LOCAL_RPC_URL", "MAINNET_RPC_URL", "MAINNET_RPC_URL_REVIEWED", "BLOCKCHAIN_PROCESS_NAMES", "SYNC_OBSERVE_RPC_URL"),
        questions=(
            "LOCAL_RPC_URL",
            "SYNC_OBSERVE_RPC_URL",
            "MAINNET_RPC_URL_REVIEWED",
            "BLOCKCHAIN_PROCESS_NAMES",
            "custom_rpc_endpoint",
            "custom_rpc_method",
            "custom_rpc_schema_evidence",
            "custom_rpc_schema_confirm",
            "custom_rpc_adapter_family_confirm",
            "custom_rpc_continue",
            "custom_rpc_scope",
            "custom_rpc_single_method",
            "custom_rpc_weights",
            "new_chain_endpoint",
            "new_chain_method",
            "new_chain_schema_evidence",
            "new_chain_schema_confirm",
            "new_chain_method_continue",
            "new_chain_workload_scope",
            "new_chain_single_method",
            "new_chain_custom_weights",
        ),
        product_node="endpoint_process",
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
        questions=(
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
        questions=(
            "advanced_tuning_confirm",
            "advanced_tuning_adjust_field",
            "advanced_tuning_adjust_value",
        ),
        product_node="advanced_threshold_review",
    ),
    WorkflowGroup(
        name="preflight_smoke_execution",
        fields=("preflight_result", "smoke_result", "approval"),
        questions=("preflight_smoke_confirm",),
        product_node="preflight_smoke",
        category="execution",
    ),
    WorkflowGroup(
        name="job_monitoring",
        fields=("job", "latest_job_id"),
        product_node="job_monitoring",
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
        "opening_next_action",
        "target_mode",
        "target_mode_change_confirm",
        "chain_ambiguity_confirm",
        "chain_change_confirm",
        "unknown_chain_identity_confirm",
        "adapter_family_confirm",
        "real_node_local_rpc_url",
        "LOCAL_RPC_URL",
        "SYNC_OBSERVE_RPC_URL",
        "MAINNET_RPC_URL_REVIEWED",
        "custom_rpc_endpoint",
        "custom_rpc_method",
        "custom_rpc_schema_evidence",
        "custom_rpc_schema_confirm",
        "custom_rpc_adapter_family_confirm",
        "custom_rpc_continue",
        "custom_rpc_scope",
        "custom_rpc_single_method",
        "custom_rpc_weights",
        "new_chain_endpoint",
        "new_chain_method",
        "new_chain_schema_evidence",
        "new_chain_schema_confirm",
        "new_chain_method_continue",
        "new_chain_workload_scope",
        "new_chain_single_method",
        "new_chain_custom_weights",
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
        "opening_next_action": "opening",
        "target_mode": "target_mode",
        "target_mode_change_confirm": "target_mode",
        "chain": "chain_identity",
        "chain_selection": "chain_identity",
        "chain_ambiguity_confirm": "chain_identity",
        "chain_change_confirm": "chain_identity",
        "unknown_chain_identity_confirm": "chain_identity",
        "adapter_family_confirm": "chain_identity",
        "real_node_local_rpc_url": "endpoint_process",
        "local_rpc_url": "endpoint_process",
        "LOCAL_RPC_URL": "endpoint_process",
        "SYNC_OBSERVE_RPC_URL": "endpoint_process",
        "MAINNET_RPC_URL_REVIEWED": "endpoint_process",
        "BLOCKCHAIN_PROCESS_NAMES": "endpoint_process",
        "rpc_mode_choice": "workload_rpc",
        "default_workload_confirm": "workload_rpc",
        "workload_confirm": "workload_rpc",
        "workload_customization_choice": "workload_rpc",
        "mixed_weights_confirm": "workload_rpc",
        "custom_rpc_endpoint_gate": "endpoint_process",
        "custom_rpc_endpoint": "endpoint_process",
        "custom_rpc_method": "endpoint_process",
        "custom_rpc_schema_evidence": "endpoint_process",
        "custom_rpc_schema_confirm": "endpoint_process",
        "custom_rpc_adapter_family_confirm": "endpoint_process",
        "custom_rpc_continue": "endpoint_process",
        "custom_rpc_scope": "endpoint_process",
        "custom_rpc_single_method": "endpoint_process",
        "custom_rpc_weights": "endpoint_process",
        "benchmark_profile_choice": "qps_profile",
        "benchmark_profile_confirm": "qps_profile",
        "benchmark_profile_adjust_item": "qps_profile",
        "benchmark_profile_adjust_value": "qps_profile",
        "observability_mode_choice": "observability",
        "observability_ports_confirm": "observability",
        "smoke_run_confirm": "preflight_smoke_execution",
        "apply_pasted_evidence": "error_evidence_analysis",
        "chain_identity_resolution": "chain_identity",
        "chain_protocol_resolution": "chain_identity",
        "unsupported_chain_endpoint_gate": "endpoint_process",
        "unsupported_chain_handoff_confirm": "chain_identity",
        "new_chain_endpoint": "endpoint_process",
        "new_chain_method": "endpoint_process",
        "new_chain_schema_evidence": "endpoint_process",
        "new_chain_schema_confirm": "endpoint_process",
        "new_chain_method_continue": "endpoint_process",
        "new_chain_workload_scope": "endpoint_process",
        "new_chain_single_method": "endpoint_process",
        "new_chain_custom_weights": "endpoint_process",
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
