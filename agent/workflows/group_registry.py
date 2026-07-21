"""Authoritative workflow-group specifications.

This pure metadata module is the single source for group order, ownership,
owned fields, and registered question ids. Runtime domains own
question construction and readiness behavior; this registry does not parse
user text, mutate state, or render responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


NavigationEntry = Literal["question_or_status", "action_only"]


@dataclass(frozen=True)
class GroupSpec:
    name: str
    owner: str
    fields: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    product_node: str = ""
    category: str = "setup"
    workflow_modes: tuple[str, ...] = ()
    fallback: bool = True
    navigation_entry: NavigationEntry = "question_or_status"


GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec(
        name="opening",
        owner="orientation",
        questions=("opening_next_action", "resume_harness_session"),
        product_node="opening",
    ),
    GroupSpec(
        name="target_mode",
        owner="chain_rpc",
        fields=("target_mode", "workflow_mode", "use_fake_node"),
        questions=("target_mode_select", "target_mode_change_confirm"),
        invalidates=(
            "endpoint_process", "workload_rpc", "target_samples_fixtures",
            "qps_profile", "sync_observe", "preflight_smoke_execution", "job_monitoring",
        ),
        product_node="target_mode",
    ),
    GroupSpec(
        name="chain_identity",
        owner="chain_rpc",
        fields=("BLOCKCHAIN_NODE", "chain_identity", "secondary_handoff"),
        questions=(
            "chain",
            "chain_change_input",
            "chain_ambiguity_confirm",
            "chain_change_confirm",
            "unknown_chain_identity_confirm",
            "adapter_family_confirm",
            "case3_protocol_evidence",
            "case3_evidence_next",
        ),
        depends_on=("target_mode",),
        invalidates=(
            "endpoint_process", "chain_auxiliary_endpoints", "workload_rpc",
            "target_samples_fixtures", "preflight_smoke_execution", "job_monitoring",
        ),
        product_node="chain_selection",
    ),
    GroupSpec(
        name="provider_deployment",
        owner="environment",
        fields=(
            "CLOUD_PROVIDER",
            "CLOUD_REGION",
            "CLOUD_ZONE",
            "MACHINE_TYPE",
        ),
        questions=("CLOUD_REGION", "CLOUD_ZONE", "MACHINE_TYPE", "inferred_config_review"),
        invalidates=("preflight_smoke_execution", "job_monitoring"),
        product_node="environment_config",
    ),
    GroupSpec(
        name="ledger_disk",
        owner="environment",
        fields=(
            "LEDGER_DEVICE",
            "DATA_VOL_TYPE",
            "DATA_VOL_SIZE",
            "DATA_VOL_MAX_IOPS",
            "DATA_VOL_MAX_THROUGHPUT",
        ),
        invalidates=("preflight_smoke_execution", "job_monitoring"),
        questions=(
            "LEDGER_DEVICE",
            "DATA_VOL_TYPE",
            "DATA_VOL_SIZE",
            "DATA_VOL_MAX_IOPS",
            "DATA_VOL_MAX_THROUGHPUT",
            "inferred_config_review",
        ),
        product_node="environment_config",
    ),
    GroupSpec(
        name="accounts_disk",
        owner="environment",
        fields=(
            "has_accounts_device",
            "ACCOUNTS_DEVICE",
            "ACCOUNTS_VOL_TYPE",
            "ACCOUNTS_VOL_SIZE",
            "ACCOUNTS_VOL_MAX_IOPS",
            "ACCOUNTS_VOL_MAX_THROUGHPUT",
        ),
        invalidates=("preflight_smoke_execution", "job_monitoring"),
        questions=(
            "has_accounts_device",
            "ACCOUNTS_DEVICE",
            "ACCOUNTS_VOL_TYPE",
            "ACCOUNTS_VOL_SIZE",
            "ACCOUNTS_VOL_MAX_IOPS",
            "ACCOUNTS_VOL_MAX_THROUGHPUT",
            "inferred_config_review",
        ),
        product_node="environment_config",
    ),
    GroupSpec(
        name="network",
        owner="environment",
        fields=("NETWORK_INTERFACE", "NETWORK_MAX_BANDWIDTH_GBPS"),
        questions=("network_interface", "NETWORK_MAX_BANDWIDTH_GBPS", "inferred_config_review"),
        invalidates=("preflight_smoke_execution", "job_monitoring"),
        product_node="environment_config",
    ),
    GroupSpec(
        name="endpoint_process",
        owner="chain_rpc",
        fields=("LOCAL_RPC_URL", "MAINNET_RPC_URL", "MAINNET_RPC_URL_REVIEWED", "BLOCKCHAIN_PROCESS_NAMES", "SYNC_OBSERVE_RPC_URL", "endpoint_evidence"),
        questions=(
            "LOCAL_RPC_URL",
            "SYNC_OBSERVE_RPC_URL",
            "MAINNET_RPC_URL_REVIEWED",
            "BLOCKCHAIN_PROCESS_NAMES",
            "custom_rpc_endpoint",
            "custom_rpc_method",
            "custom_rpc_schema_evidence",
            "custom_rpc_schema_confirm",
            "custom_rpc_response_confirm",
            "custom_rpc_adapter_family_confirm",
            "custom_rpc_continue",
            "custom_rpc_scope",
            "custom_rpc_single_method",
            "custom_rpc_weights",
            "new_chain_endpoint",
            "new_chain_method",
            "new_chain_schema_evidence",
            "new_chain_schema_confirm",
            "new_chain_response_confirm",
            "new_chain_method_continue",
            "new_chain_workload_scope",
            "new_chain_single_method",
            "new_chain_custom_weights",
        ),
        depends_on=("target_mode", "chain_identity"),
        invalidates=("target_samples_fixtures", "preflight_smoke_execution", "job_monitoring"),
        product_node="endpoint_process",
    ),
    GroupSpec(
        name="chain_auxiliary_endpoints",
        owner="chain_rpc",
        fields=(
            "CHAIN_REST_URL",
            "CHAIN_INDEXER_URL",
            "CHAIN_SIDECAR_URL",
            "CHAIN_EVM_RPC_URL",
            "CHAIN_JSON_RPC_URL",
            "CHAIN_MIRROR_URL",
            "RPC_API_KEY",
        ),
        depends_on=("chain_identity",),
        invalidates=("preflight_smoke_execution", "job_monitoring"),
        questions=(
            "RPC_API_KEY",
        ),
        product_node="real_node_endpoint",
    ),
    GroupSpec(
        name="workload_rpc",
        owner="chain_rpc",
        fields=("rpc_mode", "workload", "custom_rpc"),
        questions=(
            "rpc_mode",
            "workload_confirm",
            "target_change_scope",
        ),
        depends_on=("target_mode", "chain_identity"),
        invalidates=("target_samples_fixtures", "preflight_smoke_execution", "job_monitoring"),
        product_node="rpc_workload",
        workflow_modes=("rpc_benchmark",),
    ),
    GroupSpec(
        name="target_samples_fixtures",
        owner="chain_rpc",
        fields=("target_samples", "fixture_evidence"),
        questions=("new_chain_runtime_choice", "custom_rpc_fixture_choice"),
        depends_on=("chain_identity", "workload_rpc"),
        invalidates=("preflight_smoke_execution", "job_monitoring"),
        product_node="custom_rpc",
        workflow_modes=("rpc_benchmark",),
    ),
    GroupSpec(
        name="qps_profile",
        owner="performance",
        fields=("qps_profile",),
        questions=("benchmark_mode", "qps_profile_confirm", "qps_adjust_field", "qps_adjust_value"),
        depends_on=("target_mode",),
        invalidates=("preflight_smoke_execution", "job_monitoring"),
        product_node="rpc_workload",
        workflow_modes=("rpc_benchmark",),
    ),
    GroupSpec(
        name="sync_observe",
        owner="sync_observe",
        fields=(
            "sync_observe",
            "NODE_PROMETHEUS_METRICS_URL",
            "NODE_PROCESS_PID",
        ),
        depends_on=("target_mode", "chain_identity", "endpoint_process"),
        invalidates=("preflight_smoke_execution", "job_monitoring"),
        questions=(
            "sync_observe_source",
            "sync_observe_after_client_setup",
            "sync_observe_stop_condition",
            "sync_observe_duration_seconds",
        ),
        product_node="sync_observe",
        workflow_modes=("sync_observe",),
    ),
    GroupSpec(
        name="observability",
        owner="performance",
        fields=("observability",),
        questions=("observability_mode",),
        invalidates=("preflight_smoke_execution", "job_monitoring"),
        product_node="observability",
    ),
    GroupSpec(
        name="advanced_tuning",
        owner="performance",
        fields=("advanced_tuning",),
        questions=(
            "advanced_tuning_confirm",
            "advanced_tuning_adjust_field",
            "advanced_tuning_adjust_value",
        ),
        invalidates=("preflight_smoke_execution", "job_monitoring"),
        product_node="advanced_threshold_review",
    ),
    GroupSpec(
        name="preflight_smoke_execution",
        owner="execution",
        fields=("preflight", "plan", "plan_file", "smoke", "final_benchmark"),
        questions=("preflight_smoke_confirm",),
        product_node="preflight_smoke",
        category="execution",
    ),
    GroupSpec(
        name="job_monitoring",
        owner="execution",
        fields=("job",),
        product_node="job_monitoring",
        category="execution",
        fallback=False,
        navigation_entry="action_only",
    ),
    GroupSpec(
        name="failure_recovery",
        owner="recovery",
        fields=("failure_recovery",),
        questions=("failure_recovery_action",),
        product_node="failure_recovery",
        category="execution",
        fallback=False,
        navigation_entry="action_only",
    ),
    GroupSpec(
        name="error_evidence_analysis",
        owner="analysis",
        fields=("evidence_buffer", "evidence_collection"),
        product_node="evidence_review",
        category="analysis",
        fallback=False,
        navigation_entry="action_only",
    ),
    GroupSpec(
        name="report_artifact_analysis",
        owner="analysis",
        fields=("report_context",),
        depends_on=("job_monitoring",),
        product_node="analysis",
        category="analysis",
        fallback=False,
        navigation_entry="action_only",
    ),
)


SUPPORTED_WORKFLOW_MODES = frozenset({"rpc_benchmark", "sync_observe"})


def validate_group_registry(groups: Iterable[GroupSpec]) -> tuple[GroupSpec, ...]:
    """Build a registry only when ownership and graph metadata are unambiguous."""

    registry = tuple(groups)
    names = tuple(group.name for group in registry)
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise RuntimeError(f"duplicate group names in GroupSpec registry: {duplicate_names}")
    if not all(group.owner for group in registry):
        raise RuntimeError("every GroupSpec must declare one owner")

    field_groups: dict[str, list[str]] = {}
    for group in registry:
        for field in group.fields:
            field_groups.setdefault(field, []).append(group.name)
    duplicate_fields = {
        field: owners for field, owners in field_groups.items() if len(owners) > 1
    }
    if duplicate_fields:
        details = ", ".join(
            f"{field}={owners}" for field, owners in sorted(duplicate_fields.items())
        )
        raise RuntimeError(f"duplicate persisted field ownership: {details}")

    known = set(names)
    for group in registry:
        if group.navigation_entry not in {"question_or_status", "action_only"}:
            raise RuntimeError(
                f"invalid navigation entry for GroupSpec {group.name}: {group.navigation_entry}"
            )
        if group.navigation_entry == "action_only" and group.fallback:
            raise RuntimeError(
                f"action-only GroupSpec {group.name} cannot be a fallback destination"
            )
        unknown_dependencies = sorted(set(group.depends_on) - known)
        unknown_invalidations = sorted(set(group.invalidates) - known)
        unknown_modes = sorted(set(group.workflow_modes) - SUPPORTED_WORKFLOW_MODES)
        if unknown_dependencies or unknown_invalidations or unknown_modes:
            raise RuntimeError(
                f"invalid GroupSpec metadata for {group.name}: "
                f"dependencies={unknown_dependencies}, "
                f"invalidations={unknown_invalidations}, modes={unknown_modes}"
            )
        if group.name in group.depends_on:
            raise RuntimeError(f"GroupSpec {group.name} cannot depend on itself")

    dependencies = {group.name: group.depends_on for group in registry}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise RuntimeError(f"GroupSpec dependency cycle includes {name}")
        if name in visited:
            return
        visiting.add(name)
        for dependency in dependencies[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in names:
        visit(name)
    return registry


GROUPS = validate_group_registry(GROUPS)
GROUP_ORDER: tuple[str, ...] = tuple(group.name for group in GROUPS)
GROUP_OWNER: dict[str, str] = {group.name: group.owner for group in GROUPS}
GROUP_FIELD_MAP: dict[str, set[str]] = {group.name: set(group.fields) for group in GROUPS}
FIELD_GROUP: dict[str, str] = {
    field: group.name for group in GROUPS for field in group.fields
}
FIELD_OWNER: dict[str, str] = {
    field: group.owner for group in GROUPS for field in group.fields
}
GROUP_QUESTION_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (group.name, group.questions) for group in GROUPS if group.questions
)
GROUP_TO_PRODUCT_NODE: dict[str, str] = {group.name: group.product_node for group in GROUPS if group.product_node}
GROUP_SPEC_BY_NAME: dict[str, GroupSpec] = {group.name: group for group in GROUPS}
USER_NAVIGABLE_GROUPS: tuple[str, ...] = tuple(
    group.name for group in GROUPS if group.navigation_entry == "question_or_status"
)


def normalize_group_name(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text if text in GROUP_FIELD_MAP else ""


def group_for_field(field_name: str) -> str:
    field = str(field_name or "").strip()
    return FIELD_GROUP.get(field, "") if field else ""


def fallback_groups_for_workflow(workflow_mode: str) -> tuple[GroupSpec, ...]:
    """Return the registry-ordered prerequisites applicable to one product path."""

    mode = str(workflow_mode or "").strip()
    return tuple(
        group
        for group in GROUPS
        if group.fallback and (not group.workflow_modes or mode in group.workflow_modes)
    )


def product_node_for_group(group_name: str) -> str:
    return GROUP_TO_PRODUCT_NODE.get(normalize_group_name(group_name), "")


def invalidation_targets(group_name: str) -> tuple[str, ...]:
    spec = GROUP_SPEC_BY_NAME.get(normalize_group_name(group_name))
    return spec.invalidates if spec else ()


def is_user_navigable_group(group_name: object) -> bool:
    """Return whether ``change_group`` may enter this registry destination."""

    return normalize_group_name(group_name) in USER_NAVIGABLE_GROUPS
