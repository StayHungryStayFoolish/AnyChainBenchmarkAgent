"""Authoritative workflow-group specifications.

This pure metadata module is the single source for group order, ownership,
owned fields, and registered question ids. Runtime domains own
question construction and readiness behavior; this registry does not parse
user text, mutate state, or render responses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping


NavigationEntry = Literal["question_or_status", "action_only"]
RPC_EXTENSION_ENDPOINT_STATUSES = frozenset({
    "needs_endpoint",
    "needs_method",
    "needs_schema_evidence",
    "schema_needs_confirmation",
    "needs_adapter_family_confirmation",
    "needs_scope",
    "needs_single_method",
    "needs_weights",
    "probe_failed",
})
NEW_CHAIN_ENDPOINT_STATUSES = frozenset({
    "existing_family_needs_endpoint",
    "existing_family_needs_method",
    "existing_family_needs_schema_evidence",
    "existing_family_schema_needs_confirmation",
    "existing_family_needs_workload_scope",
    "existing_family_needs_single_method",
    "existing_family_needs_weights",
})


@dataclass(frozen=True)
class GroupSpec:
    name: str
    owner: str
    fields: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()
    sensitive_fields: tuple[str, ...] = ()
    sensitive_questions: tuple[str, ...] = ()
    reconfiguration_questions: tuple[tuple[str, str], ...] = ()
    immutable_fields: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    product_node: str = ""
    category: str = "setup"
    workflow_modes: tuple[str, ...] = ()
    target_modes: tuple[str, ...] = ()
    fallback: bool = True
    navigation_entry: NavigationEntry = "question_or_status"
    resume_selector: bool = True
    generic_navigation: bool = True


GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec(
        name="opening",
        owner="orientation",
        questions=(
            "opening_next_action",
            "resume_harness_session",
            "resume_modify_group",
        ),
        product_node="opening",
        resume_selector=False,
        generic_navigation=False,
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
        reconfiguration_questions=(
            ("CLOUD_REGION", "CLOUD_REGION"),
            ("CLOUD_ZONE", "CLOUD_ZONE"),
            ("MACHINE_TYPE", "MACHINE_TYPE"),
        ),
        immutable_fields=("CLOUD_PROVIDER",),
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
        reconfiguration_questions=(
            ("LEDGER_DEVICE", "LEDGER_DEVICE"),
            ("DATA_VOL_TYPE", "DATA_VOL_TYPE"),
            ("DATA_VOL_SIZE", "DATA_VOL_SIZE"),
            ("DATA_VOL_MAX_IOPS", "DATA_VOL_MAX_IOPS"),
            ("DATA_VOL_MAX_THROUGHPUT", "DATA_VOL_MAX_THROUGHPUT"),
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
        reconfiguration_questions=(
            ("has_accounts_device", "has_accounts_device"),
            ("ACCOUNTS_DEVICE", "ACCOUNTS_DEVICE"),
            ("ACCOUNTS_VOL_TYPE", "ACCOUNTS_VOL_TYPE"),
            ("ACCOUNTS_VOL_SIZE", "ACCOUNTS_VOL_SIZE"),
            ("ACCOUNTS_VOL_MAX_IOPS", "ACCOUNTS_VOL_MAX_IOPS"),
            ("ACCOUNTS_VOL_MAX_THROUGHPUT", "ACCOUNTS_VOL_MAX_THROUGHPUT"),
        ),
        product_node="environment_config",
    ),
    GroupSpec(
        name="network",
        owner="environment",
        fields=("NETWORK_INTERFACE", "NETWORK_MAX_BANDWIDTH_GBPS"),
        questions=("network_interface", "NETWORK_MAX_BANDWIDTH_GBPS", "inferred_config_review"),
        reconfiguration_questions=(
            ("NETWORK_INTERFACE", "network_interface"),
            ("NETWORK_MAX_BANDWIDTH_GBPS", "NETWORK_MAX_BANDWIDTH_GBPS"),
        ),
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
        sensitive_fields=(
            "LOCAL_RPC_URL",
            "MAINNET_RPC_URL",
            "SYNC_OBSERVE_RPC_URL",
            "custom_rpc_endpoint",
            "new_chain_endpoint",
        ),
        sensitive_questions=(
            "LOCAL_RPC_URL",
            "SYNC_OBSERVE_RPC_URL",
            "MAINNET_RPC_URL_REVIEWED",
            "custom_rpc_endpoint",
            "new_chain_endpoint",
        ),
        depends_on=("target_mode", "chain_identity"),
        invalidates=("target_samples_fixtures", "preflight_smoke_execution", "job_monitoring"),
        product_node="endpoint_process",
        target_modes=("real-node", "sync-observe"),
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
        sensitive_fields=(
            "CHAIN_REST_URL",
            "CHAIN_INDEXER_URL",
            "CHAIN_SIDECAR_URL",
            "CHAIN_EVM_RPC_URL",
            "CHAIN_JSON_RPC_URL",
            "CHAIN_MIRROR_URL",
            "RPC_API_KEY",
        ),
        sensitive_questions=(
            "RPC_API_KEY",
        ),
        product_node="real_node_endpoint",
        target_modes=("real-node", "sync-observe"),
        resume_selector=False,
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
        target_modes=("fake-node", "real-node"),
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
        target_modes=("fake-node", "real-node"),
        resume_selector=False,
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
        target_modes=("fake-node", "real-node"),
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
        target_modes=("sync-observe",),
    ),
    GroupSpec(
        name="observability",
        owner="performance",
        fields=("observability",),
        questions=("observability_mode",),
        depends_on=("target_mode",),
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
        resume_selector=False,
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
SUPPORTED_TARGET_MODES = frozenset({"fake-node", "real-node", "sync-observe"})


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
        duplicate_sensitive_fields = sorted({
            field
            for field in group.sensitive_fields
            if group.sensitive_fields.count(field) > 1
        })
        duplicate_sensitive_questions = sorted({
            question
            for question in group.sensitive_questions
            if group.sensitive_questions.count(question) > 1
        })
        unknown_sensitive_questions = sorted(
            set(group.sensitive_questions) - set(group.questions)
        )
        if (
            duplicate_sensitive_fields
            or duplicate_sensitive_questions
            or unknown_sensitive_questions
        ):
            raise RuntimeError(
                f"invalid sensitive intake metadata for GroupSpec {group.name}: "
                f"fields={duplicate_sensitive_fields}, "
                f"questions={duplicate_sensitive_questions}, "
                f"unknown_questions={unknown_sensitive_questions}"
            )
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
        unknown_target_modes = sorted(set(group.target_modes) - SUPPORTED_TARGET_MODES)
        if unknown_dependencies or unknown_invalidations or unknown_modes or unknown_target_modes:
            raise RuntimeError(
                f"invalid GroupSpec metadata for {group.name}: "
                f"dependencies={unknown_dependencies}, "
                f"invalidations={unknown_invalidations}, modes={unknown_modes}, "
                f"target_modes={unknown_target_modes}"
            )
        if group.name in group.depends_on:
            raise RuntimeError(f"GroupSpec {group.name} cannot depend on itself")
        reconfiguration_fields = [field for field, _question in group.reconfiguration_questions]
        duplicate_reconfiguration_fields = sorted({
            field for field in reconfiguration_fields if reconfiguration_fields.count(field) > 1
        })
        unknown_reconfiguration_fields = sorted(set(reconfiguration_fields) - set(group.fields))
        unknown_reconfiguration_questions = sorted({
            question
            for _field, question in group.reconfiguration_questions
            if question not in group.questions
        })
        unknown_immutable_fields = sorted(set(group.immutable_fields) - set(group.fields))
        conflicting_field_policies = sorted(
            set(group.immutable_fields).intersection(reconfiguration_fields)
        )
        if (
            duplicate_reconfiguration_fields
            or unknown_reconfiguration_fields
            or unknown_reconfiguration_questions
            or unknown_immutable_fields
            or conflicting_field_policies
        ):
            raise RuntimeError(
                f"invalid reconfiguration questions for GroupSpec {group.name}: "
                f"duplicates={duplicate_reconfiguration_fields}, "
                f"fields={unknown_reconfiguration_fields}, "
                f"questions={unknown_reconfiguration_questions}, "
                f"immutable={unknown_immutable_fields}, conflicts={conflicting_field_policies}"
            )

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


def navigation_prerequisites(group: GroupSpec) -> tuple[str, ...]:
    """Return every group that can visibly defer public navigation.

    A workflow-mode mismatch is resolved through ``target_mode`` before the
    destination's declared dependencies. Keep this ordering aligned with the
    coordinator's prerequisite routing contract.
    """

    prerequisites: list[str] = []
    if group.workflow_modes:
        prerequisites.append("target_mode")
    prerequisites.extend(group.depends_on)
    return tuple(dict.fromkeys(prerequisites))


def validate_action_group_requirements(
    groups: Iterable[GroupSpec],
    action_specs: Iterable[Any],
) -> None:
    """Reject action prerequisites that their destination group does not own.

    ActionSpec remains owned by the Harness action registry. This validation
    accepts structural objects to avoid coupling the pure group registry back
    to that module while still providing one cross-registry consistency gate.
    """

    registry = validate_group_registry(groups)
    by_name = {group.name: group for group in registry}

    def transitive_dependencies(group_name: str) -> set[str]:
        discovered: set[str] = set()
        pending = list(by_name[group_name].depends_on)
        while pending:
            dependency = pending.pop()
            if dependency in discovered:
                continue
            discovered.add(dependency)
            pending.extend(by_name[dependency].depends_on)
        return discovered

    for action in action_specs:
        action_type = str(getattr(action, "action_type", "") or "")
        target_group = str(getattr(action, "target_group", "") or "")
        requirements = {
            str(item)
            for item in (getattr(action, "requires_capabilities", ()) or ())
            if str(item)
        }
        if not target_group:
            continue
        if target_group not in by_name:
            raise RuntimeError(
                f"ActionSpec {action_type or '<unknown>'} targets unknown GroupSpec "
                f"{target_group}"
            )
        unknown_requirements = sorted(requirements - set(by_name))
        missing_dependencies = sorted(
            requirements - transitive_dependencies(target_group)
        )
        if unknown_requirements or missing_dependencies:
            raise RuntimeError(
                f"ActionSpec {action_type or '<unknown>'} prerequisites conflict "
                f"with GroupSpec {target_group}: unknown={unknown_requirements}, "
                f"missing_dependencies={missing_dependencies}"
            )


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
FIELD_RECONFIGURATION_QUESTION: dict[str, str] = {
    field: question
    for group in GROUPS
    for field, question in group.reconfiguration_questions
}
GROUP_QUESTION_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (group.name, group.questions) for group in GROUPS if group.questions
)
GROUP_TO_PRODUCT_NODE: dict[str, str] = {group.name: group.product_node for group in GROUPS if group.product_node}
GROUP_SPEC_BY_NAME: dict[str, GroupSpec] = {group.name: group for group in GROUPS}
USER_NAVIGABLE_GROUPS: tuple[str, ...] = tuple(
    group.name
    for group in GROUPS
    if group.navigation_entry == "question_or_status" and group.generic_navigation
)


def normalize_group_name(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text if text in GROUP_FIELD_MAP else ""


def group_for_field(field_name: str) -> str:
    field = str(field_name or "").strip()
    return FIELD_GROUP.get(field, "") if field else ""


def is_sensitive_question(
    group_name: object,
    question_id: object,
    field_name: object,
) -> bool:
    """Return the registry-owned sensitivity contract for one intake."""

    spec = GROUP_SPEC_BY_NAME.get(normalize_group_name(group_name))
    if spec is None:
        return False
    return (
        str(question_id or "").strip() in spec.sensitive_questions
        or str(field_name or "").strip() in spec.sensitive_fields
    )


def reconfiguration_question_for_field(field_name: str) -> str:
    """Return the registered typed question for an explicitly edited field."""

    field = str(field_name or "").strip()
    return FIELD_RECONFIGURATION_QUESTION.get(field, "") if field else ""


def group_registry_contract_hash() -> str:
    """Return the content identity of field ownership and intake contracts."""

    payload = [
        {
            "name": group.name,
            "owner": group.owner,
            "fields": list(group.fields),
            "questions": list(group.questions),
            "sensitive_fields": list(group.sensitive_fields),
            "sensitive_questions": list(group.sensitive_questions),
            "reconfiguration_questions": list(group.reconfiguration_questions),
            "immutable_fields": list(group.immutable_fields),
            "depends_on": list(group.depends_on),
            "invalidates": list(group.invalidates),
            "product_node": group.product_node,
            "category": group.category,
            "workflow_modes": list(group.workflow_modes),
            "target_modes": list(group.target_modes),
            "fallback": group.fallback,
            "navigation_entry": group.navigation_entry,
            "resume_selector": group.resume_selector,
            "generic_navigation": group.generic_navigation,
        }
        for group in GROUPS
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def group_applicable(state: Mapping[str, Any], spec: GroupSpec) -> bool:
    """Return whether one group belongs to the state's selected product path."""

    workflow_mode = str(state.get("workflow_mode") or "").strip()
    target_mode = str(state.get("target_mode") or "").strip()
    if spec.name == "endpoint_process" and target_mode == "fake-node":
        identity_status = str(
            (state.get("chain_identity") or {}).get("status") or ""
        )
        custom_status = str((state.get("custom_rpc") or {}).get("status") or "")
        return (
            workflow_mode == "rpc_benchmark"
            and (
                identity_status in NEW_CHAIN_ENDPOINT_STATUSES
                or custom_status in RPC_EXTENSION_ENDPOINT_STATUSES
            )
        )
    if spec.workflow_modes and workflow_mode not in spec.workflow_modes:
        return False
    if spec.target_modes and target_mode not in spec.target_modes:
        return False
    return True


def fallback_groups_for_state(state: Mapping[str, Any]) -> tuple[GroupSpec, ...]:
    """Return registry-ordered fallback groups for the selected product path."""

    return tuple(
        group
        for group in GROUPS
        if group.fallback and group_applicable(state, group)
    )


def product_node_for_group(group_name: str) -> str:
    return GROUP_TO_PRODUCT_NODE.get(normalize_group_name(group_name), "")


def invalidation_targets(group_name: str) -> tuple[str, ...]:
    spec = GROUP_SPEC_BY_NAME.get(normalize_group_name(group_name))
    return spec.invalidates if spec else ()


def is_user_navigable_group(group_name: object) -> bool:
    """Return whether ``change_group`` may enter this registry destination."""

    return normalize_group_name(group_name) in USER_NAVIGABLE_GROUPS
