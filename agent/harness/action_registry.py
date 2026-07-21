"""Authoritative model-action schema for the Harness."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

from .input_values import (
    looks_like_wire_method_identity,
    normalize_observability_mode,
    normalize_target_mode,
)


ActionLifetime = Literal["turn_local", "durable"]
ActionEffect = Literal[
    "read_only",
    "workflow_navigation",
    "configuration_mutation",
    "workflow_state_mutation",
    "execution",
]
ActionValidator = Callable[[Mapping[str, Any]], None]

SEMANTIC_SUPPORT_RELATIONS = frozenset({
    "explanatory_context",
    "operation_restatement",
    "provenance",
    "format_scope",
    "temporal_scope",
    "evidence_completeness",
    "non_mutation_scope",
})
FRAMED_OPERATION_SUPPORT_RELATIONS = (
    "explanatory_context",
    "operation_restatement",
    "provenance",
    "format_scope",
    "temporal_scope",
)
EVIDENCE_OPERATION_SUPPORT_RELATIONS = (
    *FRAMED_OPERATION_SUPPORT_RELATIONS,
    "evidence_completeness",
)


SEMANTIC_SCOPE_POLICIES: dict[str, dict[str, Any]] = {
    "consultation_only": {
        "description": "Permit read-only answers only; workflow navigation is not authorized.",
        "allowed_effects": ("read_only",),
    },
    "no_configuration_mutation": {
        "description": "Permit state inspection and workflow navigation without changing benchmark configuration values.",
        "allowed_effects": ("read_only", "workflow_navigation"),
    },
    "no_execution": {
        "description": "Permit inspection, navigation, and configuration while prohibiting execution or retry side effects.",
        "forbidden_effects": ("execution",),
    },
}


def _validate_chain_selection(action: Mapping[str, Any]) -> None:
    chain_text = str(action.get("chain_text") or "").strip()
    source = str(action.get("source_evidence") or "").strip()
    candidates = action.get("chain_candidates")
    candidate_values = [
        str(item).strip()
        for item in candidates
        if str(item).strip()
    ] if isinstance(candidates, list) else []
    if not chain_text and not candidate_values:
        raise ValueError("chain selection requires chain_text or chain_candidates")
    selected_values = [chain_text] if chain_text else candidate_values
    missing = [value for value in selected_values if value.casefold() not in source.casefold()]
    if not source or missing:
        raise ValueError(
            "chain selection values must appear verbatim in source_evidence: "
            + ", ".join(missing or selected_values)
        )


def _validate_rpc_catalog_command(action: Mapping[str, Any]) -> None:
    command = str(action.get("catalog_command") or "")
    required_by_command = {
        "set_endpoint": "rpc_endpoint",
        "set_method": "rpc_method",
        "append_evidence": "rpc_schema_evidence",
    }
    required = required_by_command.get(command)
    if required and action.get(required) in (None, ""):
        raise ValueError(f"rpc_catalog_command {command} requires {required}")
    if command == "set_method" and not looks_like_wire_method_identity(action.get("rpc_method")):
        raise ValueError(
            "rpc_catalog_command set_method requires an exact wire method token "
            "or HTTP_VERB /path identity"
        )
    payload_fields = {
        key
        for key in ("rpc_endpoint", "rpc_method", "rpc_schema_evidence")
        if key in action and action.get(key) not in (None, "")
    }
    allowed_payload = {required} if required else set()
    unexpected = sorted(payload_fields - allowed_payload)
    if unexpected:
        raise ValueError(
            f"rpc_catalog_command {command} does not accept: {', '.join(unexpected)}"
        )


def _validate_rpc_workload_command(action: Mapping[str, Any]) -> None:
    scope = str(action.get("workload_scope") or "")
    weights = action.get("rpc_weights")
    if scope == "single_replace" and weights:
        raise ValueError("single_replace does not accept rpc_weights")
    if weights is not None and sum(int(value) for value in weights.values()) != 100:
        raise ValueError("rpc_weights must total 100")


def _validate_sync_observe_options(action: Mapping[str, Any]) -> None:
    stop = action.get("sync_observe_stop_condition")
    duration = action.get("sync_observe_duration_seconds")
    if stop is None and duration is None:
        raise ValueError(
            "set_sync_observe_options requires sync_observe_stop_condition "
            "or sync_observe_duration_seconds"
        )
    if stop == "duration" and duration is None:
        raise ValueError("duration stop condition requires sync_observe_duration_seconds")
    if stop not in (None, "duration") and duration is not None:
        raise ValueError("sync_observe_duration_seconds is valid only for duration stop condition")


def _validate_group_navigation(action: Mapping[str, Any]) -> None:
    from agent.workflows.group_registry import is_user_navigable_group

    group = str(action.get("group") or "").strip()
    if not is_user_navigable_group(group):
        raise ValueError(f"change_group requires a user-navigable workflow group: {group or '<missing>'}")


def _validate_evidence_collection_append(action: Mapping[str, Any]) -> None:
    evidence = str(action.get("evidence") or "")
    source = str(action.get("source_evidence") or "")
    if not evidence.strip() or not source.strip() or evidence not in source:
        raise ValueError("append_evidence_collection requires exact current-turn evidence")


@dataclass(frozen=True)
class ActionSpec:
    action_type: str
    owner: str
    purpose: str
    allowed_arguments: tuple[str, ...] = ()
    execution_phase: int = 50
    target_group: str = ""
    merge_identity: tuple[str, ...] = ()
    preserve_pending: bool = False
    mutation_dimension: str = ""
    allows_followup_actions: bool = False
    requires_capabilities: tuple[str, ...] = ()
    provides_capabilities: tuple[str, ...] = ()
    merge_mapping_fields: tuple[str, ...] = ()
    merge_sequence_fields: tuple[str, ...] = ()
    lifetime: ActionLifetime = "durable"
    effect: ActionEffect = "configuration_mutation"
    turn_local_result_roots: tuple[str, ...] = ()
    crosses_pending_barrier: bool = False
    requires_specific_change: bool = False
    incomplete_mutation_intake: bool = False
    required_arguments: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    suppressed_by: tuple[str, ...] = ()
    semantic_recovery_source_argument: str = ""
    semantic_support_relations: tuple[str, ...] = ()
    pending_option_semantic: str = ""
    option_navigation_groups: tuple[str, ...] = ()
    pending_option_admission: bool = True
    incompatible_target_modes: tuple[str, ...] = ()
    validator: ActionValidator | None = None

    @property
    def arguments(self) -> tuple[str, ...]:
        """Compatibility alias for coverage/report consumers."""

        return self.allowed_arguments


ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(
        "greeting",
        "orientation",
        "Greet without changing workflow state.",
        ("source_evidence",),
        execution_phase=5,
        lifetime="turn_local",
        effect="read_only",
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "set_response_language",
        "orientation",
        "Persist an explicit user preference for Chinese or English responses without changing benchmark configuration.",
        ("language", "source_evidence"),
        execution_phase=2,
        lifetime="turn_local",
        effect="workflow_state_mutation",
        required_arguments=("language", "source_evidence"),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "clarify_unresolved",
        "orientation",
        "Ask the user to clarify only the clauses that could not be represented safely; do not commit a partial plan.",
        ("clauses",),
        execution_phase=1,
        lifetime="turn_local",
        effect="read_only",
        required_arguments=("clauses",),
    ),
    ActionSpec("reset_session", "orientation", "Clear workflow configuration while preserving the workflow job receipt; startup discovery and historical jobs remain external read models.", execution_phase=0, effect="workflow_state_mutation"),
    ActionSpec("ask_capabilities", "orientation", "Explain supported product capabilities from framework facts.", execution_phase=5, lifetime="turn_local", effect="read_only"),
    ActionSpec(
        "answer_opening_question",
        "orientation",
        "Answer a product, state, preparation, status, or explanation question.",
        ("topic", "subject", "source_evidence"),
        5,
        merge_identity=("topic", "subject"),
        allows_followup_actions=True,
        lifetime="turn_local",
        effect="read_only",
        required_arguments=("topic",),
        pending_option_admission=False,
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec("choose_target_mode", "chain_rpc", "Select fake-node, real-node, or sync-observe only when the user explicitly requests that mutation.", ("target_mode", "target_mode_explicit", "source_evidence"), 10, "target_mode", mutation_dimension="target_mode", provides_capabilities=("target_mode",), required_arguments=("target_mode", "target_mode_explicit", "source_evidence"), semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS),
    ActionSpec("choose_chain", "chain_rpc", "Select a chain when none is confirmed, or expose multiple candidates without choosing silently.", ("chain_text", "chain_candidates", "source_evidence", "chain_exists", "canonical_chain_name", "adapter_family", "possible_known_chain", "evidence_summary"), 20, "chain_identity", mutation_dimension="chain", provides_capabilities=("chain_identity",), required_arguments=("source_evidence",), semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS, validator=_validate_chain_selection),
    ActionSpec("change_chain", "chain_rpc", "Request a different chain when one is confirmed, or expose multiple candidates without choosing silently.", ("chain_text", "chain_candidates", "source_evidence", "chain_exists", "canonical_chain_name", "adapter_family", "possible_known_chain", "evidence_summary"), 20, "chain_identity", mutation_dimension="chain", provides_capabilities=("chain_identity",), required_arguments=("source_evidence",), semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS, validator=_validate_chain_selection),
    ActionSpec(
        "request_chain_selection",
        "chain_rpc",
        "Open typed chain-selection intake when the user requests a chain selection or replacement but supplies no concrete chain.",
        ("chain_candidates", "source_evidence"),
        execution_phase=25,
        target_group="chain_identity",
        requires_capabilities=("target_mode",),
        provides_capabilities=("chain_identity",),
        incomplete_mutation_intake=True,
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "request_target_mode_selection",
        "chain_rpc",
        "Open typed target-mode intake when it is unresolved, including initial selection or replacement requests that supply no concrete target mode.",
        ("source_evidence",),
        execution_phase=24,
        target_group="target_mode",
        provides_capabilities=("target_mode",),
        incomplete_mutation_intake=True,
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "change_group",
        "coordinator",
        "Temporarily route to a named workflow group only for an explicit navigation request.",
        ("group", "navigation_explicit", "source_evidence"),
        50,
        merge_identity=("group",),
        crosses_pending_barrier=True,
        required_arguments=("group", "navigation_explicit", "source_evidence"),
        effect="workflow_navigation",
        semantic_support_relations=(
            *FRAMED_OPERATION_SUPPORT_RELATIONS,
            "non_mutation_scope",
        ),
        validator=_validate_group_navigation,
    ),
    ActionSpec(
        "resume_current_flow",
        "coordinator",
        "Resume the active typed configuration question when the user asks to return to the current benchmark/configuration flow without naming a different configuration area.",
        ("source_evidence",),
        execution_phase=5,
        lifetime="turn_local",
        effect="workflow_navigation",
        required_arguments=("source_evidence",),
        suppressed_by=("change_group", "go_back"),
        pending_option_semantic="continue_current_flow",
        semantic_support_relations=(
            *FRAMED_OPERATION_SUPPORT_RELATIONS,
            "non_mutation_scope",
        ),
    ),
    ActionSpec(
        "go_back",
        "coordinator",
        "Return to the most recent relevant interrupted group.",
        ("source_evidence",),
        effect="workflow_navigation",
        semantic_recovery_source_argument="source_evidence",
        semantic_support_relations=(
            *FRAMED_OPERATION_SUPPORT_RELATIONS,
            "non_mutation_scope",
        ),
    ),
    ActionSpec(
        "queue_workflow_goal",
        "coordinator",
        "Persist a later mutually exclusive benchmark/observation goal without applying it to the current configuration.",
        ("target_mode", "goal", "source_evidence"),
        70,
        merge_identity=("target_mode", "goal"),
        effect="workflow_state_mutation",
        required_arguments=("target_mode", "source_evidence"),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "activate_next_workflow_goal",
        "coordinator",
        "Activate the oldest explicitly queued workflow goal when the user explicitly asks to start, continue, or advance to that saved/deferred follow-up. Do not use request_target_mode_selection or choose_target_mode for a referenced queued goal.",
        ("source_evidence",),
        execution_phase=10,
        effect="workflow_state_mutation",
        required_arguments=("source_evidence",),
        constraints=("workflow_goals must contain at least one saved goal",),
        semantic_recovery_source_argument="source_evidence",
        semantic_support_relations=(
            *FRAMED_OPERATION_SUPPORT_RELATIONS,
            "non_mutation_scope",
        ),
    ),
    ActionSpec(
        "discard_next_workflow_goal",
        "coordinator",
        "Discard only the oldest explicitly queued workflow goal when the user explicitly cancels or removes that saved/deferred follow-up.",
        ("source_evidence",),
        execution_phase=10,
        effect="workflow_state_mutation",
        required_arguments=("source_evidence",),
        constraints=("workflow_goals must contain at least one saved goal",),
        semantic_recovery_source_argument="source_evidence",
        semantic_support_relations=(
            *FRAMED_OPERATION_SUPPORT_RELATIONS,
            "non_mutation_scope",
        ),
    ),
    ActionSpec(
        "set_rpc_mode",
        "chain_rpc",
        "Select the RPC workload cardinality: one method (single) or multiple weighted methods (mixed).",
        ("rpc_mode", "mutation_explicit", "source_evidence"),
        30,
        "workload_rpc",
        mutation_dimension="rpc_mode",
        requires_capabilities=("target_mode", "chain_identity"),
        required_arguments=("rpc_mode", "mutation_explicit", "source_evidence"),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec("choose_adapter_family", "chain_rpc", "Confirm the adapter family for an already identified unknown chain.", ("adapter_family",), 22, "chain_identity", required_arguments=("adapter_family",)),
    ActionSpec(
        "rpc_catalog_command",
        "chain_rpc",
        "Apply exactly one catalog transition: enter, set_endpoint, set_method, or append_evidence.",
        ("catalog_command", "rpc_method", "rpc_endpoint", "rpc_schema_evidence", "source_evidence"),
        40,
        "endpoint_process",
        preserve_pending=True,
        requires_capabilities=("chain_identity",),
        required_arguments=("catalog_command",),
        constraints=(
            "set_endpoint requires only rpc_endpoint; set_method requires only rpc_method; append_evidence requires only rpc_schema_evidence; enter accepts no payload",
        ),
        incompatible_target_modes=("sync-observe",),
        semantic_support_relations=EVIDENCE_OPERATION_SUPPORT_RELATIONS,
        validator=_validate_rpc_catalog_command,
    ),
    ActionSpec(
        "secondary_handoff_command",
        "chain_rpc",
        "Append one source-grounded protocol-development evidence item to an active Case 3 handoff.",
        ("handoff_command", "handoff_evidence", "source_evidence"),
        40,
        "chain_identity",
        preserve_pending=True,
        required_arguments=("handoff_command", "handoff_evidence"),
        constraints=(
            "append_evidence is valid only while a Case 3 handoff is collecting official protocol, endpoint, request, or response evidence",
        ),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "rpc_workload_command",
        "chain_rpc",
        "Apply the workload selection only after catalog methods are validated.",
        ("workload_scope", "rpc_weights", "finish_methods"),
        41,
        "endpoint_process",
        preserve_pending=True,
        requires_capabilities=("target_mode", "chain_identity"),
        required_arguments=("workload_scope",),
        constraints=(
            "workload_scope is single_replace, mixed_replace, or mixed_add; supplied rpc_weights must total 100; single_replace accepts no weights",
        ),
        validator=_validate_rpc_workload_command,
    ),
    ActionSpec("use_default_workload", "chain_rpc", "Accept the active chain template workload.", execution_phase=40, requires_capabilities=("target_mode", "chain_identity")),
    ActionSpec("configure_workload_weights", "chain_rpc", "Configure mixed weights for the active template or custom methods.", execution_phase=40, requires_capabilities=("target_mode", "chain_identity")),
    ActionSpec(
        "request_target_change",
        "chain_rpc",
        "Open a typed choice for changing the chain or target mode.",
        option_navigation_groups=("chain_identity", "target_mode"),
    ),
    ActionSpec("cancel_target_change", "chain_rpc", "Return to workload configuration without changing chain or target mode."),
    ActionSpec("set_qps_mode", "performance", "Select quick, standard, or intensive profile. Do not use set_qps_override unless concrete numeric values were supplied.", ("qps_mode", "mutation_explicit", "source_evidence"), 50, "qps_profile", preserve_pending=True, mutation_dimension="qps_profile", requires_capabilities=("target_mode",), crosses_pending_barrier=True, required_arguments=("qps_mode", "mutation_explicit", "source_evidence"), semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS),
    ActionSpec("request_qps_customization", "performance", "Enter QPS customization when the user wants to adjust the selected profile but has not supplied every numeric value yet.", ("qps_fields", "source_evidence"), 50, "qps_profile", preserve_pending=True, mutation_dimension="qps_profile", requires_capabilities=("target_mode",), crosses_pending_barrier=True, requires_specific_change=True, required_arguments=("source_evidence",), semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS),
    ActionSpec("set_qps_override", "performance", "Apply concrete QPS profile overrides only when qps_overrides contains user-supplied numeric values. For an adjustment request without values, use request_qps_customization.", ("qps_overrides",), 50, "qps_profile", preserve_pending=True, mutation_dimension="qps_profile", requires_capabilities=("target_mode",), crosses_pending_barrier=True, required_arguments=("qps_overrides",)),
    ActionSpec("set_observability", "performance", "Select disabled, local, or exporter-only observability.", ("observability_mode", "mutation_explicit", "source_evidence"), 60, "observability", preserve_pending=True, mutation_dimension="observability", requires_capabilities=("target_mode",), crosses_pending_barrier=True, required_arguments=("observability_mode", "mutation_explicit", "source_evidence"), semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS),
    ActionSpec("set_sync_observe_source", "sync_observe", "Select the real sync-observe data source.", ("sync_observe_source",), target_group="sync_observe", required_arguments=("sync_observe_source",)),
    ActionSpec("clear_sync_observe_source", "sync_observe", "Clear a sync-observe source after its endpoint setup is cancelled.", target_group="sync_observe"),
    ActionSpec(
        "set_sync_observe_options",
        "sync_observe",
        "Apply user-reviewed sync-observe stopping options.",
        ("sync_observe_stop_condition", "sync_observe_duration_seconds"),
        target_group="sync_observe",
        constraints=(
            "duration requires sync_observe_duration_seconds; other stop conditions reject duration_seconds",
        ),
        validator=_validate_sync_observe_options,
    ),
    ActionSpec("approve_preflight_smoke", "execution", "Approve one idempotent preflight/smoke submission.", effect="execution"),
    ActionSpec("reject_preflight_smoke", "execution", "Pause before preflight/smoke without submitting a job."),
    ActionSpec("approve_final_benchmark", "execution", "Approve final real-node benchmark submission after isolated smoke success.", effect="execution"),
    ActionSpec("reject_final_benchmark", "execution", "Pause after successful real-node smoke without submitting the final benchmark."),
    ActionSpec("set_accounts_presence", "environment", "Confirm whether a separate accounts/state disk exists.", ("has_accounts_device",), target_group="accounts_disk", required_arguments=("has_accounts_device",)),
    ActionSpec(
        "propose_config_values",
        "environment",
        "Propose values extracted from natural language, YAML, JSON, env, or tables; after user review, normal fallback asks for required values that the partial input did not supply.",
        ("config_values", "unmapped_values", "conflicts", "source_format", "source_evidence"),
        25,
        preserve_pending=True,
        merge_mapping_fields=("config_values", "unmapped_values"),
        merge_sequence_fields=("conflicts",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "append_evidence_collection",
        "analysis",
        "Append the current user-supplied evidence fragment to an active evidence collection; never use this for questions, navigation, corrections, or unrelated conversation.",
        ("evidence", "source_evidence"),
        execution_phase=4,
        crosses_pending_barrier=True,
        required_arguments=("evidence", "source_evidence"),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
        validator=_validate_evidence_collection_append,
    ),
    ActionSpec(
        "finish_evidence_collection",
        "analysis",
        "Finish the active evidence collection only when the user explicitly says the evidence block is complete.",
        ("source_evidence",),
        execution_phase=4,
        crosses_pending_barrier=True,
        required_arguments=("source_evidence",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "pause_evidence_collection",
        "analysis",
        "Pause and preserve the active evidence collection only when the user explicitly requests a temporary detour.",
        ("source_evidence",),
        execution_phase=3,
        crosses_pending_barrier=True,
        required_arguments=("source_evidence",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "resume_evidence_collection",
        "analysis",
        "Resume a paused evidence collection only when the user explicitly asks to continue that collection.",
        ("source_evidence",),
        execution_phase=3,
        crosses_pending_barrier=True,
        required_arguments=("source_evidence",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "cancel_evidence_collection",
        "analysis",
        "Discard the active evidence collection only when the user explicitly cancels it.",
        ("source_evidence",),
        execution_phase=4,
        crosses_pending_barrier=True,
        required_arguments=("source_evidence",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "analyze_evidence",
        "analysis",
        "Analyze pasted or previously collected logs/errors/evidence without becoming a deferred workflow command.",
        ("evidence", "question"),
        execution_phase=5,
        lifetime="turn_local",
        effect="read_only",
        turn_local_result_roots=("evidence_buffer",),
        crosses_pending_barrier=True,
        semantic_recovery_source_argument="evidence",
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "analyze_report",
        "analysis",
        "Analyze a job, report, metrics, logs, or artifacts.",
        ("job_id", "subject"),
        execution_phase=5,
        allows_followup_actions=True,
        lifetime="turn_local",
        effect="read_only",
        turn_local_result_roots=("report_context",),
    ),
    ActionSpec("correct_failure", "recovery", "Reopen only the policy-declared affected configuration group."),
    ActionSpec(
        "inspect_failure",
        "recovery",
        "Inspect structured failure evidence without changing workflow configuration.",
        effect="read_only",
    ),
    ActionSpec("retry_failure", "recovery", "Clear a retryable external-service failure without executing a benchmark side effect."),
    ActionSpec("cancel_failure_recovery", "recovery", "Pause recovery while preserving evidence and confirmed configuration."),
    ActionSpec("answer_pending", "coordinator", "Answer the active typed question after interpreting non-exact user language.", ("answer", "selected_value", "source_evidence"), 0, required_arguments=("source_evidence",), semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS),
    ActionSpec("unknown", "orientation", "Report that no safe action could be resolved.", ("reason",), effect="read_only", required_arguments=("reason",)),
)


CONSULTATION_TOPICS: tuple[str, ...] = (
    "identity",
    "capabilities",
    "supported_chains",
    "extension",
    "config_explanation",
    "current_config",
    "workload_config",
    "current_context",
    "next_action",
    "startup_discovery",
    "environment_readiness",
    "requirements",
    "workflow",
    "mode_comparison",
    "performance_benchmark_guidance",
    "execution_preflight_smoke",
    "recommendation",
    "reset_help",
    "evidence_help",
    "correction",
    "current_job",
    "job_status",
    "execution_status",
)

CONSULTATION_TOPIC_ALIASES: Mapping[str, str] = {
    "who": "identity",
    "who_are_you": "identity",
    "origin": "identity",
    "purpose": "identity",
    "what_can_you_do": "capabilities",
    "agent_capabilities": "capabilities",
    "agent_capability": "capabilities",
    "supported": "supported_chains",
    "chains": "supported_chains",
    "state": "current_config",
    "settings": "current_config",
    "pending": "current_context",
    "current_prompt": "current_context",
    "prepare": "requirements",
    "prerequisites": "requirements",
    "modes": "mode_comparison",
    "mode": "mode_comparison",
    "restart": "reset_help",
    "start_over": "reset_help",
    "env_readiness": "environment_readiness",
    "recommend_start": "recommendation",
}

STATE_AUDIT_TOPICS = frozenset({"current_config", "current_context", "next_action", "execution_status"})

ACTION_BY_TYPE = {spec.action_type: spec for spec in ACTION_SPECS}

for _spec in ACTION_SPECS:
    _unknown_support_relations = (
        set(_spec.semantic_support_relations) - SEMANTIC_SUPPORT_RELATIONS
    )
    if _unknown_support_relations:
        raise ValueError(
            f"{_spec.action_type} declares unknown semantic support relations: "
            + ", ".join(sorted(_unknown_support_relations))
        )
def lifecycle_rejected_action_indexes(
    state: Mapping[str, Any],
    actions: list[dict[str, Any]],
) -> tuple[int, ...]:
    """Return actions incompatible with the ordered target-mode transaction.

    Target-mode changes take effect in action order. This lets one explicit
    transaction leave a workflow and then use its newly compatible actions,
    while preventing a domain action from mutating state owned by the current
    workflow before that transition occurs.
    """

    target_mode = normalize_target_mode(state.get("target_mode"))
    rejected: list[int] = []
    for index, action in enumerate(actions):
        action_type = str(action.get("type") or "")
        if action_type == "choose_target_mode":
            replacement = normalize_target_mode(action.get("target_mode"))
            if replacement:
                target_mode = replacement
            continue
        spec = ACTION_BY_TYPE.get(action_type)
        if spec is not None and target_mode in spec.incompatible_target_modes:
            rejected.append(index)
    return tuple(rejected)


def normalize_action_relations(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply registry-declared relations across one proposed transaction.

    Relations operate on typed actions only. They never inspect user prose or
    infer a replacement action. This keeps transaction semantics centralized
    while allowing an explicit navigation to own its existing interruption
    frame instead of competing with a same-turn generic resume action.
    """

    action_types = {
        str(action.get("type") or "").strip()
        for action in actions
        if isinstance(action, dict)
    }
    interruption_owner_present = any(
        action_preserves_pending(action)
        for action in actions
        if isinstance(action, dict)
    )
    output: list[dict[str, Any]] = []
    merge_indexes: dict[tuple[Any, ...], int] = {}
    for action in actions:
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if str(action.get("type") or "") == "resume_current_flow" and interruption_owner_present:
            continue
        if spec is not None and set(spec.suppressed_by).intersection(action_types):
            continue
        if spec is not None and spec.merge_identity:
            merge_key = action_merge_key(action)
            prior_index = merge_indexes.get(merge_key)
            if prior_index is not None:
                prior = output[prior_index]
                for field_name in TRUSTED_ACTION_METADATA_FIELDS:
                    if action.get(field_name) is True:
                        prior[field_name] = True
                if action.get("navigation_explicit") is True:
                    prior["navigation_explicit"] = True
                continue
            merge_indexes[merge_key] = len(output)
        output.append(action)
    return output

ACTION_METADATA_FIELDS = frozenset({"type", "confidence", "reason", "action_id"})
TRUSTED_ACTION_METADATA_FIELDS = frozenset({
    "selection_contract_verified",
    "semantic_purpose_verified",
    "pending_option_semantic_verified",
    "chain_selection_semantic_verified",
    "target_mode_semantic_verified",
    "group_navigation_semantic_verified",
    "_origin_text",
    "_queue_origin_group",
    "_submitted_turn_index",
    "_plan_scope",
    "_plan_index",
    "_merged_origin_texts",
})
LEGACY_ARGUMENTS_ENVELOPE_VERSION = "arguments.v1"
_COMPATIBILITY_USAGE: Counter[str] = Counter()

ACTION_ARGUMENT_SCHEMAS: dict[str, Mapping[str, Any]] = {
    "adapter_family": {"type": "string", "minLength": 1},
    "answer": {},
    "canonical_chain_name": {"type": "string", "minLength": 1},
    "catalog_command": {"type": "string", "enum": ["enter", "set_endpoint", "set_method", "append_evidence"]},
    "chain_candidates": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
    "chain_exists": {"type": "boolean"},
    "chain_text": {"type": "string", "minLength": 1},
    "clauses": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
    "config_values": {"type": "object"},
    "conflicts": {"type": "array"},
    "evidence": {},
    "evidence_summary": {"type": "string", "minLength": 1},
    "finish_methods": {"type": "boolean"},
    "goal": {"type": "string", "minLength": 1},
    "group": {"type": "string", "minLength": 1},
    "handoff_command": {"type": "string", "enum": ["append_evidence"]},
    "handoff_evidence": {"type": "string", "minLength": 1},
    "has_accounts_device": {"type": "boolean"},
    "job_id": {"type": "string", "minLength": 1},
    "language": {"type": "string", "enum": ["zh", "en"]},
    "mutation_explicit": {"type": "boolean"},
    "navigation_explicit": {"type": "boolean"},
    "observability_mode": {"type": "string", "enum": ["disabled", "local", "exporter"]},
    "possible_known_chain": {"type": "string", "minLength": 1},
    "question": {"type": "string", "minLength": 1},
    "qps_fields": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
    "qps_mode": {"type": "string", "enum": ["quick", "standard", "intensive"]},
    "qps_overrides": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 1}, "minProperties": 1},
    "reason": {"type": "string", "minLength": 1},
    "rpc_endpoint": {"type": "string", "minLength": 1},
    "rpc_method": {"type": "string", "minLength": 1},
    "rpc_mode": {"type": "string", "enum": ["single", "mixed"]},
    "rpc_schema_evidence": {},
    "rpc_weights": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 1}, "minProperties": 1},
    "selected_value": {},
    "source_evidence": {"type": "string", "minLength": 1},
    "source_format": {"type": "string", "enum": ["yaml", "json", "env", "mixed"]},
    "subject": {"type": "string"},
    "sync_observe_source": {"type": "string", "enum": ["existing_local_node", "endpoint_only", "client_setup"]},
    "sync_observe_stop_condition": {"type": "string", "enum": ["until_stopped", "duration", "until_synced"]},
    "sync_observe_duration_seconds": {"type": "integer", "minimum": 1},
    "target_mode": {"type": "string", "enum": ["fake-node", "real-node", "sync-observe"]},
    "target_mode_explicit": {"type": "boolean"},
    "topic": {"type": "string", "minLength": 1},
    "unmapped_values": {"type": "object"},
    "workload_scope": {"type": "string", "enum": ["single_replace", "mixed_replace", "mixed_add"]},
}


def compile_legacy_custom_rpc_action(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compile the retired omnibus action into ordered catalog/workload commands."""

    action = dict(raw)
    if str(action.get("type") or "") != "start_custom_rpc":
        return [action]
    # Expanded commands receive fresh action IDs later. Carrying the retired
    # action identity here would either restore its old type or create several
    # commands with the same durable ID.
    metadata = {
        key: value
        for key, value in action.items()
        if key in {"confidence", "reason"}
    }
    commands: list[dict[str, Any]] = []
    if action.get("rpc_endpoint"):
        commands.append({"type": "rpc_catalog_command", "catalog_command": "set_endpoint", "rpc_endpoint": action["rpc_endpoint"], **metadata})
    if action.get("rpc_method"):
        commands.append({"type": "rpc_catalog_command", "catalog_command": "set_method", "rpc_method": action["rpc_method"], **metadata})
    if action.get("rpc_schema_evidence"):
        commands.append({"type": "rpc_catalog_command", "catalog_command": "append_evidence", "rpc_schema_evidence": action["rpc_schema_evidence"], **metadata})
    if action.get("workload_scope"):
        workload = {
            "type": "rpc_workload_command",
            "workload_scope": action["workload_scope"],
            **metadata,
        }
        if isinstance(action.get("rpc_weights"), dict):
            workload["rpc_weights"] = dict(action["rpc_weights"])
        if "finish_methods" in action:
            workload["finish_methods"] = bool(action.get("finish_methods"))
        commands.append(workload)
    if not commands:
        commands.append({"type": "rpc_catalog_command", "catalog_command": "enter", **metadata})
    _COMPATIBILITY_USAGE["start_custom_rpc.v1"] += 1
    return commands


def normalize_action_envelope(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a model action into the registry's flat wire contract.

    Nested ``arguments`` is the measured ``arguments.v1`` compatibility path.
    The current wire contract is flat. Explicit top-level values win and the
    compatibility object cannot overwrite action identity or audit metadata.
    """

    action = dict(raw)
    nested = action.pop("arguments", None)
    if isinstance(nested, dict):
        _COMPATIBILITY_USAGE[LEGACY_ARGUMENTS_ENVELOPE_VERSION] += 1
        for key, value in nested.items():
            if key not in {"type", "intent", "action_id", "confidence", "reason"}:
                action.setdefault(str(key), value)
    if "type" not in action and "intent" in action:
        action["type"] = action["intent"]
    action.pop("intent", None)
    return action


def compatibility_usage() -> dict[str, int]:
    """Return a snapshot of legacy action-envelope usage."""

    return dict(_COMPATIBILITY_USAGE)


def validate_action_contract(
    raw: dict[str, Any],
    *,
    trusted_metadata: bool = False,
) -> dict[str, Any]:
    """Normalize and validate one model action against its ActionSpec."""

    action = normalize_action_envelope(raw)
    action_type = str(action.get("type") or "").strip()
    spec = ACTION_BY_TYPE.get(action_type)
    if spec is None:
        raise ValueError(f"undeclared action type: {action_type or '<missing>'}")
    _normalize_action_values(action)
    for key in spec.allowed_arguments:
        if key not in spec.required_arguments and action.get(key) in (None, ""):
            action.pop(key, None)
    allowed = ACTION_METADATA_FIELDS | frozenset(spec.allowed_arguments)
    if trusted_metadata:
        allowed |= TRUSTED_ACTION_METADATA_FIELDS
    undeclared = sorted(str(key) for key in action if key not in allowed)
    if undeclared:
        raise ValueError(f"undeclared arguments for {action_type}: {', '.join(undeclared)}")
    missing = [
        key
        for key in spec.required_arguments
        if key not in action or action[key] is None or (isinstance(action[key], str) and not action[key].strip())
    ]
    if missing:
        raise ValueError(f"missing required arguments for {action_type}: {', '.join(missing)}")
    for key in spec.allowed_arguments:
        if key in action:
            _validate_argument_value(action[key], ACTION_ARGUMENT_SCHEMAS[key], f"{action_type}.{key}")
    if action_type == "answer_opening_question" and action.get("topic") not in CONSULTATION_TOPICS:
        raise ValueError(f"invalid consultation topic: {action.get('topic')!r}")
    if action_type == "answer_pending":
        answer = action.get("answer")
        selected = action.get("selected_value")
        if not (
            (not isinstance(answer, str) and answer is not None)
            or (isinstance(answer, str) and answer.strip())
            or (not isinstance(selected, str) and selected is not None)
            or (isinstance(selected, str) and selected.strip())
        ):
            raise ValueError("answer_pending requires answer or selected_value")
    if spec.validator is not None:
        spec.validator(action)
    return action


def _normalize_action_values(action: dict[str, Any]) -> None:
    """Canonicalize registry-owned enums before their schemas are applied."""

    if "target_mode" in action:
        normalized = normalize_target_mode(action.get("target_mode"))
        if normalized:
            action["target_mode"] = normalized
    if "observability_mode" in action:
        normalized = normalize_observability_mode(action.get("observability_mode"))
        if normalized:
            action["observability_mode"] = normalized
    if str(action.get("type") or "") == "answer_opening_question" and "topic" in action:
        topic = str(action.get("topic") or "").strip().lower()
        action["topic"] = CONSULTATION_TOPIC_ALIASES.get(topic, topic)


def canonical_consultation_topic(value: Any) -> str:
    """Return the registry-owned canonical consultation topic."""

    topic = str(value or "capabilities").strip().lower()
    return CONSULTATION_TOPIC_ALIASES.get(topic, topic)


def _validate_argument_value(value: Any, schema: Mapping[str, Any], path: str) -> None:
    expected = schema.get("type")
    if expected == "string" and not isinstance(value, str):
        raise ValueError(f"invalid type for {path}: expected string")
    if expected == "boolean" and not isinstance(value, bool):
        raise ValueError(f"invalid type for {path}: expected boolean")
    if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"invalid type for {path}: expected integer")
    if expected == "array" and not isinstance(value, list):
        raise ValueError(f"invalid type for {path}: expected array")
    if expected == "object" and not isinstance(value, dict):
        raise ValueError(f"invalid type for {path}: expected object")
    if isinstance(value, str) and len(value) < int(schema.get("minLength") or 0):
        raise ValueError(f"invalid value for {path}: empty string")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"invalid value for {path}: {value!r}")
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems") or 0):
            raise ValueError(f"invalid value for {path}: too few items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_argument_value(item, item_schema, f"{path}[{index}]")
    if isinstance(value, dict):
        if len(value) < int(schema.get("minProperties") or 0):
            raise ValueError(f"invalid value for {path}: empty object")
        additional = schema.get("additionalProperties")
        if isinstance(additional, Mapping):
            for key, item in value.items():
                _validate_argument_value(item, additional, f"{path}.{key}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"invalid value for {path}: below minimum")


def action_execution_phase(action: dict[str, Any]) -> int:
    """Return registry-owned turn ordering for one normalized action."""

    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    if spec is None:
        return 999
    if spec.action_type == "answer_opening_question" and str(action.get("topic") or "").strip().lower() in STATE_AUDIT_TOPICS:
        return 45
    return spec.execution_phase


def action_lifetime(action: dict[str, Any]) -> ActionLifetime:
    """Return the registry-owned persistence lifetime for one action."""

    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    return spec.lifetime if spec is not None else "durable"


def action_is_turn_local(action: dict[str, Any]) -> bool:
    return action_lifetime(action) == "turn_local"


def action_effect(action: Mapping[str, Any]) -> ActionEffect:
    """Return the registry-owned business effect independently of lifetime."""

    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    return spec.effect if spec is not None else "configuration_mutation"


def semantic_scope_schema() -> list[dict[str, Any]]:
    """Expose the single semantic-scope contract to planners and validators."""

    return [
        {
            "name": name,
            "description": str(policy["description"]),
            "allowed_effects": list(policy.get("allowed_effects") or ()),
            "forbidden_effects": list(policy.get("forbidden_effects") or ()),
        }
        for name, policy in SEMANTIC_SCOPE_POLICIES.items()
    ]


def semantic_scope_accepts_action(scope: str, action: Mapping[str, Any]) -> bool:
    """Validate a semantic scope against registry effect metadata."""

    policy = SEMANTIC_SCOPE_POLICIES.get(str(scope or ""))
    if policy is None:
        return False
    effect = action_effect(action)
    allowed = tuple(policy.get("allowed_effects") or ())
    forbidden = tuple(policy.get("forbidden_effects") or ())
    return (not allowed or effect in allowed) and effect not in forbidden


def action_crosses_pending_barrier(action: dict[str, Any]) -> bool:
    """Return whether one action may execute while a typed question waits."""

    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    return bool(
        spec
        and (
            spec.lifetime == "turn_local"
            or spec.crosses_pending_barrier
        )
    )


def action_preserves_pending(action: Mapping[str, Any]) -> bool:
    """Return whether one action owns restoration of an interrupted question."""

    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    return bool(spec and spec.preserve_pending)


def action_merge_key(action: dict[str, Any]) -> tuple[Any, ...]:
    """Return the registry-declared semantic identity used across planner passes."""

    action_type = str(action.get("type") or "")
    semantic_type = "chain_selection" if action_type in {"choose_chain", "change_chain"} else action_type
    spec = ACTION_BY_TYPE.get(action_type)
    if not spec or not spec.merge_identity:
        return (semantic_type,)
    return (semantic_type,) + tuple(
        str(action.get(argument) or "").strip().casefold()
        for argument in spec.merge_identity
    )


def merge_semantic_actions(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Apply the action registry's durable merge contract.

    Incoming scalar decisions supersede older values. Registry-declared
    mapping fields merge by key and sequence fields retain unique evidence in
    encounter order. Actions without a merge declaration keep normal
    supersede semantics.
    """

    merged = dict(existing)
    merged.update(incoming)
    spec = ACTION_BY_TYPE.get(str(incoming.get("type") or ""))
    if spec is None:
        return merged
    for field_name in spec.merge_mapping_fields:
        values: dict[str, Any] = {}
        old_value = existing.get(field_name)
        new_value = incoming.get(field_name)
        if isinstance(old_value, dict):
            values.update(old_value)
        if isinstance(new_value, dict):
            values.update(new_value)
        merged[field_name] = values
    for field_name in spec.merge_sequence_fields:
        values: list[Any] = []
        for source in (existing.get(field_name), incoming.get(field_name)):
            if isinstance(source, list):
                items = source
            elif source is None or source == "":
                items = []
            else:
                items = [source]
            for item in items:
                if item not in values:
                    values.append(item)
        merged[field_name] = values
    if spec.merge_mapping_fields or spec.merge_sequence_fields:
        origins: list[str] = []
        for source in (existing, incoming):
            for value in source.get("_merged_origin_texts") or [source.get("_origin_text")]:
                text = str(value or "").strip()
                if text and text not in origins:
                    origins.append(text)
        merged["_merged_origin_texts"] = origins
    return merged


def validate_action_registry() -> None:
    names = [spec.action_type for spec in ACTION_SPECS]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate Harness action type")
    for spec in ACTION_SPECS:
        undeclared_required = set(spec.required_arguments) - set(spec.allowed_arguments)
        if undeclared_required:
            raise RuntimeError(
                f"required arguments must be allowed for {spec.action_type}: "
                f"{', '.join(sorted(undeclared_required))}"
            )
        if len(spec.allowed_arguments) != len(set(spec.allowed_arguments)):
            raise RuntimeError(f"duplicate allowed action argument: {spec.action_type}")
        missing_schemas = set(spec.allowed_arguments) - set(ACTION_ARGUMENT_SCHEMAS)
        if missing_schemas:
            raise RuntimeError(
                f"action arguments require value schemas for {spec.action_type}: "
                f"{', '.join(sorted(missing_schemas))}"
            )
        if spec.lifetime == "turn_local" and (
            spec.mutation_dimension
            or spec.requires_capabilities
            or spec.provides_capabilities
            or spec.preserve_pending
        ):
            raise RuntimeError(f"turn-local action cannot own durable workflow state: {spec.action_type}")
        if spec.turn_local_result_roots and spec.lifetime != "turn_local":
            raise RuntimeError(
                f"turn-local result roots require turn-local lifetime: {spec.action_type}"
            )
        if spec.requires_specific_change and (
            not spec.target_group or "source_evidence" not in spec.required_arguments
        ):
            raise RuntimeError(
                "actions requiring a specific change need a target group and required "
                f"source evidence: {spec.action_type}"
            )
        if spec.incomplete_mutation_intake and (
            not spec.target_group or "source_evidence" not in spec.allowed_arguments
        ):
            raise RuntimeError(
                "incomplete mutation intake actions need a target group and source evidence: "
                f"{spec.action_type}"
            )
    intake_groups = [
        spec.target_group
        for spec in ACTION_SPECS
        if spec.incomplete_mutation_intake
    ]
    if len(intake_groups) != len(set(intake_groups)):
        raise RuntimeError("only one incomplete mutation intake action is allowed per group")


validate_action_registry()


def assign_action_ids(thread_id: str, user_text: str, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign Harness-owned stable ids so retries cannot apply an action twice.

    Provider output is untrusted planning data. In particular, an LLM-provided
    ``action_id`` cannot be allowed to collide with another action or a prior
    turn, so identity is always recomputed at this control-plane boundary.
    """

    output: list[dict[str, Any]] = []
    for index, raw in enumerate(actions):
        action = dict(raw)
        action.pop("action_id", None)
        payload = json.dumps(
            {"thread_id": thread_id, "user_text": user_text, "index": index, "action": action},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        action["action_id"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        output.append(action)
    return output
