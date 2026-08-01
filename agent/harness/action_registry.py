"""Authoritative model-action schema for the Harness."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

from agent.knowledge.chain_identity import canonical_chain_aliases, repo_chain_names
from agent.workflows.group_registry import GROUP_SPEC_BY_NAME, USER_NAVIGABLE_GROUPS

from .input_values import (
    extract_json_object_or_array,
    extract_rpc_wire_evidence_spans,
    extract_url_candidates,
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
StateTransitionResolver = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    tuple[tuple[str, ...], Any] | None,
]
STRUCTURED_INTAKE_VALUE_SEMANTICS = frozenset({
    "boolean_true",
    "direct_value",
})

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
SEMANTIC_OPERATIONS = frozenset({
    "pending_answer",
    "consultation",
    "navigation",
    "administrative",
    "domain_request",
    "evidence_analysis",
    "report_analysis",
    "context",
    "unresolved",
})
ROUTED_UNIVERSAL_SEMANTIC_OPERATIONS = frozenset({
    "pending_answer",
    "consultation",
    "navigation",
    "administrative",
    "evidence_analysis",
    "report_analysis",
})
SEMANTIC_OPERATION_PURPOSES: Mapping[str, str] = {
    "pending_answer": (
        "A present, user-authorized commitment or answer to the active pending "
        "question. Hypothetical, counterfactual, consequence, explanation, or "
        "capability questions are consultation, not pending answers."
    ),
    "consultation": (
        "A read-only request for an answer, explanation, comparison, status, "
        "guidance, or hypothetical consequences. It never authorizes mutation."
    ),
    "navigation": (
        "An explicit request to move, return, resume, or change workflow position "
        "without itself supplying a configuration value."
    ),
    "administrative": (
        "An explicit present authorization for an Agent lifecycle operation such "
        "as clearing, resetting, or managing a persisted session."
    ),
    "domain_request": (
        "A present request to inspect, collect, confirm, or mutate one registered "
        "benchmark workflow domain."
    ),
    "evidence_analysis": (
        "A request to ingest, diagnose, or explain logs, errors, traces, diagnostics, failures, or "
        "other evidence, including a request made before the evidence is pasted; "
        "or the evidence contribution itself."
    ),
    "report_analysis": (
        "A request to inspect or explain completed benchmark reports, artifacts, "
        "metrics, or job results."
    ),
    "context": (
        "Non-actionable framing or support that contains no independent present "
        "request, answer, authorization, navigation, or evidence contribution."
    ),
    "unresolved": (
        "A source demand whose operation or authoritative owner cannot be safely "
        "determined without clarification."
    ),
}
if set(SEMANTIC_OPERATION_PURPOSES) != set(SEMANTIC_OPERATIONS):
    raise RuntimeError(
        "semantic operation purposes must cover every registered operation exactly"
    )
SEMANTIC_VALUE_DOMAIN_POLICY: Mapping[str, Any] = {
    "schema_version": 2,
    "identifier_boundary": "ascii_alnum_underscore_hyphen",
    "specialized_input_exemption": "validated_candidate_shape",
    "declared_pending_options_own_values": True,
    "targetless_domain_owner": "action_type",
}
PENDING_BARRIER_POLICIES: Mapping[str, Mapping[str, Any]] = {
    "none": {
        "answer_ownership": "pending_owner_only",
        "registered_cross_group_values": "route_to_registered_owner",
        "new_question_same_turn": "question_contract",
        "independent_cross_group_request": "route_then_apply_queue_policy",
        "declared_entry_intake": True,
        "preserving_detour": True,
        "explicit_navigation": True,
    },
    "exclusive_owner": {
        "answer_ownership": "pending_owner_only",
        "registered_cross_group_values": "route_to_registered_owner",
        "new_question_same_turn": "block_durable_actions",
        "independent_cross_group_request": "route_then_apply_queue_policy",
        "declared_entry_intake": True,
        "preserving_detour": False,
        "explicit_navigation": True,
    },
    "explicit_detour_only": {
        "answer_ownership": "pending_owner_only",
        "registered_cross_group_values": "route_to_registered_owner",
        "new_question_same_turn": "explicit_detours_only",
        "independent_cross_group_request": "route_then_apply_queue_policy",
        "declared_entry_intake": False,
        "preserving_detour": False,
        "explicit_navigation": True,
    },
}
_SEMANTIC_VALUE_IDENTIFIER_CHARACTER = r"A-Za-z0-9_-"


def pending_barrier_semantics(
    pending_question: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the authoritative semantics for one pending queue barrier.

    A barrier owns pending-answer binding and queue scheduling. It never
    changes the semantic owner of an independent user request.
    """

    pending = dict(pending_question or {})
    queue_barrier = pending.get("queue_barrier") is True
    policy = str(pending.get("barrier_policy") or "").strip()
    if not policy:
        policy = "exclusive_owner" if queue_barrier else "none"
    if policy not in PENDING_BARRIER_POLICIES:
        raise ValueError(f"unknown pending barrier policy: {policy}")
    if not queue_barrier and policy != "none":
        raise ValueError(
            "pending barrier policy requires queue_barrier=true"
        )
    return {
        "schema_version": 1,
        "policy": policy,
        "queue_barrier": queue_barrier,
        "pending_group": str(pending.get("group") or ""),
        "pending_owner": str(pending.get("owner") or ""),
        **dict(PENDING_BARRIER_POLICIES[policy]),
    }


def answer_pending_representation_conflict(action: Mapping[str, Any]) -> bool:
    """Return whether one pending answer carries two different facts."""

    if str(action.get("type") or "") != "answer_pending":
        return False
    supplied: list[Any] = []
    for key in ("answer", "selected_value"):
        if key not in action:
            continue
        value = action.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        supplied.append(value.strip() if isinstance(value, str) else value)
    if len(supplied) < 2:
        return False
    identities = {
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        for value in supplied
    }
    return len(identities) > 1


def state_has_capability(state: Mapping[str, Any], capability: str) -> bool:
    """Return whether authoritative state already satisfies a registry capability."""

    if capability == "chain_identity":
        from .routing import group_readiness

        return group_readiness(dict(state), "chain_identity").ready
    if capability == "target_mode":
        return bool(normalize_target_mode(state.get("target_mode")))
    return False


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
    if command not in {
        "enter",
        "set_endpoint",
        "set_method",
        "append_evidence",
        "keep_current_method",
        "replace_current_method",
        "finish",
    }:
        raise ValueError(f"unknown rpc_catalog_command: {command or '<missing>'}")
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


def _evidence_append_transition(
    state: Mapping[str, Any],
    action: Mapping[str, Any],
) -> tuple[tuple[str, ...], Any] | None:
    """Project the same completion transition owned by the analysis domain."""

    from .domains.analysis import (
        bounded_evidence_input,
        evidence_collection_complete,
    )

    collection = state.get("evidence_collection")
    active = collection if isinstance(collection, Mapping) else {}
    lines = [str(item) for item in active.get("lines") or []]
    incoming, terminated = bounded_evidence_input(
        str(action.get("evidence") or "")
    )
    lines.extend(incoming)
    if terminated or evidence_collection_complete(lines):
        return ("evidence_collection", "status"), None
    return None


def _validate_config_field_intake(action: Mapping[str, Any]) -> None:
    from agent.workflows.group_registry import reconfiguration_question_for_field

    field = str(action.get("config_field") or "").strip()
    if not reconfiguration_question_for_field(field):
        raise ValueError(
            f"request_config_field_input requires a registered reconfigurable field: {field or '<missing>'}"
        )


def _content_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StructuredIntakeSpec:
    alias: str
    fixed_arguments: tuple[tuple[str, Any], ...]
    value_semantics: str
    value_argument: str = ""


@dataclass(frozen=True)
class ActionSpec:
    action_type: str
    owner: str
    purpose: str
    allowed_arguments: tuple[str, ...] = ()
    execution_phase: int = 50
    target_group: str = ""
    target_field_argument: str = ""
    compiler_groups: tuple[str, ...] = ()
    route_groups: tuple[str, ...] = ()
    semantic_operations: tuple[str, ...] = ("domain_request",)
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
    interrupts_pending: bool = False
    replaces_deferred_queue: bool = False
    requires_specific_change: bool = False
    incomplete_mutation_intake: bool = False
    incomplete_read_intake: bool = False
    required_arguments: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    suppressed_by: tuple[str, ...] = ()
    semantic_recovery_source_argument: str = ""
    semantic_support_relations: tuple[str, ...] = ()
    semantic_value_grounding_arguments: tuple[str, ...] = ()
    open_identity_grounding_arguments: tuple[str, ...] = ()
    semantic_value_representative: bool = False
    explicit_scope_authorization: bool = False
    exact_source_value_arguments: tuple[str, ...] = ()
    pending_option_semantic: str = ""
    option_navigation_groups: tuple[str, ...] = ()
    pending_option_admission: bool = True
    incompatible_target_modes: tuple[str, ...] = ()
    required_state_path: tuple[str, ...] = ()
    required_state_values: tuple[Any, ...] = ()
    state_transition_path: tuple[str, ...] = ()
    state_transition_value: Any = None
    state_transition_resolver: StateTransitionResolver | None = None
    entry_intake: bool = False
    entry_intake_purpose: str = ""
    entry_intake_fixed_arguments: tuple[tuple[str, Any], ...] = ()
    entry_intake_value_arguments: tuple[str, ...] = ()
    structured_intake: tuple[StructuredIntakeSpec, ...] = ()
    invalidates_groups: tuple[str, ...] = ()
    internal_only: bool = False
    typed_option_only: bool = False
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
        semantic_operations=("consultation",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "set_response_language",
        "orientation",
        "Override the response language for the current turn without changing benchmark configuration.",
        ("language", "source_evidence"),
        execution_phase=2,
        lifetime="turn_local",
        effect="workflow_state_mutation",
        turn_local_result_roots=("language",),
        semantic_operations=("administrative",),
        required_arguments=("language", "source_evidence"),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
        semantic_value_grounding_arguments=("language",),
    ),
    ActionSpec(
        "clarify_unresolved",
        "orientation",
        "Ask the user to clarify only the clauses that could not be represented safely; do not commit a partial plan.",
        ("clauses",),
        execution_phase=1,
        lifetime="turn_local",
        effect="read_only",
        semantic_operations=("unresolved",),
        required_arguments=("clauses",),
    ),
    ActionSpec(
        "request_session_reset",
        "orientation",
        "Request a signed confirmation before clearing workflow configuration; startup discovery and historical jobs remain external read models.",
        execution_phase=0,
        effect="workflow_navigation",
        preserve_pending=True,
        crosses_pending_barrier=True,
        interrupts_pending=True,
        replaces_deferred_queue=True,
        semantic_operations=("administrative",),
        route_groups=("opening",),
        explicit_scope_authorization=True,
    ),
    ActionSpec(
        "reset_session",
        "orientation",
        "Clear workflow configuration after a typed confirmation contract has authorized the destructive transition.",
        execution_phase=0,
        effect="workflow_state_mutation",
        crosses_pending_barrier=True,
        interrupts_pending=True,
        replaces_deferred_queue=True,
        internal_only=True,
        typed_option_only=True,
    ),
    ActionSpec(
        "prepare_session_entry",
        "orientation",
        "Open the typed fresh-session or resumable-session question at terminal startup.",
        execution_phase=0,
        effect="workflow_navigation",
        crosses_pending_barrier=True,
        semantic_operations=("administrative",),
        internal_only=True,
    ),
    ActionSpec("ask_capabilities", "orientation", "Explain supported product capabilities from framework facts.", execution_phase=5, lifetime="turn_local", effect="read_only", semantic_operations=("consultation",)),
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
        semantic_operations=("consultation",),
        semantic_support_relations=(
            *FRAMED_OPERATION_SUPPORT_RELATIONS,
            "non_mutation_scope",
        ),
    ),
    ActionSpec(
        "choose_target_mode",
        "chain_rpc",
        "Select fake-node, real-node, or sync-observe only when the user explicitly requests that mutation.",
        ("target_mode", "target_mode_explicit", "source_evidence"),
        10,
        "target_mode",
        preserve_pending=True,
        mutation_dimension="target_mode",
        provides_capabilities=("target_mode",),
        crosses_pending_barrier=True,
        required_arguments=("target_mode", "target_mode_explicit", "source_evidence"),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
        semantic_value_grounding_arguments=("target_mode",),
        semantic_value_representative=True,
        entry_intake=True,
        entry_intake_purpose="Enter target-mode selection from any active workflow group.",
        entry_intake_fixed_arguments=(("target_mode_explicit", True),),
        entry_intake_value_arguments=("target_mode",),
    ),
    ActionSpec(
        "choose_chain",
        "chain_rpc",
        "Select the exact user-supplied raw chain identity as the current benchmark target, or preserve multiple user-supplied candidates for typed disambiguation without choosing silently.",
        ("chain_text", "chain_candidates", "source_evidence"),
        20,
        "chain_identity",
        preserve_pending=True,
        mutation_dimension="chain",
        provides_capabilities=("chain_identity",),
        crosses_pending_barrier=True,
        required_arguments=("source_evidence",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
        semantic_value_grounding_arguments=("chain_text", "chain_candidates"),
        open_identity_grounding_arguments=("chain_text", "chain_candidates"),
        semantic_value_representative=True,
        entry_intake=True,
        entry_intake_purpose="Enter source-grounded chain target selection from any active workflow group.",
        entry_intake_value_arguments=("chain_text", "chain_candidates"),
        validator=_validate_chain_selection,
    ),
    ActionSpec(
        "request_chain_selection",
        "chain_rpc",
        "Open typed chain-selection intake when the user requests a chain selection or replacement but supplies no concrete chain.",
        ("chain_candidates", "source_evidence"),
        execution_phase=25,
        target_group="chain_identity",
        preserve_pending=True,
        requires_capabilities=("target_mode",),
        provides_capabilities=("chain_identity",),
        crosses_pending_barrier=True,
        effect="workflow_navigation",
        incomplete_mutation_intake=True,
        required_arguments=("source_evidence",),
        semantic_support_relations=(
            *FRAMED_OPERATION_SUPPORT_RELATIONS,
            "non_mutation_scope",
        ),
    ),
    ActionSpec(
        "request_target_mode_selection",
        "chain_rpc",
        "Open typed target-mode intake when no single replacement target mode is affirmatively selected, including initial selection, replacement requests with no named mode, and requests that only exclude the current or another candidate mode.",
        ("source_evidence",),
        execution_phase=24,
        target_group="target_mode",
        preserve_pending=True,
        provides_capabilities=("target_mode",),
        crosses_pending_barrier=True,
        effect="workflow_navigation",
        incomplete_mutation_intake=True,
        required_arguments=("source_evidence",),
        semantic_support_relations=(
            *FRAMED_OPERATION_SUPPORT_RELATIONS,
            "non_mutation_scope",
        ),
    ),
    ActionSpec(
        "change_group",
        "coordinator",
        "Temporarily suspend the active group, route to one explicitly named workflow group, and resume the interrupted group after the destination work completes.",
        ("group", "navigation_explicit", "source_evidence"),
        50,
        merge_identity=("group",),
        preserve_pending=True,
        crosses_pending_barrier=True,
        required_arguments=("group", "navigation_explicit", "source_evidence"),
        effect="workflow_navigation",
        semantic_support_relations=(
            *FRAMED_OPERATION_SUPPORT_RELATIONS,
            "non_mutation_scope",
        ),
        validator=_validate_group_navigation,
        semantic_operations=("navigation",),
        route_groups=tuple(USER_NAVIGABLE_GROUPS),
    ),
    ActionSpec(
        "request_config_field_input",
        "coordinator",
        "Open the registered typed question for one explicitly named scalar configuration field when the user asks to change it but supplies no new value. Never copy a value from current state, discovery, examples, or defaults.",
        ("config_field", "source_evidence"),
        execution_phase=25,
        target_field_argument="config_field",
        preserve_pending=True,
        crosses_pending_barrier=True,
        requires_specific_change=True,
        incomplete_mutation_intake=True,
        required_arguments=("config_field", "source_evidence"),
        effect="workflow_navigation",
        semantic_support_relations=(
            *FRAMED_OPERATION_SUPPORT_RELATIONS,
            "non_mutation_scope",
        ),
        validator=_validate_config_field_intake,
        semantic_operations=("navigation",),
        route_groups=tuple(
            group_name
            for group_name, group in GROUP_SPEC_BY_NAME.items()
            if group.reconfiguration_questions
        ),
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
        semantic_operations=("navigation",),
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
        crosses_pending_barrier=True,
        interrupts_pending=True,
        semantic_recovery_source_argument="source_evidence",
        semantic_operations=("navigation",),
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
        semantic_value_grounding_arguments=("target_mode",),
        semantic_operations=("administrative",),
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
        semantic_operations=("administrative",),
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
        semantic_operations=("administrative",),
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
        semantic_value_grounding_arguments=("rpc_mode",),
    ),
    ActionSpec(
        "choose_adapter_family",
        "chain_rpc",
        "Confirm the adapter family for an already identified unknown chain.",
        ("adapter_family",),
        22,
        "chain_identity",
        required_arguments=("adapter_family",),
        structured_intake=(
            StructuredIntakeSpec(
                alias="protocol_family",
                fixed_arguments=(),
                value_semantics="direct_value",
                value_argument="adapter_family",
            ),
            StructuredIntakeSpec(
                alias="adapter_family",
                fixed_arguments=(),
                value_semantics="direct_value",
                value_argument="adapter_family",
            ),
        ),
        semantic_value_grounding_arguments=("adapter_family",),
    ),
    ActionSpec(
        "rpc_catalog_command",
        "chain_rpc",
        "Apply exactly one catalog transition, including explicit method-conflict resolution. Use finish only when the user says the already validated methods are sufficient.",
        ("catalog_command", "rpc_method", "rpc_endpoint", "rpc_schema_evidence", "source_evidence"),
        40,
        "endpoint_process",
        preserve_pending=True,
        crosses_pending_barrier=True,
        requires_capabilities=("chain_identity",),
        required_arguments=("catalog_command", "source_evidence"),
        constraints=(
            "set_endpoint requires only rpc_endpoint; set_method requires only rpc_method; append_evidence requires only rpc_schema_evidence; enter, keep_current_method, replace_current_method, and finish accept no payload",
        ),
        entry_intake=True,
        entry_intake_purpose="Enter custom RPC method catalog setup and collect its endpoint, method, and schema evidence.",
        entry_intake_fixed_arguments=(("catalog_command", "enter"),),
        structured_intake=(
            StructuredIntakeSpec(
                alias="custom_rpc",
                fixed_arguments=(("catalog_command", "enter"),),
                value_semantics="boolean_true",
            ),
            StructuredIntakeSpec(
                alias="validation_endpoint",
                fixed_arguments=(("catalog_command", "set_endpoint"),),
                value_semantics="direct_value",
                value_argument="rpc_endpoint",
            ),
            StructuredIntakeSpec(
                alias="rpc_request",
                fixed_arguments=(("catalog_command", "append_evidence"),),
                value_semantics="direct_value",
                value_argument="rpc_schema_evidence",
            ),
        ),
        invalidates_groups=("workload_rpc",),
        incompatible_target_modes=("sync-observe",),
        semantic_support_relations=EVIDENCE_OPERATION_SUPPORT_RELATIONS,
        semantic_value_grounding_arguments=("rpc_endpoint", "rpc_method", "rpc_schema_evidence"),
        exact_source_value_arguments=("rpc_endpoint", "rpc_method", "rpc_schema_evidence"),
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
        semantic_support_relations=EVIDENCE_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "rpc_workload_command",
        "chain_rpc",
        "Apply the workload selection only after catalog methods are validated.",
        ("workload_scope", "rpc_weights", "finish_methods", "source_evidence"),
        41,
        "endpoint_process",
        preserve_pending=True,
        crosses_pending_barrier=True,
        requires_capabilities=("target_mode", "chain_identity"),
        required_arguments=("workload_scope",),
        constraints=(
            "workload_scope is single_replace, mixed_replace, or mixed_add; supplied rpc_weights must total 100; single_replace accepts no weights",
        ),
        semantic_value_grounding_arguments=("workload_scope",),
        validator=_validate_rpc_workload_command,
    ),
    ActionSpec("use_default_workload", "chain_rpc", "Accept the active chain template workload.", execution_phase=40, compiler_groups=("workload_rpc",), requires_capabilities=("target_mode", "chain_identity")),
    ActionSpec("configure_workload_weights", "chain_rpc", "Configure mixed weights for the active template or custom methods.", execution_phase=40, compiler_groups=("workload_rpc",), requires_capabilities=("target_mode", "chain_identity")),
    ActionSpec(
        "request_target_change",
        "chain_rpc",
        "Open a typed choice for changing the chain or target mode.",
        compiler_groups=("workload_rpc",),
        option_navigation_groups=("chain_identity", "target_mode"),
    ),
    ActionSpec("cancel_target_change", "chain_rpc", "Return to workload configuration without changing chain or target mode.", compiler_groups=("workload_rpc",)),
    ActionSpec("set_qps_mode", "performance", "Select quick, standard, or intensive profile. Do not use set_qps_override unless concrete numeric values were supplied.", ("qps_mode", "mutation_explicit", "source_evidence"), 50, "qps_profile", preserve_pending=True, mutation_dimension="qps_profile", requires_capabilities=("target_mode",), crosses_pending_barrier=True, required_arguments=("qps_mode", "mutation_explicit", "source_evidence"), semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS, semantic_value_grounding_arguments=("qps_mode",)),
    ActionSpec("request_qps_customization", "performance", "Enter QPS customization when the user wants to adjust the selected profile but has not supplied every numeric value yet.", ("qps_fields", "source_evidence"), 50, "qps_profile", preserve_pending=True, mutation_dimension="qps_profile", requires_capabilities=("target_mode",), crosses_pending_barrier=True, requires_specific_change=True, incomplete_mutation_intake=True, required_arguments=("source_evidence",), semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS),
    ActionSpec(
        "set_qps_override",
        "performance",
        "Apply concrete QPS profile overrides only when qps_overrides contains user-supplied numeric values. For an adjustment request without values, use request_qps_customization.",
        ("qps_overrides", "source_evidence"),
        50,
        "qps_profile",
        preserve_pending=True,
        mutation_dimension="qps_profile",
        requires_capabilities=("target_mode",),
        crosses_pending_barrier=True,
        required_arguments=("qps_overrides",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec("set_observability", "performance", "Select disabled, local, or exporter-only observability.", ("observability_mode", "mutation_explicit", "source_evidence"), 60, "observability", preserve_pending=True, mutation_dimension="observability", requires_capabilities=("target_mode",), crosses_pending_barrier=True, required_arguments=("observability_mode", "mutation_explicit", "source_evidence"), semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS, semantic_value_grounding_arguments=("observability_mode",)),
    ActionSpec("set_sync_observe_source", "sync_observe", "Select the real sync-observe data source.", ("sync_observe_source",), target_group="sync_observe", required_arguments=("sync_observe_source",), semantic_value_grounding_arguments=("sync_observe_source",)),
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
        semantic_value_grounding_arguments=("sync_observe_stop_condition",),
        validator=_validate_sync_observe_options,
    ),
    ActionSpec("approve_preflight_smoke", "execution", "Approve one idempotent preflight/smoke submission.", compiler_groups=("preflight_smoke_execution",), effect="execution", internal_only=True, typed_option_only=True),
    ActionSpec("reject_preflight_smoke", "execution", "Pause before preflight/smoke without submitting a job.", compiler_groups=("preflight_smoke_execution",), internal_only=True, typed_option_only=True),
    ActionSpec("approve_final_benchmark", "execution", "Approve final real-node benchmark submission after isolated smoke success.", compiler_groups=("job_monitoring",), effect="execution", internal_only=True, typed_option_only=True),
    ActionSpec("reject_final_benchmark", "execution", "Pause after successful real-node smoke without submitting the final benchmark.", compiler_groups=("job_monitoring",), internal_only=True, typed_option_only=True),
    ActionSpec(
        "set_accounts_presence",
        "environment",
        "Confirm whether a separate accounts/state disk exists.",
        ("has_accounts_device", "source_evidence"),
        target_group="accounts_disk",
        preserve_pending=True,
        crosses_pending_barrier=True,
        required_arguments=("has_accounts_device",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
        semantic_value_grounding_arguments=("has_accounts_device",),
    ),
    ActionSpec(
        "propose_config_values",
        "environment",
        "Propose values extracted from natural language, YAML, JSON, env, or tables; after user review, normal fallback asks for required values that the partial input did not supply.",
        ("config_values", "unmapped_values", "conflicts", "source_format", "source_evidence"),
        25,
        compiler_groups=(
            "provider_deployment",
            "ledger_disk",
            "accounts_disk",
            "network",
        ),
        preserve_pending=True,
        merge_mapping_fields=("config_values", "unmapped_values"),
        merge_sequence_fields=("conflicts",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
    ),
    ActionSpec(
        "start_evidence_collection",
        "analysis",
        "Start framed multi-line evidence collection for the active evidence question.",
        ("evidence", "source_evidence"),
        execution_phase=3,
        preserve_pending=True,
        crosses_pending_barrier=True,
        required_arguments=("evidence", "source_evidence"),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
        semantic_operations=("evidence_analysis",),
        route_groups=("error_evidence_analysis",),
    ),
    ActionSpec(
        "append_evidence_collection",
        "analysis",
        "Append the current user-supplied evidence fragment to an active evidence collection; never use this for questions, navigation, corrections, or unrelated conversation.",
        ("evidence", "source_evidence"),
        execution_phase=4,
        preserve_pending=True,
        crosses_pending_barrier=True,
        required_arguments=("evidence", "source_evidence"),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
        required_state_path=("evidence_collection", "status"),
        required_state_values=("active",),
        state_transition_resolver=_evidence_append_transition,
        validator=_validate_evidence_collection_append,
        semantic_operations=("evidence_analysis",),
        route_groups=("error_evidence_analysis",),
    ),
    ActionSpec(
        "finish_evidence_collection",
        "analysis",
        "Finish the active evidence collection only when the user explicitly says the evidence block is complete.",
        ("source_evidence",),
        execution_phase=4,
        preserve_pending=True,
        crosses_pending_barrier=True,
        required_arguments=("source_evidence",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
        required_state_path=("evidence_collection", "status"),
        required_state_values=("active",),
        state_transition_path=("evidence_collection", "status"),
        state_transition_value=None,
        semantic_operations=("evidence_analysis",),
        route_groups=("error_evidence_analysis",),
    ),
    ActionSpec(
        "pause_evidence_collection",
        "analysis",
        "Pause and preserve the active evidence collection only when the user explicitly requests a temporary detour.",
        ("source_evidence",),
        execution_phase=3,
        preserve_pending=True,
        crosses_pending_barrier=True,
        required_arguments=("source_evidence",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
        required_state_path=("evidence_collection", "status"),
        required_state_values=("active",),
        state_transition_path=("evidence_collection", "status"),
        state_transition_value="paused",
        semantic_operations=("evidence_analysis",),
        route_groups=("error_evidence_analysis",),
    ),
    ActionSpec(
        "resume_evidence_collection",
        "analysis",
        "Resume a paused evidence collection only when the user explicitly asks to continue that collection.",
        ("source_evidence",),
        execution_phase=3,
        preserve_pending=True,
        crosses_pending_barrier=True,
        required_arguments=("source_evidence",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
        required_state_path=("evidence_collection", "status"),
        required_state_values=("paused",),
        state_transition_path=("evidence_collection", "status"),
        state_transition_value="active",
        semantic_operations=("evidence_analysis",),
        route_groups=("error_evidence_analysis",),
    ),
    ActionSpec(
        "cancel_evidence_collection",
        "analysis",
        "Discard the active evidence collection only when the user explicitly cancels it.",
        ("source_evidence",),
        execution_phase=4,
        preserve_pending=True,
        crosses_pending_barrier=True,
        required_arguments=("source_evidence",),
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
        required_state_path=("evidence_collection", "status"),
        required_state_values=("active", "paused"),
        state_transition_path=("evidence_collection", "status"),
        state_transition_value=None,
        semantic_operations=("evidence_analysis",),
        route_groups=("error_evidence_analysis",),
    ),
    ActionSpec(
        "request_evidence_analysis",
        "analysis",
        "Request analysis of logs/errors/evidence when no analyzable payload has been supplied yet; open one typed freeform evidence collection for the later bounded payload.",
        ("question",),
        execution_phase=5,
        lifetime="turn_local",
        effect="read_only",
        turn_local_result_roots=("evidence_collection",),
        crosses_pending_barrier=True,
        incomplete_read_intake=True,
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
        semantic_operations=("evidence_analysis",),
        route_groups=("error_evidence_analysis",),
    ),
    ActionSpec(
        "analyze_evidence",
        "analysis",
        "Analyze a real pasted, saved, or completed logs/errors/evidence payload. Request wording that merely asks for analysis is not evidence and belongs to request_evidence_analysis.",
        ("evidence", "question"),
        execution_phase=5,
        lifetime="turn_local",
        effect="read_only",
        turn_local_result_roots=("evidence_buffer", "evidence_collection"),
        crosses_pending_barrier=True,
        required_arguments=("evidence",),
        semantic_recovery_source_argument="evidence",
        semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS,
        semantic_operations=("evidence_analysis",),
        route_groups=("error_evidence_analysis",),
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
        semantic_operations=("report_analysis",),
        route_groups=("report_artifact_analysis",),
    ),
    ActionSpec("correct_failure", "recovery", "Reopen only the policy-declared affected configuration group.", compiler_groups=("failure_recovery",)),
    ActionSpec(
        "inspect_failure",
        "recovery",
        "Inspect structured failure evidence without changing workflow configuration.",
        compiler_groups=("failure_recovery",),
        lifetime="turn_local",
        effect="read_only",
    ),
    ActionSpec("retry_failure", "recovery", "Clear a retryable external-service failure without executing a benchmark side effect.", compiler_groups=("failure_recovery",)),
    ActionSpec("cancel_failure_recovery", "recovery", "Pause recovery while preserving evidence and confirmed configuration.", compiler_groups=("failure_recovery",)),
    ActionSpec(
        "activate_harness_recovery",
        "recovery",
        "Install one validated Harness invariant failure for typed recovery.",
        ("failure_record",),
        execution_phase=0,
        target_group="failure_recovery",
        effect="workflow_state_mutation",
        crosses_pending_barrier=True,
        semantic_operations=("administrative",),
        required_arguments=("failure_record",),
        internal_only=True,
    ),
    ActionSpec(
        "reenter_secret_reference",
        "coordinator",
        "Restore one exact process-local secret reference after restart without changing durable workflow state.",
        (
            "scope_id",
            "owner_kind",
            "owner_id",
            "owner_revision",
            "reference",
            "atom_id",
            "value_hash",
            "secret_value",
            "source_evidence",
        ),
        execution_phase=0,
        effect="workflow_state_mutation",
        preserve_pending=True,
        crosses_pending_barrier=True,
        semantic_operations=("pending_answer",),
        required_arguments=(
            "scope_id",
            "owner_kind",
            "owner_id",
            "owner_revision",
            "reference",
            "atom_id",
            "value_hash",
            "secret_value",
            "source_evidence",
        ),
        internal_only=True,
    ),
    ActionSpec(
        "resolve_semantic_draft_atom",
        "coordinator",
        "Resolve exactly one pending semantic draft atom under its bound draft identity and revision.",
        ("draft_id", "revision", "atom_id", "resolution", "source_evidence"),
        execution_phase=0,
        effect="workflow_state_mutation",
        crosses_pending_barrier=True,
        semantic_operations=("pending_answer",),
        required_arguments=(
            "draft_id",
            "revision",
            "atom_id",
            "resolution",
            "source_evidence",
        ),
        internal_only=True,
    ),
    ActionSpec(
        "previous_semantic_draft_atom",
        "coordinator",
        "Reopen exactly the preceding semantic draft atom under its bound draft identity and revision.",
        ("draft_id", "revision", "atom_id", "source_evidence"),
        execution_phase=0,
        effect="workflow_state_mutation",
        crosses_pending_barrier=True,
        semantic_operations=("navigation",),
        required_arguments=("draft_id", "revision", "atom_id", "source_evidence"),
        internal_only=True,
    ),
    ActionSpec(
        "cancel_semantic_draft",
        "coordinator",
        "Cancel the complete pending semantic draft without applying candidate actions.",
        ("draft_id", "revision", "reason", "source_evidence"),
        execution_phase=0,
        effect="workflow_state_mutation",
        crosses_pending_barrier=True,
        semantic_operations=("pending_answer",),
        required_arguments=("draft_id", "revision", "reason", "source_evidence"),
        internal_only=True,
    ),
    ActionSpec("answer_pending", "coordinator", "Answer the active typed question after interpreting non-exact user language.", ("answer", "selected_value", "source_evidence"), 0, required_arguments=("source_evidence",), semantic_support_relations=FRAMED_OPERATION_SUPPORT_RELATIONS, semantic_operations=("pending_answer",)),
    ActionSpec("unknown", "orientation", "Report that no safe action could be resolved.", ("reason",), effect="read_only", required_arguments=("reason",), semantic_operations=("unresolved",)),
)


MODE_COMPARISON_TOPIC = "mode_comparison"

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
    MODE_COMPARISON_TOPIC,
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

CONSULTATION_TOPIC_PURPOSES: Mapping[str, str] = {
    "identity": "Identify the Agent, its runtime role, and where it operates.",
    "capabilities": "Explain what the Agent and benchmark product can do.",
    "supported_chains": "List or explain supported blockchain chains.",
    "extension": "Explain custom RPC, new-chain, or unsupported-family extension paths.",
    "config_explanation": "Explain one current configuration field, prompt, or option.",
    "current_config": "Report retained and confirmed workflow configuration values.",
    "workload_config": "Report effective or default RPC methods and weights.",
    "current_context": "Report the active workflow context and pending question.",
    "next_action": "Explain the next workflow action or blocker.",
    "startup_discovery": "Report values inferred by startup environment discovery.",
    "environment_readiness": "Report environment and dependency readiness.",
    "requirements": "Explain what the user must prepare for a test.",
    "workflow": "Explain the benchmark workflow and configuration sequence.",
    MODE_COMPARISON_TOPIC: "Compare fake-node, real-node, and sync-observe modes.",
    "performance_benchmark_guidance": "Recommend a mode for a stated performance goal.",
    "execution_preflight_smoke": "Explain preflight and smoke execution semantics.",
    "recommendation": "Recommend a safe next benchmark starting path.",
    "reset_help": "Explain how to clear or modify retained workflow configuration.",
    "evidence_help": "Explain how to provide logs, errors, reports, or multiline evidence.",
    "correction": "Acknowledge and recover from an incorrect or irrelevant prior answer.",
    "current_job": "Report whether a current or historical benchmark job exists and identify it.",
    "job_status": "Report the status of a selected current or historical job.",
    "execution_status": "Report workflow execution, preflight, smoke, or benchmark status.",
}

if set(CONSULTATION_TOPIC_PURPOSES) != set(CONSULTATION_TOPICS):
    raise RuntimeError("consultation topic purposes must cover the consultation registry exactly")

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
    "modes": MODE_COMPARISON_TOPIC,
    "mode": MODE_COMPARISON_TOPIC,
    "restart": "reset_help",
    "start_over": "reset_help",
    "env_readiness": "environment_readiness",
    "recommend_start": "recommendation",
}

STATE_AUDIT_TOPICS = frozenset({"current_config", "current_context", "next_action", "execution_status"})

ACTION_BY_TYPE = {spec.action_type: spec for spec in ACTION_SPECS}


def action_route_groups(spec: ActionSpec) -> frozenset[str]:
    """Return the registry-authoritative groups one action can serve.

    Planner stages must consume this projection instead of independently
    inferring reachability from only a static target or compiler owner.
    """

    return frozenset(
        group
        for group in (
            spec.target_group,
            *spec.compiler_groups,
            *spec.route_groups,
        )
        if group
    )


def action_spec_serves_route(
    spec: ActionSpec,
    *,
    operation: str,
    group: str = "",
) -> bool:
    """Resolve one semantic operation/group pair to an executable spec."""

    if spec.internal_only or operation not in spec.semantic_operations:
        return False
    route_groups = action_route_groups(spec)
    if not route_groups:
        return operation != "domain_request"
    if not group:
        return operation != "domain_request"
    return group in route_groups


def project_action_specs(
    *,
    owners: frozenset[str] | None = None,
    groups: frozenset[str] | None = None,
    lifetimes: frozenset[ActionLifetime] | None = None,
    action_types: frozenset[str] | None = None,
) -> tuple[ActionSpec, ...]:
    """Return one deterministic registry projection for a bounded planner."""

    return tuple(
        spec
        for spec in ACTION_SPECS
        if (owners is None or spec.owner in owners)
        and not spec.internal_only
        and (
            groups is None
            or bool(groups.intersection(action_route_groups(spec)))
        )
        and (lifetimes is None or spec.lifetime in lifetimes)
        and (action_types is None or spec.action_type in action_types)
    )


for _spec in ACTION_SPECS:
    _unknown_route_groups = set(_spec.route_groups) - set(GROUP_SPEC_BY_NAME)
    if _unknown_route_groups:
        raise ValueError(
            f"{_spec.action_type} declares unknown route groups: "
            + ", ".join(sorted(_unknown_route_groups))
        )
    _unknown_invalidated_groups = (
        set(_spec.invalidates_groups) - set(GROUP_SPEC_BY_NAME)
    )
    if _unknown_invalidated_groups:
        raise ValueError(
            f"{_spec.action_type} declares unknown invalidated groups: "
            + ", ".join(sorted(_unknown_invalidated_groups))
        )
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
    """Return actions incompatible with the current ordered lifecycle state.

    Target-mode changes take effect in action order. This lets one explicit
    transaction leave a workflow and then use its newly compatible actions,
    while preventing a domain action from mutating state owned by the current
    workflow before that transition occurs.
    """

    target_mode = normalize_target_mode(state.get("target_mode"))
    projected_capabilities = {
        capability
        for spec in ACTION_SPECS
        for capability in spec.provides_capabilities
        if state_has_capability(state, capability)
    }
    projected_state_values: dict[tuple[str, ...], Any] = {}
    rejected: list[int] = []
    for index, action in enumerate(actions):
        action_type = str(action.get("type") or "")
        if action_type == "choose_target_mode":
            replacement = normalize_target_mode(action.get("target_mode"))
            if replacement:
                target_mode = replacement
                projected_capabilities.add("target_mode")
            continue
        spec = ACTION_BY_TYPE.get(action_type)
        if spec is None:
            continue
        if (
            spec.incomplete_mutation_intake
            and spec.provides_capabilities
            and bool(str(action.get("source_evidence") or "").strip())
            and action.get("selection_contract_verified") is not True
            and not replacement_intake_admission_verified(action)
            and all(
                capability in projected_capabilities
                for capability in spec.provides_capabilities
            )
        ):
            rejected.append(index)
            continue
        incompatible_mode = target_mode in spec.incompatible_target_modes
        current: Any = projected_state_values.get(spec.required_state_path)
        if spec.required_state_path not in projected_state_values:
            current = state
            for key in spec.required_state_path:
                current = current.get(key) if isinstance(current, Mapping) else None
        incompatible_state = bool(
            spec.required_state_path
            and current not in spec.required_state_values
        )
        if incompatible_mode or incompatible_state:
            rejected.append(index)
            continue
        transition = (
            spec.state_transition_resolver(state, action)
            if spec.state_transition_resolver is not None
            else (
                (spec.state_transition_path, spec.state_transition_value)
                if spec.state_transition_path
                else None
            )
        )
        if transition is not None:
            projected_state_values[transition[0]] = transition[1]
        if not spec.incomplete_mutation_intake:
            projected_capabilities.update(spec.provides_capabilities)
    return tuple(rejected)


def resolve_action_target_group(action: Mapping[str, Any]) -> str:
    """Resolve one action's static or registry-derived workflow destination."""

    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    if spec is None:
        return ""
    if spec.target_group:
        return str(spec.target_group)
    if not spec.target_field_argument:
        return ""
    from agent.workflows.group_registry import group_for_field

    return group_for_field(
        str(action.get(spec.target_field_argument) or "").strip()
    )


ACTION_METADATA_FIELDS = frozenset({"type", "confidence", "reason", "action_id"})
TRUSTED_ACTION_METADATA_FIELDS = frozenset({
    "selection_contract_verified",
    "semantic_purpose_verified",
    "_replacement_intake_receipt",
    "pending_option_semantic_verified",
    "chain_selection_semantic_verified",
    "target_mode_semantic_verified",
    "group_navigation_semantic_verified",
    "_semantic_admission_receipt",
    "_semantic_consensus_receipt",
    "_proposal_field_receipts",
    "_proposal_transaction_hashes",
    "_admission_action_id",
    "_transaction_action_ids",
    "_plan_transaction_hash",
    "_origin_text",
    "_queue_origin_group",
    "_submitted_turn_index",
    "_plan_scope",
    "_plan_index",
    "_merged_origin_texts",
})

SEMANTIC_ADMISSION_RECEIPT_VERSION = 4


def action_registry_contract_hash() -> str:
    """Return the complete deterministic action-contract identity."""

    specs = []
    for spec in ACTION_SPECS:
        specs.append({
            "action_type": spec.action_type,
            "owner": spec.owner,
            "purpose": spec.purpose,
            "allowed_arguments": list(spec.allowed_arguments),
            "execution_phase": spec.execution_phase,
            "target_group": spec.target_group,
            "target_field_argument": spec.target_field_argument,
            "compiler_groups": list(spec.compiler_groups),
            "route_groups": list(spec.route_groups),
            "semantic_operations": list(spec.semantic_operations),
            "merge_identity": list(spec.merge_identity),
            "preserve_pending": spec.preserve_pending,
            "mutation_dimension": spec.mutation_dimension,
            "allows_followup_actions": spec.allows_followup_actions,
            "requires_capabilities": list(spec.requires_capabilities),
            "provides_capabilities": list(spec.provides_capabilities),
            "merge_mapping_fields": list(spec.merge_mapping_fields),
            "merge_sequence_fields": list(spec.merge_sequence_fields),
            "lifetime": spec.lifetime,
            "effect": spec.effect,
            "turn_local_result_roots": list(spec.turn_local_result_roots),
            "crosses_pending_barrier": spec.crosses_pending_barrier,
            "interrupts_pending": spec.interrupts_pending,
            "replaces_deferred_queue": spec.replaces_deferred_queue,
            "requires_specific_change": spec.requires_specific_change,
            "incomplete_mutation_intake": spec.incomplete_mutation_intake,
            "incomplete_read_intake": spec.incomplete_read_intake,
            "entry_intake": spec.entry_intake,
            "entry_intake_purpose": spec.entry_intake_purpose,
            "entry_intake_fixed_arguments": [
                [key, value]
                for key, value in spec.entry_intake_fixed_arguments
            ],
            "entry_intake_value_arguments": list(
                spec.entry_intake_value_arguments
            ),
            "structured_intake": [
                {
                    "alias": intake.alias,
                    "fixed_arguments": [
                        [key, value]
                        for key, value in intake.fixed_arguments
                    ],
                    "value_semantics": intake.value_semantics,
                    "value_argument": intake.value_argument,
                }
                for intake in spec.structured_intake
            ],
            "invalidates_groups": list(spec.invalidates_groups),
            "required_arguments": list(spec.required_arguments),
            "constraints": list(spec.constraints),
            "suppressed_by": list(spec.suppressed_by),
            "semantic_recovery_source_argument": spec.semantic_recovery_source_argument,
            "semantic_support_relations": list(spec.semantic_support_relations),
            "semantic_value_grounding_arguments": list(spec.semantic_value_grounding_arguments),
            "open_identity_grounding_arguments": list(
                spec.open_identity_grounding_arguments
            ),
            "semantic_value_representative": spec.semantic_value_representative,
            "explicit_scope_authorization": spec.explicit_scope_authorization,
            "exact_source_value_arguments": list(spec.exact_source_value_arguments),
            "pending_option_semantic": spec.pending_option_semantic,
            "option_navigation_groups": list(spec.option_navigation_groups),
            "pending_option_admission": spec.pending_option_admission,
            "incompatible_target_modes": list(spec.incompatible_target_modes),
            "required_state_path": list(spec.required_state_path),
            "required_state_values": list(spec.required_state_values),
            "state_transition_path": list(spec.state_transition_path),
            "state_transition_value": spec.state_transition_value,
            "internal_only": spec.internal_only,
            "typed_option_only": spec.typed_option_only,
            "state_transition_resolver": (
                f"{spec.state_transition_resolver.__module__}.{spec.state_transition_resolver.__qualname__}"
                if spec.state_transition_resolver is not None
                else ""
            ),
            "validator": (
                f"{spec.validator.__module__}.{spec.validator.__qualname__}"
                if spec.validator is not None
                else ""
            ),
        })
    return _content_hash({
        "specs": specs,
        "argument_schemas": ACTION_ARGUMENT_SCHEMAS,
        "semantic_scope_policies": SEMANTIC_SCOPE_POLICIES,
        "semantic_operation_purposes": SEMANTIC_OPERATION_PURPOSES,
        "consultation_topics": sorted(CONSULTATION_TOPICS),
        "semantic_value_domain_policy": SEMANTIC_VALUE_DOMAIN_POLICY,
        "semantic_value_domains": registered_semantic_value_domains(),
        "pending_barrier_policies": PENDING_BARRIER_POLICIES,
    })


def registered_semantic_value_domains() -> tuple[dict[str, Any], ...]:
    """Expose exact values whose workflow ownership is already authoritative.

    Closed enum values come from action schemas. Supported chain identities
    and their exact aliases come from the repository chain catalog. The
    planner and final admission boundary use this one registry to prevent a
    value owned by one workflow group from being consumed by an unrelated
    manual-text question.
    """

    records: list[dict[str, Any]] = []
    argument_specs: dict[str, list[ActionSpec]] = {}
    for spec in ACTION_SPECS:
        for argument in spec.semantic_value_grounding_arguments:
            argument_specs.setdefault(argument, []).append(spec)
    for argument, specs in argument_specs.items():
        schema = ACTION_ARGUMENT_SCHEMAS.get(argument) or {}
        enum = schema.get("enum")
        if not isinstance(enum, list):
            continue
        canonical_groups = {
            spec.target_group for spec in specs if spec.target_group
        }
        canonical_group = (
            next(iter(canonical_groups))
            if len(canonical_groups) == 1
            else ""
        )
        explicit_representatives = [
            spec for spec in specs if spec.semantic_value_representative
        ]
        representatives = explicit_representatives or [
            spec
            for spec in specs
            if (
                spec.target_group == canonical_group
                if canonical_group
                else not spec.target_group
            )
        ]
        if len(representatives) != 1:
            raise RuntimeError(
                "semantic value argument requires exactly one canonical "
                f"representative action: {argument}"
            )
        representative = representatives[0]
        if (
            canonical_group
            and representative.target_group != canonical_group
        ):
            raise RuntimeError(
                "semantic value representative does not own the canonical "
                f"workflow group: {argument}/{representative.action_type}"
            )
        for value in enum:
            canonical = str(value).strip()
            if not canonical:
                continue
            records.append({
                "action_type": representative.action_type,
                "argument": argument,
                "target_group": canonical_group,
                "semantic_owner": (
                    canonical_group
                    or f"action:{representative.action_type}"
                ),
                "value": canonical,
                "canonical_value": canonical,
                "domain_kind": "closed_enum",
            })
    known_chains = set(repo_chain_names())
    chain_values = {
        chain: chain for chain in known_chains
    }
    chain_values.update({
        alias: canonical
        for alias, canonical in canonical_chain_aliases().items()
        if canonical in known_chains
    })
    chain_identity_specs = [
        spec
        for spec in ACTION_SPECS
        if spec.target_group == "chain_identity"
        and "chain_text" in spec.entry_intake_value_arguments
        and spec.semantic_value_representative
    ]
    if len(chain_identity_specs) != 1:
        raise RuntimeError(
            "known chain identities require exactly one canonical "
            "semantic-value representative action"
        )
    chain_identity_spec = chain_identity_specs[0]
    records.extend({
        "action_type": chain_identity_spec.action_type,
        "argument": "chain_text",
        "target_group": chain_identity_spec.target_group,
        "semantic_owner": chain_identity_spec.target_group,
        "value": alias,
        "canonical_value": canonical,
        "domain_kind": "known_identity",
    } for alias, canonical in sorted(chain_values.items()))
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record.get("semantic_owner") or ""),
            str(record.get("value") or "").strip().casefold(),
        )
        existing = unique.get(key)
        if existing is None:
            unique[key] = record
            continue
        authority_fields = (
            "action_type",
            "argument",
            "canonical_value",
            "target_group",
        )
        conflicting_fields = [
            field
            for field in authority_fields
            if str(existing.get(field) or "").strip()
            != str(record.get(field) or "").strip()
        ]
        if conflicting_fields:
            raise RuntimeError(
                "conflicting semantic value domain registration for "
                f"{key[0] or '<missing-owner>'}/{key[1] or '<missing-value>'}: "
                f"{', '.join(conflicting_fields)}"
            )
    return tuple(unique.values())


def semantic_value_domain_conflicts(
    text: Any,
    *,
    owning_group: str,
    pending_question: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return exact registered values owned by another workflow group."""

    source = str(text or "").strip()
    pending = dict(pending_question or {})
    if not source:
        return ()
    conflicts: list[dict[str, Any]] = []
    for record in registered_semantic_value_domains():
        semantic_owner = str(record.get("semantic_owner") or "")
        value = str(record.get("value") or "").strip()
        if (
            semantic_owner == owning_group
            or not value
            or _pending_option_owns_semantic_value(pending, record)
        ):
            continue
        pattern = (
            rf"(?<![{_SEMANTIC_VALUE_IDENTIFIER_CHARACTER}])"
            rf"{re.escape(value)}"
            rf"(?![{_SEMANTIC_VALUE_IDENTIFIER_CHARACTER}])"
        )
        if re.search(pattern, source, flags=re.IGNORECASE):
            conflicts.append(record)
    if not conflicts or not pending:
        return tuple(conflicts)
    specialized_spans = _pending_specialized_value_spans(source, pending)
    return tuple(
        record
        for record in conflicts
        if not _semantic_value_occurs_only_within_spans(
            source,
            str(record.get("value") or ""),
            specialized_spans,
        )
    )


def _pending_specialized_value_spans(
    source: str,
    pending: Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    """Return source spans whose structure proves specialized field ownership."""

    kind = str(pending.get("kind") or "")
    validation = (
        dict(pending.get("validation") or {})
        if isinstance(pending.get("validation"), Mapping)
        else {}
    )
    value_type = str(validation.get("value_type") or "")
    input_mode = str(validation.get("input_mode") or "")
    candidates: list[str] = []
    if kind == "url" or value_type == "url":
        candidates.extend(extract_url_candidates(source))
    elif kind == "json" or value_type == "json":
        candidate = extract_json_object_or_array(source)
        if candidate:
            candidates.append(candidate)
    elif (
        kind == "evidence"
        or value_type == "evidence_contribution"
        or input_mode == "rpc_method_or_schema_evidence"
    ):
        candidates.extend(extract_url_candidates(source))
    spans: list[tuple[int, int]] = []
    lowered = source.casefold()
    for candidate in candidates:
        needle = candidate.casefold()
        start = 0
        while needle:
            index = lowered.find(needle, start)
            if index < 0:
                break
            spans.append((index, index + len(candidate)))
            start = index + len(candidate)
    if (
        kind == "evidence"
        or value_type == "evidence_contribution"
        or input_mode == "rpc_method_or_schema_evidence"
    ):
        spans.extend(
            extract_rpc_wire_evidence_spans(
                source,
                allow_params_only=True,
            )
        )
    return tuple(spans)


def _semantic_value_occurs_only_within_spans(
    source: str,
    value: str,
    spans: tuple[tuple[int, int], ...],
) -> bool:
    if not value or not spans:
        return False
    pattern = (
        rf"(?<![{_SEMANTIC_VALUE_IDENTIFIER_CHARACTER}])"
        rf"{re.escape(value)}"
        rf"(?![{_SEMANTIC_VALUE_IDENTIFIER_CHARACTER}])"
    )
    occurrences = tuple(re.finditer(pattern, source, flags=re.IGNORECASE))
    return bool(occurrences) and all(
        any(
            span_start <= occurrence.start()
            and occurrence.end() <= span_end
            for span_start, span_end in spans
        )
        for occurrence in occurrences
    )


def _pending_option_owns_semantic_value(
    pending: Mapping[str, Any],
    record: Mapping[str, Any],
) -> bool:
    value_identities = {
        str(record.get("value") or "").strip().casefold(),
        str(record.get("canonical_value") or "").strip().casefold(),
    } - {""}
    action_type = str(record.get("action_type") or "")
    argument = str(record.get("argument") or "")
    for option in pending.get("options") or []:
        if not isinstance(option, Mapping):
            continue
        option_value = str(option.get("value") or "").strip().casefold()
        action = (
            dict(option.get("action") or {})
            if isinstance(option.get("action"), Mapping)
            else {}
        )
        action_value = str(action.get(argument) or "").strip().casefold()
        if (
            option_value in value_identities
            or (
                str(action.get("type") or "") == action_type
                and action_value in value_identities
            )
        ):
            return True
    return False


def admission_contract_hash() -> str:
    """Bind receipts to both workflow ownership and action semantics."""

    from agent.workflows.group_registry import group_registry_contract_hash

    return _content_hash({
        "group_registry_hash": group_registry_contract_hash(),
        "action_registry_hash": action_registry_contract_hash(),
    })


def build_admission_transaction_hash(
    *,
    thread_id: str,
    session_id: str,
    submitted_turn_index: int,
    actions: list[Mapping[str, Any]],
    semantic_units: list[Mapping[str, Any]],
    admission_action_ids: list[str],
) -> str:
    """Return the identity of one ordered, source-owned admission transaction."""

    canonical_actions = []
    for action in actions:
        canonical_actions.append({
            str(key): value
            for key, value in action.items()
            if key not in TRUSTED_ACTION_METADATA_FIELDS
            and key not in ACTION_METADATA_FIELDS
            and not str(key).startswith("_")
        } | {"type": str(action.get("type") or "")})
    canonical_units = [
        {
            "unit_id": str(unit.get("unit_id") or ""),
            "clause_id": str(unit.get("clause_id") or ""),
            "source_text": str(unit.get("source_text") or ""),
            "disposition": str(unit.get("disposition") or ""),
            "action_indexes": list(unit.get("action_indexes") or []),
        }
        for unit in semantic_units
    ]
    return _content_hash({
        "thread_id": str(thread_id or ""),
        "session_id": str(session_id or ""),
        "submitted_turn_index": int(submitted_turn_index),
        "admission_action_ids": list(admission_action_ids),
        "actions": canonical_actions,
        "semantic_units": canonical_units,
    })


def build_semantic_consensus_receipt(
    *,
    thread_id: str,
    session_id: str,
    submitted_turn_index: int,
    transaction_hash: str,
    plan_hash: str,
    admission_action_ids: Sequence[str],
    review_hashes: Sequence[str],
    review_ids: Sequence[str],
    request_count: int,
    request_sizes: Sequence[int],
) -> dict[str, Any]:
    """Mint one durable receipt for every grounded-mutation authority."""

    normalized_review_hashes = [
        str(value).strip()
        for value in review_hashes
        if str(value).strip()
    ]
    normalized_review_ids = [
        str(value).strip()
        for value in review_ids
        if str(value).strip()
    ]

    payload = {
        "version": SEMANTIC_ADMISSION_RECEIPT_VERSION,
        "receipt_type": "semantic_consensus",
        "thread_id": str(thread_id or "").strip(),
        "session_id": str(session_id or "").strip(),
        "submitted_turn_index": int(submitted_turn_index),
        "transaction_hash": str(transaction_hash or "").strip(),
        "plan_hash": str(plan_hash or "").strip(),
        "admission_action_ids": [
            str(value).strip()
            for value in admission_action_ids
            if str(value).strip()
        ],
        "review_hashes": normalized_review_hashes,
        "review_ids": normalized_review_ids,
        "request_count": int(request_count),
        "request_sizes": [int(value) for value in request_sizes],
        "action_contract_hash": action_registry_contract_hash(),
        "admission_contract_hash": admission_contract_hash(),
    }
    if (
        not all(
            str(payload[key] or "").strip()
            for key in (
                "thread_id",
                "session_id",
                "transaction_hash",
                "plan_hash",
            )
        )
        or len(payload["review_hashes"]) < 3
        or len(payload["review_ids"]) != len(payload["review_hashes"])
        or not payload["admission_action_ids"]
        or payload["request_count"] != len(payload["request_sizes"])
        or payload["request_count"] != len(payload["review_hashes"])
        or payload["request_count"] < 3
        or any(size <= 0 for size in payload["request_sizes"])
        or not _valid_semantic_jury_review_ids(payload["review_ids"])
    ):
        raise ValueError("semantic consensus receipt is incomplete")
    return {**payload, "receipt_id": _content_hash(payload)}


def _valid_semantic_jury_review_ids(review_ids: Sequence[str]) -> bool:
    """Require complete, ordered evidence for all jury members and attempts."""

    ids = [str(value) for value in review_ids]
    if ids and ids[-1] == "closed_enum_grounding":
        ids = ids[:-1]
    members: dict[int, list[int]] = {1: [], 2: [], 3: []}
    observed_members: list[int] = []
    for review_id in ids:
        match = re.fullmatch(r"jury_([123])/attempt_([12])", review_id)
        if match is None:
            return False
        member = int(match.group(1))
        attempt = int(match.group(2))
        observed_members.append(member)
        members[member].append(attempt)
    return (
        observed_members == sorted(observed_members)
        and all(members[member] in ([1], [1, 2]) for member in (1, 2, 3))
    )


def validate_semantic_consensus_receipt(
    action: Mapping[str, Any],
    *,
    thread_id: str | None = None,
    session_id: str | None = None,
    submitted_turn_index: int | None = None,
) -> None:
    """Validate one grounded-mutation consensus receipt against its action."""

    receipt = action.get("_semantic_consensus_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("semantic consensus receipt is missing")
    expected = build_semantic_consensus_receipt(
        thread_id=str(receipt.get("thread_id") or ""),
        session_id=str(receipt.get("session_id") or ""),
        submitted_turn_index=int(receipt.get("submitted_turn_index") or 0),
        transaction_hash=str(receipt.get("transaction_hash") or ""),
        plan_hash=str(receipt.get("plan_hash") or ""),
        admission_action_ids=receipt.get("admission_action_ids") or (),
        review_hashes=receipt.get("review_hashes") or (),
        review_ids=receipt.get("review_ids") or (),
        request_count=int(receipt.get("request_count") or 0),
        request_sizes=receipt.get("request_sizes") or (),
    )
    if dict(receipt) != expected:
        raise ValueError("semantic consensus receipt does not match its contract")
    if str(action.get("_plan_transaction_hash") or "") != str(
        receipt.get("transaction_hash") or ""
    ):
        raise ValueError("semantic consensus receipt transaction mismatch")
    if str(action.get("_admission_action_id") or "") not in {
        str(value)
        for value in receipt.get("admission_action_ids") or ()
    }:
        raise ValueError("semantic consensus receipt action identity mismatch")
    if thread_id is not None and str(receipt.get("thread_id") or "") != str(
        thread_id
    ):
        raise ValueError("semantic consensus receipt thread mismatch")
    if session_id is not None and str(receipt.get("session_id") or "") != str(
        session_id
    ):
        raise ValueError("semantic consensus receipt session mismatch")
    if (
        submitted_turn_index is not None
        and int(receipt.get("submitted_turn_index") or 0)
        != int(submitted_turn_index)
    ):
        raise ValueError("semantic consensus receipt belongs to another turn")


def build_replacement_intake_admission_receipt(
    *,
    thread_id: str,
    session_id: str,
    submitted_turn_index: int,
    transaction_hash: str,
    admission_action_id: str,
    action_type: str,
    source_evidence: str,
    target_group: str,
    provides_capabilities: Sequence[str],
    reviewer_evidence_hash: str,
) -> dict[str, Any]:
    """Mint one reviewed authorization to reopen a completed typed intake."""

    payload = {
        "version": SEMANTIC_ADMISSION_RECEIPT_VERSION,
        "action_type": str(action_type or "").strip(),
        "thread_id": str(thread_id or "").strip(),
        "session_id": str(session_id or "").strip(),
        "submitted_turn_index": int(submitted_turn_index),
        "transaction_hash": str(transaction_hash or "").strip(),
        "admission_action_id": str(admission_action_id or "").strip(),
        "target_group": str(target_group or "").strip(),
        "provides_capabilities": sorted(
            str(value).strip()
            for value in provides_capabilities
            if str(value).strip()
        ),
        "source_hash": hashlib.sha256(
            str(source_evidence or "").strip().encode("utf-8")
        ).hexdigest(),
        "reviewer_evidence_hash": str(reviewer_evidence_hash or "").strip(),
        "action_contract_hash": action_registry_contract_hash(),
        "admission_contract_hash": admission_contract_hash(),
    }
    if not all(
        str(payload[key] or "").strip()
        for key in (
            "action_type",
            "thread_id",
            "session_id",
            "transaction_hash",
            "admission_action_id",
            "target_group",
            "reviewer_evidence_hash",
        )
    ) or not payload["provides_capabilities"]:
        raise ValueError("replacement intake admission receipt is incomplete")
    return {**payload, "receipt_id": _content_hash(payload)}


def validate_replacement_intake_admission_receipt(
    action: Mapping[str, Any],
    *,
    thread_id: str | None = None,
    session_id: str | None = None,
    submitted_turn_index: int | None = None,
) -> None:
    """Validate a reviewed replacement authorization against its action."""

    receipt = action.get("_replacement_intake_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("replacement intake requires an admission receipt")
    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    if (
        spec is None
        or not spec.incomplete_mutation_intake
        or not spec.provides_capabilities
    ):
        raise ValueError("replacement intake receipt belongs to an invalid action")
    expected = build_replacement_intake_admission_receipt(
        thread_id=str(receipt.get("thread_id") or ""),
        session_id=str(receipt.get("session_id") or ""),
        submitted_turn_index=int(receipt.get("submitted_turn_index") or 0),
        transaction_hash=str(receipt.get("transaction_hash") or ""),
        admission_action_id=str(receipt.get("admission_action_id") or ""),
        action_type=str(action.get("type") or ""),
        source_evidence=str(action.get("source_evidence") or ""),
        target_group=str(spec.target_group or ""),
        provides_capabilities=spec.provides_capabilities,
        reviewer_evidence_hash=str(
            receipt.get("reviewer_evidence_hash") or ""
        ),
    )
    if dict(receipt) != expected:
        raise ValueError("replacement intake admission receipt does not match the action")
    if str(action.get("_plan_transaction_hash") or "") != str(
        receipt.get("transaction_hash") or ""
    ):
        raise ValueError("replacement intake receipt transaction mismatch")
    if str(action.get("_admission_action_id") or "") != str(
        receipt.get("admission_action_id") or ""
    ):
        raise ValueError("replacement intake receipt action identity mismatch")
    if thread_id is not None and str(receipt.get("thread_id") or "") != str(
        thread_id
    ):
        raise ValueError("replacement intake receipt thread mismatch")
    if session_id is not None and str(receipt.get("session_id") or "") != str(
        session_id
    ):
        raise ValueError("replacement intake receipt session mismatch")
    if (
        submitted_turn_index is not None
        and int(receipt.get("submitted_turn_index") or 0)
        != int(submitted_turn_index)
    ):
        raise ValueError("replacement intake receipt belongs to another turn")


def replacement_intake_admission_verified(action: Mapping[str, Any]) -> bool:
    """Return whether one action carries a self-consistent trusted receipt."""

    try:
        validate_replacement_intake_admission_receipt(action)
    except (TypeError, ValueError):
        return False
    return True


def build_field_intake_admission_receipt(
    *,
    thread_id: str,
    session_id: str,
    submitted_turn_index: int,
    transaction_hash: str,
    admission_action_id: str,
    config_field: str,
    source_evidence: str,
    source_unit_id: str,
    source_unit_text: str,
    source_quote: str,
    semantic_unit_hash: str,
    reviewer_evidence_hash: str,
) -> dict[str, Any]:
    """Mint one Harness-owned receipt for a value-less field edit."""

    from agent.workflows.group_registry import (
        group_for_field,
        group_registry_contract_hash,
        reconfiguration_question_for_field,
    )

    field = str(config_field or "").strip()
    source = str(source_evidence or "").strip()
    payload = {
        "version": SEMANTIC_ADMISSION_RECEIPT_VERSION,
        "action_type": "request_config_field_input",
        "thread_id": str(thread_id or "").strip(),
        "session_id": str(session_id or "").strip(),
        "submitted_turn_index": int(submitted_turn_index),
        "transaction_hash": str(transaction_hash or "").strip(),
        "admission_action_id": str(admission_action_id or "").strip(),
        "target_group": group_for_field(field),
        "config_field": field,
        "question_id": reconfiguration_question_for_field(field),
        "source_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_unit_id": str(source_unit_id or "").strip(),
        "source_unit_text": str(source_unit_text or ""),
        "source_unit_hash": _content_hash(str(source_unit_text or "")),
        "source_quote": str(source_quote or ""),
        "semantic_unit_hash": str(semantic_unit_hash or "").strip(),
        "reviewer_evidence_hash": str(reviewer_evidence_hash or "").strip(),
        "registry_hash": group_registry_contract_hash(),
        "action_contract_hash": action_registry_contract_hash(),
        "admission_contract_hash": admission_contract_hash(),
        "replacement_supplied": False,
    }
    if not all(
        str(payload[key] or "").strip()
        for key in (
            "target_group",
            "thread_id",
            "session_id",
            "transaction_hash",
            "admission_action_id",
            "config_field",
            "question_id",
            "source_unit_id",
            "source_quote",
            "semantic_unit_hash",
            "reviewer_evidence_hash",
        )
    ):
        raise ValueError("field intake admission receipt is incomplete")
    if payload["source_quote"] not in str(source_unit_text or ""):
        raise ValueError("field intake source quote is outside its semantic unit")
    return {**payload, "receipt_id": _content_hash(payload)}


def build_proposal_field_receipt(
    *,
    thread_id: str,
    session_id: str,
    submitted_turn_index: int,
    transaction_hash: str,
    admission_action_id: str,
    config_field: str,
    canonical_value: Any,
    source_unit_id: str,
    source_unit_text: str,
    source_quote: str,
) -> dict[str, Any]:
    """Mint one Harness-owned receipt for one source-grounded config value."""

    from agent.workflows.group_registry import group_for_field, group_registry_contract_hash

    field = str(config_field or "").strip().upper()
    unit_text = str(source_unit_text or "")
    quote = str(source_quote or "")
    payload = {
        "version": SEMANTIC_ADMISSION_RECEIPT_VERSION,
        "action_type": "propose_config_values",
        "thread_id": str(thread_id or "").strip(),
        "session_id": str(session_id or "").strip(),
        "submitted_turn_index": int(submitted_turn_index),
        "transaction_hash": str(transaction_hash or "").strip(),
        "admission_action_id": str(admission_action_id or "").strip(),
        "config_field": field,
        "canonical_value": canonical_value,
        "canonical_value_hash": _content_hash(canonical_value),
        "target_group": group_for_field(field),
        "source_unit_id": str(source_unit_id or "").strip(),
        "source_unit_text": unit_text,
        "source_unit_hash": _content_hash(unit_text),
        "source_quote": quote,
        "source_quote_hash": _content_hash(quote),
        "registry_hash": group_registry_contract_hash(),
        "action_contract_hash": action_registry_contract_hash(),
        "admission_contract_hash": admission_contract_hash(),
    }
    if not all(str(payload[key] or "").strip() for key in (
        "thread_id", "session_id", "transaction_hash", "admission_action_id",
        "config_field", "source_unit_id", "source_quote",
    )):
        raise ValueError("proposal field receipt is incomplete")
    if quote not in unit_text:
        raise ValueError("proposal source quote is outside its semantic unit")
    return {**payload, "receipt_id": _content_hash(payload)}


def validate_field_intake_admission_receipt(
    action: Mapping[str, Any],
    *,
    thread_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Validate a trusted receipt against its current action and registry."""

    receipt = action.get("_semantic_admission_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("request_config_field_input requires a semantic admission receipt")
    expected = build_field_intake_admission_receipt(
        thread_id=str(receipt.get("thread_id") or ""),
        session_id=str(receipt.get("session_id") or ""),
        submitted_turn_index=int(receipt.get("submitted_turn_index") or 0),
        transaction_hash=str(receipt.get("transaction_hash") or ""),
        admission_action_id=str(receipt.get("admission_action_id") or ""),
        config_field=str(action.get("config_field") or ""),
        source_evidence=str(action.get("source_evidence") or ""),
        source_unit_id=str(receipt.get("source_unit_id") or ""),
        source_unit_text=str(receipt.get("source_unit_text") or ""),
        source_quote=str(receipt.get("source_quote") or ""),
        semantic_unit_hash=str(receipt.get("semantic_unit_hash") or ""),
        reviewer_evidence_hash=str(receipt.get("reviewer_evidence_hash") or ""),
    )
    if dict(receipt) != expected:
        raise ValueError("field intake semantic admission receipt does not match the action")
    if str(action.get("_plan_transaction_hash") or "") != str(receipt.get("transaction_hash") or ""):
        raise ValueError("field intake receipt transaction mismatch")
    if str(action.get("_admission_action_id") or "") != str(receipt.get("admission_action_id") or ""):
        raise ValueError("field intake receipt action identity mismatch")
    if thread_id is not None and str(receipt.get("thread_id") or "") != str(thread_id):
        raise ValueError("field intake receipt thread mismatch")
    if session_id is not None and str(receipt.get("session_id") or "") != str(session_id):
        raise ValueError("field intake receipt session mismatch")


def validate_proposal_field_receipts(
    action: Mapping[str, Any],
    *,
    thread_id: str | None = None,
    session_id: str | None = None,
    submitted_turn_index: int | None = None,
) -> None:
    """Validate every proposed field against its immutable source receipt."""

    values = action.get("config_values")
    receipts = action.get("_proposal_field_receipts")
    if not isinstance(values, Mapping) or not values:
        raise ValueError("propose_config_values requires non-empty config_values")
    if not isinstance(receipts, Mapping) or set(receipts) != set(values):
        raise ValueError("propose_config_values requires one receipt per field")
    transaction_hashes = {
        str(value)
        for value in action.get("_proposal_transaction_hashes") or []
        if str(value)
    }
    if not transaction_hashes:
        transaction_hashes = {str(action.get("_plan_transaction_hash") or "")}
    single_transaction = len(transaction_hashes) == 1
    from agent.workflows.group_registry import group_registry_contract_hash

    for field, value in values.items():
        receipt = receipts.get(field)
        if not isinstance(receipt, Mapping):
            raise ValueError(f"proposal field receipt is missing: {field}")
        unsigned = {key: item for key, item in receipt.items() if key != "receipt_id"}
        if receipt.get("receipt_id") != _content_hash(unsigned):
            raise ValueError(f"proposal field receipt hash mismatch: {field}")
        if str(receipt.get("action_type") or "") != "propose_config_values":
            raise ValueError(f"proposal field receipt action mismatch: {field}")
        if str(receipt.get("config_field") or "") != str(field):
            raise ValueError(f"proposal field receipt field mismatch: {field}")
        if receipt.get("canonical_value") != value or receipt.get("canonical_value_hash") != _content_hash(value):
            raise ValueError(f"proposal field receipt value mismatch: {field}")
        source_unit_text = str(receipt.get("source_unit_text") or "")
        source_quote = str(receipt.get("source_quote") or "")
        if (
            not source_quote
            or source_quote not in source_unit_text
            or receipt.get("source_unit_hash") != _content_hash(source_unit_text)
            or receipt.get("source_quote_hash") != _content_hash(source_quote)
        ):
            raise ValueError(f"proposal field receipt source mismatch: {field}")
        if str(receipt.get("transaction_hash") or "") not in transaction_hashes:
            raise ValueError(f"proposal field receipt transaction mismatch: {field}")
        if (
            single_transaction
            and str(receipt.get("admission_action_id") or "")
            != str(action.get("_admission_action_id") or "")
        ):
            raise ValueError(f"proposal field receipt action identity mismatch: {field}")
        if receipt.get("registry_hash") != group_registry_contract_hash():
            raise ValueError(f"proposal field receipt registry mismatch: {field}")
        if receipt.get("action_contract_hash") != action_registry_contract_hash():
            raise ValueError(f"proposal field receipt action-contract mismatch: {field}")
        if receipt.get("admission_contract_hash") != admission_contract_hash():
            raise ValueError(f"proposal field receipt admission-contract mismatch: {field}")
        if thread_id is not None and str(receipt.get("thread_id") or "") != str(thread_id):
            raise ValueError(f"proposal field receipt thread mismatch: {field}")
        if session_id is not None and str(receipt.get("session_id") or "") != str(session_id):
            raise ValueError(f"proposal field receipt session mismatch: {field}")
        if submitted_turn_index is not None and int(receipt.get("submitted_turn_index") or 0) != int(submitted_turn_index):
            raise ValueError(f"proposal field receipt turn mismatch: {field}")
ACTION_ARGUMENT_SCHEMAS: dict[str, Mapping[str, Any]] = {
    "adapter_family": {"type": "string", "minLength": 1},
    "answer": {},
    "canonical_chain_name": {"type": "string", "minLength": 1},
    "catalog_command": {
        "type": "string",
        "enum": [
            "enter",
            "set_endpoint",
            "set_method",
            "append_evidence",
            "keep_current_method",
            "replace_current_method",
            "finish",
        ],
    },
    "chain_candidates": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
    "chain_exists": {"type": "boolean"},
    "chain_text": {"type": "string", "minLength": 1},
    "clauses": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
    "config_values": {"type": "object"},
    "config_field": {"type": "string", "minLength": 1},
    "conflicts": {"type": "array"},
    "evidence": {},
    "evidence_summary": {"type": "string", "minLength": 1},
    "finish_methods": {"type": "boolean"},
    "failure_record": {"type": "object", "minProperties": 1},
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
    "revision": {"type": "integer", "minimum": 1},
    "draft_id": {"type": "string", "minLength": 1},
    "scope_id": {"type": "string", "minLength": 1},
    "owner_kind": {
        "type": "string",
        "enum": ["semantic_draft", "durable_state"],
    },
    "owner_id": {"type": "string", "minLength": 1},
    "owner_revision": {"type": "integer", "minimum": 0},
    "atom_id": {"type": "string", "minLength": 1},
    "resolution": {"type": "string", "minLength": 1},
    "reference": {
        "type": "string",
        "pattern": r"^semantic-secret:[A-Za-z0-9_-]+$",
    },
    "secret_value": {"type": "string", "minLength": 1, "maxLength": 65536},
    "value_hash": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
    "rpc_endpoint": {"type": "string", "minLength": 1},
    "rpc_method": {"type": "string", "minLength": 1},
    "rpc_mode": {"type": "string", "enum": ["single", "mixed"]},
    "rpc_schema_evidence": {},
    "rpc_weights": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 1}, "minProperties": 1},
    "selected_value": {},
    "source_evidence": {"type": "string", "minLength": 1},
    "source_format": {
        "type": "string",
        "enum": ["prose", "yaml", "json", "env", "mixed"],
    },
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

CANDIDATE_BINDING_SCHEMAS: dict[str, Mapping[str, Mapping[str, Any]]] = {
    "rpc_workload_command": {
        "rpc_weights": {
            "mapping_mode": "whole_value",
        },
    },
    "set_qps_override": {
        "qps_overrides": {
            "mapping_mode": "required_key",
            "mapping_keys": ("INITIAL_QPS", "MAX_QPS", "QPS_STEP", "DURATION"),
        },
    },
    "set_sync_observe_options": {
        "sync_observe_duration_seconds": {
            "mapping_mode": "scalar",
        },
    },
}


def validate_candidate_binding_contract(raw: Mapping[str, Any]) -> dict[str, str]:
    """Validate one question-to-action candidate binding against the registry."""

    binding = {
        "type": str(raw.get("type") or "").strip(),
        "value_argument": str(raw.get("value_argument") or "").strip(),
    }
    mapping_key = str(raw.get("mapping_key") or "").strip()
    action_type = binding["type"]
    value_argument = binding["value_argument"]
    spec = ACTION_BY_TYPE.get(action_type)
    if spec is None:
        raise ValueError(
            f"unknown candidate binding action type: {action_type or '<missing>'}"
        )
    if not value_argument or value_argument not in spec.allowed_arguments:
        raise ValueError(
            f"candidate binding {action_type} requires an allowed value_argument"
        )
    binding_schema = (
        CANDIDATE_BINDING_SCHEMAS.get(action_type, {}).get(value_argument)
    )
    if binding_schema is None:
        raise ValueError(
            f"candidate binding {action_type}.{value_argument} is not "
            "registered as a business value"
        )
    mapping_mode = str(binding_schema.get("mapping_mode") or "")
    if mapping_mode == "required_key":
        allowed_keys = {
            str(value) for value in binding_schema.get("mapping_keys") or ()
        }
        if not mapping_key:
            raise ValueError(
                f"candidate binding {action_type}.{value_argument} "
                "requires mapping_key"
            )
        if mapping_key not in allowed_keys:
            raise ValueError(
                f"candidate binding {action_type}.{value_argument} has "
                f"invalid mapping_key: {mapping_key}"
            )
    elif mapping_key:
        raise ValueError(
            f"candidate binding {action_type}.{value_argument} "
            "cannot declare mapping_key"
        )
    if mapping_key:
        binding["mapping_key"] = mapping_key
    return binding


def normalize_current_action_envelope(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the current flat action wire contract without legacy repair."""

    action = dict(raw)
    if "arguments" in action:
        raise ValueError("arguments.v1 is retired for current-turn actions")
    if "intent" in action:
        raise ValueError("intent.v1 is retired for current-turn actions")
    return action


def lower_empty_entry_action_to_registered_intake(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Lower one value-less entry action to its unique typed intake.

    Stage B may identify the correct workflow dimension before it has a
    concrete business value. The registry already declares both the concrete
    entry action and the typed intake that collects the missing value. This
    function connects those two contracts without guessing a value or changing
    an action that supplied one.
    """

    action = normalize_current_action_envelope(raw)
    spec = ACTION_BY_TYPE.get(str(action.get("type") or "").strip())
    if (
        spec is None
        or not spec.entry_intake
        or not spec.target_group
        or not spec.entry_intake_value_arguments
        or any(
            action.get(argument) not in (None, "", [], {})
            for argument in spec.entry_intake_value_arguments
        )
    ):
        return action
    intake_specs = [
        candidate
        for candidate in ACTION_SPECS
        if candidate.owner == spec.owner
        and candidate.target_group == spec.target_group
        and candidate.incomplete_mutation_intake
        and set(candidate.required_arguments).issubset({"source_evidence"})
    ]
    if len(intake_specs) != 1:
        return action
    intake = intake_specs[0]
    replacement: dict[str, Any] = {"type": intake.action_type}
    for key in ACTION_METADATA_FIELDS:
        if key != "type" and key in action:
            replacement[key] = action[key]
    if (
        "source_evidence" in intake.allowed_arguments
        and action.get("source_evidence") not in (None, "")
    ):
        replacement["source_evidence"] = action["source_evidence"]
    return replacement


def validate_action_contract(
    raw: dict[str, Any],
    *,
    trusted_metadata: bool = False,
) -> dict[str, Any]:
    """Normalize and validate one model action against its ActionSpec."""

    action = normalize_current_action_envelope(raw)
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
        if answer_pending_representation_conflict(action):
            raise ValueError(
                "answer_pending has conflicting answer and selected_value representations"
            )
    if spec.validator is not None:
        spec.validator(action)
    if (
        trusted_metadata
        and action_type == "request_config_field_input"
        and "_semantic_admission_receipt" in action
    ):
        validate_field_intake_admission_receipt(action)
    if trusted_metadata and action_type == "propose_config_values":
        validate_proposal_field_receipts(action)
    return action


def validate_action_transaction_contract(
    actions: list[Mapping[str, Any]],
) -> None:
    """Reject plans that would clarify and partially commit the same turn."""

    action_types = [
        str(action.get("type") or "").strip()
        for action in actions
        if isinstance(action, Mapping)
    ]
    if "clarify_unresolved" in action_types and any(
        action_type != "clarify_unresolved"
        for action_type in action_types
    ):
        raise ValueError(
            "clarify_unresolved is a whole-turn transaction barrier and cannot "
            "coexist with another action"
        )


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


def semantic_grounding_arguments(action: Mapping[str, Any]) -> tuple[str, ...]:
    """Return source-grounded operation values declared by the registry.

    Specs explicitly distinguish user-selected values from internal operation
    codes. This prevents planner commands such as ``set_endpoint`` from being
    treated as words the user had to type while keeping concrete workflow
    selections and RPC payload values source-grounded.
    """

    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    if spec is None or spec.effect == "read_only":
        return ()
    return tuple(
        argument
        for argument in spec.semantic_value_grounding_arguments
        if argument in action
    )


def semantic_grounding_value_projection(
    value: Any,
    schema: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Project one schema-owned argument into independently grounded values."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return ()
    if schema.get("type") == "array":
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            return (value,)
        item_schema = schema.get("items")
        if not isinstance(item_schema, Mapping):
            item_schema = {}
        return tuple(
            projected
            for item in value
            for projected in semantic_grounding_value_projection(
                item,
                item_schema,
            )
        )
    return (value,)


def semantic_grounding_values(action: Mapping[str, Any]) -> tuple[Any, ...]:
    """Project registry-declared grounding arguments into concrete values.

    Array arguments represent several independently grounded values, whereas
    structured objects are one value whose identity is the complete object.
    Keeping this projection beside the argument schemas prevents semantic
    consumers from inventing their own string representations for containers.
    """

    return tuple(
        value
        for argument in semantic_grounding_arguments(action)
        for value in semantic_grounding_value_projection(
            action.get(argument),
            ACTION_ARGUMENT_SCHEMAS.get(argument) or {},
        )
    )


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


def action_merge_key(action: dict[str, Any]) -> tuple[Any, ...]:
    """Return the registry-declared semantic identity used across planner passes."""

    action_type = str(action.get("type") or "")
    spec = ACTION_BY_TYPE.get(action_type)
    if not spec or not spec.merge_identity:
        return (action_type,)
    return (action_type,) + tuple(
        str(action.get(argument) or "").strip().casefold()
        for argument in spec.merge_identity
    )


def merge_semantic_actions(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Apply the action registry's durable merge contract.

    Actions without an explicit merge declaration are replaced wholesale by
    the incoming reviewed action. Registry-declared mapping fields merge by key
    and sequence fields retain unique evidence in encounter order.
    """

    spec = ACTION_BY_TYPE.get(str(incoming.get("type") or ""))
    if (
        spec is None
        or not spec.merge_mapping_fields
        and not spec.merge_sequence_fields
    ):
        return dict(incoming)
    merged = dict(incoming)
    for field_name in spec.merge_mapping_fields:
        values: dict[str, Any] = {}
        old_value = existing.get(field_name)
        new_value = incoming.get(field_name)
        if isinstance(old_value, dict):
            values.update(old_value)
        if isinstance(new_value, dict):
            values.update(new_value)
        merged[field_name] = values
    if str(incoming.get("type") or "") == "propose_config_values":
        validate_proposal_field_receipts(existing)
        validate_proposal_field_receipts(incoming)
        receipts: dict[str, Any] = {}
        receipts.update(dict(existing.get("_proposal_field_receipts") or {}))
        receipts.update(dict(incoming.get("_proposal_field_receipts") or {}))
        merged["_proposal_field_receipts"] = receipts
        transaction_hashes: list[str] = []
        for source in (existing, incoming):
            values = source.get("_proposal_transaction_hashes") or [source.get("_plan_transaction_hash")]
            for value in values:
                transaction_hash = str(value or "").strip()
                if transaction_hash and transaction_hash not in transaction_hashes:
                    transaction_hashes.append(transaction_hash)
        merged["_proposal_transaction_hashes"] = transaction_hashes
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


def _entry_intake_probe_value(schema: Mapping[str, Any]) -> Any:
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    value_type = str(schema.get("type") or "")
    if value_type == "boolean":
        return True
    if value_type == "integer":
        return max(1, int(schema.get("minimum") or 1))
    if value_type == "number":
        return max(1.0, float(schema.get("minimum") or 1.0))
    if value_type == "array":
        item_schema = schema.get("items")
        return [
            _entry_intake_probe_value(
                item_schema if isinstance(item_schema, Mapping) else {"type": "string"}
            )
        ]
    if value_type == "object":
        return {}
    return "entry-value"


def _entry_intake_probe_tokens(action: Mapping[str, Any]) -> tuple[str, ...]:
    tokens: list[str] = []
    for key, value in action.items():
        if key in {"type", "source_evidence"}:
            continue
        values = value if isinstance(value, list) else [value]
        tokens.extend(str(item) for item in values if str(item))
    return tuple(tokens) or ("entry",)


def validate_action_registry() -> None:
    from agent.workflows.group_registry import GROUP_SPEC_BY_NAME

    names = [spec.action_type for spec in ACTION_SPECS]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate Harness action type")
    structured_alias_owners: dict[str, str] = {}
    for spec in ACTION_SPECS:
        undeclared_required = set(spec.required_arguments) - set(spec.allowed_arguments)
        if undeclared_required:
            raise RuntimeError(
                f"required arguments must be allowed for {spec.action_type}: "
                f"{', '.join(sorted(undeclared_required))}"
            )
        if len(spec.allowed_arguments) != len(set(spec.allowed_arguments)):
            raise RuntimeError(f"duplicate allowed action argument: {spec.action_type}")
        if len(spec.compiler_groups) != len(set(spec.compiler_groups)):
            raise RuntimeError(f"duplicate compiler group: {spec.action_type}")
        if len(spec.semantic_operations) != len(set(spec.semantic_operations)):
            raise RuntimeError(f"duplicate semantic operation: {spec.action_type}")
        if not spec.semantic_operations:
            raise RuntimeError(f"action has no semantic operation: {spec.action_type}")
        unknown_compiler_groups = set(spec.compiler_groups) - set(GROUP_SPEC_BY_NAME)
        unknown_semantic_operations = (
            set(spec.semantic_operations) - SEMANTIC_OPERATIONS
        )
        if unknown_compiler_groups or unknown_semantic_operations:
            raise RuntimeError(
                f"invalid compiler authority for {spec.action_type}: "
                f"groups={sorted(unknown_compiler_groups)}, "
                f"operations={sorted(unknown_semantic_operations)}"
            )
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
        if spec.interrupts_pending and not spec.crosses_pending_barrier:
            raise RuntimeError(
                "pending interruption requires barrier crossing authority: "
                f"{spec.action_type}"
            )
        if spec.replaces_deferred_queue and spec.lifetime == "turn_local":
            raise RuntimeError(
                "turn-local action cannot replace the durable queue: "
                f"{spec.action_type}"
            )
        if spec.target_field_argument and spec.target_field_argument not in spec.required_arguments:
            raise RuntimeError(
                f"dynamic target field must be a required argument: {spec.action_type}"
            )
        if spec.requires_specific_change and (
            (not spec.target_group and not spec.target_field_argument)
            or "source_evidence" not in spec.required_arguments
        ):
            raise RuntimeError(
                "actions requiring a specific change need a target group and required "
                f"source evidence: {spec.action_type}"
            )
        if spec.incomplete_mutation_intake and (
            (not spec.target_group and not spec.target_field_argument)
            or "source_evidence" not in spec.allowed_arguments
        ):
            raise RuntimeError(
                "incomplete mutation intake actions need a target group and source evidence: "
                f"{spec.action_type}"
            )
        if spec.incomplete_read_intake and (
            spec.effect != "read_only"
            or spec.lifetime != "turn_local"
            or bool(spec.required_arguments)
            or bool(spec.required_state_path)
        ):
            raise RuntimeError(
                "incomplete read intake actions must be turn-local, read-only, "
                "and require no arguments or prior lifecycle state: "
                f"{spec.action_type}"
            )
        if spec.entry_intake:
            fixed_arguments = dict(spec.entry_intake_fixed_arguments)
            value_arguments = set(spec.entry_intake_value_arguments)
            if not spec.target_group or not spec.entry_intake_purpose.strip():
                raise RuntimeError(
                    f"entry intake requires a target group and purpose: {spec.action_type}"
                )
            if set(fixed_arguments) - set(spec.allowed_arguments):
                raise RuntimeError(
                    f"entry intake fixed arguments must be allowed: {spec.action_type}"
                )
            if value_arguments - set(spec.allowed_arguments):
                raise RuntimeError(
                    f"entry intake value arguments must be allowed: {spec.action_type}"
                )
            if set(fixed_arguments).intersection(value_arguments):
                raise RuntimeError(
                    f"entry intake argument ownership overlaps: {spec.action_type}"
                )
            uncovered_required = (
                set(spec.required_arguments)
                - set(fixed_arguments)
                - value_arguments
                - {"source_evidence"}
            )
            if uncovered_required:
                raise RuntimeError(
                    "entry intake metadata does not cover required arguments for "
                    f"{spec.action_type}: {sorted(uncovered_required)}"
                )
            alternatives = spec.entry_intake_value_arguments or ("",)
            for value_argument in alternatives:
                probe: dict[str, Any] = {
                    "type": spec.action_type,
                    **fixed_arguments,
                }
                if value_argument:
                    probe[value_argument] = _entry_intake_probe_value(
                        ACTION_ARGUMENT_SCHEMAS[value_argument]
                    )
                if "source_evidence" in spec.required_arguments:
                    probe["source_evidence"] = " ".join(
                        _entry_intake_probe_tokens(probe)
                    )
                try:
                    validate_action_contract(probe)
                except ValueError as exc:
                    raise RuntimeError(
                        f"invalid entry intake metadata for {spec.action_type}: {exc}"
                    ) from exc
        elif (
            spec.entry_intake_purpose
            or spec.entry_intake_fixed_arguments
            or spec.entry_intake_value_arguments
        ):
            raise RuntimeError(
                f"entry intake metadata requires entry_intake=true: {spec.action_type}"
            )
        for intake in spec.structured_intake:
            alias = intake.alias.strip()
            normalized_alias = alias.casefold()
            if not alias:
                raise RuntimeError(
                    f"structured intake alias cannot be empty: {spec.action_type}"
                )
            previous_owner = structured_alias_owners.get(normalized_alias)
            if previous_owner is not None:
                raise RuntimeError(
                    "structured intake aliases must be globally unique: "
                    f"{alias} belongs to {previous_owner} and {spec.action_type}"
                )
            structured_alias_owners[normalized_alias] = spec.action_type
            if intake.value_semantics not in STRUCTURED_INTAKE_VALUE_SEMANTICS:
                raise RuntimeError(
                    "invalid structured intake value semantics for "
                    f"{spec.action_type}.{alias}: {intake.value_semantics}"
                )
            fixed_argument_names = [key for key, _value in intake.fixed_arguments]
            if len(fixed_argument_names) != len(set(fixed_argument_names)):
                raise RuntimeError(
                    f"duplicate structured intake fixed argument: {spec.action_type}.{alias}"
                )
            undeclared_fixed = set(fixed_argument_names) - set(spec.allowed_arguments)
            if undeclared_fixed:
                raise RuntimeError(
                    "structured intake fixed arguments must be allowed for "
                    f"{spec.action_type}.{alias}: {sorted(undeclared_fixed)}"
                )
            for argument, value in intake.fixed_arguments:
                try:
                    _validate_argument_value(
                        value,
                        ACTION_ARGUMENT_SCHEMAS[argument],
                        f"{spec.action_type}.{argument}",
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        "invalid structured intake fixed argument for "
                        f"{spec.action_type}.{alias}: {exc}"
                    ) from exc
            value_argument = intake.value_argument.strip()
            if value_argument and value_argument not in spec.allowed_arguments:
                raise RuntimeError(
                    "structured intake value argument must be allowed for "
                    f"{spec.action_type}.{alias}: {value_argument}"
                )
            if value_argument and value_argument in fixed_argument_names:
                raise RuntimeError(
                    "structured intake value argument overlaps fixed arguments: "
                    f"{spec.action_type}.{alias}.{value_argument}"
                )
            if (intake.value_semantics == "direct_value") != bool(value_argument):
                raise RuntimeError(
                    "direct structured intake semantics require exactly one value argument: "
                    f"{spec.action_type}.{alias}"
                )
            uncovered_required = (
                set(spec.required_arguments)
                - set(fixed_argument_names)
                - ({value_argument} if value_argument else set())
                - {"source_evidence"}
            )
            if uncovered_required:
                raise RuntimeError(
                    "structured intake metadata does not cover required arguments for "
                    f"{spec.action_type}.{alias}: {sorted(uncovered_required)}"
                )
            probe: dict[str, Any] = {
                "type": spec.action_type,
                **dict(intake.fixed_arguments),
            }
            if value_argument:
                probe[value_argument] = _entry_intake_probe_value(
                    ACTION_ARGUMENT_SCHEMAS[value_argument]
                )
            if "source_evidence" in spec.required_arguments:
                probe["source_evidence"] = alias
            try:
                validate_action_contract(probe)
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid structured intake contract for {spec.action_type}.{alias}: {exc}"
                ) from exc
        if bool(spec.required_state_path) != bool(spec.required_state_values):
            raise RuntimeError(
                f"state lifecycle requirements must declare both path and values: {spec.action_type}"
            )
        if set(spec.exact_source_value_arguments) - set(spec.allowed_arguments):
            raise RuntimeError(
                f"exact source arguments must be allowed: {spec.action_type}"
            )
        if set(spec.open_identity_grounding_arguments) - set(
            spec.semantic_value_grounding_arguments
        ):
            raise RuntimeError(
                "open identity arguments must be semantic grounding arguments: "
                f"{spec.action_type}"
            )
    intake_groups = [
        spec.target_group
        for spec in ACTION_SPECS
        if spec.incomplete_mutation_intake and spec.target_group
    ]
    if len(intake_groups) != len(set(intake_groups)):
        raise RuntimeError("only one incomplete mutation intake action is allowed per group")
    semantic_argument_groups: dict[str, set[str]] = {}
    for spec in ACTION_SPECS:
        if not spec.target_group:
            continue
        for argument in spec.semantic_value_grounding_arguments:
            semantic_argument_groups.setdefault(argument, set()).add(
                spec.target_group
            )
    ambiguous_semantic_arguments = {
        argument: sorted(groups)
        for argument, groups in semantic_argument_groups.items()
        if len(groups) > 1
    }
    if ambiguous_semantic_arguments:
        raise RuntimeError(
            "semantic value arguments cannot belong to several workflow groups: "
            f"{ambiguous_semantic_arguments}"
        )
    for spec in ACTION_SPECS:
        if not spec.semantic_value_representative:
            continue
        owns_grounded_value = bool(spec.semantic_value_grounding_arguments)
        owns_chain_identity = (
            spec.target_group == "chain_identity"
            and "chain_text" in spec.entry_intake_value_arguments
        )
        if not owns_grounded_value and not owns_chain_identity:
            raise RuntimeError(
                "semantic value representative has no declared value domain: "
                f"{spec.action_type}"
            )
    registered_semantic_value_domains()


validate_action_registry()


def universal_semantic_operation_owners() -> dict[str, tuple[str, ...]]:
    """Return model-reachable owners for non-domain semantic operations."""

    owners = {
        operation: tuple(sorted({
            spec.owner
            for spec in ACTION_SPECS
            if not spec.internal_only
            and operation in spec.semantic_operations
        }))
        for operation in ROUTED_UNIVERSAL_SEMANTIC_OPERATIONS
    }
    missing = sorted(
        operation
        for operation, operation_owners in owners.items()
        if not operation_owners
    )
    if missing:
        raise RuntimeError(
            "routed universal semantic operations require a model-reachable "
            f"action owner: {missing}"
        )
    return owners


UNIVERSAL_SEMANTIC_OPERATION_OWNERS = universal_semantic_operation_owners()


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
