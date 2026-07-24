"""Typed contracts shared by the AnyChain Harness control plane.

The model may propose actions, but only these contracts can authorize state
transitions or user-visible choices.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field as dataclass_field, is_dataclass
from typing import Any, Literal, Mapping


ReturnPolicy = Literal["stay", "resume_interrupted", "fallback", "stop_after_response"]
CompletionStatus = Literal["unchanged", "in_progress", "completed", "blocked"]
RecoveryOperation = Literal["activate", "resolve"]
NavigationOperation = Literal["change_group", "go_back"]
WorkflowGoalOperation = Literal["enqueue", "remove_first"]
ActionEffect = Literal["pure", "read_only", "external"]
ActionEnvelopeStatus = Literal[
    "admitted",
    "selected",
    "prepared",
    "committed",
    "blocked",
    "failed",
]
SideEffectStatus = Literal["prepared", "invoking", "succeeded", "reused", "blocked", "failed"]
TurnReceiptStatus = Literal["prepared", "planned", "executing", "completed", "blocked", "failed"]
StatePath = tuple[str, ...]


DOMAIN_CONTROL_ROOTS = frozenset(
    {
        "active_group",
        "active_subgroup",
        "action_errors",
        "action_queue",
        "applied_action_ids",
        "audit_events",
        "completed_actions",
        "control",
        "current_action",
        "group_history",
        "group_states",
        "interruption_stack",
        "invalidated_groups",
        "pending_question",
        "proposed_actions",
        "turn_context",
        "visible_response",
    }
)


@dataclass(frozen=True)
class DeltaWrite:
    """One immutable write to a leaf or complete value in graph state."""

    path: StatePath
    value: Any


@dataclass(frozen=True)
class StateDelta:
    """A recursive state change that can be owner-validated before commit."""

    writes: tuple[DeltaWrite, ...] = ()
    deletes: tuple[StatePath, ...] = ()

    @classmethod
    def between(
        cls,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        *,
        ignored_roots: frozenset[str] = DOMAIN_CONTROL_ROOTS,
    ) -> "StateDelta":
        writes: list[DeltaWrite] = []
        deletes: list[StatePath] = []
        _diff_mapping(before, after, (), writes, deletes, ignored_roots)
        return cls(tuple(writes), tuple(deletes))

    @classmethod
    def set_values(cls, values: Mapping[str, Any]) -> "StateDelta":
        """Create recursive writes for new domain-owned values."""

        writes: list[DeltaWrite] = []
        for key, value in values.items():
            _flatten_new_value((str(key),), value, writes)
        return cls(tuple(writes))

    def is_empty(self) -> bool:
        return not self.writes and not self.deletes


@dataclass(frozen=True)
class RecoveryCommand:
    """A typed failure-recovery transition committed by the coordinator."""

    operation: RecoveryOperation
    record: Mapping[str, Any] = dataclass_field(default_factory=dict)
    validation_receipt: str = ""

    def __post_init__(self) -> None:
        if self.operation == "activate" and not self.record:
            raise ValueError("activate recovery command requires a failure record")
        if self.operation == "resolve" and not self.validation_receipt:
            raise ValueError("resolve recovery command requires a validation receipt")


@dataclass(frozen=True)
class CheckpointCommand:
    """Typed checkpoint mutation executed only by the Harness control plane."""

    command: Literal["reset", "retain_safe"]
    confirmed_config: Mapping[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class NavigationCommand:
    """A navigation request interpreted only by the atomic commit boundary."""

    operation: NavigationOperation
    target_group: str = ""
    origin_group: str = ""


@dataclass(frozen=True)
class FieldReconfigurationCommand:
    """Install a typed field-reconfiguration continuation at commit time."""

    group: str
    config_field: str
    prerequisite_question_id: str = ""


@dataclass(frozen=True)
class WorkflowGoalCommand:
    """Mutate the durable workflow-goal queue through one explicit operation."""

    operation: WorkflowGoalOperation
    goal: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation == "change_group" and not self.target_group:
            raise ValueError("change_group navigation requires target_group")


def _diff_mapping(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    prefix: StatePath,
    writes: list[DeltaWrite],
    deletes: list[StatePath],
    ignored_roots: frozenset[str],
) -> None:
    keys = set(before) | set(after)
    for raw_key in sorted(keys, key=str):
        key = str(raw_key)
        if not prefix and key in ignored_roots:
            continue
        path = (*prefix, key)
        if raw_key not in after:
            deletes.append(path)
            continue
        if raw_key not in before:
            _flatten_new_value(path, after[raw_key], writes)
            continue
        old_value = before[raw_key]
        new_value = after[raw_key]
        if isinstance(old_value, Mapping) and isinstance(new_value, Mapping):
            _diff_mapping(old_value, new_value, path, writes, deletes, frozenset())
        elif old_value != new_value:
            writes.append(DeltaWrite(path, deepcopy(new_value)))


def _flatten_new_value(path: StatePath, value: Any, writes: list[DeltaWrite]) -> None:
    if isinstance(value, Mapping) and value:
        for raw_key in sorted(value, key=str):
            _flatten_new_value((*path, str(raw_key)), value[raw_key], writes)
        return
    writes.append(DeltaWrite(path, deepcopy(value)))


@dataclass(frozen=True)
class ActionProposal:
    """A model-proposed action before product validation."""

    action_id: str
    action_type: str
    arguments: Mapping[str, Any] = dataclass_field(default_factory=dict)
    confidence: Literal["low", "medium", "high"] = "low"
    reason: str = ""


@dataclass(frozen=True)
class OptionContract:
    """Executable behavior attached to one rendered option."""

    option_id: str
    value: Any
    action: ActionProposal
    expected_patch: Mapping[str, Any]
    description: str = ""
    completion_effect: str = ""
    manual_entry: bool = False
    return_policy: ReturnPolicy = "fallback"
    preconditions: Mapping[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class QuestionContract:
    """A blocking question owned by exactly one workflow group."""

    question_id: str
    group: str
    kind: str
    prompt_key: str
    field: str = ""
    options: tuple[OptionContract, ...] = ()
    manual_input_allowed: bool = False
    validation: Mapping[str, Any] = dataclass_field(default_factory=dict)
    requires_capabilities: tuple[str, ...] = ()
    help_text: str = ""
    completion_effect: str = ""

    def __post_init__(self) -> None:
        option_ids = [option.option_id for option in self.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError(f"duplicate option ids for {self.question_id}")
        if self.kind in {"numbered_choice", "yes_no"} and not self.options:
            raise ValueError(f"{self.question_id} requires executable options")
        for option in self.options:
            if not option.action.action_type:
                raise ValueError(f"{self.question_id}/{option.option_id} has no action")
            delegated_transition = (
                option.action.action_type != "answer_pending"
            )
            if (
                not option.expected_patch
                and option.return_policy != "stop_after_response"
                and not delegated_transition
            ):
                raise ValueError(f"{self.question_id}/{option.option_id} has no postcondition")


@dataclass(frozen=True)
class HandlerResult:
    """The only result a domain handler may return to the coordinator."""

    delta: StateDelta = dataclass_field(default_factory=StateDelta)
    control_receipts: tuple[Mapping[str, Any], ...] = ()
    consumed_action_ids: tuple[str, ...] = ()
    invalidated_groups: tuple[str, ...] = ()
    reconfigured_groups: tuple[str, ...] = ()
    invalidated_fields: tuple[str, ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()
    action_errors: tuple[Mapping[str, Any], ...] = ()
    visible_result: str = ""
    visible_results: tuple[str, ...] = ()
    pending_question: QuestionContract | Mapping[str, Any] | None = None
    clear_pending: bool = False
    next_group: str = ""
    navigation_resume_group: str = ""
    followup_actions: tuple[Mapping[str, Any], ...] = ()
    checkpoint_command: CheckpointCommand | None = None
    recovery_command: RecoveryCommand | None = None
    navigation_command: NavigationCommand | None = None
    field_reconfiguration_command: FieldReconfigurationCommand | None = None
    workflow_goal_command: WorkflowGoalCommand | None = None
    completion: CompletionStatus = "unchanged"
    blocker: str = ""
    stop_after_response: bool = False


@dataclass(frozen=True)
class ActionEnvelope:
    """Harness-owned provenance around one validated model action."""

    action_id: str
    action_type: str
    owner: str
    target_group: str
    arguments: Mapping[str, Any] = dataclass_field(default_factory=dict)
    confidence: Literal["low", "medium", "high"] = "medium"
    reason: str = ""
    source_unit_ids: tuple[str, ...] = ()
    source_evidence_hash: str = ""
    semantic_order: int = 0
    execution_order: int = 0
    submitted_turn_index: int = 0
    origin_group: str = ""
    origin_text: str = ""
    plan_scope: str = ""
    admission_metadata: Mapping[str, Any] = dataclass_field(default_factory=dict)
    effect_kind: ActionEffect = "pure"
    idempotency_key: str = ""
    status: ActionEnvelopeStatus = "admitted"


@dataclass(frozen=True)
class PendingDomainResult:
    """One prepared owner result waiting for the graph commit node."""

    action: ActionEnvelope
    result: HandlerResult
    prepared_state_hash: str


@dataclass(frozen=True)
class SideEffectIntent:
    """Durable authorization written before invoking an external operation."""

    intent_id: str
    turn_id: str
    action_id: str
    operation: str
    idempotency_key: str
    request: Mapping[str, Any]
    request_fingerprint: str
    expected_receipt_kind: str
    status: SideEffectStatus = "prepared"
    attempt_count: int = 0


@dataclass(frozen=True)
class SideEffectReceipt:
    """Durable result of one idempotent external operation."""

    receipt_id: str
    intent_id: str
    action_id: str
    status: SideEffectStatus
    idempotency_key: str
    result: Mapping[str, Any] = dataclass_field(default_factory=dict)
    job_id: str = ""
    execution_key: str = ""
    failure_code: str = ""
    retryable: bool = False


@dataclass(frozen=True)
class TurnReceipt:
    """Durable accounting for one complete user turn."""

    turn_id: str
    input_hash: str
    language: str
    input_shape: str
    clauses: tuple[Mapping[str, Any], ...] = ()
    semantic_units: tuple[Mapping[str, Any], ...] = ()
    pending_before: Mapping[str, Any] = dataclass_field(default_factory=dict)
    admitted_action_ids: tuple[str, ...] = ()
    semantic_order: tuple[str, ...] = ()
    execution_order: tuple[str, ...] = ()
    owner_bindings: Mapping[str, str] = dataclass_field(default_factory=dict)
    action_unit_bindings: Mapping[str, tuple[str, ...]] = dataclass_field(
        default_factory=dict
    )
    unit_action_bindings: Mapping[str, tuple[str, ...]] = dataclass_field(
        default_factory=dict
    )
    pending_candidate_verdicts: tuple[Mapping[str, Any], ...] = ()
    sibling_omission_checks: tuple[Mapping[str, Any], ...] = ()
    unresolved_units: tuple[str, ...] = ()
    pending_after: Mapping[str, Any] = dataclass_field(default_factory=dict)
    response_count: int = 0
    status: TurnReceiptStatus = "prepared"


def action_envelope_to_dict(envelope: ActionEnvelope) -> dict[str, Any]:
    return {
        "action_id": envelope.action_id,
        "action_type": envelope.action_type,
        "owner": envelope.owner,
        "target_group": envelope.target_group,
        "arguments": deepcopy(dict(envelope.arguments)),
        "confidence": envelope.confidence,
        "reason": envelope.reason,
        "source_unit_ids": list(envelope.source_unit_ids),
        "source_evidence_hash": envelope.source_evidence_hash,
        "semantic_order": envelope.semantic_order,
        "execution_order": envelope.execution_order,
        "submitted_turn_index": envelope.submitted_turn_index,
        "origin_group": envelope.origin_group,
        "origin_text": envelope.origin_text,
        "plan_scope": envelope.plan_scope,
        "admission_metadata": deepcopy(dict(envelope.admission_metadata)),
        "effect_kind": envelope.effect_kind,
        "idempotency_key": envelope.idempotency_key,
        "status": envelope.status,
    }


def action_envelope_from_dict(payload: Mapping[str, Any]) -> ActionEnvelope:
    return ActionEnvelope(
        action_id=str(payload.get("action_id") or ""),
        action_type=str(payload.get("action_type") or ""),
        owner=str(payload.get("owner") or ""),
        target_group=str(payload.get("target_group") or ""),
        arguments=deepcopy(dict(payload.get("arguments") or {})),
        confidence=str(payload.get("confidence") or "medium"),  # type: ignore[arg-type]
        reason=str(payload.get("reason") or ""),
        source_unit_ids=tuple(str(item) for item in payload.get("source_unit_ids") or ()),
        source_evidence_hash=str(payload.get("source_evidence_hash") or ""),
        semantic_order=int(payload.get("semantic_order") or 0),
        execution_order=int(payload.get("execution_order") or 0),
        submitted_turn_index=int(payload.get("submitted_turn_index") or 0),
        origin_group=str(payload.get("origin_group") or ""),
        origin_text=str(payload.get("origin_text") or ""),
        plan_scope=str(payload.get("plan_scope") or ""),
        admission_metadata=deepcopy(dict(payload.get("admission_metadata") or {})),
        effect_kind=str(payload.get("effect_kind") or "pure"),  # type: ignore[arg-type]
        idempotency_key=str(payload.get("idempotency_key") or ""),
        status=str(payload.get("status") or "admitted"),  # type: ignore[arg-type]
    )


def handler_result_to_dict(result: HandlerResult) -> dict[str, Any]:
    return {
        "delta": {
            "writes": [
                {"path": list(write.path), "value": deepcopy(write.value)}
                for write in result.delta.writes
            ],
            "deletes": [list(path) for path in result.delta.deletes],
        },
        "control_receipts": [
            deepcopy(dict(item)) for item in result.control_receipts
        ],
        "consumed_action_ids": list(result.consumed_action_ids),
        "invalidated_groups": list(result.invalidated_groups),
        "reconfigured_groups": list(result.reconfigured_groups),
        "invalidated_fields": list(result.invalidated_fields),
        "evidence": [deepcopy(dict(item)) for item in result.evidence],
        "action_errors": [deepcopy(dict(item)) for item in result.action_errors],
        "visible_result": result.visible_result,
        "visible_results": list(result.visible_results),
        "pending_question": (
            deepcopy(dict(result.pending_question))
            if isinstance(result.pending_question, Mapping)
            else (
                deepcopy(asdict(result.pending_question))
                if is_dataclass(result.pending_question)
                else deepcopy(dict(result.pending_question))
                if result.pending_question is not None
                else None
            )
        ),
        "clear_pending": result.clear_pending,
        "next_group": result.next_group,
        "navigation_resume_group": result.navigation_resume_group,
        "followup_actions": [deepcopy(dict(item)) for item in result.followup_actions],
        "checkpoint_command": (
            {
                "command": result.checkpoint_command.command,
                "confirmed_config": deepcopy(dict(result.checkpoint_command.confirmed_config)),
            }
            if result.checkpoint_command
            else None
        ),
        "recovery_command": (
            {
                "operation": result.recovery_command.operation,
                "record": deepcopy(dict(result.recovery_command.record)),
                "validation_receipt": result.recovery_command.validation_receipt,
            }
            if result.recovery_command
            else None
        ),
        "navigation_command": (
            {
                "operation": result.navigation_command.operation,
                "target_group": result.navigation_command.target_group,
                "origin_group": result.navigation_command.origin_group,
            }
            if result.navigation_command
            else None
        ),
        "field_reconfiguration_command": (
            {
                "group": result.field_reconfiguration_command.group,
                "config_field": result.field_reconfiguration_command.config_field,
                "prerequisite_question_id": result.field_reconfiguration_command.prerequisite_question_id,
            }
            if result.field_reconfiguration_command
            else None
        ),
        "workflow_goal_command": (
            {
                "operation": result.workflow_goal_command.operation,
                "goal": deepcopy(dict(result.workflow_goal_command.goal)),
            }
            if result.workflow_goal_command
            else None
        ),
        "completion": result.completion,
        "blocker": result.blocker,
        "stop_after_response": result.stop_after_response,
    }


def handler_result_from_dict(payload: Mapping[str, Any]) -> HandlerResult:
    raw_delta = dict(payload.get("delta") or {})
    delta = StateDelta(
        writes=tuple(
            DeltaWrite(
                tuple(str(part) for part in item.get("path") or ()),
                deepcopy(item.get("value")),
            )
            for item in raw_delta.get("writes") or ()
        ),
        deletes=tuple(
            tuple(str(part) for part in path)
            for path in raw_delta.get("deletes") or ()
        ),
    )
    checkpoint = payload.get("checkpoint_command")
    recovery = payload.get("recovery_command")
    navigation = payload.get("navigation_command")
    field_reconfiguration = payload.get("field_reconfiguration_command")
    workflow_goal = payload.get("workflow_goal_command")
    return HandlerResult(
        delta=delta,
        control_receipts=tuple(
            deepcopy(dict(item))
            for item in payload.get("control_receipts") or ()
            if isinstance(item, Mapping)
        ),
        consumed_action_ids=tuple(str(item) for item in payload.get("consumed_action_ids") or ()),
        invalidated_groups=tuple(str(item) for item in payload.get("invalidated_groups") or ()),
        reconfigured_groups=tuple(str(item) for item in payload.get("reconfigured_groups") or ()),
        invalidated_fields=tuple(str(item) for item in payload.get("invalidated_fields") or ()),
        evidence=tuple(deepcopy(dict(item)) for item in payload.get("evidence") or ()),
        action_errors=tuple(deepcopy(dict(item)) for item in payload.get("action_errors") or ()),
        visible_result=str(payload.get("visible_result") or ""),
        visible_results=tuple(str(item) for item in payload.get("visible_results") or ()),
        pending_question=deepcopy(payload.get("pending_question")),
        clear_pending=bool(payload.get("clear_pending")),
        next_group=str(payload.get("next_group") or ""),
        navigation_resume_group=str(payload.get("navigation_resume_group") or ""),
        followup_actions=tuple(deepcopy(dict(item)) for item in payload.get("followup_actions") or ()),
        checkpoint_command=(
            CheckpointCommand(
                command=str(checkpoint.get("command") or ""),  # type: ignore[arg-type]
                confirmed_config=deepcopy(dict(checkpoint.get("confirmed_config") or {})),
            )
            if isinstance(checkpoint, Mapping)
            else None
        ),
        recovery_command=(
            RecoveryCommand(
                operation=str(recovery.get("operation") or ""),  # type: ignore[arg-type]
                record=deepcopy(dict(recovery.get("record") or {})),
                validation_receipt=str(recovery.get("validation_receipt") or ""),
            )
            if isinstance(recovery, Mapping)
            else None
        ),
        navigation_command=(
            NavigationCommand(
                operation=str(navigation.get("operation") or ""),  # type: ignore[arg-type]
                target_group=str(navigation.get("target_group") or ""),
                origin_group=str(navigation.get("origin_group") or ""),
            )
            if isinstance(navigation, Mapping)
            else None
        ),
        field_reconfiguration_command=(
            FieldReconfigurationCommand(
                group=str(field_reconfiguration.get("group") or ""),
                config_field=str(field_reconfiguration.get("config_field") or ""),
                prerequisite_question_id=str(field_reconfiguration.get("prerequisite_question_id") or ""),
            )
            if isinstance(field_reconfiguration, Mapping)
            else None
        ),
        workflow_goal_command=(
            WorkflowGoalCommand(
                operation=str(workflow_goal.get("operation") or ""),  # type: ignore[arg-type]
                goal=deepcopy(dict(workflow_goal.get("goal") or {})),
            )
            if isinstance(workflow_goal, Mapping)
            else None
        ),
        completion=str(payload.get("completion") or "unchanged"),  # type: ignore[arg-type]
        blocker=str(payload.get("blocker") or ""),
        stop_after_response=bool(payload.get("stop_after_response")),
    )


def pending_domain_result_to_dict(value: PendingDomainResult) -> dict[str, Any]:
    return {
        "action": action_envelope_to_dict(value.action),
        "result": handler_result_to_dict(value.result),
        "prepared_state_hash": value.prepared_state_hash,
    }


def pending_domain_result_from_dict(payload: Mapping[str, Any]) -> PendingDomainResult:
    return PendingDomainResult(
        action=action_envelope_from_dict(dict(payload.get("action") or {})),
        result=handler_result_from_dict(dict(payload.get("result") or {})),
        prepared_state_hash=str(payload.get("prepared_state_hash") or ""),
    )


def side_effect_intent_to_dict(intent: SideEffectIntent) -> dict[str, Any]:
    return {
        **intent.__dict__,
        "request": deepcopy(dict(intent.request)),
    }


def side_effect_receipt_to_dict(receipt: SideEffectReceipt) -> dict[str, Any]:
    return {
        **receipt.__dict__,
        "result": deepcopy(dict(receipt.result)),
    }


def turn_receipt_to_dict(receipt: TurnReceipt) -> dict[str, Any]:
    return {
        "turn_id": receipt.turn_id,
        "input_hash": receipt.input_hash,
        "language": receipt.language,
        "input_shape": receipt.input_shape,
        "clauses": [deepcopy(dict(item)) for item in receipt.clauses],
        "semantic_units": [
            deepcopy(dict(item)) for item in receipt.semantic_units
        ],
        "pending_before": deepcopy(dict(receipt.pending_before)),
        "admitted_action_ids": list(receipt.admitted_action_ids),
        "semantic_order": list(receipt.semantic_order),
        "execution_order": list(receipt.execution_order),
        "owner_bindings": deepcopy(dict(receipt.owner_bindings)),
        "action_unit_bindings": {
            str(action_id): list(unit_ids)
            for action_id, unit_ids in receipt.action_unit_bindings.items()
        },
        "unit_action_bindings": {
            str(unit_id): list(action_ids)
            for unit_id, action_ids in receipt.unit_action_bindings.items()
        },
        "pending_candidate_verdicts": [
            deepcopy(dict(item))
            for item in receipt.pending_candidate_verdicts
        ],
        "sibling_omission_checks": [
            deepcopy(dict(item))
            for item in receipt.sibling_omission_checks
        ],
        "unresolved_units": list(receipt.unresolved_units),
        "pending_after": deepcopy(dict(receipt.pending_after)),
        "response_count": receipt.response_count,
        "status": receipt.status,
    }
