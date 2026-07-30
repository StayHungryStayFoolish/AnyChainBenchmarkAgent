"""Typed contracts shared by the AnyChain Harness control plane.

The model may propose actions, but only these contracts can authorize state
transitions or user-visible choices.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field as dataclass_field, is_dataclass
import math
import re
from typing import Any, Literal, Mapping


ReturnPolicy = Literal["stay", "resume_interrupted", "fallback", "stop_after_response"]
CompletionStatus = Literal["unchanged", "in_progress", "completed", "blocked"]
RecoveryOperation = Literal["activate", "resolve"]
NavigationOperation = Literal["change_group", "go_back"]
WorkflowGoalOperation = Literal["enqueue", "remove_first"]
SemanticDraftOperation = Literal["resolve", "cancel", "previous", "invalidate"]
ActionEffect = Literal["pure", "read_only", "external"]
SECRET_REFERENCE_RE = re.compile(r"semantic-secret:[A-Za-z0-9_-]+")
ActionEnvelopeStatus = Literal[
    "admitted",
    "selected",
    "prepared",
    "committed",
    "blocked",
    "failed",
]


def is_secret_reference(value: Any) -> bool:
    """Return whether a value is exactly one opaque secret capability."""

    return SECRET_REFERENCE_RE.fullmatch(str(value or "")) is not None
AdmissionStatus = Literal["accepted", "rejected"]
ResponseFragmentKind = Literal["message", "status", "evidence", "warning", "error"]
ResponseArgument = str | int | float | bool | None
FailureSeverity = Literal["info", "warning", "blocking", "critical"]
SideEffectStatus = Literal["prepared", "invoking", "succeeded", "reused", "blocked", "failed"]
TurnReceiptStatus = Literal["prepared", "planned", "executing", "completed", "blocked", "failed"]
SemanticPlanDraftStatus = Literal[
    "awaiting_clarification",
    "ready_for_review",
    "stale",
    "cancelled",
]
SemanticUnresolvedReason = Literal[
    "semantic_ambiguity",
    "missing_user_evidence",
]
StatePath = tuple[str, ...]


DOMAIN_CONTROL_ROOTS = frozenset(
    {
        "active_group",
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
        "response_fragments",
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


@dataclass(frozen=True)
class SemanticDraftCommand:
    """Mutate only the coordinator-owned semantic draft."""

    operation: SemanticDraftOperation
    draft_id: str
    revision: int
    atom_id: str = ""
    resolution: str = ""
    resolution_hash: str = ""
    resolution_ref: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.draft_id or self.revision < 1:
            raise ValueError("semantic draft command requires bound identity")
        if self.operation == "resolve" and (
            not self.atom_id or not self.resolution.strip()
        ):
            raise ValueError("resolve semantic draft command requires atom and value")
        if self.operation == "resolve" and (
            len(self.resolution_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.resolution_hash
            )
        ):
            raise ValueError(
                "resolve semantic draft command requires a value hash"
            )
        if self.resolution_ref and (
            not is_secret_reference(self.resolution_ref)
            or "***REDACTED***" not in self.resolution
        ):
            raise ValueError(
                "semantic draft secret resolution reference is invalid"
            )
        if self.operation == "previous" and not self.atom_id:
            raise ValueError("previous semantic draft command requires atom")


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
class AdmissionRejection:
    """One stable reason an immutable proposed transaction was rejected."""

    code: str
    message: str
    action_indexes: tuple[int, ...] = ()


@dataclass(frozen=True)
class AdmissionResult:
    """Pure validation result for one complete proposed action transaction."""

    status: AdmissionStatus
    actions: tuple[Mapping[str, Any], ...] = ()
    rejections: tuple[AdmissionRejection, ...] = ()

    def __post_init__(self) -> None:
        if self.status == "accepted" and self.rejections:
            raise ValueError("accepted admission result cannot contain rejections")
        if self.status == "rejected" and not self.rejections:
            raise ValueError("rejected admission result requires a reason")
        if self.status == "rejected" and self.actions:
            raise ValueError("rejected admission result cannot contain actions")


@dataclass(frozen=True)
class OptionContract:
    """Executable behavior attached to one rendered option."""

    option_id: str
    value: Any
    action: ActionProposal
    expected_patch: Mapping[str, Any]
    label: TextRef
    description: TextRef | None = None
    completion_effect: TextRef | None = None
    manual_entry: bool = False
    return_policy: ReturnPolicy = "fallback"
    preconditions: Mapping[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class QuestionContract:
    """A blocking question owned by exactly one workflow group."""

    question_id: str
    group: str
    owner: str
    kind: str
    prompt: TextRef
    field: str = ""
    options: tuple[OptionContract, ...] = ()
    manual_input_allowed: bool = False
    validation: Mapping[str, Any] = dataclass_field(default_factory=dict)
    requires_capabilities: tuple[str, ...] = ()
    domain_context: Mapping[str, Any] = dataclass_field(default_factory=dict)
    help_text: TextRef | None = None
    completion_effect: TextRef | None = None

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


def _validate_response_arguments(arguments: Mapping[str, ResponseArgument]) -> None:
    if not isinstance(arguments, Mapping):
        raise TypeError("response arguments must be a mapping")
    for key, value in arguments.items():
        if not isinstance(key, str) or not key:
            raise ValueError("response argument names must be non-empty strings")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"response argument {key!r} must be a scalar value")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"response argument {key!r} must be finite")


@dataclass(frozen=True)
class TextRef:
    """Language-independent reference to one centrally registered message."""

    message_id: str
    arguments: Mapping[str, ResponseArgument] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message_id.strip():
            raise ValueError("text reference message_id cannot be empty")
        _validate_response_arguments(self.arguments)


def text_ref_to_dict(reference: TextRef) -> dict[str, Any]:
    """Serialize one registered text reference without rendered prose."""

    if not isinstance(reference, TextRef):
        raise TypeError("question text must be a TextRef")
    return {
        "message_id": reference.message_id,
        "arguments": dict(reference.arguments),
    }


def text_ref_from_dict(value: Mapping[str, Any]) -> TextRef:
    """Deserialize a strict TextRef mapping and reject compatibility prose."""

    if not isinstance(value, Mapping):
        raise TypeError("question text reference must be a mapping")
    if set(value) != {"message_id", "arguments"}:
        raise ValueError(
            "question text reference requires exactly message_id and arguments"
        )
    arguments = value.get("arguments")
    if not isinstance(arguments, Mapping):
        raise TypeError("question text reference arguments must be a mapping")
    return TextRef(
        message_id=str(value.get("message_id") or ""),
        arguments=dict(arguments),
    )


@dataclass(frozen=True)
class ResponseFragment:
    """Typed semantic output rendered only by the central response catalog."""

    kind: ResponseFragmentKind
    message_id: str
    arguments: Mapping[str, ResponseArgument] = dataclass_field(default_factory=dict)
    payload: Mapping[str, Any] = dataclass_field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"message", "status", "evidence", "warning", "error"}:
            raise ValueError(f"unsupported response fragment kind: {self.kind}")
        TextRef(self.message_id, self.arguments)
        if not isinstance(self.payload, Mapping):
            raise TypeError("response fragment payload must be a mapping")

    @property
    def text_ref(self) -> TextRef:
        return TextRef(self.message_id, self.arguments)


@dataclass(frozen=True)
class FailureDescriptor:
    """Language-independent failure facts rendered by the response authority."""

    code: str
    arguments: Mapping[str, ResponseArgument] = dataclass_field(default_factory=dict)
    payload: Mapping[str, Any] = dataclass_field(default_factory=dict)
    source: str = ""
    retryable: bool = False
    severity: FailureSeverity = "blocking"

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("failure descriptor code cannot be empty")
        _validate_response_arguments(self.arguments)
        if not isinstance(self.payload, Mapping):
            raise TypeError("failure descriptor payload must be a mapping")
        if self.severity not in {"info", "warning", "blocking", "critical"}:
            raise ValueError(f"unsupported failure severity: {self.severity}")


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
    response_fragments: tuple[ResponseFragment, ...] = ()
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
    semantic_draft_command: SemanticDraftCommand | None = None
    completion: CompletionStatus = "unchanged"
    blocker: FailureDescriptor | None = None
    stop_after_response: bool = False
    suppress_pending_render: bool = False

    def __post_init__(self) -> None:
        if not all(
            isinstance(fragment, ResponseFragment)
            for fragment in self.response_fragments
        ):
            raise TypeError(
                "handler result response_fragments must contain ResponseFragment values"
            )
        if self.blocker is not None and not isinstance(
            self.blocker, FailureDescriptor
        ):
            raise TypeError("handler result blocker must be a FailureDescriptor")


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
    execution_request_id: str
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


@dataclass(frozen=True)
class SemanticDraftCandidate:
    """One validated but deliberately non-admitted draft action."""

    candidate_id: str
    source_atom_ids: tuple[str, ...]
    owner: str
    group: str
    action: Mapping[str, Any]
    validation_hash: str
    contract_hashes: Mapping[str, str] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class SemanticUnresolvedAtom:
    """One exact semantic atom that requires user clarification."""

    atom_id: str
    unit_id: str
    clause_id: str
    source_text: str
    source_path: str
    semantic_source: str
    owner: str
    group: str
    sensitive_input: bool
    reason: SemanticUnresolvedReason
    reason_detail: str
    resolution: str = ""
    resolution_hash: str = ""
    resolution_ref: str = ""


@dataclass(frozen=True)
class SemanticPlanDraft:
    """Durable coordinator-owned plan that has not crossed admission."""

    draft_id: str
    revision: int
    status: SemanticPlanDraftStatus
    session_id: str
    creation_turn: int
    source_product_authority_id: str
    source_checkpoint_revision: int
    source_checkpoint_thread_id: str
    source_checkpoint_id: str
    source_schema_version: int
    source_checkpoint_fingerprint: str
    original_input: str
    original_input_hash: str
    source_clauses: tuple[Mapping[str, Any], ...]
    source_partition: tuple[Mapping[str, Any], ...]
    source_partition_hash: str
    source_active_group: str
    source_pending_question: Mapping[str, Any]
    source_secret_bindings: tuple[Mapping[str, str], ...]
    candidates: tuple[SemanticDraftCandidate, ...]
    unresolved_atoms: tuple[SemanticUnresolvedAtom, ...]
    active_atom_id: str
    registry_hash: str
    action_registry_hash: str
    pending_contract_hash: str
    pending_authority_hash: str
    workflow_precondition_hash: str
    lifecycle_receipts: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class SemanticDraftFinalizationReceipt:
    """Proof binding a resolved draft to its later admission transaction."""

    draft_id: str
    draft_revision: int
    session_id: str
    source_product_authority_id: str
    source_checkpoint_revision: int
    source_checkpoint_thread_id: str
    source_checkpoint_id: str
    source_checkpoint_fingerprint: str
    source_partition_hash: str
    candidate_ids: tuple[str, ...]
    candidate_convergence_hash: str
    secret_binding_manifest_hash: str
    secret_binding_hashes: Mapping[str, str]
    resolution_hash: str
    final_plan_hash: str
    admission_transaction_hash: str
    final_action_ids: tuple[str, ...]
    receipt_hash: str


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


def response_fragment_to_dict(fragment: ResponseFragment) -> dict[str, Any]:
    return {
        "kind": fragment.kind,
        "message_id": fragment.message_id,
        "arguments": deepcopy(dict(fragment.arguments)),
        "payload": deepcopy(dict(fragment.payload)),
        "source": fragment.source,
    }


def response_fragment_from_dict(payload: Mapping[str, Any]) -> ResponseFragment:
    expected_keys = {"kind", "message_id", "arguments", "payload", "source"}
    if set(payload) != expected_keys:
        raise ValueError(
            "response fragment mapping must contain exactly "
            "kind, message_id, arguments, payload, and source"
        )
    return ResponseFragment(
        kind=str(payload.get("kind") or ""),  # type: ignore[arg-type]
        message_id=str(payload.get("message_id") or ""),
        arguments=deepcopy(dict(payload.get("arguments") or {})),
        payload=deepcopy(dict(payload.get("payload") or {})),
        source=str(payload.get("source") or ""),
    )


def failure_descriptor_to_dict(failure: FailureDescriptor) -> dict[str, Any]:
    return {
        "code": failure.code,
        "arguments": deepcopy(dict(failure.arguments)),
        "payload": deepcopy(dict(failure.payload)),
        "source": failure.source,
        "retryable": failure.retryable,
        "severity": failure.severity,
    }


def failure_descriptor_from_dict(payload: Mapping[str, Any]) -> FailureDescriptor:
    expected_keys = {
        "code",
        "arguments",
        "payload",
        "source",
        "retryable",
        "severity",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            "failure descriptor mapping must contain exactly "
            "code, arguments, payload, source, retryable, and severity"
        )
    return FailureDescriptor(
        code=str(payload.get("code") or ""),
        arguments=deepcopy(dict(payload.get("arguments") or {})),
        payload=deepcopy(dict(payload.get("payload") or {})),
        source=str(payload.get("source") or ""),
        retryable=bool(payload.get("retryable")),
        severity=str(payload.get("severity") or ""),  # type: ignore[arg-type]
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
        "response_fragments": [
            response_fragment_to_dict(item) for item in result.response_fragments
        ],
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
        "semantic_draft_command": (
            {
                "operation": result.semantic_draft_command.operation,
                "draft_id": result.semantic_draft_command.draft_id,
                "revision": result.semantic_draft_command.revision,
                "atom_id": result.semantic_draft_command.atom_id,
                "resolution": result.semantic_draft_command.resolution,
                "resolution_hash": result.semantic_draft_command.resolution_hash,
                "resolution_ref": result.semantic_draft_command.resolution_ref,
                "reason": result.semantic_draft_command.reason,
            }
            if result.semantic_draft_command
            else None
        ),
        "completion": result.completion,
        "blocker": (
            failure_descriptor_to_dict(result.blocker) if result.blocker else None
        ),
        "stop_after_response": result.stop_after_response,
        "suppress_pending_render": result.suppress_pending_render,
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
    semantic_draft = payload.get("semantic_draft_command")
    raw_response_fragments = tuple(payload.get("response_fragments") or ())
    if not all(isinstance(item, Mapping) for item in raw_response_fragments):
        raise TypeError("handler result response_fragments must contain mappings")
    response_fragments = tuple(
        response_fragment_from_dict(item)
        for item in raw_response_fragments
    )
    blocker = payload.get("blocker")
    if blocker is not None and not isinstance(blocker, Mapping):
        raise TypeError("handler result blocker must be a FailureDescriptor mapping")
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
        response_fragments=response_fragments,
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
        semantic_draft_command=(
            SemanticDraftCommand(
                operation=str(semantic_draft.get("operation") or ""),  # type: ignore[arg-type]
                draft_id=str(semantic_draft.get("draft_id") or ""),
                revision=int(semantic_draft.get("revision") or 0),
                atom_id=str(semantic_draft.get("atom_id") or ""),
                resolution=str(semantic_draft.get("resolution") or ""),
                resolution_hash=str(
                    semantic_draft.get("resolution_hash") or ""
                ),
                resolution_ref=str(
                    semantic_draft.get("resolution_ref") or ""
                ),
                reason=str(semantic_draft.get("reason") or ""),
            )
            if isinstance(semantic_draft, Mapping)
            else None
        ),
        completion=str(payload.get("completion") or "unchanged"),  # type: ignore[arg-type]
        blocker=(
            failure_descriptor_from_dict(blocker)
            if isinstance(blocker, Mapping)
            else None
        ),
        stop_after_response=bool(payload.get("stop_after_response")),
        suppress_pending_render=bool(payload.get("suppress_pending_render")),
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
