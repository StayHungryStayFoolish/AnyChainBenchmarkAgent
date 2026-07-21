"""Typed contracts shared by the AnyChain Harness control plane.

The model may propose actions, but only these contracts can authorize state
transitions or user-visible choices.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Literal, Mapping


ReturnPolicy = Literal["stay", "resume_interrupted", "fallback", "stop_after_response"]
CompletionStatus = Literal["unchanged", "in_progress", "completed", "blocked"]
RecoveryOperation = Literal["activate", "resolve"]
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
            if not option.expected_patch and option.return_policy != "stop_after_response":
                raise ValueError(f"{self.question_id}/{option.option_id} has no postcondition")


@dataclass(frozen=True)
class HandlerResult:
    """The only result a domain handler may return to the coordinator."""

    delta: StateDelta = dataclass_field(default_factory=StateDelta)
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
    completion: CompletionStatus = "unchanged"
    blocker: str = ""
    stop_after_response: bool = False
