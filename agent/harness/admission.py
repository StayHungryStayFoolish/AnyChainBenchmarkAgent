"""Semantic action admission authority for the AnyChain Harness."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .action_registry import (
    ACTION_BY_TYPE,
    lifecycle_rejected_action_indexes,
    validate_action_contract,
    validate_action_transaction_contract,
    validate_field_intake_admission_receipt,
    validate_proposal_field_receipts,
)
from .contracts import AdmissionRejection, AdmissionResult
from .invariants import StateInvariantError
from .questions import (
    pending_option_value_exists as _pending_option_value_exists,
    value_satisfies_pending_contract as _value_satisfies_pending_contract,
)
from .state import AgentGraphState
from agent.workflows.group_registry import invalidation_targets


def reconcile_admission_coverage(
    receipt: Mapping[str, Any],
    actions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconcile semantic coverage after rejected proposals are removed."""

    reconciled = deepcopy(dict(receipt))
    action_ids_by_unit: dict[str, list[str]] = {}
    for action in actions:
        action_type = str(action.get("type") or "")
        if action_type == "unknown" or action_type not in ACTION_BY_TYPE:
            continue
        action_id = str(action.get("action_id") or "")
        if not action_id:
            continue
        for unit_id in action.get("_source_unit_ids") or []:
            normalized_unit_id = str(unit_id or "")
            if normalized_unit_id:
                action_ids_by_unit.setdefault(normalized_unit_id, []).append(action_id)

    unresolved: list[str] = []
    semantic_units: list[dict[str, Any]] = []
    for raw_unit in reconciled.get("semantic_units") or []:
        unit = deepcopy(dict(raw_unit))
        unit_id = str(unit.get("unit_id") or "")
        if (
            unit_id
            and str(unit.get("disposition") or "") == "action"
            and unit_id not in action_ids_by_unit
        ):
            unit["proposed_action_indexes"] = list(unit.get("action_indexes") or [])
            unit["action_indexes"] = []
            unit["disposition"] = "unresolved"
        if str(unit.get("disposition") or "") == "unresolved" and unit_id:
            unresolved.append(unit_id)
        semantic_units.append(unit)

    omission_checks: list[dict[str, Any]] = []
    for raw_check in reconciled.get("sibling_omission_checks") or []:
        check = deepcopy(dict(raw_check))
        unit_id = str(check.get("unit_id") or "")
        action_ids = list(dict.fromkeys(action_ids_by_unit.get(unit_id, [])))
        check["action_ids"] = action_ids
        if str(check.get("disposition") or "") == "action" and not action_ids:
            check["proposed_action_indexes"] = list(check.get("action_indexes") or [])
            check["action_indexes"] = []
            check["disposition"] = "unresolved"
            check["verdict"] = "unresolved"
        omission_checks.append(check)

    reconciled["semantic_units"] = semantic_units
    reconciled["unresolved_units"] = list(dict.fromkeys(unresolved))
    reconciled["sibling_omission_checks"] = omission_checks
    return reconciled


def _normalized_action_queue(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_actions = payload.get("actions") if isinstance(payload, dict) else None
    if isinstance(raw_actions, dict):
        raw_actions = [raw_actions]
    if not isinstance(raw_actions, list):
        return []
    output: list[dict[str, Any]] = []
    current_actions = [raw for raw in raw_actions[:12] if isinstance(raw, dict)]
    for raw in current_actions:
        try:
            # The resolver is the sole producer of trusted admission receipts.
            # Model documents are validated before those receipts are attached.
            action = validate_action_contract(raw, trusted_metadata=True)
        except ValueError as exc:
            action = {"type": "unknown", "reason": str(exc), "confidence": "low"}
        action_type = str(action.get("type") or "unknown").strip()
        action["type"] = action_type
        action["confidence"] = str(action.get("confidence") or "medium").strip().lower()
        output.append(action)
    return output


def _canonical_pending_choice_matches(
    state: AgentGraphState,
    action: Mapping[str, Any],
) -> bool:
    """Verify one admitted semantic choice against its immutable contract."""

    pending = dict(state.get("pending_question") or {})
    selected = action.get("selected_value")
    action_id = str(action.get("_admission_action_id") or "")
    matches = []
    for row in (state.get("turn_context") or {}).get("pending_choice_contracts") or []:
        if not isinstance(row, Mapping):
            continue
        question = row.get("question") if isinstance(row.get("question"), Mapping) else {}
        option = row.get("option") if isinstance(row.get("option"), Mapping) else {}
        provenance = row.get("semantic_units") if isinstance(row.get("semantic_units"), list) else []
        if (
            str(question.get("id") or "") == str(pending.get("id") or "")
            and str(question.get("group") or "") == str(pending.get("group") or "")
            and option.get("selected_value") == selected
            and str(row.get("admission_action_id") or "") == action_id
            and provenance
        ):
            matches.append(row)
    return len(matches) == 1


def _bypasses_canonical_pending_choice(
    state: AgentGraphState,
    action: Mapping[str, Any],
) -> bool:
    """Reject a declared option owner that bypassed canonical admission."""

    if str(action.get("type") or "") == "answer_pending":
        return False
    pending = dict(state.get("pending_question") or {})
    for option in pending.get("options") or []:
        declared = option.get("action") if isinstance(option, Mapping) else None
        if not isinstance(declared, Mapping):
            continue
        if str(declared.get("type") or "") != str(action.get("type") or ""):
            continue
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if spec is None or not spec.pending_option_admission:
            continue
        if all(key == "type" or action.get(key) == value for key, value in declared.items()):
            return True
    return False


def validate_action_plan(
    state: AgentGraphState,
    actions: list[dict[str, Any]],
) -> AdmissionResult:
    """Validate one immutable planner transaction without repairing it."""

    if not actions:
        return AdmissionResult(status="accepted")
    proposed = tuple(dict(item) for item in actions)
    try:
        validate_action_transaction_contract(list(proposed))
    except ValueError as exc:
        return AdmissionResult(
            status="rejected",
            rejections=(AdmissionRejection("transaction_invalid", str(exc)),),
        )
    unknown = tuple(
        index
        for index, item in enumerate(proposed)
        if str(item.get("type") or "") not in ACTION_BY_TYPE
        or str(item.get("type") or "") == "unknown"
    )
    if unknown:
        return AdmissionResult(
            status="rejected",
            rejections=(
                AdmissionRejection(
                    "unknown_action",
                    "the proposed transaction contains an unregistered action",
                    unknown,
                ),
            ),
        )
    bypasses = tuple(
        index
        for index, item in enumerate(proposed)
        if _bypasses_canonical_pending_choice(state, item)
    )
    if bypasses:
        return AdmissionResult(
            status="rejected",
            rejections=(
                AdmissionRejection(
                    "pending_choice_bypass",
                    "a declared pending option bypassed its canonical contract",
                    bypasses,
                ),
            ),
        )
    pending_answers = tuple(
        index
        for index, item in enumerate(proposed)
        if str(item.get("type") or "") == "answer_pending"
    )
    prepared = tuple(
        _attach_canonical_admission_metadata(state, item)
        for item in proposed
    )
    if len(pending_answers) > 1 or any(
        not _action_answers_pending_contract(state, dict(prepared[index]))
        for index in pending_answers
    ):
        return AdmissionResult(
            status="rejected",
            rejections=(
                AdmissionRejection(
                    "pending_contract_mismatch",
                    "the proposed pending answer does not match the active contract",
                    pending_answers,
                ),
            ),
        )
    if _pending_answer_is_invalidated(state, prepared):
        return AdmissionResult(
            status="rejected",
            rejections=(
                AdmissionRejection(
                    "pending_answer_invalidated",
                    "a sibling mutation invalidates the active pending answer",
                    pending_answers,
                ),
            ),
        )
    lifecycle_rejected = lifecycle_rejected_action_indexes(state, list(prepared))
    if lifecycle_rejected:
        return AdmissionResult(
            status="rejected",
            rejections=(
                AdmissionRejection(
                    "lifecycle_incompatible",
                    "the proposed transaction is incompatible with current lifecycle state",
                    lifecycle_rejected,
                ),
            ),
        )
    return AdmissionResult(
        status="accepted",
        actions=prepared,
    )


def _attach_canonical_admission_metadata(
    state: AgentGraphState,
    action: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Attach only contract-derived metadata without changing action meaning."""

    canonical = dict(action)
    if (
        canonical.get("pending_option_semantic_verified") is True
        and _canonical_pending_choice_matches(state, canonical)
    ):
        canonical["selection_contract_verified"] = True
    return canonical


def _action_satisfies_pending_manual_effect(
    pending: Mapping[str, Any],
    action: Mapping[str, Any],
) -> bool:
    """Return whether one typed action supplies the pending manual contract."""

    manual = pending.get("manual_action")
    if not isinstance(manual, Mapping):
        return False
    if str(action.get("type") or "") != str(manual.get("type") or ""):
        return False
    value_argument = str(manual.get("value_argument") or "").strip()
    fixed = {
        str(key): value
        for key, value in manual.items()
        if str(key) not in {"type", "value_argument", "use_complete_turn"}
    }
    if any(action.get(key) != value for key, value in fixed.items()):
        return False
    return bool(value_argument and action.get(value_argument) not in (None, ""))


def _pending_answer_is_invalidated(
    state: AgentGraphState,
    actions: tuple[Mapping[str, Any], ...],
) -> bool:
    """Return whether a sibling mutation invalidates a pending answer."""

    pending_group = str((state.get("pending_question") or {}).get("group") or "").strip()
    if not pending_group:
        return False
    for action in actions:
        if str(action.get("type") or "") == "answer_pending":
            continue
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if spec is None or not spec.mutation_dimension:
            continue
        source_group = str(spec.target_group or spec.mutation_dimension).strip()
        if pending_group in set(invalidation_targets(source_group)):
            return True
    return False


def _action_answers_pending_contract(state: AgentGraphState, action: dict[str, Any]) -> bool:
    """Reject model answers that bypass the active typed question contract."""

    pending = state.get("pending_question") or {}
    if not pending:
        return False
    if str(pending.get("kind") or "") in {"numbered_choice", "yes_no"}:
        selected = action.get("selected_value")
        if not _pending_option_value_exists(selected, pending):
            answer = str(action.get("answer") or "").strip()
            evidence = str(action.get("source_evidence") or "").strip()
            user_text = str(state.get("last_user_input") or "")
            return bool(
                pending.get("manual_input_allowed") is True
                and answer
                and evidence
                and evidence in user_text
                and answer in evidence
                and _value_satisfies_pending_contract(answer, pending)
                and action.get("semantic_purpose_verified") is True
            )
        if not (
            action.get("selection_contract_verified") is True
            and _canonical_pending_choice_matches(state, action)
        ):
            return False
        return True
    selected = action.get("selected_value")
    if isinstance(selected, str) and not selected.strip():
        selected = None
    raw_answer = selected if selected is not None else action.get("answer")
    answer = str(raw_answer or "").strip()
    evidence = str(action.get("source_evidence") or "").strip()
    user_text = str(state.get("last_user_input") or "")
    declared_option = _pending_option_value_exists(selected, pending)
    if not evidence or evidence not in user_text:
        return False
    if (
        not declared_option
        and answer not in evidence
        and action.get("semantic_purpose_verified") is not True
    ):
        return False
    reserved_identifiers = {
        str(pending.get("id") or "").strip().casefold(),
        str(pending.get("field") or "").strip().casefold(),
        str(pending.get("group") or "").strip().casefold(),
    }
    if answer.casefold() in reserved_identifiers:
        return False
    return bool(
        raw_answer not in (None, "")
        and (
            declared_option
            or _value_satisfies_pending_contract(raw_answer, pending)
        )
    )


def _has_meaningful_queue(actions: list[dict[str, Any]]) -> bool:
    return any(
        action_type in ACTION_BY_TYPE and action_type != "unknown"
        for item in actions
        if (action_type := str(item.get("type") or "").strip())
    )


def _validate_admission_transaction(
    state: AgentGraphState,
    actions: list[dict[str, Any]],
    *,
    current_submission: bool,
) -> None:
    """Validate Harness receipts against their session and ordered transaction."""

    thread_id = str(state.get("thread_id") or "default")
    session_id = str((state.get("session") or {}).get("id") or thread_id)
    turn_index = int(state.get("turn_index") or 0)
    receipt_actions = [
        action for action in actions
        if isinstance(action, dict) and action.get("_admission_action_id")
    ]
    if receipt_actions:
        declared_orders = {
            tuple(str(value) for value in action.get("_transaction_action_ids") or [])
            for action in receipt_actions
        }
        if len(declared_orders) != 1:
            raise StateInvariantError("admission transaction order metadata is inconsistent")
        declared_order = next(iter(declared_orders))
        actual_order = tuple(str(action.get("_admission_action_id") or "") for action in receipt_actions)
        if actual_order != tuple(value for value in declared_order if value in set(actual_order)):
            raise StateInvariantError("admission transaction action order mismatch")
    for action in actions:
        action_type = str(action.get("type") or "")
        if action_type == "request_config_field_input":
            validate_field_intake_admission_receipt(
                action,
                thread_id=thread_id,
                session_id=session_id,
            )
            receipt = action.get("_semantic_admission_receipt") or {}
            if current_submission and int(receipt.get("submitted_turn_index") or 0) != turn_index:
                raise StateInvariantError("field intake receipt belongs to another turn")
        elif action_type == "propose_config_values":
            validate_proposal_field_receipts(
                action,
                thread_id=thread_id,
                session_id=session_id,
                submitted_turn_index=turn_index if current_submission else None,
            )
