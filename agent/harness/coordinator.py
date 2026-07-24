"""Deterministic group workflow engine for the LangGraph Harness."""

from __future__ import annotations

import json
import hashlib
import re
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
from typing import Any, Mapping

from .state import AgentGraphState, PendingQuestion, RESET_PRESERVED_KEYS, new_state
from .intent import (
    ALLOWED_GROUPS,
)
from .hierarchical_planner import resolve_product_action_queue as resolve_action_queue
from .oracle import (
    compute_next_action,
    format_recommended_next_action,
)
from .routing import (
    chain_identity_confirmed,
    group_readiness,
    navigation_prerequisite as _navigation_prerequisite,
    next_group_and_reason,
    option_return_policy as _option_return_policy,
)
from .turns import adjudicate_turn
from .plan_coverage import segment_user_turn
from .action_registry import ACTION_BY_TYPE, ACTION_METADATA_FIELDS, TRUSTED_ACTION_METADATA_FIELDS, action_crosses_pending_barrier, action_execution_phase, action_is_turn_local, action_merge_key, action_preserves_pending, assign_action_ids, lifecycle_rejected_action_indexes, merge_semantic_actions, normalize_action_relations, validate_action_contract, validate_action_transaction_contract, validate_field_intake_admission_receipt, validate_proposal_field_receipts
from .contracts import (
    ActionEnvelope,
    ActionProposal,
    CheckpointCommand,
    FieldReconfigurationCommand,
    HandlerResult,
    NavigationCommand,
    RecoveryCommand,
    PendingDomainResult,
    SideEffectIntent,
    SideEffectReceipt,
    TurnReceipt,
    WorkflowGoalCommand,
    StateDelta,
    action_envelope_from_dict,
    action_envelope_to_dict,
    handler_result_from_dict,
    handler_result_to_dict,
    pending_domain_result_from_dict,
    pending_domain_result_to_dict,
    side_effect_intent_to_dict,
    side_effect_receipt_to_dict,
    turn_receipt_to_dict,
)
from .control_receipts import validate_domain_control_receipt
from .localization import localized as _localized
from .domains.orientation import completed_group_status
from .domains.environment import (
    apply_inferred_config_review,
    config_proposal_review_question,
)
from .domains.chain_rpc_support import is_existing_family_lifecycle
from .domains.analysis import (
    JOB_ID_RE,
    is_evidence_completion_command,
    prompt_evidence_collection_waiting,
    should_start_evidence_collection,
)
from .domains.execution import reconcile_execution_state
from .domains.recovery import question_for_recovery
from .failures import domain_blocker_failure_record, model_provider_failure_record, render_failure_summary
from .domains.registry import GROUP_OWNER
from .domains.runtime import DOMAIN_RUNTIME, DomainRuntime
from .invariants import StateInvariantError, apply_state_delta, validate_state
from .transitions import mark_group_reconfigured, mark_group_reconfiguring
from .questions import (
    action_for_value,
    answer_fits_pending as _answer_fits_pending,
    coerce_pending_answer as _coerce_answer,
    exact_answer as contract_exact_answer,
    manual_literal_violation,
    matches_numbered_option as _matches_numbered_option,
    pending_option_value_exists as _pending_option_value_exists,
    value_satisfies_pending_contract as _value_satisfies_pending_contract,
    render_question as _render_question,
)
from .response import (
    append_active_question_once as _append_active_question_once,
    finalize_turn_response as _finalize_turn_response,
    question_is_actionable_in_responses as _question_is_actionable_in_responses,
    without_superseded_question as _without_superseded_question,
)
from .queue import (
    action_can_run_while_pending as _action_can_run_while_pending,
    order_action_queue as _order_action_queue,
)
from .admission import (
    _action_answers_pending_contract,
    _action_satisfies_pending_manual_effect,
    _drop_conflicting_answer_actions,
    _has_meaningful_queue,
    _normalized_action_queue,
    _validate_action_plan,
    _validate_admission_transaction,
    reconcile_admission_coverage,
)
from .input_values import normalize_target_mode, target_mode_evidence_matches
from agent.workflows.group_registry import (
    GROUP_ORDER,
    GROUP_SPEC_BY_NAME,
    group_for_field,
    invalidation_targets,
    is_user_navigable_group,
    reconfiguration_question_for_field,
)
QUEUE_RESUME_PENDING_IDS = {
    "inferred_config_review",
    "target_mode_change_confirm",
    "chain_change_confirm",
    "chain_ambiguity_confirm",
    "unknown_chain_identity_confirm",
}

_ADMISSION_METADATA_KEYS = (
    "_semantic_admission_receipt",
    "_proposal_field_receipts",
    "_proposal_transaction_hashes",
    "_admission_action_id",
    "_transaction_action_ids",
    "_plan_transaction_hash",
    "_merged_origin_texts",
)


def apply_coordinator_action(state: AgentGraphState, action: ActionProposal) -> HandlerResult:
    """Apply navigation actions owned by the single workflow coordinator."""

    if action.action_type == "resume_current_flow":
        pending = dict(state.get("pending_question") or {})
        if not pending:
            return HandlerResult(blocker="no active typed question is available to resume")
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            visible_result=_render_question(pending, state.get("language", "en")),
            pending_question=pending,
            completion="unchanged",
            stop_after_response=True,
        )
    if action.action_type == "request_config_field_input":
        field = str(action.arguments.get("config_field") or "").strip()
        group = group_for_field(field)
        question_id = reconfiguration_question_for_field(field)
        if not group or not question_id:
            return HandlerResult(blocker=f"field is not registered for typed reconfiguration: {field or '<missing>'}")
        runtime = DOMAIN_RUNTIME.get(GROUP_OWNER.get(group, ""))
        question = (
            runtime.field_question_factory(state, group, field)
            if runtime and runtime.field_question_factory
            else None
        )
        owner_spec = GROUP_SPEC_BY_NAME.get(group)
        if (
            not question
            or not owner_spec
            or str(question.get("group") or "") != group
            or str(question.get("id") or "") not in owner_spec.questions
            or str(question.get("field") or "") not in owner_spec.fields
        ):
            return HandlerResult(blocker=f"registered field question is unavailable: {field}")
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            visible_result=_render_question(question, state.get("language", "en")),
            pending_question=question,
            next_group=group,
            field_reconfiguration_command=FieldReconfigurationCommand(
                group=group,
                config_field=field,
                prerequisite_question_id=(
                    str(question.get("id") or "")
                    if str(question.get("field") or "") != field
                    else ""
                ),
            ),
            completion="blocked",
            stop_after_response=True,
        )
    if action.action_type == "change_group":
        group = str(action.arguments.get("group") or "").strip()
        if group not in ALLOWED_GROUPS or not is_user_navigable_group(group):
            return HandlerResult(blocker=f"group is not a user-navigable destination: {group or '<missing>'}")
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            navigation_command=NavigationCommand(
                operation="change_group",
                target_group=group,
                origin_group=str(action.arguments.get("queue_origin_group") or "").strip(),
            ),
            completion="completed",
            stop_after_response=True,
        )
    if action.action_type == "go_back":
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            navigation_command=NavigationCommand(operation="go_back"),
            completion="completed",
            stop_after_response=True,
        )
    if action.action_type == "queue_workflow_goal":
        target_mode = normalize_target_mode(action.arguments.get("target_mode"))
        goal = str(action.arguments.get("goal") or "").strip()
        source = str(action.arguments.get("source_evidence") or "").strip()
        if not target_mode or not goal or not source:
            return HandlerResult(blocker="queued workflow goal requires target_mode, goal, and source_evidence")
        goals = [dict(item) for item in state.get("workflow_goals") or [] if isinstance(item, dict)]
        candidate = {"target_mode": target_mode, "goal": goal, "source_evidence": source}
        already_queued = any(
            item.get("target_mode") == target_mode and item.get("goal") == goal
            for item in goals
        )
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            workflow_goal_command=(
                None
                if already_queued
                else WorkflowGoalCommand(operation="enqueue", goal=candidate)
            ),
            visible_result=_localized(
                state.get("language", "en"),
                f"已保存后续目标：完成当前流程后进入 `{target_mode}`（{goal}）。",
                f"Saved a later goal: enter `{target_mode}` after the current workflow ({goal}).",
            ),
            completion="completed",
        )
    if action.action_type in {"activate_next_workflow_goal", "discard_next_workflow_goal"}:
        goals = [dict(item) for item in state.get("workflow_goals") or [] if isinstance(item, dict)]
        if not goals:
            return HandlerResult(blocker="no queued workflow goal is available")
        goal = goals.pop(0)
        if action.action_type == "discard_next_workflow_goal":
            return HandlerResult(
                consumed_action_ids=(action.action_id,),
                workflow_goal_command=WorkflowGoalCommand(operation="remove_first"),
                visible_result=_localized(
                    state.get("language", "en"),
                    "已移除最早保存的后续测试目标。",
                    "Removed the oldest saved workflow goal.",
                ),
                completion="completed",
            )
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            workflow_goal_command=WorkflowGoalCommand(operation="remove_first"),
            followup_actions=({
                "type": "choose_target_mode",
                "target_mode": str(goal.get("target_mode") or ""),
                "target_mode_explicit": True,
                "source_evidence": str(goal.get("source_evidence") or "").strip(),
                "selection_contract_verified": True,
                "confidence": "high",
            },),
            visible_result=_localized(
                state.get("language", "en"),
                f"正在切换到已保存的后续目标：{goal.get('goal') or goal.get('target_mode')}。",
                f"Activating the saved workflow goal: {goal.get('goal') or goal.get('target_mode')}.",
            ),
            completion="completed",
        )
    if action.action_type == "answer_pending":
        return HandlerResult(blocker="answer_pending must be dispatched by the pending-question coordinator")
    return HandlerResult(blocker=f"unsupported coordinator action: {action.action_type}")

COORDINATOR_RUNTIME = DomainRuntime(apply_action=apply_coordinator_action)


def _set_turn_phase(state: AgentGraphState, phase: str, reason: str = "") -> AgentGraphState:
    control = dict(state.get("control") or {})
    control["phase"] = phase
    if reason:
        control["reason"] = reason
    state["control"] = control
    return state


def _record_admitted_action(
    state: AgentGraphState,
    action: Mapping[str, Any],
    *,
    source: str,
) -> None:
    """Record only actions admitted for the current user turn.

    ``completed_actions`` is execution history for the current graph pass and
    can be replaced by reset-style transitions.  Live evidence therefore uses
    this turn-scoped admission ledger as the authoritative control-plane fact.
    """

    action_type = str(
        action.get("type")
        or action.get("action_type")
        or ""
    ).strip()
    if not action_type:
        return
    turn = state.setdefault("turn_context", {})
    admitted = turn.setdefault("admitted_actions", [])
    action_id = str(action.get("action_id") or "").strip()
    identity = (action_type, action_id)
    if any(
        (str(item.get("type") or ""), str(item.get("action_id") or "")) == identity
        for item in admitted
        if isinstance(item, Mapping)
    ):
        return
    admitted_action = {
        "type": action_type,
        "action_id": action_id,
        "source": source,
    }
    spec = ACTION_BY_TYPE.get(action_type)
    if spec is not None:
        admitted_action["owner"] = spec.owner
        admitted_action["effect"] = spec.effect
    argument_payload = {
        str(key): value
        for key, value in action.items()
        if key not in ACTION_METADATA_FIELDS
        and key not in TRUSTED_ACTION_METADATA_FIELDS
        and not str(key).startswith("_")
    }
    admitted_action["argument_names"] = sorted(argument_payload)
    admitted_action["arguments_hash"] = hashlib.sha256(
        json.dumps(
            argument_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    admitted_action["source_unit_ids"] = [
        str(item)
        for item in (
            action.get("source_unit_ids")
            or action.get("_source_unit_ids")
            or ()
        )
        if str(item)
    ]
    source_evidence = str(action.get("source_evidence") or "")
    admitted_action["source_hash"] = hashlib.sha256(
        source_evidence.encode("utf-8")
    ).hexdigest()
    admission_receipt = action.get("_semantic_admission_receipt")
    if isinstance(admission_receipt, Mapping):
        admitted_action["admission_receipt_id"] = str(
            admission_receipt.get("receipt_id") or ""
        )
    if action_type == "change_group":
        target_group = str(
            action.get("group")
            or (action.get("arguments") or {}).get("group")
            or ""
        ).strip()
        if target_group:
            admitted_action["group"] = target_group
    admitted.append(admitted_action)
    receipt = dict(state.get("turn_receipt") or {})
    if not str(receipt.get("turn_id") or ""):
        return
    admitted_ids = list(receipt.get("admitted_action_ids") or [])
    semantic_order = list(receipt.get("semantic_order") or [])
    owners = dict(receipt.get("owner_bindings") or {})
    action_units = {
        str(key): list(value)
        for key, value in dict(receipt.get("action_unit_bindings") or {}).items()
    }
    unit_actions = {
        str(key): list(value)
        for key, value in dict(receipt.get("unit_action_bindings") or {}).items()
    }
    if action_id and action_id not in admitted_ids:
        admitted_ids.append(action_id)
        semantic_order.append(action_id)
    if action_id and spec is not None:
        owners[action_id] = (
            GROUP_OWNER.get(
                str((state.get("pending_question") or {}).get("group") or ""),
                spec.owner,
            )
            if action_type == "answer_pending"
            else spec.owner
        )
    source_unit_ids = [
        str(item)
        for item in (
            action.get("source_unit_ids")
            or action.get("_source_unit_ids")
            or ()
        )
        if str(item)
    ]
    if action_id and source_unit_ids:
        action_units[action_id] = list(dict.fromkeys(source_unit_ids))
        for unit_id in source_unit_ids:
            bindings = unit_actions.setdefault(unit_id, [])
            if action_id not in bindings:
                bindings.append(action_id)
    receipt["admitted_action_ids"] = admitted_ids
    receipt["semantic_order"] = semantic_order
    receipt["owner_bindings"] = owners
    receipt["action_unit_bindings"] = action_units
    receipt["unit_action_bindings"] = unit_actions
    receipt["status"] = "planned"
    state["turn_receipt"] = receipt


def _receipt_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _append_control_receipt(
    state: AgentGraphState,
    receipt_type: str,
    payload: Mapping[str, Any],
) -> None:
    """Append one secret-free observation emitted by the current owner."""

    body = {
        "receipt_type": str(receipt_type),
        "turn_index": int(state.get("turn_index") or 0),
        **deepcopy(dict(payload)),
    }
    body["receipt_id"] = _receipt_hash(body)
    turn_context = dict(state.get("turn_context") or {})
    receipts = [
        dict(item)
        for item in turn_context.get("control_receipts") or ()
        if isinstance(item, Mapping)
    ]
    if not any(item.get("receipt_id") == body["receipt_id"] for item in receipts):
        receipts.append(body)
    turn_context["control_receipts"] = receipts
    state["turn_context"] = turn_context


def _append_domain_control_receipts(
    state: AgentGraphState,
    receipts_to_commit: tuple[Mapping[str, Any], ...],
    *,
    owner: str,
) -> None:
    """Commit only original, owner-validated domain observation receipts."""

    turn_index = int(state.get("turn_index") or 0)
    turn_context = dict(state.get("turn_context") or {})
    receipts = [
        dict(item)
        for item in turn_context.get("control_receipts") or ()
        if isinstance(item, Mapping)
    ]
    known_ids = {
        str(item.get("receipt_id") or "")
        for item in receipts
        if str(item.get("receipt_id") or "")
    }
    for raw in receipts_to_commit:
        receipt = deepcopy(dict(raw))
        valid, reason = validate_domain_control_receipt(
            receipt,
            handler_owner=owner,
            turn_index=turn_index,
        )
        if not valid:
            raise StateInvariantError(f"invalid domain control receipt: {reason}")
        receipt_id = str(receipt.get("receipt_id") or "")
        if receipt_id not in known_ids:
            receipts.append(receipt)
            known_ids.add(receipt_id)
    turn_context["control_receipts"] = receipts
    state["turn_context"] = turn_context


def _record_semantic_plan_receipt(
    state: AgentGraphState,
    semantic_units: list[Mapping[str, Any]],
    pending_choice_contracts: list[Mapping[str, Any]],
) -> None:
    """Persist lossless unit coverage before any admitted action can execute."""

    receipt = dict(state.get("turn_receipt") or {})
    if not receipt:
        return
    normalized_units = [
        deepcopy(dict(unit))
        for unit in semantic_units
        if str(unit.get("unit_id") or "")
    ]
    unresolved = [
        str(unit.get("unit_id") or "")
        for unit in normalized_units
        if str(unit.get("disposition") or "") == "unresolved"
    ]
    omission_checks: list[dict[str, Any]] = []
    for unit in normalized_units:
        unit_id = str(unit.get("unit_id") or "")
        disposition = str(unit.get("disposition") or "")
        indexes = [
            int(index)
            for index in unit.get("action_indexes") or []
            if isinstance(index, int) and not isinstance(index, bool)
        ]
        verdict = (
            "covered"
            if disposition == "action" and indexes
            else "context"
            if disposition == "context" and not indexes
            else "unresolved"
            if disposition == "unresolved" and not indexes
            else "invalid"
        )
        omission_checks.append({
            "unit_id": unit_id,
            "disposition": disposition,
            "action_indexes": indexes,
            "verdict": verdict,
        })
    pending_verdicts = [
        {
            "action_index": contract.get("action_index"),
            "admission_action_id": str(
                contract.get("admission_action_id") or ""
            ),
            "candidate_value": deepcopy(contract.get("candidate_value")),
            "semantic_unit_ids": [
                str(unit.get("unit_id") or "")
                for unit in contract.get("semantic_units") or []
                if isinstance(unit, Mapping)
                and str(unit.get("unit_id") or "")
            ],
            "verdict": "admitted",
        }
        for contract in pending_choice_contracts
        if isinstance(contract, Mapping)
    ]
    receipt["semantic_units"] = normalized_units
    receipt["unresolved_units"] = unresolved
    receipt["pending_candidate_verdicts"] = pending_verdicts
    receipt["sibling_omission_checks"] = omission_checks
    state["turn_receipt"] = receipt


def prepare_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Create one immutable turn snapshot before any routing or mutation."""

    state = _copy_state(state)
    state["action_queue"] = _durable_actions(state.get("action_queue") or [])
    resumed_phase = _inflight_transition_phase(state)
    if resumed_phase:
        validate_state(state)
        return _set_turn_phase(
            state,
            resumed_phase,
            "inflight_transition_resumed",
        )
    state = _apply_handler_result(state, reconcile_execution_state(state), owner="execution")
    pending_owner = str((state.get("pending_question") or {}).get("group") or "").strip()
    if pending_owner:
        state["active_group"] = pending_owner
    state["turn_index"] = int(state.get("turn_index") or 0) + 1
    text = str(state.get("last_user_input") or "").strip()
    clauses = segment_user_turn(text)
    clause_shapes = {str(clause.input_shape or "prose") for clause in clauses}
    input_shape = (
        "structured"
        if clause_shapes == {"structured"}
        else "mixed"
        if "structured" in clause_shapes
        else "prose"
    )
    state["input_shape"] = input_shape
    state["visible_response"] = []
    state["current_action"] = {}
    state["completed_actions"] = []
    state["action_errors"] = []
    turn = adjudicate_turn(state, text)
    state["turn_context"] = {
        "id": int(state.get("turn_index") or 0),
        "kind": str(turn.kind),
        "text": text,
        "input_shape": input_shape,
        "origin_group": str(state.get("active_group") or ""),
        "pending_snapshot": dict(state.get("pending_question") or {}),
        "admitted_actions": [],
    }
    turn_id = (
        f"{state.get('thread_id') or 'default'}:"
        f"{int(state.get('turn_index') or 0)}"
    )
    state["turn_receipt"] = turn_receipt_to_dict(
        TurnReceipt(
            turn_id=turn_id,
            input_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            language=str(state.get("language") or "en"),
            input_shape=input_shape,
            clauses=tuple(clause.as_dict() for clause in clauses),
            pending_before=deepcopy(state.get("pending_question") or {}),
        )
    )
    state["proposed_actions"] = []
    return _set_turn_phase(state, "adjudicate")


def _inflight_transition_phase(state: AgentGraphState) -> str:
    """Return the first incomplete checkpointed action phase, if any."""

    selected = dict(state.get("selected_action") or {})
    if not selected:
        return ""
    if state.get("pending_domain_result"):
        return "commit"
    intent = dict(state.get("side_effect_intent") or {})
    receipt = dict(state.get("side_effect_receipt") or {})
    if receipt:
        return "commit_receipt"
    if not intent:
        return "execute"
    status = str(intent.get("status") or "")
    if status == "prepared":
        return "invoke_effect"
    if status == "invoking":
        return "perform_effect"
    return ""


def adjudicate_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Handle only deterministic contracts; free text proceeds to planning."""

    text = str((state.get("turn_context") or {}).get("text") or "")
    runtime_action = dict(
        (state.get("turn_context") or {}).get("runtime_action") or {}
    )
    if runtime_action:
        return _admit_deterministic_action(
            state,
            runtime_action,
            source="runtime_command",
        )
    turn_kind = str((state.get("turn_context") or {}).get("kind") or "free_text")
    language = str(state.get("language") or "en")
    collecting = state.get("evidence_collection") or {}

    if turn_kind == "evidence_continuation":
        if not text:
            state = _apply_handler_result(state, prompt_evidence_collection_waiting(state, collecting), owner="analysis")
            return _set_turn_phase(state, "compose", "evidence_waiting")
        if is_evidence_completion_command(text):
            return _admit_deterministic_action(
                state,
                {
                    "type": "finish_evidence_collection",
                    "source_evidence": text,
                    "confidence": "high",
                },
                source="evidence_transport",
            )
        return _set_turn_phase(state, "plan", "typed_evidence_collection_turn")

    if turn_kind == "empty":
        return _set_turn_phase(state, "fallback", "empty_turn")

    pending = state.get("pending_question") or {}
    input_shape = str((state.get("turn_context") or {}).get("input_shape") or "prose")

    # Evidence framing is a terminal transport concern: collect a complete
    # multiline block before asking the semantic planner to classify it. The
    # completed block re-enters the normal semantic owner as one turn.
    if pending and str(pending.get("kind") or "") == "evidence" and should_start_evidence_collection(text):
        return _admit_deterministic_action(
            state,
            {
                "type": "start_evidence_collection",
                "evidence": text,
                "source_evidence": text,
                "confidence": "high",
            },
            source="evidence_transport",
        )

    pending_fits = bool(pending and _answer_fits_pending(text, pending))
    pending_owns_structured_input = bool(
        pending_fits
        and pending.get("structured_input_owner") is True
        and input_shape == "structured"
    )

    # Structured configuration is review-owned unless the active typed domain
    # contract explicitly owns a structured answer. This keeps environment
    # proposals on the inferred-config review path while allowing contracts
    # such as RPC weight maps to validate their own declared data shape without
    # model arbitration.
    if input_shape in {"structured", "mixed"} and not pending_owns_structured_input:
        return _set_turn_phase(state, "plan", "structured_turn_requires_semantic_ownership")

    violation = manual_literal_violation(text, pending) if pending else {}
    if violation:
        max_length = int(violation.get("max_length") or 0)
        if violation.get("code") == "max_length":
            message_zh = f"输入无效：该字段最多允许 {max_length} 个字符。当前问题保持不变。"
            message_en = f"Invalid input: this field allows at most {max_length} characters. The current question remains active."
        else:
            message_zh = "输入无效：请输入大于 0 的数值。当前问题保持不变。"
            message_en = "Invalid input: enter a number greater than zero. The current question remains active."
        state["visible_response"] = [_localized(
            language,
            message_zh,
            message_en,
        ), _render_question(pending, language)]
        return _set_turn_phase(state, "compose", "pending_literal_rejected")
    if pending_fits:
        pending_question_id = str(pending.get("id") or "")
        resume_queue = bool(
            pending.get("resume_action_queue")
            or pending_question_id in QUEUE_RESUME_PENDING_IDS
        )
        matched, selected_value = contract_exact_answer(text, pending)
        action = {
            "type": "answer_pending",
            "answer": text,
            "source_evidence": text,
            "confidence": "high",
            "selection_contract_verified": True,
        }
        if matched:
            action["selected_value"] = selected_value
        else:
            action["selected_value"] = _coerce_answer(text, pending)
        selected_option = next(
            (
                option
                for option in pending.get("options") or ()
                if isinstance(option, Mapping)
                and option.get("value") == action["selected_value"]
            ),
            {},
        )
        _append_control_receipt(
            state,
            "pending_resolution",
            {
                "pending_id": pending_question_id,
                "pending_group": str(pending.get("group") or ""),
                "pending_contract_hash": _receipt_hash(pending),
                "resolution_path": (
                    "exact_contract" if matched else "typed_manual_value"
                ),
                "selected_option_id": str(
                    selected_option.get("id")
                    or selected_option.get("value")
                    or ""
                ),
                "selected_value_hash": _receipt_hash(action["selected_value"]),
                "input_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "normalizer": str(
                    (
                        (pending.get("validation") or {}).get("normalization")
                        or "exact_contract"
                    )
                    if matched
                    else "declared_value_type"
                ),
                "verdict": "accepted",
            },
        )
        action = assign_action_ids(
            f"{state.get('thread_id') or 'default'}:{int(state.get('turn_index') or 0)}",
            text,
            [action],
        )[0]
        source_clauses = [
            dict(clause)
            for clause in (state.get("turn_receipt") or {}).get("clauses") or []
            if isinstance(clause, Mapping)
            and str(clause.get("clause_id") or "")
        ]
        source_unit_ids = [
            str(clause.get("clause_id") or "")
            for clause in source_clauses
        ]
        action["_source_unit_ids"] = source_unit_ids
        _record_semantic_plan_receipt(
            state,
            [
                {
                    "unit_id": str(clause.get("clause_id") or ""),
                    "clause_id": str(clause.get("clause_id") or ""),
                    "source_text": str(clause.get("text") or ""),
                    "start": 0,
                    "end": len(str(clause.get("text") or "")),
                    "disposition": "action",
                    "action_indexes": [0],
                }
                for clause in source_clauses
            ],
            [],
        )
        scope = (
            f"{state.get('thread_id') or 'default'}:"
            f"{int(state.get('turn_index') or 0)}"
        )
        action = action_envelope_to_dict(
            _build_action_envelope(
                state,
                action,
                semantic_order=0,
                submitted_turn_index=int(state.get("turn_index") or 0),
                origin_group=str(state.get("active_group") or ""),
                origin_text=text,
                plan_scope=scope,
            )
        )
        existing = (
            [dict(item) for item in state.get("action_queue") or [] if isinstance(item, Mapping)]
            if resume_queue
            else []
        )
        state["action_queue"] = [action, *existing]
        _record_admitted_action(
            state,
            action,
            source="pending_question_contract",
        )
        return _set_turn_phase(state, "execute", "pending_answer_admitted")

    if pending and str(pending.get("kind") or "") == "device" and text.lower() in {"y", "yes", "n", "no"}:
        state["visible_response"] = [_localized(
            language,
            "这个问题需要选择编号或直接输入设备/接口名；`Y/N` 不能唯一确定候选。",
            "This question needs an option number or a device/interface name; `Y/N` does not uniquely identify a candidate.",
        ), _render_question(pending, language)]
        return _set_turn_phase(state, "compose", "ambiguous_device_answer")

    if pending and str(pending.get("kind") or "") in {"numbered_choice", "yes_no"}:
        bare = text.strip().rstrip(".)、。 ").strip()
        if str(pending.get("kind") or "") == "numbered_choice" and str(pending.get("id") or "") != "unknown_chain_identity_confirm" and bare.casefold() in {"y", "yes", "n", "no"}:
            state["visible_response"] = [_localized(
                language,
                "这是编号选项，不是 Y/N 确认。请回复显示的编号或选项名称。",
                "This is a numbered menu, not a Y/N confirmation. Reply with a displayed number or option name.",
            ), _render_question(pending, language)]
            return _set_turn_phase(state, "compose", "wrong_choice_shape")
        if bare.isdigit() and not _matches_numbered_option(bare, pending):
            option_count = len(pending.get("options") or [])
            state["visible_response"] = [_localized(
                language,
                f"请输入 1 到 {option_count} 之间的选项编号，或直接说明要切换到哪个配置项。",
                f"Please enter an option number between 1 and {option_count}, or say which configuration area to switch to.",
            ), _render_question(pending, language)]
            return _set_turn_phase(state, "compose", "invalid_choice")

    if pending and str(pending.get("kind") or "") == "numbered_choice" and _looks_like_assignment_answer(text):
        state["visible_response"] = [_localized(
            language,
            "这条回复不像当前问题的答案。我会先保持当前问题不变；你也可以直接说明要切换到哪个配置项。",
            "That reply does not look like an answer to the current question. I will keep the current question active; you can also describe which configuration area to change.",
        ), _render_question(pending, language)]
        return _set_turn_phase(state, "compose", "assignment_does_not_answer_choice")
    return _set_turn_phase(state, "plan", "semantic_input")


def _admit_deterministic_action(
    state: AgentGraphState,
    action: Mapping[str, Any],
    *,
    source: str,
) -> AgentGraphState:
    """Admit one exact local contract through the normal graph lifecycle."""

    text = str((state.get("turn_context") or {}).get("text") or "")
    scope = f"{state.get('thread_id') or 'default'}:{int(state.get('turn_index') or 0)}"
    admitted = assign_action_ids(scope, text, [dict(action)])[0]
    source_clauses = [
        dict(clause)
        for clause in (state.get("turn_receipt") or {}).get("clauses") or []
        if isinstance(clause, Mapping)
        and str(clause.get("clause_id") or "")
    ]
    source_unit_ids = [
        str(clause.get("clause_id") or "")
        for clause in source_clauses
    ]
    admitted["_source_unit_ids"] = source_unit_ids
    _record_semantic_plan_receipt(
        state,
        [
            {
                "unit_id": str(clause.get("clause_id") or ""),
                "clause_id": str(clause.get("clause_id") or ""),
                "source_text": str(clause.get("text") or ""),
                "start": 0,
                "end": len(str(clause.get("text") or "")),
                "disposition": "action",
                "action_indexes": [0],
            }
            for clause in source_clauses
        ],
        [],
    )
    admitted = action_envelope_to_dict(
        _build_action_envelope(
            state,
            admitted,
            semantic_order=0,
            submitted_turn_index=int(state.get("turn_index") or 0),
            origin_group=str(state.get("active_group") or ""),
            origin_text=text,
            plan_scope=scope,
        )
    )
    existing = [
        dict(item)
        for item in state.get("action_queue") or []
        if isinstance(item, Mapping)
    ]
    state["action_queue"] = [admitted, *existing]
    _record_admitted_action(state, admitted, source=source)
    return _set_turn_phase(state, "execute", "deterministic_action_admitted")


def plan_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Call the configured LLM once to propose typed actions."""

    text = str((state.get("turn_context") or {}).get("text") or "")
    queue = resolve_action_queue(state, text)
    _append_control_receipt(
        state,
        "semantic_planner",
        {
            "input_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "pending_contract_hash": _receipt_hash(
                state.get("pending_question") or {}
            ),
            "resolver_invoked": True,
            "result_reason_hash": hashlib.sha256(
                str(queue.get("reason") or "").encode("utf-8")
            ).hexdigest(),
            "planned_action_types": [
                str(item.get("type") or "")
                for item in queue.get("actions") or ()
                if isinstance(item, Mapping) and str(item.get("type") or "")
            ],
            "semantic_units": [
                {
                    "unit_id": str(item.get("unit_id") or ""),
                    "disposition": str(item.get("disposition") or ""),
                }
                for item in queue.get("semantic_units") or ()
                if isinstance(item, Mapping) and str(item.get("unit_id") or "")
            ],
            "planner_metrics": {
                str(key): value
                for key, value in dict(queue.get("planner_metrics") or {}).items()
                if isinstance(value, (int, float, bool))
            },
        },
    )
    if str(queue.get("reason") or "") == "resolver failed":
        record = model_provider_failure_record("resolver_failed")
        state["failure_recovery"] = {"status": "pending", "record": record}
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            "当前模型服务无法完成意图识别。本轮没有修改配置或推进 workflow；请检查模型额度/连接后重试。",
            "The model service could not resolve this request. No configuration or workflow state changed; check model quota/connectivity and retry.",
        )]
        return _set_turn_phase(state, "compose", "planner_failed")
    actions = _normalized_action_queue(queue)
    state.setdefault("turn_context", {})["pending_choice_contracts"] = [
        dict(row)
        for row in queue.get("pending_choice_contracts") or []
        if isinstance(row, Mapping)
    ]
    state["turn_context"]["semantic_units"] = [
        dict(row)
        for row in queue.get("semantic_units") or []
        if isinstance(row, Mapping)
    ]
    for index, action in enumerate(actions):
        action["_source_unit_ids"] = [
            str(unit.get("unit_id") or "")
            for unit in state["turn_context"]["semantic_units"]
            if index in (
                unit.get("action_indexes")
                if isinstance(unit.get("action_indexes"), list)
                else []
            )
            and str(unit.get("unit_id") or "")
        ]
    _record_semantic_plan_receipt(
        state,
        state["turn_context"]["semantic_units"],
        state["turn_context"]["pending_choice_contracts"],
    )
    state["proposed_actions"] = actions
    return _set_turn_phase(state, "admit", "planner_completed")


def admit_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Admit every reviewed action to the graph-owned execution queue."""

    text = str((state.get("turn_context") or {}).get("text") or "")
    actions = _validate_action_plan(state, list(state.get("proposed_actions") or []))
    _validate_admission_transaction(state, actions, current_submission=True)
    scope = f"{state.get('thread_id') or 'default'}:{int(state.get('turn_index') or 0)}"
    actions = assign_action_ids(scope, text, actions)
    for index, action in enumerate(actions):
        action["_plan_scope"] = scope
        action["_plan_index"] = index
        action["_submitted_turn_index"] = int(state.get("turn_index") or 0)
        action["_origin_text"] = text
        action["_source_unit_ids"] = list(dict.fromkeys(
            str(unit_id)
            for unit_id in action.get("_source_unit_ids") or []
            if str(unit_id)
        ))
    receipt = reconcile_admission_coverage(
        state.get("turn_receipt") or {},
        actions,
    )
    pending_verdicts = [
        dict(row)
        for row in receipt.get("pending_candidate_verdicts") or []
        if isinstance(row, Mapping)
    ]
    for verdict in pending_verdicts:
        index = verdict.get("action_index")
        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < len(actions)
        ):
            verdict["action_id"] = str(actions[index].get("action_id") or "")
    omission_checks = [
        dict(row)
        for row in receipt.get("sibling_omission_checks") or []
        if isinstance(row, Mapping)
    ]
    for check in omission_checks:
        unit_id = str(check.get("unit_id") or "")
        check["action_ids"] = [
            str(action.get("action_id") or "")
            for action in actions
            if unit_id
            and unit_id in {
                str(item)
                for item in action.get("_source_unit_ids") or []
            }
        ]
    receipt["pending_candidate_verdicts"] = pending_verdicts
    receipt["sibling_omission_checks"] = omission_checks
    state["turn_receipt"] = receipt
    origin_group = str(state.get("active_group") or "")
    turn_local_actions = [item for item in actions if action_is_turn_local(item)]
    durable_actions = [item for item in actions if not action_is_turn_local(item)]
    if not _has_meaningful_queue(actions):
        pending = state.get("pending_question") or {}
        if pending:
            state["visible_response"] = [_localized(
                state.get("language", "en"),
                "这条回复不像当前问题的答案。我会先保持当前问题不变；你也可以直接说明要切换到哪个配置项。",
                "That reply does not look like an answer to the current question. I will keep the current question active; you can also describe which configuration area to change.",
            ), _render_question(pending, state.get("language", "en"))]
            return _set_turn_phase(state, "compose", "semantic_input_not_admitted")
        return _set_turn_phase(state, "fallback", "no_admitted_actions")
    for action in actions:
        _record_admitted_action(state, action, source="semantic_plan")
    pending = state.get("pending_question") or {}
    existing_queue = [
        _admission_action_from_envelope(item)
        for item in state.get("action_queue") or []
        if isinstance(item, dict)
    ]
    accepted_types = {
        str(item).strip()
        for item in pending.get("accepted_action_types") or []
        if str(item).strip()
    }
    answers_active_semantic_contract = bool(
        accepted_types
        and any(str(item.get("type") or "") in accepted_types for item in durable_actions)
    )
    extends_active_config_review = bool(
        str(pending.get("id") or "") == "inferred_config_review"
        and any(
            str(item.get("type") or "") == "propose_config_values"
            for item in durable_actions
        )
    )
    if answers_active_semantic_contract and not pending.get("resume_action_queue"):
        # The question contract is the sole authority for retaining deferred
        # work. Without resume_action_queue, accepting the current semantic
        # answer supersedes every pre-existing queue entry regardless of
        # whether a legacy checkpoint happens to contain plan-like metadata.
        existing_queue = []
    ordered_actions = _order_action_queue(state, [
        *turn_local_actions,
        *_merge_durable_action_queue(existing_queue, durable_actions),
    ])
    ordered_queue = _serialize_admitted_actions(
        state,
        ordered_actions,
        default_origin_text=text,
        default_origin_group=origin_group,
        default_plan_scope=scope,
    )
    if (
        str(pending.get("id") or "") == "inferred_config_review"
        and not any(str(item.get("type") or "") == "answer_pending" for item in durable_actions)
        and not extends_active_config_review
        and not any(
            _action_can_run_while_pending(state, dict(item))
            for item in ordered_actions
        )
    ):
        state["action_queue"] = ordered_queue
        return _set_turn_phase(state, "compose", "config_review_barrier")
    state["action_queue"] = ordered_queue
    state["completed_actions"] = []
    state["action_errors"] = []
    if state.get("action_queue"):
        return _set_turn_phase(state, "execute", "actions_admitted")
    return _set_turn_phase(state, "fallback", "no_durable_actions")


def select_action_step(state: AgentGraphState) -> AgentGraphState:
    """Select one durable action without applying domain state."""

    state = _copy_state(state)
    queue = [
        dict(item)
        for item in state.get("action_queue") or []
        if isinstance(item, Mapping)
    ]
    state["action_queue"] = queue
    state["selected_action"] = {}
    state["current_action"] = {}
    if not queue:
        if state.get("pending_question"):
            return _set_turn_phase(state, "compose", "queue_blocked_or_complete")
        return _set_turn_phase(state, "fallback", "queue_complete")
    pending = dict(state.get("pending_question") or {})
    if pending:
        eligible_index = next(
            (
                index
                for index, queued_action in enumerate(queue)
                if _action_can_run_while_pending(
                    state,
                    _queue_action(queued_action),
                )
            ),
            None,
        )
        if eligible_index is None:
            return _set_turn_phase(state, "compose", "pending_barrier_blocks_queue")
        if eligible_index:
            action = queue.pop(eligible_index)
            queue.insert(0, action)
            state["action_queue"] = queue
        queued_envelope = dict(queue[0])
    else:
        queued_envelope = dict(queue[0])
    envelope = action_envelope_from_dict(queued_envelope)
    action = _action_from_envelope(queued_envelope)
    if lifecycle_rejected_action_indexes(state, [action]):
        state["action_queue"] = queue[1:]
        state.setdefault("action_errors", []).append({
            "action": action,
            "error": "lifecycle_inapplicable",
        })
        return _set_turn_phase(state, "execute", "lifecycle_action_rejected")
    action_id = str(action.get("action_id") or "")
    if action_id and action_id in set(state.get("applied_action_ids") or []):
        state["action_queue"] = queue[1:]
        return _set_turn_phase(state, "execute", "already_applied_action_skipped")
    spec = ACTION_BY_TYPE.get(envelope.action_type)
    if spec is None:
        state["action_queue"] = queue[1:]
        state.setdefault("action_errors", []).append({
            "action": action,
            "error": "unregistered_action",
        })
        return _set_turn_phase(state, "execute", "unregistered_action_rejected")
    _record_admitted_action(state, action, source="durable_queue")
    state["selected_action"] = action_envelope_to_dict(
        replace(envelope, status="selected")
    )
    state["current_action"] = action
    control = dict(state.get("control") or {})
    control["selected_owner"] = envelope.owner
    state["control"] = control
    validate_state(state)
    return _set_turn_phase(state, "route_owner", "action_selected")


def _build_action_envelope(
    state: AgentGraphState,
    action: Mapping[str, Any],
    *,
    semantic_order: int | None = None,
    submitted_turn_index: int | None = None,
    origin_group: str | None = None,
    origin_text: str | None = None,
    plan_scope: str | None = None,
) -> ActionEnvelope:
    """Separate validated action arguments from Harness-owned provenance."""

    action_type = str(action.get("type") or "")
    spec = ACTION_BY_TYPE.get(action_type)
    if spec is None:
        raise StateInvariantError(f"cannot envelope unregistered action: {action_type}")
    arguments = {
        str(key): deepcopy(value)
        for key, value in action.items()
        if key not in {"type", "action_id", "confidence", "reason"}
        and not str(key).startswith("_")
    }
    exact_origin_text = str(
        origin_text
        if origin_text is not None
        else action.get("_origin_text")
        or (state.get("turn_context") or {}).get("text")
        or ""
    )
    source_evidence = str(
        arguments.get("source_evidence")
        or exact_origin_text
        or ""
    )
    receipt = action.get("_semantic_admission_receipt")
    source_unit_ids = tuple(
        str(item)
        for item in (
            action.get("_source_unit_ids")
            or (
                (receipt or {}).get("source_unit_ids")
                if isinstance(receipt, Mapping)
                else ()
            )
        )
        or ()
    )
    effect_kind = (
        "external"
        if spec.effect == "execution"
        else "read_only"
        if spec.effect == "read_only"
        else "pure"
    )
    action_id = str(action.get("action_id") or "")
    pending = state.get("pending_question") or {}
    owner = spec.owner
    if action_type == "answer_pending":
        owner = (
            "environment"
            if str(pending.get("id") or "") == "inferred_config_review"
            else GROUP_OWNER.get(str(pending.get("group") or ""), spec.owner)
        )
    return ActionEnvelope(
        action_id=action_id,
        action_type=action_type,
        owner=owner,
        target_group=spec.target_group,
        arguments=arguments,
        confidence=str(action.get("confidence") or "medium"),  # type: ignore[arg-type]
        reason=str(action.get("reason") or ""),
        source_unit_ids=source_unit_ids,
        source_evidence_hash=hashlib.sha256(source_evidence.encode("utf-8")).hexdigest(),
        semantic_order=int(
            semantic_order
            if semantic_order is not None
            else action.get("_plan_index")
            or 0
        ),
        execution_order=len(state.get("completed_actions") or []),
        submitted_turn_index=int(
            submitted_turn_index
            if submitted_turn_index is not None
            else action.get("_submitted_turn_index")
            or state.get("turn_index")
            or 0
        ),
        origin_group=str(
            origin_group
            if origin_group is not None
            else action.get("_queue_origin_group")
            or (state.get("turn_context") or {}).get("origin_group")
            or ""
        ),
        origin_text=exact_origin_text,
        plan_scope=str(
            plan_scope
            if plan_scope is not None
            else action.get("_plan_scope")
            or ""
        ),
        admission_metadata={
            key.removeprefix("_"): deepcopy(action[key])
            for key in _ADMISSION_METADATA_KEYS
            if key in action
        },
        effect_kind=effect_kind,  # type: ignore[arg-type]
        idempotency_key=(
            f"{state.get('thread_id') or 'default'}:"
            f"{int(state.get('turn_index') or 0)}:{action_id}"
        ),
    )


def _action_from_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    envelope = action_envelope_from_dict(payload)
    return {
        "type": envelope.action_type,
        "action_id": envelope.action_id,
        "confidence": envelope.confidence,
        "reason": envelope.reason,
        **deepcopy(dict(envelope.arguments)),
    }


def _admission_action_from_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct a proposal only at registry and admission boundaries."""

    envelope = action_envelope_from_dict(payload)
    action = _action_from_envelope(payload)
    for key, value in envelope.admission_metadata.items():
        action[f"_{key}"] = deepcopy(value)
    action["_origin_text"] = envelope.origin_text
    action["_queue_origin_group"] = envelope.origin_group
    action["_submitted_turn_index"] = envelope.submitted_turn_index
    action["_plan_scope"] = envelope.plan_scope
    action["_plan_index"] = envelope.semantic_order
    return action


def _is_serialized_action_envelope(payload: Mapping[str, Any]) -> bool:
    return bool(
        str(payload.get("action_type") or "")
        and str(payload.get("owner") or "")
        and isinstance(payload.get("arguments"), Mapping)
    )


def _queue_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not _is_serialized_action_envelope(payload):
        raise StateInvariantError("durable action queue contains a raw proposal")
    envelope = action_envelope_from_dict(payload)
    action = _action_from_envelope(payload)
    action["_origin_text"] = envelope.origin_text
    return action


def _serialize_admitted_actions(
    state: AgentGraphState,
    actions: list[dict[str, Any]],
    *,
    default_origin_text: str,
    default_origin_group: str,
    default_plan_scope: str,
) -> list[dict[str, Any]]:
    """Create the only durable queue representation after admission."""

    serialized: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        envelope = _build_action_envelope(
            state,
            action,
            semantic_order=int(action.get("_plan_index") or index),
            submitted_turn_index=int(
                action.get("_submitted_turn_index")
                or state.get("turn_index")
                or 0
            ),
            origin_group=str(
                action.get("_queue_origin_group")
                or default_origin_group
            ),
            origin_text=str(
                action.get("_origin_text")
                or default_origin_text
            ),
            plan_scope=str(
                action.get("_plan_scope")
                or default_plan_scope
            ),
        )
        serialized.append(action_envelope_to_dict(envelope))
    return serialized


def _domain_action_from_envelope(
    state: AgentGraphState,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the owner input from trusted envelope provenance."""

    action = _action_from_envelope(payload)
    envelope = action_envelope_from_dict(payload)
    origin_text = envelope.origin_text
    if envelope.action_type == "propose_config_values":
        action["source_text"] = origin_text
    if envelope.owner == "chain_rpc":
        action["origin_text"] = origin_text
    if envelope.owner == "coordinator":
        action["queue_origin_group"] = envelope.origin_group
    if envelope.action_type == "analyze_report" and not action.get("job_id"):
        requested_job = JOB_ID_RE.search(origin_text)
        if requested_job:
            action["job_id"] = requested_job.group(0)
    return action


def _queue_has_eligible_action(state: AgentGraphState) -> bool:
    queue = [
        dict(item)
        for item in state.get("action_queue") or []
        if isinstance(item, Mapping)
    ]
    if not queue:
        return False
    if not state.get("pending_question"):
        return True
    return any(
        _action_can_run_while_pending(state, _queue_action(action))
        for action in queue
    )


def execute_selected_owner_step(
    state: AgentGraphState,
    *,
    expected_owner: str,
) -> AgentGraphState:
    """Prepare one owner result or run an explicitly unmigrated lifecycle."""

    selected_envelope = dict(state.get("selected_action") or {})
    envelope = (
        action_envelope_from_dict(selected_envelope)
        if selected_envelope
        else None
    )
    selected = _action_from_envelope(selected_envelope) if selected_envelope else {}
    spec = ACTION_BY_TYPE.get(str(selected.get("type") or ""))
    if not selected or spec is None or envelope is None or envelope.owner != expected_owner:
        raise StateInvariantError(
            f"selected action owner mismatch: {expected_owner}/"
            f"{str(selected.get('type') or '<missing>')}"
        )
    if spec.effect == "execution":
        return prepare_execution_intent_step(
            state,
            expected_owner=expected_owner,
        )
    return prepare_selected_owner_result_step(
        state,
        expected_owner=expected_owner,
    )


def prepare_execution_intent_step(
    state: AgentGraphState,
    *,
    expected_owner: str,
) -> AgentGraphState:
    """Persist one external execution authorization before invocation."""

    candidate = _copy_state(state)
    envelope = action_envelope_from_dict(candidate.get("selected_action") or {})
    if envelope.owner != "execution" or expected_owner != "execution":
        raise StateInvariantError("only the execution owner may prepare a side effect")
    request_id = str(
        (candidate.get("preflight") or {}).get("execution_request_id")
        or envelope.action_id
    )
    idempotency_key = f"harness:{request_id}"
    turn_id = (
        f"{candidate.get('thread_id') or 'default'}:"
        f"{int(candidate.get('turn_index') or 0)}"
    )
    request = {
        "action": action_envelope_to_dict(
            replace(
                envelope,
                idempotency_key=idempotency_key,
                status="prepared",
            )
        ),
        "workflow_mode": str(candidate.get("workflow_mode") or ""),
        "target_mode": str(candidate.get("target_mode") or ""),
        "plan_file": str(candidate.get("plan_file") or ""),
    }
    request_fingerprint = hashlib.sha256(
        json.dumps(
            request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    intent_id = hashlib.sha256(
        f"{turn_id}:{envelope.action_id}:{idempotency_key}".encode("utf-8")
    ).hexdigest()
    candidate.setdefault("preflight", {})["execution_request_id"] = request_id
    candidate["side_effect_intent"] = side_effect_intent_to_dict(
        SideEffectIntent(
            intent_id=intent_id,
            turn_id=turn_id,
            action_id=envelope.action_id,
            operation=envelope.action_type,
            idempotency_key=idempotency_key,
            request=request,
            request_fingerprint=request_fingerprint,
            expected_receipt_kind="execution_handler_result",
        )
    )
    candidate["side_effect_receipt"] = {}
    validate_state(candidate)
    return _set_turn_phase(candidate, "invoke_effect", "execution_intent_committed")


def mark_side_effect_invoking_step(state: AgentGraphState) -> AgentGraphState:
    """Checkpoint the execution attempt before entering the application service."""

    candidate = _copy_state(state)
    intent = dict(candidate.get("side_effect_intent") or {})
    if not intent:
        raise StateInvariantError("side-effect invocation has no durable intent")
    if intent.get("status") not in {"prepared", "invoking"}:
        raise StateInvariantError(
            f"side-effect intent cannot be invoked from status {intent.get('status')}"
        )
    intent["status"] = "invoking"
    intent["attempt_count"] = int(intent.get("attempt_count") or 0) + 1
    candidate["side_effect_intent"] = intent
    validate_state(candidate)
    return _set_turn_phase(candidate, "perform_effect", "execution_attempt_checkpointed")


def invoke_idempotent_side_effect_step(state: AgentGraphState) -> AgentGraphState:
    """Invoke one external execution operation under its persisted identity."""

    candidate = _copy_state(state)
    intent = dict(candidate.get("side_effect_intent") or {})
    selected = dict(candidate.get("selected_action") or {})
    if not intent or not selected:
        raise StateInvariantError("external invocation is missing intent or selected action")
    envelope = action_envelope_from_dict(selected)
    if intent.get("action_id") != envelope.action_id:
        raise StateInvariantError("side-effect intent does not match the selected action")
    action = _domain_action_from_envelope(candidate, selected)
    result = DOMAIN_RUNTIME["execution"].apply_action(
        deepcopy(candidate),
        _action_proposal(action, envelope.confidence),
    )
    serialized_result = handler_result_to_dict(result)
    job_id = ""
    for write in result.delta.writes:
        if write.path == ("job", "job_id"):
            job_id = str(write.value or "")
            break
    status = "blocked" if result.blocker or result.completion == "blocked" else "succeeded"
    receipt_id = hashlib.sha256(
        f"{intent.get('intent_id')}:{intent.get('attempt_count')}".encode("utf-8")
    ).hexdigest()
    candidate["side_effect_receipt"] = side_effect_receipt_to_dict(
        SideEffectReceipt(
            receipt_id=receipt_id,
            intent_id=str(intent.get("intent_id") or ""),
            action_id=envelope.action_id,
            status=status,  # type: ignore[arg-type]
            idempotency_key=str(intent.get("idempotency_key") or ""),
            result={"handler_result": serialized_result},
            job_id=job_id,
            failure_code=result.blocker,
            retryable=False,
        )
    )
    validate_state(candidate)
    return _set_turn_phase(candidate, "commit_receipt", "execution_receipt_observed")


def commit_side_effect_receipt_step(state: AgentGraphState) -> AgentGraphState:
    """Admit a persisted external receipt to the common action commit path."""

    candidate = _copy_state(state)
    intent = dict(candidate.get("side_effect_intent") or {})
    receipt = dict(candidate.get("side_effect_receipt") or {})
    if not intent or not receipt:
        raise StateInvariantError("execution receipt commit requires intent and receipt")
    if receipt.get("intent_id") != intent.get("intent_id"):
        raise StateInvariantError("execution receipt does not match its intent")
    envelope = action_envelope_from_dict(candidate.get("selected_action") or {})
    handler_payload = dict((receipt.get("result") or {}).get("handler_result") or {})
    result = handler_result_from_dict(handler_payload)
    intent["status"] = str(receipt.get("status") or "failed")
    candidate["side_effect_intent"] = intent
    candidate = _set_turn_phase(candidate, "commit", "execution_receipt_committed")
    candidate["pending_domain_result"] = pending_domain_result_to_dict(
        PendingDomainResult(
            action=replace(envelope, status="prepared"),
            result=result,
            prepared_state_hash=_prepared_state_hash(candidate),
        )
    )
    validate_state(candidate)
    return candidate


def prepare_selected_owner_result_step(
    state: AgentGraphState,
    *,
    expected_owner: str,
) -> AgentGraphState:
    """Run one side-effect-free domain handler and persist its typed result."""

    candidate = _copy_state(state)
    envelope = action_envelope_from_dict(candidate.get("selected_action") or {})
    if envelope.owner != expected_owner:
        raise StateInvariantError(
            f"prepared action owner mismatch: {expected_owner}/{envelope.owner}"
        )
    if envelope.effect_kind == "external":
        raise StateInvariantError(
            "external action reached pure owner preparation without a side-effect intent"
        )
    action = _domain_action_from_envelope(
        candidate,
        candidate["selected_action"],
    )
    if envelope.action_type == "answer_pending":
        result = _prepare_pending_answer_result(
            candidate,
            action,
            envelope,
        )
        candidate = _set_turn_phase(candidate, "commit", "pending_result_prepared")
        candidate["pending_domain_result"] = pending_domain_result_to_dict(
            PendingDomainResult(
                action=replace(envelope, status="prepared"),
                result=result,
                prepared_state_hash=_prepared_state_hash(candidate),
            )
        )
        validate_state(candidate)
        return candidate
    runtime = (
        COORDINATOR_RUNTIME
        if expected_owner == "coordinator"
        else DOMAIN_RUNTIME.get(expected_owner)
    )
    if runtime is None:
        raise StateInvariantError(f"no runtime is registered for owner: {expected_owner}")
    handler_state = deepcopy(candidate)
    result = runtime.apply_action(
        handler_state,
        _action_proposal(action, envelope.confidence),
    )
    spec = ACTION_BY_TYPE.get(envelope.action_type)
    if spec is not None and spec.lifetime == "turn_local":
        allowed_roots = set(spec.turn_local_result_roots)
        overlay_question = (
            result.pending_question
            if not candidate.get("pending_question") and result.pending_question is not None
            else None
        )
        turn_local_delta = StateDelta(
            writes=tuple(
                write
                for write in result.delta.writes
                if write.path and write.path[0] in allowed_roots
            ),
            deletes=tuple(
                path
                for path in result.delta.deletes
                if path and path[0] in allowed_roots
            ),
        )
        result = replace(
            result,
            delta=turn_local_delta,
            next_group=(
                str(
                    (
                        asdict(overlay_question)
                        if is_dataclass(overlay_question)
                        else dict(overlay_question or {})
                    ).get("group")
                    or ""
                )
                if overlay_question is not None
                else ""
            ),
            pending_question=overlay_question,
            clear_pending=False,
            navigation_command=None,
            checkpoint_command=None,
            invalidated_groups=(),
            reconfigured_groups=(),
            invalidated_fields=(),
            followup_actions=(),
        )
    candidate = _set_turn_phase(candidate, "commit", "domain_result_prepared")
    candidate["pending_domain_result"] = pending_domain_result_to_dict(
        PendingDomainResult(
            action=replace(envelope, status="prepared"),
            result=result,
            prepared_state_hash=_prepared_state_hash(candidate),
        )
    )
    validate_state(candidate)
    return candidate


def _prepare_pending_answer_result(
    state: AgentGraphState,
    action: dict[str, Any],
    envelope: ActionEnvelope,
) -> HandlerResult:
    """Translate one admitted answer into a domain result or one follow-up."""

    pending = dict(state.get("pending_question") or {})
    if not pending:
        return HandlerResult(blocker="no pending question is available for this answer")
    raw_answer = action.get("answer", "")
    selected = action.get("selected_value")
    if isinstance(selected, str) and not selected.strip():
        selected = None
    choice_question = str(pending.get("kind") or "") in {
        "numbered_choice",
        "yes_no",
    }
    interpreted: Any = selected if selected is not None else raw_answer
    if not choice_question and (
        interpreted is None
        or (isinstance(interpreted, str) and not interpreted.strip())
    ):
        return HandlerResult(blocker="the pending answer is empty")
    if not choice_question and not (
        _pending_option_value_exists(selected, pending)
        or _value_satisfies_pending_contract(interpreted, pending)
    ):
        return HandlerResult(
            blocker=_localized(
                state.get("language", "en"),
                "模型给出的结构化值不符合当前字段契约，当前问题保持不变。",
                "The model-derived value does not satisfy the current field contract. The question remains active.",
            )
        )
    manual_choice_value = bool(
        choice_question
        and pending.get("manual_input_allowed") is True
        and not _pending_option_value_exists(selected, pending)
        and _value_satisfies_pending_contract(
            selected if selected is not None else str(raw_answer),
            pending,
        )
    )
    if (
        choice_question
        and not _pending_option_value_exists(selected, pending)
        and not manual_choice_value
    ):
        return HandlerResult(
            blocker=_localized(
                state.get("language", "en"),
                "模型没有把这段回复映射到一个已声明选项，当前问题保持不变。请换一种说法，或回复显示的选项。",
                "The model did not map that reply to a declared option. The question remains active; rephrase or use a displayed option.",
            )
        )
    value = selected if _pending_option_value_exists(selected, pending) else interpreted
    declared_action = action_for_value(pending, value)
    if declared_action and str(declared_action.get("type") or "") != "answer_pending":
        followup = dict(declared_action)
        spec = ACTION_BY_TYPE.get(str(followup.get("type") or ""))
        if (
            spec is not None
            and "source_evidence" in spec.allowed_arguments
            and not str(followup.get("source_evidence") or "").strip()
        ):
            followup["source_evidence"] = str(raw_answer or "").strip()
        followup["confidence"] = "high"
        followup["selection_contract_verified"] = True
        policy = _option_return_policy(pending, value)
        return HandlerResult(
            consumed_action_ids=(envelope.action_id,),
            clear_pending=policy != "stay",
            pending_question=pending if policy == "stay" else None,
            followup_actions=(followup,),
            completion="completed",
        )
    question_id = str(pending.get("id") or "")
    if question_id == "inferred_config_review":
        return replace(
            apply_inferred_config_review(state, bool(value)),
            consumed_action_ids=(envelope.action_id,),
        )
    group = str(pending.get("group") or "")
    runtime = DOMAIN_RUNTIME.get(GROUP_OWNER.get(group, ""))
    if runtime is None or runtime.apply_answer is None:
        return HandlerResult(
            blocker=f"no answer handler owns question: {group}/{question_id}"
        )
    return replace(
        runtime.apply_answer(
            deepcopy(state),
            pending,
            value,
            str(raw_answer or ""),
        ),
        consumed_action_ids=(envelope.action_id,),
    )


def commit_selected_action_step(state: AgentGraphState) -> AgentGraphState:
    """Commit exactly one prepared owner result and disposition its action."""

    candidate = _copy_state(state)
    prepared_payload = dict(candidate.get("pending_domain_result") or {})
    if not prepared_payload:
        raise StateInvariantError("commit node has no pending domain result")
    prepared = pending_domain_result_from_dict(prepared_payload)
    envelope = prepared.action
    if prepared.prepared_state_hash != _prepared_state_hash(candidate):
        raise StateInvariantError("prepared domain result state fingerprint changed before commit")
    queue = [
        dict(item)
        for item in candidate.get("action_queue") or []
        if isinstance(item, Mapping)
    ]
    if not queue or str(queue[0].get("action_id") or "") != envelope.action_id:
        raise StateInvariantError("prepared action is not the admitted queue head")
    action = _action_from_envelope(action_envelope_to_dict(envelope))
    before_pending = dict(candidate.get("pending_question") or {})
    before_pending_id = str(before_pending.get("id") or "")
    before_responses = list(candidate.get("visible_response") or [])
    spec = ACTION_BY_TYPE.get(envelope.action_type)
    candidate["action_queue"] = queue[1:]
    if envelope.action_type == "answer_pending":
        _discard_superseded_queue_actions(candidate, before_pending)
    candidate["pending_domain_result"] = {}
    candidate["selected_action"] = {}
    candidate["current_action"] = action
    control = dict(candidate.get("control") or {})
    control.pop("selected_owner", None)
    candidate["control"] = control
    if (
        envelope.action_type != "answer_pending"
        and not (spec is not None and spec.lifetime == "turn_local")
        and envelope.action_type
        in {
            str(item).strip()
            for item in before_pending.get("accepted_action_types") or []
            if str(item).strip()
        }
    ):
        candidate["pending_question"] = {}
    committed = _apply_handler_result(
        candidate,
        prepared.result,
        owner=envelope.owner,
    )
    if (
        prepared.result.completion == "in_progress"
        and prepared.result.next_group
        and not committed.get("pending_question")
    ):
        active_group = str(prepared.result.next_group)
        internal_group_progression = bool(
            active_group
            and active_group
            in {
                str(envelope.target_group or ""),
                str(before_pending.get("group") or ""),
                str(candidate.get("active_group") or ""),
            }
        )
        prerequisite = (
            _navigation_prerequisite(committed, active_group)
            if active_group and not internal_group_progression
            else ""
        )
        if prerequisite:
            control = dict(committed.get("control") or {})
            control["deferred_group"] = active_group
            committed["control"] = control
            _record_group_transition(committed, prerequisite)
            active_group = prerequisite
        next_question = _question_for_group(committed, active_group) if active_group else None
        if next_question:
            next_question["same_turn_navigation_allowed"] = (
                prepared.result.completion != "blocked"
            )
            _install_pending_question(committed, next_question)
    if committed.get("pending_question") and committed.get("action_queue"):
        committed["pending_question"]["resume_action_queue"] = True
    defer_after_answer = False
    if envelope.action_type == "answer_pending":
        committed = _resume_field_reconfiguration_after_prerequisite(
            committed,
            answered_question=before_pending,
        )
        if (
            before_pending_id
            and str((committed.get("pending_question") or {}).get("id") or "")
            == before_pending_id
            and prepared.result.completion == "blocked"
        ):
            committed["pending_question"]["created_turn_index"] = int(
                committed.get("turn_index") or 0
            )
        if (
            prepared.result.completion == "blocked"
            and committed.get("pending_question")
            and committed.get("action_queue")
        ):
            committed["pending_question"]["resume_action_queue"] = True
            defer_after_answer = True
        elif committed.get("pending_question") and committed.get("action_queue"):
            committed["pending_question"]["resume_action_queue"] = True
            if (
                not _queue_has_eligible_action(committed)
            ):
                defer_after_answer = True
    if envelope.action_type == "reset_session" and queue[1:]:
        committed["action_queue"] = queue[1:]
    after_pending_id = str((committed.get("pending_question") or {}).get("id") or "")
    if (
        spec is not None
        and spec.preserve_pending
        and before_pending_id
        and after_pending_id != before_pending_id
        and not _action_satisfies_pending_manual_effect(before_pending, action)
    ):
        _push_interruption_frame(
            committed,
            before_pending,
            reason=f"{envelope.action_type}_overlay",
        )
    if before_responses:
        merged = list(before_responses)
        if before_pending_id and after_pending_id != before_pending_id:
            merged = _without_superseded_question(
                merged,
                before_pending,
                committed.get("language", "en"),
            )
        for response in committed.get("visible_response") or []:
            if response not in merged:
                merged.append(response)
        committed["visible_response"] = merged
    rejected = bool(prepared.result.blocker)
    if not rejected:
        if spec is None or spec.lifetime != "turn_local":
            completed = list(committed.get("completed_actions") or [])
            completed.append(action)
            committed["completed_actions"] = completed[-20:]
        if envelope.action_id:
            applied = list(committed.get("applied_action_ids") or [])
            if envelope.action_id not in applied:
                applied.append(envelope.action_id)
            committed["applied_action_ids"] = applied[-200:]
        receipt = dict(committed.get("turn_receipt") or {})
        execution_order = list(receipt.get("execution_order") or [])
        if envelope.action_id and envelope.action_id not in execution_order:
            execution_order.append(envelope.action_id)
        receipt["execution_order"] = execution_order
        receipt["status"] = "executing"
        committed["turn_receipt"] = receipt
        if spec is not None and spec.lifetime == "turn_local":
            result_count = len(
                [
                    response
                    for response in (
                        *prepared.result.visible_results,
                        prepared.result.visible_result,
                    )
                    if response
                ]
            )
            if result_count:
                turn_context = dict(committed.get("turn_context") or {})
                turn_context["turn_local_result_count"] = int(
                    turn_context.get("turn_local_result_count") or 0
                ) + result_count
                committed["turn_context"] = turn_context
    committed["current_action"] = {}
    if (
        spec is not None
        and spec.lifetime == "turn_local"
        and not committed.get("action_queue")
    ):
        validate_state(committed)
        return _set_turn_phase(
            committed,
            "compose",
            "turn_local_result_completed",
        )
    if defer_after_answer:
        validate_state(committed)
        return _set_turn_phase(
            committed,
            "compose",
            "new_pending_contract_defers_remaining_queue",
        )
    queue_can_continue = _queue_has_eligible_action(committed)
    if rejected or (
        prepared.result.stop_after_response
        and not committed.get("action_queue")
    ):
        return _set_turn_phase(committed, "compose", "action_commit_stopped")
    if queue_can_continue:
        validate_state(committed)
        return _set_turn_phase(committed, "execute", "action_committed_queue_continues")
    if committed.get("pending_question"):
        validate_state(committed)
        return _set_turn_phase(committed, "compose", "action_committed_pending")
    validate_state(committed)
    return _set_turn_phase(committed, "fallback", "action_committed_queue_complete")


def _prepared_state_hash(state: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in state.items()
        if key not in {
            "discovery",
            "framework_summary",
            "web_research",
            "pending_domain_result",
        }
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fallback_turn_step(state: AgentGraphState) -> AgentGraphState:
    selected_group, reason = next_group_and_reason(state)
    readiness = {}
    for group in GROUP_ORDER:
        fact = group_readiness(state, group)
        readiness[group] = {
            "ready": bool(fact.ready),
            "continuation": bool(fact.continuation),
            "reason_hash": hashlib.sha256(
                str(fact.reason or "").encode("utf-8")
            ).hexdigest(),
        }
    state = _ask_next_blocking_question(state)
    _append_control_receipt(
        state,
        "fallback_selection",
        {
            "selected_group": selected_group,
            "reason_hash": hashlib.sha256(
                str(reason or "").encode("utf-8")
            ).hexdigest(),
            "group_readiness": readiness,
            "pending_after_id": str(
                (state.get("pending_question") or {}).get("id") or ""
            ),
        },
    )
    return _set_turn_phase(state, "compose", "fallback_resolved")


def compose_turn_step(state: AgentGraphState) -> AgentGraphState:
    state = _finalize_turn_response(state)
    receipt = dict(state.get("turn_receipt") or {})
    if receipt:
        receipt["pending_after"] = deepcopy(state.get("pending_question") or {})
        receipt["response_count"] = len(state.get("visible_response") or [])
        receipt["status"] = (
            "blocked"
            if state.get("pending_question") or state.get("failure_recovery")
            else "completed"
        )
        state["turn_receipt"] = receipt
    pending = dict(state.get("pending_question") or {})
    rendered_pending = (
        _render_question(pending, state.get("language", "en"))
        if pending
        else ""
    )
    _append_control_receipt(
        state,
        "response_composition",
        {
            "language": str(state.get("language") or ""),
            "active_group": str(state.get("active_group") or ""),
            "source_action_ids": [
                str(item)
                for item in (state.get("turn_receipt") or {}).get(
                    "execution_order"
                )
                or ()
                if str(item)
            ],
            "pending_contract_hash": _receipt_hash(pending),
            "fragments": [
                {
                    "fragment_hash": hashlib.sha256(
                        str(fragment).encode("utf-8")
                    ).hexdigest(),
                    "role": (
                        "pending_question"
                        if rendered_pending and str(fragment) == rendered_pending
                        else "visible_result"
                    ),
                }
                for fragment in state.get("visible_response") or ()
            ],
        },
    )
    return _set_turn_phase(state, "end", "response_composed")


def _copy_state(state: AgentGraphState) -> AgentGraphState:
    output: AgentGraphState = dict(state)
    output.setdefault("action_queue", [])
    output.setdefault("current_action", {})
    output.setdefault("completed_actions", [])
    output.setdefault("action_errors", [])
    output.setdefault("group_states", {})
    output.setdefault("group_history", [])
    output.setdefault("confirmed_config", {})
    output.setdefault("inferred_config", {})
    output.setdefault("invalidated_groups", [])
    output.setdefault("interruption_stack", [])
    output.setdefault("chain_identity", {})
    output.setdefault("workload", {})
    output.setdefault("custom_rpc", {})
    output.setdefault("endpoint_evidence", {})
    output.setdefault("fixture_evidence", {})
    output.setdefault("sync_observe", {})
    output.setdefault("observability", {})
    output.setdefault("evidence_buffer", [])
    output.setdefault("evidence_collection", {})
    output.setdefault("failure_recovery", {})
    output.setdefault("final_benchmark", {})
    output.setdefault("audit_events", [])
    output.setdefault("resume_context", {})
    output.setdefault("workflow_goals", [])
    return output



def _next_action_was_submitted_after_pending(state: AgentGraphState) -> bool:
    queue = state.get("action_queue") or []
    pending = state.get("pending_question") or {}
    if not queue or not pending:
        return False
    submitted = action_envelope_from_dict(queue[0]).submitted_turn_index
    created = int(pending.get("created_turn_index") or 0)
    return submitted > created


def _merge_durable_action_queue(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge a new user turn with deferred actions from earlier turns.

    Incoming actions run first because they may answer the current barrier or
    revise an earlier request. A new action supersedes an older action with the
    same semantic identity or mutation dimension. Reset intentionally discards
    deferred work from the prior workflow.
    """

    existing = [
        (
            _admission_action_from_envelope(item)
            if _is_serialized_action_envelope(item)
            else dict(item)
        )
        for item in _durable_actions(existing)
    ]
    incoming = [
        (
            _admission_action_from_envelope(item)
            if _is_serialized_action_envelope(item)
            else dict(item)
        )
        for item in _durable_actions(incoming)
    ]
    if any(str(item.get("type") or "") == "reset_session" for item in incoming):
        return incoming
    merged_incoming = [dict(item) for item in incoming]
    incoming_index = {
        action_merge_key(item): index
        for index, item in enumerate(merged_incoming)
    }
    incoming_dimensions = {
        spec.mutation_dimension
        for item in incoming
        if (spec := ACTION_BY_TYPE.get(str(item.get("type") or ""))) is not None
        and spec.mutation_dimension
    }
    retained: list[dict[str, Any]] = []
    for item in existing:
        spec = ACTION_BY_TYPE.get(str(item.get("type") or ""))
        merge_key = action_merge_key(item)
        if merge_key in incoming_index:
            index = incoming_index[merge_key]
            merged_incoming[index] = merge_semantic_actions(item, merged_incoming[index])
            continue
        if spec is not None and spec.mutation_dimension in incoming_dimensions:
            continue
        retained.append(item)
    return merged_incoming + retained


def _durable_actions(actions: Any) -> list[dict[str, Any]]:
    """Drop turn-local actions from current and legacy durable queues."""

    return [
        dict(item)
        for item in actions or []
        if (
            isinstance(item, dict)
            and not action_is_turn_local(
                _action_from_envelope(item)
                if _is_serialized_action_envelope(item)
                else item
            )
        )
    ]


def _queue_has_admitted_durable_work(state: AgentGraphState) -> bool:
    """Distinguish admitted commands from unannotated legacy stale entries."""

    return any(
        _is_serialized_action_envelope(item)
        and not action_is_turn_local(_action_from_envelope(item))
        for item in state.get("action_queue") or []
        if isinstance(item, dict)
    )


def _discard_superseded_queue_actions(state: AgentGraphState, question: PendingQuestion) -> None:
    """Drop queued actions whose decision was resolved by this question.

    Interruption questions declare this relationship in their typed contract;
    the coordinator only enforces it and does not infer domain intent.
    """

    superseded = {
        str(action_type).strip()
        for action_type in question.get("supersedes_action_types") or []
        if str(action_type).strip()
    }
    if not superseded:
        return
    state["action_queue"] = [
        action
        for action in state.get("action_queue") or []
        if str(_queue_action(action).get("type") or "").strip() not in superseded
    ]


def _queue_can_continue_through_pending(state: AgentGraphState) -> bool:
    """Return whether the next typed action can consume this pending contract."""

    pending_group = str((state.get("pending_question") or {}).get("group") or "").strip()
    queue = state.get("action_queue") or []
    if not pending_group or not queue:
        return not pending_group
    next_action_type = str(_queue_action(queue[0]).get("type") or "").strip()
    accepted = {
        str(action_type).strip()
        for action_type in (state.get("pending_question") or {}).get("accepted_action_types") or []
        if str(action_type).strip()
    }
    return next_action_type in accepted


def _next_queue_action_crosses_pending_barrier(state: AgentGraphState) -> bool:
    queue = state.get("action_queue") or []
    if not queue:
        return False
    envelope = action_envelope_from_dict(queue[0])
    action = _action_from_envelope(queue[0])
    if not action_crosses_pending_barrier(action):
        return False
    pending = state.get("pending_question") or {}
    if not pending:
        return True
    # Navigation may detour from a question that was already visible when the
    # user submitted this turn. A separately admitted explicit navigation in
    # the same semantic transaction may also suspend a question created by an
    # earlier sibling action; the interruption stack, not the barrier, owns
    # resumption. Other actions cannot bypass a newly created contract.
    submitted_turn = envelope.submitted_turn_index
    created_turn = int(pending.get("created_turn_index") or 0)
    if submitted_turn and created_turn < submitted_turn:
        return True
    return bool(
        submitted_turn
        and created_turn == submitted_turn
        and str(action.get("type") or "") == "change_group"
        and action.get("group_navigation_semantic_verified") is True
        and str(action.get("source_evidence") or "").strip()
    )


def _pending_is_queue_barrier(state: AgentGraphState) -> bool:
    return bool((state.get("pending_question") or {}).get("queue_barrier"))


def _suspend_pending_for_remaining_queue(state: AgentGraphState) -> None:
    """Defer a derived next question while applying independent user actions."""

    pending = dict(state.get("pending_question") or {})
    if not pending:
        return
    _remove_rendered_question(state, pending)
    _push_interruption_frame(state, pending, reason="same_turn_action_queue")
    state["pending_question"] = {}


def _push_interruption_frame(state: AgentGraphState, pending: PendingQuestion, *, reason: str) -> None:
    """Persist one typed return point without copying stale prompt text."""

    frame = {
        "group": str(pending.get("group") or state.get("active_group") or ""),
        "question_id": str(pending.get("id") or ""),
        "reason": reason,
    }
    field = str(pending.get("field") or "").strip()
    if field:
        frame["field"] = field
    if not frame["group"] or not frame["question_id"]:
        return
    stack = list(state.get("interruption_stack") or [])
    frame_identity = (
        frame["group"],
        frame["question_id"],
        str(frame.get("field") or ""),
    )
    top_identity = (
        str((stack[-1] if stack else {}).get("group") or ""),
        str((stack[-1] if stack else {}).get("question_id") or ""),
        str((stack[-1] if stack else {}).get("field") or ""),
    )
    if top_identity != frame_identity:
        stack.append(frame)
    state["interruption_stack"] = stack[-40:]


def _resume_suspended_question(state: AgentGraphState) -> PendingQuestion | None:
    """Restore the newest still-relevant typed question after a detour.

    Frames store identity only.  The owning domain reconstructs the question
    from current state, so stale prompts cannot revive after invalidation.
    """

    # A detour owns control until its current domain has no blocking question.
    # Restoring an older frame first would abandon a partially configured group
    # and make the next user answer bind to stale work.
    stack = list(state.get("interruption_stack") or [])
    active_group = str(state.get("active_group") or "").strip()
    if active_group and active_group in ALLOWED_GROUPS:
        active_question = _question_for_group(state, active_group)
        if active_question:
            # Questions derived from one admitted transaction retain canonical
            # dependency order. An explicit user detour is different: its
            # active group owns control until that group is complete.
            active_order = GROUP_ORDER.index(active_group) if active_group in GROUP_ORDER else len(GROUP_ORDER)
            for frame_index in range(len(stack) - 1, -1, -1):
                frame = stack[frame_index]
                if str((frame or {}).get("reason") or "") != "same_turn_action_queue":
                    continue
                group = str((frame or {}).get("group") or "").strip()
                question_id = str((frame or {}).get("question_id") or "").strip()
                if group not in GROUP_ORDER or GROUP_ORDER.index(group) >= active_order:
                    continue
                question = _reconstruct_question(state, frame)
                if not question or str(question.get("id") or "") != question_id:
                    continue
                del stack[frame_index]
                state["interruption_stack"] = stack
                _record_group_transition(state, group, record_history=False)
                return question
            return None

    return _pop_interruption_question(state)


def _pop_interruption_question(state: AgentGraphState) -> PendingQuestion | None:
    """Consume the newest reconstructable interruption frame."""

    stack = list(state.get("interruption_stack") or [])
    while stack:
        frame = stack.pop()
        group = str((frame or {}).get("group") or "").strip()
        question_id = str((frame or {}).get("question_id") or "").strip()
        if not group or group not in ALLOWED_GROUPS:
            continue
        question = _reconstruct_question(state, frame)
        if not question or str(question.get("id") or "") != question_id:
            continue
        state["interruption_stack"] = stack
        _record_group_transition(state, group, record_history=False)
        return question
    state["interruption_stack"] = []
    return None


def _apply_handler_result(
    state: AgentGraphState,
    result: HandlerResult,
    *,
    owner: str,
    control_state: AgentGraphState | None = None,
) -> AgentGraphState:
    """Atomically validate and commit one domain result.

    Domain handlers run against isolated copies. A blocker therefore rejects
    every proposed domain and control mutation; only its response is appended.
    A business failure expressed as ``completion='blocked'`` without a blocker
    still commits the owner's failure delta.
    """

    previous_pending = dict(state.get("pending_question") or {})
    navigation_command = result.navigation_command
    if result.blocker:
        candidate: AgentGraphState = deepcopy(state)
        _append_domain_control_receipts(
            candidate,
            result.control_receipts,
            owner=owner,
        )
        current_action = dict(candidate.get("current_action") or {})
        if current_action and owner not in {"coordinator", "orientation", "recovery"}:
            retained_state = {
                "affected_group": str(
                    (ACTION_BY_TYPE.get(str(current_action.get("type") or "")) or ACTION_BY_TYPE["unknown"]).target_group
                    or candidate.get("active_group")
                    or (candidate.get("pending_question") or {}).get("group")
                    or "opening"
                ),
                "active_group": str(candidate.get("active_group") or ""),
                "pending_question": deepcopy(candidate.get("pending_question") or {}),
                "confirmed_config": deepcopy(candidate.get("confirmed_config") or {}),
                "durable_queue": _durable_actions(candidate.get("action_queue") or []),
            }
            record = domain_blocker_failure_record(
                current_action,
                owner=owner,
                validation_detail=result.blocker,
                retained_state=retained_state,
            )
            candidate["failure_recovery"] = {"status": "pending", "record": record}
            candidate.setdefault("action_errors", []).append({
                "action": record["action_identity"],
                "error": "domain_blocked",
                "failure_id": record["failure_id"],
            })
            blocker_response = render_failure_summary(record, str(candidate.get("language") or "en"))
        else:
            blocker_response = result.blocker
        responses = list(candidate.get("visible_response") or [])
        responses.append(blocker_response)
        candidate["visible_response"] = responses
        _append_control_receipt(
            candidate,
            "domain_commit",
            {
                "owner": owner,
                "completion": "rejected",
                "blocker_hash": hashlib.sha256(
                    str(result.blocker).encode("utf-8")
                ).hexdigest(),
                "pending_before_hash": _receipt_hash(previous_pending),
                "pending_after_hash": _receipt_hash(
                    candidate.get("pending_question") or {}
                ),
                "consumed_action_ids": [],
                "invalidated_groups": [],
                "invalidated_fields": [],
                "response_fragment_hashes": [
                    hashlib.sha256(str(item).encode("utf-8")).hexdigest()
                    for item in candidate.get("visible_response") or ()
                ],
            },
        )
        validate_state(candidate)
        return candidate

    navigation_state: AgentGraphState | None = None
    navigation_followups: tuple[Mapping[str, Any], ...] = ()
    if result.navigation_command is not None:
        navigation_state, navigation_followups = _commit_navigation_command(
            state,
            result.navigation_command,
        )
        result = replace(
            result,
            navigation_command=None,
            followup_actions=(
                *navigation_followups,
                *result.followup_actions,
            ),
        )
    if result.checkpoint_command is not None:
        candidate = _apply_checkpoint_command(state, result.checkpoint_command)
    elif navigation_state is not None:
        candidate = navigation_state
    elif owner == "coordinator" and control_state is not None:
        candidate = deepcopy(control_state)
    else:
        candidate = deepcopy(state)
    if result.field_reconfiguration_command is not None:
        command = result.field_reconfiguration_command
        control = dict(candidate.get("control") or {})
        if command.prerequisite_question_id:
            control["field_reconfiguration_continuation"] = {
                "group": command.group,
                "config_field": command.config_field,
                "prerequisite_question_id": command.prerequisite_question_id,
            }
        else:
            control.pop("field_reconfiguration_continuation", None)
        candidate["control"] = control
    if result.workflow_goal_command is not None:
        command = result.workflow_goal_command
        goals = [
            dict(item)
            for item in candidate.get("workflow_goals") or []
            if isinstance(item, dict)
        ]
        if command.operation == "enqueue":
            goal = dict(command.goal)
            if not any(
                item.get("target_mode") == goal.get("target_mode")
                and item.get("goal") == goal.get("goal")
                for item in goals
            ):
                goals.append(goal)
        elif command.operation == "remove_first":
            if goals:
                goals.pop(0)
        else:
            raise StateInvariantError(
                f"unsupported workflow goal command: {command.operation}"
            )
        candidate["workflow_goals"] = goals
    candidate = apply_state_delta(candidate, result.delta, owner=owner)

    recovery_pending: dict[str, Any] | None = None
    if result.recovery_command is not None:
        recovery_pending = _apply_recovery_command(candidate, result.recovery_command)

    if result.invalidated_groups:
        _apply_group_invalidation_state(candidate, result.invalidated_groups)
        invalidated = set(candidate.get("invalidated_groups") or [])
        invalidated.update(result.invalidated_groups)
        candidate["invalidated_groups"] = sorted(invalidated)
        group_states = candidate.setdefault("group_states", {})
        for group in result.invalidated_groups:
            group_states.setdefault(group, {})["status"] = "invalidated"
    for group in result.reconfigured_groups:
        mark_group_reconfigured(candidate, group)
    if result.invalidated_fields:
        confirmed = candidate.setdefault("confirmed_config", {})
        inferred = candidate.setdefault("inferred_config", {})
        for field in result.invalidated_fields:
            confirmed.pop(field, None)
            inferred.pop(field, None)
    if result.evidence:
        candidate.setdefault("audit_events", []).extend(dict(item) for item in result.evidence)
    if result.action_errors:
        candidate.setdefault("action_errors", []).extend(dict(item) for item in result.action_errors)

    if result.clear_pending:
        candidate["pending_question"] = {}
    effective_next_group = "failure_recovery" if recovery_pending else result.next_group
    if effective_next_group:
        _record_group_transition(candidate, effective_next_group)
    responses = list(candidate.get("visible_response") or [])
    for response in (*result.visible_results, result.visible_result):
        if response and response not in responses:
            responses.append(response)
    candidate["visible_response"] = responses
    effective_pending = recovery_pending if recovery_pending is not None else result.pending_question
    if effective_pending is not None:
        pending = asdict(effective_pending) if is_dataclass(effective_pending) else dict(effective_pending)
        if pending != previous_pending:
            pending["same_turn_navigation_allowed"] = (
                result.completion != "blocked"
            )
            _install_pending_question(candidate, pending)
            if not _next_queue_action_crosses_pending_barrier(candidate):
                rendered = _render_question(pending, candidate.get("language", "en"))
                responses = _without_superseded_question(
                    list(candidate.get("visible_response") or []),
                    previous_pending,
                    candidate.get("language", "en"),
                )
                if rendered not in responses and not _question_is_actionable_in_responses(responses, pending):
                    responses.append(rendered)
                candidate["visible_response"] = responses
    pending_owner = str((candidate.get("pending_question") or {}).get("group") or "").strip()
    if pending_owner:
        candidate["active_group"] = pending_owner
        mark_group_reconfiguring(candidate, pending_owner)
    elif result.completion == "completed":
        completed_group = str(candidate.get("active_group") or "").strip()
        if completed_group and completed_group in set(candidate.get("invalidated_groups") or []):
            mark_group_reconfigured(candidate, completed_group)
    if result.consumed_action_ids:
        applied = list(candidate.get("applied_action_ids") or [])
        for action_id in result.consumed_action_ids:
            if action_id and action_id not in applied:
                applied.append(action_id)
        candidate["applied_action_ids"] = applied[-200:]
    if result.followup_actions:
        origin_text = str((candidate.get("turn_context") or {}).get("text") or candidate.get("last_user_input") or "")
        scope = (
            f"{candidate.get('thread_id') or 'default'}:"
            f"{int(candidate.get('turn_index') or 0)}:followup"
        )
        followups = assign_action_ids(
            scope,
            origin_text,
            [dict(item) for item in result.followup_actions],
        )
        for index, action in enumerate(followups):
            action["_origin_text"] = origin_text
            action["_queue_origin_group"] = str(candidate.get("active_group") or "")
            action["_submitted_turn_index"] = int(candidate.get("turn_index") or 0)
            action["_plan_scope"] = scope
            action["_plan_index"] = index
        ordered_followups = _order_action_queue(
            candidate,
            [
                *followups,
                *[
                    _admission_action_from_envelope(item)
                    for item in candidate.get("action_queue") or []
                ],
            ],
        )
        candidate["action_queue"] = _serialize_admitted_actions(
            candidate,
            ordered_followups,
            default_origin_text=origin_text,
            default_origin_group=str(candidate.get("active_group") or ""),
            default_plan_scope=scope,
        )
    if candidate.get("pending_question") and not candidate.get("action_queue"):
        candidate["pending_question"].pop("resume_action_queue", None)

    candidate = _activate_ready_deferred_group(candidate)
    _append_domain_control_receipts(
        candidate,
        result.control_receipts,
        owner=owner,
    )
    delta_paths = [
        {
            "operation": "write",
            "path": ".".join(write.path),
            "value_hash": _receipt_hash(write.value),
        }
        for write in result.delta.writes
    ]
    delta_paths.extend(
        {
            "operation": "delete",
            "path": ".".join(path),
            "value_hash": "",
        }
        for path in result.delta.deletes
    )
    _append_control_receipt(
        candidate,
        "domain_commit",
        {
            "owner": owner,
            "completion": str(result.completion or ""),
            "consumed_action_ids": [
                str(item) for item in result.consumed_action_ids if str(item)
            ],
            "invalidated_groups": [
                str(item) for item in result.invalidated_groups if str(item)
            ],
            "invalidated_fields": [
                str(item) for item in result.invalidated_fields if str(item)
            ],
            "reconfigured_groups": [
                str(item) for item in result.reconfigured_groups if str(item)
            ],
            "material_delta": delta_paths,
            "navigation_operation": str(
                navigation_command.operation if navigation_command else ""
            ),
            "navigation_target_group": str(
                navigation_command.target_group if navigation_command else ""
            ),
            "pending_before_hash": _receipt_hash(previous_pending),
            "pending_after_hash": _receipt_hash(
                candidate.get("pending_question") or {}
            ),
            "pending_after_id": str(
                (candidate.get("pending_question") or {}).get("id") or ""
            ),
            "response_fragment_hashes": [
                hashlib.sha256(str(item).encode("utf-8")).hexdigest()
                for item in candidate.get("visible_response") or ()
            ],
        },
    )

    validate_state(candidate)
    return candidate


def _commit_navigation_command(
    state: AgentGraphState,
    command: NavigationCommand,
) -> tuple[AgentGraphState, tuple[Mapping[str, Any], ...]]:
    """Commit one typed navigation transaction at the control boundary."""

    candidate = deepcopy(state)
    followups: tuple[Mapping[str, Any], ...] = ()
    if command.operation == "change_group":
        target_group = command.target_group
        interrupted_pending = deepcopy(candidate.get("pending_question") or {})
        pending_group = str(interrupted_pending.get("group") or "").strip()
        if (
            interrupted_pending
            and target_group != pending_group
            and _reconstruct_question(candidate, interrupted_pending) is not None
        ):
            _push_interruption_frame(
                candidate,
                interrupted_pending,
                reason="explicit_navigation",
            )
        prerequisite = _navigation_prerequisite(candidate, target_group)
        if prerequisite:
            control = dict(candidate.get("control") or {})
            control["deferred_group"] = target_group
            candidate["control"] = control
            return _activate_group_question(candidate, prerequisite), followups
        if (
            target_group == command.origin_group
            and _active_group_has_blocking_question(candidate)
        ):
            pending = dict(candidate.get("pending_question") or {})
            candidate["visible_response"] = [
                _render_question(pending, candidate.get("language", "en"))
            ]
            return candidate, followups
        if _queue_has_followup_for_group(candidate, target_group):
            _record_group_transition(candidate, target_group)
            candidate["pending_question"] = {}
            return candidate, followups
        if (
            target_group == "chain_identity"
            and (candidate.get("chain_identity") or {}).get("canonical")
        ):
            candidate["pending_question"] = {}
            followups = (
                {
                    "type": "request_chain_selection",
                    "confidence": "high",
                },
            )
            return candidate, followups
        return (
            _activate_group_question(
                candidate,
                target_group,
                reconfigure=True,
            ),
            followups,
        )

    control = dict(candidate.get("control") or {})
    control.pop("field_reconfiguration_continuation", None)
    candidate["control"] = control
    resume_group = ""
    pending = dict(candidate.get("pending_question") or {})
    pending_owner = GROUP_OWNER.get(str(pending.get("group") or ""), "")
    runtime = DOMAIN_RUNTIME.get(pending_owner)
    if pending and runtime is not None and runtime.cancel_question is not None:
        cancellation = runtime.cancel_question(deepcopy(candidate), pending)
        resume_group = str(cancellation.navigation_resume_group or "").strip()
        candidate = _apply_handler_result(
            candidate,
            replace(
                cancellation,
                followup_actions=(),
                stop_after_response=False,
            ),
            owner=pending_owner,
        )
        followups = tuple(cancellation.followup_actions)
    interrupted_question = _pop_interruption_question(candidate)
    if interrupted_question:
        _install_pending_question(candidate, interrupted_question)
        candidate["visible_response"] = [
            _render_question(
                interrupted_question,
                candidate.get("language", "en"),
            )
        ]
        return candidate, followups
    if resume_group:
        _discard_cancelled_origin_from_history(candidate)
    previous_group = resume_group or _pop_previous_group(candidate)
    if previous_group:
        candidate = _activate_group_question(
            candidate,
            previous_group,
            record_history=False,
        )
    else:
        candidate["pending_question"] = {}
        candidate["visible_response"] = [
            _localized(
                candidate.get("language", "en"),
                "当前没有可回退的配置组。你可以直接说明要回到哪个配置项，例如 RPC、QPS、磁盘或可观测性。",
                "There is no previous configuration group to return to. Name the area to revisit, such as RPC, QPS, disk, or observability.",
            )
        ]
    return candidate, followups


def _activate_ready_deferred_group(state: AgentGraphState) -> AgentGraphState:
    """Resume an explicit group detour before registry fallback can take over."""

    if state.get("pending_question"):
        return state
    control = dict(state.get("control") or {})
    deferred_group = str(control.get("deferred_group") or "").strip()
    if not deferred_group or deferred_group not in ALLOWED_GROUPS:
        return state
    prerequisite = _navigation_prerequisite(state, deferred_group)
    if prerequisite:
        if prerequisite != str(state.get("active_group") or ""):
            return _activate_group_question(state, prerequisite)
        return state
    control.pop("deferred_group", None)
    state["control"] = control
    return _activate_group_question(state, deferred_group)


def _apply_recovery_command(
    state: AgentGraphState,
    command: RecoveryCommand,
) -> dict[str, Any] | None:
    """Commit an execution failure signal at the control-plane boundary."""

    if command.operation == "activate":
        state["failure_recovery"] = {
            "status": "pending",
            "record": dict(command.record),
        }
        return question_for_recovery(state, "failure_recovery")
    recovery = state.get("failure_recovery") or {}
    if recovery.get("status") == "correcting":
        recovery["status"] = "resolved"
        recovery["validation_receipt"] = command.validation_receipt
    return None


def _apply_group_invalidation_state(
    state: AgentGraphState,
    groups: tuple[str, ...],
) -> None:
    """Apply cross-domain invalidation only at the coordinator commit boundary."""

    invalidated = set(groups)
    if "preflight_smoke_execution" in invalidated:
        state["plan"] = {}
        state["plan_file"] = ""
        state["preflight"] = {}
        state["smoke"] = {}
        state["final_benchmark"] = {}
    if "job_monitoring" in invalidated:
        state["job"] = {}
    if "workload_rpc" in invalidated:
        state["workload"] = {}
        state["rpc_mode"] = ""
    if "target_samples_fixtures" in invalidated:
        state["fixture_evidence"] = {}
    if "qps_profile" in invalidated:
        state["qps_profile"] = {}
    if "sync_observe" in invalidated:
        state["sync_observe"] = {}
    if "endpoint_process" in invalidated:
        evidence = state.setdefault("endpoint_evidence", {})
        evidence.pop("local_rpc_url_ready", None)
        evidence.pop("sync_rpc_url_ready", None)


def _apply_checkpoint_command(state: AgentGraphState, command: CheckpointCommand) -> AgentGraphState:
    language = str(state.get("language") or "en")
    session = state.get("session") or {}
    candidate = new_state(
        str(state.get("thread_id") or "default"),
        language=language,
        session_purpose=str(session.get("purpose") or "user"),
    )
    for key in RESET_PRESERVED_KEYS:
        if key in state:
            candidate[key] = deepcopy(state[key])  # type: ignore[literal-required]
    # A workflow reset clears durable configuration, not the control-plane
    # identity of the turn that requested it.  Retaining this snapshot keeps
    # the reset auditable through the same admission/evidence path as every
    # other pending-question action.
    candidate["turn_index"] = int(state.get("turn_index") or 0)
    candidate["last_user_input"] = str(state.get("last_user_input") or "")
    candidate["turn_context"] = deepcopy(dict(state.get("turn_context") or {}))
    candidate["turn_receipt"] = deepcopy(dict(state.get("turn_receipt") or {}))
    candidate["audit_events"] = list(state.get("audit_events") or []) + [{"event": "workflow_reset"}]
    if command.command == "retain_safe":
        candidate["confirmed_config"] = deepcopy(dict(command.confirmed_config))
    return candidate


def _action_proposal(action: dict[str, Any], confidence: str) -> ActionProposal:
    return ActionProposal(
        action_id=str(action.get("action_id") or ""),
        action_type=str(action.get("type") or "unknown"),
        arguments={
            key: value
            for key, value in action.items()
            if key not in {"action_id", "type", "confidence", "reason"} and not key.startswith("_")
        },
        confidence=confidence if confidence in {"low", "medium", "high"} else "medium",  # type: ignore[arg-type]
        reason=str(action.get("reason") or ""),
    )


def _queue_has_followup_for_group(state: AgentGraphState, group: str) -> bool:
    action_types = {
        str(_queue_action(item).get("type") or "").strip()
        for item in state.get("action_queue") or []
        if isinstance(item, dict)
    }
    if group == "qps_profile":
        return bool(action_types & {"set_qps_mode", "request_qps_customization", "set_qps_override"})
    if group == "observability":
        return "set_observability" in action_types
    if group == "workload_rpc":
        return "set_rpc_mode" in action_types
    if group == "sync_observe":
        return bool(
            action_types
            & {
                "set_sync_observe_source",
                "clear_sync_observe_source",
                "set_sync_observe_options",
            }
        )
    return False


def _active_group_has_blocking_question(state: AgentGraphState) -> bool:
    active_group = str(state.get("active_group") or "").strip()
    if not active_group:
        return False
    return bool(_question_for_group(state, active_group))


def _ask_next_blocking_question(state: AgentGraphState) -> AgentGraphState:
    interrupted_question = _resume_suspended_question(state)
    if interrupted_question:
        current_question = dict(state.get("pending_question") or {})
        if current_question:
            _remove_rendered_question(state, current_question)
        _install_pending_question(state, interrupted_question)
        prefix = list(state.get("visible_response") or [])
        state["visible_response"] = prefix + [
            _render_question(interrupted_question, state.get("language", "en"))
        ]
        return state
    control = dict(state.get("control") or {})
    deferred_group = str(control.get("deferred_group") or "").strip()
    if deferred_group and deferred_group in ALLOWED_GROUPS:
        prerequisite = _navigation_prerequisite(state, deferred_group)
        if not prerequisite:
            control.pop("deferred_group", None)
            state["control"] = control
            return _activate_group_question(state, deferred_group)
        if prerequisite != str(state.get("active_group") or ""):
            return _activate_group_question(state, prerequisite)
    active_group = str(state.get("active_group") or "").strip()
    if active_group and active_group not in {"opening", "job_monitoring", "error_evidence_analysis", "report_artifact_analysis"}:
        active_question = _question_for_group(state, active_group)
        if active_question:
            _install_pending_question(state, active_question)
            if state.get("action_queue"):
                state["pending_question"]["resume_action_queue"] = True
            prefix = list(state.get("visible_response") or [])
            state["visible_response"] = prefix + [_render_question(active_question, state.get("language", "en"))]
            return state
    # A completed handler relinquishes its active group. Shared fallback then
    # computes the earliest relevant incomplete group from validated state.
    group = _next_group(state)
    _record_group_transition(state, group)
    question = _question_for_group(state, group)
    prefix = list(state.get("visible_response") or [])
    if question:
        _install_pending_question(state, question)
        if state.get("action_queue"):
            state["pending_question"]["resume_action_queue"] = True
        state["visible_response"] = prefix + [_render_question(question, state.get("language", "en"))]
    else:
        next_action = compute_next_action(state)
        state["pending_question"] = {}
        state["visible_response"] = prefix + [_localized(
            state.get("language", "en"),
            f"当前配置没有新的阻塞项。建议下一步：{format_recommended_next_action(next_action, state.get('language', 'zh'))}",
            f"No blocking configuration item remains. Recommended next action: {format_recommended_next_action(next_action, state.get('language', 'en'))}",
        )]
    return state


def _remove_rendered_question(state: AgentGraphState, question: PendingQuestion) -> None:
    """Remove only responses that render one exact typed question contract."""

    responses = list(state.get("visible_response") or [])
    if not responses or not question:
        return
    state["visible_response"] = [
        response
        for response in responses
        if not _question_is_actionable_in_responses([response], question)
    ]


def _reconfiguration_question(state: AgentGraphState, group: str) -> PendingQuestion | None:
    """Ask a domain's first owned question without discarding confirmed state.

    The projection is used only to construct the question. The real state and
    its evidence remain intact until the owning domain applies the user's
    replacement value and dependency invalidation.
    """

    spec = GROUP_SPEC_BY_NAME.get(group)
    if spec is None or not spec.fields:
        return None
    projected = deepcopy(state)
    confirmed = dict(projected.get("confirmed_config") or {})
    for field in spec.fields:
        confirmed.pop(field, None)
        projected.pop(field, None)
    projected["confirmed_config"] = confirmed
    projected.setdefault("group_states", {}).setdefault(group, {})["status"] = "reconfiguring"
    return _question_for_group(projected, group)


def _activate_group_question(
    state: AgentGraphState,
    group: str,
    *,
    record_history: bool = True,
    reconfigure: bool = False,
) -> AgentGraphState:
    _record_group_transition(state, group, record_history=record_history)
    state.setdefault("group_states", {}).setdefault(group, {})["status"] = "in_progress"
    # An explicit navigation to chain identity means "choose or change the
    # chain" even when the current chain is already confirmed. Keep the old
    # value until the user's replacement is confirmed; the chain owner then
    # applies dependency invalidation atomically.
    question = _question_for_group(state, group)
    if reconfigure and not question:
        question = _reconfiguration_question(state, group)
    if question:
        _install_pending_question(state, question)
        state["visible_response"] = [_render_question(question, state.get("language", "en"))]
    else:
        status = completed_group_status(state, group)
        if status:
            state["pending_question"] = {}
            state["visible_response"] = [status]
            return state
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            f"已进入 `{group}`。该配置组当前没有需要确认的阻塞项；你可以说明要修改的内容，或在下一轮继续默认配置流程。",
            f"Entered `{group}`. This configuration group currently has no blocking item; describe what to change, or continue the default configuration flow on the next turn.",
        )]
        return state
    return state


def _record_group_transition(state: AgentGraphState, next_group: str, *, record_history: bool = True) -> None:
    current = str(state.get("active_group") or "").strip()
    if record_history and current and current != next_group:
        history = list(state.get("group_history") or [])
        if not history or history[-1] != current:
            history.append(current)
        state["group_history"] = history[-20:]
    state["active_group"] = next_group
    mark_group_reconfiguring(state, next_group)


def _install_pending_question(state: AgentGraphState, question: PendingQuestion) -> None:
    """Install one blocking contract and enter its owner's repair lifecycle."""

    pending = dict(question)
    pending.setdefault("created_turn_index", int(state.get("turn_index") or 0))
    turn_context = dict(state.get("turn_context") or {})
    installed = [
        dict(item)
        for item in turn_context.get("installed_questions") or []
        if isinstance(item, dict)
    ]
    rendered = _render_question(pending, state.get("language", "en")).strip()
    if rendered and not any(
        _render_question(item, state.get("language", "en")).strip() == rendered
        for item in installed
    ):
        installed.append(dict(pending))
    turn_context["installed_questions"] = installed
    state["turn_context"] = turn_context
    state["pending_question"] = pending
    group = str(pending.get("group") or "").strip()
    if group:
        state["active_group"] = group
        mark_group_reconfiguring(state, group)


def _pop_previous_group(state: AgentGraphState) -> str:
    history = list(state.get("group_history") or [])
    current = str(state.get("active_group") or "").strip()
    while history:
        candidate = str(history.pop() or "").strip()
        if candidate and candidate != current and candidate in ALLOWED_GROUPS:
            state["group_history"] = history
            return candidate
    state["group_history"] = history
    return ""


def _discard_cancelled_origin_from_history(state: AgentGraphState) -> None:
    """Consume the abandoned group's history entry exactly once."""

    history = list(state.get("group_history") or [])
    origin = str(state.get("active_group") or "").strip()
    if history and str(history[-1] or "").strip() == origin:
        history.pop()
    state["group_history"] = history


def _next_group(state: AgentGraphState) -> str:
    """Return the next blocking group name.

    Delegates to `routing.next_group_and_reason`, the single shared
    implementation also used by `oracle.py`'s status/explanation text, so
    the two can never disagree about what group is next (see architecture
    audit Finding B1).
    """

    group, _reason = next_group_and_reason(state)
    return group


def _question_for_group(state: AgentGraphState, group: str) -> PendingQuestion | None:
    owner = GROUP_OWNER.get(group, "")
    runtime = DOMAIN_RUNTIME.get(owner)
    return runtime.question_factory(state, group) if runtime and runtime.question_factory else None


def _reconstruct_question(
    state: AgentGraphState,
    identity: Mapping[str, Any],
) -> PendingQuestion | None:
    """Rebuild one suspended typed question from current authoritative state."""

    group = str(identity.get("group") or "").strip()
    question_id = str(identity.get("question_id") or identity.get("id") or "").strip()
    if not group or not question_id:
        return None
    if question_id == "inferred_config_review":
        proposal = (state.get("inferred_config") or {}).get("pending_review")
        if isinstance(proposal, dict) and proposal:
            return config_proposal_review_question(
                group,
                proposal,
                language=str(state.get("language") or "en"),
            )
        return None
    field = str(identity.get("field") or "").strip()
    if field:
        runtime = DOMAIN_RUNTIME.get(GROUP_OWNER.get(group, ""))
        question = (
            runtime.field_question_factory(state, group, field)
            if runtime and runtime.field_question_factory
            else None
        )
        if question and str(question.get("id") or "") == question_id:
            return question
    question = _question_for_group(state, group)
    if question and str(question.get("id") or "") == question_id:
        return question
    return None

def _resume_field_reconfiguration_after_prerequisite(
    state: AgentGraphState,
    *,
    answered_question: PendingQuestion,
) -> AgentGraphState:
    """Consume one durable field-edit continuation after its prerequisite."""

    control = dict(state.get("control") or {})
    continuation = control.get("field_reconfiguration_continuation")
    if not isinstance(continuation, dict):
        return state
    if str(answered_question.get("id") or "") != str(
        continuation.get("prerequisite_question_id") or ""
    ):
        return state
    group = str(continuation.get("group") or "").strip()
    field = str(continuation.get("config_field") or "").strip()
    if group_for_field(field) != group:
        control.pop("field_reconfiguration_continuation", None)
        state["control"] = control
        return state
    runtime = DOMAIN_RUNTIME.get(GROUP_OWNER.get(group, ""))
    question = (
        runtime.field_question_factory(state, group, field)
        if runtime and runtime.field_question_factory
        else None
    )
    owner_spec = GROUP_SPEC_BY_NAME.get(group)
    if (
        not question
        or not owner_spec
        or str(question.get("group") or "") != group
        or str(question.get("id") or "") not in owner_spec.questions
        or str(question.get("field") or "") not in owner_spec.fields
    ):
        return state
    if str(question.get("field") or "") == field:
        control.pop("field_reconfiguration_continuation", None)
    else:
        continuation["prerequisite_question_id"] = str(question.get("id") or "")
        control["field_reconfiguration_continuation"] = continuation
    state["control"] = control
    _record_group_transition(state, group)
    _install_pending_question(state, question)
    state["visible_response"] = [_render_question(question, state.get("language", "en"))]
    return state


def _looks_like_assignment_answer(text: str) -> bool:
    raw = str(text or "").strip().rstrip(",，;；、").strip()
    if not raw or "\n" in raw:
        return False
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return False
    return all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:/-]*\s*=\s*[^=,]+", part) for part in parts)


# Typed actions that mean the user is navigating away from the secondary-handoff
# evidence collection (resetting, jumping to another group, asking to analyze a
# past report, or asking a question) rather than pasting development evidence.
# Deliberately excludes chain/mode selection and analyze_evidence: pasted
# development evidence routinely names chains, protocols, and endpoints, so the
# resolver classifies it as choose_chain/analyze_evidence — treating those as
# navigation would drop real evidence (verified via live DeepSeek runs).
_HANDOFF_NAVIGATION_ACTIONS = {
    "reset_session",
    "change_group",
    "go_back",
    "ask_capabilities",
    "answer_opening_question",
    "analyze_report",
}
