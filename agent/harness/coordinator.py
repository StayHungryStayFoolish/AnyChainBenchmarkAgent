"""Deterministic group workflow engine for the LangGraph Harness."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
from typing import Any, Mapping

from .state import AgentGraphState, PendingQuestion, RESET_PRESERVED_KEYS, new_state
from .intent import (
    ALLOWED_GROUPS,
    resolve_action_queue,
)
from .oracle import (
    compute_next_action,
    format_recommended_next_action,
)
from .routing import chain_identity_confirmed, next_group_and_reason
from .turns import adjudicate_turn
from .action_registry import ACTION_BY_TYPE, action_crosses_pending_barrier, action_execution_phase, action_is_turn_local, action_merge_key, assign_action_ids, compile_legacy_custom_rpc_action, merge_semantic_actions, normalize_action_relations, validate_action_contract
from .contracts import ActionProposal, CheckpointCommand, HandlerResult, RecoveryCommand
from .localization import localized as _localized
from .domains.orientation import completed_group_status
from .domains.environment import (
    apply_direct_config_assignments,
    apply_inferred_config_review,
    config_proposal_review_question,
    merge_config_proposal_from_text,
    is_assignment_only_config_text,
    parse_known_config_assignments,
)
from .domains.chain_rpc import apply_chain_rpc_action
from .domains.analysis import (
    JOB_ID_RE,
    EvidenceCollectionOutcome,
    cancel_evidence_collection,
    continue_evidence_collection,
    finish_evidence_collection,
    pause_evidence_collection,
    resume_evidence_collection,
    is_evidence_completion_command,
    prompt_evidence_collection_waiting,
    should_start_evidence_collection,
    start_evidence_collection,
)
from .domains.execution import reconcile_execution_state
from .domains.recovery import question_for_recovery
from .failures import domain_blocker_failure_record, model_provider_failure_record, render_failure_summary
from .domains.registry import GROUP_OWNER
from .domains.runtime import DOMAIN_RUNTIME, DomainRuntime
from .invariants import StateInvariantError, apply_state_delta, validate_state, verify_expected_patch
from .transitions import mark_group_reconfigured, mark_group_reconfiguring
from .questions import (
    action_for_value,
    answer_fits_pending as _answer_fits_pending,
    coerce_pending_answer as _coerce_answer,
    exact_answer as contract_exact_answer,
    expected_patch_for_value,
    manual_literal_violation,
    matches_numbered_option as _matches_numbered_option,
    pending_option_value_exists as _pending_option_value_exists,
    render_question as _render_question,
)
from .input_values import normalize_target_mode, target_mode_evidence_matches
from agent.workflows.group_registry import GROUP_ORDER, invalidation_targets
QUEUE_RESUME_PENDING_IDS = {
    "inferred_config_review",
    "target_mode_change_confirm",
    "chain_change_confirm",
    "chain_ambiguity_confirm",
    "unknown_chain_identity_confirm",
}


def apply_coordinator_action(state: AgentGraphState, action: ActionProposal) -> HandlerResult:
    """Apply navigation actions owned by the single workflow coordinator."""

    next_state = state
    if action.action_type == "resume_current_flow":
        pending = dict(next_state.get("pending_question") or {})
        if not pending:
            return HandlerResult(blocker="no active typed question is available to resume")
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            visible_result=_render_question(pending, next_state.get("language", "en")),
            pending_question=pending,
            completion="unchanged",
            stop_after_response=True,
        )
    if action.action_type == "change_group":
        group = str(action.arguments.get("group") or "").strip()
        if group not in ALLOWED_GROUPS:
            return HandlerResult(blocker=f"unknown workflow group: {group or '<missing>'}")
        if group == "workload_rpc" and not chain_identity_confirmed(next_state):
            control = dict(next_state.get("control") or {})
            control["deferred_group"] = group
            next_state["control"] = control
            next_state = _activate_group_question(next_state, "chain_identity")
            return HandlerResult(
                consumed_action_ids=(action.action_id,),
                completion="blocked",
                stop_after_response=True,
            )
        if group == str(action.arguments.get("queue_origin_group") or "").strip() and _active_group_has_blocking_question(next_state):
            pending = dict(next_state.get("pending_question") or {})
            return HandlerResult(
                consumed_action_ids=(action.action_id,),
                visible_result=_render_question(pending, next_state.get("language", "en")),
                completion="unchanged",
                stop_after_response=True,
            )
        if _queue_has_followup_for_group(next_state, group):
            _record_group_transition(next_state, group)
            next_state["pending_question"] = {}
            return HandlerResult(
                consumed_action_ids=(action.action_id,),
                completion="completed",
            )
        if group == "chain_identity" and (next_state.get("chain_identity") or {}).get("canonical"):
            return HandlerResult(
                consumed_action_ids=(action.action_id,),
                followup_actions=(
                    {
                        "type": "request_chain_selection",
                        "confidence": "high",
                    },
                ),
                completion="completed",
            )
        next_state = _activate_group_question(next_state, group)
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            completion="blocked" if next_state.get("pending_question") else "completed",
            stop_after_response=True,
        )
    if action.action_type == "go_back":
        resume_group = str(action.arguments.get("cancellation_resume_group") or "").strip()
        if resume_group:
            _discard_cancelled_origin_from_history(next_state)
        previous_group = resume_group or _pop_previous_group(next_state)
        if previous_group:
            next_state = _activate_group_question(next_state, previous_group, record_history=False)
        else:
            next_state["visible_response"] = [_localized(
                next_state.get("language", "en"),
                "当前没有可回退的配置组。你可以直接说明要回到哪个配置项，例如 RPC、QPS、磁盘或可观测性。",
                "There is no previous configuration group to return to. Name the area to revisit, such as RPC, QPS, disk, or observability.",
            )]
            next_state["_stop_after_response"] = True
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            completion="completed",
            stop_after_response=True,
        )
    if action.action_type == "queue_workflow_goal":
        target_mode = normalize_target_mode(action.arguments.get("target_mode"))
        goal = str(action.arguments.get("goal") or "").strip()
        source = str(action.arguments.get("source_evidence") or "").strip()
        if not target_mode or not goal or not source:
            return HandlerResult(blocker="queued workflow goal requires target_mode, goal, and source_evidence")
        goals = [dict(item) for item in next_state.get("workflow_goals") or [] if isinstance(item, dict)]
        candidate = {"target_mode": target_mode, "goal": goal, "source_evidence": source}
        if not any(
            item.get("target_mode") == target_mode and item.get("goal") == goal
            for item in goals
        ):
            goals.append(candidate)
        next_state["workflow_goals"] = goals
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            visible_result=_localized(
                next_state.get("language", "en"),
                f"已保存后续目标：完成当前流程后进入 `{target_mode}`（{goal}）。",
                f"Saved a later goal: enter `{target_mode}` after the current workflow ({goal}).",
            ),
            completion="completed",
        )
    if action.action_type in {"activate_next_workflow_goal", "discard_next_workflow_goal"}:
        goals = [dict(item) for item in next_state.get("workflow_goals") or [] if isinstance(item, dict)]
        if not goals:
            return HandlerResult(blocker="no queued workflow goal is available")
        goal = goals.pop(0)
        next_state["workflow_goals"] = goals
        if action.action_type == "discard_next_workflow_goal":
            return HandlerResult(
                consumed_action_ids=(action.action_id,),
                visible_result=_localized(
                    next_state.get("language", "en"),
                    "已移除最早保存的后续测试目标。",
                    "Removed the oldest saved workflow goal.",
                ),
                completion="completed",
            )
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            followup_actions=({
                "type": "choose_target_mode",
                "target_mode": str(goal.get("target_mode") or ""),
                "target_mode_explicit": True,
                "source_evidence": str(goal.get("source_evidence") or "").strip(),
                "selection_contract_verified": True,
                "confidence": "high",
            },),
            visible_result=_localized(
                next_state.get("language", "en"),
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

    action_type = str(action.get("type") or "").strip()
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
    if action_type == "change_group":
        target_group = str(action.get("group") or "").strip()
        if target_group:
            admitted_action["group"] = target_group
    admitted.append(admitted_action)


def prepare_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Create one immutable turn snapshot before any routing or mutation."""

    state = _copy_state(state)
    state["action_queue"] = _durable_actions(state.get("action_queue") or [])
    state = _apply_handler_result(state, reconcile_execution_state(state), owner="execution")
    pending_owner = str((state.get("pending_question") or {}).get("group") or "").strip()
    if pending_owner:
        state["active_group"] = pending_owner
    state["turn_index"] = int(state.get("turn_index") or 0) + 1
    text = str(state.get("last_user_input") or "").strip()
    state["visible_response"] = []
    state["current_action"] = {}
    state["completed_actions"] = []
    state["action_errors"] = []
    turn = adjudicate_turn(state, text)
    state["turn_context"] = {
        "id": int(state.get("turn_index") or 0),
        "kind": str(turn.kind),
        "text": text,
        "origin_group": str(state.get("active_group") or ""),
        "pending_snapshot": dict(state.get("pending_question") or {}),
        "admitted_actions": [],
    }
    state["proposed_actions"] = []
    return _set_turn_phase(state, "adjudicate")


def adjudicate_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Handle only deterministic contracts; free text proceeds to planning."""

    text = str((state.get("turn_context") or {}).get("text") or "")
    turn_kind = str((state.get("turn_context") or {}).get("kind") or "free_text")
    language = str(state.get("language") or "en")
    collecting = state.get("evidence_collection") or {}

    if turn_kind == "evidence_continuation":
        if not text:
            state = _apply_handler_result(state, prompt_evidence_collection_waiting(state, collecting), owner="analysis")
            return _set_turn_phase(state, "compose", "evidence_waiting")
        if is_evidence_completion_command(text):
            state = _apply_evidence_outcome(state, continue_evidence_collection(state, text, collecting))
            if state.pop("_stop_after_response", False) or state.get("pending_question"):
                return _set_turn_phase(state, "compose", "evidence_collected")
            return _set_turn_phase(state, "fallback", "evidence_collected")
        return _set_turn_phase(state, "plan", "typed_evidence_collection_turn")

    if turn_kind == "empty":
        return _set_turn_phase(state, "fallback", "empty_turn")

    pending = state.get("pending_question") or {}
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
    if str(pending.get("id") or "") == "inferred_config_review":
        merged = _merge_pending_config_proposal_from_text(state, text)
        if merged:
            return _set_turn_phase(merged, "compose", "merged_config_proposal")

    direct_assignments = parse_known_config_assignments(text) if is_assignment_only_config_text(text) else {}
    if direct_assignments:
        state = _apply_handler_result(state, apply_direct_config_assignments(state, direct_assignments), owner="environment")
        return _set_turn_phase(state, "fallback", "direct_assignments")

    if pending and str(pending.get("kind") or "") == "evidence" and should_start_evidence_collection(text):
        state = _apply_evidence_outcome(state, start_evidence_collection(state, text, pending))
        return _set_turn_phase(state, "compose", "evidence_collection_started")

    if pending and _answer_fits_pending(text, pending):
        pending_question_id = str(pending.get("id") or "")
        resume_queue = bool(
            _queue_has_admitted_durable_work(state)
            or pending.get("resume_action_queue")
            or pending_question_id in QUEUE_RESUME_PENDING_IDS
        )
        if not resume_queue:
            # Discard only deferred work that predates this blocking answer.
            # The answer handler may emit new follow-up actions (for example a
            # recommended chain+mode setup); clearing after the handler would
            # silently erase those newly authorized actions.
            state["action_queue"] = []
        state = _apply_pending_answer(state, text, pending)
        state.pop("_handler_completion", None)
        if resume_queue:
            _discard_superseded_queue_actions(state, pending)
        if resume_queue and state.get("pending_question"):
            state["pending_question"]["resume_action_queue"] = True
        if resume_queue and not state.get("_stop_after_response") and state.get("action_queue") and _queue_can_continue_through_pending(state):
            return _set_turn_phase(state, "execute", "resume_queue_after_answer")
        if not state.get("pending_question") and not state.get("_stop_after_response") and state.get("action_queue"):
            return _set_turn_phase(state, "execute", "continue_queue_after_answer")
        if state.pop("_stop_after_response", False) or state.get("pending_question"):
            return _set_turn_phase(state, "compose", "pending_answer_applied")
        return _set_turn_phase(state, "fallback", "pending_answer_applied")

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


def plan_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Call the configured LLM once to propose typed actions."""

    text = str((state.get("turn_context") or {}).get("text") or "")
    queue = resolve_action_queue(state, text)
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
    state["proposed_actions"] = actions
    return _set_turn_phase(state, "admit", "planner_completed")


def admit_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Execute turn-local results and admit durable commands to the queue."""

    text = str((state.get("turn_context") or {}).get("text") or "")
    actions = _validate_action_plan(state, list(state.get("proposed_actions") or []))
    scope = f"{state.get('thread_id') or 'default'}:{int(state.get('turn_index') or 0)}"
    actions = assign_action_ids(scope, text, actions)
    origin_group = str(state.get("active_group") or "")
    for plan_index, action in enumerate(actions):
        action.setdefault("_origin_text", text)
        action.setdefault("_queue_origin_group", origin_group)
        action.setdefault("_submitted_turn_index", int(state.get("turn_index") or 0))
        action.setdefault("_plan_scope", scope)
        action.setdefault("_plan_index", plan_index)
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
    if turn_local_actions:
        state = _execute_turn_local_actions(state, turn_local_actions, text)
    pending = state.get("pending_question") or {}
    ordered_queue = _order_action_queue(
        state,
        _merge_durable_action_queue(
            [dict(item) for item in state.get("action_queue") or [] if isinstance(item, dict)],
            durable_actions,
        ),
    )
    if (
        str(pending.get("id") or "") == "inferred_config_review"
        and not any(str(item.get("type") or "") == "answer_pending" for item in durable_actions)
        and not (ordered_queue and action_crosses_pending_barrier(ordered_queue[0]))
    ):
        state["action_queue"] = ordered_queue
        return _set_turn_phase(state, "compose", "config_review_barrier")
    state["action_queue"] = ordered_queue
    state["completed_actions"] = []
    state["action_errors"] = []
    if state.get("action_queue"):
        if (
            turn_local_actions
            and state.get("pending_question")
            and _pending_is_queue_barrier(state)
            and not _queue_can_continue_through_pending(state)
        ):
            return _set_turn_phase(state, "compose", "turn_local_barrier_preserved")
        return _set_turn_phase(state, "execute", "actions_admitted")
    if turn_local_actions:
        return _set_turn_phase(state, "compose", "turn_local_actions_completed")
    return _set_turn_phase(state, "fallback", "no_durable_actions")


def _execute_turn_local_actions(
    state: AgentGraphState,
    actions: list[dict[str, Any]],
    text: str,
) -> AgentGraphState:
    """Dispatch read-only actions without committing them to workflow state."""

    pending_snapshot = deepcopy(state.get("pending_question") or {})
    responses = list(state.get("visible_response") or [])
    suppress_pending_render = False
    for action in actions:
        isolated = deepcopy(state)
        isolated["action_queue"] = []
        isolated["visible_response"] = []
        routed = _apply_queue_action(isolated, action, text)
        if routed is None:
            continue
        action_type = str(action.get("type") or "")
        suppress_pending_render = suppress_pending_render or bool(
            routed.get("_stop_after_response")
            and action_type in {"analyze_evidence", "analyze_report"}
        )
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        for root in tuple(spec.turn_local_result_roots if spec else ()):
            if root in routed:
                state[root] = deepcopy(routed[root])
            else:
                state.pop(root, None)
        for response in routed.get("visible_response") or []:
            if pending_snapshot and _question_is_actionable_in_responses([str(response)], pending_snapshot):
                continue
            if response and response not in responses:
                responses.append(response)
        if not pending_snapshot and routed.get("pending_question"):
            state["pending_question"] = deepcopy(routed.get("pending_question") or {})
            state["active_group"] = str(routed.get("active_group") or state.get("active_group") or "")
    state["visible_response"] = responses
    if pending_snapshot:
        state["pending_question"] = pending_snapshot
    if suppress_pending_render:
        state.setdefault("turn_context", {})["suppress_pending_render"] = True
    return state


def execute_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Execute exactly one admitted action through its owning domain."""

    queued = state.get("action_queue") or []
    text = str(
        (state.get("turn_context") or {}).get("text")
        or ((queued[0] or {}).get("_origin_text") if queued else "")
        or ""
    )
    routed = _process_action_queue(state, [], text, max_actions=1)
    if routed is not None:
        state = routed
    stop = bool(state.pop("_stop_after_response", False))
    if (
        not stop
        and state.get("action_queue")
        and (
            not state.get("pending_question")
            or _queue_can_continue_through_pending(state)
            or _next_queue_action_crosses_pending_barrier(state)
        )
    ):
        return _set_turn_phase(state, "execute", "queue_has_next_action")
    if stop or state.get("pending_question"):
        return _set_turn_phase(state, "compose", "queue_blocked_or_complete")
    return _set_turn_phase(state, "fallback", "queue_complete")


def fallback_turn_step(state: AgentGraphState) -> AgentGraphState:
    state = _ask_next_blocking_question(state)
    return _set_turn_phase(state, "compose", "fallback_resolved")


def compose_turn_step(state: AgentGraphState) -> AgentGraphState:
    state = _finalize_turn_response(state)
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


def _merge_pending_config_proposal_from_text(state: AgentGraphState, text: str) -> AgentGraphState | None:
    inferred = state.setdefault("inferred_config", {})
    current = inferred.get("pending_review") if isinstance(inferred.get("pending_review"), dict) else {}
    merged = merge_config_proposal_from_text(current, text)
    if merged is None:
        return None
    inferred["pending_review"] = merged
    current_group = str((state.get("pending_question") or {}).get("group") or state.get("active_group") or "provider_deployment")
    state["pending_question"] = config_proposal_review_question(
        current_group,
        merged,
        language=str(state.get("language") or "en"),
    )
    state["visible_response"] = [_render_question(state["pending_question"], state.get("language", "en"))]
    return state


def _normalized_action_queue(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_actions = payload.get("actions") if isinstance(payload, dict) else None
    if isinstance(raw_actions, dict):
        raw_actions = [raw_actions]
    if not isinstance(raw_actions, list):
        return []
    output: list[dict[str, Any]] = []
    compiled_actions = [
        compiled
        for raw in raw_actions[:12]
        if isinstance(raw, dict)
        for compiled in compile_legacy_custom_rpc_action(raw)
    ]
    for raw in compiled_actions[:12]:
        try:
            # The resolver is the sole producer of trusted admission receipts.
            # Model documents are validated before those receipts are attached.
            action = validate_action_contract(raw, trusted_metadata=True)
        except ValueError as exc:
            action = {"type": "unknown", "reason": str(exc), "confidence": "low"}
        action_type = str(action.get("type") or action.get("intent") or "unknown").strip()
        action["type"] = action_type
        action["confidence"] = str(action.get("confidence") or "medium").strip().lower()
        output.append(action)
    return output


def _bind_declared_option_actions(
    state: AgentGraphState,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Route semantic menu selections back through the option contract."""

    pending = dict(state.get("pending_question") or {})
    options = list(pending.get("options") or [])
    if not options:
        return actions
    raw_matched, raw_value = contract_exact_answer(
        str(state.get("last_user_input") or ""),
        pending,
    )
    output: list[dict[str, Any]] = []
    for action in actions:
        action_type = str(action.get("type") or "")
        if action_type == "answer_pending":
            # A model may return the exact displayed option label/id while the
            # contract stores a typed value (for example label ``Y`` and value
            # ``True``). Resolve only against the declared option contract and
            # carry its typed value forward; no aliases or fuzzy matching are
            # allowed at this boundary.
            if raw_matched:
                normalized = dict(action)
                normalized["answer"] = raw_value
                normalized["selected_value"] = raw_value
                normalized["selection_contract_verified"] = True
                output.append(normalized)
            else:
                output.append(action)
            continue
        spec = ACTION_BY_TYPE.get(action_type)
        semantic = str(spec.pending_option_semantic if spec else "").strip()
        declared_option = next(
            (
                option for option in options
                if semantic and str(option.get("semantic_action") or "").strip() == semantic
            ),
            None,
        )
        if declared_option is not None:
            selected = declared_option.get("value")
            output.append({
                "type": "answer_pending",
                "answer": selected,
                "selected_value": selected,
                "source_evidence": action.get("source_evidence"),
                "pending_option_semantic_verified": True,
            })
            continue
        # A domain action that happens to match one displayed option is not
        # proof that the user selected that option. Keep it as a domain action
        # so its owner-specific source-evidence guard remains authoritative.
        output.append(action)
    return output


def _action_matches_declared_pending_option(
    state: AgentGraphState,
    action: Mapping[str, Any],
) -> bool:
    """Verify a semantic selection against the active typed option contract."""

    pending = state.get("pending_question") or {}
    for option in pending.get("options") or []:
        declared = option.get("action") if isinstance(option, Mapping) else None
        if not isinstance(declared, Mapping):
            continue
        if all(
            key == "type" or action.get(key) == value
            for key, value in declared.items()
        ):
            return True
    return False


def _validate_action_plan(state: AgentGraphState, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate typed plan semantics without reinterpreting user language."""
    if not actions:
        return actions
    prepared = normalize_action_relations(
        _bind_declared_option_actions(state, [dict(item) for item in actions])
    )
    for item in prepared:
        if (
            item.get("pending_option_semantic_verified") is True
            and _action_matches_declared_pending_option(state, item)
        ):
            item["selection_contract_verified"] = True
    identity = state.get("chain_identity") or {}
    if identity.get("case") == "case3" and identity.get("adapter_family") == "unsupported":
        # An unsupported-family handoff cannot also mutate the RPC catalog.
        # The user must explicitly leave Case 3 by changing chain/family first.
        changes_case = any(
            str(item.get("type") or "") in {"choose_chain", "change_chain", "choose_adapter_family"}
            for item in prepared
        )
        if not changes_case:
            prepared = [
                item for item in prepared
                if str(item.get("type") or "") not in {"rpc_catalog_command", "rpc_workload_command"}
            ]
    if any(str(item.get("type") or "") != "unknown" for item in prepared):
        # ``unknown`` is the whole-turn fallback, not an executable sibling.
        # A provider may emit one invalid proposal and other valid actions in
        # the same plan; retaining the normalized fallback would block those
        # valid actions and surface a false "no safe action" response.
        prepared = [item for item in prepared if str(item.get("type") or "") != "unknown"]
    prepared = _drop_conflicting_answer_actions(state, prepared)
    prepared = _drop_answers_to_pending_invalidated_by_plan(state, prepared)
    if state.get("target_mode"):
        # A free-standing model request cannot reopen replacement of a
        # confirmed mode. Declared menu choices have already been rebound to
        # answer_pending; explicit user navigation uses change_group.
        prepared = [
            item
            for item in prepared
            if str(item.get("type") or "") != "request_target_mode_selection"
        ]
    admitted: list[dict[str, Any]] = []
    current_answer_checked = False
    for item in prepared:
        if str(item.get("type") or "") != "answer_pending":
            admitted.append(item)
            continue
        # A user turn can answer only the question that existed when the turn
        # began. Future decisions must be represented by domain actions, not by
        # speculative answer_pending entries for questions that do not exist.
        if not current_answer_checked:
            current_answer_checked = True
            if _action_answers_pending_contract(state, item):
                admitted.append(item)
            continue
        continue
    prepared = admitted
    pending_id = str((state.get("pending_question") or {}).get("id") or "")
    # A model plan cannot answer a domain question that does not exist yet.
    # Protocol selection is intentionally a separate user-confirmed decision
    # after an unknown-chain identity choice. Exact menu answers are applied
    # locally and never need this speculative domain action.
    if pending_id not in {"adapter_family_confirm", "custom_rpc_adapter_family_confirm"}:
        prepared = [
            item for item in prepared
            if str(item.get("type") or "") != "choose_adapter_family"
        ]
    if any(str(item.get("type") or "") == "answer_pending" for item in prepared):
        accepted = {
            str(action_type).strip()
            for action_type in (state.get("pending_question") or {}).get("accepted_action_types") or []
            if str(action_type).strip() and str(action_type).strip() != "answer_pending"
        }
        # One turn may express the choice both as an answer to the current
        # contract and as a domain mutation. The answered contract is the
        # authority; replaying its equivalent setter can recreate the same
        # question and reorder unrelated queued work.
        prepared = [
            item for item in prepared
            if str(item.get("type") or "") == "answer_pending"
            or str(item.get("type") or "") not in accepted
        ]
    prepared = [
        item
        for item in prepared
        if str(item.get("type") or "") != "choose_target_mode"
        or (
            bool(normalize_target_mode(item.get("target_mode")))
            and (
                item.get("selection_contract_verified") is True
                or item.get("target_mode_semantic_verified") is True
                or target_mode_evidence_matches(
                    item.get("target_mode"),
                    item.get("source_evidence"),
                    state.get("last_user_input"),
                )
            )
        )
    ]
    prepared = [
        item
        for item in prepared
        if str(item.get("type") or "") != "queue_workflow_goal"
        or (
            bool(normalize_target_mode(item.get("target_mode")))
            and bool(str(item.get("goal") or "").strip())
            and bool(str(item.get("source_evidence") or "").strip())
            and str(item.get("source_evidence") or "").strip() in str(state.get("last_user_input") or "")
        )
    ]
    normalized_partial_actions: list[dict[str, Any]] = []
    for item in prepared:
        if str(item.get("type") or "") == "set_qps_override" and not isinstance(item.get("qps_overrides"), dict):
            item = {
                **item,
                "type": "request_qps_customization",
                "qps_fields": item.get("qps_fields") or [],
            }
            item.pop("qps_overrides", None)
        normalized_partial_actions.append(item)
    prepared = normalized_partial_actions
    prepared = [
        item
        for item in prepared
        if str(item.get("type") or "") != "change_group"
        or item.get("selection_contract_verified") is True
        or (
            item.get("navigation_explicit") is True
            and bool(str(item.get("source_evidence") or "").strip())
            and str(item.get("source_evidence") or "").strip() in str(state.get("last_user_input") or "")
        )
    ]
    extension_consultation = any(
        str(item.get("type") or "") == "answer_opening_question"
        and str(item.get("topic") or "").strip().lower() == "extension"
        for item in prepared
    )
    if extension_consultation:
        prepared = [
            item
            for item in prepared
            if str(item.get("type") or "") != "rpc_catalog_command"
            or str(item.get("catalog_command") or "") != "enter"
        ]
    mutation_types = {
        "choose_chain", "change_chain", "set_rpc_mode", "set_qps_mode", "request_qps_customization",
        "set_qps_override", "set_observability", "rpc_catalog_command", "rpc_workload_command",
    }
    has_benchmark_mutation = any(str(item.get("type") or "") in mutation_types for item in prepared)
    chooses_mode = any(
        str(item.get("type") or "") == "choose_target_mode"
        or (
            str(item.get("type") or "") == "answer_pending"
            and _pending_answer_declared_action_type(state, item) == "choose_target_mode"
        )
        for item in prepared
    )
    requests_mode = any(str(item.get("type") or "") == "request_target_mode_selection" for item in prepared)
    if has_benchmark_mutation and not state.get("target_mode") and not chooses_mode and not requests_mode:
        prepared.append({
            "type": "request_target_mode_selection",
            "confidence": "high",
            "reason": "benchmark actions require an explicit target mode",
        })
    prepared = _drop_redundant_group_navigation(state, prepared)
    return _ensure_action_prerequisites(state, prepared)


def _drop_answers_to_pending_invalidated_by_plan(
    state: AgentGraphState,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reject answers bound to a question made stale by the same transaction."""

    pending_group = str((state.get("pending_question") or {}).get("group") or "").strip()
    if not pending_group:
        return actions
    for action in actions:
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if spec is None or not spec.mutation_dimension:
            continue
        source_group = str(spec.target_group or spec.mutation_dimension).strip()
        if pending_group in set(invalidation_targets(source_group)):
            return [item for item in actions if str(item.get("type") or "") != "answer_pending"]
    return actions


def _drop_redundant_group_navigation(state: AgentGraphState, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove navigation when a concrete action already owns that destination."""

    concrete_groups = {
        "chain_identity": {"choose_chain", "change_chain", "request_chain_selection"},
        "workload_rpc": {"set_rpc_mode", "rpc_catalog_command", "rpc_workload_command", "use_default_workload", "configure_workload_weights"},
        "qps_profile": {"set_qps_mode", "request_qps_customization", "set_qps_override"},
        "observability": {"set_observability"},
        "sync_observe": {
            "set_sync_observe_source",
            "clear_sync_observe_source",
            "set_sync_observe_options",
        },
    }
    action_types = {str(item.get("type") or "") for item in actions}
    output: list[dict[str, Any]] = []
    for item in actions:
        if str(item.get("type") or "") == "change_group":
            group = str(item.get("group") or "")
            if action_types & concrete_groups.get(group, set()):
                continue
            if "answer_pending" in action_types and group == str(state.get("active_group") or ""):
                continue
        output.append(item)
    return output


def _ensure_action_prerequisites(state: AgentGraphState, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add typed prerequisite questions without inventing user choices."""

    output = list(actions)
    action_types = {str(item.get("type") or "") for item in output}
    chain_required = {"set_rpc_mode", "rpc_workload_command", "use_default_workload", "configure_workload_weights"}
    chain_confirmed = chain_identity_confirmed(state)
    identity = state.get("chain_identity") or {}
    catalog_continues_case2 = (
        identity.get("case") == "case2"
        and str(identity.get("status") or "").startswith("existing_family_")
    )
    needs_chain_prerequisite = bool(action_types & chain_required) or (
        "rpc_catalog_command" in action_types and not catalog_continues_case2
    )
    chain_planned = bool(action_types & {"choose_chain", "change_chain", "request_chain_selection"})
    if needs_chain_prerequisite and not chain_confirmed and not chain_planned:
        output.insert(0, {
            "type": "change_group",
            "group": "workload_rpc",
            "navigation_explicit": True,
            "source_evidence": str(state.get("last_user_input") or ""),
            "confidence": "high",
            "reason": "workload actions require confirmed chain identity",
        })
    return output


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
                and _answer_fits_pending(answer, pending)
                and action.get("semantic_purpose_verified") is True
            )
        if not (
            action.get("selection_contract_verified") is True
            or action.get("pending_option_semantic_verified") is True
        ):
            return False
        return True
    selected = action.get("selected_value")
    if isinstance(selected, str) and not selected.strip():
        selected = None
    answer = str(selected if selected is not None else action.get("answer") or "").strip()
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
        answer
        and (
            declared_option
            or _answer_fits_pending(answer, pending)
        )
    )


def _pending_answer_declared_action_type(state: AgentGraphState, action: dict[str, Any]) -> str:
    pending = state.get("pending_question") or {}
    selected = action.get("selected_value")
    declared = action_for_value(pending, selected)
    return str(declared.get("type") or "")


def _drop_conflicting_answer_actions(state: AgentGraphState, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user_text = str(state.get("last_user_input") or "")
    pending = dict(state.get("pending_question") or {})
    evidenced_consultation_topics = {
        str(item.get("topic") or "").strip().lower()
        for item in actions
        if str(item.get("type") or "") == "answer_opening_question"
        and bool(str(item.get("source_evidence") or "").strip())
        and str(item.get("source_evidence") or "").strip() in user_text
    }
    if evidenced_consultation_topics:
        actions = [
            item
            for item in actions
            if not (
                str(item.get("type") or "") == "answer_opening_question"
                and str(item.get("topic") or "").strip().lower() in evidenced_consultation_topics
                and not (
                    bool(str(item.get("source_evidence") or "").strip())
                    and str(item.get("source_evidence") or "").strip() in user_text
                )
            )
        ]
        raw_selects_pending, _selected_value = contract_exact_answer(user_text, pending)
        if pending and not raw_selects_pending:
            # A read-only consultation is an overlay on the active workflow.
            # The model may not turn that prose into a menu value and thereby
            # consume the pending contract. Exact option selection is owned by
            # the raw-input contract matcher above.
            actions = [
                item
                for item in actions
                if str(item.get("type") or "") != "answer_pending"
                or item.get("pending_option_semantic_verified") is True
            ]
    requests_target_mode = any(
        str(item.get("type") or "") == "request_target_mode_selection"
        for item in actions
    )
    if requests_target_mode and str((state.get("pending_question") or {}).get("id") or "") == "opening_next_action":
        actions = [
            item
            for item in actions
            if str(item.get("type") or "") != "answer_pending"
        ]
    if any(str(item.get("type") or "") == "change_group" for item in actions):
        actions = [item for item in actions if str(item.get("type") or "") != "go_back"]
    if any(str(item.get("type") or "") == "analyze_report" for item in actions):
        actions = [
            item
            for item in actions
            if not (
                str(item.get("type") or "") in {"ask_capabilities", "answer_opening_question"}
                and str(item.get("topic") or "capabilities").strip().lower()
                in {"capabilities", "current_job", "job_status", "execution_status", "evidence_help", "log_help"}
            )
        ]
    consultation_topics = {
        str(item.get("topic") or "").strip().lower()
        for item in actions
        if str(item.get("type") or "") == "answer_opening_question"
    }
    if "workload_config" in consultation_topics:
        workload_will_be_rendered_by_mutation = any(
            str(item.get("type") or "") in {
                "set_rpc_mode",
                "use_default_workload",
                "configure_workload_weights",
                "rpc_catalog_command",
                "rpc_workload_command",
            }
            for item in actions
        )
        actions = [
            item
            for item in actions
            if not (
                str(item.get("type") or "") == "answer_opening_question"
                and (
                    str(item.get("topic") or "").strip().lower()
                    in {"current_config", "config_explanation"}
                    or (
                        workload_will_be_rendered_by_mutation
                        and str(item.get("topic") or "").strip().lower() == "workload_config"
                    )
                )
            )
        ]
        consultation_topics = {
            str(item.get("topic") or "").strip().lower()
            for item in actions
            if str(item.get("type") or "") == "answer_opening_question"
        }
    if "current_config" in consultation_topics:
        # The current-config renderer owns current context and includes the
        # computed next blocker. These two topics are proven output subsets,
        # unlike requirements/workflow questions which must remain independent.
        actions = [
            item
            for item in actions
            if not (
                str(item.get("type") or "") == "answer_opening_question"
                and str(item.get("topic") or "").strip().lower() in {"current_context", "next_action"}
            )
        ]
    topics = {
        str(item.get("topic") or "").strip().lower()
        for item in actions
        if str(item.get("type") or "") == "answer_opening_question"
    }
    specific_chain_consultation = any(
        str(item.get("type") or "") == "answer_opening_question"
        and str(item.get("topic") or "").strip().lower() == "supported_chains"
        and bool(str(item.get("subject") or "").strip())
        for item in actions
    )
    if specific_chain_consultation:
        actions = [
            item
            for item in actions
            if not (
                str(item.get("type") or "") in {"ask_capabilities", "answer_opening_question"}
                and str(item.get("topic") or "capabilities").strip().lower() in {"identity", "capabilities", "agent_capability"}
            )
        ]
        topics = {
            str(item.get("topic") or "").strip().lower()
            for item in actions
            if str(item.get("type") or "") == "answer_opening_question"
        }
    if not (topics & {"performance_benchmark_guidance", "mode_comparison"}):
        return actions
    pruned: list[dict[str, Any]] = []
    for item in actions:
        action_type = str(item.get("type") or "").strip()
        topic = str(item.get("topic") or "").strip().lower()
        target_mode = _normalized_target_mode(item.get("target_mode"))
        if action_type == "answer_opening_question" and topic in {"recommendation", "recommend_start"}:
            continue
        if "performance_benchmark_guidance" in topics and action_type == "choose_target_mode" and target_mode == "fake-node":
            continue
        pruned.append(item)
    return pruned


def _has_meaningful_queue(actions: list[dict[str, Any]]) -> bool:
    return any(
        action_type in ACTION_BY_TYPE and action_type != "unknown"
        for item in actions
        if (action_type := str(item.get("type") or "").strip())
    )


def _process_action_queue(
    state: AgentGraphState,
    actions: list[dict[str, Any]],
    text: str,
    *,
    max_actions: int | None = None,
) -> AgentGraphState | None:
    turn_local_actions = [dict(item) for item in actions if isinstance(item, dict) and action_is_turn_local(item)]
    durable_incoming = [dict(item) for item in actions if isinstance(item, dict) and not action_is_turn_local(item)]
    if turn_local_actions:
        state = _execute_turn_local_actions(state, turn_local_actions, text)
    prepared_actions = []
    origin_group = str(state.get("active_group") or "").strip()
    for item in durable_incoming:
        action = dict(item)
        action.setdefault("_origin_text", text)
        action.setdefault("_queue_origin_group", origin_group)
        action.setdefault("_submitted_turn_index", int(state.get("turn_index") or 0))
        prepared_actions.append(action)
    prepared_actions = _merge_durable_action_queue(
        [dict(item) for item in state.get("action_queue") or [] if isinstance(item, dict)],
        prepared_actions,
    )
    prepared_actions = _order_action_queue(state, prepared_actions)
    state["action_queue"] = prepared_actions
    if (
        turn_local_actions
        and state.get("pending_question")
        and state.get("action_queue")
        and _pending_is_queue_barrier(state)
        and not _queue_can_continue_through_pending(state)
    ):
        _append_active_question_once(state)
        state["current_action"] = {}
        return state
    if actions:
        state["completed_actions"] = []
        state["action_errors"] = []
    changed = False
    executed = 0
    while state.get("action_queue"):
        action = dict(state["action_queue"].pop(0))
        remaining_actions = [dict(item) for item in state.get("action_queue") or [] if isinstance(item, dict)]
        action_id = str(action.get("action_id") or "")
        if action_id and action_id in set(state.get("applied_action_ids") or []):
            continue
        state["current_action"] = action
        before_pending = dict(state.get("pending_question") or {})
        before_pending_id = str(before_pending.get("id") or "")
        before_responses = list(state.get("visible_response") or [])
        routed = _apply_queue_action(state, action, text)
        if routed is None:
            state.setdefault("action_errors", []).append({"action": action, "error": "unsupported_or_low_confidence"})
            continue
        state = routed
        if action.get("type") == "reset_session" and remaining_actions:
            # Reset clears state from earlier turns, not explicit work still
            # queued from the current user turn.  The orientation domain uses
            # replace-state semantics, so the coordinator must reattach the
            # already-validated remainder and allow it to continue.
            state["action_queue"] = remaining_actions
            state.pop("_stop_after_response", None)
        action_spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        after_action_pending_id = str((state.get("pending_question") or {}).get("id") or "")
        if (
            action_spec is not None
            and action_spec.preserve_pending
            and before_pending_id
            and after_action_pending_id != before_pending_id
        ):
            _push_interruption_frame(state, before_pending, reason=f"{action_spec.action_type}_overlay")
        if before_responses and action.get("type") != "reset_session":
            merged_responses = list(before_responses)
            if before_pending_id and after_action_pending_id != before_pending_id:
                merged_responses = _without_superseded_question(
                    merged_responses,
                    before_pending,
                    state.get("language", "en"),
                )
            for response in state.get("visible_response") or []:
                if response not in merged_responses:
                    merged_responses.append(response)
            state["visible_response"] = merged_responses
        completion = str(state.pop("_handler_completion", "unchanged"))
        rejected = bool(state.pop("_handler_rejected", False))
        changed = True
        executed += 1
        if rejected:
            break
        state.setdefault("completed_actions", []).append(action)
        state["completed_actions"] = state["completed_actions"][-20:]
        if action_id:
            applied = list(state.get("applied_action_ids") or [])
            if action_id not in applied:
                applied.append(action_id)
            state["applied_action_ids"] = applied[-200:]
        if (
            not state.get("pending_question")
            and state.get("action_queue")
            and str((state["action_queue"][0] or {}).get("type") or "") == "answer_pending"
        ):
            active_group = str(state.get("active_group") or "")
            next_question = _question_for_group(state, active_group) if active_group else None
            if next_question:
                _install_pending_question(state, next_question)
                state["pending_question"]["resume_action_queue"] = True
        if completion == "in_progress":
            active_group = str(state.get("active_group") or "")
            if not _queue_has_followup_for_group(state, active_group):
                question = _question_for_group(state, active_group)
                if question:
                    _install_pending_question(state, question)
                    if state.get("action_queue"):
                        state["pending_question"]["resume_action_queue"] = True
                    if (
                        state.get("action_queue")
                        and not _queue_can_continue_through_pending(state)
                        and not _next_queue_action_crosses_pending_barrier(state)
                        and not _pending_is_queue_barrier(state)
                    ):
                        _suspend_pending_for_remaining_queue(state)
                    elif not _queue_can_continue_through_pending(state) and not _next_queue_action_crosses_pending_barrier(state):
                        state["visible_response"] = list(state.get("visible_response") or []) + [
                            _render_question(question, state.get("language", "en"))
                        ]
                        break
        after_pending_id = str((state.get("pending_question") or {}).get("id") or "")
        pending_changed = bool(after_pending_id and after_pending_id != before_pending_id)
        if pending_changed:
            state["pending_question"]["resume_action_queue"] = True
            if _queue_can_continue_through_pending(state) or _next_queue_action_crosses_pending_barrier(state):
                # A domain may establish the next pending contract, but the
                # coordinator owns turn-level presentation.  An action that
                # consumes this pending contract, or a consultation queued in
                # the same turn, runs before the question is rendered.
                state["visible_response"] = before_responses
            elif state.get("action_queue") and not _pending_is_queue_barrier(state):
                state["visible_response"] = before_responses
                _suspend_pending_for_remaining_queue(state)
        if state.pop("_queue_pause", False):
            if state.get("pending_question") and state.get("action_queue"):
                state["pending_question"]["resume_action_queue"] = True
            break
        if state.get("_stop_after_response"):
            if action_spec and action_spec.allows_followup_actions and state.get("action_queue"):
                # A consultation/report answer completes its own work but must
                # not discard another validated request from the same user
                # turn. The registry, rather than individual domains, declares
                # which read-only actions are safe to compose this way.
                state.pop("_stop_after_response", None)
            else:
                break
        if (
            state.get("pending_question")
            and state.get("action_queue")
            and _pending_is_queue_barrier(state)
            and not _queue_can_continue_through_pending(state)
            and not _next_queue_action_crosses_pending_barrier(state)
        ):
            # Read-only consultations may overlay a blocking question. Once
            # those consultations finish, unrelated mutations remain queued
            # until the user resolves the active typed contract.
            if _next_action_was_submitted_after_pending(state):
                _suspend_pending_for_remaining_queue(state)
            else:
                state["pending_question"]["resume_action_queue"] = True
                _append_active_question_once(state)
                break
        if (
            pending_changed
            and state.get("pending_question")
            and not _queue_can_continue_through_pending(state)
            and not _next_queue_action_crosses_pending_barrier(state)
        ):
            break
        if max_actions is not None and executed >= max_actions:
            break
    state["current_action"] = {}
    return state if changed else None


def _append_active_question_once(state: AgentGraphState) -> None:
    pending = state.get("pending_question") or {}
    if not pending:
        return
    rendered = _render_question(pending, state.get("language", "en"))
    responses = list(state.get("visible_response") or [])
    already_actionable = _question_is_actionable_in_responses(responses, pending)
    if rendered not in responses and not already_actionable:
        responses.append(rendered)
    state["visible_response"] = responses


def _question_is_actionable_in_responses(
    responses: list[str],
    pending: PendingQuestion,
) -> bool:
    """Recognize one rendered question across equivalent presenter formats."""

    prompt = str(pending.get("prompt") or "").strip()
    if not prompt:
        return False
    option_labels = [
        str(option.get("label") or "").strip()
        for option in pending.get("options") or []
        if str(option.get("label") or "").strip()
    ]
    return any(
        prompt in response
        and (not option_labels or all(label in response for label in option_labels))
        for response in responses
    )


def _finalize_turn_response(state: AgentGraphState) -> AgentGraphState:
    """Apply one response-composition policy after every coordinator path."""

    responses: list[str] = []
    seen: set[str] = set()
    for item in state.get("visible_response") or []:
        rendered = str(item or "").strip()
        if not rendered or rendered in seen:
            continue
        seen.add(rendered)
        responses.append(rendered)

    turn_context = dict(state.get("turn_context") or {})
    pending = state.get("pending_question") or {}
    active_rendered = _render_question(pending, state.get("language", "en")).strip() if pending else ""
    for installed in turn_context.get("installed_questions") or []:
        if not isinstance(installed, dict):
            continue
        installed_rendered = _render_question(installed, state.get("language", "en")).strip()
        if not installed_rendered or installed_rendered == active_rendered:
            continue
        responses = _without_superseded_question(
            responses,
            installed,
            state.get("language", "en"),
        )
    suppress_pending_render = bool(turn_context.pop("suppress_pending_render", False))
    state["turn_context"] = turn_context
    if pending and not suppress_pending_render:
        actionable = [
            index
            for index, response in enumerate(responses)
            if _question_is_actionable_in_responses([response], pending)
        ]
        if not actionable:
            responses.append(_render_question(pending, state.get("language", "en")))
        elif len(actionable) > 1:
            keep = actionable[0]
            responses = [
                response
                for index, response in enumerate(responses)
                if index == keep or index not in actionable
            ]
    state["visible_response"] = responses
    validate_state(state)
    return state


def _next_action_was_submitted_after_pending(state: AgentGraphState) -> bool:
    queue = state.get("action_queue") or []
    pending = state.get("pending_question") or {}
    if not queue or not pending:
        return False
    submitted = int((queue[0] or {}).get("_submitted_turn_index") or 0)
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

    existing = _durable_actions(existing)
    incoming = _durable_actions(incoming)
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
        if isinstance(item, dict) and not action_is_turn_local(item)
    ]


def _queue_has_admitted_durable_work(state: AgentGraphState) -> bool:
    """Distinguish admitted commands from unannotated legacy stale entries."""

    return any(
        not action_is_turn_local(item)
        and ("_plan_scope" in item or "_submitted_turn_index" in item)
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
        if str((action or {}).get("type") or "").strip() not in superseded
    ]


def _queue_can_continue_through_pending(state: AgentGraphState) -> bool:
    """Return whether the next typed action can consume this pending contract."""

    pending_group = str((state.get("pending_question") or {}).get("group") or "").strip()
    queue = state.get("action_queue") or []
    if not pending_group or not queue:
        return not pending_group
    next_action_type = str((queue[0] or {}).get("type") or "").strip()
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
    action = dict(queue[0] or {})
    if not action_crosses_pending_barrier(action):
        return False
    pending = state.get("pending_question") or {}
    if not pending:
        return True
    # Navigation may detour from a question that was already visible when the
    # user submitted this turn. It must not bypass an approval or validation
    # barrier produced by an earlier action in the same transaction.
    submitted_turn = int(action.get("_submitted_turn_index") or 0)
    created_turn = int(pending.get("created_turn_index") or 0)
    return bool(submitted_turn and created_turn < submitted_turn)


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
    if not frame["group"] or not frame["question_id"]:
        return
    stack = list(state.get("interruption_stack") or [])
    if not stack or stack[-1] != frame:
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


def _order_action_queue(
    state: AgentGraphState,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Topologically order typed actions without rewriting user intent order.

    The intent planner returns an ordered plan. Dependencies declared by action
    and question contracts may move a prerequisite before its consumer, but a
    static registry phase must not otherwise reverse explicit user sequencing.
    For legacy/deferred actions that predate plan metadata, current queue order
    is the stable fallback. Dependencies are never inferred from prose.
    """

    requirements = [_action_requirements(state, action) for action in actions]
    providers = [_action_provisions(action) for action in actions]
    incoming: list[set[int]] = [set() for _ in actions]
    outgoing: list[set[int]] = [set() for _ in actions]
    for consumer_index, required in enumerate(requirements):
        missing = {item for item in required if not _state_has_capability(state, item)}
        if not missing:
            continue
        for provider_index, provided in enumerate(providers):
            if provider_index == consumer_index or not (missing & provided):
                continue
            incoming[consumer_index].add(provider_index)
            outgoing[provider_index].add(consumer_index)

    # Structured configuration is an untrusted proposal that must cross its
    # review barrier before independent mutations from the same turn execute.
    # This is a typed transaction dependency, not a numeric phase convention.
    proposal_indexes = [
        index
        for index, action in enumerate(actions)
        if str(action.get("type") or "") == "propose_config_values"
    ]
    proposal_blocked_types = {
        "rpc_catalog_command",
        "rpc_workload_command",
        "set_rpc_mode",
        "use_default_workload",
        "configure_workload_weights",
        "set_qps_mode",
        "request_qps_customization",
        "set_qps_override",
        "set_observability",
        "set_sync_observe_source",
        "clear_sync_observe_source",
        "set_sync_observe_options",
        "approve_preflight_smoke",
        "approve_final_benchmark",
    }
    for proposal_index in proposal_indexes:
        for consumer_index, action in enumerate(actions):
            if consumer_index == proposal_index:
                continue
            if str(action.get("type") or "") not in proposal_blocked_types:
                continue
            incoming[consumer_index].add(proposal_index)
            outgoing[proposal_index].add(consumer_index)

    ready = [index for index, dependencies in enumerate(incoming) if not dependencies]
    ordered: list[dict[str, Any]] = []
    while ready:
        ready.sort(key=lambda index: _action_plan_order(actions[index], index))
        current = ready.pop(0)
        ordered.append(actions[current])
        for consumer in sorted(outgoing[current]):
            incoming[consumer].discard(current)
            if not incoming[consumer] and consumer not in ready:
                ready.append(consumer)
    if len(ordered) != len(actions):
        raise RuntimeError("Harness action dependency cycle")
    return ordered


def _action_plan_order(action: dict[str, Any], queue_index: int) -> tuple[int, int, int]:
    """Return a durable plan-first order for one ready action.

    Newer turns are merged before retained deferred work, and each turn's model
    order is persisted as metadata. Registry phase remains only a legacy tie
    breaker for checkpoints created before ordered plans existed.
    """

    plan_index = action.get("_plan_index")
    if isinstance(plan_index, int) and plan_index >= 0:
        return (0, plan_index, queue_index)
    return (1, action_execution_phase(action), queue_index)


def _action_requirements(state: AgentGraphState, action: dict[str, Any]) -> set[str]:
    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    required = set(spec.requires_capabilities if spec else ())
    if str(action.get("type") or "") == "answer_pending":
        required.update(
            str(item).strip()
            for item in (state.get("pending_question") or {}).get("requires_capabilities") or []
            if str(item).strip()
        )
    return required


def _action_provisions(action: dict[str, Any]) -> set[str]:
    spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
    return set(spec.provides_capabilities if spec else ())


def _state_has_capability(state: AgentGraphState, capability: str) -> bool:
    if capability == "chain_identity":
        identity = state.get("chain_identity") or {}
        return bool(str(identity.get("canonical") or identity.get("raw") or "").strip())
    if capability == "target_mode":
        return bool(normalize_target_mode(state.get("target_mode")))
    return False


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
    if result.blocker:
        candidate: AgentGraphState = deepcopy(state)
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
        candidate["_stop_after_response"] = True
        candidate["_handler_completion"] = "blocked"
        candidate["_handler_rejected"] = True
        validate_state(candidate)
        return candidate

    if result.checkpoint_command is not None:
        candidate = _apply_checkpoint_command(state, result.checkpoint_command)
    elif owner == "coordinator" and control_state is not None:
        candidate = deepcopy(control_state)
    else:
        candidate = deepcopy(state)
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
        followups = assign_action_ids(
            str(candidate.get("thread_id") or "default"),
            origin_text,
            [dict(item) for item in result.followup_actions],
        )
        for action in followups:
            action.setdefault("_origin_text", origin_text)
            action.setdefault("_queue_origin_group", str(candidate.get("active_group") or ""))
            action.setdefault("_submitted_turn_index", int(candidate.get("turn_index") or 0))
        candidate["action_queue"] = _order_action_queue(
            candidate,
            [*followups, *[dict(item) for item in candidate.get("action_queue") or []]],
        )

    candidate = _activate_ready_deferred_group(candidate)

    if result.stop_after_response and not candidate.get("action_queue"):
        candidate["_stop_after_response"] = True
    candidate["_handler_completion"] = result.completion
    validate_state(candidate)
    return candidate


def _activate_ready_deferred_group(state: AgentGraphState) -> AgentGraphState:
    """Resume an explicit group detour before registry fallback can take over."""

    if state.get("pending_question"):
        return state
    control = dict(state.get("control") or {})
    deferred_group = str(control.get("deferred_group") or "").strip()
    if not deferred_group or deferred_group not in ALLOWED_GROUPS:
        return state
    if deferred_group == "workload_rpc" and not chain_identity_confirmed(state):
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
    candidate["audit_events"] = list(state.get("audit_events") or []) + [{"event": "workflow_reset"}]
    if command.command == "retain_safe":
        candidate["confirmed_config"] = deepcopy(dict(command.confirmed_config))
    return candidate


def _without_superseded_question(
    responses: list[str],
    previous_pending: PendingQuestion,
    language: str,
) -> list[str]:
    """Remove only a superseded pending question from the response composition.

    A single user turn may legitimately apply several typed actions. Each
    handler can create a newer blocking question, but the product terminal must
    never display every intermediate question. Keeping non-question results
    while replacing the rendered previous question preserves useful evidence
    and enforces the one-active-question contract.
    """

    if not previous_pending:
        return responses
    candidates = {
        str(previous_pending.get("prompt") or "").strip(),
        _render_question(previous_pending, language).strip(),
    }
    candidates.discard("")
    return [response for response in responses if str(response).strip() not in candidates]


def _apply_evidence_outcome(
    state: AgentGraphState,
    outcome: EvidenceCollectionOutcome,
) -> AgentGraphState:
    state = _apply_handler_result(state, outcome.result, owner="analysis")
    if outcome.disposition == "pending_answer":
        return _apply_pending_answer(
            state,
            outcome.collected_text,
            dict(outcome.pending_question),
        )
    return state


def _apply_queue_action(state: AgentGraphState, action: dict[str, Any], text: str) -> AgentGraphState | None:
    action = validate_action_contract(action, trusted_metadata=True)
    action_type = str(action.get("type") or "unknown").strip()
    confidence = str(action.get("confidence") or "medium").strip().lower()
    origin_text = str(action.get("_origin_text") or text)
    if confidence == "low" and action_type not in {"greeting", "unknown"}:
        state.setdefault("action_errors", []).append({"action": action, "error": "low_confidence"})
        return None
    spec = ACTION_BY_TYPE.get(action_type)
    owner = spec.owner if spec else ""
    if action_type == "answer_pending":
        return _dispatch_pending_action(state, action)
    if action_type == "append_evidence_collection":
        outcome = continue_evidence_collection(
            state,
            str(action.get("evidence") or origin_text),
            state.get("evidence_collection") or {},
        )
        return _apply_evidence_outcome(state, outcome)
    if action_type == "finish_evidence_collection":
        return _apply_evidence_outcome(
            state,
            finish_evidence_collection(state, state.get("evidence_collection") or {}),
        )
    if action_type == "pause_evidence_collection":
        return _apply_handler_result(state, pause_evidence_collection(state), owner="analysis")
    if action_type == "resume_evidence_collection":
        return _apply_handler_result(state, resume_evidence_collection(state), owner="analysis")
    if action_type == "cancel_evidence_collection":
        return _apply_handler_result(state, cancel_evidence_collection(state), owner="analysis")
    runtime = COORDINATOR_RUNTIME if owner == "coordinator" else DOMAIN_RUNTIME.get(owner)
    if runtime is not None:
        domain_action = dict(action)
        if action_type == "propose_config_values":
            domain_action["source_text"] = origin_text
        if spec and spec.owner == "chain_rpc":
            domain_action["origin_text"] = origin_text
        if spec and spec.owner == "coordinator":
            domain_action["queue_origin_group"] = action.get("_queue_origin_group")
        if action_type == "analyze_report" and not domain_action.get("job_id"):
            requested_job = JOB_ID_RE.search(origin_text)
            if requested_job:
                domain_action["job_id"] = requested_job.group(0)
        dispatch_state = state
        cancellation_resume_group = ""
        if owner == "coordinator" and action_type in {"change_group", "go_back"}:
            # Navigation may abandon a domain-owned transient workflow. Commit
            # that owner's cancellation delta first, then let the coordinator
            # mutate only control-plane state. This keeps ownership explicit
            # and prevents a locally rebound cancellation snapshot from being
            # discarded when the coordinator result is committed.
            target_group = str(domain_action.get("group") or "").strip()
            interrupted_pending = deepcopy(state.get("pending_question") or {})
            pending_group = str(interrupted_pending.get("group") or "").strip()
            same_group_navigation = action_type == "change_group" and target_group == pending_group
            if not same_group_navigation:
                dispatch_state, cancellation_resume_group = _cancel_transient_question(state)
                if (
                    action_type == "change_group"
                    and interrupted_pending
                    and _reconstruct_question(dispatch_state, interrupted_pending) is not None
                ):
                    _push_interruption_frame(
                        dispatch_state,
                        interrupted_pending,
                        reason="explicit_navigation",
                    )
            if cancellation_resume_group:
                domain_action["cancellation_resume_group"] = cancellation_resume_group
        handler_state = deepcopy(dispatch_state)
        result = runtime.apply_action(handler_state, _action_proposal(domain_action, confidence))
        return _apply_handler_result(
            dispatch_state,
            result,
            owner=owner,
            control_state=handler_state if owner == "coordinator" else None,
        )
    return None


def _dispatch_pending_action(state: AgentGraphState, action: dict[str, Any]) -> AgentGraphState:
    pending = dict(state.get("pending_question") or {})
    if not pending:
        return _apply_handler_result(
            state,
            HandlerResult(blocker="no pending question is available for this answer"),
            owner="coordinator",
        )
    answer = str(action.get("answer") or action.get("_origin_text") or "").strip()
    selected = action.get("selected_value")
    if isinstance(selected, str) and not selected.strip():
        selected = None
    choice_question = str(pending.get("kind") or "") in {"numbered_choice", "yes_no"}
    interpreted = str(selected) if selected is not None else answer
    if not choice_question and not interpreted:
        return _apply_handler_result(
            state,
            HandlerResult(blocker="the pending answer is empty"),
            owner="coordinator",
        )
    if not choice_question and not (
        _pending_option_value_exists(selected, pending)
        or _answer_fits_pending(interpreted, pending)
    ):
        return _apply_handler_result(
            state,
            HandlerResult(
                blocker=_localized(
                    state.get("language", "en"),
                    "模型给出的结构化值不符合当前字段契约，当前问题保持不变。",
                    "The model-derived value does not satisfy the current field contract. The question remains active.",
                )
            ),
            owner="coordinator",
        )
    manual_choice_value = bool(
        choice_question
        and pending.get("manual_input_allowed") is True
        and selected is None
        and _answer_fits_pending(answer, pending)
    )
    if choice_question and not _pending_option_value_exists(selected, pending) and not manual_choice_value:
        return _apply_handler_result(
            state,
            HandlerResult(
                blocker=_localized(
                    state.get("language", "en"),
                    "模型没有把这段回复映射到一个已声明选项，当前问题保持不变。请换一种说法，或回复显示的选项。",
                    "The model did not map that reply to a declared option. The question remains active; rephrase or use a displayed option.",
                )
            ),
            owner="coordinator",
        )
    return _apply_pending_answer(state, answer if manual_choice_value else interpreted, pending)


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
    action_types = {str(item.get("type") or "").strip() for item in state.get("action_queue") or [] if isinstance(item, dict)}
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
        if deferred_group != "workload_rpc" or chain_identity_confirmed(state):
            control.pop("deferred_group", None)
            state["control"] = control
            return _activate_group_question(state, deferred_group)
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


def _activate_group_question(state: AgentGraphState, group: str, *, record_history: bool = True) -> AgentGraphState:
    if group == "sync_observe":
        current_mode = str(state.get("target_mode") or "").strip()
        if current_mode != "sync-observe":
            action = ActionProposal(
                action_id=f"{state.get('thread_id') or 'default'}:{state.get('turn_index') or 0}:sync-observe-jump",
                action_type="choose_target_mode",
                arguments={"target_mode": "sync-observe", "target_mode_explicit": True, "selection_contract_verified": True},
                confidence="high",
            )
            updated = _apply_handler_result(
                state,
                apply_chain_rpc_action(state, action),
                owner="chain_rpc",
            )
            # Coordinator actions execute against an isolated control-state
            # snapshot. Keep that snapshot identity stable when a domain owner
            # returns a new committed value; otherwise the caller retains the
            # pre-dispatch object and silently loses the dependency-change
            # confirmation contract.
            state.clear()
            state.update(updated)
            if state.get("pending_question"):
                return state
    _record_group_transition(state, group, record_history=record_history)
    state.setdefault("group_states", {}).setdefault(group, {})["status"] = "in_progress"
    # An explicit navigation to chain identity means "choose or change the
    # chain" even when the current chain is already confirmed. Keep the old
    # value until the user's replacement is confirmed; the chain owner then
    # applies dependency invalidation atomically.
    question = _question_for_group(state, group)
    if question:
        _install_pending_question(state, question)
        state["visible_response"] = [_render_question(question, state.get("language", "en"))]
    else:
        status = completed_group_status(state, group)
        if status:
            state["pending_question"] = {}
            state["visible_response"] = [status]
            state["_stop_after_response"] = True
            return state
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            "这个配置组当前没有阻塞项。我会继续寻找下一项必须确认的配置。",
            "This configuration group has no blocking item right now. I will continue to the next required configuration item.",
        )]
        state = _ask_next_blocking_question(state)
        return state
    state["_stop_after_response"] = True
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


def _cancel_transient_question(state: AgentGraphState) -> tuple[AgentGraphState, str]:
    pending = state.get("pending_question") or {}
    owner = GROUP_OWNER.get(str(pending.get("group") or ""), "")
    runtime = DOMAIN_RUNTIME.get(owner)
    if runtime is None or runtime.cancel_question is None:
        return state, ""
    result = runtime.cancel_question(deepcopy(state), pending)
    resume_group = str(result.navigation_resume_group or "").strip()
    candidate = _apply_handler_result(
        state,
        replace(result, followup_actions=()),
        owner=owner,
    )
    for raw_followup in result.followup_actions:
        followup = validate_action_contract(dict(raw_followup))
        spec = ACTION_BY_TYPE[str(followup["type"])]
        followup_runtime = DOMAIN_RUNTIME.get(spec.owner)
        if followup_runtime is None:
            raise StateInvariantError(
                f"cancel follow-up has no domain runtime: {followup['type']}"
            )
        followup_result = followup_runtime.apply_action(
            deepcopy(candidate),
            _action_proposal(followup, "high"),
        )
        candidate = _apply_handler_result(
            candidate,
            followup_result,
            owner=spec.owner,
        )
    return candidate, resume_group


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
    question = _question_for_group(state, group)
    if question and str(question.get("id") or "") == question_id:
        return question
    return None

def _apply_pending_answer(state: AgentGraphState, text: str, question: PendingQuestion) -> AgentGraphState:
    group = str(question.get("group") or "")
    question_id = str(question.get("id") or "")
    if question.get("contract_version") == 1:
        matched, value = contract_exact_answer(text, question)
        if not matched:
            return state
    else:
        value = _coerce_answer(text, question)
    declared_action = action_for_value(question, value)
    if declared_action and str(declared_action.get("type") or "") != "answer_pending":
        return_policy = _option_return_policy(question, value)
        action_spec = ACTION_BY_TYPE.get(str(declared_action.get("type") or ""))
        runtime_arguments: dict[str, Any] = {}
        if (
            action_spec is not None
            and "source_evidence" in action_spec.allowed_arguments
            and not str(declared_action.get("source_evidence") or "").strip()
        ):
            # The pending-question contract owns option interpretation. Once
            # it has matched an exact or model-selected declared option, the
            # original answer is trusted provenance for the declared action;
            # individual domains must not reconstruct that provenance.
            runtime_arguments["source_evidence"] = str(text or "").strip()
        action = {
            **declared_action,
            **runtime_arguments,
            "action_id": f"{state.get('thread_id') or 'default'}:{int(state.get('turn_index') or 0)}:{question_id}:{value}",
            "confidence": "high",
            "selection_contract_verified": True,
            "_origin_text": text,
        }
        dispatch_state = state
        if return_policy == "stop_after_response":
            dispatch_state = deepcopy(state)
            dispatch_state["pending_question"] = {}
        result = _apply_queue_action(dispatch_state, action, text)
        if result is None:
            raise RuntimeError(f"option action was not executable: {question_id}/{value}")
        expected = expected_patch_for_value(question, value)
        if expected:
            verify_expected_patch(result, expected)
        result = _apply_option_return_policy(result, question, value)
        _record_admitted_action(result, action, source="pending_question_contract")
        return result
    if question_id == "inferred_config_review":
        state["action_queue"] = [
            item
            for item in state.get("action_queue") or []
            if str(item.get("type") or "") != "propose_config_values"
        ]
        result = _apply_handler_result(state, apply_inferred_config_review(state, bool(value)), owner="environment")
        _record_admitted_action(
            result,
            {
                "type": "answer_pending",
                "action_id": f"{state.get('thread_id') or 'default'}:{int(state.get('turn_index') or 0)}:{question_id}",
            },
            source="pending_question_contract",
        )
        return result
    runtime = DOMAIN_RUNTIME.get(GROUP_OWNER.get(group, ""))
    if runtime is None or runtime.apply_answer is None:
        raise RuntimeError(f"no answer handler owns question: {group}/{question_id}")
    expected = expected_patch_for_value(question, value)
    result = _apply_handler_result(
        state,
        runtime.apply_answer(deepcopy(state), question, value, text),
        owner=GROUP_OWNER.get(group, ""),
    )
    if expected and (GROUP_OWNER.get(group) == "execution" or not result.get("pending_question")):
        verify_expected_patch(result, expected)
    _record_admitted_action(
        result,
        {
            "type": "answer_pending",
            "action_id": f"{state.get('thread_id') or 'default'}:{int(state.get('turn_index') or 0)}:{question_id}",
        },
        source="pending_question_contract",
    )
    return result


def _option_return_policy(question: PendingQuestion, value: Any) -> str:
    selected = next(
        (option for option in question.get("options") or [] if option.get("value") == value),
        {},
    )
    return str(selected.get("return_policy") or "fallback")


def _apply_option_return_policy(
    state: AgentGraphState,
    question: PendingQuestion,
    value: Any,
) -> AgentGraphState:
    """Apply the selected option's declared control-plane return policy."""

    policy = _option_return_policy(question, value)
    if policy == "stay":
        state["pending_question"] = deepcopy(question)
        state["active_group"] = str(question.get("group") or state.get("active_group") or "")
        state["_stop_after_response"] = True
    if policy == "stop_after_response":
        state["_stop_after_response"] = True
    return state


def _looks_like_assignment_answer(text: str) -> bool:
    raw = str(text or "").strip().rstrip(",，;；、").strip()
    if not raw or "\n" in raw:
        return False
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return False
    return all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:/-]*\s*=\s*[^=,]+", part) for part in parts)


def _normalized_target_mode(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "fake": "fake-node",
        "fake-node": "fake-node",
        "fakenode": "fake-node",
        "real": "real-node",
        "real-node": "real-node",
        "realnode": "real-node",
        "sync": "sync-observe",
        "sync-observe": "sync-observe",
        "syncobserve": "sync-observe",
    }
    return aliases.get(text, "")


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
