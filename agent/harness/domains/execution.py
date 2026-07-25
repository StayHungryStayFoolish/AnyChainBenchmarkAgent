"""Execution domain for explicit preflight/smoke approval and job handoff."""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import replace
from functools import partial
from typing import Any

from agent.runners.job_manager import get_job, verify_job_receipt

from ..contracts import ActionProposal, HandlerResult, RecoveryCommand, ResponseFragment, StateDelta
from .execution_runtime import execute_approved_final_benchmark, execute_approved_preflight_and_smoke
from ..questions import choice_question as _choice_question, question_text
from ..routing import next_group_and_reason
from ..state import AgentGraphState
from .response_fragments import failure

choice_question = partial(_choice_question, owner="execution")


def _fragment(message_id: str) -> ResponseFragment:
    return ResponseFragment(kind="message", message_id=message_id, source=__name__)


EXECUTION_GROUPS = {"preflight_smoke_execution", "job_monitoring"}


def question_for_execution(state: AgentGraphState, group: str) -> dict[str, Any] | None:
    if group == "job_monitoring" and _needs_real_node_smoke(state):
        question = choice_question(
            group,
            "real_node_smoke_confirm",
            question_text("question.execution.real_node_smoke.prompt"),
            field="real_node_smoke_confirmed",
            kind="yes_no",
            options=[
                {
                    "label": question_text("question.control.option.yes"),
                    "value": True,
                    "action": {"type": "approve_preflight_smoke"},
                    "expected_patch": {"preflight.approved": True},
                    "completion_effect": question_text(
                        "question.execution.real_node_smoke.approve.completion"
                    ),
                },
                {
                    "label": question_text("question.control.option.no"),
                    "value": False,
                    "action": {"type": "reject_preflight_smoke"},
                    "expected_patch": {"preflight.approved": False},
                    "completion_effect": question_text(
                        "question.execution.real_node_smoke.reject.completion"
                    ),
                },
            ],
            queue_barrier=True,
        )
        question["execution_request_id"] = str((state.get("preflight") or {}).get("execution_request_id") or uuid.uuid4().hex)
        return question
    if group == "job_monitoring" and _ready_for_final_benchmark(state):
        return choice_question(
            group,
            "real_node_final_benchmark_confirm",
            question_text("question.execution.final_benchmark.prompt"),
            field="final_benchmark_confirmed",
            kind="yes_no",
            options=[
                {
                    "label": question_text("question.control.option.yes"),
                    "value": True,
                    "action": {"type": "approve_final_benchmark"},
                    "expected_patch": {"final_benchmark.approved": True},
                    "completion_effect": question_text(
                        "question.execution.final_benchmark.approve.completion"
                    ),
                },
                {
                    "label": question_text("question.control.option.no"),
                    "value": False,
                    "action": {"type": "reject_final_benchmark"},
                    "expected_patch": {"final_benchmark.approved": False},
                    "completion_effect": question_text(
                        "question.execution.final_benchmark.reject.completion"
                    ),
                },
            ],
            queue_barrier=True,
        )
    if group != "preflight_smoke_execution":
        return None
    next_group, _reason = next_group_and_reason(state)
    if next_group != "preflight_smoke_execution":
        return None
    question = choice_question(
        group,
        "preflight_smoke_confirm",
        question_text("question.execution.preflight_smoke.prompt"),
        field="preflight_smoke_confirmed",
        kind="yes_no",
        options=[
            {
                "label": question_text("question.control.option.yes"),
                "value": True,
                "action": {"type": "approve_preflight_smoke"},
                "expected_patch": {"preflight.approved": True},
            },
            {
                "label": question_text("question.control.option.no"),
                "value": False,
                "action": {"type": "reject_preflight_smoke"},
                "expected_patch": {"preflight.approved": False},
            },
        ],
        queue_barrier=True,
    )
    question["execution_request_id"] = uuid.uuid4().hex
    return question


def apply_execution_answer(
    state: AgentGraphState,
    value: Any,
    question: dict[str, Any] | None = None,
) -> HandlerResult:
    next_state: AgentGraphState = deepcopy(state)
    active_question = dict(question or next_state.get("pending_question") or {})
    question_id = str(active_question.get("id") or "")
    if question_id == "real_node_final_benchmark_confirm":
        return _apply_final_benchmark_answer(next_state, bool(value))
    request_id = str(active_question.get("execution_request_id") or "").strip()
    approved = bool(value)
    next_state.setdefault("preflight", {})["approved"] = approved
    if request_id:
        next_state["preflight"]["execution_request_id"] = request_id
    if approved:
        runtime_result = execute_approved_preflight_and_smoke(next_state)
        return replace(
            runtime_result,
            delta=_merge_deltas(StateDelta.between(state, next_state), runtime_result.delta),
            clear_pending=True,
        )
    return HandlerResult(
        delta=StateDelta.between(state, next_state),
        response_fragments=(_fragment("execution.response.preflight_paused"),),
        clear_pending=True,
        completion="blocked",
        stop_after_response=True,
    )


def apply_execution_action(state: AgentGraphState, action: ActionProposal) -> HandlerResult:
    if action.action_type == "approve_preflight_smoke":
        result = apply_execution_answer(state, True)
    elif action.action_type == "reject_preflight_smoke":
        result = apply_execution_answer(state, False)
    elif action.action_type == "approve_final_benchmark":
        result = _apply_final_benchmark_answer(deepcopy(state), True)
    elif action.action_type == "reject_final_benchmark":
        result = _apply_final_benchmark_answer(deepcopy(state), False)
    else:
        return HandlerResult(blocker=failure(
            "execution.failure.unsupported_action",
            arguments={"action_type": action.action_type},
            source=__name__,
        ))
    return replace(result, consumed_action_ids=(action.action_id,))


def reconcile_execution_state(state: AgentGraphState) -> HandlerResult:
    """Refresh the workflow-owned job and advance only proven execution phases."""

    next_state: AgentGraphState = deepcopy(state)
    current_job = dict(next_state.get("job") or {})
    job_id = str(current_job.get("job_id") or "").strip()
    if not job_id:
        return HandlerResult()
    try:
        persisted = get_job(job_id)
    except (FileNotFoundError, OSError, ValueError):
        return HandlerResult()
    receipts = persisted.get("execution_receipts") or {}
    read_receipt = (
        dict(receipts.get("last_read") or {})
        if isinstance(receipts, dict)
        else {}
    )
    if (
        not verify_job_receipt(read_receipt)
        or read_receipt.get("job_id") != job_id
        or read_receipt.get("observed_status") != persisted.get("status")
    ):
        return HandlerResult(blocker=failure(
            "execution.failure.invalid_job_receipt",
            source=__name__,
        ))
    next_state["job"] = persisted
    status = str(persisted.get("status") or "unknown")

    smoke = next_state.get("smoke") or {}
    if smoke.get("purpose") == "real_node_isolated_smoke" and smoke.get("job_id") == job_id:
        smoke["status"] = status
        smoke["job"] = persisted
        if status == "completed":
            pending = None
            if not next_state.get("pending_question"):
                pending = question_for_execution(next_state, "job_monitoring")
            return HandlerResult(
                delta=StateDelta.between(state, next_state),
                pending_question=pending,
                next_group="job_monitoring",
                completion="completed",
            )
        elif status in {"failed", "partial"}:
            recovery_command = _job_failure_command(next_state, persisted)
            return HandlerResult(
                delta=StateDelta.between(state, next_state),
                recovery_command=recovery_command,
                completion="blocked",
            )
        return HandlerResult(delta=StateDelta.between(state, next_state))

    final = next_state.get("final_benchmark") or {}
    if final.get("job_id") == job_id:
        final["status"] = status
        final["job"] = persisted
    if status in {"failed", "partial"}:
        recovery_command = _job_failure_command(next_state, persisted)
        return HandlerResult(
            delta=StateDelta.between(state, next_state),
            recovery_command=recovery_command,
            completion="blocked",
        )
    return HandlerResult(delta=StateDelta.between(state, next_state))


def _ready_for_final_benchmark(state: AgentGraphState) -> bool:
    smoke = state.get("smoke") or {}
    final = state.get("final_benchmark") or {}
    return (
        state.get("target_mode") == "real-node"
        and smoke.get("purpose") == "real_node_isolated_smoke"
        and smoke.get("status") == "completed"
        and not final.get("job_id")
        and final.get("decision") != "declined"
    )


def _needs_real_node_smoke(state: AgentGraphState) -> bool:
    return (
        state.get("target_mode") == "real-node"
        and bool((state.get("preflight") or {}).get("approved"))
        and (state.get("preflight") or {}).get("status") == "passed"
        and not (state.get("smoke") or {}).get("job_id")
    )


def _apply_final_benchmark_answer(state: AgentGraphState, approved: bool) -> HandlerResult:
    original = deepcopy(state)
    final = state.setdefault("final_benchmark", {})
    final["approved"] = approved
    if not approved:
        final["decision"] = "declined"
        return HandlerResult(
            delta=StateDelta.between(original, state),
            response_fragments=(_fragment("execution.response.final_paused"),),
            clear_pending=True,
            completion="blocked",
            stop_after_response=True,
        )
    if not _ready_for_final_benchmark(state):
        return HandlerResult(blocker=failure(
            "execution.failure.real_smoke_required",
            source=__name__,
        ))
    runtime_result = execute_approved_final_benchmark(state)
    return replace(
        runtime_result,
        delta=_merge_deltas(StateDelta.between(original, state), runtime_result.delta),
        clear_pending=True,
    )


def _job_failure_command(state: AgentGraphState, job: dict[str, Any]) -> RecoveryCommand | None:
    from ..failures import failure_record_from_job

    record = failure_record_from_job(job, confirmed_config=state.get("confirmed_config") or {})
    recovery = dict(state.get("failure_recovery") or {})
    previous = dict(recovery.get("record") or {})
    if previous.get("failure_id") == record.get("failure_id") and recovery.get("status") in {
        "pending", "correcting", "resolved", "cancelled",
    }:
        return None
    return RecoveryCommand(operation="activate", record=record)


def _merge_deltas(*deltas: StateDelta) -> StateDelta:
    return StateDelta(
        writes=tuple(write for delta in deltas for write in delta.writes),
        deletes=tuple(path for delta in deltas for path in delta.deletes),
    )
