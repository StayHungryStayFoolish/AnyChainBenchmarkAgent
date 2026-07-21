"""Execution domain for explicit preflight/smoke approval and job handoff."""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import replace
from typing import Any

from ..contracts import ActionProposal, HandlerResult, RecoveryCommand, StateDelta
from ..localization import localized
from .execution_runtime import execute_approved_final_benchmark, execute_approved_preflight_and_smoke
from ..questions import choice_question
from ..routing import next_group_and_reason
from ..state import AgentGraphState


EXECUTION_GROUPS = {"preflight_smoke_execution", "job_monitoring"}


def question_for_execution(state: AgentGraphState, group: str) -> dict[str, Any] | None:
    if group == "job_monitoring" and _needs_real_node_smoke(state):
        language = str(state.get("language") or "en")
        question = choice_question(
            group,
            "real_node_smoke_confirm",
            localized(
                language,
                "preflight 已通过，但还没有执行隔离的 real-node smoke。现在重新校验并提交安全小流量 smoke？",
                "Preflight passed, but no isolated real-node smoke has run. Revalidate and submit the safe low-traffic smoke now?",
            ),
            field="real_node_smoke_confirmed",
            kind="yes_no",
            options=[
                {
                    "label": "Y",
                    "value": True,
                    "action": {"type": "approve_preflight_smoke"},
                    "expected_patch": {"preflight.approved": True},
                    "completion_effect": localized(
                        language,
                        "重新校验已确认的 endpoint、自定义 RPC method/schema 和执行前置条件，然后提交一次隔离的安全小流量 real-node smoke。",
                        "Revalidate the confirmed endpoint, custom RPC method/schema, and execution prerequisites, then submit one isolated safe low-traffic real-node smoke.",
                    ),
                },
                {
                    "label": "N",
                    "value": False,
                    "action": {"type": "reject_preflight_smoke"},
                    "expected_patch": {"preflight.approved": False},
                    "completion_effect": localized(
                        language,
                        "不重新校验或提交 real-node smoke，并返回配置流程。",
                        "Do not revalidate or submit the real-node smoke; return to configuration.",
                    ),
                },
            ],
            queue_barrier=True,
        )
        question["execution_request_id"] = str((state.get("preflight") or {}).get("execution_request_id") or uuid.uuid4().hex)
        return question
    if group == "job_monitoring" and _ready_for_final_benchmark(state):
        language = str(state.get("language") or "en")
        return choice_question(
            group,
            "real_node_final_benchmark_confirm",
            localized(
                language,
                "隔离的 real-node smoke 已成功完成。是否按已确认的 QPS profile 提交正式 benchmark？",
                "The isolated real-node smoke completed successfully. Submit the final benchmark with the confirmed QPS profile?",
            ),
            field="final_benchmark_confirmed",
            kind="yes_no",
            options=[
                {"label": "Y", "value": True, "action": {"type": "approve_final_benchmark"}, "expected_patch": {"final_benchmark.approved": True}},
                {"label": "N", "value": False, "action": {"type": "reject_final_benchmark"}, "expected_patch": {"final_benchmark.approved": False}},
            ],
            queue_barrier=True,
        )
    if group != "preflight_smoke_execution":
        return None
    next_group, _reason = next_group_and_reason(state)
    if next_group != "preflight_smoke_execution":
        return None
    language = str(state.get("language") or "en")
    question = choice_question(
        group,
        "preflight_smoke_confirm",
        localized(language, "配置已收集。是否运行 preflight 和 smoke？", "Configuration is collected. Run preflight and smoke?"),
        field="preflight_smoke_confirmed",
        kind="yes_no",
        options=[
            {"label": "Y", "value": True, "action": {"type": "approve_preflight_smoke"}, "expected_patch": {"preflight.approved": True}},
            {"label": "N", "value": False, "action": {"type": "reject_preflight_smoke"}, "expected_patch": {"preflight.approved": False}},
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
        visible_result=localized(
            next_state.get("language", "en"),
            "已暂停 preflight/smoke。可以继续修改链、RPC、QPS、磁盘或可观测性；准备好后再批准执行。",
            "Preflight/smoke is paused. You can change chain, RPC, QPS, disk, or observability and approve execution when ready.",
        ),
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
        return HandlerResult(blocker=f"unsupported execution action: {action.action_type}")
    return replace(result, consumed_action_ids=(action.action_id,))


def reconcile_execution_state(state: AgentGraphState) -> HandlerResult:
    """Refresh the workflow-owned job and advance only proven execution phases."""

    next_state: AgentGraphState = deepcopy(state)
    current_job = dict(next_state.get("job") or {})
    job_id = str(current_job.get("job_id") or "").strip()
    if not job_id:
        return HandlerResult()
    from agent.runners.job_manager import get_job
    try:
        persisted = get_job(job_id)
    except (FileNotFoundError, OSError, ValueError):
        return HandlerResult()
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
            visible_result=localized(
                state.get("language", "en"),
                "已暂停正式 benchmark。隔离 smoke 的结果和当前配置仍保留；你可以修改配置，或稍后明确要求提交正式 benchmark。",
                "The final benchmark is paused. The isolated-smoke result and current configuration are preserved; change the configuration or explicitly request final submission later.",
            ),
            clear_pending=True,
            completion="blocked",
            stop_after_response=True,
        )
    if not _ready_for_final_benchmark(state):
        return HandlerResult(blocker="final benchmark requires a completed isolated real-node smoke")
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
