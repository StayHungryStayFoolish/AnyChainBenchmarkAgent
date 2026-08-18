"""Policy-constrained failure recovery domain."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
import json
from typing import Any

from ..contracts import ActionProposal, HandlerResult, ResponseFragment, StateDelta
from ..failures import (
    RECOVERY_POLICIES,
    failure_record_response_fragment,
    unresolved_recovery,
)
from ..advisory import analyze_evidence_with_model
from ..localization import localized
from ..questions import choice_question as _choice_question, question_text
from ..state import AgentGraphState
from .analysis_receipts import analysis_hash
from .response_fragments import failure

choice_question = partial(_choice_question, owner="recovery")


RECOVERY_GROUPS = {"failure_recovery"}


def _fragment(message_id: str, *, kind: str = "status") -> ResponseFragment:
    return ResponseFragment(
        kind=kind,  # type: ignore[arg-type]
        message_id=message_id,
        source=__name__,
    )


def question_for_recovery(
    state: AgentGraphState,
    group: str,
    *,
    include_summary: bool = True,
) -> dict[str, Any] | None:
    recovery = dict(state.get("failure_recovery") or {})
    record = dict(recovery.get("record") or {})
    if group != "failure_recovery" or not unresolved_recovery(recovery):
        return None
    options: list[dict[str, Any]] = []
    if "correct_failure" in set(record.get("allowed_actions") or []):
        options.append({
            "label": question_text("question.recovery.option.correct"),
            "value": "correct",
            "action": {"type": "correct_failure"},
            "expected_patch": {"failure_recovery.status": "correcting"},
        })
    options.append({
        "label": question_text("question.recovery.option.inspect"),
        "value": "inspect",
        "action": {"type": "inspect_failure"},
        "expected_patch": {"failure_recovery.status": recovery.get("status") or "pending"},
        "return_policy": "stay",
    })
    if "retry_failure" in set(record.get("allowed_actions") or []):
        options.append({
            "label": question_text("question.recovery.option.retry"),
            "value": "retry",
            "action": {"type": "retry_failure"},
            "expected_patch": {"failure_recovery.status": "resolved"},
        })
    options.append({
        "label": question_text("question.recovery.option.cancel"),
        "value": "cancel",
        "action": {"type": "cancel_failure_recovery"},
        "expected_patch": {"failure_recovery.status": "cancelled"},
        "return_policy": "stop_after_response",
    })
    prompt = question_text(
        (
            "question.recovery.action_with_summary.prompt"
            if include_summary
            else "question.recovery.action.prompt"
        ),
        **(
            {
                "severity": str(record.get("severity") or "blocking"),
                "code": str(record.get("code") or "UNKNOWN"),
                "failure_id": str(record.get("failure_id") or "<unknown>"),
                "facts": json.dumps(
                    record.get("facts") or [],
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
                "evidence_paths": json.dumps(
                    record.get("evidence_paths") or [],
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            }
            if include_summary
            else {}
        ),
    )
    return choice_question(
        "failure_recovery",
        "failure_recovery_action",
        prompt,
        field="failure_recovery_action",
        options=options,
        queue_barrier=True,
    )


def apply_recovery_action(state: AgentGraphState, action: ActionProposal) -> HandlerResult:
    next_state: AgentGraphState = deepcopy(state)
    recovery = next_state.setdefault("failure_recovery", {})
    if action.action_type == "activate_harness_recovery":
        failure_record = action.arguments.get("failure_record")
        if not isinstance(failure_record, dict) or not failure_record.get("code"):
            return HandlerResult(blocker=failure(
                "recovery.failure.activation_record_required",
                source=__name__,
            ))
        next_state["failure_recovery"] = {
            "status": "pending",
            "record": deepcopy(failure_record),
        }
        pending = question_for_recovery(next_state, "failure_recovery")
        if pending is None:
            return HandlerResult(blocker=failure(
                "recovery.failure.contract_unavailable",
                source=__name__,
            ))
        return HandlerResult(
            delta=StateDelta.between(state, next_state),
            consumed_action_ids=(action.action_id,),
            pending_question=pending,
            next_group="failure_recovery",
            completion="blocked",
            stop_after_response=True,
        )
    record = dict(recovery.get("record") or {})
    allowed = set(record.get("allowed_actions") or [])
    if not record:
        return HandlerResult(blocker=failure(
            "recovery.failure.record_required",
            source=__name__,
        ))
    if action.action_type not in allowed:
        return HandlerResult(blocker=failure(
            "recovery.failure.action_not_allowed",
            arguments={
                "action_type": action.action_type,
                "code": str(record.get("code") or ""),
            },
            source=__name__,
        ))

    if action.action_type == "inspect_failure":
        recovery["selected_action"] = "inspect_failure"
        active_question = deepcopy(state.get("pending_question") or {})
        responses: list[ResponseFragment] = (
            [] if active_question else [failure_record_response_fragment(record)]
        )
        if record.get("llm_analysis_useful"):
            advisory = analyze_evidence_with_model(
                next_state,
                json.dumps(record, ensure_ascii=False, sort_keys=True),
                localized(
                    next_state.get("language", "en"),
                    "只补充可能原因和验证步骤，不要重复 failure heading、已观察事实、保留配置或证据路径；只能建议 failure record 中允许的动作。",
                    "Add only likely causes and validation steps. Do not repeat the failure heading, observed facts, preserved configuration, or evidence paths; recommend only actions allowed by the failure record.",
                ),
            )
            evidence_paths = [
                str(path) for path in record.get("evidence_paths") or []
            ]
            responses.append(ResponseFragment(
                kind="evidence",
                message_id="analysis.model_document",
                payload={
                    "text": advisory,
                    "source_kind": "failure_record",
                    "language": str(next_state.get("language") or "en"),
                    "evidence_hash": analysis_hash(record),
                    "evidence_paths": evidence_paths,
                },
                source=__name__,
            ))
        next_question = (
            active_question
            if active_question
            else question_for_recovery(next_state, "failure_recovery", include_summary=False)
        )
        return HandlerResult(
            delta=StateDelta.between(state, next_state),
            consumed_action_ids=(action.action_id,),
            response_fragments=tuple(responses),
            clear_pending=False,
            pending_question=next_question,
            next_group="failure_recovery",
            completion="in_progress",
            stop_after_response=True,
        )

    if action.action_type == "cancel_failure_recovery":
        recovery["status"] = "cancelled"
        recovery["selected_action"] = "cancel_failure_recovery"
        return HandlerResult(
            delta=StateDelta.between(state, next_state),
            consumed_action_ids=(action.action_id,),
            response_fragments=(_fragment("recovery.response.paused"),),
            clear_pending=True,
            next_group="failure_recovery",
            completion="completed",
            stop_after_response=True,
        )

    if action.action_type == "retry_failure":
        recovery["status"] = "resolved"
        recovery["selected_action"] = "retry_failure"
        return HandlerResult(
            delta=StateDelta.between(state, next_state),
            consumed_action_ids=(action.action_id,),
            response_fragments=(_fragment("recovery.response.retry_ready"),),
            clear_pending=True,
            next_group="opening",
            completion="completed",
            stop_after_response=True,
        )

    if action.action_type == "correct_failure":
        policy = RECOVERY_POLICIES.get(str(record.get("code") or ""))
        if policy is None or not policy.allow_correction:
            return HandlerResult(blocker=failure(
                "recovery.failure.correction_not_supported",
                arguments={"code": str(record.get("code") or "")},
                source=__name__,
            ))
        recovery["status"] = "correcting"
        recovery["selected_action"] = "correct_failure"
        return HandlerResult(
            delta=StateDelta.between(state, next_state),
            consumed_action_ids=(action.action_id,),
            invalidated_groups=policy.invalidated_groups,
            invalidated_fields=policy.invalidated_fields,
            clear_pending=True,
            next_group=policy.affected_group,
            completion="completed",
        )

    return HandlerResult(blocker=failure(
        "recovery.failure.unsupported_action",
        arguments={"action_type": action.action_type},
        source=__name__,
    ))
