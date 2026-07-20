"""Policy-constrained failure recovery domain."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from ..contracts import ActionProposal, HandlerResult, StateDelta
from ..failures import RECOVERY_POLICIES, render_failure_summary, unresolved_recovery
from ..intent import analyze_evidence_with_model
from ..localization import localized
from ..questions import choice_question
from ..state import AgentGraphState


RECOVERY_GROUPS = {"failure_recovery"}


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
    language = str(state.get("language") or "en")
    options: list[dict[str, Any]] = []
    if "correct_failure" in set(record.get("allowed_actions") or []):
        options.append({
            "label": localized(language, "修正受影响的配置并重新验证", "Correct the affected configuration and revalidate"),
            "value": "correct",
            "action": {"type": "correct_failure"},
            "expected_patch": {"failure_recovery.status": "correcting"},
        })
    options.append({
        "label": localized(language, "查看错误证据和诊断信息", "Inspect failure evidence and diagnostics"),
        "value": "inspect",
        "action": {"type": "inspect_failure"},
        "expected_patch": {"failure_recovery.status": recovery.get("status") or "pending"},
        "return_policy": "stay",
    })
    if "retry_failure" in set(record.get("allowed_actions") or []):
        options.append({
            "label": localized(language, "恢复外部服务后重试当前请求", "Retry the current request after restoring the external service"),
            "value": "retry",
            "action": {"type": "retry_failure"},
            "expected_patch": {"failure_recovery.status": "resolved"},
        })
    options.append({
        "label": localized(language, "暂不修复，保留证据和配置", "Pause recovery and preserve evidence/configuration"),
        "value": "cancel",
        "action": {"type": "cancel_failure_recovery"},
        "expected_patch": {"failure_recovery.status": "cancelled"},
        "return_policy": "stop_after_response",
    })
    decision_prompt = localized(
        language,
        "请选择下一步。修正完成后，Agent 会重新运行相关校验并回到正常配置流程。",
        "Choose the next step. After correction, the Agent will rerun the relevant validation and resume the normal configuration flow.",
    )
    prompt = (
        render_failure_summary(record, language) + "\n" + decision_prompt
        if include_summary
        else decision_prompt
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
    record = dict(recovery.get("record") or {})
    allowed = set(record.get("allowed_actions") or [])
    if not record:
        return HandlerResult(blocker="no active failure record is available")
    if action.action_type not in allowed:
        return HandlerResult(blocker=f"recovery action is not allowed for {record.get('code')}: {action.action_type}")

    if action.action_type == "inspect_failure":
        recovery["selected_action"] = "inspect_failure"
        responses = [render_failure_summary(record, str(next_state.get("language") or "en"))]
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
            responses.append(advisory)
        next_question = question_for_recovery(next_state, "failure_recovery", include_summary=False)
        return HandlerResult(
            delta=StateDelta.between(state, next_state),
            consumed_action_ids=(action.action_id,),
            visible_results=tuple(responses),
            clear_pending=True,
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
            visible_result=localized(
                next_state.get("language", "en"),
                "已暂停恢复；错误证据和当前配置仍保留。你可以稍后查看 job、日志或重新开始修正。",
                "Recovery is paused; failure evidence and the current configuration remain available. You can inspect the job/logs or resume correction later.",
            ),
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
            visible_result=localized(
                next_state.get("language", "en"),
                "已解除本次外部服务错误状态。请重新提交刚才的请求；本次操作没有修改 benchmark 配置或重复提交 job。",
                "The external-service failure state is cleared. Retry your previous request; this action did not change benchmark configuration or resubmit a job.",
            ),
            clear_pending=True,
            next_group="opening",
            completion="completed",
            stop_after_response=True,
        )

    if action.action_type == "correct_failure":
        policy = RECOVERY_POLICIES.get(str(record.get("code") or ""))
        if policy is None or not policy.allow_correction:
            return HandlerResult(blocker=f"failure does not support configuration correction: {record.get('code')}")
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

    return HandlerResult(blocker=f"unsupported recovery action: {action.action_type}")
