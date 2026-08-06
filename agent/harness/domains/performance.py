"""Performance domain: QPS profiles, observability, and advanced tuning."""

from __future__ import annotations

import re
from copy import deepcopy
from functools import partial
from typing import Any

from ..contracts import (
    ActionProposal,
    FailureDescriptor,
    HandlerResult,
    ResponseFragment,
    StateDelta,
)
from ..questions import (
    choice_question as _choice_question,
    manual_question as _manual_question,
    normalize_scalar,
    question_text,
)
from ..state import AgentGraphState
from ..input_values import normalize_observability_mode
from ..transitions import record_group_invalidations

from agent.knowledge.qps_profiles import qps_profile_defaults

choice_question = partial(_choice_question, owner="performance")
manual_question = partial(_manual_question, owner="performance")

PERFORMANCE_GROUPS = {"qps_profile", "observability", "advanced_tuning"}
QPS_FIELDS = ("INITIAL_QPS", "MAX_QPS", "QPS_STEP", "DURATION")
ADVANCED_FIELDS = {
    "MONITOR_INTERVAL": "unified monitoring interval (seconds)",
    "DISK_MONITOR_RATE": "disk-specific monitor rate",
    "SUCCESS_RATE_THRESHOLD": "QPS success-rate threshold (%)",
    "MAX_LATENCY_THRESHOLD": "QPS max latency threshold (ms)",
    "BOTTLENECK_CPU_THRESHOLD": "CPU bottleneck threshold (%)",
    "BOTTLENECK_MEMORY_THRESHOLD": "memory bottleneck threshold (%)",
    "BOTTLENECK_DISK_UTIL_THRESHOLD": "disk utilization bottleneck threshold (%)",
    "BOTTLENECK_DISK_LATENCY_THRESHOLD": "disk latency bottleneck threshold (ms)",
    "BOTTLENECK_NETWORK_THRESHOLD": "network bottleneck threshold (%)",
    "BOTTLENECK_ERROR_RATE_THRESHOLD": "error-rate bottleneck threshold (%)",
    "BOTTLENECK_DISK_IOPS_THRESHOLD": "disk IOPS bottleneck threshold (%)",
    "BOTTLENECK_DISK_THROUGHPUT_THRESHOLD": "disk throughput bottleneck threshold (%)",
}


def question_for_performance(state: AgentGraphState, group: str) -> dict[str, Any] | None:
    if group not in PERFORMANCE_GROUPS:
        return None
    if group == "qps_profile":
        qps = state.get("qps_profile") or {}
        if not qps.get("mode"):
            return choice_question(
                group,
                "benchmark_mode",
                question_text("question.performance.benchmark_mode.prompt"),
                field="benchmark_mode",
                options=[
                    {"label": question_text("question.performance.qps_mode.quick"), "value": "quick", "expected_patch": {"qps_profile.mode": "quick"}},
                    {"label": question_text("question.performance.qps_mode.standard"), "value": "standard", "expected_patch": {"qps_profile.mode": "standard"}},
                    {"label": question_text("question.performance.qps_mode.intensive"), "value": "intensive", "expected_patch": {"qps_profile.mode": "intensive"}},
                ],
                accepted_action_types=("set_qps_mode",),
            )
        if not qps.get("default_decision_made"):
            return choice_question(
                group,
                "qps_profile_confirm",
                question_text(
                    (
                        "question.performance.qps_profile_confirm_fake.prompt"
                        if state.get("target_mode") == "fake-node"
                        else "question.performance.qps_profile_confirm.prompt"
                    ),
                    mode=str(qps.get("mode") or ""),
                    profile=", ".join(
                        f"{key}={value}"
                        for key, value in qps_profile_defaults(
                            str(qps.get("mode") or "")
                        ).items()
                    ),
                ),
                field="qps_profile_confirmed",
                kind="yes_no",
                options=[
                    {"label": question_text("question.common.option.yes"), "value": True, "expected_patch": {"qps_profile.confirmed": True}},
                    {"label": question_text("question.common.option.no"), "value": False, "expected_patch": {"qps_profile.default_decision_made": True}},
                ],
                accepted_action_types=("request_qps_customization",),
                queue_barrier=True,
            )
        if not qps.get("confirmed"):
            if not qps.get("adjust_field"):
                options = [
                    {
                        "label": question_text(
                            f"question.performance.qps_field.{field.lower()}"
                        ),
                        "value": field,
                        "expected_patch": {"qps_profile.adjust_field": field},
                    }
                    for field in QPS_FIELDS
                ]
                options.append(
                    {
                        "label": question_text(
                            "question.performance.qps_adjustments.finish"
                        ),
                        "value": "done",
                        "expected_patch": {"qps_profile.confirmed": True},
                    }
                )
                return choice_question(
                    group,
                    "qps_adjust_field",
                    question_text(
                        "question.performance.qps_adjust_field.prompt",
                        mode=str(qps.get("mode") or ""),
                    ),
                    field="qps_adjust_field",
                    options=options,
                )
            return manual_question(
                group,
                "qps_adjust_value",
                question_text(
                    "question.performance.qps_adjust_value.prompt",
                    field=str(qps.get("adjust_field") or ""),
                ),
                field="qps_adjust_value",
                validation={"value_type": "positive_number"},
                candidate_bindings=({
                    "type": "set_qps_override",
                    "value_argument": "qps_overrides",
                    "mapping_key": str(qps.get("adjust_field") or ""),
                },),
            )
        return None
    if group == "observability":
        if (state.get("observability") or {}).get("mode"):
            return None
        return choice_question(
            group,
            "observability_mode",
            question_text("question.performance.observability_mode.prompt"),
            field="observability_mode",
            options=[
                {"label": question_text("question.performance.observability.disabled"), "value": "disabled", "expected_patch": {"observability.mode": "disabled"}},
                {"label": question_text("question.performance.observability.local"), "value": "local", "expected_patch": {"observability.mode": "local"}},
                {"label": question_text("question.performance.observability.exporter"), "value": "exporter", "expected_patch": {"observability.mode": "exporter"}},
            ],
            accepted_action_types=("set_observability",),
        )
    tuning = state.get("advanced_tuning") or {}
    if not tuning.get("default_decision_made"):
        return choice_question(
            group,
            "advanced_tuning_confirm",
            question_text(
                "question.performance.advanced_tuning_confirm.prompt",
                fields=", ".join(ADVANCED_FIELDS),
            ),
            field="advanced_tuning_confirmed",
            kind="yes_no",
            options=[
                {"label": question_text("question.common.option.yes"), "value": True, "expected_patch": {"advanced_tuning.confirmed": True}},
                {"label": question_text("question.common.option.no"), "value": False, "expected_patch": {"advanced_tuning.default_decision_made": True}},
            ],
        )
    if not tuning.get("confirmed"):
        if not tuning.get("adjust_field"):
            options = [
                {
                    "label": question_text(
                        "question.performance.advanced_field.option",
                        field=field,
                    ),
                    "value": field,
                    "expected_patch": {"advanced_tuning.adjust_field": field},
                }
                for field in ADVANCED_FIELDS
            ]
            options.append(
                {
                    "label": question_text(
                        "question.performance.advanced_adjustments.finish"
                    ),
                    "value": "done",
                    "expected_patch": {"advanced_tuning.confirmed": True},
                }
            )
            return choice_question(
                group,
                "advanced_tuning_adjust_field",
                question_text(
                    "question.performance.advanced_tuning_adjust_field.prompt"
                ),
                field="advanced_tuning_adjust_field",
                options=options,
            )
        return manual_question(
            group,
            "advanced_tuning_adjust_value",
            question_text(
                "question.performance.advanced_tuning_adjust_value.prompt",
                field=str(tuning.get("adjust_field") or ""),
            ),
            field="advanced_tuning_adjust_value",
            validation={"value_type": "positive_number"},
        )
    return None


def apply_performance_answer(state: AgentGraphState, question: dict[str, Any], value: Any) -> HandlerResult:
    next_state, response_fragment, blocked = _apply_performance_answer_state(
        deepcopy(state), question, value
    )
    group = str(question.get("group") or "")
    previous_invalidated = set(state.get("invalidated_groups") or [])
    next_invalidated = set(next_state.get("invalidated_groups") or [])
    return HandlerResult(
        delta=StateDelta.between(state, next_state),
        invalidated_groups=tuple(sorted(next_invalidated - previous_invalidated)),
        reconfigured_groups=tuple(sorted(previous_invalidated - next_invalidated)),
        response_fragments=(response_fragment,) if response_fragment else (),
        pending_question=question if blocked else None,
        clear_pending=not blocked,
        next_group=group,
        completion="blocked" if blocked else "in_progress",
    )


def _apply_performance_answer_state(
    state: AgentGraphState,
    question: dict[str, Any],
    value: Any,
) -> tuple[AgentGraphState, ResponseFragment | None, bool]:
    group = str(question.get("group") or "")
    field = str(question.get("field") or "")
    response_fragment = None
    if group == "qps_profile":
        qps = state.setdefault("qps_profile", {})
        if field == "benchmark_mode":
            customization_requested = bool(qps.get("customization_requested"))
            qps.clear()
            qps.update({
                "mode": str(value),
                "confirmed": False,
                "default_decision_made": customization_requested,
            })
        elif field == "qps_profile_confirmed":
            qps["default_decision_made"] = True
            qps["confirmed"] = bool(value)
        elif field == "qps_adjust_field":
            if value == "done":
                qps["confirmed"] = True
                qps.pop("adjust_field", None)
            else:
                qps["adjust_field"] = str(value)
        elif field == "qps_adjust_value":
            adjust_field = str(qps.get("adjust_field") or "")
            candidate = normalize_scalar(str(value))
            merged = {**qps_profile_defaults(qps.get("mode")), **(qps.get("overrides") or {}), adjust_field: candidate}
            error = validate_qps_profile(merged)
            if error:
                response_fragment = ResponseFragment(
                    kind="warning",
                    message_id="harness.performance.invalid_qps_value",
                    arguments={"error": error},
                    source=__name__,
                )
                return state, response_fragment, True
            qps.setdefault("overrides", {})[adjust_field] = candidate
            qps.pop("adjust_field", None)
    elif group == "observability":
        mode = str(value)
        state.setdefault("observability", {})["mode"] = mode
        if mode == "exporter":
            response_fragment = ResponseFragment(
                kind="message",
                message_id="harness.performance.exporter_selected",
                source=__name__,
            )
        elif mode == "local":
            response_fragment = ResponseFragment(
                kind="message",
                message_id="harness.performance.local_observability_selected",
                source=__name__,
            )
    else:
        tuning = state.setdefault("advanced_tuning", {})
        if field == "advanced_tuning_confirmed":
            tuning["default_decision_made"] = True
            tuning["confirmed"] = bool(value)
        elif field == "advanced_tuning_adjust_field":
            if value == "done":
                tuning["confirmed"] = True
                tuning.pop("adjust_field", None)
            else:
                tuning["adjust_field"] = str(value)
        elif field == "advanced_tuning_adjust_value":
            adjust_field = str(tuning.get("adjust_field") or "")
            candidate = normalize_scalar(str(value))
            if not valid_advanced_value(adjust_field, candidate):
                response_fragment = ResponseFragment(
                    kind="warning",
                    message_id="harness.performance.invalid_advanced_value",
                    source=__name__,
                )
                return state, response_fragment, True
            tuning.setdefault("overrides", {})[adjust_field] = candidate
            tuning.pop("adjust_field", None)
    invalidated = set(state.get("invalidated_groups") or [])
    invalidated.discard(group)
    state["invalidated_groups"] = sorted(invalidated)
    record_group_invalidations(state, group)
    return state, response_fragment, False


def validate_qps_profile(values: dict[str, Any]) -> str:
    parsed: dict[str, int] = {}
    for field in QPS_FIELDS:
        raw = str(values.get(field) or "").strip()
        if not re.fullmatch(r"[0-9]+", raw) or int(raw) <= 0:
            return f"{field}={values.get(field)}"
        parsed[field] = int(raw)
    if parsed["MAX_QPS"] < parsed["INITIAL_QPS"]:
        return f"MAX_QPS({parsed['MAX_QPS']}) < INITIAL_QPS({parsed['INITIAL_QPS']})"
    return ""


def validate_qps_overrides(
    mode: str,
    overrides: dict[str, Any],
    current_overrides: dict[str, Any] | None = None,
) -> str:
    """Validate partial overrides against the selected mode's full profile."""

    merged = {
        **qps_profile_defaults(mode),
        **(current_overrides or {}),
        **{str(key): normalize_scalar(str(value)) for key, value in overrides.items()},
    }
    return validate_qps_profile(merged)


def valid_advanced_value(field: str, value: str) -> bool:
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) or float(value) <= 0:
        return False
    return not ("(%)" in ADVANCED_FIELDS.get(field, "") and float(value) > 100)


def apply_performance_action(state: AgentGraphState, action: ActionProposal) -> HandlerResult:
    next_state: AgentGraphState = deepcopy(state)
    response_fragment = None
    completion = "completed"
    if action.action_type == "set_qps_mode":
        mode = str(action.arguments.get("qps_mode") or "").strip().lower()
        if mode not in {"quick", "standard", "intensive"}:
            return HandlerResult(
                blocker=FailureDescriptor(
                    code="harness.performance.failure.invalid_qps_mode",
                    source=__name__,
                )
            )
        current_qps = dict(next_state.get("qps_profile") or {})
        if str(current_qps.get("mode") or "").strip().lower() == mode:
            return HandlerResult(
                consumed_action_ids=(action.action_id,),
                completion="unchanged",
            )
        customization_requested = bool(current_qps.get("customization_requested"))
        next_state["qps_profile"] = {
            "mode": mode,
            "confirmed": False,
            "default_decision_made": customization_requested,
        }
        next_group = "qps_profile"
        completion = "in_progress"
    elif action.action_type == "request_qps_customization":
        qps = next_state.setdefault("qps_profile", {})
        if not qps.get("mode"):
            qps.update({
                "confirmed": False,
                "default_decision_made": False,
                "customization_requested": True,
            })
            next_group = "qps_profile"
            completion = "in_progress"
        else:
            requested = action.arguments.get("qps_fields")
            if isinstance(requested, str):
                requested = [requested]
            fields = [str(item).strip().upper() for item in requested or [] if str(item).strip().upper() in QPS_FIELDS]
            qps.update({"confirmed": False, "default_decision_made": True})
            qps.pop("customization_requested", None)
            if len(fields) == 1:
                qps["adjust_field"] = fields[0]
            next_group = "qps_profile"
            completion = "in_progress"
    elif action.action_type == "set_qps_override":
        values = action.arguments.get("qps_overrides")
        if not isinstance(values, dict) or not values:
            return HandlerResult(
                blocker=FailureDescriptor(
                    code="harness.performance.failure.qps_overrides_required",
                    source=__name__,
                )
            )
        qps = next_state.setdefault("qps_profile", {})
        normalized = {str(key): normalize_scalar(str(value)) for key, value in values.items()}
        error = validate_qps_overrides(
            str(qps.get("mode") or ""),
            normalized,
            qps.get("overrides") or {},
        )
        if error:
            return HandlerResult(
                blocker=FailureDescriptor(
                    code="harness.performance.failure.invalid_qps_profile",
                    arguments={"error": error},
                    source=__name__,
                )
            )
        qps.setdefault("overrides", {}).update({key: normalized[key] for key in QPS_FIELDS if key in normalized})
        qps.update({"confirmed": True, "default_decision_made": True})
        next_group = "qps_profile"
    elif action.action_type == "set_observability":
        mode = normalize_observability_mode(action.arguments.get("observability_mode"))
        if not mode:
            return HandlerResult(
                blocker=FailureDescriptor(
                    code="harness.performance.failure.invalid_observability_mode",
                    source=__name__,
                )
            )
        next_state["observability"] = {"mode": mode}
        next_group = "observability"
        if mode == "exporter":
            response_fragment = ResponseFragment(
                kind="message",
                message_id="harness.performance.exporter_selected",
                source=__name__,
            )
        elif mode == "local":
            response_fragment = ResponseFragment(
                kind="message",
                message_id="harness.performance.local_observability_selected",
                source=__name__,
            )
        else:
            response_fragment = ResponseFragment(
                kind="message",
                message_id="harness.performance.observability_disabled",
                source=__name__,
            )
    else:
        return HandlerResult(
            blocker=FailureDescriptor(
                code="harness.performance.failure.unsupported_action",
                arguments={"action_type": action.action_type},
                source=__name__,
            )
        )
    invalidated = set(next_state.get("invalidated_groups") or [])
    invalidated.discard(next_group)
    next_state["invalidated_groups"] = sorted(invalidated)
    record_group_invalidations(next_state, next_group)
    invalidated = set(next_state.get("invalidated_groups") or [])
    return HandlerResult(
        delta=StateDelta.between(state, next_state),
        consumed_action_ids=(action.action_id,),
        invalidated_groups=tuple(sorted(invalidated - set(state.get("invalidated_groups") or []))),
        reconfigured_groups=tuple(sorted(set(state.get("invalidated_groups") or []) - invalidated)),
        clear_pending=True,
        next_group=next_group,
        completion=completion,
        response_fragments=(response_fragment,) if response_fragment else (),
    )
