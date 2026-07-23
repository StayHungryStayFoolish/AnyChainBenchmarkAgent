"""Performance domain: QPS profiles, observability, and advanced tuning."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from ..contracts import ActionProposal, HandlerResult, StateDelta
from ..localization import localized
from ..questions import choice_question, manual_question, normalize_scalar
from ..state import AgentGraphState
from ..input_values import normalize_observability_mode
from ..transitions import record_group_invalidations

from agent.planners import question_prompts
PERFORMANCE_GROUPS = {"qps_profile", "observability", "advanced_tuning"}
QPS_FIELDS = ("INITIAL_QPS", "MAX_QPS", "QPS_STEP", "DURATION")
ADVANCED_FIELDS = {
    "MONITOR_INTERVAL": "MONITOR_INTERVAL / unified monitoring interval (seconds)",
    "DISK_MONITOR_RATE": "DISK_MONITOR_RATE / disk-specific monitor rate",
    "SUCCESS_RATE_THRESHOLD": "SUCCESS_RATE_THRESHOLD / QPS success-rate threshold (%)",
    "MAX_LATENCY_THRESHOLD": "MAX_LATENCY_THRESHOLD / QPS max latency threshold (ms)",
    "BOTTLENECK_CPU_THRESHOLD": "BOTTLENECK_CPU_THRESHOLD / CPU bottleneck threshold (%)",
    "BOTTLENECK_MEMORY_THRESHOLD": "BOTTLENECK_MEMORY_THRESHOLD / memory bottleneck threshold (%)",
    "BOTTLENECK_DISK_UTIL_THRESHOLD": "BOTTLENECK_DISK_UTIL_THRESHOLD / disk utilization bottleneck threshold (%)",
    "BOTTLENECK_DISK_LATENCY_THRESHOLD": "BOTTLENECK_DISK_LATENCY_THRESHOLD / disk latency bottleneck threshold (ms)",
    "BOTTLENECK_NETWORK_THRESHOLD": "BOTTLENECK_NETWORK_THRESHOLD / network bottleneck threshold (%)",
    "BOTTLENECK_ERROR_RATE_THRESHOLD": "BOTTLENECK_ERROR_RATE_THRESHOLD / error-rate bottleneck threshold (%)",
    "BOTTLENECK_DISK_IOPS_THRESHOLD": "BOTTLENECK_DISK_IOPS_THRESHOLD / disk IOPS bottleneck threshold (%)",
    "BOTTLENECK_DISK_THROUGHPUT_THRESHOLD": "BOTTLENECK_DISK_THROUGHPUT_THRESHOLD / disk throughput bottleneck threshold (%)",
}


def question_for_performance(state: AgentGraphState, group: str) -> dict[str, Any] | None:
    if group not in PERFORMANCE_GROUPS:
        return None
    language = str(state.get("language") or "en")
    zh = language.startswith("zh")
    if group == "qps_profile":
        qps = state.get("qps_profile") or {}
        if not qps.get("mode"):
            return choice_question(
                group,
                "benchmark_mode",
                localized(language, "请选择 benchmark 模式。", "Choose benchmark mode."),
                field="benchmark_mode",
                options=[
                    {"label": "quick", "value": "quick", "expected_patch": {"qps_profile.mode": "quick"}},
                    {"label": "standard", "value": "standard", "expected_patch": {"qps_profile.mode": "standard"}},
                    {"label": "intensive", "value": "intensive", "expected_patch": {"qps_profile.mode": "intensive"}},
                ],
                accepted_action_types=("set_qps_mode",),
            )
        if not qps.get("default_decision_made"):
            return choice_question(
                group,
                "qps_profile_confirm",
                question_prompts.qps_profile_prompt(
                    str(qps.get("mode") or ""),
                    fake_node=state.get("target_mode") == "fake-node",
                    language=language,
                ),
                field="qps_profile_confirmed",
                kind="yes_no",
                options=[
                    {"label": "Y", "value": True, "expected_patch": {"qps_profile.confirmed": True}},
                    {"label": "N", "value": False, "expected_patch": {"qps_profile.default_decision_made": True}},
                ],
                queue_barrier=True,
            )
        if not qps.get("confirmed"):
            if not qps.get("adjust_field"):
                options = [
                    {"label": f"{field} / {label}", "value": field, "expected_patch": {"qps_profile.adjust_field": field}}
                    for field, label in (
                        ("INITIAL_QPS", "起始 QPS" if zh else "initial QPS"),
                        ("MAX_QPS", "最高 QPS" if zh else "max QPS"),
                        ("QPS_STEP", "每级递增" if zh else "increment"),
                        ("DURATION", "每档持续秒数" if zh else "seconds per step"),
                    )
                ]
                options.append(
                    {
                        "label": "完成调整" if zh else "Finish QPS adjustments",
                        "value": "done",
                        "expected_patch": {"qps_profile.confirmed": True},
                    }
                )
                return choice_question(
                    group,
                    "qps_adjust_field",
                    localized(language, f"请选择要调整的 {qps.get('mode')} QPS 参数。", f"Choose the {qps.get('mode')} QPS parameter to adjust."),
                    field="qps_adjust_field",
                    options=options,
                )
            return manual_question(
                group,
                "qps_adjust_value",
                localized(language, f"请输入 {qps.get('adjust_field')} 的值。", f"Enter the value for {qps.get('adjust_field')}."),
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
            localized(language, "请选择可观测性模式。", "Choose observability mode."),
            field="observability_mode",
            options=[
                {"label": "禁用" if zh else "Disabled", "value": "disabled", "expected_patch": {"observability.mode": "disabled"}},
                {"label": "本地 Prometheus/Grafana" if zh else "Local Prometheus/Grafana", "value": "local", "expected_patch": {"observability.mode": "local"}},
                {"label": "仅 exporter，对接已有 Prometheus" if zh else "Exporter only for existing Prometheus", "value": "exporter", "expected_patch": {"observability.mode": "exporter"}},
            ],
            accepted_action_types=("set_observability",),
        )
    tuning = state.get("advanced_tuning") or {}
    if not tuning.get("default_decision_made"):
        return choice_question(
            group,
            "advanced_tuning_confirm",
            question_prompts.advanced_tuning_default_prompt(language=language),
            field="advanced_tuning_confirmed",
            kind="yes_no",
            options=[
                {"label": "Y", "value": True, "expected_patch": {"advanced_tuning.confirmed": True}},
                {"label": "N", "value": False, "expected_patch": {"advanced_tuning.default_decision_made": True}},
            ],
        )
    if not tuning.get("confirmed"):
        if not tuning.get("adjust_field"):
            options = [
                {"label": label, "value": field, "expected_patch": {"advanced_tuning.adjust_field": field}}
                for field, label in ADVANCED_FIELDS.items()
            ]
            options.append(
                {
                    "label": "完成调整" if zh else "Finish adjustments",
                    "value": "done",
                    "expected_patch": {"advanced_tuning.confirmed": True},
                }
            )
            return choice_question(
                group,
                "advanced_tuning_adjust_field",
                localized(language, "请选择要调整的高级调优参数。", "Choose the advanced tuning parameter to adjust."),
                field="advanced_tuning_adjust_field",
                options=options,
            )
        return manual_question(
            group,
            "advanced_tuning_adjust_value",
            localized(language, f"请输入 {tuning.get('adjust_field')} 的值。", f"Enter the value for {tuning.get('adjust_field')}."),
            field="advanced_tuning_adjust_value",
            validation={"value_type": "positive_number"},
        )
    return None


def apply_performance_answer(state: AgentGraphState, question: dict[str, Any], value: Any) -> HandlerResult:
    next_state, visible, blocked = _apply_performance_answer_state(deepcopy(state), question, value)
    group = str(question.get("group") or "")
    previous_invalidated = set(state.get("invalidated_groups") or [])
    next_invalidated = set(next_state.get("invalidated_groups") or [])
    return HandlerResult(
        delta=StateDelta.between(state, next_state),
        invalidated_groups=tuple(sorted(next_invalidated - previous_invalidated)),
        reconfigured_groups=tuple(sorted(previous_invalidated - next_invalidated)),
        visible_result=visible,
        pending_question=question if blocked else None,
        clear_pending=not blocked,
        next_group=group,
        completion="blocked" if blocked else "in_progress",
    )


def _apply_performance_answer_state(
    state: AgentGraphState,
    question: dict[str, Any],
    value: Any,
) -> tuple[AgentGraphState, str, bool]:
    group = str(question.get("group") or "")
    field = str(question.get("field") or "")
    visible = ""
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
            merged = {**question_prompts.qps_profile_defaults(qps.get("mode")), **(qps.get("overrides") or {}), adjust_field: candidate}
            error = validate_qps_profile(merged)
            if error:
                visible = localized(state.get("language", "en"), f"QPS 数值无效：{error}。请重新输入。", f"Invalid QPS value: {error}. Please re-enter.")
                return state, visible, True
            qps.setdefault("overrides", {})[adjust_field] = candidate
            qps.pop("adjust_field", None)
    elif group == "observability":
        mode = str(value)
        state.setdefault("observability", {})["mode"] = mode
        if mode == "exporter":
            visible = localized(state.get("language", "en"), "仅启动 exporter；请让已有 Prometheus 抓取 `http://<benchmark-host>:9108/metrics`。", "Exporter only; configure existing Prometheus to scrape `http://<benchmark-host>:9108/metrics`.")
        elif mode == "local":
            visible = localized(state.get("language", "en"), "将启动 exporter:9108、Prometheus:9091 和 Grafana:3001。", "Will start exporter:9108, Prometheus:9091, and Grafana:3001.")
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
                visible = localized(state.get("language", "en"), "数值无效：请输入正数；百分比字段不得超过 100。", "Invalid value: enter a positive number; percentage fields must not exceed 100.")
                return state, visible, True
            tuning.setdefault("overrides", {})[adjust_field] = candidate
            tuning.pop("adjust_field", None)
    invalidated = set(state.get("invalidated_groups") or [])
    invalidated.discard(group)
    state["invalidated_groups"] = sorted(invalidated)
    record_group_invalidations(state, group)
    return state, visible, False


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
        **question_prompts.qps_profile_defaults(mode),
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
    visible = ""
    completion = "completed"
    if action.action_type == "set_qps_mode":
        mode = str(action.arguments.get("qps_mode") or "").strip().lower()
        if mode not in {"quick", "standard", "intensive"}:
            return HandlerResult(blocker="qps_mode must be quick, standard, or intensive")
        customization_requested = bool((next_state.get("qps_profile") or {}).get("customization_requested"))
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
            return HandlerResult(blocker="qps_overrides is required")
        qps = next_state.setdefault("qps_profile", {})
        normalized = {str(key): normalize_scalar(str(value)) for key, value in values.items()}
        error = validate_qps_overrides(
            str(qps.get("mode") or ""),
            normalized,
            qps.get("overrides") or {},
        )
        if error:
            return HandlerResult(blocker=f"invalid QPS profile: {error}")
        qps.setdefault("overrides", {}).update({key: normalized[key] for key in QPS_FIELDS if key in normalized})
        qps.update({"confirmed": True, "default_decision_made": True})
        next_group = "qps_profile"
    elif action.action_type == "set_observability":
        mode = normalize_observability_mode(action.arguments.get("observability_mode"))
        if not mode:
            return HandlerResult(blocker="observability_mode must be disabled, local, or exporter")
        next_state["observability"] = {"mode": mode}
        next_group = "observability"
        if mode == "exporter":
            visible = localized(
                state.get("language", "en"),
                "已选择 exporter-only；已有 Prometheus 应抓取 `http://<benchmark-host>:9108/metrics`。",
                "Selected exporter-only; configure the existing Prometheus to scrape `http://<benchmark-host>:9108/metrics`.",
            )
        elif mode == "local":
            visible = localized(
                state.get("language", "en"),
                "已选择本地可观测性：exporter:9108、Prometheus:9091、Grafana:3001。",
                "Selected local observability: exporter:9108, Prometheus:9091, Grafana:3001.",
            )
        else:
            visible = localized(state.get("language", "en"), "已禁用本地可观测性组件。", "Local observability components are disabled.")
    else:
        return HandlerResult(blocker=f"unsupported performance action: {action.action_type}")
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
        visible_result=visible,
    )
