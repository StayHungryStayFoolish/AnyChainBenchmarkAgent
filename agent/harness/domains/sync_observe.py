"""Sync-observe domain.

This mode observes a real node. It never runs Vegeta and never treats fake-node
fixtures or a demo acknowledgement as node-performance evidence.
"""

from __future__ import annotations

import re
from typing import Any

from ..contracts import ActionProposal, HandlerResult, StateDelta
from ..localization import localized
from ..questions import choice_question, manual_question
from ..state import AgentGraphState
from ..sync_observe_contract import REAL_SYNC_SOURCES, SyncObserveRequest

from agent.llm.search_grounding import run_google_search_grounding
from agent.workflows.group_registry import invalidation_targets
SYNC_OBSERVE_GROUPS = {"sync_observe"}
REAL_SOURCES = set(REAL_SYNC_SOURCES)


def question_for_sync_observe(state: AgentGraphState) -> dict[str, Any] | None:
    if state.get("workflow_mode") != "sync_observe":
        return None
    language = str(state.get("language") or "en")
    zh = language.startswith("zh")
    request = SyncObserveRequest.from_state(state)
    blocker = request.blocker()
    if blocker is None or blocker.group != "sync_observe":
        return None
    if blocker.question_id == "sync_observe_source":
        return choice_question(
            "sync_observe",
            "sync_observe_source",
            localized(
                language,
                "请选择 sync-observe 的真实数据来源。该模式不运行 Vegeta，也不把 fake-node 当作性能数据源。",
                "Choose the real data source for sync-observe. This mode does not run Vegeta or use fake-node as a performance source.",
            ),
            field="sync_observe_source",
            options=[
                {
                    "label": "本机真实节点进程" if zh else "Existing local real-node process",
                    "value": "existing_local_node",
                    "expected_patch": {"sync_observe.source": "existing_local_node"},
                },
                {
                    "label": "真实 RPC/metrics endpoint" if zh else "Real RPC/metrics endpoint",
                    "value": "endpoint_only",
                    "expected_patch": {"sync_observe.source": "endpoint_only"},
                },
                {
                    "label": "生成节点客户端准备说明" if zh else "Generate real-node client setup guidance",
                    "value": "client_setup",
                    "expected_patch": {"sync_observe.source": "client_setup"},
                    "return_policy": "stop_after_response",
                },
            ],
            accepted_action_types=("set_sync_observe_source",),
        )
    if blocker.question_id == "sync_observe_after_client_setup":
        return choice_question(
            "sync_observe",
            "sync_observe_after_client_setup",
            localized(
                language,
                "准备真实节点客户端后，必须选择本机进程或真实 endpoint 才能开始观察。",
                "After preparing a real-node client, select a local process or real endpoint before observation can start.",
            ),
            field="sync_observe_after_client_setup",
            options=[
                {
                    "label": "本机真实节点进程" if zh else "Existing local process",
                    "value": "existing_local_node",
                    "expected_patch": {"sync_observe.source": "existing_local_node"},
                },
                {
                    "label": "真实 RPC/metrics endpoint" if zh else "Real RPC/metrics endpoint",
                    "value": "endpoint_only",
                    "expected_patch": {"sync_observe.source": "endpoint_only"},
                },
            ],
        )
    if blocker.question_id == "sync_observe_stop_condition":
        return choice_question(
            "sync_observe",
            "sync_observe_stop_condition",
            localized(language, "请选择停止条件。默认是一直运行直到用户停止。", "Choose a stop condition. The default is run until stopped."),
            field="sync_observe_stop_condition",
            options=[
                {
                    "label": "一直运行直到停止" if zh else "Run until stopped",
                    "value": "until_stopped",
                    "expected_patch": {"sync_observe.stop_condition": "until_stopped"},
                },
                {
                    "label": "固定时长" if zh else "Fixed duration",
                    "value": "duration",
                    "expected_patch": {"sync_observe.stop_condition": "duration"},
                },
                {
                    "label": "同步完成后停止" if zh else "Stop when synced",
                    "value": "until_synced",
                    "expected_patch": {"sync_observe.stop_condition": "until_synced"},
                },
            ],
        )
    if blocker.question_id == "sync_observe_duration_seconds":
        return manual_question(
            "sync_observe",
            "sync_observe_duration_seconds",
            localized(language, "请输入观察时长（秒），必须是正整数。", "Enter observation duration in seconds as a positive integer."),
            field="sync_observe_duration_seconds",
            kind="positive_integer",
            validation={"value_type": "positive_integer"},
            candidate_bindings=({
                "type": "set_sync_observe_options",
                "value_argument": "sync_observe_duration_seconds",
            },),
        )
    return None


def apply_sync_observe_answer(
    state: AgentGraphState, value: Any, question: dict[str, Any]
) -> HandlerResult:
    field = str(question.get("field") or "")
    sync = dict(state.get("sync_observe") or {})
    visible = ""
    stop = False
    next_group = ""
    if field in {"sync_observe_source", "sync_observe_after_client_setup"}:
        sync["source"] = str(value)
        if value in REAL_SOURCES:
            next_group = "endpoint_process"
        else:
            visible = _client_setup_guidance(state)
            stop = True
    elif field == "sync_observe_stop_condition":
        sync["stop_condition"] = str(value)
    elif field == "sync_observe_duration_seconds":
        candidate = str(value).strip()
        if not re.fullmatch(r"[0-9]+", candidate) or int(candidate) <= 0:
            return HandlerResult(
                blocker=localized(state.get("language", "en"), "观察时长无效，请输入正整数。", "Invalid duration; enter a positive integer."),
            )
        sync["duration_seconds"] = int(candidate)
    return HandlerResult(
        delta=StateDelta.set_values({"sync_observe": sync}),
        clear_pending=True,
        next_group=next_group,
        visible_result=visible,
        completion="completed" if _sync_configuration_complete(sync) else "in_progress",
        stop_after_response=stop,
    )


def _sync_configuration_complete(sync: dict[str, Any]) -> bool:
    if sync.get("source") not in REAL_SOURCES:
        return False
    condition = str(sync.get("stop_condition") or "")
    if condition == "duration":
        return bool(sync.get("duration_seconds"))
    return condition in {"until_stopped", "until_synced"}


def apply_sync_observe_action(state: AgentGraphState, action: ActionProposal) -> HandlerResult:
    if action.action_type == "clear_sync_observe_source":
        sync = dict(state.get("sync_observe") or {})
        sync.pop("source", None)
        sync.pop("local_attribution_available", None)
        return HandlerResult(
            delta=StateDelta.set_values({"sync_observe": sync}),
            consumed_action_ids=(action.action_id,),
            invalidated_groups=invalidation_targets("sync_observe"),
            clear_pending=True,
            next_group="sync_observe",
            completion="in_progress",
        )
    if action.action_type == "set_sync_observe_options":
        sync = dict(state.get("sync_observe") or {})
        condition = str(action.arguments.get("sync_observe_stop_condition") or "").strip()
        duration = action.arguments.get("sync_observe_duration_seconds")
        if condition:
            sync["stop_condition"] = condition
            if condition != "duration":
                sync.pop("duration_seconds", None)
        if duration is not None:
            if condition and condition != "duration":
                return HandlerResult(blocker="sync-observe duration requires stop_condition=duration")
            sync["stop_condition"] = "duration"
            sync["duration_seconds"] = int(duration)
        return HandlerResult(
            delta=StateDelta.set_values({"sync_observe": sync}),
            consumed_action_ids=(action.action_id,),
            invalidated_groups=invalidation_targets("sync_observe"),
            clear_pending=True,
            next_group="sync_observe",
            completion="in_progress",
        )
    if action.action_type != "set_sync_observe_source":
        return HandlerResult(blocker=f"unsupported sync-observe action: {action.action_type}")
    source = str(action.arguments.get("sync_observe_source") or "").strip().lower()
    if source not in REAL_SOURCES | {"client_setup"}:
        return HandlerResult(blocker="sync-observe requires a real source or client setup guidance")
    if state.get("workflow_mode") != "sync_observe":
        return HandlerResult(blocker="sync-observe source requires sync-observe mode")
    sync = dict(state.get("sync_observe") or {})
    if sync.get("source") != source:
        sync = {"source": source}
    return HandlerResult(
        delta=StateDelta.set_values({
            "sync_observe": sync,
        }),
        consumed_action_ids=(action.action_id,),
        invalidated_groups=invalidation_targets("sync_observe"),
        clear_pending=True,
        next_group="endpoint_process" if source in REAL_SOURCES else "sync_observe",
        visible_result=_client_setup_guidance(state) if source == "client_setup" else "",
        stop_after_response=source == "client_setup",
    )


def _client_setup_guidance(state: AgentGraphState) -> str:
    language = str(state.get("language") or "en")
    base = localized(
        language,
        "准备并启动真实节点后，回来选择本机进程或真实 endpoint。Agent 不会自动下载客户端，也不会在没有真实数据源时声称获得了性能数据。",
        "Prepare and start a real node, then return to select its local process or real endpoint. The Agent does not auto-download a client or claim performance data without a real source.",
    )
    if not bool((state.get("web_research") or {}).get("google_search_available")):
        return base + localized(
            language,
            " 当前模型没有 google_search；请提供客户端官方文档，或自行安装后提供 endpoint。",
            " The current model has no google_search; provide official client documentation, or install it and provide the endpoint.",
        )
    chain = str((state.get("chain_identity") or {}).get("canonical") or "blockchain").strip()
    result = run_google_search_grounding(f"{chain} official node client installation metrics documentation")
    summary = str(getattr(result, "text_summary", "") or "").strip()
    citations = [str(item) for item in (getattr(result, "citations", None) or []) if str(item).strip()]
    if not bool(getattr(result, "available", False)) or not summary:
        return base + localized(
            language,
            " google_search 没有返回可验证的官方准备资料；请提供官方文档或安装后的 endpoint。",
            " google_search did not return verifiable official setup material; provide official documentation or the installed endpoint.",
        )
    evidence = summary
    if citations:
        evidence += "\n" + "\n".join(f"- {item}" for item in citations)
    return base + "\n" + evidence
