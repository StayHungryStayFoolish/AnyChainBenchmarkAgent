"""Sync-observe domain.

This mode observes a real node. It never runs Vegeta and never treats fake-node
fixtures or a demo acknowledgement as node-performance evidence.
"""

from __future__ import annotations

from functools import partial
import re
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
    question_text,
)
from ..state import AgentGraphState
from ..sync_observe_contract import REAL_SYNC_SOURCES, SyncObserveRequest

from agent.llm.search_grounding import run_google_search_grounding
from agent.workflows.group_registry import invalidation_targets

choice_question = partial(_choice_question, owner="sync_observe")
manual_question = partial(_manual_question, owner="sync_observe")

SYNC_OBSERVE_GROUPS = {"sync_observe"}
REAL_SOURCES = set(REAL_SYNC_SOURCES)


def question_for_sync_observe(state: AgentGraphState) -> dict[str, Any] | None:
    if state.get("workflow_mode") != "sync_observe":
        return None
    request = SyncObserveRequest.from_state(state)
    blocker = request.blocker()
    if blocker is None or blocker.group != "sync_observe":
        return None
    if blocker.question_id == "sync_observe_source":
        return choice_question(
            "sync_observe",
            "sync_observe_source",
            question_text("question.sync_observe.source.prompt"),
            field="sync_observe_source",
            options=[
                {
                    "label": question_text(
                        "question.sync_observe.source.local_process"
                    ),
                    "value": "existing_local_node",
                    "expected_patch": {"sync_observe.source": "existing_local_node"},
                },
                {
                    "label": question_text(
                        "question.sync_observe.source.endpoint"
                    ),
                    "value": "endpoint_only",
                    "expected_patch": {"sync_observe.source": "endpoint_only"},
                },
                {
                    "label": question_text(
                        "question.sync_observe.source.client_setup"
                    ),
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
            question_text("question.sync_observe.after_client_setup.prompt"),
            field="sync_observe_after_client_setup",
            options=[
                {
                    "label": question_text(
                        "question.sync_observe.source.local_process"
                    ),
                    "value": "existing_local_node",
                    "expected_patch": {"sync_observe.source": "existing_local_node"},
                },
                {
                    "label": question_text(
                        "question.sync_observe.source.endpoint"
                    ),
                    "value": "endpoint_only",
                    "expected_patch": {"sync_observe.source": "endpoint_only"},
                },
            ],
        )
    if blocker.question_id == "sync_observe_stop_condition":
        return choice_question(
            "sync_observe",
            "sync_observe_stop_condition",
            question_text("question.sync_observe.stop_condition.prompt"),
            field="sync_observe_stop_condition",
            options=[
                {
                    "label": question_text(
                        "question.sync_observe.stop_condition.until_stopped"
                    ),
                    "value": "until_stopped",
                    "expected_patch": {"sync_observe.stop_condition": "until_stopped"},
                },
                {
                    "label": question_text(
                        "question.sync_observe.stop_condition.duration"
                    ),
                    "value": "duration",
                    "expected_patch": {"sync_observe.stop_condition": "duration"},
                },
                {
                    "label": question_text(
                        "question.sync_observe.stop_condition.until_synced"
                    ),
                    "value": "until_synced",
                    "expected_patch": {"sync_observe.stop_condition": "until_synced"},
                },
            ],
        )
    if blocker.question_id == "sync_observe_duration_seconds":
        return manual_question(
            "sync_observe",
            "sync_observe_duration_seconds",
            question_text("question.sync_observe.duration_seconds.prompt"),
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
    response_fragment = None
    stop = False
    next_group = ""
    if field in {"sync_observe_source", "sync_observe_after_client_setup"}:
        sync["source"] = str(value)
        if value in REAL_SOURCES:
            next_group = "endpoint_process"
        else:
            response_fragment = _client_setup_guidance(state)
            stop = True
    elif field == "sync_observe_stop_condition":
        sync["stop_condition"] = str(value)
    elif field == "sync_observe_duration_seconds":
        candidate = str(value).strip()
        if not re.fullmatch(r"[0-9]+", candidate) or int(candidate) <= 0:
            return HandlerResult(
                blocker=FailureDescriptor(
                    code="harness.sync_observe.failure.invalid_duration",
                    source=__name__,
                    retryable=True,
                )
            )
        sync["duration_seconds"] = int(candidate)
    return HandlerResult(
        delta=StateDelta.set_values({"sync_observe": sync}),
        clear_pending=True,
        next_group=next_group,
        response_fragments=(response_fragment,) if response_fragment else (),
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
    if action.action_type == "apply_reviewed_sync_observe_config":
        from .environment import apply_reviewed_owned_config_values

        return apply_reviewed_owned_config_values(
            state,
            action,
            owner="sync_observe",
        )
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
                return HandlerResult(
                    blocker=FailureDescriptor(
                        code="harness.sync_observe.failure.duration_condition_mismatch",
                        source=__name__,
                        retryable=True,
                    )
                )
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
        return HandlerResult(
            blocker=FailureDescriptor(
                code="harness.sync_observe.failure.unsupported_action",
                arguments={"action_type": action.action_type},
                source=__name__,
            )
        )
    source = str(action.arguments.get("sync_observe_source") or "").strip().lower()
    if source not in REAL_SOURCES | {"client_setup"}:
        return HandlerResult(
            blocker=FailureDescriptor(
                code="harness.sync_observe.failure.real_source_required",
                source=__name__,
                retryable=True,
            )
        )
    if state.get("workflow_mode") != "sync_observe":
        return HandlerResult(
            blocker=FailureDescriptor(
                code="harness.sync_observe.failure.mode_required",
                source=__name__,
            )
        )
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
        response_fragments=(
            (_client_setup_guidance(state),)
            if source == "client_setup"
            else ()
        ),
        stop_after_response=source == "client_setup",
    )


def _client_setup_guidance(state: AgentGraphState) -> ResponseFragment:
    if not bool((state.get("web_research") or {}).get("google_search_available")):
        return ResponseFragment(
            kind="message",
            message_id="harness.sync_observe.client_setup_without_search",
            source=__name__,
        )
    chain = str((state.get("chain_identity") or {}).get("canonical") or "blockchain").strip()
    result = run_google_search_grounding(f"{chain} official node client installation metrics documentation")
    summary = str(getattr(result, "text_summary", "") or "").strip()
    citations = [str(item) for item in (getattr(result, "citations", None) or []) if str(item).strip()]
    if not bool(getattr(result, "available", False)) or not summary:
        return ResponseFragment(
            kind="warning",
            message_id="harness.sync_observe.client_setup_search_unavailable",
            source=__name__,
        )
    evidence = summary
    if citations:
        evidence += "\n" + "\n".join(f"- {item}" for item in citations)
    return ResponseFragment(
        kind="evidence",
        message_id="harness.sync_observe.client_setup_grounded",
        arguments={"evidence": evidence},
        source=__name__,
    )
