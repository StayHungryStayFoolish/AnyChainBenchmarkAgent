"""Single response-composition authority for the AnyChain Harness."""

from __future__ import annotations

from .invariants import validate_state
from .questions import render_question
from .state import AgentGraphState, PendingQuestion


def append_active_question_once(state: AgentGraphState) -> None:
    pending = state.get("pending_question") or {}
    if not pending:
        return
    rendered = render_question(pending, state.get("language", "en"))
    responses = list(state.get("visible_response") or [])
    if rendered not in responses and not question_is_actionable_in_responses(responses, pending):
        responses.append(rendered)
    state["visible_response"] = responses


def question_is_actionable_in_responses(
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


def finalize_turn_response(state: AgentGraphState) -> AgentGraphState:
    """Compose one deduplicated result with at most one actionable question."""

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
    active_rendered = (
        render_question(pending, state.get("language", "en")).strip()
        if pending
        else ""
    )
    for installed in turn_context.get("installed_questions") or []:
        if not isinstance(installed, dict):
            continue
        installed_rendered = render_question(
            installed,
            state.get("language", "en"),
        ).strip()
        if not installed_rendered or installed_rendered == active_rendered:
            continue
        responses = without_superseded_question(
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
            if question_is_actionable_in_responses([response], pending)
        ]
        if not actionable:
            responses.append(render_question(pending, state.get("language", "en")))
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


def without_superseded_question(
    responses: list[str],
    previous_pending: PendingQuestion,
    language: str,
) -> list[str]:
    """Remove only a superseded question while retaining domain evidence."""

    if not previous_pending:
        return responses
    candidates = {
        str(previous_pending.get("prompt") or "").strip(),
        render_question(previous_pending, language).strip(),
    }
    candidates.discard("")
    return [response for response in responses if str(response).strip() not in candidates]
