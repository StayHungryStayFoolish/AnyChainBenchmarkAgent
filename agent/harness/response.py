"""Single response-composition authority for the AnyChain Harness."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Mapping

from .contracts import ResponseFragment, response_fragment_from_dict
from .invariants import validate_state
from .questions import render_question
from .response_catalog import render_fragment, render_hash, semantic_hash
from .state import AgentGraphState


def reset_turn_response(state: AgentGraphState) -> None:
    """Clear prior terminal output at the start of a new graph turn."""

    state["visible_response"] = []
    state["response_fragments"] = []


def finalize_turn_response(state: AgentGraphState) -> AgentGraphState:
    """Compose one deduplicated result with at most one actionable question."""

    responses: list[str] = []
    response_manifest: list[dict[str, str]] = []
    seen_semantics: set[str] = set()
    for item in state.get("response_fragments") or []:
        fragment = _response_fragment_from_state(item)
        rendered_fragment = render_fragment(fragment, state.get("language", "en"))
        if rendered_fragment.semantic_hash in seen_semantics:
            continue
        seen_semantics.add(rendered_fragment.semantic_hash)
        responses.append(rendered_fragment.text)
        response_manifest.append(
            {
                "semantic_hash": rendered_fragment.semantic_hash,
                "render_hash": rendered_fragment.render_hash,
                "role": rendered_fragment.kind,
                "message_id": rendered_fragment.message_id,
            }
        )

    turn_context = dict(state.get("turn_context") or {})
    pending = state.get("pending_question") or {}
    active_rendered = (
        render_question(pending, state.get("language", "en")).strip()
        if pending
        else ""
    )
    suppress_pending_render = bool(
        turn_context.pop("suppress_pending_render", False)
        or (
            state.get("evidence_collection")
            and str(
                (state.get("evidence_collection") or {}).get("status")
                or "active"
            )
            == "active"
        )
    )
    state["turn_context"] = turn_context
    if pending and not suppress_pending_render:
        pending_semantic_hash = semantic_hash(
            {
                "kind": "pending_question",
                "question": _plain_value(pending),
            }
        )
        if pending_semantic_hash not in seen_semantics:
            seen_semantics.add(pending_semantic_hash)
            responses.append(active_rendered)
            response_manifest.append(
                {
                    "semantic_hash": pending_semantic_hash,
                    "render_hash": render_hash(active_rendered),
                    "role": "pending_question",
                    "message_id": str(pending.get("id") or ""),
                }
            )
    terminal_response = "\n".join(responses).strip()
    state["visible_response"] = [terminal_response] if terminal_response else []
    state["response_fragments"] = []
    turn_context = dict(state.get("turn_context") or {})
    turn_context["response_manifest"] = response_manifest
    turn_context["terminal_semantic_hash"] = (
        semantic_hash(
            [item["semantic_hash"] for item in response_manifest]
        )
        if response_manifest
        else ""
    )
    turn_context["terminal_response_hash"] = (
        render_hash(terminal_response) if terminal_response else ""
    )
    state["turn_context"] = turn_context
    validate_state(state)
    return state


def _response_fragment_from_state(item: object) -> ResponseFragment:
    if isinstance(item, ResponseFragment):
        return item
    if not isinstance(item, Mapping):
        raise TypeError("response_fragments entries must be typed mappings")
    return response_fragment_from_dict(item)


def _plain_value(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value
