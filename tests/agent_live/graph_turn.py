"""Test adapter that executes the same compiled LangGraph used by the product."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from typing import Any, Mapping
from unittest.mock import patch

from agent.harness.graph import build_graph
from agent.harness.invariants import validate_state


def action_types(state: Mapping[str, Any], key: str = "action_queue") -> list[str]:
    """Project typed action envelopes without exposing their storage shape."""

    return [
        str(item.get("action_type") or item.get("type") or "")
        for item in state.get(key) or []
        if isinstance(item, Mapping)
    ]


def invoke_actions(
    state: Mapping[str, Any],
    actions: list[Mapping[str, Any]],
    text: str | None = None,
) -> dict[str, Any]:
    """Admit proposed actions through the product graph."""

    current = deepcopy(dict(state))
    if text is not None:
        current["last_user_input"] = text
    from agent.harness import coordinator

    with patch(
        "agent.harness.coordinator.resolve_action_queue",
        return_value={"actions": [deepcopy(dict(item)) for item in actions]},
    ):
        return invoke_product_graph_turn(current)


def invoke_action(
    state: Mapping[str, Any],
    action: Mapping[str, Any],
    text: str | None = None,
) -> dict[str, Any]:
    """Admit one proposed action through the product graph."""

    return invoke_actions(state, [action], text)


def answer_pending(
    state: Mapping[str, Any],
    answer: str,
    _question: Mapping[str, Any] | None = None,
    *,
    selected_value: Any = None,
    manual_value: Any = None,
) -> dict[str, Any]:
    """Answer the active question through deterministic graph admission."""

    current = deepcopy(dict(state))
    resolved = manual_value if manual_value is not None else selected_value
    current["last_user_input"] = str(resolved if resolved is not None else answer)
    return invoke_product_graph_turn(current)


def admitted_action_queue(
    state: Mapping[str, Any],
    actions: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build current durable envelopes for tests that seed deferred work."""

    from agent.harness.action_registry import ACTION_BY_TYPE
    from agent.harness.contracts import ActionEnvelope, action_envelope_to_dict
    from agent.harness.domains.registry import GROUP_OWNER

    pending = dict(state.get("pending_question") or {})
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(actions):
        action = dict(raw)
        action_type = str(action.get("type") or "")
        spec = ACTION_BY_TYPE[action_type]
        owner = (
            GROUP_OWNER.get(str(pending.get("group") or ""), spec.owner)
            if action_type == "answer_pending"
            else spec.owner
        )
        arguments = {
            name: action[name]
            for name in spec.arguments
            if name in action
        }
        envelope = ActionEnvelope(
            action_id=str(action.get("action_id") or f"test-action-{index}"),
            action_type=action_type,
            owner=owner,
            target_group=spec.target_group,
            arguments=arguments,
            confidence=str(action.get("confidence") or "high"),
            reason=str(action.get("reason") or ""),
            semantic_order=index,
            execution_order=index,
            submitted_turn_index=int(state.get("turn_index") or 0),
            origin_group=str(state.get("active_group") or ""),
            origin_text=str(
                action.get("_origin_text")
                or action.get("source_evidence")
                or ""
            ),
            plan_scope="test-fixture",
            effect_kind=(
                "external"
                if spec.effect == "execution"
                else "read_only"
                if spec.effect == "read_only"
                else "pure"
            ),
            idempotency_key=f"test-fixture:{index}:{action_type}",
        )
        output.append(action_envelope_to_dict(envelope))
    return output


def invoke_product_graph_turn(
    state: Mapping[str, Any],
    *,
    allow_semantic_resolver: bool = False,
) -> dict[str, Any]:
    """Run one deterministic uncheckpointed turn through the product graph.

    Contract and unit evidence must never depend on a live model response. A
    caller exercising semantic input has to inject the reviewed resolver it is
    testing; real provider behavior belongs to the PTY/CLI acceptance lanes.
    """

    from agent.harness.domains.rpc_catalog import migrate_legacy_catalog
    from agent.harness.state import (
        STATE_SCHEMA_VERSION,
        migrate_state,
    )
    from agent.harness import coordinator, intent
    from agent.harness import hierarchical_planner
    from agent.harness.domains import analysis, chain_identity, recovery, rpc_endpoint

    current = deepcopy(dict(state))
    migrate_legacy_catalog(current)
    if int(current.get("schema_version") or 0) < STATE_SCHEMA_VERSION:
        current = migrate_state(
            current,
            thread_id=str(current.get("thread_id") or "test-fixture"),
            language=str(current.get("language") or "en"),
            session_purpose=str((current.get("session") or {}).get("purpose") or "test"),
        )
    guarded_entries = (
        (
            "agent.harness.coordinator.resolve_action_queue",
            coordinator.resolve_action_queue,
            hierarchical_planner.resolve_product_action_queue,
        ),
        (
            "agent.harness.domains.chain_identity.resolve_unknown_chain_identity",
            chain_identity.resolve_unknown_chain_identity,
            intent.resolve_unknown_chain_identity,
        ),
        (
            "agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence",
            rpc_endpoint.extract_rpc_schema_from_evidence,
            intent.extract_rpc_schema_from_evidence,
        ),
        (
            "agent.harness.domains.analysis.analyze_evidence_with_model",
            analysis.analyze_evidence_with_model,
            intent.analyze_evidence_with_model,
        ),
        (
            "agent.harness.domains.recovery.analyze_evidence_with_model",
            recovery.analyze_evidence_with_model,
            intent.analyze_evidence_with_model,
        ),
    )
    with ExitStack() as stack:
        for index, (target, active, original) in enumerate(guarded_entries):
            if active is original and not (
                index == 0 and allow_semantic_resolver
            ):
                stack.enter_context(patch(
                    target,
                    side_effect=AssertionError(
                        "deterministic graph turn attempted to call a live model entry"
                    ),
                ))
        invocation_context = {
            key: deepcopy(current.get(key) or {})
            for key in ("discovery", "framework_summary", "web_research")
        }
        result = dict(
            build_graph(None).invoke(
                current,
                context=invocation_context,
            )
        )
    validate_state(result)
    return result
