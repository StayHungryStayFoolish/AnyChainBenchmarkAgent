"""Authoritative response-driven Journey contracts for the formal Chaos profile."""

from __future__ import annotations

from typing import Any, Mapping

from tests.agent_live.dynamic_dual_ai_chaos import (
    JourneyPostconditionResult,
    JourneyPostconditionVerifierDefinition,
    JourneyVerifierContext,
    build_journey_outcome_verifier_registry,
)


REGISTRY_IMPORT = (
    "tests.agent_live.formal_journey_catalog:FORMAL_JOURNEY_VERIFIER_REGISTRY"
)


def _result(
    postcondition_id: str,
    satisfied: bool,
    context: JourneyVerifierContext,
    **details: Any,
) -> JourneyPostconditionResult:
    return JourneyPostconditionResult(
        postcondition_id,
        bool(satisfied),
        {
            "initial_turn": context.initial_event.turn_index,
            "current_turn": context.current_event.turn_index,
            "active_group": context.current_event.active_group,
            **details,
        },
    )


def committed_state(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    return _result(
        "committed_state",
        context.current_event.turn_index > context.initial_event.turn_index
        and context.current_event.after_fingerprint != context.initial_event.after_fingerprint,
        context,
    )


def group_changed(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    return _result(
        "group_changed",
        bool(context.current_event.active_group)
        and context.current_event.active_group != context.initial_event.active_group,
        context,
        initial_group=context.initial_event.active_group,
    )


def pending_advanced(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    return _result(
        "pending_advanced",
        context.current_event.pending_question_id
        != context.initial_event.pending_question_id,
        context,
        initial_question=context.initial_event.pending_question_id,
        current_question=context.current_event.pending_question_id,
    )


def multiple_actions_admitted(
    context: JourneyVerifierContext,
) -> JourneyPostconditionResult:
    actions = tuple(context.current_event.admitted_action_types)
    return _result(
        "multiple_actions_admitted",
        len(actions) >= 2,
        context,
        admitted_action_types=list(actions),
    )


def rpc_action_admitted(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    actions = tuple(context.current_event.admitted_action_types)
    return _result(
        "rpc_action_admitted",
        any("rpc" in action.lower() for action in actions),
        context,
        admitted_action_types=list(actions),
    )


def result_returned(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    result = context.current_event.next_result
    return _result(
        "result_returned",
        isinstance(result, Mapping) and str(result.get("kind") or "") == "result",
        context,
        result_kind=str(result.get("kind") or "") if isinstance(result, Mapping) else "",
    )


def mechanical_response_loop(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    responses = [agent.strip() for _, agent in context.transcript if agent.strip()]
    repeated = len(responses) >= 2 and responses[-1] == responses[-2]
    return _result("mechanical_response_loop", repeated, context)


def state_regressed(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    return _result(
        "state_regressed",
        bool(context.completed_turns)
        and context.current_event.turn_index <= context.initial_event.turn_index,
        context,
    )


FORMAL_JOURNEY_VERIFIER_REGISTRY = build_journey_outcome_verifier_registry((
    JourneyPostconditionVerifierDefinition(
        "committed_state", "formal-committed-state", 1,
        "A response-driven turn committed a new checkpoint state.", committed_state,
    ),
    JourneyPostconditionVerifierDefinition(
        "group_changed", "formal-group-changed", 1,
        "The active configuration group changed after the user redirected the flow.",
        group_changed,
    ),
    JourneyPostconditionVerifierDefinition(
        "pending_advanced", "formal-pending-advanced", 1,
        "The pending question advanced after a valid response.", pending_advanced,
    ),
    JourneyPostconditionVerifierDefinition(
        "multiple_actions_admitted", "formal-multiple-actions", 1,
        "One natural-language turn admitted at least two typed actions.",
        multiple_actions_admitted,
    ),
    JourneyPostconditionVerifierDefinition(
        "rpc_action_admitted", "formal-rpc-action", 1,
        "The runtime admitted a typed RPC configuration action.", rpc_action_admitted,
    ),
    JourneyPostconditionVerifierDefinition(
        "result_returned", "formal-result-returned", 1,
        "The runtime returned a committed non-question result.", result_returned,
    ),
    JourneyPostconditionVerifierDefinition(
        "mechanical_response_loop", "formal-mechanical-loop", 1,
        "Detects consecutive identical Agent responses.", mechanical_response_loop,
    ),
    JourneyPostconditionVerifierDefinition(
        "state_regressed", "formal-state-regression", 1,
        "Detects a turn that failed to advance beyond the initial checkpoint.",
        state_regressed,
    ),
))


def formal_journey_definitions() -> tuple[dict[str, Any], ...]:
    """Return eight open missions; no future user message is pre-generated."""

    rows = (
        ("orientation", "new operator", "Ask what the Agent can do, then begin a valid workflow.", "committed_state", ("ask_question",)),
        ("group-jump", "impatient operator", "Interrupt a partial group and move to a different configuration group.", "group_changed", ("jump_group", "change_direction")),
        ("pending-correction", "operator correcting a mistake", "Correct the current pending value and continue without losing prior state.", "pending_advanced", ("correct_value", "backtrack")),
        ("multi-intent", "expert operator", "Express multiple configuration changes in one natural-language turn.", "multiple_actions_admitted", ("multi_intent", "change_direction")),
        ("custom-rpc", "protocol engineer", "Add or revise a custom RPC method from incomplete request and response evidence.", "rpc_action_admitted", ("multiline_evidence", "incomplete_data")),
        ("history-question", "returning operator", "Ask about saved configuration or job history and obtain a concrete result.", "result_returned", ("ask_question", "resume_session")),
        ("language-switch", "bilingual operator", "Switch language while preserving the current configuration task.", "committed_state", ("language_switch",)),
        ("mode-switch", "performance engineer", "Switch between RPC benchmark and sync observation, preserving only compatible state.", "group_changed", ("change_direction", "mode_switch")),
    )
    forbidden = (
        {"outcome_id": "mechanical-loop", "required_postcondition_ids": ["mechanical_response_loop"]},
        {"outcome_id": "state-regression", "required_postcondition_ids": ["state_regressed"]},
    )
    return tuple({
        "lane": "journey",
        "verifier_registry": REGISTRY_IMPORT,
        "journey": {
            "journey_id": journey_id,
            "start_scenario": "opening",
            "persona": persona,
            "mission": mission,
            "allowed_risk_factors": list(risks),
            "max_turns": 12,
            "terminal_outcome": {
                "outcome_id": f"{journey_id}-complete",
                "required_postcondition_ids": [postcondition],
            },
            "forbidden_outcomes": [dict(item) for item in forbidden],
        },
    } for journey_id, persona, mission, postcondition, risks in rows)
