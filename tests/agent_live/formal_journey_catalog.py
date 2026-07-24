"""Authoritative response-driven Journey contracts for the formal Chaos profile."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

from agent.harness.action_registry import ACTION_BY_TYPE
from tests.agent_live.dynamic_dual_ai_chaos import (
    JourneyPostconditionResult,
    JourneyPostconditionVerifierDefinition,
    JourneyVerifierContext,
    build_journey_outcome_verifier_registry,
)


REGISTRY_IMPORT = (
    "tests.agent_live.formal_journey_catalog:FORMAL_JOURNEY_VERIFIER_REGISTRY"
)

_NON_DOMAIN_STATE_ROOTS = frozenset({
    "turn_index",
    "turn_context",
    "visible_response",
})
_TERMINAL_PREFIX_RE = re.compile(r"^\s*(?:Agent|User)>\s*", re.IGNORECASE)
_THINKING_FRAME_RE = re.compile(r"^\s*\[thinking\]\s*", re.IGNORECASE)
_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


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


def _concrete_diff(context: JourneyVerifierContext) -> dict[str, dict[str, str]]:
    return {
        str(path): {"before": str(hashes.get("before") or ""), "after": str(hashes.get("after") or "")}
        for path, hashes in context.current_event.state_diff_hashes.items()
        if isinstance(hashes, Mapping)
        and str(hashes.get("before") or "") != str(hashes.get("after") or "")
    }


def _material_diff(context: JourneyVerifierContext) -> dict[str, dict[str, str]]:
    return {
        path: hashes
        for path, hashes in _concrete_diff(context).items()
        if path.split(".", 1)[0] not in _NON_DOMAIN_STATE_ROOTS
    }


def _registered_admitted_actions(context: JourneyVerifierContext) -> tuple[Any, ...]:
    action_types = tuple(context.current_event.admitted_action_types)
    if not action_types or len(action_types) != len(set(action_types)):
        return ()
    specs = tuple(ACTION_BY_TYPE.get(action_type) for action_type in action_types)
    return specs if all(spec is not None for spec in specs) else ()


def _turn_chain_errors(context: JourneyVerifierContext) -> tuple[str, ...]:
    errors: list[str] = []
    event = context.current_event
    turns = context.completed_turns
    latest = context.latest_turn
    if event.event_type != "turn_committed":
        errors.append("current runtime event is not a committed turn")
    if not turns or latest is None or turns[-1] != latest:
        errors.append("current event is not bound to the latest completed PTY turn")
        return tuple(errors)
    if event.turn_index != context.initial_event.turn_index + len(turns):
        errors.append("runtime turn count is not contiguous from the initial checkpoint")
    previous_after = context.initial_event.after_fingerprint
    expected_index = context.initial_event.turn_index + 1
    for turn in turns:
        if turn.turn_index != expected_index:
            errors.append("PTY turn indexes are not contiguous")
            break
        if turn.before_fingerprint != previous_after:
            errors.append("PTY fingerprint chain is broken")
            break
        previous_after = turn.after_fingerprint
        expected_index += 1
    if (
        latest.turn_index != event.turn_index
        or latest.before_fingerprint != event.before_fingerprint
        or latest.after_fingerprint != event.after_fingerprint
    ):
        errors.append("runtime event identity does not match the latest PTY turn")
    if not context.transcript or context.transcript[-1] != (
        latest.user_message,
        latest.agent_response,
    ):
        errors.append("latest PTY turn is not bound to the transcript tail")
    if not (
        latest.previous_response_received_at_ns
        <= latest.user_message_submitted_at_ns
        <= latest.agent_response_received_at_ns
    ):
        errors.append("PTY turn timestamps are not monotonic")
    if event.before_fingerprint == event.after_fingerprint and _concrete_diff(context):
        errors.append("state diff contradicts an unchanged runtime fingerprint")
    if event.before_fingerprint != event.after_fingerprint and not _concrete_diff(context):
        errors.append("changed runtime fingerprint has no concrete state diff")
    if str(event.pending_contract.get("id") or "") != event.pending_question_id:
        errors.append("pending contract identity does not match the runtime event")
    return tuple(errors)


def _next_result_errors(context: JourneyVerifierContext) -> tuple[str, ...]:
    result = context.current_event.next_result
    if not isinstance(result, Mapping):
        return ("next_result is not a mapping",)
    kind = str(result.get("kind") or "")
    if kind not in {"question", "result"}:
        return ("next_result has no supported kind",)
    question_id = str(result.get("question_id") or "")
    pending_id = context.current_event.pending_question_id
    if kind == "question" and (not pending_id or question_id != pending_id):
        return ("question result is not bound to the active pending contract",)
    if kind == "result" and question_id and question_id != pending_id:
        return ("result overlay is not bound to the active pending contract",)
    return ()


def _provenance_errors(context: JourneyVerifierContext) -> tuple[str, ...]:
    action_types = tuple(context.current_event.admitted_action_types)
    specs = _registered_admitted_actions(context)
    errors: list[str] = []
    if not specs:
        errors.append("admitted actions are empty, duplicated, or absent from the registry")
    for target in context.current_event.admitted_action_targets:
        action_type = str(target.get("type") or "")
        group = str(target.get("group") or "")
        if not action_type or action_type not in action_types or not group:
            errors.append("admitted action target is not bound to an admitted action")
    return tuple(errors)


def _strict_transition_errors(context: JourneyVerifierContext) -> tuple[str, ...]:
    return (
        *_turn_chain_errors(context),
        *_next_result_errors(context),
        *_provenance_errors(context),
    )


def _normalized_response(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", _ANSI_RE.sub("", str(value or "")))
    lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = _TERMINAL_PREFIX_RE.sub("", raw_line)
        if _THINKING_FRAME_RE.match(line):
            continue
        line = " ".join(line.split()).casefold()
        if line:
            lines.append(line)
    return "\n".join(lines)


def committed_state(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    errors = list(_strict_transition_errors(context))
    material = _material_diff(context)
    mutating_actions = tuple(
        spec.action_type
        for spec in _registered_admitted_actions(context)
        if spec.effect != "read_only"
    )
    if not material:
        errors.append("no material state path changed")
    if not mutating_actions:
        errors.append("no registered state-changing action was admitted")
    return _result(
        "committed_state",
        not errors,
        context,
        errors=errors,
        material_paths=sorted(material),
        admitted_mutating_actions=list(mutating_actions),
    )


def group_changed(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    errors = list(_strict_transition_errors(context))
    initial_group = context.initial_event.active_group
    current_group = context.current_event.active_group
    group_diff = {
        path: hashes
        for path, hashes in _concrete_diff(context).items()
        if path == "active_group" or path.startswith("control.")
    }
    specs = _registered_admitted_actions(context)
    targeted_groups = {
        str(target.get("group") or "")
        for target in context.current_event.admitted_action_targets
        if str(target.get("type") or "") in context.current_event.admitted_action_types
    }
    action_supports_transition = any(
        spec.effect == "workflow_navigation"
        or spec.mutation_dimension in {"target_mode", "chain"}
        or bool(spec.target_group)
        for spec in specs
    )
    if not initial_group or not current_group or initial_group == current_group:
        errors.append("active group did not change")
    if not group_diff:
        errors.append("group transition has no concrete control-plane state diff")
    if not action_supports_transition:
        errors.append("no admitted action declares navigation or workflow transition semantics")
    if targeted_groups and current_group not in targeted_groups:
        errors.append("visible group does not match the admitted navigation destination")
    return _result(
        "group_changed",
        not errors,
        context,
        errors=errors,
        initial_group=initial_group,
        current_group=current_group,
        group_diff_paths=sorted(group_diff),
        admitted_target_groups=sorted(targeted_groups),
    )


def pending_advanced(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    errors = list(_strict_transition_errors(context))
    initial_question = context.initial_event.pending_question_id
    current_question = context.current_event.pending_question_id
    pending_diff = {
        path: hashes
        for path, hashes in _concrete_diff(context).items()
        if path == "pending_question" or path.startswith("pending_question.")
    }
    accepted = {
        str(item)
        for item in context.initial_event.pending_contract.get("accepted_action_types") or ()
        if str(item)
    }
    admitted = set(context.current_event.admitted_action_types)
    initial_id_hash = context.initial_event.after_value_hashes.get("pending_question.id", "")
    consumed_initial = any(
        path == "pending_question.id"
        and (not initial_id_hash or hashes["before"] == initial_id_hash)
        for path, hashes in pending_diff.items()
    )
    if not initial_question:
        errors.append("initial checkpoint had no pending contract")
    if current_question == initial_question:
        errors.append("initial pending contract was not advanced")
    if not pending_diff or not consumed_initial:
        errors.append("pending lifecycle is not proven by a concrete diff from the initial contract")
    if accepted and not accepted.intersection(admitted):
        errors.append("admitted action was not accepted by the initial pending contract")
    return _result(
        "pending_advanced",
        not errors,
        context,
        errors=errors,
        initial_question=initial_question,
        current_question=current_question,
        pending_diff_paths=sorted(pending_diff),
        admitted_accepted_actions=sorted(accepted.intersection(admitted)),
    )


def multiple_actions_admitted(
    context: JourneyVerifierContext,
) -> JourneyPostconditionResult:
    errors = list(_strict_transition_errors(context))
    specs = _registered_admitted_actions(context)
    actions = tuple(spec.action_type for spec in specs if spec.effect != "read_only")
    material_roots = {
        path.split(".", 1)[0]
        for path in _material_diff(context)
    }
    if len(actions) < 2:
        errors.append("fewer than two registered state-changing actions were admitted")
    if len(material_roots) < 2:
        errors.append("multi-action turn did not produce changes in two state domains")
    return _result(
        "multiple_actions_admitted",
        not errors,
        context,
        errors=errors,
        admitted_action_types=list(actions),
        material_state_roots=sorted(material_roots),
    )


def rpc_action_admitted(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    errors = list(_strict_transition_errors(context))
    actions = tuple(context.current_event.admitted_action_types)
    catalog_paths = {
        path
        for path in _material_diff(context)
        if path == "custom_rpc" or path.startswith("custom_rpc.")
    }
    if "rpc_catalog_command" not in actions:
        errors.append("the exact registered RPC catalog action was not admitted")
    if not catalog_paths:
        errors.append("RPC catalog admission produced no concrete custom_rpc state transition")
    return _result(
        "rpc_action_admitted",
        not errors,
        context,
        errors=errors,
        admitted_action_types=list(actions),
        catalog_diff_paths=sorted(catalog_paths),
    )


def result_returned(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    errors = list(_strict_transition_errors(context))
    result = context.current_event.next_result
    response_count = int(result.get("response_count") or 0) if isinstance(result, Mapping) else 0
    specs = _registered_admitted_actions(context)
    normalized_response = _normalized_response(
        context.latest_turn.agent_response if context.latest_turn is not None else ""
    )
    if not isinstance(result, Mapping) or str(result.get("kind") or "") != "result":
        errors.append("runtime did not return a result")
    if response_count <= 0 or not normalized_response:
        errors.append("result has no observed user-visible response")
    if not any(spec.effect == "read_only" for spec in specs):
        errors.append("result is not attributable to a registered read-only action")
    return _result(
        "result_returned",
        not errors,
        context,
        errors=errors,
        result_kind=str(result.get("kind") or "") if isinstance(result, Mapping) else "",
        response_count=response_count,
    )


def mechanical_response_loop(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    responses = [
        normalized
        for _, agent in context.transcript
        if (normalized := _normalized_response(agent))
    ]
    adjacent_repeat = len(responses) >= 2 and responses[-1] == responses[-2]
    short_cycle = (
        len(responses) >= 4
        and responses[-4] == responses[-2]
        and responses[-3] == responses[-1]
    )
    return _result(
        "mechanical_response_loop",
        adjacent_repeat or short_cycle,
        context,
        normalized_response_count=len(responses),
        loop_kind="adjacent" if adjacent_repeat else "short_cycle" if short_cycle else "",
    )


def state_regressed(context: JourneyVerifierContext) -> JourneyPostconditionResult:
    if not context.completed_turns:
        return _result(
            "state_regressed",
            False,
            context,
            chain_errors=[],
            exact_prior_checkpoint_rollback=False,
            lifecycle="initial_checkpoint",
        )
    chain_errors = _turn_chain_errors(context)
    prior_fingerprints = {
        fingerprint
        for turn in context.completed_turns[:-1]
        for fingerprint in (turn.before_fingerprint, turn.after_fingerprint)
        if fingerprint
    }
    exact_rollback = bool(
        context.latest_turn is not None
        and context.current_event.before_fingerprint != context.current_event.after_fingerprint
        and context.current_event.after_fingerprint in prior_fingerprints
    )
    return _result(
        "state_regressed",
        bool(chain_errors) or exact_rollback,
        context,
        chain_errors=list(chain_errors),
        exact_prior_checkpoint_rollback=exact_rollback,
        lifecycle="committed_turn",
    )


FORMAL_JOURNEY_VERIFIER_REGISTRY = build_journey_outcome_verifier_registry((
    JourneyPostconditionVerifierDefinition(
        "committed_state", "formal-committed-state", 2,
        "A response-driven turn committed a concrete domain state transition through a registered action.", committed_state,
    ),
    JourneyPostconditionVerifierDefinition(
        "group_changed", "formal-group-changed", 2,
        "A registered navigation or workflow action produced a concrete control-plane group transition.",
        group_changed,
    ),
    JourneyPostconditionVerifierDefinition(
        "pending_advanced", "formal-pending-advanced", 2,
        "An action accepted by the original pending contract consumed it and established the next lifecycle state.", pending_advanced,
    ),
    JourneyPostconditionVerifierDefinition(
        "multiple_actions_admitted", "formal-multiple-actions", 2,
        "One natural-language turn admitted multiple registered state-changing actions across multiple state domains.",
        multiple_actions_admitted,
    ),
    JourneyPostconditionVerifierDefinition(
        "rpc_action_admitted", "formal-rpc-action", 2,
        "The runtime admitted the exact catalog action and committed a custom RPC catalog transition.", rpc_action_admitted,
    ),
    JourneyPostconditionVerifierDefinition(
        "result_returned", "formal-result-returned", 2,
        "A registered read-only action returned an observed user-visible result.", result_returned,
    ),
    JourneyPostconditionVerifierDefinition(
        "mechanical_response_loop", "formal-mechanical-loop", 2,
        "Detects normalized adjacent repetition or a normalized two-response cycle.", mechanical_response_loop,
    ),
    JourneyPostconditionVerifierDefinition(
        "state_regressed", "formal-state-regression", 2,
        "Detects broken PTY/runtime lineage or an exact rollback to a prior checkpoint.",
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
