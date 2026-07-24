"""Authoritative response-driven Journey contracts for the formal Chaos profile."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from typing import Any, Mapping, Sequence

from agent.harness.action_registry import ACTION_BY_TYPE
from agent.harness.control_receipts import (
    validate_persisted_domain_control_receipt,
)
from agent.harness.plan_coverage import segment_user_turn
from agent.workflows.group_registry import GROUP_ORDER
from tests.agent_live.dynamic_dual_ai_chaos import (
    JourneyPostconditionResult,
    JourneyPostconditionVerifierDefinition,
    JourneyVerifierContext,
    build_journey_outcome_verifier_registry,
)


REGISTRY_IMPORT = (
    "tests.agent_live.formal_journey_catalog:FORMAL_JOURNEY_VERIFIER_REGISTRY"
)

FACTOR_OBSERVATION_VALUES: Mapping[str, tuple[str, ...]] = {
    "workflow_mode": ("fake", "real", "sync"),
    "subject_group": GROUP_ORDER,
    "language": ("en", "zh"),
    "session_state": ("fresh", "partial", "complete", "quarantine"),
    "pending_state": ("none", "manual", "choice"),
    "group_state": ("partial", "completed", "invalidated"),
    "interruption_depth": ("0", "1", "2+"),
    "input_shape": ("exact", "natural", "multiline", "structured", "contradictory"),
    "chain_case": ("known", "case1", "case2", "case3"),
    "workload": (
        "default_single",
        "default_mixed",
        "custom_single",
        "custom_mixed",
        "not_applicable",
    ),
    "evidence_shape": ("none", "request", "response", "split", "docs"),
    "recovery": ("none", "back", "jump", "correct", "retry", "reset"),
}


def factor_observation_postcondition_id(name: str, value: str) -> str:
    if value not in FACTOR_OBSERVATION_VALUES.get(name, ()):
        raise ValueError(f"unknown product Chaos factor observation: {name}={value}")
    return f"factor_observed__{name}__{value.replace('+', 'plus')}"

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


def _journey_material_diff(
    context: JourneyVerifierContext,
) -> dict[str, dict[str, str]]:
    material: dict[str, dict[str, str]] = {}
    for event in context.completed_events or (context.current_event,):
        for path, hashes in event.state_diff_hashes.items():
            if (
                isinstance(hashes, Mapping)
                and str(hashes.get("before") or "")
                != str(hashes.get("after") or "")
                and path.split(".", 1)[0] not in _NON_DOMAIN_STATE_ROOTS
            ):
                material[str(path)] = {
                    "before": str(hashes.get("before") or ""),
                    "after": str(hashes.get("after") or ""),
                }
    return material


def _registered_admitted_actions(context: JourneyVerifierContext) -> tuple[Any, ...]:
    action_types = tuple(context.current_event.admitted_action_types)
    if not action_types or len(action_types) != len(set(action_types)):
        return ()
    specs = tuple(ACTION_BY_TYPE.get(action_type) for action_type in action_types)
    return specs if all(spec is not None for spec in specs) else ()


def _journey_registered_actions(
    context: JourneyVerifierContext,
) -> tuple[Any, ...]:
    action_types = tuple(dict.fromkeys(
        str(action)
        for event in context.completed_events or (context.current_event,)
        for action in event.admitted_action_types
        if str(action)
    ))
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
    material = _journey_material_diff(context)
    mutating_actions = tuple(
        spec.action_type
        for spec in _journey_registered_actions(context)
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
        for path, hashes in _journey_material_diff(context).items()
        if path == "active_group" or path.startswith("control.")
    }
    specs = _journey_registered_actions(context)
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
    pending_diffs = [
        (path, hashes)
        for event in context.completed_events or (context.current_event,)
        for path, hashes in event.state_diff_hashes.items()
        if path == "pending_question" or path.startswith("pending_question.")
    ]
    pending_diff = {path: dict(hashes) for path, hashes in pending_diffs}
    accepted = {
        str(item)
        for item in context.initial_event.pending_contract.get("accepted_action_types") or ()
        if str(item)
    }
    admitted = {
        str(action)
        for event in context.completed_events or (context.current_event,)
        for action in event.admitted_action_types
    }
    initial_id_hash = context.initial_event.after_value_hashes.get("pending_question.id", "")
    consumed_initial = any(
        path == "pending_question.id"
        and (not initial_id_hash or hashes["before"] == initial_id_hash)
        for path, hashes in pending_diffs
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
    specs = _journey_registered_actions(context)
    actions = tuple(spec.action_type for spec in specs if spec.effect != "read_only")
    material_roots = {
        path.split(".", 1)[0]
        for path in _journey_material_diff(context)
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
    actions = tuple(
        str(action)
        for event in context.completed_events or (context.current_event,)
        for action in event.admitted_action_types
    )
    catalog_paths = {
        path
        for path in _journey_material_diff(context)
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


def _value_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _all_events(context: JourneyVerifierContext) -> tuple[Any, ...]:
    return (context.initial_event, *context.completed_events)


def _has_state_value(
    context: JourneyVerifierContext,
    path: str,
    *values: Any,
) -> bool:
    expected = {_value_hash(value) for value in values}
    return any(
        str(event.after_value_hashes.get(path) or "") in expected
        for event in _all_events(context)
    )


def _event_has_state_value(event: Any, path: str, *values: Any) -> bool:
    expected = {_value_hash(value) for value in values}
    return str(event.after_value_hashes.get(path) or "") in expected


def _changed_path(context: JourneyVerifierContext, *prefixes: str) -> bool:
    return any(
        any(path == prefix or path.startswith(prefix + ".") for prefix in prefixes)
        for event in context.completed_events
        for path in event.material_state_diff_hashes
    )


def _validated_receipts(
    context: JourneyVerifierContext,
    *receipt_types: str,
) -> tuple[tuple[Any, Mapping[str, Any]], ...]:
    accepted: list[tuple[Any, Mapping[str, Any]]] = []
    allowed = set(receipt_types)
    for event in context.completed_events:
        for raw in event.control_receipts:
            if not isinstance(raw, Mapping):
                continue
            receipt_type = str(raw.get("receipt_type") or "")
            if allowed and receipt_type not in allowed:
                continue
            valid, _ = validate_persisted_domain_control_receipt(
                raw,
                turn_index=int(event.turn_index),
            )
            if valid:
                accepted.append((event, raw))
    return tuple(accepted)


def _domain_commits(
    context: JourneyVerifierContext,
    *,
    owner: str = "",
    navigation: str = "",
) -> tuple[tuple[Any, Mapping[str, Any]], ...]:
    return tuple(
        (event, receipt)
        for event, receipt in _validated_receipts(context, "domain_commit")
        if (not owner or receipt.get("owner") == owner)
        and (not navigation or receipt.get("navigation_operation") == navigation)
        and receipt.get("completion") in {"in_progress", "completed"}
        and bool(receipt.get("material_delta"))
    )


def _receipt_writes(
    receipt: Mapping[str, Any],
    prefix: str,
) -> bool:
    return any(
        item.get("operation") == "write"
        and (
            str(item.get("path") or "") == prefix
            or str(item.get("path") or "").startswith(prefix + ".")
        )
        for item in receipt.get("material_delta") or ()
        if isinstance(item, Mapping)
    )


def _receipt_changes(
    receipt: Mapping[str, Any],
    prefix: str,
) -> bool:
    return any(
        item.get("operation") in {"write", "delete"}
        and (
            str(item.get("path") or "") == prefix
            or str(item.get("path") or "").startswith(prefix + ".")
        )
        for item in receipt.get("material_delta") or ()
        if isinstance(item, Mapping)
    )


@lru_cache(maxsize=None)
def _seed_classification(scenario_id: str, factor_name: str) -> dict[str, Any]:
    from tests.agent_live.runtime_checkpoint import reviewed_scenario

    scenario = reviewed_scenario(scenario_id)
    state = dict(scenario.seed_state)
    question = dict(state.get("pending_question") or {})
    statuses = {
        str(item.get("status") or "")
        for item in (state.get("group_states") or {}).values()
        if isinstance(item, Mapping) and str(item.get("status") or "")
    }
    if factor_name == "pending_state":
        basis = {
            "pending_question_id": str(question.get("id") or ""),
            "has_options": bool(question.get("options")),
            "manual_input_allowed": question.get("manual_input_allowed") is True,
            "question_present": bool(question),
        }
        value = (
            "choice"
            if basis["has_options"]
            else "manual"
            if basis["manual_input_allowed"]
            else "none"
            if not basis["question_present"]
            else ""
        )
    elif factor_name == "group_state":
        basis = {"group_statuses": sorted(statuses)}
        value = (
            "invalidated"
            if statuses == {"invalidated"}
            else "completed"
            if statuses == {"completed"}
            else "partial"
            if statuses and statuses <= {"in_progress", "reconfiguring"}
            else ""
        )
    elif factor_name == "session_state":
        quarantine = dict(state.get("checkpoint_recovery") or {}).get("status")
        configured = bool(state.get("confirmed_config"))
        completion_roots = {
            "confirmed_config": configured,
            "target_mode": bool(state.get("target_mode")),
            "workflow_mode": bool(state.get("workflow_mode")),
            "chain_identity": bool(state.get("chain_identity")),
            "workload": bool(state.get("workload")),
            "qps_profile": bool(state.get("qps_profile")),
        }
        completed = bool(
            configured
            and completion_roots["target_mode"]
            and completion_roots["workflow_mode"]
            and completion_roots["chain_identity"]
            and (
                state.get("workflow_mode") == "sync_observe"
                or completion_roots["workload"] and completion_roots["qps_profile"]
            )
        )
        material_roots = (
            "confirmed_config", "group_states", "chain_identity", "target_mode",
            "workflow_mode", "workload", "qps_profile", "sync_observe",
        )
        basis = {
            "checkpoint_recovery_status": str(quarantine or ""),
            "completion_roots": completion_roots,
            "material_roots_present": {
                root: bool(state.get(root)) for root in material_roots
            },
        }
        value = (
            "quarantine"
            if quarantine == "quarantined"
            else "complete"
            if completed
            else "partial"
            if any(state.get(root) for root in material_roots)
            else "fresh"
        )
    else:
        raise ValueError(f"{factor_name} is not a seed factor")
    proof = {
        "proof_type": "reviewed_seed_classification",
        "proof_version": 1,
        "classification_rule_id": f"{factor_name}-v1",
        "scenario_id": scenario_id,
        "scenario_state_fingerprint": scenario.state_fingerprint,
        "factor_name": factor_name,
        "classified_value": value,
        "classification_basis": basis,
    }
    return {**proof, "classification_hash": _value_hash(proof)}


def reviewed_seed_factor_proof(
    scenario_id: str,
    factor_name: str,
) -> Mapping[str, Any]:
    """Return a deterministic typed classification of one reviewed seed."""

    return _seed_classification(scenario_id, factor_name)


def _claimed_factor(context: JourneyVerifierContext, name: str, value: str) -> bool:
    factor_id = f"{name}:{value}"
    return any(
        factor_id in decision.risk_factor_ids
        for decision in context.completed_decisions
    )


def _seed_factor_observed(
    context: JourneyVerifierContext,
    name: str,
    value: str,
) -> tuple[bool, Mapping[str, Any]]:
    scenario_id = str(getattr(context.schedule, "start_scenario", "") or "")
    proof = _seed_classification(scenario_id, name)
    fingerprint_bound = (
        str(context.initial_event.after_fingerprint)
        == str(proof["scenario_state_fingerprint"])
    )
    classified = proof["classified_value"] == value
    missing = []
    if not fingerprint_bound:
        missing.append("journey_seed_binding_receipt")
    if not classified:
        missing.append(f"reviewed_seed_classification:{name}={value}")
    return fingerprint_bound and classified, {
        "evidence_class": "seed_fact",
        "typed_seed_proof": proof,
        "actual_initial_event_fingerprint": context.initial_event.after_fingerprint,
        "missing_product_receipts": missing,
    }


def _input_shape_observed(
    context: JourneyVerifierContext,
    value: str,
) -> tuple[bool, Mapping[str, Any]]:
    observations: list[dict[str, Any]] = []
    for index, ((message, _), event) in enumerate(
        zip(context.transcript, context.completed_events, strict=False),
        start=1,
    ):
        clauses = segment_user_turn(str(message))
        shapes = {clause.input_shape for clause in clauses}
        turn_shape = str(event.turn_receipt_summary.get("input_shape") or "")
        pending_receipts = [
            receipt
            for receipt_event, receipt in _validated_receipts(
                context, "pending_resolution"
            )
            if receipt_event.turn_index == event.turn_index
        ]
        planner_receipts = [
            receipt
            for receipt_event, receipt in _validated_receipts(
                context, "semantic_planner"
            )
            if receipt_event.turn_index == event.turn_index
        ]
        unresolved = tuple(
            event.turn_receipt_summary.get("unresolved_unit_ids") or ()
        )
        raw_exact = any(
            receipt.get("resolution_path") == "exact_contract"
            for receipt in pending_receipts
        )
        raw_natural = (
            "\n" not in str(message)
            and shapes == {"prose"}
            and turn_shape == "prose"
            and bool(planner_receipts)
            and not unresolved
        )
        raw_multiline = (
            "\n" in str(message)
            and shapes == {"prose"}
            and turn_shape == "prose"
        )
        raw_structured = (
            "structured" in shapes
            and turn_shape in {"structured", "mixed"}
        )
        raw_contradictory = bool(planner_receipts and unresolved)
        classified_shape = (
            "contradictory"
            if raw_contradictory
            else "exact"
            if raw_exact
            else "structured"
            if raw_structured
            else "multiline"
            if raw_multiline
            else "natural"
            if raw_natural
            else ""
        )
        observed = classified_shape == value
        observations.append({
            "turn_index": event.turn_index,
            "syntax_shapes": sorted(shapes),
            "turn_receipt_input_shape": turn_shape,
            "pending_exact_contract": raw_exact,
            "semantic_planner_resolver_invoked": bool(planner_receipts),
            "unresolved_unit_ids": list(unresolved),
            "classified_shape": classified_shape,
            "observed": observed,
        })
    return any(item["observed"] for item in observations), {
        "evidence_class": "stimulus_property",
        "turn_observations": observations,
        "missing_product_receipts": (
            [] if any(item["observed"] for item in observations)
            else [f"exact_input_shape_consequence:{value}"]
        ),
    }


def _evidence_shape_observed(
    context: JourneyVerifierContext,
    value: str,
) -> tuple[bool, Mapping[str, Any]]:
    provenance = _validated_receipts(context, "rpc_schema_provenance")
    evidence_receipts = _validated_receipts(
        context,
        "rpc_schema_provenance",
        "rpc_catalog_transition",
        "analysis_evidence_block",
    )
    evidence_delta = _changed_path(
        context,
        "custom_rpc.catalog",
        "chain_identity.catalog",
        "evidence_collection",
        "evidence_buffer",
    )
    fields = [
        dict(field)
        for _, receipt in provenance
        for field in receipt.get("fields") or ()
        if isinstance(field, Mapping)
    ]
    request_fields = [
        field for field in fields
        if field.get("source_kind") == "protocol_request_parser"
    ]
    response_fields = [
        field for field in fields
        if str(field.get("field_path") or "").startswith("response")
    ]
    docs_fields = [
        field for field in fields
        if field.get("source_kind") == "official_document"
    ]
    request_turns = {
        int(event.turn_index)
        for event, receipt in provenance
        if any(
            isinstance(field, Mapping)
            and field.get("source_kind") == "protocol_request_parser"
            for field in receipt.get("fields") or ()
        )
    }
    response_turns = {
        int(event.turn_index)
        for event, receipt in provenance
        if any(
            isinstance(field, Mapping)
            and str(field.get("field_path") or "").startswith("response")
            for field in receipt.get("fields") or ()
        )
    }
    revisions = {
        int(revision)
        for field in fields
        for revision in field.get("source_revisions") or ()
        if isinstance(revision, int) and not isinstance(revision, bool)
    }
    receipt_turns = {int(event.turn_index) for event, _ in provenance}
    request_precedes_response = bool(
        request_turns
        and response_turns
        and min(request_turns) < max(response_turns)
    )
    observed = {
        "none": not evidence_receipts and not evidence_delta,
        "request": bool(request_fields) and not response_fields and not docs_fields,
        "response": bool(response_fields) and not request_fields and not docs_fields,
        "split": (
            len(revisions) >= 2
            and len(receipt_turns) >= 2
            and bool(request_fields)
            and bool(response_fields)
            and request_precedes_response
        ),
        "docs": bool(docs_fields) and not request_fields and not response_fields,
    }[value]
    missing = []
    if not observed:
        if value == "docs" and not docs_fields:
            missing.append("rpc_schema_provenance.source_kind=official_document")
        elif value == "split":
            missing.append("ordered_multi_turn_rpc_schema_provenance")
        else:
            missing.append(f"typed_rpc_evidence_receipt:{value}")
    return observed, {
        "evidence_class": "stimulus_property",
        "request_field_count": len(request_fields),
        "response_field_count": len(response_fields),
        "docs_field_count": len(docs_fields),
        "source_revisions": sorted(revisions),
        "receipt_turns": sorted(receipt_turns),
        "request_turns": sorted(request_turns),
        "response_turns": sorted(response_turns),
        "request_precedes_response": request_precedes_response,
        "evidence_state_delta": evidence_delta,
        "missing_product_receipts": missing,
    }


def _recovery_observed(
    context: JourneyVerifierContext,
    value: str,
) -> tuple[bool, Mapping[str, Any]]:
    requirements = {
        "back": ("go_back", "go_back", "interruption_stack"),
        "jump": ("change_group", "change_group", "active_group"),
        "correct": ("correct_failure", "", "failure_recovery"),
        "retry": ("retry_failure", "", "failure_recovery"),
        "reset": ("reset_session", "", "confirmed_config"),
    }
    recovery_actions = {
        "go_back", "change_group", "correct_failure", "retry_failure",
        "reset_session",
    }
    admitted = {
        action
        for event in context.completed_events
        for action in event.admitted_action_types
    }
    recovery_commits = [
        (event, receipt)
        for event, receipt in _domain_commits(context)
        if (
            receipt.get("navigation_operation") in {"go_back", "change_group"}
            or any(
                action in recovery_actions
                for action in event.admitted_action_types
            )
        )
    ]
    if value == "none":
        observed = (
            not admitted.intersection(recovery_actions)
            and not recovery_commits
        )
        return observed, {
            "evidence_class": "runtime_fact",
            "admitted_recovery_actions": sorted(admitted.intersection(recovery_actions)),
            "missing_product_receipts": [] if observed else ["absence_of_recovery_commit"],
        }
    action, navigation, delta_prefix = requirements[value]
    matching = [
        receipt
        for event, receipt in _domain_commits(context, navigation=navigation)
        if (
            action in event.admitted_action_types
            and _receipt_changes(receipt, delta_prefix)
        )
    ]
    observed = bool(matching)
    return observed, {
        "evidence_class": "runtime_fact",
        "required_action": action,
        "required_navigation": navigation,
        "required_material_delta": delta_prefix,
        "matching_commit_receipt_ids": [
            str(receipt.get("receipt_id") or "") for receipt in matching
        ],
        "missing_product_receipts": (
            [] if observed else [f"domain_commit:{action}:{delta_prefix}"]
        ),
    }


def _group_state_observed(
    context: JourneyVerifierContext,
    value: str,
) -> tuple[bool, Mapping[str, Any]]:
    subject_group = str(
        getattr(context.schedule, "subject_group", "") or ""
    )
    if not subject_group:
        return False, {
            "evidence_class": "runtime_fact",
            "expected_statuses": [],
            "subject_group": "",
            "observations": [],
            "missing_product_receipts": ["schedule:subject_group"],
        }
    expected_statuses = {
        "partial": frozenset({"in_progress", "reconfiguring"}),
        "completed": frozenset({"completed"}),
        "invalidated": frozenset({"invalidated"}),
    }[value]
    observations: list[dict[str, Any]] = []
    for event, receipt in _validated_receipts(context, "domain_commit"):
        if receipt.get("completion") not in {
            "in_progress",
            "completed",
            "blocked",
        }:
            continue
        for transition in receipt.get("group_state_transitions") or ():
            if (
                not isinstance(transition, Mapping)
                or str(transition.get("group") or "") != subject_group
                or str(transition.get("after") or "") not in expected_statuses
            ):
                continue
            observations.append({
                "turn_index": int(event.turn_index),
                "group": str(transition.get("group") or ""),
                "status": str(transition.get("after") or ""),
                "before_status": str(transition.get("before") or ""),
                "receipt_id": str(receipt.get("receipt_id") or ""),
            })
    return bool(observations), {
        "evidence_class": "runtime_fact",
        "subject_group": subject_group,
        "expected_statuses": sorted(expected_statuses),
        "observations": observations,
        "missing_product_receipts": (
            []
            if observations
            else [f"domain_commit:group_states.<subject>.status:{value}"]
        ),
    }


def _factor_structural_observation(
    context: JourneyVerifierContext,
    name: str,
    value: str,
) -> tuple[bool, Mapping[str, Any]]:
    if name in {"session_state", "pending_state"}:
        return _seed_factor_observed(context, name, value)
    if name == "subject_group":
        scheduled_group = str(
            getattr(context.schedule, "subject_group", "") or ""
        )
        matching_events = [
            event
            for event in context.completed_events
            if any(
                isinstance(transition, Mapping)
                and str(transition.get("group") or "") == value
                for receipt_event, receipt in _validated_receipts(
                    context,
                    "domain_commit",
                )
                if receipt_event.turn_index == event.turn_index
                for transition in receipt.get(
                    "group_state_transitions"
                ) or ()
            )
        ]
        observed = scheduled_group == value and bool(matching_events)
        return observed, {
            "evidence_class": "runtime_fact",
            "scheduled_subject_group": scheduled_group,
            "observed_turn_indexes": [
                int(event.turn_index) for event in matching_events
            ],
            "missing_product_receipts": (
                [] if observed else [f"domain_commit:subject_group:{value}"]
            ),
        }
    if name == "group_state":
        return _group_state_observed(context, value)
    if name == "workflow_mode":
        expected = {
            "fake": ("rpc_benchmark", "fake-node"),
            "real": ("rpc_benchmark", "real-node"),
            "sync": ("sync_observe", "sync-observe"),
        }[value]
        commits = [
            receipt
            for event, receipt in _domain_commits(context)
            if (
                _receipt_writes(receipt, "workflow_mode")
                and _receipt_writes(receipt, "target_mode")
                and _event_has_state_value(event, "workflow_mode", expected[0])
                and _event_has_state_value(event, "target_mode", expected[1])
            )
        ]
        observed = len(commits) == 1
        return observed, {
            "evidence_class": "runtime_fact",
            "expected_workflow_mode": expected[0],
            "expected_target_mode": expected[1],
            "missing_product_receipts": [] if observed else ["domain_commit:workflow_mode"],
        }
    if name == "language":
        receipts = [
            receipt for event, receipt in _validated_receipts(
                context, "response_composition"
            )
            if receipt.get("language") == value
            and _event_has_state_value(event, "language", value)
        ]
        observed = bool(receipts)
        return observed, {
            "evidence_class": "runtime_fact",
            "response_composition_receipts": len(receipts),
            "missing_product_receipts": [] if observed else ["response_composition:language"],
        }
    if name == "interruption_depth":
        navigation = [
            receipt
            for event, receipt in _domain_commits(context)
            if (
                receipt.get("navigation_operation") in {"change_group", "go_back"}
                and (
                    (
                        receipt.get("navigation_operation") == "change_group"
                        and "change_group" in event.admitted_action_types
                    )
                    or (
                        receipt.get("navigation_operation") == "go_back"
                        and "go_back" in event.admitted_action_types
                    )
                )
            )
            and _receipt_changes(receipt, "interruption_stack")
        ]
        connected = all(
            navigation[index - 1].get("navigation_target_group")
            == navigation[index].get("navigation_origin_group")
            for index in range(1, len(navigation))
        )
        observed = (
            len(navigation) == 0 if value == "0"
            else len(navigation) == 1 if value == "1"
            else len(navigation) >= 2 and connected
        )
        return observed, {
            "evidence_class": "runtime_fact",
            "navigation_commit_count": len(navigation),
            "navigation_sequence_connected": connected,
            "missing_product_receipts": (
                [] if observed else ["ordered_navigation_domain_commits"]
            ),
        }
    if name == "input_shape":
        return _input_shape_observed(context, value)
    if name == "chain_case":
        owner_receipts = _validated_receipts(
            context, "chain_identity_resolution", "domain_commit"
        )
        exact_state = _has_state_value(context, "chain_identity.case", value)
        confirmed_state = _has_state_value(
            context,
            "chain_identity.status",
            "confirmed",
        )
        owner_bound = any(
            (
                receipt.get("receipt_type") == "chain_identity_resolution"
                and _event_has_state_value(event, "chain_identity.case", value)
                and _event_has_state_value(
                    event, "chain_identity.status", "confirmed"
                )
            )
            or (
                receipt.get("receipt_type") == "domain_commit"
                and receipt.get("owner") == "chain_identity"
                and _receipt_writes(receipt, "chain_identity")
                and _event_has_state_value(event, "chain_identity.case", value)
                and _event_has_state_value(
                    event, "chain_identity.status", "confirmed"
                )
            )
            for event, receipt in owner_receipts
        )
        observed = exact_state and confirmed_state and owner_bound
        return observed, {
            "evidence_class": "runtime_fact",
            "exact_case": value,
            "required_status": "confirmed",
            "owner_receipt_count": len(owner_receipts),
            "missing_product_receipts": (
                []
                if observed
                else [f"chain_identity_owner_receipt:{value}:confirmed"]
            ),
        }
    if name == "workload":
        if value == "not_applicable":
            commits = _validated_receipts(context, "rpc_workload_commit")
            exclusion_commits = [
                receipt
                for event, receipt in _domain_commits(context)
                if (
                    (
                        _receipt_writes(receipt, "workflow_mode")
                        and _event_has_state_value(
                            event, "workflow_mode", "sync_observe"
                        )
                    )
                    or (
                        _receipt_writes(receipt, "chain_identity")
                        and _event_has_state_value(
                            event, "chain_identity.case", "case3"
                        )
                    )
                )
            ]
            observed = not commits and len(exclusion_commits) == 1
        else:
            mode = "single" if value.endswith("single") else "mixed"
            custom = value.startswith("custom")
            commits = [
                receipt for _, receipt in _validated_receipts(
                    context, "rpc_workload_commit"
                )
                if receipt.get("rpc_mode") == mode
                and receipt.get("job_local_override") is custom
                and receipt.get("replace_defaults") is custom
                and receipt.get("choice")
                == ("custom_rpc" if custom else "template_default")
                and bool(receipt.get("methods"))
            ]
            observed = len(commits) == 1
        return observed, {
            "evidence_class": "runtime_fact",
            "matching_workload_commit_count": len(commits),
            "missing_product_receipts": (
                [] if observed else [f"rpc_workload_commit:{value}"]
            ),
        }
    if name == "evidence_shape":
        return _evidence_shape_observed(context, value)
    if name == "recovery":
        return _recovery_observed(context, value)
    raise ValueError(f"unsupported factor observation: {name}={value}")


def factor_observed(
    context: JourneyVerifierContext,
) -> JourneyPostconditionResult:
    postcondition_id = str(context.evaluating_postcondition_id or "")
    prefix = "factor_observed__"
    if not postcondition_id.startswith(prefix):
        raise ValueError("factor verifier has no bound postcondition identity")
    factor_name, separator, encoded_value = postcondition_id[len(prefix):].partition("__")
    if not separator:
        raise ValueError("factor verifier postcondition identity is malformed")
    factor_value = encoded_value.replace("plus", "+")
    postcondition_id = factor_observation_postcondition_id(
        factor_name, factor_value
    )
    claimed = _claimed_factor(context, factor_name, factor_value)
    observed, details = _factor_structural_observation(
        context, factor_name, factor_value
    )
    return _result(
        postcondition_id,
        claimed and observed,
        context,
        factor_name=factor_name,
        factor_value=factor_value,
        response_bound_claim_observed=claimed,
        structural_observation=observed,
        **dict(details),
    )


_BASE_VERIFIER_DEFINITIONS = (
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
)

_FACTOR_VERIFIER_DEFINITIONS = tuple(
    JourneyPostconditionVerifierDefinition(
        factor_observation_postcondition_id(name, value),
        f"formal-factor-observation-{name}-{value.replace('+', 'plus')}",
        2,
        (
            "A response-bound simulator decision declared the frozen factor and "
            "runtime state, action, or input-shape evidence independently observed it."
        ),
        factor_observed,
    )
    for name, values in FACTOR_OBSERVATION_VALUES.items()
    for value in values
)

FORMAL_JOURNEY_VERIFIER_REGISTRY = build_journey_outcome_verifier_registry(
    (*_BASE_VERIFIER_DEFINITIONS, *_FACTOR_VERIFIER_DEFINITIONS)
)


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
