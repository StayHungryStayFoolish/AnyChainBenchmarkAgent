"""Shared semantic predicates for retained-regression journey evidence.

The predicates combine immutable test-side contracts, response-bound external
attestations, and product-process evidence. They never inspect retained
regression case IDs, user phrases, or chain names.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from agent.harness.control_receipts import (
    validate_persisted_domain_control_receipt,
)
from agent.harness.state import DURABLE_ENVIRONMENT_STATE_ROOTS
from agent.workflows.group_registry import GROUP_OWNER
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.retained_regression_attestations import (
    collect_valid_variant_attestations,
    validate_verifier_input_contract,
)


PredicateResult = tuple[bool, Mapping[str, Any]]
PostconditionEvaluator = Callable[[Any], PredicateResult]

_HASH_LENGTH = 64
_ENVIRONMENT_INTERRUPT_ROLES = frozenset({
    "chain_change_request",
    "chain_mode_change_request",
})
_RPC_GROUPS = frozenset({
    "chain_identity",
    "endpoint_process",
    "workload_rpc",
    "target_samples_fixtures",
})
_DISK_LIMIT_FIELDS = frozenset({
    "DATA_VOL_MAX_IOPS",
    "DATA_VOL_MAX_THROUGHPUT",
})


def _events(context: Any) -> tuple[Any, ...]:
    return tuple(getattr(context, "completed_events", ()) or ())


def _turns(context: Any) -> tuple[Any, ...]:
    return tuple(getattr(context, "completed_turns", ()) or ())


def _verifier_input(context: Any) -> tuple[dict[str, Any] | None, str]:
    try:
        return validate_verifier_input_contract(
            getattr(context, "verifier_input_contract", {}) or {}
        ), ""
    except (TypeError, ValueError) as exc:
        return None, str(exc)


def _is_hash(value: Any, *, allow_empty: bool = False) -> bool:
    text = str(value or "")
    if allow_empty and not text:
        return True
    return (
        len(text) == _HASH_LENGTH
        and text == text.lower()
        and all(character in "0123456789abcdef" for character in text)
    )


def _valid_receipts(
    context: Any,
    *receipt_types: str,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    wanted = set(receipt_types)
    valid: list[Mapping[str, Any]] = []
    invalid: list[Mapping[str, Any]] = []
    for event in _events(context):
        for receipt in tuple(getattr(event, "control_receipts", ()) or ()):
            if not isinstance(receipt, Mapping):
                continue
            receipt_type = str(receipt.get("receipt_type") or "")
            if receipt_type not in wanted:
                continue
            accepted, reason = validate_persisted_domain_control_receipt(
                receipt,
                turn_index=int(getattr(event, "turn_index", -1)),
            )
            record = {
                "receipt": receipt,
                "receipt_id": str(receipt.get("receipt_id") or ""),
                "turn_index": int(getattr(event, "turn_index", -1)),
                "reason": reason,
            }
            (valid if accepted else invalid).append(record)
    return valid, invalid


def _receipt_details(
    family: str,
    valid: Sequence[Mapping[str, Any]],
    invalid: Sequence[Mapping[str, Any]],
    **observations: Any,
) -> dict[str, Any]:
    return {
        "evidence_family": family,
        "receipt_ids": [
            str(item.get("receipt_id") or "")
            for item in valid
            if str(item.get("receipt_id") or "")
        ],
        "invalid_receipts": [
            {
                "receipt_id": str(item.get("receipt_id") or ""),
                "reason": str(item.get("reason") or ""),
            }
            for item in invalid
        ],
        **observations,
    }


def _receipt_predicate(
    context: Any,
    *,
    family: str,
    receipt_types: Sequence[str],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> PredicateResult:
    valid, invalid = _valid_receipts(context, *receipt_types)
    matched = [
        item for item in valid if predicate(item["receipt"])
    ]
    return bool(matched) and not invalid, _receipt_details(
        family,
        matched,
        invalid,
        valid_receipt_count=len(valid),
        matched_receipt_count=len(matched),
    )


def _valid_pending_transition(event: Any) -> Mapping[str, Any] | None:
    transition = dict(getattr(event, "pending_transition", {}) or {})
    required = {
        "transition",
        "before_id",
        "before_group",
        "before_hash",
        "after_id",
        "after_group",
        "after_hash",
        "consumer_action_ids",
    }
    if (
        set(transition) != required
        or transition.get("transition")
        not in {"preserved", "replaced", "consumed", "created", "absent"}
        or not _is_hash(transition.get("before_hash"))
        or not _is_hash(transition.get("after_hash"))
        or not isinstance(transition.get("consumer_action_ids"), list)
        or any(
            not isinstance(item, str)
            for item in transition.get("consumer_action_ids") or ()
        )
        or str(transition.get("after_id") or "")
        != str(getattr(event, "pending_question_id", "") or "")
    ):
        return None
    return transition


def _valid_material_diffs(event: Any) -> Mapping[str, Mapping[str, str]]:
    output: dict[str, Mapping[str, str]] = {}
    for path, hashes in dict(
        getattr(event, "material_state_diff_hashes", {}) or {}
    ).items():
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(hashes, Mapping)
            or set(hashes) != {"before", "after"}
            or not _is_hash(hashes.get("before"), allow_empty=True)
            or not _is_hash(hashes.get("after"), allow_empty=True)
            or hashes.get("before") == hashes.get("after")
        ):
            continue
        output[path] = hashes
    return output


def _domain_commits(context: Any) -> tuple[
    list[Mapping[str, Any]], list[Mapping[str, Any]]
]:
    return _valid_receipts(context, "domain_commit")


def _material_delta_records(
    context: Any,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    valid, invalid = _domain_commits(context)
    records: list[Mapping[str, Any]] = []
    for item in valid:
        receipt = item["receipt"]
        if receipt.get("completion") == "rejected":
            continue
        for delta in receipt.get("material_delta") or ():
            records.append({
                **item,
                "operation": str(delta.get("operation") or ""),
                "path": str(delta.get("path") or ""),
                "value_hash": str(delta.get("value_hash") or ""),
            })
    return records, invalid


def _path_leaf(path: str) -> str:
    return str(path or "").rsplit(".", 1)[-1]


def _path_root(path: str) -> str:
    return str(path or "").split(".", 1)[0]


def _exact_fixture_turns_observed(context: Any) -> PredicateResult:
    contract, error = _verifier_input(context)
    observed_turns = tuple(
        str(getattr(turn, "user_message", ""))
        for turn in _turns(context)
    )
    observed_hash = content_hash(observed_turns)
    if contract is None:
        return False, {
            "evidence_family": "immutable_fixture_replay",
            "reason": error,
            "observed_turn_count": len(observed_turns),
            "observed_turns_hash": observed_hash,
        }
    source = contract["source_contract"]
    variant = contract["variant_contract"]
    expected_count = len(source["source_steps"])
    expected_hash = str(source["source_turns_hash"])
    satisfied = bool(
        contract["mode"] == "exact_fixture_replay"
        and variant["variant"] == "exact"
        and len(observed_turns) == expected_count
        and observed_hash == expected_hash
    )
    return satisfied, {
        "evidence_family": "immutable_fixture_replay",
        "source_contract_hash": contract["source_contract_hash"],
        "variant_contract_hash": contract["variant_contract_hash"],
        "expected_turn_count": expected_count,
        "observed_turn_count": len(observed_turns),
        "expected_turns_hash": expected_hash,
        "observed_turns_hash": observed_hash,
    }


def _variant_attested(
    context: Any,
    *,
    variant: str,
    relation: str,
) -> PredicateResult:
    contract, error = _verifier_input(context)
    if contract is None:
        return False, {
            "evidence_family": "response_bound_variant_attestation",
            "reason": error,
            "valid_attestation_ids": [],
            "invalid_attestations": [],
        }
    variant_contract = contract["variant_contract"]
    if (
        variant_contract["variant"] != variant
        or variant_contract["relation"] != relation
    ):
        return False, {
            "evidence_family": "response_bound_variant_attestation",
            "reason": "retained variant contract does not match the predicate",
            "source_contract_hash": contract["source_contract_hash"],
            "variant_contract_hash": contract["variant_contract_hash"],
            "valid_attestation_ids": [],
            "invalid_attestations": [],
        }
    valid, invalid = collect_valid_variant_attestations(context)
    matching = [
        item for item in valid
        if item.get("relation") == relation
    ]
    required_steps = tuple(variant_contract["required_source_step_ids"])
    binding_counts = Counter(
        str(item.get("source_step_id") or "")
        for item in matching
    )
    missing_steps = sorted(set(required_steps) - set(binding_counts))
    duplicate_steps = sorted(
        step_id
        for step_id, count in binding_counts.items()
        if count != 1
    )
    unexpected_steps = sorted(set(binding_counts) - set(required_steps))
    satisfied = bool(
        len(matching) == len(required_steps)
        and not invalid
        and not missing_steps
        and not duplicate_steps
        and not unexpected_steps
    )
    return satisfied, {
        "evidence_family": "response_bound_variant_attestation",
        "source_contract_hash": contract["source_contract_hash"],
        "variant_contract_hash": contract["variant_contract_hash"],
        "required_relation": relation,
        "required_source_step_ids": list(required_steps),
        "valid_attestation_ids": [
            str(item.get("attestation_id") or "")
            for item in matching
        ],
        "attested_source_step_ids": sorted({
            str(item.get("source_step_id") or "")
            for item in matching
        }),
        "missing_source_step_ids": missing_steps,
        "duplicate_source_step_ids": duplicate_steps,
        "unexpected_source_step_ids": unexpected_steps,
        "invalid_attestations": invalid,
    }


def _isomorphic_meaning_attested(context: Any) -> PredicateResult:
    return _variant_attested(
        context,
        variant="isomorphic",
        relation="isomorphic_meaning",
    )


def _adjacent_non_trigger_attested(context: Any) -> PredicateResult:
    return _variant_attested(
        context,
        variant="negative",
        relation="adjacent_non_trigger",
    )


def _neighboring_transition_attested(context: Any) -> PredicateResult:
    return _variant_attested(
        context,
        variant="neighboring",
        relation="neighboring_transition",
    )


def _semantic_roles_by_turn(
    context: Any,
) -> tuple[dict[int, set[str]], list[dict[str, str]], str]:
    contract, error = _verifier_input(context)
    if contract is None:
        return {}, [], error
    roles: dict[int, set[str]] = defaultdict(set)
    variant = str(contract["variant_contract"]["variant"])
    if variant == "exact":
        observed_turns = _turns(context)
        observed_hash = content_hash(tuple(
            str(getattr(turn, "user_message", ""))
            for turn in observed_turns
        ))
        if observed_hash != contract["source_contract"]["source_turns_hash"]:
            return {}, [], "exact observed turns do not match the source contract"
        source_steps = tuple(contract["source_contract"]["source_steps"])
        if len(source_steps) != len(observed_turns):
            return {}, [], "exact source steps do not match observed turns"
        for step, turn in zip(source_steps, observed_turns, strict=True):
            roles[int(getattr(turn, "turn_index", -1))].add(
                str(step["semantic_role"])
            )
        return roles, [], ""
    valid, invalid = collect_valid_variant_attestations(context)
    for attestation in valid:
        roles[int(attestation["turn_index"])].add(
            str(attestation["semantic_role"])
        )
    return roles, invalid, ""


def _environment_text_consumed_as_region(context: Any) -> PredicateResult:
    roles_by_turn, invalid_attestations, role_error = _semantic_roles_by_turn(
        context
    )
    valid, invalid_receipts = _valid_receipts(context, "pending_resolution")
    turns = {
        int(getattr(turn, "turn_index", -1)): turn
        for turn in _turns(context)
    }
    events = {
        int(getattr(event, "turn_index", -1)): event
        for event in _events(context)
    }
    interruption_turns = {
        turn_index
        for turn_index, roles in roles_by_turn.items()
        if roles & _ENVIRONMENT_INTERRUPT_ROLES
    }
    interruption_turns.update(
        turn_index
        for turn_index, roles in roles_by_turn.items()
        if "confirmation" in roles
        and roles_by_turn.get(turn_index - 1, set())
        & _ENVIRONMENT_INTERRUPT_ROLES
    )
    violations: list[dict[str, Any]] = []
    for item in valid:
        receipt = item["receipt"]
        turn_index = int(item["turn_index"])
        turn = turns.get(turn_index)
        event = events.get(turn_index)
        if turn is None or event is None:
            continue
        action_id = str(receipt.get("resolved_action_id") or "")
        admitted_ids = set(
            dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
                "admitted_action_ids"
            )
            or ()
        )
        transition = _valid_pending_transition(event)
        input_hash = hashlib.sha256(
            str(getattr(turn, "user_message", "")).encode("utf-8")
        ).hexdigest()
        if (
            receipt.get("pending_id") == "CLOUD_REGION"
            and receipt.get("verdict") == "accepted"
            and receipt.get("input_hash") == input_hash
            and turn_index in interruption_turns
            and action_id in admitted_ids
            and transition is not None
            and action_id in set(transition["consumer_action_ids"])
        ):
            violations.append({
                "turn_index": turn_index,
                "receipt_id": str(item.get("receipt_id") or ""),
                "resolved_action_id": action_id,
                "semantic_roles": sorted(roles_by_turn[turn_index]),
                "resolution_path": str(
                    receipt.get("resolution_path") or ""
                ),
            })
    return (
        bool(violations)
        and not invalid_receipts
        and not invalid_attestations
        and not role_error
    ), _receipt_details(
        "semantic_intent_pending_consumption",
        valid,
        invalid_receipts,
        violations=violations,
        invalid_attestations=invalid_attestations,
        role_error=role_error,
    )


def _pending_preserved_with_receipt(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "orientation_response")
    matched_events: list[int] = []
    receipt_turns = {int(item["turn_index"]) for item in valid}
    for event in _events(context):
        transition = _valid_pending_transition(event)
        if (
            int(getattr(event, "turn_index", -1)) in receipt_turns
            and transition
            and transition["transition"] == "preserved"
            and transition["before_id"]
        ):
            matched_events.append(int(getattr(event, "turn_index", -1)))
    return bool(matched_events) and not invalid, _receipt_details(
        "read_only_pending_lineage",
        valid,
        invalid,
        matched_turn_indexes=matched_events,
    )


def _orientation_answered_read_only(context: Any) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="orientation_read_only",
        receipt_types=("orientation_response",),
        predicate=lambda receipt: receipt.get("read_only") is True,
    )


def _consultation_answered_read_only(context: Any) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="orientation_read_only",
        receipt_types=("orientation_response",),
        predicate=lambda receipt: (
            receipt.get("read_only") is True
            and receipt.get("action_type") == "answer_opening_question"
        ),
    )


def _retained_state_described(context: Any) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="orientation_read_only",
        receipt_types=("orientation_response",),
        predicate=lambda receipt: (
            receipt.get("read_only") is True
            and receipt.get("topic") in {"current_config", "current_context"}
        ),
    )


def _resume_action_contract_exposed(context: Any) -> PredicateResult:
    from tests.agent_live.harness_contract_scenarios import (
        canonical_question_contract,
    )
    from tests.agent_live.runtime_checkpoint import reviewed_scenario

    scenario_id = str(
        getattr(getattr(context, "schedule", None), "start_scenario", "") or ""
    )
    try:
        scenario = reviewed_scenario(scenario_id)
    except (KeyError, TypeError, ValueError):
        return False, {
            "evidence_family": "resume_contract",
            "scenario_id": scenario_id,
            "expected_pending_contract_hash": "",
            "matched_pending_contract_hashes": [],
            "matched_turn_indexes": [],
            "error": "reviewed start scenario is unavailable",
        }
    expected_contract = dict(scenario.question or {})
    expected_hash = (
        content_hash(canonical_question_contract(expected_contract))
        if expected_contract
        else ""
    )
    matched = []
    matched_hashes: list[str] = []
    initial_event = getattr(context, "initial_event", None)
    events = (
        *((initial_event,) if initial_event is not None else ()),
        *_events(context),
    )
    seen_event_ids: set[str] = set()
    for event in events:
        event_id = str(getattr(event, "runtime_event_id", "") or "")
        if event_id and event_id in seen_event_ids:
            continue
        if event_id:
            seen_event_ids.add(event_id)
        contract = dict(getattr(event, "pending_contract", {}) or {})
        contract_hash = (
            content_hash(canonical_question_contract(contract))
            if contract
            else ""
        )
        options = tuple(contract.get("options") or ())
        option_ids = [
            str(option.get("id") or option.get("option_id") or "")
            for option in options
            if isinstance(option, Mapping)
        ]
        if (
            contract.get("id") == "resume_harness_session"
            and expected_hash
            and contract_hash == expected_hash
            and len(option_ids) >= 2
            and all(option_ids)
            and len(option_ids) == len(set(option_ids))
            and all(
                isinstance(option, Mapping)
                and (
                    option.get("action")
                    or option.get("expected_patch")
                    or option.get("semantic_action")
                )
                for option in options
            )
        ):
            matched.append(int(getattr(event, "turn_index", -1)))
            matched_hashes.append(contract_hash)
    return bool(matched), {
        "evidence_family": "resume_contract",
        "scenario_id": scenario_id,
        "expected_pending_contract_hash": expected_hash,
        "matched_pending_contract_hashes": matched_hashes,
        "matched_turn_indexes": matched,
    }


def _workflow_state_mutated_by_consultation(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "orientation_response")
    turns = {int(item["turn_index"]) for item in valid}
    changed = {
        int(getattr(event, "turn_index", -1)): sorted(_valid_material_diffs(event))
        for event in _events(context)
        if int(getattr(event, "turn_index", -1)) in turns
        and _valid_material_diffs(event)
    }
    return bool(changed) and not invalid, _receipt_details(
        "read_only_material_diff",
        valid,
        invalid,
        changed_paths_by_turn=changed,
    )


def _current_turn_language_preserved(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "response_composition")
    matches: list[int] = []
    for item in valid:
        event = next(
            (
                candidate
                for candidate in _events(context)
                if int(getattr(candidate, "turn_index", -1))
                == int(item["turn_index"])
            ),
            None,
        )
        if event is None:
            continue
        receipt = item["receipt"]
        turn_language = str(
            dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
                "language"
            )
            or ""
        )
        render_language = str(
            dict(getattr(event, "render_manifest", {}) or {}).get("language")
            or ""
        )
        if (
            turn_language
            and turn_language == receipt.get("language")
            and (not render_language or render_language == turn_language)
        ):
            matches.append(int(item["turn_index"]))
    return bool(matches) and not invalid, _receipt_details(
        "language_lineage",
        valid,
        invalid,
        matched_turn_indexes=matches,
    )


def _consultation_consumed_as_pending_value(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "orientation_response")
    receipt_turns = {int(item["turn_index"]) for item in valid}
    consumed = []
    for event in _events(context):
        transition = _valid_pending_transition(event)
        if (
            int(getattr(event, "turn_index", -1)) in receipt_turns
            and transition
            and transition["before_id"]
            and transition["transition"] in {"consumed", "replaced"}
        ):
            consumed.append(int(getattr(event, "turn_index", -1)))
    return bool(consumed) and not invalid, _receipt_details(
        "read_only_pending_lineage",
        valid,
        invalid,
        consumed_turn_indexes=consumed,
    )


def _response_driven_selection_observed(context: Any) -> PredicateResult:
    turns = tuple(getattr(context, "completed_turns", ()) or ())
    decisions = tuple(getattr(context, "completed_decisions", ()) or ())
    matched = bool(turns) and len(turns) == len(decisions)
    bindings = []
    if matched:
        for turn, decision in zip(turns, decisions):
            bindings.append(
                int(getattr(decision, "turn_index", -1))
                == int(getattr(turn, "turn_index", -2))
                and str(getattr(decision, "previous_response_hash", ""))
                == content_hash(str(getattr(turn, "previous_agent_response", "")))
                and str(getattr(decision, "user_message_hash", ""))
                == content_hash(str(getattr(turn, "user_message", "")))
            )
    return matched and all(bindings), {
        "evidence_family": "response_bound_decisions",
        "turn_count": len(turns),
        "decision_count": len(decisions),
        "binding_results": bindings,
    }


def _visible_option_action_executed(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "pending_resolution")
    matches = []
    for item in valid:
        receipt = item["receipt"]
        if (
            receipt.get("resolution_path") == "exact_contract"
            and receipt.get("selected_option_id")
        ):
            event = next(
                (
                    candidate
                    for candidate in _events(context)
                    if int(getattr(candidate, "turn_index", -1))
                    == int(item["turn_index"])
                ),
                None,
            )
            if event is None:
                continue
            executed = set(
                dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
                    "execution_order"
                )
                or ()
            )
            consumers = set(
                (_valid_pending_transition(event) or {}).get(
                    "consumer_action_ids"
                )
                or ()
            )
            resolved_action_id = str(receipt.get("resolved_action_id") or "")
            if (
                resolved_action_id
                and resolved_action_id in executed
                and resolved_action_id in consumers
            ):
                matches.append(item)
    return bool(matches) and not invalid, _receipt_details(
        "visible_option_binding",
        matches,
        invalid,
        matched_receipt_count=len(matches),
    )


def _current_menu_binding_preserved(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "pending_resolution")
    matches = []
    for item in valid:
        receipt = item["receipt"]
        event = next(
            (
                candidate
                for candidate in _events(context)
                if int(getattr(candidate, "turn_index", -1))
                == int(item["turn_index"])
            ),
            None,
        )
        transition = _valid_pending_transition(event) if event else None
        execution_order = set(
            dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
                "execution_order"
            )
            or ()
        ) if event else set()
        resolved_action_id = str(receipt.get("resolved_action_id") or "")
        if (
            transition
            and receipt.get("resolution_path") == "exact_contract"
            and receipt.get("pending_id") == transition.get("before_id")
            and receipt.get("pending_group") == transition.get("before_group")
            and receipt.get("pending_contract_hash")
            == transition.get("before_hash")
            and resolved_action_id
            and resolved_action_id in execution_order
            and resolved_action_id
            in set(transition.get("consumer_action_ids") or ())
        ):
            matches.append(item)
    return bool(matches) and not invalid, _receipt_details(
        "visible_option_binding",
        matches,
        invalid,
        matched_receipt_count=len(matches),
    )


def _unhandled_visible_option(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "pending_resolution")
    unhandled = []
    for item in valid:
        receipt = item["receipt"]
        if (
            receipt.get("resolution_path") != "exact_contract"
            or not receipt.get("selected_option_id")
        ):
            continue
        event = next(
            (
                candidate
                for candidate in _events(context)
                if int(getattr(candidate, "turn_index", -1))
                == int(item["turn_index"])
            ),
            None,
        )
        execution_order = tuple(
            dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
                "execution_order"
            )
            or ()
        ) if event else ()
        if str(receipt.get("resolved_action_id") or "") not in execution_order:
            unhandled.append(item)
    return bool(unhandled) and not invalid, _receipt_details(
        "visible_option_binding",
        unhandled,
        invalid,
        unhandled_count=len(unhandled),
    )


def _stale_menu_choice_applied(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "pending_resolution")
    stale = []
    for item in valid:
        receipt = item["receipt"]
        event = next(
            (
                candidate
                for candidate in _events(context)
                if int(getattr(candidate, "turn_index", -1))
                == int(item["turn_index"])
            ),
            None,
        )
        transition = _valid_pending_transition(event) if event else None
        if transition and (
            receipt.get("pending_id") != transition.get("before_id")
            or receipt.get("pending_contract_hash")
            != transition.get("before_hash")
        ):
            stale.append(item)
    return bool(stale) and not invalid, _receipt_details(
        "visible_option_binding",
        stale,
        invalid,
        stale_count=len(stale),
    )


def _source_step_event(context: Any, position: int) -> Any | None:
    source_step_id = f"source-{position}"
    valid_attestations, invalid_attestations = collect_valid_variant_attestations(
        context
    )
    if invalid_attestations:
        return None
    attested = [
        int(item["turn_index"])
        for item in valid_attestations
        if item.get("source_step_id") == source_step_id
    ]
    if len(attested) > 1:
        return None
    if attested:
        turn_index = attested[0]
    else:
        verifier_input = getattr(context, "verifier_input_contract", {}) or {}
        source_steps = (
            (verifier_input.get("source_contract") or {}).get("source_steps")
            or ()
        )
        source_step = next(
            (
                item
                for item in source_steps
                if isinstance(item, Mapping)
                and item.get("step_id") == source_step_id
            ),
            None,
        )
        turns = _turns(context)
        if not isinstance(source_step, Mapping):
            if verifier_input:
                return None
            source_turn_position = position
        else:
            source_turn_position = int(source_step.get("turn_index") or -1)
        if source_turn_position < 1 or source_turn_position > len(turns):
            return None
        turn_index = int(
            getattr(turns[source_turn_position - 1], "turn_index", -1)
        )
    return next(
        (
            event
            for event in _events(context)
            if int(getattr(event, "turn_index", -2)) == turn_index
        ),
        None,
    )


def _planned_action_types(event: Any) -> tuple[str, ...]:
    if event is None:
        return ()
    planned: list[str] = []
    for receipt in tuple(getattr(event, "control_receipts", ()) or ()):
        if not isinstance(receipt, Mapping):
            continue
        if receipt.get("receipt_type") != "semantic_planner":
            continue
        accepted, _ = validate_persisted_domain_control_receipt(
            receipt,
            turn_index=int(getattr(event, "turn_index", -1)),
        )
        if accepted:
            planned.extend(
                str(action_type)
                for action_type in receipt.get("planned_action_types") or ()
            )
    return tuple(planned)


def _value_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _admitted_actions(
    event: Any,
    action_type: str,
) -> tuple[Mapping[str, Any], ...]:
    if event is None:
        return ()
    return tuple(
        item
        for item in tuple(
            getattr(event, "admitted_action_provenance", ()) or ()
        )
        if isinstance(item, Mapping)
        and item.get("type") == action_type
    )


def _mode_change_request_routed_from_chain_pending(
    context: Any,
) -> PredicateResult:
    event = _source_step_event(context, 3)
    transition = _valid_pending_transition(event) if event else None
    owner_bindings = dict(
        dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
            "owner_bindings"
        )
        or {}
    ) if event else {}
    chain_paths = sorted(
        path
        for path in _valid_material_diffs(event)
        if _path_root(path) == "chain_identity"
    ) if event else []
    admitted = _admitted_actions(event, "choose_target_mode")
    transition_consumers = set(
        (transition or {}).get("consumer_action_ids") or ()
    )
    execution_order = set(
        dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
            "execution_order"
        )
        or ()
    ) if event else set()
    matching_actions = [
        action
        for action in admitted
        if action.get("owner") == "chain_rpc"
        and action.get("effect") == "configuration_mutation"
        and action.get("group") == "target_mode"
        and dict(action.get("argument_value_hashes") or {}).get("target_mode")
        == _value_hash("fake-node")
        and str(action.get("action_id") or "") in transition_consumers
        and str(action.get("action_id") or "") in execution_order
    ]
    satisfied = bool(
        transition
        and transition["before_id"] == "chain"
        and transition["before_group"] == "chain_identity"
        and transition["after_id"] == "target_mode_change_confirm"
        and transition["after_group"] == "target_mode"
        and "choose_target_mode" in _planned_action_types(event)
        and "chain_rpc" in set(owner_bindings.values())
        and len(matching_actions) == 1
        and not chain_paths
    )
    return satisfied, {
        "evidence_family": "turn_bound_domain_routing",
        "source_step_position": 3,
        "turn_index": int(getattr(event, "turn_index", -1)) if event else -1,
        "planned_action_types": list(_planned_action_types(event)),
        "pending_transition": dict(transition or {}),
        "owner_bindings": owner_bindings,
        "matching_admitted_action_ids": [
            str(action.get("action_id") or "")
            for action in matching_actions
        ],
        "chain_identity_changed_paths": chain_paths,
    }


def _declined_mode_change_resumes_chain_pending(
    context: Any,
) -> PredicateResult:
    event = _source_step_event(context, 4)
    transition = _valid_pending_transition(event) if event else None
    valid, invalid = _valid_receipts(context, "pending_resolution")
    turn_index = int(getattr(event, "turn_index", -1)) if event else -1
    matching = [
        item
        for item in valid
        if int(item["turn_index"]) == turn_index
        and item["receipt"].get("pending_id") == "target_mode_change_confirm"
        and item["receipt"].get("selected_option_id") == "no"
    ]
    admitted_ids = set(
        dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
            "admitted_action_ids"
        )
        or ()
    ) if event else set()
    execution_order = set(
        dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
            "execution_order"
        )
        or ()
    ) if event else set()
    transition_consumers = set(
        (transition or {}).get("consumer_action_ids") or ()
    )
    matching = [
        item
        for item in matching
        if str(item["receipt"].get("resolved_action_id") or "")
        in admitted_ids & execution_order & transition_consumers
    ]
    target_mode_paths = sorted(
        path
        for path in _valid_material_diffs(event)
        if path == "target_mode" or path.startswith("target_mode.")
    ) if event else []
    satisfied = bool(
        transition
        and transition["before_id"] == "target_mode_change_confirm"
        and transition["before_group"] == "target_mode"
        and transition["after_id"] == "chain"
        and transition["after_group"] == "chain_identity"
        and matching
        and not invalid
        and not target_mode_paths
    )
    return satisfied, _receipt_details(
        "turn_bound_pending_resume",
        matching,
        invalid,
        source_step_position=4,
        turn_index=turn_index,
        pending_transition=dict(transition or {}),
        target_mode_changed_paths=target_mode_paths,
    )


def _mode_request_consumed_as_chain_identity(
    context: Any,
) -> PredicateResult:
    event = _source_step_event(context, 3)
    transition = _valid_pending_transition(event) if event else None
    chain_paths = sorted(
        path
        for path in _valid_material_diffs(event)
        if _path_root(path) == "chain_identity"
    ) if event else []
    observed = bool(
        chain_paths
        or (
            transition
            and transition["after_group"] == "chain_identity"
            and transition["after_id"] != "chain"
        )
        or "answer_pending" in _planned_action_types(event)
    )
    return observed, {
        "evidence_family": "turn_bound_domain_misroute",
        "source_step_position": 3,
        "turn_index": int(getattr(event, "turn_index", -1)) if event else -1,
        "planned_action_types": list(_planned_action_types(event)),
        "pending_transition": dict(transition or {}),
        "chain_identity_changed_paths": chain_paths,
    }


def _unknown_chain_identity_resolution_started(context: Any) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="unknown_chain_identity",
        receipt_types=("chain_identity_resolution",),
        predicate=lambda receipt: bool(receipt.get("resolver_source")),
    )


def _chain_confirmation_required(context: Any) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="unknown_chain_identity",
        receipt_types=("chain_identity_resolution",),
        predicate=lambda receipt: receipt.get("confirmation_required") is True,
    )


def _unknown_chain_silently_configured(context: Any) -> PredicateResult:
    identities, invalid_identity = _valid_receipts(
        context, "chain_identity_resolution"
    )
    deltas, invalid_commits = _material_delta_records(context)
    identity_turns = {int(item["turn_index"]) for item in identities}
    silent = [
        item
        for item in deltas
        if int(item["turn_index"]) in identity_turns
        and item["operation"] == "write"
        and _path_root(item["path"]) == "chain_identity"
        and not any(
            (
                transition := _valid_pending_transition(event)
            )
            and transition["after_id"]
            for event in _events(context)
            if int(getattr(event, "turn_index", -1))
            == int(item["turn_index"])
        )
    ]
    invalid = [*invalid_identity, *invalid_commits]
    return bool(silent) and not invalid, _receipt_details(
        "unknown_chain_identity",
        silent,
        invalid,
        silent_commit_count=len(silent),
    )


def _chain_mode_change_confirmed(context: Any) -> PredicateResult:
    deltas, invalid = _material_delta_records(context)
    roles_by_turn, invalid_attestations, role_error = _semantic_roles_by_turn(
        context
    )
    events_by_turn = {
        int(getattr(event, "turn_index", -1)): event
        for event in _events(context)
    }
    by_turn: dict[int, set[str]] = defaultdict(set)
    for item in deltas:
        path = str(item["path"])
        if path == "chain_identity.canonical":
            by_turn[int(item["turn_index"])].add("chain_identity")
        elif path in {"target_mode", "workflow_mode"}:
            by_turn[int(item["turn_index"])].add(path)
    confirmed = {
        turn: sorted(roots)
        for turn, roots in by_turn.items()
        if "chain_identity" in roots
        or {"target_mode", "workflow_mode"} & roots
    }
    compound_requests: list[dict[str, Any]] = []
    for event in _events(context):
        turn_index = int(getattr(event, "turn_index", -1))
        if "chain_mode_change_request" not in roles_by_turn.get(
            turn_index, set()
        ):
            continue
        provenance = [
            action
            for action in tuple(
                getattr(event, "admitted_action_provenance", ()) or ()
            )
            if isinstance(action, Mapping)
            and str(action.get("action_id") or "")
            and str(action.get("owner") or "") == "chain_rpc"
        ]
        chain_actions = {
            str(action["action_id"]): str(action["type"])
            for action in provenance
            if str(action.get("type") or "") in {
                "change_chain",
                "choose_chain",
            }
        }
        mode_actions = {
            str(action["action_id"]): str(action["type"])
            for action in provenance
            if str(action.get("type") or "") in {
                "choose_target_mode",
                "request_target_mode_selection",
            }
        }
        if chain_actions and mode_actions:
            compound_requests.append({
                "turn_index": turn_index,
                "chain_action_ids": sorted(chain_actions),
                "chain_actions": sorted(chain_actions.values()),
                "mode_action_ids": sorted(mode_actions),
                "mode_actions": sorted(mode_actions.values()),
            })
    commit_records = [
        item
        for item in deltas
        if item["receipt"].get("completion") != "rejected"
    ]
    mode_progress_turns: set[int] = set()
    chain_progress_turns: set[int] = set()
    completed_compound_turns: set[int] = set()
    for request in compound_requests:
        compound_turn = int(request["turn_index"])
        chain_ids = set(request["chain_action_ids"])
        mode_ids = set(request["mode_action_ids"])
        staged_chain_turns = {
            int(item["turn_index"])
            for item in commit_records
            if int(item["turn_index"]) >= compound_turn
            and chain_ids.intersection(
                str(value)
                for value in item["receipt"].get("consumed_action_ids") or ()
            )
            and str(item["path"])
            == "chain_identity.change_candidate.canonical"
        }
        mode_turns: set[int] = set()
        for turn_index, event in events_by_turn.items():
            if turn_index < compound_turn:
                continue
            transition = _valid_pending_transition(event)
            if (
                transition is not None
                and transition["after_group"] == "target_mode"
                and transition["after_id"]
                and mode_ids.intersection(
                    str(value)
                    for value in transition["consumer_action_ids"]
                )
            ):
                mode_turns.add(turn_index)
        chain_progress_turns.update(staged_chain_turns)
        mode_progress_turns.update(mode_turns)
        if staged_chain_turns and mode_turns:
            completed_compound_turns.add(compound_turn)
    correlated_changes = {
        turn: roots
        for turn, roots in confirmed.items()
        if any(
            turn >= int(request["turn_index"])
            for request in compound_requests
        )
    }
    complete = (
        bool(completed_compound_turns)
        and not invalid
        and not invalid_attestations
        and not role_error
    )
    return complete, _receipt_details(
        "registry_invalidation",
        deltas,
        invalid,
        changed_roots_by_turn=correlated_changes,
        compound_requests=compound_requests,
        chain_progress_turns=sorted(chain_progress_turns),
        mode_progress_turns=sorted(mode_progress_turns),
        completed_compound_turns=sorted(completed_compound_turns),
        invalid_attestations=invalid_attestations,
        role_error=role_error,
    )


def _incompatible_state_invalidated(context: Any) -> PredicateResult:
    valid, invalid = _domain_commits(context)
    matched = [
        item
        for item in valid
        if item["receipt"].get("completion") != "rejected"
        and (
            item["receipt"].get("invalidated_groups")
            or item["receipt"].get("invalidated_fields")
        )
    ]
    return bool(matched) and not invalid, _receipt_details(
        "registry_invalidation",
        matched,
        invalid,
        invalidation_count=len(matched),
    )


def _fallback_resumed(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "fallback_selection")
    matches = []
    for item in valid:
        selected = str(item["receipt"].get("selected_group") or "")
        event = next(
            (
                candidate
                for candidate in _events(context)
                if int(getattr(candidate, "turn_index", -1))
                == int(item["turn_index"])
            ),
            None,
        )
        if selected and event and selected in {
            str(getattr(event, "active_group", "") or ""),
            str(getattr(event, "pending_contract", {}).get("group") or ""),
        }:
            matches.append(item)
    return bool(matches) and not invalid, _receipt_details(
        "fallback_selection",
        matches,
        invalid,
        matched_receipt_count=len(matches),
    )


def _stale_fallback_emitted(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "fallback_selection")
    stale = []
    for item in valid:
        selected = str(item["receipt"].get("selected_group") or "")
        event = next(
            (
                candidate
                for candidate in _events(context)
                if int(getattr(candidate, "turn_index", -1))
                == int(item["turn_index"])
            ),
            None,
        )
        if selected and event and selected not in {
            str(getattr(event, "active_group", "") or ""),
            str(getattr(event, "pending_contract", {}).get("group") or ""),
        }:
            stale.append(item)
    return bool(stale) and not invalid, _receipt_details(
        "fallback_selection",
        stale,
        invalid,
        stale_count=len(stale),
    )


def _confirmed_pending_lineage(
    context: Any,
    *,
    expected_field: str,
    expected_role: str,
    expected_group: str,
    allowed_resolution_paths: frozenset[str],
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    Mapping[str, Any],
]:
    """Bind one semantic source turn to its accepted configuration write."""

    roles_by_turn, invalid_attestations, role_error = _semantic_roles_by_turn(
        context
    )
    pending, invalid_pending = _valid_receipts(
        context,
        "pending_resolution",
    )
    commits, invalid_commits = _domain_commits(context)
    turns = {
        int(getattr(turn, "turn_index", -1)): turn
        for turn in _turns(context)
    }
    commits_by_turn: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in commits:
        commits_by_turn[int(item["turn_index"])].append(item)

    matches: list[Mapping[str, Any]] = []
    for item in pending:
        receipt = item["receipt"]
        turn_index = int(item["turn_index"])
        turn = turns.get(turn_index)
        event = next(
            (
                candidate
                for candidate in _events(context)
                if int(getattr(candidate, "turn_index", -1)) == turn_index
            ),
            None,
        )
        if turn is None or event is None:
            continue
        transition = _valid_pending_transition(event)
        summary = dict(getattr(event, "turn_receipt_summary", {}) or {})
        input_hash = hashlib.sha256(
            str(getattr(turn, "user_message", "")).encode("utf-8")
        ).hexdigest()
        action_id = str(receipt.get("resolved_action_id") or "")
        pending_hash = str(receipt.get("pending_contract_hash") or "")
        expected_owner = str(GROUP_OWNER.get(expected_group) or "")
        if (
            receipt.get("verdict") != "accepted"
            or receipt.get("pending_id") != expected_field
            or receipt.get("pending_group") != expected_group
            or receipt.get("resolution_path") not in allowed_resolution_paths
            or receipt.get("input_hash") != input_hash
            or expected_role not in roles_by_turn.get(turn_index, set())
            or not action_id
            or transition is None
            or transition.get("before_id") != expected_field
            or transition.get("before_group") != expected_group
            or transition.get("before_hash") != pending_hash
            or action_id not in set(transition.get("consumer_action_ids") or ())
            or action_id not in set(summary.get("admitted_action_ids") or ())
            or dict(summary.get("owner_bindings") or {}).get(action_id)
            != expected_owner
        ):
            continue
        matching_commits = []
        expected_path = f"confirmed_config.{expected_field}"
        material_diff = _valid_material_diffs(event).get(expected_path)
        selected_value_hash = str(receipt.get("selected_value_hash") or "")
        for commit_item in commits_by_turn.get(turn_index, ()):
            commit = commit_item["receipt"]
            matching_deltas = [
                delta
                for delta in commit.get("material_delta") or ()
                if (
                delta.get("operation") == "write"
                and delta.get("path") == expected_path
                )
            ]
            if (
                commit.get("completion") != "rejected"
                and commit.get("owner") == expected_owner
                and commit.get("pending_before_hash") == pending_hash
                and action_id in set(commit.get("consumed_action_ids") or ())
                and len(matching_deltas) == 1
                and material_diff is not None
                and selected_value_hash
                and matching_deltas[0].get("value_hash")
                == selected_value_hash
                == material_diff.get("after")
            ):
                matching_commits.append(commit_item)
        if len(matching_commits) == 1:
            matches.append({
                "field": expected_field,
                "turn_index": turn_index,
                "pending_receipt": receipt,
                "pending_receipt_id": item["receipt_id"],
                "domain_commit": matching_commits[0]["receipt"],
                "domain_commit_receipt_id": matching_commits[0]["receipt_id"],
            })

    invalid = [*invalid_pending, *invalid_commits]
    if invalid_attestations or role_error:
        invalid.append({
            "receipt_id": "",
            "reason": role_error or "invalid semantic-role attestation",
        })
    return matches, invalid, {
        "expected_field": expected_field,
        "expected_role": expected_role,
        "invalid_attestations": invalid_attestations,
        "role_error": role_error,
    }


def _disk_limit_lineage(
    context: Any,
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    Mapping[str, Any],
]:
    matches: list[Mapping[str, Any]] = []
    invalid: list[Mapping[str, Any]] = []
    observations: dict[str, Any] = {}
    roles = {
        "DATA_VOL_MAX_IOPS": "provide_disk_iops",
        "DATA_VOL_MAX_THROUGHPUT": "provide_disk_throughput",
    }
    for field, role in roles.items():
        field_matches, field_invalid, field_observations = (
            _confirmed_pending_lineage(
                context,
                expected_field=field,
                expected_role=role,
                expected_group="ledger_disk",
                allowed_resolution_paths=frozenset({"typed_manual_value"}),
            )
        )
        matches.extend(field_matches)
        invalid.extend(field_invalid)
        observations[field] = field_observations
    return matches, invalid, {"field_lineage": observations}


def _copied_scalar_normalized(context: Any) -> PredicateResult:
    matches, invalid, observations = _confirmed_pending_lineage(
        context,
        expected_field="DATA_VOL_TYPE",
        expected_role="provide_disk_type",
        expected_group="ledger_disk",
        allowed_resolution_paths=frozenset({"typed_manual_value"}),
    )
    normalized = [
        item for item in matches
        if item["pending_receipt"].get("normalizer") not in {"", "identity"}
    ]
    return bool(normalized) and not invalid, _receipt_details(
        "scalar_normalization",
        normalized,
        invalid,
        **observations,
        matched_receipt_count=len(normalized),
    )


def _typed_detected_value_confirmed(context: Any) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="pending_lineage",
        receipt_types=("pending_resolution",),
        predicate=lambda receipt: (
            receipt.get("resolution_path") == "exact_contract"
            and bool(receipt.get("selected_value_hash"))
        ),
    )


def _disk_size_resolved(context: Any) -> PredicateResult:
    matches, invalid, observations = _confirmed_pending_lineage(
        context,
        expected_field="DATA_VOL_SIZE",
        expected_role="provide_disk_size",
        expected_group="ledger_disk",
        allowed_resolution_paths=frozenset({
            "typed_manual_value",
            "exact_contract",
        }),
    )
    resolved = [
        item for item in matches
        if (
            item["pending_receipt"].get("resolution_path")
            == "typed_manual_value"
            or bool(item["pending_receipt"].get("selected_option_id"))
        )
        and bool(item["pending_receipt"].get("selected_value_hash"))
    ]
    return bool(resolved) and not invalid, _receipt_details(
        "pending_lineage",
        resolved,
        invalid,
        **observations,
        matched_receipt_count=len(resolved),
    )


def _disk_limits_collected_once(context: Any) -> PredicateResult:
    matches, invalid, observations = _disk_limit_lineage(context)
    counts = Counter(item["field"] for item in matches)
    satisfied = set(counts) == set(_DISK_LIMIT_FIELDS) and all(
        count == 1 for count in counts.values()
    )
    return satisfied and not invalid, _receipt_details(
        "subgroup_progression",
        matches,
        invalid,
        **observations,
        field_write_counts=dict(counts),
    )


def _disk_subgroup_repeated(context: Any) -> PredicateResult:
    matches, invalid, observations = _disk_limit_lineage(context)
    counts = Counter(item["field"] for item in matches)
    repeated = {
        field: count for field, count in counts.items() if count > 1
    }
    return bool(repeated) and not invalid, _receipt_details(
        "subgroup_progression",
        matches,
        invalid,
        **observations,
        repeated_field_writes=repeated,
    )


def _typed_confirmation_rejected_by_side_channel(
    context: Any,
) -> PredicateResult:
    pending, invalid_pending = _valid_receipts(context, "pending_resolution")
    commits, invalid_commits = _domain_commits(context)
    accepted_turns = {int(item["turn_index"]) for item in pending}
    rejected = [
        item
        for item in commits
        if int(item["turn_index"]) in accepted_turns
        and item["receipt"].get("completion") == "rejected"
    ]
    invalid = [*invalid_pending, *invalid_commits]
    return bool(rejected) and not invalid, _receipt_details(
        "pending_commit_lineage",
        rejected,
        invalid,
        rejected_after_accepted_count=len(rejected),
    )


def _owned_group_backtrack_completed(context: Any) -> PredicateResult:
    valid, invalid = _domain_commits(context)
    matched = [
        item
        for item in valid
        if item["receipt"].get("completion") != "rejected"
        and item["receipt"].get("navigation_operation") == "go_back"
        and item["receipt"].get("navigation_origin_group")
        and item["receipt"].get("navigation_target_group")
    ]
    return bool(matched) and not invalid, _receipt_details(
        "navigation",
        matched,
        invalid,
        matched_navigation_count=len(matched),
    )


def _backtrack_lost_configuration_state(context: Any) -> PredicateResult:
    valid, invalid = _domain_commits(context)
    lost = []
    for item in valid:
        receipt = item["receipt"]
        if receipt.get("navigation_operation") != "go_back":
            continue
        destructive = [
            delta
            for delta in receipt.get("material_delta") or ()
            if delta.get("operation") == "delete"
            and _path_root(str(delta.get("path") or ""))
            in DURABLE_ENVIRONMENT_STATE_ROOTS
        ]
        if destructive:
            lost.append(item)
    return bool(lost) and not invalid, _receipt_details(
        "navigation",
        lost,
        invalid,
        destructive_backtrack_count=len(lost),
    )


def _new_chain_request_routed(context: Any) -> PredicateResult:
    identity, invalid_identity = _valid_receipts(
        context, "chain_identity_resolution"
    )
    routed_actions = []
    for event in _events(context):
        for target in tuple(
            getattr(event, "admitted_action_targets", ()) or ()
        ):
            if (
                isinstance(target, Mapping)
                and target.get("type") == "change_group"
                and target.get("group") == "chain_identity"
            ):
                routed_actions.append(target)
        if set(getattr(event, "admitted_action_types", ()) or ()) & {
            "set_chain_candidate",
            "change_chain",
            "select_chain",
        }:
            routed_actions.append({
                "type": "chain_identity_action",
                "turn_index": int(getattr(event, "turn_index", -1)),
            })
    return bool(identity or routed_actions) and not invalid_identity, _receipt_details(
        "chain_request_routing",
        identity,
        invalid_identity,
        routed_action_count=len(routed_actions),
    )


def _stale_preflight_executed(context: Any) -> PredicateResult:
    commits, invalid = _domain_commits(context)
    invalidation_turns = {
        int(item["turn_index"])
        for item in commits
        if "preflight_smoke_execution"
        in set(item["receipt"].get("invalidated_groups") or ())
    }
    stale = []
    for event in _events(context):
        if int(getattr(event, "turn_index", -1)) not in invalidation_turns:
            continue
        summary = dict(getattr(event, "execution_receipt_summary", {}) or {})
        if summary.get("manager_submission_receipt_id"):
            stale.append(int(getattr(event, "turn_index", -1)))
    return bool(stale) and not invalid, _receipt_details(
        "execution_invalidation",
        commits,
        invalid,
        stale_submission_turns=stale,
    )


def _custom_method_collection_exited(context: Any) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="rpc_catalog",
        receipt_types=("rpc_catalog_transition",),
        predicate=lambda receipt: (
            receipt.get("accepted") is True
            and receipt.get("finished") is True
            and int(receipt.get("method_count") or 0) > 0
        ),
    )


def _custom_method_collection_looped(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "rpc_catalog_transition")
    identities = Counter(
        (
            int(item["receipt"].get("catalog_revision") or 0),
            tuple(item["receipt"].get("method_hashes") or ()),
            str(item["receipt"].get("phase") or ""),
            bool(item["receipt"].get("finished")),
        )
        for item in valid
    )
    loops = {
        repr(identity): count
        for identity, count in identities.items()
        if count > 1 and identity[-1] is False
    }
    return bool(loops) and not invalid, _receipt_details(
        "rpc_catalog",
        valid,
        invalid,
        repeated_catalog_states=loops,
    )


def _effective_workload_commit_replaces_defaults(
    context: Any,
) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="rpc_workload",
        receipt_types=("rpc_workload_commit",),
        predicate=lambda receipt: (
            receipt.get("replace_defaults") is True
            and bool(receipt.get("method_hashes"))
        ),
    )


def _removed_default_method_committed(context: Any) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="rpc_workload",
        receipt_types=("rpc_workload_commit",),
        predicate=lambda receipt: (
            receipt.get("choice") == "custom"
            and receipt.get("replace_defaults") is not True
            and bool(receipt.get("method_hashes"))
        ),
    )


def _example_endpoint_scope_preserved(context: Any) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="rpc_endpoint_scope",
        receipt_types=("rpc_endpoint_role",),
        predicate=lambda receipt: receipt.get("role") == "validation",
    )


def _example_endpoint_replaced_runtime_endpoint(context: Any) -> PredicateResult:
    endpoints, invalid_endpoints = _valid_receipts(context, "rpc_endpoint_role")
    deltas, invalid_commits = _material_delta_records(context)
    validation_turns = {
        int(item["turn_index"])
        for item in endpoints
        if item["receipt"].get("role") == "validation"
    }
    replaced = [
        item
        for item in deltas
        if int(item["turn_index"]) in validation_turns
        and item["operation"] == "write"
        and _path_leaf(item["path"])
        in {"LOCAL_RPC_URL", "MAINNET_RPC_URL", "SYNC_ENDPOINT"}
    ]
    invalid = [*invalid_endpoints, *invalid_commits]
    return bool(replaced) and not invalid, _receipt_details(
        "rpc_endpoint_scope",
        replaced,
        invalid,
        runtime_endpoint_write_count=len(replaced),
    )


def _rpc_schema_evidence_extracted(context: Any) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="rpc_schema_provenance",
        receipt_types=("rpc_schema_provenance",),
        predicate=lambda receipt: bool(receipt.get("fields")),
    )


def _schema_intake_looped(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "rpc_schema_provenance")
    identities = Counter(
        (
            str(item["receipt"].get("method_hash") or ""),
            str(item["receipt"].get("fields_hash") or ""),
            int(item["receipt"].get("catalog_revision") or 0),
        )
        for item in valid
    )
    loops = {
        repr(identity): count
        for identity, count in identities.items()
        if count > 1
    }
    return bool(loops) and not invalid, _receipt_details(
        "rpc_schema_provenance",
        valid,
        invalid,
        repeated_schema_states=loops,
    )


def _multiline_evidence_block_collected_once(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "analysis_evidence_block")
    completed = [
        item
        for item in valid
        if item["receipt"].get("operation") == "finish"
        and int(item["receipt"].get("line_count") or 0) > 1
        and item["receipt"].get("status") in {"saved", "pending_answer"}
    ]
    block_counts = Counter(
        str(item["receipt"].get("block_id") or "") for item in completed
    )
    satisfied = bool(block_counts) and all(
        count == 1 for count in block_counts.values()
    )
    return satisfied and not invalid, _receipt_details(
        "analysis_evidence_block",
        completed,
        invalid,
        completed_block_counts=dict(block_counts),
    )


def _blank_prompt_counted_as_evidence(context: Any) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="analysis_evidence_block",
        receipt_types=("analysis_evidence_block",),
        predicate=lambda receipt: (
            receipt.get("input_disposition") == "accepted"
            and int(receipt.get("input_non_empty_line_count") or 0) == 0
        ),
    )


def _ordinary_question_counted_as_evidence(context: Any) -> PredicateResult:
    evidence, invalid_evidence = _valid_receipts(
        context, "analysis_evidence_block"
    )
    orientation, invalid_orientation = _valid_receipts(
        context, "orientation_response"
    )
    orientation_turns = {int(item["turn_index"]) for item in orientation}
    counted = [
        item
        for item in evidence
        if int(item["turn_index"]) in orientation_turns
        and item["receipt"].get("input_disposition") == "accepted"
        and int(item["receipt"].get("input_non_empty_line_count") or 0) > 0
    ]
    invalid = [*invalid_evidence, *invalid_orientation]
    return bool(counted) and not invalid, _receipt_details(
        "analysis_evidence_classification",
        counted,
        invalid,
        counted_consultation_turns=sorted({
            int(item["turn_index"]) for item in counted
        }),
    )


def _evidence_analysis_returned(context: Any) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="analysis_invocation",
        receipt_types=("analysis_invocation", "analysis_report"),
        predicate=lambda receipt: (
            receipt.get("invoked") is True
            and receipt.get("visible_result_hash")
            and receipt.get("visible_result_hash")
            != hashlib.sha256(b'""').hexdigest()
        ),
    )


def _execution_stage_explained(context: Any) -> PredicateResult:
    return _receipt_predicate(
        context,
        family="orientation_read_only",
        receipt_types=("orientation_response",),
        predicate=lambda receipt: (
            receipt.get("topic") == "execution_preflight_smoke"
            and receipt.get("read_only") is True
        ),
    )


def _approved_execution_submitted_once(context: Any) -> PredicateResult:
    submissions: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for event in _events(context):
        summary = dict(
            getattr(event, "execution_receipt_summary", {}) or {}
        )
        key = str(
            summary.get("receipt_idempotency_key")
            or summary.get("intent_idempotency_key")
            or ""
        )
        manager_receipt = str(
            summary.get("manager_submission_receipt_id") or ""
        )
        job_id = str(summary.get("job_id") or "")
        if (
            key
            and _is_hash(manager_receipt)
            and job_id
            and summary.get("manager_submission_disposition")
            in {"created", "reused"}
            and int(summary.get("manager_matching_job_count") or 0) == 1
        ):
            submissions[key].add((manager_receipt, job_id))
    satisfied = len(submissions) == 1 and all(
        len(identities) == 1 for identities in submissions.values()
    )
    return satisfied, {
        "evidence_family": "job_submission",
        "submission_identities": {
            key: sorted(identities)
            for key, identities in submissions.items()
        },
    }


def _duplicate_job_submission(context: Any) -> PredicateResult:
    submissions: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for event in _events(context):
        summary = dict(
            getattr(event, "execution_receipt_summary", {}) or {}
        )
        key = str(
            summary.get("receipt_idempotency_key")
            or summary.get("intent_idempotency_key")
            or ""
        )
        identity = (
            str(summary.get("manager_submission_receipt_id") or ""),
            str(summary.get("job_id") or ""),
        )
        if key and any(identity):
            submissions[key].add(identity)
    duplicates = {
        key: len(identities)
        for key, identities in submissions.items()
        if len(identities) > 1
    }
    return bool(duplicates), {
        "evidence_family": "job_submission",
        "duplicate_submission_counts": duplicates,
    }


def _status_uses_job_evidence(context: Any) -> PredicateResult:
    matched = []
    for event in _events(context):
        summary = dict(
            getattr(event, "execution_receipt_summary", {}) or {}
        )
        if (
            summary.get("job_id")
            and _is_hash(summary.get("manager_read_receipt_id"))
            and summary.get("manager_observed_status")
            == summary.get("job_status")
        ):
            matched.append(int(getattr(event, "turn_index", -1)))
    return bool(matched), {
        "evidence_family": "job_read",
        "verified_status_turn_indexes": matched,
    }


def _invented_job_status(context: Any) -> PredicateResult:
    invented = []
    for event in _events(context):
        summary = dict(
            getattr(event, "execution_receipt_summary", {}) or {}
        )
        if (
            summary.get("intent_action_type")
            in {"status", "read_status", "analyze_job"}
            and summary.get("job_status")
            and not _is_hash(summary.get("manager_read_receipt_id"))
        ):
            invented.append(int(getattr(event, "turn_index", -1)))
    return bool(invented), {
        "evidence_family": "job_read",
        "unverified_status_turn_indexes": invented,
    }


def _compatible_environment_retained(context: Any) -> PredicateResult:
    commits, invalid = _domain_commits(context)
    mode_turns = {
        int(item["turn_index"])
        for item in commits
        if item["receipt"].get("completion") != "rejected"
        and any(
            _path_root(str(delta.get("path") or ""))
            in {"target_mode", "workflow_mode"}
            for delta in item["receipt"].get("material_delta") or ()
        )
    }
    environment_changes = {
        int(getattr(event, "turn_index", -1)): sorted(
            path
            for path in _valid_material_diffs(event)
            if _path_root(path) in DURABLE_ENVIRONMENT_STATE_ROOTS
        )
        for event in _events(context)
        if int(getattr(event, "turn_index", -1)) in mode_turns
    }
    retained = bool(mode_turns) and all(
        not paths for paths in environment_changes.values()
    )
    return retained and not invalid, _receipt_details(
        "mode_compatibility",
        commits,
        invalid,
        mode_change_turns=sorted(mode_turns),
        environment_changes=environment_changes,
    )


def _mode_specific_state_invalidated(context: Any) -> PredicateResult:
    commits, invalid = _domain_commits(context)
    matched = [
        item
        for item in commits
        if item["receipt"].get("completion") != "rejected"
        and item["receipt"].get("invalidated_groups")
        and any(
            _path_root(str(delta.get("path") or ""))
            in {"target_mode", "workflow_mode"}
            for delta in item["receipt"].get("material_delta") or ()
        )
    ]
    return bool(matched) and not invalid, _receipt_details(
        "mode_compatibility",
        matched,
        invalid,
        matched_receipt_count=len(matched),
    )


def _rpc_groups_rerequired(context: Any) -> PredicateResult:
    commits, invalid = _domain_commits(context)
    rerequired = [
        item
        for item in commits
        if _RPC_GROUPS
        & set(item["receipt"].get("invalidated_groups") or ())
    ]
    return bool(rerequired) and not invalid, _receipt_details(
        "mode_compatibility",
        rerequired,
        invalid,
        matched_receipt_count=len(rerequired),
    )


def _sync_observe_ran_rpc_load(context: Any) -> PredicateResult:
    rpc_operations = {
        "fake_node_smoke",
        "real_node_smoke",
        "final_benchmark",
    }
    sync_hash = content_hash("sync_observe")
    violations: list[dict[str, Any]] = []
    for event in _events(context):
        summary = dict(
            getattr(event, "execution_receipt_summary", {}) or {}
        )
        operation = str(summary.get("intent_action_type") or "")
        workflow_hash = str(
            getattr(event, "after_value_hashes", {}).get("workflow_mode")
            or ""
        )
        submission_receipt_id = str(
            summary.get("manager_submission_receipt_id") or ""
        )
        read_receipt_id = str(summary.get("manager_read_receipt_id") or "")
        completed = (
            summary.get("manager_observed_status") == "completed"
            and summary.get("job_status") == "completed"
        )
        if (
            workflow_hash == sync_hash
            and operation in rpc_operations
            and _is_hash(submission_receipt_id)
            and _is_hash(read_receipt_id)
            and completed
        ):
            violations.append({
                "turn_index": int(getattr(event, "turn_index", -1)),
                "operation": operation,
                "workflow_mode_hash": workflow_hash,
                "manager_submission_receipt_id": submission_receipt_id,
                "manager_read_receipt_id": read_receipt_id,
            })
    return bool(violations), {
        "evidence_family": "execution_mode_policy",
        "violations": violations,
    }


def _semantic_units_partitioned_in_order(context: Any) -> PredicateResult:
    valid, invalid = _valid_receipts(context, "semantic_planner")
    matched = []
    for item in valid:
        event = next(
            (
                candidate
                for candidate in _events(context)
                if int(getattr(candidate, "turn_index", -1))
                == int(item["turn_index"])
            ),
            None,
        )
        summary = dict(
            getattr(event, "turn_receipt_summary", {}) or {}
        ) if event else {}
        units = tuple(summary.get("semantic_units") or ())
        unit_ids = [
            str(unit.get("unit_id") or "")
            for unit in units
            if isinstance(unit, Mapping)
        ]
        order = [str(value) for value in summary.get("semantic_order") or ()]
        spans = [
            (unit.get("start"), unit.get("end"))
            for unit in units
            if isinstance(unit, Mapping)
        ]
        if (
            unit_ids
            and unit_ids == order
            and len(unit_ids) == len(set(unit_ids))
            and all(
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end
                for start, end in spans
            )
            and all(
                spans[index][1] <= spans[index + 1][0]
                for index in range(len(spans) - 1)
            )
        ):
            matched.append(item)
    return bool(matched) and not invalid, _receipt_details(
        "semantic_units",
        matched,
        invalid,
        matched_receipt_count=len(matched),
    )


def _admitted_mutations_only(context: Any) -> PredicateResult:
    commits, invalid = _domain_commits(context)
    violations = []
    mutation_count = 0
    for item in commits:
        receipt = item["receipt"]
        if not receipt.get("material_delta"):
            continue
        mutation_count += 1
        event = next(
            (
                candidate
                for candidate in _events(context)
                if int(getattr(candidate, "turn_index", -1))
                == int(item["turn_index"])
            ),
            None,
        )
        admitted = set(
            dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
                "admitted_action_ids"
            )
            or ()
        ) if event else set()
        consumed = set(receipt.get("consumed_action_ids") or ())
        if not consumed or not consumed <= admitted:
            violations.append(str(item.get("receipt_id") or ""))
    return mutation_count > 0 and not violations and not invalid, _receipt_details(
        "admission_lineage",
        commits,
        invalid,
        mutation_count=mutation_count,
        violating_receipt_ids=violations,
    )


def _unresolved_units_preserved(context: Any) -> PredicateResult:
    preserved = []
    for event in _events(context):
        summary = dict(
            getattr(event, "turn_receipt_summary", {}) or {}
        )
        units = {
            str(item.get("unit_id") or "")
            for item in summary.get("semantic_units") or ()
            if isinstance(item, Mapping)
        }
        unresolved = set(summary.get("unresolved_unit_ids") or ())
        if unresolved and unresolved <= units:
            preserved.append(int(getattr(event, "turn_index", -1)))
    return bool(preserved), {
        "evidence_family": "semantic_units",
        "preserved_turn_indexes": preserved,
    }


def _ambiguous_change_silently_committed(context: Any) -> PredicateResult:
    commits, invalid = _domain_commits(context)
    violations = []
    for item in commits:
        event = next(
            (
                candidate
                for candidate in _events(context)
                if int(getattr(candidate, "turn_index", -1))
                == int(item["turn_index"])
            ),
            None,
        )
        summary = dict(
            getattr(event, "turn_receipt_summary", {}) or {}
        ) if event else {}
        unresolved = set(summary.get("unresolved_unit_ids") or ())
        action_units = dict(summary.get("action_unit_bindings") or {})
        consumed = set(item["receipt"].get("consumed_action_ids") or ())
        if any(
            unresolved & set(action_units.get(action_id) or ())
            for action_id in consumed
        ):
            violations.append(str(item.get("receipt_id") or ""))
    return bool(violations) and not invalid, _receipt_details(
        "semantic_units",
        commits,
        invalid,
        violating_receipt_ids=violations,
    )


def _semantic_unit_dropped(context: Any) -> PredicateResult:
    dropped: dict[int, list[str]] = {}
    for event in _events(context):
        summary = dict(
            getattr(event, "turn_receipt_summary", {}) or {}
        )
        units = {
            str(item.get("unit_id") or "")
            for item in summary.get("semantic_units") or ()
            if isinstance(item, Mapping)
            and str(item.get("unit_id") or "")
        }
        handled = set(summary.get("unresolved_unit_ids") or ())
        handled.update(
            str(unit_id)
            for unit_id, action_ids in dict(
                summary.get("unit_action_bindings") or {}
            ).items()
            if str(unit_id) and action_ids
        )
        missing = sorted(units - handled)
        if missing:
            dropped[int(getattr(event, "turn_index", -1))] = missing
    return bool(dropped), {
        "evidence_family": "semantic_units",
        "dropped_units_by_turn": dropped,
    }


POSTCONDITION_EVALUATORS: Mapping[str, PostconditionEvaluator] = {
    "exact_fixture_turns_observed": _exact_fixture_turns_observed,
    "isomorphic_meaning_attested": _isomorphic_meaning_attested,
    "adjacent_non_trigger_attested": _adjacent_non_trigger_attested,
    "neighboring_transition_attested": _neighboring_transition_attested,
    "current_turn_language_preserved": _current_turn_language_preserved,
    "orientation_answered_read_only": _orientation_answered_read_only,
    "consultation_answered_read_only": _consultation_answered_read_only,
    "retained_state_described": _retained_state_described,
    "resume_action_contract_exposed": _resume_action_contract_exposed,
    "workflow_state_mutated_by_consultation": (
        _workflow_state_mutated_by_consultation
    ),
    "blocking_question_preserved": _pending_preserved_with_receipt,
    "consultation_preserved_pending_work": _pending_preserved_with_receipt,
    "consultation_consumed_as_pending_value": (
        _consultation_consumed_as_pending_value
    ),
    "response_driven_selection_observed": (
        _response_driven_selection_observed
    ),
    "visible_option_action_executed": _visible_option_action_executed,
    "current_menu_binding_preserved": _current_menu_binding_preserved,
    "mode_change_request_routed_from_chain_pending": (
        _mode_change_request_routed_from_chain_pending
    ),
    "declined_mode_change_resumes_chain_pending": (
        _declined_mode_change_resumes_chain_pending
    ),
    "mode_request_consumed_as_chain_identity": (
        _mode_request_consumed_as_chain_identity
    ),
    "unhandled_visible_option": _unhandled_visible_option,
    "stale_menu_choice_applied": _stale_menu_choice_applied,
    "unknown_chain_identity_resolution_started": (
        _unknown_chain_identity_resolution_started
    ),
    "chain_confirmation_required": _chain_confirmation_required,
    "unknown_chain_silently_configured": _unknown_chain_silently_configured,
    "environment_text_consumed_as_region": (
        _environment_text_consumed_as_region
    ),
    "chain_mode_change_confirmed": _chain_mode_change_confirmed,
    "incompatible_state_invalidated": _incompatible_state_invalidated,
    "fallback_resumed": _fallback_resumed,
    "stale_fallback_emitted": _stale_fallback_emitted,
    "copied_scalar_normalized": _copied_scalar_normalized,
    "disk_size_resolved": _disk_size_resolved,
    "disk_limits_collected_once": _disk_limits_collected_once,
    "disk_subgroup_repeated": _disk_subgroup_repeated,
    "typed_detected_value_confirmed": _typed_detected_value_confirmed,
    "typed_confirmation_rejected_by_side_channel": (
        _typed_confirmation_rejected_by_side_channel
    ),
    "owned_group_backtrack_completed": _owned_group_backtrack_completed,
    "backtrack_lost_configuration_state": (
        _backtrack_lost_configuration_state
    ),
    "new_chain_request_routed": _new_chain_request_routed,
    "stale_preflight_executed": _stale_preflight_executed,
    "custom_method_collection_exited": _custom_method_collection_exited,
    "custom_method_collection_looped": _custom_method_collection_looped,
    "effective_workload_commit_replaces_defaults": (
        _effective_workload_commit_replaces_defaults
    ),
    "removed_default_method_committed": _removed_default_method_committed,
    "example_endpoint_scope_preserved": _example_endpoint_scope_preserved,
    "example_endpoint_replaced_runtime_endpoint": (
        _example_endpoint_replaced_runtime_endpoint
    ),
    "rpc_schema_evidence_extracted": _rpc_schema_evidence_extracted,
    "schema_intake_looped": _schema_intake_looped,
    "multiline_evidence_block_collected_once": (
        _multiline_evidence_block_collected_once
    ),
    "blank_prompt_counted_as_evidence": _blank_prompt_counted_as_evidence,
    "ordinary_question_counted_as_evidence": (
        _ordinary_question_counted_as_evidence
    ),
    "evidence_analysis_returned": _evidence_analysis_returned,
    "execution_stage_explained": _execution_stage_explained,
    "approved_execution_submitted_once": (
        _approved_execution_submitted_once
    ),
    "duplicate_job_submission": _duplicate_job_submission,
    "status_uses_job_evidence": _status_uses_job_evidence,
    "invented_job_status": _invented_job_status,
    "compatible_environment_retained": _compatible_environment_retained,
    "mode_specific_state_invalidated": _mode_specific_state_invalidated,
    "rpc_groups_rerequired": _rpc_groups_rerequired,
    "sync_observe_ran_rpc_load": _sync_observe_ran_rpc_load,
    "semantic_units_partitioned_in_order": (
        _semantic_units_partitioned_in_order
    ),
    "admitted_mutations_only": _admitted_mutations_only,
    "unresolved_units_preserved": _unresolved_units_preserved,
    "ambiguous_change_silently_committed": (
        _ambiguous_change_silently_committed
    ),
    "semantic_unit_dropped": _semantic_unit_dropped,
}


UNSUPPORTED_WITHOUT_IMMUTABLE_VARIANT_CONTRACT = frozenset()
