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

from agent.harness.action_registry import MODE_COMPARISON_TOPIC
from agent.harness.control_receipts import (
    validate_persisted_domain_control_receipt,
)
from agent.harness.contracts import ResponseFragment
from agent.harness.domains.rpc_receipts import evidence_hash
from agent.harness.questions import render_question
from agent.harness.response_catalog import (
    MESSAGE_CATALOG,
    render_fragment,
    render_hash,
    semantic_hash,
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


def _visible_terminal_response_hashes(context: Any) -> dict[int, str]:
    """Project complete PTY responses to their terminal payload hashes."""

    hashes: dict[int, str] = {}
    prefix = "Agent> "
    boundary = "\nAgent> "
    for turn in _turns(context):
        response = str(getattr(turn, "agent_response", "") or "")
        if boundary in response:
            terminal = response.split(boundary, 1)[1]
        elif response.startswith(prefix):
            terminal = response[len(prefix):]
        else:
            continue
        if terminal:
            hashes[int(getattr(turn, "turn_index", -1))] = render_hash(
                terminal
            )
    return hashes


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


def _domain_consumers_by_turn(
    context: Any,
    *,
    owner: str,
    events: Sequence[Any] | None = None,
) -> tuple[dict[int, set[str]], list[Mapping[str, Any]]]:
    if events is None:
        valid, invalid = _domain_commits(context)
    else:
        valid = []
        invalid = []
        for event in events:
            for receipt in tuple(
                getattr(event, "control_receipts", ()) or ()
            ):
                if (
                    not isinstance(receipt, Mapping)
                    or receipt.get("receipt_type") != "domain_commit"
                ):
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
    consumers: dict[int, set[str]] = defaultdict(set)
    for item in valid:
        receipt = item["receipt"]
        if (
            receipt.get("owner") != owner
            or receipt.get("completion") == "rejected"
        ):
            continue
        consumers[int(item["turn_index"])].update(
            str(action_id)
            for action_id in receipt.get("consumed_action_ids") or ()
            if str(action_id)
        )
    return consumers, invalid


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


def _visible_orientation_authority(
    context: Any,
    *,
    topics: frozenset[str],
    required_message_ids: frozenset[str] = frozenset(),
) -> tuple[dict[int, set[str]], list[dict[str, str]]]:
    """Bind an orientation action through commit, composition, and rendering."""

    orientations, invalid_orientations = _valid_receipts(
        context,
        "orientation_response",
    )
    commits, invalid_commits = _domain_commits(context)
    compositions, invalid_compositions = _valid_receipts(
        context,
        "response_composition",
    )
    orientations_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    commits_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    compositions_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    events_by_turn = {
        int(getattr(event, "turn_index", -1)): event
        for event in _events(context)
    }
    visible_terminal_hashes = _visible_terminal_response_hashes(context)
    for item in orientations:
        orientations_by_turn[int(item["turn_index"])].append(item)
    for item in commits:
        commits_by_turn[int(item["turn_index"])].append(item)
    for item in compositions:
        compositions_by_turn[int(item["turn_index"])].append(item)
    matches: dict[int, set[str]] = defaultdict(set)
    for turn_index, event in events_by_turn.items():
        admitted = {
            str(action.get("action_id") or ""): action
            for action in (
                getattr(event, "admitted_action_provenance", ()) or ()
            )
            if isinstance(action, Mapping)
            and str(action.get("action_id") or "")
        }
        manifest = dict(getattr(event, "render_manifest", {}) or {})
        visible_hashes = list(manifest.get("fragment_hashes") or ())
        if (
            manifest.get("fragment_count") != 1
            or len(visible_hashes) != 1
            or visible_terminal_hashes.get(turn_index)
            != visible_hashes[0]
        ):
            continue
        for orientation in orientations_by_turn.get(turn_index, ()):
            receipt = orientation["receipt"]
            action_id = str(receipt.get("action_id") or "")
            action = admitted.get(action_id) or {}
            if (
                receipt.get("read_only") is not True
                or receipt.get("topic") not in topics
                or action.get("type") != receipt.get("action_type")
            ):
                continue
            for commit in commits_by_turn.get(turn_index, ()):
                commit_receipt = commit["receipt"]
                fragments = [
                    dict(fragment)
                    for fragment in (
                        commit_receipt.get("response_fragments") or ()
                    )
                    if isinstance(fragment, Mapping)
                ]
                message_ids = {
                    str(fragment.get("message_id") or "")
                    for fragment in fragments
                }
                job_fragment_matches = True
                if (
                    "harness.orientation.consultation.job_verified"
                    in message_ids
                ):
                    projection = dict(
                        receipt.get("state_projection") or {}
                    )
                    status_fragments = [
                        fragment
                        for fragment in fragments
                        if fragment.get("message_id")
                        == "harness.orientation.consultation.job_verified"
                    ]
                    language = str(manifest.get("language") or "")
                    if language not in {"en", "zh"}:
                        job_fragment_matches = False
                    else:
                        expected = render_fragment(
                            ResponseFragment(
                                kind="message",
                                message_id=(
                                    "harness.orientation.consultation."
                                    "job_verified"
                                ),
                                arguments={
                                    "job_id": str(
                                        projection.get("job_id") or ""
                                    ),
                                    "status": str(
                                        projection.get("job_status") or ""
                                    ),
                                },
                                source=__name__,
                            ),
                            language,
                        )
                        job_fragment_matches = (
                            len(status_fragments) == 1
                            and status_fragments[0].get("semantic_hash")
                            == expected.semantic_hash
                            and status_fragments[0].get("render_hash")
                            == expected.render_hash
                        )
                if (
                    commit_receipt.get("owner") != "orientation"
                    or action_id
                    not in set(
                        commit_receipt.get("consumed_action_ids") or ()
                    )
                    or not fragments
                    or any(
                        str(fragment.get("message_id") or "")
                        not in MESSAGE_CATALOG
                        for fragment in fragments
                    )
                    or not job_fragment_matches
                    or not required_message_ids.issubset(message_ids)
                    or receipt.get("response_hash")
                    != semantic_hash([
                        str(fragment.get("semantic_hash") or "")
                        for fragment in fragments
                    ])
                ):
                    continue
                for composition in compositions_by_turn.get(turn_index, ()):
                    composed_receipt = composition["receipt"]
                    composed = [
                        dict(fragment)
                        for fragment in (
                            composed_receipt.get("fragments") or ()
                        )
                        if isinstance(fragment, Mapping)
                    ]
                    if (
                        action_id
                        in set(
                            composed_receipt.get("source_action_ids")
                            or ()
                        )
                        and all(
                            fragment in composed
                            for fragment in fragments
                        )
                        and composed_receipt.get(
                            "pending_contract_hash"
                        )
                        == manifest.get("pending_contract_hash")
                        and visible_hashes[0]
                        == composed_receipt.get(
                            "terminal_response_hash"
                        )
                    ):
                        matches[turn_index].add(action_id)
                        break
    return matches, [
        *invalid_orientations,
        *invalid_commits,
        *invalid_compositions,
    ]


def _retained_state_described(context: Any) -> PredicateResult:
    contract, contract_error = _verifier_input(context)
    required_turns = {
        int(step.get("turn_index") or -1)
        for step in (
            (contract or {}).get("source_contract", {}).get(
                "source_steps"
            )
            or ()
        )
        if isinstance(step, Mapping)
        and step.get("semantic_role") == "retained_state_consultation"
    } if contract is not None else set()
    orientations, invalid_orientations = _valid_receipts(
        context,
        "orientation_response",
    )
    commits, invalid_commits = _domain_commits(context)
    compositions, invalid_compositions = _valid_receipts(
        context,
        "response_composition",
    )
    orientation_by_turn = defaultdict(list)
    commit_by_turn = defaultdict(list)
    composition_by_turn = defaultdict(list)
    for item in orientations:
        orientation_by_turn[int(item["turn_index"])].append(item)
    for item in commits:
        commit_by_turn[int(item["turn_index"])].append(item)
    for item in compositions:
        composition_by_turn[int(item["turn_index"])].append(item)
    matched: list[int] = []
    for source_turn_index, event in enumerate(_events(context), start=1):
        runtime_turn_index = int(getattr(event, "turn_index", -1))
        admitted = {
            str(action.get("action_id") or ""): action
            for action in (
                getattr(event, "admitted_action_provenance", ()) or ()
            )
            if isinstance(action, Mapping)
            and str(action.get("action_id") or "")
        }
        manifest = dict(getattr(event, "render_manifest", {}) or {})
        visible_hashes = list(manifest.get("fragment_hashes") or ())
        if (
            manifest.get("fragment_count") != 1
            or len(visible_hashes) != 1
        ):
            continue
        for orientation in orientation_by_turn.get(
            runtime_turn_index,
            (),
        ):
            receipt = orientation["receipt"]
            action_id = str(receipt.get("action_id") or "")
            action = admitted.get(action_id) or {}
            if (
                receipt.get("read_only") is not True
                or receipt.get("topic")
                not in {"current_config", "current_context"}
                or action.get("type")
                != receipt.get("action_type")
            ):
                continue
            for commit in commit_by_turn.get(runtime_turn_index, ()):
                commit_receipt = commit["receipt"]
                fragments = [
                    dict(fragment)
                    for fragment in (
                        commit_receipt.get("response_fragments") or ()
                    )
                    if isinstance(fragment, Mapping)
                ]
                if (
                    commit_receipt.get("owner") != "orientation"
                    or action_id
                    not in set(
                        commit_receipt.get("consumed_action_ids") or ()
                    )
                    or not fragments
                    or receipt.get("response_hash")
                    != semantic_hash([
                        str(fragment.get("semantic_hash") or "")
                        for fragment in fragments
                    ])
                ):
                    continue
                for composition in composition_by_turn.get(
                    runtime_turn_index,
                    (),
                ):
                    composition_receipt = composition["receipt"]
                    composed = [
                        dict(fragment)
                        for fragment in (
                            composition_receipt.get("fragments") or ()
                        )
                        if isinstance(fragment, Mapping)
                    ]
                    if (
                        action_id
                        in set(
                            composition_receipt.get(
                                "source_action_ids"
                            )
                            or ()
                        )
                        and all(
                            fragment in composed
                            for fragment in fragments
                        )
                        and visible_hashes[0]
                        == composition_receipt.get(
                            "terminal_response_hash"
                        )
                    ):
                        matched.append(source_turn_index)
                        break
    invalid = [
        *invalid_orientations,
        *invalid_commits,
        *invalid_compositions,
    ]
    return (
        bool(matched)
        and not invalid
        and (
            not required_turns
            or required_turns.issubset(set(matched))
        )
        and not contract_error
    ), {
        "evidence_family": "orientation_visible_read_only",
        "matched_source_turn_indexes": sorted(set(matched)),
        "required_source_turn_indexes": sorted(required_turns),
        "invalid_receipts": invalid,
        "verifier_contract_error": contract_error,
    }


def _admitted_action_matches_pending_option(
    action: Mapping[str, Any],
    option: Mapping[str, Any],
) -> bool:
    declared_action = option.get("action")
    action_type = str(action.get("type") or "")
    if isinstance(declared_action, Mapping):
        if action_type != str(declared_action.get("type") or ""):
            return False
        value_hashes = dict(action.get("argument_value_hashes") or {})
        return all(
            value_hashes.get(str(key)) == content_hash(value)
            for key, value in declared_action.items()
            if str(key) != "type"
        )
    return bool(
        action_type == "answer_pending"
        and (
            option.get("semantic_action")
            or option.get("expected_patch")
        )
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
    displayed_contract_turns: dict[str, int] = {}
    resolution_matches: list[int] = []
    resolution_semantic_hashes: list[str] = []
    invalid_resolutions: list[dict[str, str]] = []
    mismatched_resolutions: list[dict[str, Any]] = []
    expected_options: dict[str, Mapping[str, Any]] = {}
    duplicate_expected_option_ids: set[str] = set()
    for option in tuple(expected_contract.get("options") or ()):
        if not isinstance(option, Mapping):
            continue
        option_id = str(option.get("id") or option.get("option_id") or "")
        if (
            not option_id
            or not (
                option.get("action")
                or option.get("expected_patch")
                or option.get("semantic_action")
            )
        ):
            continue
        if option_id in expected_options:
            duplicate_expected_option_ids.add(option_id)
        expected_options[option_id] = option
    initial_event = getattr(context, "initial_event", None)
    events = (
        *tuple(getattr(context, "setup_events", ()) or ()),
        *((initial_event,) if initial_event is not None else ()),
        *_events(context),
    )
    orientation_consumers, invalid_domain_commits = (
        _domain_consumers_by_turn(
            context,
            owner="orientation",
            events=events,
        )
    )
    seen_event_ids: set[str] = set()
    for event in events:
        event_id = str(getattr(event, "runtime_event_id", "") or "")
        if event_id and event_id in seen_event_ids:
            continue
        if event_id:
            seen_event_ids.add(event_id)
        contract = dict(getattr(event, "pending_contract", {}) or {})
        render_manifest = dict(
            getattr(event, "render_manifest", {}) or {}
        )
        semantic_contract_hash = (
            content_hash(canonical_question_contract(contract))
            if contract
            else ""
        )
        runtime_contract_hash = content_hash(contract) if contract else ""
        options = tuple(contract.get("options") or ())
        option_ids = [
            str(option.get("id") or option.get("option_id") or "")
            for option in options
            if isinstance(option, Mapping)
        ]
        if (
            contract.get("id") == "resume_harness_session"
            and expected_hash
            and semantic_contract_hash == expected_hash
            and len(option_ids) >= 2
            and all(option_ids)
            and len(option_ids) == len(set(option_ids))
            and render_manifest.get("pending_contract_hash")
            == runtime_contract_hash
            and dict(render_manifest.get("result") or {}).get(
                "question_id"
            )
            == "resume_harness_session"
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
            turn_index = int(getattr(event, "turn_index", -1))
            matched.append(turn_index)
            matched_hashes.append(semantic_contract_hash)
            displayed_contract_turns[runtime_contract_hash] = turn_index
        for receipt in tuple(
            getattr(event, "control_receipts", ()) or ()
        ):
            if (
                not isinstance(receipt, Mapping)
                or receipt.get("receipt_type") != "pending_resolution"
                or receipt.get("pending_id") != "resume_harness_session"
            ):
                continue
            accepted, reason = validate_persisted_domain_control_receipt(
                receipt,
                turn_index=int(getattr(event, "turn_index", -1)),
            )
            if not accepted:
                invalid_resolutions.append({
                    "receipt_id": str(receipt.get("receipt_id") or ""),
                    "reason": reason,
                })
                continue
            selected_option_id = str(
                receipt.get("selected_option_id") or ""
            )
            selected_option = expected_options.get(selected_option_id)
            transition = _valid_pending_transition(event)
            admitted_actions = {
                str(action.get("action_id") or ""): action
                for action in (
                    getattr(event, "admitted_action_provenance", ()) or ()
                )
                if isinstance(action, Mapping)
            }
            resolved_action_id = str(
                receipt.get("resolved_action_id") or ""
            )
            resolved_action = admitted_actions.get(resolved_action_id)
            turn_index = int(getattr(event, "turn_index", -1))
            resolved_contract_hash = str(
                receipt.get("pending_contract_hash") or ""
            )
            if (
                receipt.get("verdict") == "accepted"
                and receipt.get("normalizer") == "exact_contract"
                and receipt.get("resolution_path") == "exact_contract"
                and resolved_contract_hash in displayed_contract_turns
                and displayed_contract_turns[resolved_contract_hash]
                < turn_index
                and selected_option is not None
                and selected_option_id not in duplicate_expected_option_ids
                and receipt.get("selected_value_hash")
                == content_hash(selected_option.get("value"))
                and resolved_action is not None
                and _admitted_action_matches_pending_option(
                    resolved_action,
                    selected_option,
                )
                and resolved_action_id
                in orientation_consumers.get(turn_index, set())
                and transition is not None
                and transition.get("before_id")
                == "resume_harness_session"
                and transition.get("before_hash")
                == resolved_contract_hash
                and resolved_action_id
                in set(transition.get("consumer_action_ids") or ())
            ):
                resolution_matches.append(turn_index)
                resolution_semantic_hashes.append(
                    content_hash(selected_option)
                )
            elif receipt.get("verdict") == "accepted":
                mismatched_resolutions.append({
                    "turn_index": int(
                        getattr(event, "turn_index", -1)
                    ),
                    "pending_contract_hash": str(
                        receipt.get("pending_contract_hash") or ""
                    ),
                    "selected_option_id": selected_option_id,
                    "resolution_path": str(
                        receipt.get("resolution_path") or ""
                    ),
                    "normalizer": str(receipt.get("normalizer") or ""),
                })
    resolved_after_display = [
        turn_index
        for turn_index in resolution_matches
        if any(display_turn < turn_index for display_turn in matched)
    ]
    return (
        bool(matched)
        and (
            not resolution_matches
            or len(resolved_after_display) == len(resolution_matches)
        )
        and not invalid_resolutions
        and not mismatched_resolutions
        and not invalid_domain_commits
    ), {
        "evidence_family": "resume_contract",
        "scenario_id": scenario_id,
        "expected_pending_contract_hash": expected_hash,
        "matched_pending_contract_hashes": matched_hashes,
        "matched_turn_indexes": matched,
        "matched_resolution_turn_indexes": resolved_after_display,
        "matched_resolution_option_semantic_hashes": (
            resolution_semantic_hashes
        ),
        "invalid_resolution_receipts": invalid_resolutions,
        "mismatched_resolution_receipts": mismatched_resolutions,
        "invalid_domain_commit_receipts": invalid_domain_commits,
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


def _real_node_selection_executed_at_source(
    context: Any,
) -> PredicateResult:
    event = _source_step_event(context, 1)
    transition = _valid_pending_transition(event) if event else None
    valid, invalid = _valid_receipts(context, "pending_resolution")
    turn_index = int(getattr(event, "turn_index", -1)) if event else -1
    execution_order = set(
        dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
            "execution_order"
        )
        or ()
    ) if event else set()
    admitted_ids = set(
        dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
            "admitted_action_ids"
        )
        or ()
    ) if event else set()
    transition_consumers = set(
        (transition or {}).get("consumer_action_ids") or ()
    )
    admitted = _admitted_actions(event, "choose_target_mode")
    pending_actions = _admitted_actions(event, "answer_pending")
    matching_actions = [
        action
        for action in admitted
        if action.get("owner") == "chain_rpc"
        and action.get("effect") == "configuration_mutation"
        and action.get("group") == "target_mode"
        and dict(action.get("argument_value_hashes") or {}).get("target_mode")
        == _value_hash("real-node")
        and str(action.get("action_id") or "")
        in admitted_ids & execution_order & transition_consumers
    ]
    matching_action_ids = {
        str(action.get("action_id") or "")
        for action in matching_actions
    }
    matching_pending_action_ids = {
        str(action.get("action_id") or "")
        for action in pending_actions
        if action.get("owner") == "coordinator"
        and action.get("effect") == "configuration_mutation"
        and action.get("group") == ""
        and str(action.get("action_id") or "")
        in admitted_ids & execution_order & transition_consumers
    }
    matching_receipts = [
        item
        for item in valid
        if int(item["turn_index"]) == turn_index
        and item["receipt"].get("resolution_path") == "exact_contract"
        and item["receipt"].get("pending_id") == "opening_next_action"
        and item["receipt"].get("pending_group") == "opening"
        and item["receipt"].get("pending_contract_hash")
        == (transition or {}).get("before_hash")
        and item["receipt"].get("selected_value_hash")
        == _value_hash("real-node")
        and str(item["receipt"].get("resolved_action_id") or "")
        in matching_pending_action_ids
    ]
    execution_sequence = tuple(
        dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
            "execution_order"
        )
        or ()
    ) if event else ()
    causally_ordered = bool(
        len(matching_pending_action_ids) == 1
        and len(matching_action_ids) == 1
        and execution_sequence.index(next(iter(matching_pending_action_ids)))
        < execution_sequence.index(next(iter(matching_action_ids)))
    )
    material_diffs = _valid_material_diffs(event) if event else {}
    material_paths = sorted(material_diffs)
    satisfied = bool(
        transition
        and transition["transition"] == "replaced"
        and transition["before_id"] == "opening_next_action"
        and transition["before_group"] == "opening"
        and transition["after_id"] == "chain"
        and transition["after_group"] == "chain_identity"
        and len(matching_actions) == 1
        and causally_ordered
        and len(matching_receipts) == 1
        and (material_diffs.get("target_mode") or {}).get("after")
        == _value_hash("real-node")
        and not invalid
    )
    return satisfied, _receipt_details(
        "turn_bound_real_node_selection",
        matching_receipts,
        invalid,
        source_step_position=1,
        turn_index=turn_index,
        pending_transition=dict(transition or {}),
        matching_admitted_action_ids=sorted(matching_action_ids),
        matching_pending_action_ids=sorted(matching_pending_action_ids),
        causally_ordered=causally_ordered,
        material_state_paths=material_paths,
    )


def _mode_consultation_preserves_chain_pending(
    context: Any,
) -> PredicateResult:
    event = _source_step_event(context, 2)
    transition = _valid_pending_transition(event) if event else None
    valid, invalid = _valid_receipts(context, "orientation_response")
    turn_index = int(getattr(event, "turn_index", -1)) if event else -1
    execution_order = set(
        dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
            "execution_order"
        )
        or ()
    ) if event else set()
    admitted_ids = set(
        dict(getattr(event, "turn_receipt_summary", {}) or {}).get(
            "admitted_action_ids"
        )
        or ()
    ) if event else set()
    transition_consumers = set(
        (transition or {}).get("consumer_action_ids") or ()
    )
    admitted = _admitted_actions(event, "answer_opening_question")
    matching_actions = [
        action
        for action in admitted
        if action.get("owner") == "orientation"
        and action.get("effect") == "read_only"
        and dict(action.get("argument_value_hashes") or {}).get("topic")
        == _value_hash(MODE_COMPARISON_TOPIC)
        and str(action.get("action_id") or "")
        in admitted_ids & execution_order & transition_consumers
    ]
    matching_action_ids = {
        str(action.get("action_id") or "")
        for action in matching_actions
    }
    matching_receipts = [
        item
        for item in valid
        if int(item["turn_index"]) == turn_index
        and item["receipt"].get("read_only") is True
        and item["receipt"].get("action_type") == "answer_opening_question"
        and item["receipt"].get("topic") == MODE_COMPARISON_TOPIC
        and item["receipt"].get("pending_contract_hash")
        == (transition or {}).get("before_hash")
        and str(item["receipt"].get("action_id") or "")
        in matching_action_ids
    ]
    material_paths = sorted(_valid_material_diffs(event)) if event else []
    satisfied = bool(
        transition
        and transition["transition"] == "preserved"
        and transition["before_id"] == "chain"
        and transition["before_group"] == "chain_identity"
        and transition["after_id"] == transition["before_id"]
        and transition["after_group"] == transition["before_group"]
        and transition["after_hash"] == transition["before_hash"]
        and len(matching_actions) == 1
        and len(matching_receipts) == 1
        and not material_paths
        and not invalid
    )
    return satisfied, _receipt_details(
        "turn_bound_mode_consultation",
        matching_receipts,
        invalid,
        source_step_position=2,
        turn_index=turn_index,
        pending_transition=dict(transition or {}),
        matching_admitted_action_ids=sorted(matching_action_ids),
        material_state_paths=material_paths,
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


def _bound_unknown_chain_resolutions(
    context: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    identities, invalid = _valid_receipts(
        context, "chain_identity_resolution"
    )
    chain_consumers, invalid_commits = _domain_consumers_by_turn(
        context,
        owner="chain_rpc",
    )
    events_by_turn = {
        int(getattr(event, "turn_index", -1)): event
        for event in _events(context)
    }
    bound: list[dict[str, Any]] = []
    for item in identities:
        event = events_by_turn.get(int(item["turn_index"]))
        transition = (
            _valid_pending_transition(event)
            if event is not None
            else None
        )
        if (
            transition is None
            or transition.get("after_group") != "chain_identity"
        ):
            continue
        consumer_ids = set(transition.get("consumer_action_ids") or ())
        committed_ids = chain_consumers.get(int(item["turn_index"]), set())
        candidate_hash = str(
            item["receipt"].get("candidate_hash") or ""
        )
        actions = [
            action
            for action in (
                getattr(event, "admitted_action_provenance", ()) or ()
            )
            if isinstance(action, Mapping)
            and str(action.get("action_id") or "") in consumer_ids
            and str(action.get("action_id") or "") in committed_ids
            and str(action.get("type") or "")
            in {"choose_chain", "change_chain"}
            and dict(action.get("argument_value_hashes") or {}).get(
                "chain_text"
            )
            == candidate_hash
        ]
        if actions:
            bound.append(item)
    return bound, [*invalid, *invalid_commits]


def _unknown_chain_identity_resolution_started(context: Any) -> PredicateResult:
    bound, invalid = _bound_unknown_chain_resolutions(context)
    matches = [
        item
        for item in bound
        if bool(item["receipt"].get("resolver_source"))
    ]
    return bool(matches) and not invalid, _receipt_details(
        "unknown_chain_identity",
        matches,
        invalid,
    )


def _chain_confirmation_required(context: Any) -> PredicateResult:
    bound, invalid = _bound_unknown_chain_resolutions(context)
    matches = [
        item
        for item in bound
        if item["receipt"].get("confirmation_required") is True
    ]
    return bool(matches) and not invalid, _receipt_details(
        "unknown_chain_identity",
        matches,
        invalid,
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
    identities_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in identity:
        identities_by_turn[int(item["turn_index"])].append(item)
    route_action_types_requiring_identity = {
        "set_chain_candidate",
        "change_chain",
        "choose_chain",
        "select_chain",
    }
    chain_consumers, invalid_commits = _domain_consumers_by_turn(
        context,
        owner="chain_rpc",
    )
    routed_actions: list[dict[str, Any]] = []
    typed_intakes: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for event in _events(context):
        turn_index = int(getattr(event, "turn_index", -1))
        transition = _valid_pending_transition(event)
        typed_intake = (
            transition is not None
            and transition.get("transition") in {"created", "replaced"}
            and transition.get("after_group") == "chain_identity"
            and transition.get("after_id") in {"chain", "chain_change_input"}
        )
        if typed_intake:
            typed_intakes.append({
                "turn_index": turn_index,
                "question_id": str(transition.get("after_id") or ""),
                "consumer_action_ids": list(
                    transition.get("consumer_action_ids") or ()
                ),
            })
        provenance = [
            action
            for action in tuple(
                getattr(event, "admitted_action_provenance", ()) or ()
            )
            if isinstance(action, Mapping)
            and str(action.get("action_id") or "")
        ]
        for action in provenance:
            action_type = str(action.get("type") or "")
            is_chain_route = (
                action_type in {
                    "set_chain_candidate",
                    "change_chain",
                    "choose_chain",
                    "request_chain_selection",
                    "select_chain",
                }
                or (
                    action_type == "change_group"
                    and action.get("group") == "chain_identity"
                )
            )
            if not is_chain_route:
                continue
            action_id = str(action["action_id"])
            record = {
                "turn_index": turn_index,
                "action_id": action_id,
                "action_type": action_type,
            }
            routed_actions.append(record)
            same_turn_identity = identities_by_turn.get(turn_index, [])
            identity_required = (
                action_type in route_action_types_requiring_identity
            )
            action_candidate_hash = str(
                dict(action.get("argument_value_hashes") or {}).get(
                    "chain_text"
                )
                or ""
            )
            bound_identity = [
                item
                for item in same_turn_identity
                if str(item["receipt"].get("candidate_hash") or "")
                == action_candidate_hash
            ]
            if (
                typed_intake
                and action_id
                in set(transition.get("consumer_action_ids") or ())
                and action_id
                in chain_consumers.get(turn_index, set())
                and (
                    not identity_required
                    or (
                        _is_hash(action_candidate_hash)
                        and bool(bound_identity)
                    )
                )
            ):
                matches.append({
                    **record,
                    "question_id": str(transition.get("after_id") or ""),
                    "identity_receipt_ids": [
                        str(item.get("receipt_id") or "")
                        for item in bound_identity
                    ],
                })
    invalid = [*invalid_identity, *invalid_commits]
    return bool(matches) and not invalid, _receipt_details(
        "chain_request_routing",
        identity,
        invalid,
        routed_action_count=len(routed_actions),
        typed_intake_count=len(typed_intakes),
        typed_intakes=typed_intakes,
        matches=matches,
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


def _rpc_schema_progressions(
    context: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[Mapping[str, Any]],
]:
    schemas, invalid_schemas = _valid_receipts(
        context, "rpc_schema_provenance"
    )
    catalogs, invalid_catalogs = _valid_receipts(
        context, "rpc_catalog_transition"
    )
    catalogs_by_turn: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in catalogs:
        catalogs_by_turn[int(item["turn_index"])].append(item)
    events_by_turn = {
        int(getattr(event, "turn_index", -1)): event
        for event in _events(context)
    }
    consumers, invalid_commits = _domain_consumers_by_turn(
        context,
        owner="chain_rpc",
    )
    lineage_by_method: dict[str, dict[str, Any]] = {}
    progression: list[dict[str, Any]] = []
    complete: list[dict[str, Any]] = []
    endpoint_scoped: list[dict[str, Any]] = []

    for item in sorted(
        schemas,
        key=lambda record: (
            int(record["turn_index"]),
            int(record["receipt"].get("catalog_revision") or -1),
        ),
    ):
        receipt = item["receipt"]
        turn_index = int(item["turn_index"])
        method_hash = str(receipt.get("method_hash") or "")
        event = events_by_turn.get(turn_index)
        transition = (
            _valid_pending_transition(event)
            if event is not None
            else None
        )
        before_id = str((transition or {}).get("before_id") or "")
        after_id = str((transition or {}).get("after_id") or "")
        starts_lineage = before_id.endswith("_schema_evidence")
        continues_lineage = before_id.endswith("_schema_confirm")
        prior = lineage_by_method.get(method_hash)
        if (
            not _is_hash(method_hash)
            or transition is None
            or not after_id.endswith("_schema_confirm")
            or not (starts_lineage or (continues_lineage and prior))
        ):
            continue

        consumer_ids = set(transition.get("consumer_action_ids") or ())
        admitted = [
            action
            for action in (
                getattr(event, "admitted_action_provenance", ()) or ()
            )
            if isinstance(action, Mapping)
            and str(action.get("action_id") or "") in consumer_ids
            and str(action.get("action_id") or "")
            in consumers.get(turn_index, set())
            and str(action.get("type") or "") == "rpc_catalog_command"
            and dict(action.get("argument_value_hashes") or {}).get(
                "catalog_command"
            )
            == content_hash("append_evidence")
            and str(action.get("action_id") or "")
            == str(receipt.get("producer_action_id") or "")
        ]
        if not admitted:
            continue
        producer_ids = {
            str(action.get("action_id") or "") for action in admitted
        }
        current_action_hashes = {
            str(value_hash)
            for action in admitted
            for argument, value_hash in dict(
                action.get("argument_value_hashes") or {}
            ).items()
            if argument
            in {
                "source_evidence",
                "rpc_schema_evidence",
                "selected_value",
                "answer",
            }
            and _is_hash(str(value_hash or ""))
        }
        accepted_appends = [
            catalog
            for catalog in catalogs_by_turn.get(turn_index, [])
            if catalog["receipt"].get("accepted") is True
            and catalog["receipt"].get("command") == "append_evidence"
            and str(catalog["receipt"].get("producer_action_id") or "")
            in producer_ids
            and (
                not catalog["receipt"].get("draft_method_hash")
                or catalog["receipt"].get("draft_method_hash") == method_hash
            )
            and catalog["receipt"].get("source_evidence_hashes")
            == receipt.get("source_evidence_hashes")
            and catalog["receipt"].get("source_action_value_hashes")
            == receipt.get("source_action_value_hashes")
        ]
        accepted_schema_transitions = [
            catalog
            for catalog in catalogs_by_turn.get(turn_index, [])
            if catalog["receipt"].get("accepted") is True
            and catalog["receipt"].get("command") == "correct_draft"
            and str(catalog["receipt"].get("producer_action_id") or "")
            in producer_ids
            and int(catalog["receipt"].get("catalog_revision") or -1)
            == int(receipt.get("catalog_revision") or -2)
            and (
                not catalog["receipt"].get("draft_method_hash")
                or catalog["receipt"].get("draft_method_hash") == method_hash
            )
        ]
        if not accepted_appends or not accepted_schema_transitions:
            continue

        append_revisions = {
            int(catalog["receipt"].get("catalog_revision") or -1)
            for catalog in accepted_appends
        }
        source_action_hashes = set(
            receipt.get("source_action_value_hashes") or ()
        )
        source_evidence_hashes = set(
            receipt.get("source_evidence_hashes") or ()
        )
        prior_action_hashes = set(
            (prior or {}).get("source_action_hashes") or ()
        )
        prior_evidence_hashes = set(
            (prior or {}).get("source_evidence_hashes") or ()
        )
        prior_append_revisions = set(
            (prior or {}).get("append_revisions") or ()
        )
        lineage_append_revisions = prior_append_revisions | append_revisions
        if (
            not current_action_hashes
            or not current_action_hashes <= source_action_hashes
            or not prior_action_hashes <= source_action_hashes
            or not prior_evidence_hashes <= source_evidence_hashes
            or (
                prior
                and int(receipt.get("catalog_revision") or -1)
                <= int(prior.get("catalog_revision") or -1)
            )
            or (
                prior_append_revisions
                and min(append_revisions) <= max(prior_append_revisions)
            )
        ):
            continue

        fields = {
            str(field.get("field_path") or ""): field
            for field in receipt.get("fields") or ()
            if isinstance(field, Mapping)
        }
        source_kinds = {
            str(field.get("source_kind") or "")
            for field in fields.values()
        }
        status = fields.get("exchange_correlation.status") or {}
        request_ids = fields.get(
            "exchange_correlation.request_id_hashes"
        ) or {}
        response_ids = fields.get(
            "exchange_correlation.response_id_hashes"
        ) or {}
        provenance_fields = [
            field
            for field in (
                status,
                request_ids,
                response_ids,
                *[
                    field
                    for field in fields.values()
                    if field.get("source_kind")
                    in {
                        "protocol_request_parser",
                        "protocol_response_parser",
                    }
                ],
            )
            if isinstance(field, Mapping)
        ]
        lineage_revisions = {
            tuple(field.get("source_revisions") or ())
            for field in provenance_fields
        }
        common_revisions = next(iter(lineage_revisions), ())
        if (
            len(lineage_revisions) != 1
            or not common_revisions
            or not lineage_append_revisions <= set(common_revisions)
        ):
            continue

        record = {
            **item,
            "append_revisions": tuple(sorted(lineage_append_revisions)),
            "source_action_hashes": tuple(sorted(source_action_hashes)),
            "source_evidence_hashes": tuple(sorted(source_evidence_hashes)),
        }
        lineage_by_method[method_hash] = {
            "catalog_revision": int(
                receipt.get("catalog_revision") or -1
            ),
            "append_revisions": lineage_append_revisions,
            "source_action_hashes": source_action_hashes,
            "source_evidence_hashes": source_evidence_hashes,
        }
        progression.append(record)

        validation_endpoint = fields.get("validation_endpoint") or {}
        if (
            validation_endpoint.get("source_kind") == "rpc_endpoint_role"
            and tuple(validation_endpoint.get("source_revisions") or ())
            == tuple(common_revisions)
            and _is_hash(
                str(validation_endpoint.get("value_hash") or "")
            )
        ):
            endpoint_scoped.append(record)

        if (
            "protocol_request_parser" in source_kinds
            and "protocol_response_parser" in source_kinds
            and status.get("source_kind") == "protocol_exchange_correlator"
            and status.get("value_hash") == evidence_hash("correlated")
            and request_ids.get("source_kind")
            == "protocol_exchange_correlator"
            and response_ids.get("source_kind")
            == "protocol_exchange_correlator"
            and request_ids.get("value_hash")
            == response_ids.get("value_hash")
            and _is_hash(str(request_ids.get("value_hash") or ""))
        ):
            complete.append(record)

    return (
        progression,
        complete,
        endpoint_scoped,
        [*invalid_schemas, *invalid_catalogs, *invalid_commits],
    )


def _example_endpoint_scope_preserved(context: Any) -> PredicateResult:
    endpoints, invalid_endpoints = _valid_receipts(
        context,
        "rpc_endpoint_role",
    )
    consumers, invalid_commits = _domain_consumers_by_turn(
        context,
        owner="chain_rpc",
    )
    events = {
        int(getattr(event, "turn_index", -1)): event
        for event in _events(context)
    }
    matched = []
    for item in endpoints:
        receipt = item["receipt"]
        turn_index = int(item["turn_index"])
        producer = str(receipt.get("producer_action_id") or "")
        event = events.get(turn_index)
        admitted_ids = {
            str(action.get("action_id") or "")
            for action in (
                getattr(event, "admitted_action_provenance", ()) or ()
            )
            if isinstance(action, Mapping)
        } if event is not None else set()
        if (
            receipt.get("role") == "validation"
            and receipt.get("ready") is True
            and producer
            and producer in admitted_ids
            and producer in consumers.get(turn_index, set())
        ):
            matched.append(item)
    (
        _schema_progression,
        _complete_schemas,
        schema_endpoint_scope,
        invalid_schema_lineage,
    ) = _rpc_schema_progressions(context)
    invalid = [
        *invalid_endpoints,
        *invalid_commits,
        *invalid_schema_lineage,
    ]
    matched.extend(schema_endpoint_scope)
    return bool(matched) and not invalid, _receipt_details(
        "rpc_endpoint_scope",
        matched,
        invalid,
    )


def _example_endpoint_replaced_runtime_endpoint(context: Any) -> PredicateResult:
    endpoints, invalid_endpoints = _valid_receipts(context, "rpc_endpoint_role")
    deltas, invalid_commits = _material_delta_records(context)
    (
        _schema_progression,
        _complete_schemas,
        schema_endpoint_scope,
        invalid_schema_lineage,
    ) = _rpc_schema_progressions(context)
    validation_turns = sorted({
        int(item["turn_index"])
        for item in endpoints
        if item["receipt"].get("role") == "validation"
    } | {
        int(item["turn_index"])
        for item in schema_endpoint_scope
    })
    if not validation_turns:
        return False, _receipt_details(
            "rpc_endpoint_scope",
            [],
            [*invalid_endpoints, *invalid_commits],
            runtime_endpoint_write_count=0,
        )
    consumers, invalid_consumer_commits = _domain_consumers_by_turn(
        context,
        owner="chain_rpc",
    )
    events = {
        int(getattr(event, "turn_index", -1)): event
        for event in _events(context)
    }
    endpoint_receipts_by_id = {
        str(item["receipt"].get("receipt_id") or ""): item
        for item in endpoints
        if str(item["receipt"].get("receipt_id") or "")
    }
    authorized_runtime_writes: set[tuple[int, str, str, str]] = set()
    for item in endpoints:
        receipt = item["receipt"]
        if receipt.get("role") not in {
            "final_benchmark",
            "sync_observe",
        }:
            continue
        turn_index = int(item["turn_index"])
        producer = str(receipt.get("producer_action_id") or "")
        event = events.get(turn_index)
        admitted_ids = {
            str(action.get("action_id") or "")
            for action in (
                getattr(event, "admitted_action_provenance", ()) or ()
            )
            if isinstance(action, Mapping)
        } if event is not None else set()
        admitted_value_hashes = {
            str(action.get("action_id") or ""): set(
                str(value_hash)
                for value_hash in dict(
                    action.get("argument_value_hashes") or {}
                ).values()
            )
            for action in (
                getattr(event, "admitted_action_provenance", ()) or ()
            )
            if isinstance(action, Mapping)
        } if event is not None else {}
        direct_source = (
            receipt.get("source_kind") == "direct_action"
            and receipt.get("source_value_hash")
            in admitted_value_hashes.get(producer, set())
        )
        source_item = endpoint_receipts_by_id.get(
            str(receipt.get("source_receipt_id") or "")
        )
        source_receipt = (
            source_item["receipt"]
            if isinstance(source_item, Mapping)
            else {}
        )
        promoted_source = (
            receipt.get("source_kind") == "validated_endpoint"
            and source_item is not None
            and int(source_item["turn_index"]) < turn_index
            and source_receipt.get("role") == "validation"
            and source_receipt.get("ready") is True
            and source_receipt.get("endpoint_hash")
            == receipt.get("endpoint_hash")
            and source_receipt.get("source_value_hash")
            == receipt.get("source_value_hash")
        )
        if (
            receipt.get("ready") is True
            and producer
            and producer in admitted_ids
            and producer in consumers.get(turn_index, set())
            and (direct_source or promoted_source)
        ):
            authorized_runtime_writes.add((
                turn_index,
                producer,
                str(receipt.get("config_field") or ""),
                str(receipt.get("source_value_hash") or ""),
            ))
    replaced = [
        item
        for item in deltas
        if int(item["turn_index"]) >= min(validation_turns)
        and item["operation"] == "write"
        and _path_leaf(item["path"])
        in {
            "LOCAL_RPC_URL",
            "MAINNET_RPC_URL",
            "SYNC_OBSERVE_RPC_URL",
        }
        and not any(
            (
                int(item["turn_index"]),
                str(producer),
                _path_leaf(item["path"]),
                str(item.get("value_hash") or ""),
            )
            in authorized_runtime_writes
            for producer in (
                item["receipt"].get("consumed_action_ids") or ()
            )
        )
    ]
    invalid = [
        *invalid_endpoints,
        *invalid_commits,
        *invalid_consumer_commits,
        *invalid_schema_lineage,
    ]
    return bool(replaced) and not invalid, _receipt_details(
        "rpc_endpoint_scope",
        replaced,
        invalid,
        runtime_endpoint_write_count=len(replaced),
    )


def _rpc_schema_evidence_extracted(context: Any) -> PredicateResult:
    (
        _progression,
        correlated,
        _endpoint_scoped,
        invalid_receipts,
    ) = _rpc_schema_progressions(context)
    return bool(correlated) and not invalid_receipts, _receipt_details(
        "rpc_schema_provenance",
        correlated,
        invalid_receipts,
        correlated_receipt_count=len(correlated),
        progressed_turn_indexes=sorted(
            {int(item["turn_index"]) for item in correlated}
        ),
    )


def _schema_intake_looped(context: Any) -> PredicateResult:
    valid_catalog, invalid_catalog = _valid_receipts(
        context, "rpc_catalog_transition"
    )
    catalog_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in valid_catalog:
        catalog_by_turn[int(item["turn_index"])].append(item)
    accepted_turns: set[int] = set()
    chain_consumers, invalid_commits = _domain_consumers_by_turn(
        context,
        owner="chain_rpc",
    )
    mutation_actions_by_turn: dict[int, dict[str, str]] = defaultdict(dict)
    for event in _events(context):
        turn_index = int(getattr(event, "turn_index", -1))
        transition = _valid_pending_transition(event)
        consumer_ids = set(
            (transition or {}).get("consumer_action_ids") or ()
        )
        for action in (
            getattr(event, "admitted_action_provenance", ()) or ()
        ):
            if (
                not isinstance(action, Mapping)
                or str(action.get("action_id") or "") not in consumer_ids
                or str(action.get("action_id") or "")
                not in chain_consumers.get(turn_index, set())
                or str(action.get("type") or "")
                != "rpc_catalog_command"
            ):
                continue
            command_hash = str(
                dict(action.get("argument_value_hashes") or {}).get(
                    "catalog_command"
                )
                or ""
            )
            action_id = str(action.get("action_id") or "")
            mutation_actions_by_turn[turn_index][action_id] = command_hash
            if (
                command_hash == content_hash("append_evidence")
                and any(
                    item["receipt"].get("accepted") is True
                    and item["receipt"].get("command")
                    == "append_evidence"
                    and item["receipt"].get("producer_action_id")
                    == action_id
                    for item in catalog_by_turn.get(turn_index, ())
                )
            ):
                accepted_turns.add(turn_index)
    loops: list[dict[str, Any]] = []
    for event in _events(context):
        transition = _valid_pending_transition(event)
        if transition is None:
            continue
        before_id = str(transition.get("before_id") or "")
        after_id = str(transition.get("after_id") or "")
        evidence_stalled = (
            before_id.endswith("_schema_evidence")
            and content_hash("append_evidence")
            in set(
                mutation_actions_by_turn.get(
                    int(getattr(event, "turn_index", -1)),
                    {},
                ).values()
            )
            and int(getattr(event, "turn_index", -1))
            not in accepted_turns
        )
        confirmation_stalled = (
            before_id.endswith("_schema_confirm")
            and bool(
                set(mutation_actions_by_turn.get(
                    int(getattr(event, "turn_index", -1)),
                    {},
                ).values())
                & {
                    content_hash("confirm_request"),
                    content_hash("confirm_response"),
                }
            )
        )
        if (
            after_id == before_id
            and bool(transition.get("consumer_action_ids"))
            and (evidence_stalled or confirmation_stalled)
        ):
            loops.append({
                "turn_index": int(getattr(event, "turn_index", -1)),
                "pending_question_id": before_id,
                "consumer_action_ids": list(
                    transition.get("consumer_action_ids") or ()
                ),
            })
    return bool(loops), {
        "evidence_family": "rpc_schema_lifecycle",
        "repeated_schema_intake_transitions": loops,
        "accepted_incremental_evidence_turns": sorted(accepted_turns),
        "invalid_catalog_receipts": invalid_catalog,
        "invalid_domain_commit_receipts": invalid_commits,
    }


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
            and receipt.get("response_rendered") is True
            and _is_hash(str(receipt.get("response_render_hash") or ""))
            and _is_hash(str(receipt.get("response_semantic_hash") or ""))
            and bool(receipt.get("response_message_ids"))
        ),
    )


def _execution_stage_explained(context: Any) -> PredicateResult:
    orientations, invalid_orientations = _valid_receipts(
        context, "orientation_response"
    )
    commits, invalid_commits = _domain_commits(context)
    compositions, invalid_compositions = _valid_receipts(
        context, "response_composition"
    )
    execution_explanation_message = (
        "harness.orientation.consultation.preflight_smoke"
    )
    compositions_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for composition in compositions:
        compositions_by_turn[int(composition["turn_index"])].append(
            composition
        )
    events_by_turn = {
        int(getattr(event, "turn_index", -1)): event
        for event in _events(context)
    }
    visible_terminal_hashes = _visible_terminal_response_hashes(context)
    matches: list[dict[str, Any]] = []
    for orientation in orientations:
        receipt = orientation["receipt"]
        if (
            receipt.get("read_only") is not True
            or receipt.get("action_type") != "answer_opening_question"
            or receipt.get("topic")
            not in {
                "requirements",
                "workflow",
                "execution_preflight_smoke",
            }
        ):
            continue
        turn_index = int(orientation["turn_index"])
        action_id = str(receipt.get("action_id") or "")
        event = events_by_turn.get(turn_index)
        admitted_orientation_actions = [
            action
            for action in (
                getattr(event, "admitted_action_provenance", ()) or ()
            )
            if isinstance(action, Mapping)
            and str(action.get("action_id") or "") == action_id
            and str(action.get("type") or "")
            == str(receipt.get("action_type") or "")
        ] if event is not None else []
        if not admitted_orientation_actions:
            continue
        for commit in commits:
            commit_receipt = commit["receipt"]
            if (
                int(commit["turn_index"]) != turn_index
                or commit_receipt.get("owner") != "orientation"
                or action_id
                not in set(commit_receipt.get("consumed_action_ids") or ())
            ):
                continue
            explanation_fragments = [
                dict(fragment)
                for fragment in commit_receipt.get("response_fragments") or ()
                if isinstance(fragment, Mapping)
                and fragment.get("message_id")
                == execution_explanation_message
            ]
            if not explanation_fragments:
                continue
            domain_fragments = [
                dict(fragment)
                for fragment in commit_receipt.get("response_fragments") or ()
                if isinstance(fragment, Mapping)
            ]
            pending_transition = (
                _valid_pending_transition(event)
                if event is not None
                else None
            )
            if (
                pending_transition is None
                or pending_transition.get("transition") != "preserved"
                or pending_transition.get("before_group")
                != "preflight_smoke_execution"
                or pending_transition.get("after_group")
                != "preflight_smoke_execution"
                or not pending_transition.get("before_id")
                or not pending_transition.get("after_id")
                or receipt.get("pending_contract_hash")
                != pending_transition.get("before_hash")
                or pending_transition.get("consumer_action_ids")
            ):
                continue
            render_manifest = (
                dict(getattr(event, "render_manifest", {}) or {})
                if event is not None
                else {}
            )
            visible_hashes = list(
                render_manifest.get("fragment_hashes") or ()
            )
            fragment_count = render_manifest.get("fragment_count")
            domain_response_hash = semantic_hash([
                str(fragment.get("semantic_hash") or "")
                for fragment in domain_fragments
            ])
            if receipt.get("response_hash") != domain_response_hash:
                continue
            for composition in compositions_by_turn.get(turn_index, []):
                composition_receipt = composition["receipt"]
                if action_id not in set(
                    composition_receipt.get("source_action_ids") or ()
                ):
                    continue
                composed_fragments = [
                    dict(fragment)
                    for fragment in composition_receipt.get("fragments") or ()
                    if isinstance(fragment, Mapping)
                ]
                recomputed = _recompute_preflight_composition(
                    composed_fragments,
                    pending_contract=dict(
                        getattr(event, "pending_contract", {}) or {}
                    ),
                    language=str(
                        composition_receipt.get("language") or ""
                    ),
                )
                for fragment in explanation_fragments:
                    render_hash = str(fragment.get("render_hash") or "")
                    if (
                        all(
                            domain_fragment in composed_fragments
                            for domain_fragment in domain_fragments
                        )
                        and fragment in composed_fragments
                        and composition_receipt.get(
                            "pending_contract_hash"
                        )
                        == render_manifest.get("pending_contract_hash")
                        and len(visible_hashes) == 1
                        and visible_hashes[0]
                        == composition_receipt.get(
                            "terminal_response_hash"
                        )
                        and visible_terminal_hashes.get(turn_index)
                        == visible_hashes[0]
                        and recomputed is not None
                        and recomputed["terminal_response_hash"]
                        == composition_receipt.get(
                            "terminal_response_hash"
                        )
                        and recomputed["terminal_semantic_hash"]
                        == composition_receipt.get(
                            "terminal_semantic_hash"
                        )
                        and isinstance(fragment_count, int)
                        and fragment_count == 1
                    ):
                        matches.append({
                            "turn_index": turn_index,
                            "orientation_receipt_id": str(
                                orientation.get("receipt_id") or ""
                            ),
                            "domain_commit_receipt_id": str(
                                commit.get("receipt_id") or ""
                            ),
                            "composition_receipt_id": str(
                                composition.get("receipt_id") or ""
                            ),
                            "message_id": execution_explanation_message,
                            "render_hash": render_hash,
                            "visible_render_authority": "render_manifest_hash",
                        })
    invalid = [
        *invalid_orientations,
        *invalid_commits,
        *invalid_compositions,
    ]
    return bool(matches) and not invalid, _receipt_details(
        "orientation_execution_explanation",
        orientations,
        invalid,
        matches=matches,
    )


def _recompute_preflight_composition(
    fragments: Sequence[Mapping[str, Any]],
    *,
    pending_contract: Mapping[str, Any],
    language: str,
) -> dict[str, str] | None:
    allowed_messages = {
        "harness.orientation.consultation.preflight_smoke",
        "harness.orientation.consultation.requirements",
        "harness.orientation.consultation.requirements_with_recommendation",
    }
    texts: list[str] = []
    semantic_hashes: list[str] = []
    for item in fragments:
        role = str(item.get("role") or "")
        message_id = str(item.get("message_id") or "")
        if role == "pending_question":
            if (
                not pending_contract
                or message_id
                != str(pending_contract.get("id") or "")
            ):
                return None
            text = render_question(pending_contract, language).strip()
            item_semantic_hash = semantic_hash({
                "kind": "pending_question",
                "question": dict(pending_contract),
            })
            item_render_hash = render_hash(text)
        elif message_id in allowed_messages and role == "message":
            rendered = render_fragment(
                ResponseFragment(
                    kind="message",
                    message_id=message_id,
                    source=__name__,
                ),
                language,
            )
            text = rendered.text
            item_semantic_hash = rendered.semantic_hash
            item_render_hash = rendered.render_hash
        else:
            return None
        if (
            item.get("semantic_hash") != item_semantic_hash
            or item.get("render_hash") != item_render_hash
        ):
            return None
        texts.append(text)
        semantic_hashes.append(item_semantic_hash)
    if not any(
        item.get("message_id")
        == "harness.orientation.consultation.preflight_smoke"
        for item in fragments
    ):
        return None
    terminal = "\n".join(texts).strip()
    return {
        "terminal_response_hash": render_hash(terminal),
        "terminal_semantic_hash": semantic_hash(semantic_hashes),
    }


def _approved_execution_submitted_once(context: Any) -> PredicateResult:
    from agent.runners.job_manager import verify_job_receipt

    contract, contract_error = _verifier_input(context)
    required_turns = {
        int(step.get("turn_index") or -1)
        for step in (
            (contract or {}).get("source_contract", {}).get(
                "source_steps"
            )
            or ()
        )
        if isinstance(step, Mapping)
        and step.get("semantic_role") == "approve_execution"
    } if contract is not None else set()
    approvals, invalid_approvals = _valid_receipts(
        context,
        "execution_approval",
    )
    pending_resolutions, invalid_pending = _valid_receipts(
        context,
        "pending_resolution",
    )
    commits, invalid_commits = _domain_commits(context)
    approvals_by_turn = defaultdict(list)
    pending_by_turn = defaultdict(list)
    commits_by_turn = defaultdict(list)
    for item in approvals:
        approvals_by_turn[int(item["turn_index"])].append(item)
    for item in pending_resolutions:
        pending_by_turn[int(item["turn_index"])].append(item)
    for item in commits:
        commits_by_turn[int(item["turn_index"])].append(item)
    submissions: dict[str, set[tuple[str, str]]] = defaultdict(set)
    verified_turns: list[int] = []
    for source_turn_index, event in enumerate(_events(context), start=1):
        runtime_turn_index = int(getattr(event, "turn_index", -1))
        summary = dict(
            getattr(event, "execution_receipt_summary", {}) or {}
        )
        manager_receipt = dict(
            summary.get("manager_submission_receipt") or {}
        )
        key = str(
            summary.get("receipt_idempotency_key")
            or summary.get("intent_idempotency_key")
            or ""
        )
        manager_receipt_id = str(
            summary.get("manager_submission_receipt_id") or ""
        )
        job_id = str(summary.get("job_id") or "")
        all_admitted = [
            action
            for action in (
                getattr(event, "admitted_action_provenance", ()) or ()
            )
            if isinstance(action, Mapping)
            and str(action.get("action_id") or "")
        ]
        admitted = [
            action
            for action in all_admitted
            if str(action.get("type") or "")
            in {
                "approve_preflight_smoke",
                "approve_final_benchmark",
            }
        ]
        consumed = {
            str(action_id)
            for commit in commits_by_turn.get(runtime_turn_index, ())
            if commit["receipt"].get("owner") == "execution"
            for action_id in (
                commit["receipt"].get("consumed_action_ids") or ()
            )
        }
        bound_approvals = [
            approval["receipt"]
            for approval in approvals_by_turn.get(
                runtime_turn_index,
                (),
            )
            if approval["receipt"].get("approval_action_id")
            in {
                str(action.get("action_id") or "")
                for action in admitted
                if str(action.get("action_id") or "") in consumed
            }
        ]
        intent_projection = dict(
            summary.get("side_effect_intent_projection") or {}
        )
        effect_projection = dict(
            summary.get("side_effect_receipt_projection") or {}
        )
        pending_receipts = {
            str(item["receipt"].get("receipt_id") or ""): item["receipt"]
            for item in pending_by_turn.get(runtime_turn_index, ())
        }
        if (
            key
            and verify_job_receipt(manager_receipt)
            and manager_receipt.get("receipt_type")
            == "job_submission"
            and manager_receipt_id
            == manager_receipt.get("receipt_id")
            and job_id
            and manager_receipt.get("job_id") == job_id
            and summary.get("manager_submission_disposition")
            in {"created", "reused"}
            and manager_receipt.get("disposition")
            == summary.get("manager_submission_disposition")
            and int(summary.get("manager_matching_job_count") or 0) == 1
            and manager_receipt.get("matching_job_count") == 1
            and manager_receipt.get("execution_key_hash")
            == summary.get("manager_execution_key_hash")
            == content_hash(key)
            and any(
                approval.get("job_id") == job_id
                and approval.get("job_submission_receipt_id")
                == manager_receipt_id
                and approval.get("job_submission_receipt_hash")
                == content_hash(manager_receipt)
                and approval.get("approval_action_type")
                in {
                    str(action.get("type") or "")
                    for action in admitted
                }
                and approval.get("idempotency_key_hash")
                == content_hash(key)
                and intent_projection.get("execution_request_id")
                == approval.get("execution_request_id")
                and key
                == f"harness:{approval.get('execution_request_id')}"
                and approval.get("side_effect_intent_id")
                == intent_projection.get("intent_id")
                == summary.get("intent_id")
                and approval.get("side_effect_intent_hash")
                == content_hash(intent_projection)
                and approval.get("side_effect_receipt_id")
                == effect_projection.get("receipt_id")
                == summary.get("receipt_id")
                and approval.get("side_effect_receipt_hash")
                == content_hash(effect_projection)
                and intent_projection.get("action_id")
                == effect_projection.get("action_id")
                == approval.get("approval_action_id")
                and effect_projection.get("intent_id")
                == intent_projection.get("intent_id")
                and effect_projection.get("job_id") == job_id
                and (
                    lambda pending_receipt: (
                        bool(pending_receipt)
                        and pending_receipt.get("resolved_action_id")
                        == approval.get("answer_action_id")
                        and pending_receipt.get("selected_value_hash")
                        == content_hash(True)
                        and any(
                            str(action.get("action_id") or "")
                            == approval.get("answer_action_id")
                            and str(action.get("type") or "")
                            == "answer_pending"
                            for action in all_admitted
                        )
                    )
                )(
                    pending_receipts.get(
                        str(
                            approval.get(
                                "pending_resolution_receipt_id"
                            )
                            or ""
                        ),
                        {},
                    )
                )
                for approval in bound_approvals
            )
        ):
            submissions[key].add((manager_receipt_id, job_id))
            verified_turns.append(source_turn_index)
    satisfied = (
        len(submissions) == 1
        and all(
        len(identities) == 1 for identities in submissions.values()
        )
        and not invalid_approvals
        and not invalid_pending
        and not invalid_commits
        and not contract_error
        and (
            not required_turns
            or required_turns.issubset(set(verified_turns))
        )
    )
    return satisfied, {
        "evidence_family": "job_submission",
        "submission_identities": {
            key: sorted(identities)
            for key, identities in submissions.items()
        },
        "verified_source_turn_indexes": sorted(set(verified_turns)),
        "required_source_turn_indexes": sorted(required_turns),
        "invalid_execution_approval_receipts": invalid_approvals,
        "invalid_pending_resolution_receipts": invalid_pending,
        "invalid_domain_commit_receipts": invalid_commits,
        "verifier_contract_error": contract_error,
    }


def _duplicate_job_submission(context: Any) -> PredicateResult:
    from agent.runners.job_manager import verify_job_receipt

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
        receipt = dict(summary.get("manager_submission_receipt") or {})
        identity = (
            str(receipt.get("receipt_id") or ""),
            str(receipt.get("job_id") or ""),
        )
        if (
            key
            and verify_job_receipt(receipt)
            and receipt.get("receipt_type") == "job_submission"
        ):
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
    from agent.runners.job_manager import verify_job_receipt

    contract, contract_error = _verifier_input(context)
    if contract is None:
        return False, {
            "evidence_family": "job_read",
            "status_intent_turn_indexes": [],
            "verified_status_turn_indexes": [],
            "unverified_status_turn_indexes": [],
            "required_status_turn_indexes": [],
            "verifier_contract_error": contract_error,
        }
    required_status_turns = {
        int(step.get("turn_index") or -1)
        for step in (
            (contract or {}).get("source_contract", {}).get(
                "source_steps"
            )
            or ()
        )
        if isinstance(step, Mapping)
        and step.get("semantic_role")
        in {
            "execution_evidence_consultation",
            "job_status_consultation",
        }
    }
    status_intents: list[int] = []
    matched: list[int] = []
    unverified: list[int] = []
    orientation_receipts, invalid_orientations = _valid_receipts(
        context,
        "orientation_response",
    )
    visible_orientation, invalid_visible = _visible_orientation_authority(
        context,
        topics=frozenset({
            "current_context",
            "current_job",
            "job_status",
            "execution_status",
        }),
        required_message_ids=frozenset({
            "harness.orientation.consultation.job_verified",
        }),
    )
    commits, invalid_commits = _domain_commits(context)
    orientation_by_turn: dict[int, dict[str, Mapping[str, Any]]] = (
        defaultdict(dict)
    )
    for item in orientation_receipts:
        receipt = item["receipt"]
        if (
            receipt.get("read_only") is True
            and receipt.get("action_type") == "answer_opening_question"
            and receipt.get("topic")
            in {
                "current_context",
                "current_job",
                "job_status",
                "execution_status",
            }
        ):
            orientation_by_turn[int(item["turn_index"])][
                str(receipt.get("action_id") or "")
            ] = receipt
    verified_orientation_consumers: dict[int, set[str]] = defaultdict(set)
    for item in commits:
        receipt = item["receipt"]
        if (
            receipt.get("owner") == "orientation"
            and any(
                isinstance(fragment, Mapping)
                and fragment.get("message_id")
                == "harness.orientation.consultation.job_verified"
                for fragment in receipt.get("response_fragments") or ()
            )
        ):
            verified_orientation_consumers[int(item["turn_index"])].update(
                str(action_id)
                for action_id in receipt.get("consumed_action_ids") or ()
                if str(action_id)
            )
    for source_turn_index, event in enumerate(_events(context), start=1):
        runtime_turn_index = int(getattr(event, "turn_index", -1))
        admitted_actions = [
            action
            for action in (
                getattr(event, "admitted_action_provenance", ()) or ()
            )
            if isinstance(action, Mapping)
            and str(action.get("action_id") or "")
        ]
        explicit_status_actions = [
            action
            for action in admitted_actions
            if str(action.get("type") or "")
            in {"status", "read_status", "analyze_job"}
        ]
        consumed_explicit_action_ids = {
            str(action_id)
            for commit in commits
            if int(commit["turn_index"]) == runtime_turn_index
            for action_id in (
                commit["receipt"].get("consumed_action_ids") or ()
            )
            if commit["receipt"].get("owner")
            in {
                str(action.get("owner") or "")
                for action in explicit_status_actions
            }
        }
        orientation_status_actions = [
            action
            for action in admitted_actions
            if (
                str(action.get("type") or "")
                == "answer_opening_question"
                and str(action.get("action_id") or "")
                in orientation_by_turn.get(runtime_turn_index, {})
                and str(action.get("action_id") or "")
                in verified_orientation_consumers.get(
                    runtime_turn_index,
                    set(),
                )
                and str(action.get("action_id") or "")
                in visible_orientation.get(runtime_turn_index, set())
            )
        ]
        orientation_job_reads = []
        for action in orientation_status_actions:
            orientation_receipt = orientation_by_turn.get(
                runtime_turn_index,
                {},
            ).get(str(action.get("action_id") or ""), {})
            orientation_job_reads.extend(
                dict(receipt)
                for receipt in (
                    orientation_receipt.get("source_receipts") or ()
                )
                if isinstance(receipt, Mapping)
                and receipt.get("receipt_type") == "job_read"
                and verify_job_receipt(dict(receipt))
            )
        summary = dict(
            getattr(event, "execution_receipt_summary", {}) or {}
        )
        if not explicit_status_actions and not orientation_status_actions:
            continue
        status_intents.append(source_turn_index)
        admitted_types = {
            str(action.get("type") or "")
            for action in explicit_status_actions
        }
        orientation_verified = bool(
            orientation_status_actions
            and orientation_job_reads
            and all(
                receipt.get("job_id")
                and receipt.get("observed_status")
                == receipt.get("persisted_status")
                for receipt in orientation_job_reads
            )
        )
        execution_verified = bool(
            summary.get("intent_action_type") in admitted_types
            and any(
                str(action.get("action_id") or "")
                in consumed_explicit_action_ids
                for action in explicit_status_actions
            )
            and (
                lambda receipt: (
                    verify_job_receipt(receipt)
                    and receipt.get("receipt_type") == "job_read"
                    and receipt.get("receipt_id")
                    == summary.get("manager_read_receipt_id")
                    and receipt.get("job_id") == summary.get("job_id")
                    and receipt.get("observed_status")
                    == receipt.get("persisted_status")
                    and receipt.get("observed_status")
                    == summary.get("manager_observed_status")
                    == summary.get("job_status")
                )
            )(dict(summary.get("manager_read_receipt") or {}))
        )
        verified = orientation_verified or execution_verified
        if verified:
            matched.append(source_turn_index)
        else:
            unverified.append(source_turn_index)
    return (
        bool(status_intents)
        and len(matched) == len(status_intents)
        and not unverified
        and not invalid_orientations
        and not invalid_commits
        and not invalid_visible
        and (
            not required_status_turns
            or required_status_turns.issubset(set(matched))
        )
    ), {
        "evidence_family": "job_read",
        "status_intent_turn_indexes": status_intents,
        "verified_status_turn_indexes": matched,
        "unverified_status_turn_indexes": unverified,
        "required_status_turn_indexes": sorted(required_status_turns),
        "verifier_contract_error": contract_error,
        "invalid_orientation_receipts": invalid_orientations,
        "invalid_domain_commit_receipts": invalid_commits,
        "invalid_visible_orientation_receipts": invalid_visible,
    }


def _invented_job_status(context: Any) -> PredicateResult:
    from agent.runners.job_manager import verify_job_receipt

    invented = []
    for event in _events(context):
        admitted_types = {
            str(action.get("type") or "")
            for action in (
                getattr(event, "admitted_action_provenance", ()) or ()
            )
            if isinstance(action, Mapping)
            and str(action.get("action_id") or "")
        }
        summary = dict(
            getattr(event, "execution_receipt_summary", {}) or {}
        )
        receipt = dict(summary.get("manager_read_receipt") or {})
        if (
            summary.get("intent_action_type")
            in {"status", "read_status", "analyze_job"}
            and summary.get("intent_action_type") in admitted_types
            and summary.get("job_status")
            and not (
                verify_job_receipt(receipt)
                and receipt.get("receipt_type") == "job_read"
                and receipt.get("receipt_id")
                == summary.get("manager_read_receipt_id")
                and receipt.get("job_id") == summary.get("job_id")
                and receipt.get("observed_status")
                == receipt.get("persisted_status")
                and receipt.get("observed_status")
                == summary.get("manager_observed_status")
                == summary.get("job_status")
            )
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
        action_order = [
            str(value) for value in summary.get("semantic_order") or ()
        ]
        action_bindings = {
            str(action_id): [
                str(unit_id)
                for unit_id in bound_units or ()
                if str(unit_id)
            ]
            for action_id, bound_units in dict(
                summary.get("action_unit_bindings") or {}
            ).items()
        }
        bound_unit_order: list[str] = []
        for action_id in action_order:
            for unit_id in action_bindings.get(action_id, ()):
                if unit_id not in bound_unit_order:
                    bound_unit_order.append(unit_id)
        expected_action_units = [
            str(unit.get("unit_id") or "")
            for unit in units
            if isinstance(unit, Mapping)
            and str(unit.get("disposition") or "") == "action"
        ]
        spans_by_clause: dict[str, list[tuple[int, int]]] = defaultdict(list)
        valid_spans = True
        for unit in units:
            if not isinstance(unit, Mapping):
                valid_spans = False
                continue
            start = unit.get("start")
            end = unit.get("end")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or not 0 <= start < end
            ):
                valid_spans = False
                continue
            spans_by_clause[str(unit.get("clause_id") or "")].append(
                (start, end)
            )
        clause_spans_ordered = all(
            all(
                spans[index][1] <= spans[index + 1][0]
                for index in range(len(spans) - 1)
            )
            for spans in spans_by_clause.values()
        )
        if (
            unit_ids
            and len(unit_ids) == len(set(unit_ids))
            and valid_spans
            and clause_spans_ordered
            and bound_unit_order == expected_action_units
        ):
            matched.append(item)
    return bool(matched) and not invalid, _receipt_details(
        "semantic_units",
        matched,
        invalid,
        matched_receipt_count=len(matched),
    )


def _admitted_mutations_only(context: Any) -> PredicateResult:
    events = _events(context)
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
    return bool(events) and not violations and not invalid, _receipt_details(
        "admission_lineage",
        commits,
        invalid,
        event_count=len(events),
        mutation_count=mutation_count,
        violating_receipt_ids=violations,
    )


def _unresolved_units_preserved(context: Any) -> PredicateResult:
    preserved = []
    for event in _events(context):
        summary = dict(
            getattr(event, "turn_receipt_summary", {}) or {}
        )
        units = [
            dict(item)
            for item in summary.get("semantic_units") or ()
            if isinstance(item, Mapping)
            and str(item.get("unit_id") or "")
        ]
        unit_ids = {str(item["unit_id"]) for item in units}
        unresolved = {
            str(unit_id)
            for unit_id in summary.get("unresolved_unit_ids") or ()
            if str(unit_id)
        }
        admitted = {
            str(item.get("action_id") or ""): dict(item)
            for item in getattr(event, "admitted_action_provenance", ()) or ()
            if isinstance(item, Mapping)
            and str(item.get("action_id") or "")
        }
        executed = {
            str(action_id)
            for action_id in summary.get("execution_order") or ()
            if str(action_id)
        }
        deferred = [
            item
            for action_id, item in admitted.items()
            if action_id not in executed
        ]
        queued_types = Counter(
            str(action_type)
            for action_type in getattr(event, "action_queue_types", ()) or ()
            if str(action_type)
        )
        deferred_types = Counter(
            str(item.get("type") or "")
            for item in deferred
            if str(item.get("type") or "")
        )
        queued_actions_preserved = all(
            queued_types[action_type] >= count
            for action_type, count in deferred_types.items()
        )
        bindings = {
            str(action_id): {
                str(unit_id)
                for unit_id in bound_units or ()
                if str(unit_id)
            }
            for action_id, bound_units in dict(
                summary.get("action_unit_bindings") or {}
            ).items()
        }
        represented_units = set(unresolved)
        for action_id in executed:
            represented_units.update(bindings.get(action_id, set()))
        if queued_actions_preserved:
            for item in deferred:
                represented_units.update(
                    bindings.get(str(item.get("action_id") or ""), set())
                )
        required_units = {
            str(item["unit_id"])
            for item in units
            if str(item.get("disposition") or "") in {"action", "unresolved"}
        }
        if (
            unresolved <= unit_ids
            and required_units <= represented_units
            and (unresolved or deferred)
        ):
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
    "real_node_selection_executed_at_source": (
        _real_node_selection_executed_at_source
    ),
    "mode_consultation_preserves_chain_pending": (
        _mode_consultation_preserves_chain_pending
    ),
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
