"""Semantic document preparation and admission for hierarchical planning."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from ..llm.types import ReasoningMode
from .action_registry import (
    ACTION_ARGUMENT_SCHEMAS,
    ACTION_BY_TYPE,
    ACTION_SPECS,
    CONSULTATION_TOPIC_PURPOSES,
    TRUSTED_ACTION_METADATA_FIELDS,
    answer_pending_representation_conflict,
    build_admission_transaction_hash,
    build_field_intake_admission_receipt,
    build_proposal_field_receipt,
    build_replacement_intake_admission_receipt,
    build_semantic_consensus_receipt,
    canonical_consultation_topic,
    normalize_current_action_envelope,
    resolve_action_target_group,
    semantic_value_domain_conflicts,
    semantic_grounding_arguments,
    semantic_scope_accepts_action,
    semantic_scope_schema,
    state_has_capability,
    validate_action_contract,
    validate_action_transaction_contract,
)
from .context import action_schema, group_schema, workflow_snapshot
from .domains.environment import (
    extract_structured_input_candidates,
    normalize_proposed_config_value,
)
from .plan_coverage import (
    PlanCoverageResult,
    TurnClause,
    validate_plan_coverage,
)
from .questions import (
    coerce_pending_answer,
    declared_option_label_variants,
    exact_answer,
    manual_action_for_value,
    pending_option_value_exists,
    pending_value_identity,
    semantic_pending_question,
    typed_pending_value_candidates,
    value_satisfies_pending_contract,
)
from .semantic_compiler import (
    ImmutableSemanticPlan,
    STRICT_JSON_REASONING_MODE,
    WholePlanAdmission,
    freeze_semantic_plan,
    request_whole_plan_admission,
)
from .semantic_policy import FRAMED_REQUEST_SEMANTIC_POLICY
from .state import AgentGraphState
from agent.workflows.group_registry import GROUP_SPEC_BY_NAME

ALLOWED_ACTION_TYPES = [spec.action_type for spec in ACTION_SPECS]

_ADMISSION_RECEIPT_KEYS = frozenset({
    "pending_choice_contracts",
    "pending_answer_admissions",
    "pending_support_unit_ids",
    "semantic_support_unit_ids",
    "chain_selection_admissions",
    "target_mode_selection_admissions",
    "consultation_admissions",
    "group_navigation_admissions",
    "field_reconfiguration_admissions",
    "replacement_intake_admissions",
    "admission_rejections",
    "admission_action_ids",
})


def _content_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

GROUP_NAVIGATION_SEMANTIC_POLICY = (
    "A group-navigation purpose is supported when the source asks to visit, return to, or configure a named "
    "area without making a more specific mutation request. It intentionally opens that group's later typed "
    "questions and does not carry destination fields or values. A value-less statement that the user intends to "
    "inspect, revise, or configure that same destination explains why the group is being opened; it is part of "
    "the navigation/intake transaction, not a second mutation that navigation must execute immediately. A request "
    "to enter the destination so the user can inspect, compare, or decide among that group's later typed options "
    "is also one navigation/intake transaction; it neither selects an option nor creates a separate unresolved demand. "
    "However, when the supplied group_schema for that destination declares an entry_action whose purpose matches "
    "the source request, generic navigation is unsupported: the declared typed entry action owns that subflow and "
    "its registry-supplied entry arguments. An action that merely shares target_group but is not listed in "
    "entry_actions is never inferred as an entry. "
    "becomes an owner mutation only when the source explicitly asks to alter, customize, override, or select a "
    "concrete field, value, or default that a registered owner action can apply. Naming the destination resource "
    "again, or expressing a general desire to revise it without a concrete setting or value, is not sufficient. "
    "Never manufacture a second owner mutation merely because navigation intentionally defers those questions. "
)


def prepare_hierarchical_candidate(
    raw_response: str,
    state: AgentGraphState,
    clauses: tuple[TurnClause, ...],
    *,
    pending_choice_unit_ids: frozenset[str],
    semantic_support_unit_ids: frozenset[str] = frozenset(),
) -> tuple[str, PlanCoverageResult]:
    """Prepare Stage A/B output without invoking legacy planner repair.

    Hierarchical planning already owns source partitioning, owner routing, and
    action compilation. This boundary may apply registry/state policy and the
    typed pending contract, but it must not synthesize omitted semantic units,
    reinterpret structured ownership, or repair planner output.
    """

    try:
        candidate = _prepare_untrusted_action_document(raw_response)
        _reject_conflicting_pending_representations(candidate)
        candidate = _reset_candidate_action_ids(candidate)
        candidate = _canonicalize_pending_choice_actions(
            candidate,
            state,
            eligible_unit_ids=pending_choice_unit_ids,
        )
        candidate = _mark_pending_owner_candidates(
            candidate,
            state,
            eligible_unit_ids=pending_choice_unit_ids,
        )
        if semantic_support_unit_ids:
            trusted_payload = _parse_json_object(candidate)
            trusted_payload["semantic_support_unit_ids"] = sorted(
                semantic_support_unit_ids
            )
            candidate = json.dumps(
                trusted_payload,
                ensure_ascii=False,
                sort_keys=True,
            )
        validation = _validate_action_document(candidate, clauses, state)
    except ValueError as exc:
        candidate = "{}"
        validation = _invalid_plan_coverage(clauses, (str(exc),))
    return candidate, validation


def _canonicalize_pending_choice_actions(
    text: str,
    state: AgentGraphState,
    *,
    eligible_unit_ids: frozenset[str] | None = None,
) -> str:
    """Bind every semantic option selection to one canonical pending answer.

    The compiler may describe a selection as ``answer_pending`` or as the
    option's declared owner action.  Both representations become the same
    pre-admission action.  The owner action is intentionally not executed here;
    the pending-question contract dispatches it after whole-plan admission.
    """

    pending = dict(state.get("pending_question") or {})
    options = [item for item in pending.get("options") or [] if isinstance(item, dict)]
    if not options and not isinstance(pending.get("manual_action"), Mapping):
        return text
    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    contracts: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        units = _source_units_for_action(payload, index)
        if eligible_unit_ids is not None and not any(
            str(unit.get("unit_id") or "") in eligible_unit_ids
            for unit in units
        ):
            continue
        option = _matching_pending_option(action, state)
        manual_value = None
        if not option:
            manual_value = (
                _manual_value_from_answer_pending(action, pending)
                if str(action.get("type") or "") == "answer_pending"
                else _matching_pending_manual_value(action, pending)
            )
        if manual_value is not None:
            source = str(action.get("source_evidence") or "").strip()
            units = _source_units_for_action(payload, index)
            if not source or not any(
                source in str(unit.get("source_text") or "") for unit in units
            ):
                continue
            if _action_answers_bound_semantic_draft(action, pending):
                continue
            declared_manual_action = manual_action_for_value(
                pending,
                manual_value,
            )
            if (
                declared_manual_action
                and str(declared_manual_action.get("type") or "")
                != "answer_pending"
                and (
                    isinstance(
                        pending.get("semantic_draft_binding"),
                        Mapping,
                    )
                    or isinstance(
                        pending.get("secret_reentry_binding"),
                        Mapping,
                    )
                )
            ):
                declared_spec = ACTION_BY_TYPE.get(
                    str(declared_manual_action.get("type") or "")
                )
                if (
                    declared_spec is not None
                    and "source_evidence" in declared_spec.allowed_arguments
                    and not str(
                        declared_manual_action.get("source_evidence") or ""
                    ).strip()
                ):
                    declared_manual_action["source_evidence"] = source
                declared_manual_action["confidence"] = str(
                    action.get("confidence") or "medium"
                )
                actions[index] = validate_action_contract(
                    declared_manual_action
                )
                continue
            canonical = {
                "type": "answer_pending",
                "answer": manual_value,
                "source_evidence": source,
                "confidence": str(action.get("confidence") or "medium"),
            }
            if not _manual_answer_has_literal_source(canonical, source):
                if (
                    str(pending.get("value_domain") or "")
                    != "researched_identity"
                    or not value_satisfies_pending_contract(source, pending)
                ):
                    continue
                manual_value = coerce_pending_answer(source, pending)
                canonical["answer"] = manual_value
                if not _manual_answer_has_literal_source(canonical, source):
                    continue
            actions[index] = canonical
            continue
        if not option:
            continue
        action_type = str(action.get("type") or "")
        if action_type != "answer_pending":
            spec = ACTION_BY_TYPE.get(action_type)
            if spec is None or not spec.pending_option_admission:
                continue
        if not units:
            raise ValueError("pending choice has no mapped semantic-unit provenance")
        source = str(action.get("source_evidence") or "").strip()
        if not any(source and source in str(unit.get("source_text") or "") for unit in units):
            source = str(units[0].get("source_text") or "").strip()
        if not source:
            raise ValueError("pending choice has no exact semantic-unit source")
        selected = option.get("value")
        actions[index] = {
            "type": "answer_pending",
            "selected_value": selected,
            "source_evidence": source,
            "confidence": str(action.get("confidence") or "medium"),
        }
        contracts.append({
            "action_index": index,
            "question": {
                "id": str(pending.get("id") or ""),
                "group": str(pending.get("group") or ""),
                "contract_version": pending.get("contract_version"),
            },
            "option": {
                "id": str(option.get("id") or ""),
                "selected_value": selected,
            },
            "semantic_units": [
                {
                    "unit_id": str(unit.get("unit_id") or ""),
                    "clause_id": str(unit.get("clause_id") or ""),
                    "source_text": str(unit.get("source_text") or ""),
                }
                for unit in units
            ],
        })
    if len(contracts) > 1:
        raise ValueError("multiple actions claim the active pending choice")
    payload["actions"] = actions
    if contracts:
        payload["pending_choice_contracts"] = contracts
    payload = _coalesce_equivalent_manual_pending_answers(payload, pending)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _coalesce_equivalent_manual_pending_answers(
    payload: dict[str, Any],
    pending: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge duplicate typed answers while preserving every source-unit owner.

    A compound manual value may be repeated by the semantic compiler with a
    different exact evidence excerpt for each source unit. The pending
    contract owns one value, so those equivalent operations are one immutable
    action with multiple source units, not competing mutations.
    """

    if pending.get("manual_input_allowed") is not True:
        return payload
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    first_by_value: dict[str, int] = {}
    retained: list[Any] = []
    old_to_new: dict[int, int] = {}
    changed = False
    for old_index, action in enumerate(actions):
        if not isinstance(action, Mapping) or str(action.get("type") or "") != "answer_pending":
            old_to_new[old_index] = len(retained)
            retained.append(action)
            continue
        option = _declared_option_for_pending_answer(dict(action), dict(pending))
        answer = _manual_value_from_answer_pending(action, pending)
        if option or answer is None:
            old_to_new[old_index] = len(retained)
            retained.append(action)
            continue
        canonical = dict(action)
        canonical["answer"] = answer
        canonical.pop("selected_value", None)
        identity = json.dumps(answer, ensure_ascii=False, sort_keys=True, default=str)
        prior = first_by_value.get(identity)
        if prior is None:
            prior = len(retained)
            first_by_value[identity] = prior
            retained.append(canonical)
            changed = changed or canonical != dict(action)
        else:
            changed = True
        old_to_new[old_index] = prior
    if not changed:
        return payload
    payload["actions"] = retained
    for unit in payload.get("semantic_units") or []:
        if not isinstance(unit, dict) or not isinstance(unit.get("action_indexes"), list):
            continue
        unit["action_indexes"] = list(dict.fromkeys(
            old_to_new[index]
            for index in unit["action_indexes"]
            if isinstance(index, int) and index in old_to_new
        ))
    for contract in payload.get("pending_choice_contracts") or []:
        if not isinstance(contract, dict):
            continue
        index = contract.get("action_index")
        if isinstance(index, int) and index in old_to_new:
            contract["action_index"] = old_to_new[index]
    return payload


def _pending_choice_contract_for_action(
    payload: Mapping[str, Any],
    action_index: int,
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in payload.get("pending_choice_contracts") or []
        if isinstance(row, Mapping) and row.get("action_index") == action_index
    ]
    if len(matches) > 1:
        raise ValueError("pending choice has duplicate canonical contracts")
    return matches[0] if matches else {}


def _validate_pending_choice_contracts(
    payload: Mapping[str, Any],
    state: AgentGraphState,
) -> tuple[str, ...]:
    pending = dict(state.get("pending_question") or {})
    errors: list[str] = []
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping) or str(action.get("type") or "") != "answer_pending":
            continue
        option = _declared_option_for_pending_answer(dict(action), pending)
        if not option:
            continue
        contract = _pending_choice_contract_for_action(payload, index)
        question = contract.get("question") if isinstance(contract.get("question"), Mapping) else {}
        selected = (
            contract.get("option", {}).get("selected_value")
            if isinstance(contract.get("option"), Mapping)
            else None
        )
        units = contract.get("semantic_units") if isinstance(contract.get("semantic_units"), list) else []
        mapped_units = _source_units_for_action(payload, index)
        if not contract:
            errors.append(f"action {index} pending choice lacks canonical contract")
        elif str(question.get("id") or "") != str(pending.get("id") or ""):
            errors.append(f"action {index} pending choice question identity differs from active contract")
        elif selected != option.get("value") or selected != action.get("selected_value"):
            errors.append(f"action {index} pending choice selected value differs from canonical contract")
        elif not units or [str(unit.get("unit_id") or "") for unit in units if isinstance(unit, Mapping)] != [
            str(unit.get("unit_id") or "") for unit in mapped_units
        ]:
            errors.append(f"action {index} pending choice provenance differs from mapped semantic units")
    return tuple(errors)


def _reset_candidate_action_ids(text: str) -> str:
    payload = _parse_json_object(text)
    payload.pop("admission_action_ids", None)
    return _prepare_untrusted_action_document(
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _mark_pending_owner_candidates(
    text: str,
    state: AgentGraphState,
    *,
    eligible_unit_ids: frozenset[str] | None = None,
) -> str:
    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    admitted = [
        index
        for index, action in enumerate(actions)
        if (
            isinstance(action, dict)
            and (
                eligible_unit_ids is None
                or any(
                    str(unit.get("unit_id") or "") in eligible_unit_ids
                    for unit in _source_units_for_action(payload, index)
                )
            )
            and _action_owns_pending_candidate(action, state)
        )
    ]
    if len(admitted) > 1:
        raise ValueError("multiple actions claim the active pending-question owner")
    if admitted:
        payload["pending_answer_admissions"] = admitted
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _action_owns_pending_candidate(
    action: dict[str, Any],
    state: AgentGraphState,
) -> bool:
    pending = dict(state.get("pending_question") or {})
    if not pending:
        return False
    if _action_answers_bound_semantic_draft(action, pending):
        return True
    if str(action.get("type") or "") != "answer_pending":
        option = _matching_pending_option(action, state)
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        return bool(option and (spec is None or spec.pending_option_admission))
    if _declared_option_for_pending_answer(action, pending):
        return True
    answer = _manual_value_from_answer_pending(action, pending)
    if answer is None:
        return False
    canonical = dict(action)
    canonical["answer"] = answer
    source = str(action.get("source_evidence") or "")
    return bool(
        pending.get("manual_input_allowed") is True
        and _manual_answer_has_literal_source(canonical, source)
    )


def _validate_action_source_grounding(
    text: str,
    validation: PlanCoverageResult,
) -> PlanCoverageResult:
    if not validation.valid:
        return validation
    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    errors: list[str] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        exact_arguments = tuple(spec.exact_source_value_arguments if spec is not None else ())
        source = str(action.get("source_evidence") or "").strip()
        if not source:
            if exact_arguments:
                errors.append(
                    f"action {index} has exact source values but no source_evidence"
                )
            continue
        units = _source_units_for_action(payload, index)
        if not any(
            source in evidence_source
            for unit in units
            for evidence_source in _semantic_unit_evidence_sources(unit)
        ):
            errors.append(f"action {index} source_evidence is not an exact mapped-unit quote")
            continue
        for value_argument in exact_arguments:
            value = action.get(value_argument)
            if value is not None and value != "" and not _exact_value_has_source(
                value,
                source,
            ):
                errors.append(
                    f"action {index} {value_argument} is not present in its exact source_evidence"
                )
    return _merge_plan_errors(validation, tuple(errors)) if errors else validation


def _exact_value_has_source(value: Any, source: str) -> bool:
    """Compare structured evidence by value and scalar evidence by exact text."""

    if isinstance(value, (dict, list)):
        try:
            return json.loads(source) == value
        except (json.JSONDecodeError, TypeError):
            return False
    return str(value).strip() in source


def _semantic_unit_evidence_sources(
    unit: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return original and Harness-bound clarification evidence."""

    return tuple(
        value
        for value in (
            str(unit.get("source_text") or ""),
            str(unit.get("resolution_evidence") or ""),
        )
        if value
    )


def _review_bounded_semantic_candidate(
    provider: Any,
    candidate: str,
    validation: PlanCoverageResult,
    state: AgentGraphState,
    clauses: tuple[TurnClause, ...],
    *,
    allowed_action_types: frozenset[str] | None = None,
    whole_plan_contract_repair: bool = False,
    compact_pending_review: bool = False,
    reasoning_mode: ReasoningMode = STRICT_JSON_REASONING_MODE,
    authoritative_direct_unit_ids: frozenset[str] = frozenset(),
) -> tuple[ImmutableSemanticPlan | None, WholePlanAdmission | None, tuple[str, ...]]:
    if not validation.valid:
        return None, None, tuple(validation.errors)
    try:
        plan = _freeze_bounded_semantic_plan(
            candidate,
            state,
            clauses,
            allowed_action_types=allowed_action_types,
            compact_pending_review=compact_pending_review,
            authoritative_direct_unit_ids=authoritative_direct_unit_ids,
        )
    except ValueError as exc:
        return None, None, (str(exc),)
    admission = request_whole_plan_admission(
        provider,
        plan,
        semantic_policy=_semantic_fulfillment_prompt(),
        allowed_action_types=(
            sorted(allowed_action_types)
            if allowed_action_types is not None
            else ALLOWED_ACTION_TYPES
        ),
        contract_repair=whole_plan_contract_repair,
        reasoning_mode=reasoning_mode,
    )
    return plan, admission, admission.errors


def _freeze_bounded_semantic_plan(
    text: str,
    state: AgentGraphState,
    clauses: tuple[TurnClause, ...],
    *,
    allowed_action_types: frozenset[str] | None = None,
    compact_pending_review: bool = False,
    authoritative_direct_unit_ids: frozenset[str] = frozenset(),
) -> ImmutableSemanticPlan:
    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    action_ids = _ensure_admission_action_ids(payload)
    unit_ids = [str(unit.get("unit_id") or "") if isinstance(unit, dict) else "" for unit in units]
    clause_shapes = {clause.clause_id: clause.input_shape for clause in clauses}
    unit_owner_indexes = [
        _semantic_unit_owner_indexes(unit, actions)
        if isinstance(unit, dict)
        else []
        for unit in units
    ]
    action_records: list[dict[str, Any]] = []
    pending = dict(state.get("pending_question") or {})
    turn_pending_value_candidates: list[dict[str, Any]] = []
    turn_candidate_by_identity: dict[str, dict[str, Any]] = {}
    for clause in clauses:
        for value in typed_pending_value_candidates(clause.text, pending):
            identity = pending_value_identity(value, pending)
            if not identity:
                continue
            source_unit_ids = [
                unit_ids[index]
                for index, unit in enumerate(units)
                if isinstance(unit, Mapping)
                and str(unit.get("clause_id") or "") == clause.clause_id
            ]
            existing = turn_candidate_by_identity.get(identity)
            if existing is not None:
                existing["source_unit_ids"] = list(dict.fromkeys([
                    *existing["source_unit_ids"],
                    *source_unit_ids,
                ]))
                continue
            candidate = {
                "candidate_id": f"turn-candidate-{len(turn_pending_value_candidates)}",
                "identity": identity,
                "value": value,
                "source_unit_ids": source_unit_ids,
            }
            turn_candidate_by_identity[identity] = candidate
            turn_pending_value_candidates.append(candidate)
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValueError("immutable semantic plan contains a non-object action")
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if spec is None:
            raise ValueError("immutable semantic plan contains an unregistered action")
        target_group = resolve_action_target_group(action)
        if not target_group and str(action.get("type") or "") == "change_group":
            target_group = str(action.get("group") or "")
        target_spec = GROUP_SPEC_BY_NAME.get(target_group)
        operation_arguments = _semantic_operation_arguments(action)
        owned_unit_ids = [
            unit_ids[unit_index]
            for unit_index, owners in enumerate(unit_owner_indexes)
            if index in owners
        ]
        scoped_turn_candidates = [
            {
                **candidate,
                "source_unit_ids": [
                    unit_id
                    for unit_id in candidate["source_unit_ids"]
                    if unit_id in owned_unit_ids
                ],
            }
            for candidate in turn_pending_value_candidates
            if any(
                unit_id in owned_unit_ids
                for unit_id in candidate["source_unit_ids"]
            )
        ]
        scoped_candidate_identities = {
            str(candidate.get("identity") or "")
            for candidate in scoped_turn_candidates
        }
        operation_candidates = _pending_operation_value_candidates(
            action,
            operation_arguments,
            pending,
        )
        pending_value_candidates = list(operation_candidates)
        action_turn_candidates = list(scoped_turn_candidates)
        for candidate in operation_candidates:
            identity = str(candidate.get("identity") or "")
            if identity in scoped_candidate_identities:
                continue
            action_turn_candidates.append({
                "candidate_id": (
                    f"action-{index + 1}-operation-candidate-"
                    f"{len(action_turn_candidates)}"
                ),
                "identity": identity,
                "value": candidate.get("value"),
                "source_unit_ids": owned_unit_ids,
            })
        if not pending_value_candidates:
            action_turn_candidates = []
        action_records.append({
            "action_id": action_ids[index],
            "action_index": index,
            "action": action,
            "registry_owner": spec.owner,
            "registry_effect": spec.effect,
            "registry_target_group": target_group,
            "registry_target_group_owner": target_spec.owner if target_spec is not None else "",
            "registry_incomplete_mutation_intake": spec.incomplete_mutation_intake,
            "registry_incomplete_read_intake": spec.incomplete_read_intake,
            "registry_pending_option_admission": spec.pending_option_admission,
            "declared_purpose": _semantic_action_purpose(action, spec.purpose, state),
            "operation_arguments": operation_arguments,
            "allowed_support_relations": list(spec.semantic_support_relations),
            "required_evidence_relations": [
                {
                    "unit_id": unit_id,
                    "relation": "direct",
                    "support_relation": "",
                }
                for unit_id in owned_unit_ids
                if unit_id in authoritative_direct_unit_ids
            ],
            "required_value_grounding_arguments": list(
                semantic_grounding_arguments(action)
            ),
            "open_identity_grounding_arguments": [
                argument
                for argument in spec.open_identity_grounding_arguments
                if argument in action
            ],
            "closed_enum_grounding_values": {
                argument: list(
                    (ACTION_ARGUMENT_SCHEMAS.get(argument) or {}).get("enum")
                    or []
                )
                for argument in semantic_grounding_arguments(action)
                if isinstance(
                    (ACTION_ARGUMENT_SCHEMAS.get(argument) or {}).get("enum"),
                    list,
                )
            },
            "exact_source_value_arguments": list(spec.exact_source_value_arguments),
            "pending_value_candidates": pending_value_candidates,
            "turn_pending_value_candidates": action_turn_candidates,
            "unit_ids": owned_unit_ids,
        })
    unit_records = [
        {
            "unit_id": unit_ids[index],
            "unit_index": index,
            "unit": unit,
            "clause_id": str(unit.get("clause_id") or ""),
            "source_text": str(unit.get("source_text") or ""),
            "resolution_evidence": str(
                unit.get("resolution_evidence") or ""
            ),
            "resolution_atom_id": str(
                unit.get("resolution_atom_id") or ""
            ),
            "resolution_binding_hash": str(
                unit.get("resolution_binding_hash") or ""
            ),
            "evidence_sources": list(
                _semantic_unit_evidence_sources(unit)
            ),
            "input_shape": clause_shapes.get(str(unit.get("clause_id") or ""), ""),
            "disposition": str(unit.get("disposition") or ""),
            "owner_action_ids": [action_ids[action_index] for action_index in unit_owner_indexes[index]],
        }
        for index, unit in enumerate(units)
        if isinstance(unit, dict)
    ]
    review_workflow_state = workflow_snapshot(state)
    review_group_schema = group_schema()
    review_pending = semantic_pending_question(pending)
    if compact_pending_review:
        pending_group = str(pending.get("group") or "")
        review_workflow_state = {
            "active_group": str(state.get("active_group") or ""),
            "language": str(state.get("language") or ""),
            "target_mode": str(state.get("target_mode") or ""),
            "workflow_mode": str(state.get("workflow_mode") or ""),
            "pending_question": review_pending,
        }
        review_group_schema = [
            row for row in group_schema()
            if str(row.get("name") or "") == pending_group
        ]
    return freeze_semantic_plan(
        payload,
        action_records=action_records,
        unit_records=unit_records,
        review_context={
            "original_request": {
                "user_text": "\n".join(clause.text for clause in clauses),
                "clauses": [clause.as_dict() for clause in clauses],
            },
            "pending_question": review_pending,
            "turn_pending_value_candidates": turn_pending_value_candidates,
            "workflow_state": review_workflow_state,
            "action_schema": action_schema(action_types=allowed_action_types),
            "group_schema": review_group_schema,
            "semantic_scope_schema": semantic_scope_schema(),
        },
    )


def _pending_operation_value_candidates(
    action: Mapping[str, Any],
    operation_arguments: Mapping[str, Any],
    pending: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expose only operation values eligible to answer the active contract.

    Candidate eligibility follows declared ownership. A direct pending owner
    contributes only its declared value argument. Structured configuration and
    domain mutations contribute only values explicitly named by the active
    question contract. Metadata and navigation arguments never become
    pending-answer candidates.
    """

    if pending.get("manual_input_allowed") is not True:
        return []
    action_type = str(action.get("type") or "")
    manual = pending.get("manual_action")
    manual_type = str(manual.get("type") or "") if isinstance(manual, Mapping) else ""
    manual_argument = (
        str(manual.get("value_argument") or "")
        if isinstance(manual, Mapping)
        else ""
    )
    output: list[dict[str, Any]] = []
    seen_values: set[str] = set()
    eligible_arguments: list[tuple[str, tuple[str, ...], Any]] = []
    if action_type == "answer_pending":
        for argument in ("selected_value", "answer"):
            if argument in operation_arguments:
                eligible_arguments.append((argument, (), operation_arguments[argument]))
    elif manual_type and action_type == manual_type and manual_argument:
        if manual_argument in operation_arguments:
            eligible_arguments.append(
                (manual_argument, (), operation_arguments[manual_argument])
            )
    elif (
        action_type == "propose_config_values"
        and str(pending.get("structured_config_key") or "").strip()
    ):
        structured_key = str(pending["structured_config_key"]).strip()
        config_values = operation_arguments.get("config_values")
        if isinstance(config_values, Mapping):
            for key, value in config_values.items():
                if str(key).strip().casefold() == structured_key.casefold():
                    eligible_arguments.append(
                        ("config_values", (str(key),), value)
                    )
    else:
        for binding in pending.get("candidate_bindings") or []:
            if not isinstance(binding, Mapping):
                continue
            if str(binding.get("type") or "") != action_type:
                continue
            argument = str(binding.get("value_argument") or "")
            if argument not in operation_arguments:
                continue
            value = operation_arguments[argument]
            mapping_key = str(binding.get("mapping_key") or "")
            if mapping_key:
                if not isinstance(value, Mapping) or mapping_key not in value:
                    continue
                eligible_arguments.append(
                    (argument, (mapping_key,), value[mapping_key])
                )
            else:
                eligible_arguments.append((argument, (), value))

    for argument, path, candidate in eligible_arguments:
        if not value_satisfies_pending_contract(candidate, dict(pending)):
            continue
        identity = pending_value_identity(candidate, dict(pending))
        if not identity or identity in seen_values:
            continue
        seen_values.add(identity)
        output.append({
            "candidate_id": f"candidate-{len(output)}",
            "argument": str(argument),
            "identity": identity,
            "path": list(path),
            "value": candidate,
        })
    return output


def _semantic_unit_owner_indexes(
    unit: Mapping[str, Any],
    actions: list[Any],
) -> list[int]:
    indexes = unit.get("action_indexes") if isinstance(unit.get("action_indexes"), list) else []
    if indexes:
        return [int(index) for index in indexes if isinstance(index, int) and not isinstance(index, bool)]
    scope = str(unit.get("scope_constraint") or "")
    if str(unit.get("disposition") or "") != "action" or not scope:
        return []
    if actions and all(
        isinstance(action, Mapping) and semantic_scope_accepts_action(scope, action)
        for action in actions
    ):
        return list(range(len(actions)))
    return []


def _admitted_action_queue(
    plan: ImmutableSemanticPlan,
    admission: WholePlanAdmission,
    state: AgentGraphState,
) -> dict[str, Any]:
    payload = plan.document()
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    action_rows = {
        str(row.get("action_id") or ""): row
        for row in admission.action_verdicts
    }
    pending_indexes = [
        int(index)
        for index in payload.get("pending_answer_admissions") or []
        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < len(actions)
        )
    ]
    if len(pending_indexes) != len(
        [
            index
            for index in payload.get("pending_answer_admissions") or []
            if isinstance(index, int) and not isinstance(index, bool)
        ]
    ) or len(pending_indexes) != len(set(pending_indexes)):
        raise ValueError("frozen pending admission indexes are invalid")
    chain_indexes: list[int] = []
    target_indexes: list[int] = []
    consultation_indexes: list[int] = []
    navigation_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    action_ids = list(plan.action_ids)
    pending_choice_contracts = [
        dict(row)
        for row in payload.get("pending_choice_contracts") or []
        if isinstance(row, Mapping)
    ]
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValueError("admitted immutable plan contains a non-object action")
        action_type = str(action.get("type") or "")
        row = action_rows.get(action_ids[index])
        if row is None:
            raise ValueError("admitted immutable plan is missing an action verdict")
        if action_type == "choose_chain":
            chain_indexes.append(index)
        if action_type in {"choose_target_mode", "queue_workflow_goal"}:
            target_indexes.append(index)
        spec = ACTION_BY_TYPE.get(action_type)
        if (
            spec is not None
            and spec.incomplete_mutation_intake
            and spec.provides_capabilities
            and all(
                state_has_capability(state, capability)
                for capability in spec.provides_capabilities
            )
        ):
            replacement_rows.append({
                "action_index": index,
                "reviewer_evidence_hash": _content_hash(row),
            })
        if action_type == "answer_opening_question":
            consultation_indexes.append(index)
        if action_type == "change_group":
            source = str(action.get("source_evidence") or "")
            if not source:
                raise ValueError("group navigation has no exact source evidence")
            navigation_rows.append({
                "action_index": index,
                "group": str(action.get("group") or ""),
                "source_evidence": source,
                "destination_quote": source,
                "owner_unit_ids": [str(value) for value in row.get("unit_ids") or []],
            })
        if action_type == "request_config_field_input":
            field_rows.append({
                "action_index": index,
                "config_field": str(action.get("config_field") or ""),
                "source_evidence": str(action.get("source_evidence") or ""),
                "semantic_unit_hash": _semantic_unit_hash_for_action(payload, index),
                "reviewer_evidence_hash": _content_hash(row),
            })
    for contract in pending_choice_contracts:
        index = contract.get("action_index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(actions):
            raise ValueError("canonical pending choice references an invalid action")
        action = actions[index]
        row = action_rows.get(action_ids[index]) or {}
        provenance = contract.get("semantic_units") if isinstance(contract.get("semantic_units"), list) else []
        provenance_ids = [
            str(unit.get("unit_id") or "")
            for unit in provenance
            if isinstance(unit, Mapping)
        ]
        reviewer_ids = [str(value) for value in row.get("unit_ids") or []]
        if provenance_ids != reviewer_ids:
            raise ValueError("canonical pending choice provenance differs from admitted semantic units")
        if str(action.get("type") or "") != "answer_pending":
            raise ValueError("canonical pending choice does not reference answer_pending")
        contract["admission_action_id"] = action_ids[index]
    payload["pending_choice_contracts"] = pending_choice_contracts
    payload.update({
        "admission_action_ids": action_ids,
        "pending_answer_admissions": pending_indexes,
        "chain_selection_admissions": chain_indexes,
        "target_mode_selection_admissions": target_indexes,
        "consultation_admissions": consultation_indexes,
        "group_navigation_admissions": navigation_rows,
        "field_reconfiguration_admissions": field_rows,
        "replacement_intake_admissions": replacement_rows,
    })
    admitted = _attach_semantic_admission_receipts(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        state,
        semantic_consensus={
            "plan_hash": plan.plan_hash,
            "review_hashes": list(admission.review_hashes),
            "request_count": admission.request_count,
            "request_sizes": list(admission.request_sizes),
        }
        if admission.consensus_required
        else None,
    )
    admitted_payload = _parse_json_object(admitted)
    admitted_actions = (
        admitted_payload.get("actions")
        if isinstance(admitted_payload.get("actions"), list)
        else []
    )
    for contract in admitted_payload.get("pending_choice_contracts") or []:
        if not isinstance(contract, dict):
            continue
        index = contract.get("action_index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(admitted_actions):
            raise ValueError("canonical pending choice lost its admitted action")
        action = admitted_actions[index]
        if not isinstance(action, Mapping) or not str(action.get("_admission_action_id") or ""):
            raise ValueError("canonical pending choice has no final admission action id")
        contract["admission_action_id"] = str(action.get("_admission_action_id"))
    return _parse_action_queue(
        json.dumps(admitted_payload, ensure_ascii=False, sort_keys=True),
        trusted_metadata=True,
    )


def _invalid_plan_coverage(
    clauses: tuple[TurnClause, ...],
    errors: tuple[str, ...],
) -> PlanCoverageResult:
    return PlanCoverageResult(
        valid=False,
        errors=tuple(dict.fromkeys(errors)),
        unresolved_clauses=tuple(clause.text for clause in clauses),
    )


def _merge_plan_errors(
    validation: PlanCoverageResult,
    errors: tuple[str, ...],
) -> PlanCoverageResult:
    if not errors:
        return validation
    return PlanCoverageResult(
        valid=False,
        errors=tuple(dict.fromkeys([*validation.errors, *errors])),
        unresolved_clauses=validation.unresolved_clauses,
        rejected_action_indexes=validation.rejected_action_indexes,
        incomplete_unit_ids=validation.incomplete_unit_ids,
    )


def _unresolved_action_queue(
    clauses: tuple[TurnClause, ...],
    errors: tuple[str, ...],
    *,
    semantic_units: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    preserved_units = [
        {
            **dict(unit),
            "disposition": "unresolved",
            "action_indexes": [],
        }
        for unit in semantic_units or []
        if str(unit.get("unit_id") or "")
        and str(unit.get("source_text") or "")
    ]
    unresolved = tuple(
        str(unit["source_text"])
        for unit in preserved_units
    ) or tuple(clause.text for clause in clauses)
    reasons = tuple(dict.fromkeys(str(error) for error in errors if str(error)))
    return {
        "actions": [{
            "type": "clarify_unresolved",
            "clauses": list(unresolved),
            "reason": "; ".join(reasons) or "incomplete clause coverage",
            "confidence": "high",
        }],
        "reason": "resolver returned an incomplete clause plan",
        "coverage_errors": list(reasons),
        "unresolved_clauses": list(unresolved),
        "semantic_units": preserved_units,
    }


def _valid_group_navigation_admissions(
    payload: dict[str, Any],
    actions: list[Any],
) -> dict[int, dict[str, Any]]:
    """Return only receipts that exactly match their admitted navigation."""

    valid: dict[int, dict[str, Any]] = {}
    rows = payload.get("group_navigation_admissions")
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    if not isinstance(rows, list):
        return valid
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("action_index"), int):
            continue
        index = int(row["action_index"])
        if not 0 <= index < len(actions) or not isinstance(actions[index], dict):
            continue
        action = actions[index]
        source = str(action.get("source_evidence") or "")
        group = str(action.get("group") or "")
        quote = str(row.get("destination_quote") or "").strip()
        owner_unit_ids = [
            str(unit_id)
            for unit_id in row.get("owner_unit_ids", [])
            if str(unit_id)
        ] if isinstance(row.get("owner_unit_ids"), list) else []
        mapped_unit_ids = {
            str(unit.get("unit_id") or "")
            for unit in units
            if isinstance(unit, dict)
            and index in (
                unit.get("action_indexes")
                if isinstance(unit.get("action_indexes"), list)
                else []
            )
            and str(unit.get("unit_id") or "")
        }
        if (
            str(action.get("type") or "") != "change_group"
            or not group
            or str(row.get("group") or "") != group
            or str(row.get("source_evidence") or "") != source
            or not quote
            or quote not in source
            or not owner_unit_ids
            or set(owner_unit_ids) != mapped_unit_ids
        ):
            continue
        valid[index] = {
            "group": group,
            "source_evidence": source,
            "destination_quote": quote,
            "owner_unit_ids": owner_unit_ids,
            "specific_change_requested": False,
        }
    return valid


def _semantic_unit_hash_for_action(payload: Mapping[str, Any], index: int) -> str:
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    owned = [
        {
            "unit_id": str(unit.get("unit_id") or ""),
            "clause_id": str(unit.get("clause_id") or ""),
            "source_text": str(unit.get("source_text") or ""),
        }
        for unit in units
        if isinstance(unit, dict)
        and index in (
            unit.get("action_indexes")
            if isinstance(unit.get("action_indexes"), list)
            else []
        )
    ]
    return _content_hash(owned) if owned else ""


def _valid_field_reconfiguration_admissions(
    payload: dict[str, Any],
    actions: list[Any],
) -> dict[int, dict[str, Any]]:
    """Return Harness-minted field-intake receipts that still match the plan."""

    rows = payload.get("field_reconfiguration_admissions")
    if not isinstance(rows, list):
        return {}
    valid: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("action_index"), int):
            continue
        index = int(row["action_index"])
        if not 0 <= index < len(actions) or not isinstance(actions[index], dict):
            continue
        action = actions[index]
        if str(action.get("type") or "") != "request_config_field_input":
            continue
        semantic_hash = _semantic_unit_hash_for_action(payload, index)
        if not semantic_hash:
            continue
        reviewer_hash = str(row.get("reviewer_evidence_hash") or "").strip()
        if (
            not reviewer_hash
            or str(row.get("semantic_unit_hash") or "") != semantic_hash
            or str(row.get("config_field") or "") != str(action.get("config_field") or "")
            or str(row.get("source_evidence") or "") != str(action.get("source_evidence") or "")
        ):
            continue
        valid[index] = dict(row)
    return valid


def _validate_action_document(
    text: str,
    clauses: tuple[TurnClause, ...],
    state: AgentGraphState | None = None,
):
    payload = _parse_json_object(text)
    if not isinstance(payload.get("actions"), list):
        return validate_plan_coverage({}, clauses)
    action_errors: list[str] = []
    rejected_action_indexes: set[int] = set()
    try:
        validate_action_transaction_contract(payload["actions"])
    except ValueError as exc:
        action_errors.append(str(exc))
    pending = dict((state or {}).get("pending_question") or {})
    pending_admissions = {
        int(index)
        for index in payload.get("pending_answer_admissions", [])
        if isinstance(index, int)
    }
    for index, raw in enumerate(payload["actions"]):
        if not isinstance(raw, dict):
            action_errors.append(f"action {index} is not an object")
            continue
        try:
            validate_action_contract(raw)
        except ValueError as exc:
            action_errors.append(f"action {index} contract invalid: {exc}")
            rejected_action_indexes.add(index)
            continue
        if answer_pending_representation_conflict(raw):
            action_errors.append(
                f"action {index} answer_pending has conflicting answer and selected_value representations"
            )
            rejected_action_indexes.add(index)
            continue
        domain_error = _pending_value_domain_error(
            raw,
            pending,
        )
        if domain_error:
            action_errors.append(f"action {index} {domain_error}")
            rejected_action_indexes.add(index)
            continue
        if (
            str(raw.get("type") or "") == "go_back"
            and state is not None
            and not (state.get("group_history") or [])
            and not (state.get("interruption_stack") or [])
        ):
            action_errors.append(
                f"action {index} go_back has no recoverable previous or interrupted group; "
                "use named group navigation when the source supplies a destination"
            )
            rejected_action_indexes.add(index)
            continue
        if (
            pending
            and not pending_admissions
            and str(raw.get("type") or "") == "change_group"
            and str(raw.get("group") or "") == str(pending.get("group") or "")
        ):
            action_errors.append(
                f"action {index} navigates to the group already owned by active pending question "
                f"{pending.get('id')!r} without answering it"
            )
            rejected_action_indexes.add(index)
            continue
        if str(raw.get("type") or "") != "answer_pending" or state is None:
            continue
        if not pending:
            action_errors.append(f"action {index} answer_pending has no active question")
            continue
        if not pending.get("options"):
            continue
        selected = raw.get("selected_value")
        if pending_option_value_exists(selected, pending):
            continue
        answer = str(raw.get("answer") or "").strip()
        if pending.get("manual_input_allowed") is True and value_satisfies_pending_contract(answer, pending):
            continue
        matches, _ = exact_answer(answer, pending)
        if not matches:
            action_errors.append(
                f"action {index} answer_pending does not select an exact declared option "
                f"for pending question {pending.get('id')!r}"
            )
            rejected_action_indexes.add(index)
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    for clause in clauses:
        if clause.input_shape != "structured":
            continue
        candidates = extract_structured_input_candidates(clause.text) or {}
        if not candidates.get("config_values"):
            continue
        config_paths = {
            str(candidate.get("source_path") or "")
            for candidate in candidates.get("field_candidates") or []
            if (
                isinstance(candidate, Mapping)
                and str(candidate.get("candidate_kind") or "") == "config"
                and str(candidate.get("source_path") or "")
            )
        }
        for unit in units:
            if not isinstance(unit, Mapping) or str(unit.get("clause_id") or "") != clause.clause_id:
                continue
            source_path = str(unit.get("source_path") or "")
            if source_path and source_path not in config_paths:
                continue
            for index in unit.get("action_indexes") or []:
                if (
                    isinstance(index, int)
                    and not isinstance(index, bool)
                    and 0 <= index < len(payload["actions"])
                    and isinstance(payload["actions"][index], Mapping)
                    and str(payload["actions"][index].get("type") or "") == "answer_pending"
                ):
                    action_errors.append(
                        f"action {index} bypasses structured configuration review for {clause.clause_id}"
                    )
                    rejected_action_indexes.add(index)
    coverage = validate_plan_coverage(
        payload,
        clauses,
        pending_answer_action_indexes=tuple(
            index
            for index, action in enumerate(payload["actions"])
            if isinstance(action, dict)
            and _action_owns_pending_candidate(action, state or {})
        ),
    )
    if state is not None:
        action_errors.extend(_validate_pending_choice_contracts(payload, state))
    if not action_errors:
        return _validate_action_source_grounding(text, coverage)
    return PlanCoverageResult(
        valid=False,
        errors=tuple([*action_errors, *coverage.errors]),
        unresolved_clauses=coverage.unresolved_clauses,
        rejected_action_indexes=tuple(sorted({
            *coverage.rejected_action_indexes,
            *rejected_action_indexes,
        })),
        incomplete_unit_ids=coverage.incomplete_unit_ids,
    )


def _pending_value_domain_error(
    action: Mapping[str, Any],
    pending: Mapping[str, Any],
) -> str:
    """Reject a value whose exact identity belongs to another workflow group."""

    if not pending:
        return ""
    if _action_answers_bound_semantic_draft(action, pending):
        return ""
    value = _raw_pending_candidate(action, pending)
    if not isinstance(value, str) or not value.strip():
        return ""
    pending_group = str(pending.get("group") or "")
    conflicts = semantic_value_domain_conflicts(
        value,
        owning_group=pending_group,
        pending_question=pending,
    )
    if not conflicts:
        return ""
    owners = ", ".join(sorted({
        f"{record.get('semantic_owner')}:{record.get('action_type')}.{record.get('argument')}"
        for record in conflicts
    }))
    return (
        "uses a registered semantic value as an unrelated pending answer "
        f"({owners})"
    )


def _action_answers_bound_semantic_draft(
    action: Mapping[str, Any],
    pending: Mapping[str, Any],
) -> bool:
    """Honor the exact draft atom contract before ordinary value ownership."""

    binding = dict(pending.get("semantic_draft_binding") or {})
    if (
        not binding
        or str(action.get("type") or "") != "resolve_semantic_draft_atom"
    ):
        return False
    identity_matches = (
        str(action.get("draft_id") or "") == str(binding.get("draft_id") or "")
        and int(action.get("revision") or 0) == int(binding.get("revision") or 0)
        and str(action.get("atom_id") or "") == str(binding.get("atom_id") or "")
    )
    return (
        identity_matches
        and isinstance(action.get("resolution"), str)
        and bool(str(action.get("resolution") or "").strip())
    )


def _raw_pending_candidate(
    action: Mapping[str, Any],
    pending: Mapping[str, Any],
) -> Any:
    if str(action.get("type") or "") == "answer_pending":
        supplied = []
        for key in ("answer", "selected_value"):
            value = action.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            supplied.append(value)
        return supplied[0] if len(supplied) == 1 else None
    declared = pending.get("manual_action")
    if (
        not isinstance(declared, Mapping)
        or str(action.get("type") or "") != str(declared.get("type") or "")
    ):
        return None
    value_argument = str(declared.get("value_argument") or "").strip()
    if not value_argument:
        return None
    return action.get(value_argument)


def _semantic_action_purpose(
    action: dict[str, Any],
    fallback: str,
    state: AgentGraphState | None = None,
) -> str:
    """Describe the selected typed operation, not its implementation primitive."""

    action_type = str(action.get("type") or "")
    option = _matching_pending_option(action, state) if state else {}
    if option:
        declared = dict(option.get("action") or {})
        declared_spec = ACTION_BY_TYPE.get(str(declared.get("type") or ""))
        effect = declared_spec.purpose if declared_spec is not None else "apply the displayed option's declared effect"
        labels = declared_option_label_variants(option)
        expected_patch = (
            dict(option.get("expected_patch") or {})
            if isinstance(option.get("expected_patch"), Mapping)
            else {}
        )
        return (
            f"Select only the displayed pending option with id "
            f"{str(option.get('id') or '')!r}, registered labels {list(labels)!r}, "
            f"and value {option.get('value')!r}; its expected state effect is "
            f"{expected_patch!r}, and its declared dispatch effect is: {effect}"
        )
    if action_type == "answer_pending" and state:
        return "Supply a source-grounded answer that actually satisfies the active typed pending contract."
    action_spec = ACTION_BY_TYPE.get(action_type)
    if (
        action_spec is not None
        and action_spec.incomplete_mutation_intake
        and action_spec.provides_capabilities
        and state
        and all(
            state_has_capability(state, capability)
            for capability in action_spec.provides_capabilities
        )
    ):
        return (
            f"Reopen {action_spec.target_group!r} typed intake only because the "
            "source explicitly requests replacing the already confirmed value "
            "but supplies no concrete replacement."
        )
    if action_type == "change_group":
        target_group = str(action.get("group") or "").strip()
        entry_purposes = [
            spec.entry_intake_purpose
            for spec in ACTION_SPECS
            if spec.entry_intake
            and spec.target_group == target_group
            and spec.entry_intake_purpose.strip()
        ]
        exclusion = (
            " Registered typed entry purposes excluded from this generic navigation are: "
            + "; ".join(entry_purposes)
            if entry_purposes
            else ""
        )
        return (
            f"Temporarily suspend the active group and navigate to workflow group {target_group!r} "
            "for a general request to visit or revisit that area which does not match a registered "
            "typed entry purpose; after the destination work completes, resume the interrupted group."
            f"{exclusion}"
        )
    if action_type == "go_back":
        history = [str(value) for value in (state or {}).get("group_history") or [] if str(value)]
        interruptions = [
            str(row.get("group") or "")
            for row in (state or {}).get("interruption_stack") or []
            if isinstance(row, Mapping) and str(row.get("group") or "")
        ]
        recoverable = [*history, *interruptions]
        availability = (
            f"The current recoverable group sequence is {recoverable!r}."
            if recoverable
            else "The current state has no recoverable previous or interrupted group, so this operation is unavailable."
        )
        return (
            "Return only to the most recent recoverable previous/interrupted workflow group. "
            "This operation does not select an explicitly named destination when no matching recoverable group exists; "
            "that request belongs to named group navigation. "
            + availability
        )
    if action_type == "answer_opening_question":
        topic = canonical_consultation_topic(action.get("topic"))
        subject = str(action.get("subject") or "").strip()
        subject_scope = f" for the exact subject {subject!r}" if subject else ""
        topic_purpose = CONSULTATION_TOPIC_PURPOSES.get(topic, "Answer the selected read-only topic.")
        return (
            f"Answer only the user's independent read-only consultation topic {topic!r}"
            f"{subject_scope}: {topic_purpose} This purpose does not answer a different consultation topic."
        )
    if action_type == "choose_chain":
        candidates = [
            str(item).strip()
            for item in action.get("chain_candidates") or []
            if str(item).strip()
        ]
        if len(candidates) > 1:
            return "Ask the user to resolve the exact finite chain candidate set supplied in this source unit."
        return "Select the exact source-supplied named chain identity as the current benchmark target."
    if str(action.get("type") or "") == "rpc_catalog_command":
        command = str(action.get("catalog_command") or "")
        return {
            "enter": "Start a custom RPC catalog workflow explicitly requested by the user.",
            "set_endpoint": "Select the source-supplied endpoint as the RPC schema validation endpoint.",
            "set_method": "Select the exact source-supplied callable wire RPC method for validation.",
            "append_evidence": (
                "Ingest new source-attributable RPC method-schema, parameter, request, "
                "response, or documentation facts into the active catalog draft."
            ),
            "keep_current_method": (
                "Resolve the active method conflict by preserving the current draft."
            ),
            "replace_current_method": (
                "Resolve the active method conflict by replacing the draft with the "
                "already presented incoming method."
            ),
        }.get(command, fallback)
    return fallback


def _matching_pending_option(
    action: dict[str, Any],
    state: AgentGraphState | None,
) -> dict[str, Any]:
    """Return the registered pending option whose declared effect matches an action."""

    if not state:
        return {}
    pending = dict(state.get("pending_question") or {})
    options = [item for item in pending.get("options") or [] if isinstance(item, dict)]
    action_type = str(action.get("type") or "")
    if action_type == "answer_pending":
        return _declared_option_for_pending_answer(action, pending)
    for option in options:
        declared = option.get("action")
        if not isinstance(declared, dict) or str(declared.get("type") or "") != action_type:
            continue
        if all(
            key == "type" or action.get(key) == value
            for key, value in declared.items()
        ):
            return option
    candidate_spec = ACTION_BY_TYPE.get(action_type)
    explicit_flags = [
        key for key in action if str(key).endswith("_explicit")
    ]
    if candidate_spec is None or not explicit_flags or any(
        action.get(key) is not False for key in explicit_flags
    ):
        return {}
    for option in options:
        declared = option.get("action")
        if not isinstance(declared, dict):
            continue
        declared_spec = ACTION_BY_TYPE.get(str(declared.get("type") or ""))
        if (
            declared_spec is not None
            and declared_spec.incomplete_mutation_intake
            and declared_spec.target_group
            and declared_spec.target_group == candidate_spec.target_group
        ):
            return option
    return {}


def _matching_pending_manual_value(
    action: Mapping[str, Any],
    pending: Mapping[str, Any],
) -> Any:
    """Return a value supplied through the question's declared manual owner."""

    declared = pending.get("manual_action")
    if not isinstance(declared, Mapping):
        return None
    if str(action.get("type") or "") != str(declared.get("type") or ""):
        return None
    value_argument = str(declared.get("value_argument") or "").strip()
    if not value_argument or value_argument not in action:
        return None
    spec = ACTION_BY_TYPE.get(str(declared.get("type") or ""))
    allowed_arguments = set(spec.allowed_arguments if spec is not None else ())
    for key, value in declared.items():
        if key in {"type", "value_argument"}:
            continue
        if key not in allowed_arguments:
            continue
        if action.get(key) != value:
            return None
    value = action.get(value_argument)
    return value if value_satisfies_pending_contract(value, dict(pending)) else None


def _manual_value_from_answer_pending(
    action: Mapping[str, Any],
    pending: Mapping[str, Any],
) -> Any:
    """Return one unambiguous manual value from a pending-answer action.

    The current compiler contract uses ``answer`` for manual input and
    ``selected_value`` for a declared option. This reader tolerates one supplied
    representation so checkpoint migration and canonicalization cannot choose
    between contradictory facts.
    """

    if str(action.get("type") or "") != "answer_pending":
        return None
    supplied: list[Any] = []
    identities: set[str] = set()
    for key in ("answer", "selected_value"):
        if key not in action:
            continue
        value = action.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        normalized = value.strip() if isinstance(value, str) else value
        identity = json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
        if identity in identities:
            continue
        identities.add(identity)
        supplied.append(normalized)
    if len(supplied) != 1:
        return None
    value = supplied[0]
    return value if value_satisfies_pending_contract(value, dict(pending)) else None


def _reject_conflicting_pending_representations(text: str) -> None:
    """Reject raw model contradictions before any canonicalization can erase them."""

    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    conflicting = [
        index
        for index, action in enumerate(actions)
        if isinstance(action, Mapping)
        and answer_pending_representation_conflict(action)
    ]
    if conflicting:
        joined = ", ".join(str(index) for index in conflicting)
        raise ValueError(
            "answer_pending has conflicting answer and selected_value "
            f"representations at action indexes: {joined}"
        )


def _declared_option_for_pending_answer(
    action: dict[str, Any],
    pending: dict[str, Any],
) -> dict[str, Any]:
    """Resolve one answer action against the exact typed option contract.

    New compiler output uses ``selected_value`` for a declared option. Reading
    ``answer`` remains a migration tolerance for older checkpoints, while the
    registry rejects contradictory dual representations before admission.
    """

    if str(action.get("type") or "") != "answer_pending":
        return {}
    selected = (
        action.get("selected_value")
        if "selected_value" in action and action.get("selected_value") is not None
        else action.get("answer")
    )
    option = next(
        (
            item
            for item in pending.get("options") or []
            if isinstance(item, dict) and item.get("value") == selected
        ),
        {},
    )
    if (
        not option
        and ("selected_value" not in action or action.get("selected_value") is None)
        and action.get("answer") not in (None, "")
    ):
        answer_selects_option, answer_value = exact_answer(str(action.get("answer")), pending)
        if answer_selects_option:
            option = next(
                (
                    item
                    for item in pending.get("options") or []
                    if isinstance(item, dict) and item.get("value") == answer_value
                ),
                {},
            )
    if not option or "selected_value" not in action or action.get("selected_value") is None:
        return option
    answer = action.get("answer")
    if answer in (None, "") or answer == selected or str(answer).strip() == str(selected).strip():
        return option
    answer_selects_option, answer_value = exact_answer(str(answer), pending)
    if answer_selects_option and answer_value == selected:
        return option
    if (
        pending.get("manual_input_allowed") is True
        and value_satisfies_pending_contract(answer, pending)
        and _manual_answer_has_literal_source(
            action,
            str(action.get("source_evidence") or ""),
        )
    ):
        return {}
    return option


def _semantic_operation_arguments(action: dict[str, Any]) -> dict[str, Any]:
    """Expose source facts without delegating registry authority to the LLM."""

    internal_fields = {
        "type",
        "action_id",
        "confidence",
        "pending_option_semantic_verified",
        "chain_selection_semantic_verified",
        "target_mode_semantic_verified",
        "semantic_purpose_verified",
        "_replacement_intake_receipt",
        "group_navigation_semantic_verified",
    }
    return {
        key: value
        for key, value in action.items()
        if key not in internal_fields
    }


def _source_units_for_action(payload: Mapping[str, Any], action_index: int) -> list[dict[str, Any]]:
    return [
        dict(unit)
        for unit in payload.get("semantic_units") or []
        if isinstance(unit, dict)
        and action_index in (
            unit.get("action_indexes")
            if isinstance(unit.get("action_indexes"), list)
            else []
        )
    ]


def _canonicalize_proposal_values(action: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for raw_field, raw_value in dict(action.get("config_values") or {}).items():
        field, value = normalize_proposed_config_value(str(raw_field or ""), raw_value)
        if not field or value in {"", None}:
            raise ValueError("configuration proposal contains an empty canonical field or value")
        if field in canonical and canonical[field] != value:
            raise ValueError("configuration proposal contains conflicting canonical values")
        canonical[field] = value
    if not canonical:
        raise ValueError("configuration proposal has no canonical values")
    normalized = dict(action)
    normalized["config_values"] = canonical
    return normalized


def _proposal_field_source(
    action: Mapping[str, Any],
    field: str,
    value: Any,
    units: list[dict[str, Any]],
) -> tuple[str, str, str] | None:
    """Return one exact current-turn source unit for a canonical field/value."""

    action_source = str(action.get("source_evidence") or "").strip()
    for unit in units:
        unit_id = str(unit.get("unit_id") or "").strip()
        unit_text = str(unit.get("source_text") or "")
        if not unit_id or not unit_text:
            continue
        candidates = extract_structured_input_candidates(unit_text) or {}
        for raw_field, raw_value in dict(candidates.get("config_values") or {}).items():
            candidate_field, candidate_value = normalize_proposed_config_value(raw_field, raw_value)
            if candidate_field == field and candidate_value == value:
                return unit_id, unit_text, unit_text
        quote = action_source if action_source and action_source in unit_text else unit_text
        value_text = str(value).strip()
        if value_text and value_text.casefold() in quote.casefold():
            return unit_id, unit_text, quote
    return None


def _attach_semantic_admission_receipts(
    text: str,
    state: AgentGraphState | None = None,
    *,
    semantic_consensus: Mapping[str, Any] | None = None,
) -> str:
    """Mark actions whose purpose passed the final semantic admission review.

    This helper is called only after structural and semantic validation have
    both succeeded. The marker is trusted runtime metadata, so model-produced
    documents cannot supply it through the normal action contract.
    """

    payload = _parse_json_object(text)
    payload.pop("admission_action_ids", None)
    payload.pop("admission_rejections", None)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    pending_admissions = {
        int(index)
        for index in payload.pop("pending_answer_admissions", [])
        if isinstance(index, int) and 0 <= index < len(actions)
    }
    chain_admissions = {
        int(index)
        for index in payload.pop("chain_selection_admissions", [])
        if isinstance(index, int) and 0 <= index < len(actions)
    }
    target_mode_admissions = {
        int(index)
        for index in payload.pop("target_mode_selection_admissions", [])
        if isinstance(index, int) and 0 <= index < len(actions)
    }
    navigation_admissions = set(
        _valid_group_navigation_admissions(payload, actions)
    )
    field_admissions = _valid_field_reconfiguration_admissions(payload, actions)
    replacement_admissions = {
        int(row["action_index"]): str(
            row.get("reviewer_evidence_hash") or ""
        )
        for row in payload.pop("replacement_intake_admissions", [])
        if isinstance(row, Mapping)
        and isinstance(row.get("action_index"), int)
        and not isinstance(row.get("action_index"), bool)
        and 0 <= int(row["action_index"]) < len(actions)
        and str(row.get("reviewer_evidence_hash") or "")
    }
    payload.pop("consultation_admissions", None)
    payload.pop("group_navigation_admissions", None)
    payload.pop("field_reconfiguration_admissions", None)
    admission_action_ids = _ensure_admission_action_ids(payload)
    state = state or {"thread_id": "admission-test", "session": {"id": "admission-test"}, "turn_index": 0}
    thread_id = str(state.get("thread_id") or "default")
    session_id = str((state.get("session") or {}).get("id") or thread_id)
    submitted_turn_index = int(state.get("turn_index") or 0)
    semantic_units = [
        unit for unit in payload.get("semantic_units") or []
        if isinstance(unit, dict)
    ]
    transaction_hash = build_admission_transaction_hash(
        thread_id=thread_id,
        session_id=session_id,
        submitted_turn_index=submitted_turn_index,
        actions=[action for action in actions if isinstance(action, dict)],
        semantic_units=semantic_units,
        admission_action_ids=admission_action_ids,
    )
    consensus_receipt = (
        build_semantic_consensus_receipt(
            thread_id=thread_id,
            session_id=session_id,
            submitted_turn_index=submitted_turn_index,
            transaction_hash=transaction_hash,
            plan_hash=str(semantic_consensus.get("plan_hash") or ""),
            admission_action_ids=admission_action_ids,
            review_hashes=semantic_consensus.get("review_hashes") or (),
            request_count=int(semantic_consensus.get("request_count") or 0),
            request_sizes=semantic_consensus.get("request_sizes") or (),
        )
        if semantic_consensus is not None
        else None
    )
    for index, action in enumerate(actions):
        if isinstance(action, dict):
            action["_admission_action_id"] = admission_action_ids[index]
            action["_transaction_action_ids"] = list(admission_action_ids)
            action["_plan_transaction_hash"] = transaction_hash
            if consensus_receipt is not None:
                action["_semantic_consensus_receipt"] = consensus_receipt
        if isinstance(action, dict) and _requires_semantic_fulfillment_review(action):
            action["semantic_purpose_verified"] = True
            spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
            if spec is not None and index in replacement_admissions:
                action["_replacement_intake_receipt"] = (
                    build_replacement_intake_admission_receipt(
                        thread_id=thread_id,
                        session_id=session_id,
                        submitted_turn_index=submitted_turn_index,
                        transaction_hash=transaction_hash,
                        admission_action_id=admission_action_ids[index],
                        action_type=str(action.get("type") or ""),
                        source_evidence=str(
                            action.get("source_evidence") or ""
                        ),
                        target_group=str(spec.target_group or ""),
                        provides_capabilities=spec.provides_capabilities,
                        reviewer_evidence_hash=replacement_admissions[index],
                    )
                )
        if isinstance(action, dict) and index in pending_admissions:
            action["pending_option_semantic_verified"] = True
        if isinstance(action, dict) and index in chain_admissions:
            action["chain_selection_semantic_verified"] = True
        if isinstance(action, dict) and index in target_mode_admissions:
            action["target_mode_semantic_verified"] = True
        if isinstance(action, dict) and index in navigation_admissions:
            action["group_navigation_semantic_verified"] = True
        if isinstance(action, dict) and index in field_admissions:
            units = _source_units_for_action(payload, index)
            source = str(action.get("source_evidence") or "")
            source_unit = next(
                (
                    unit for unit in units
                    if source and source in str(unit.get("source_text") or "")
                ),
                None,
            )
            if source_unit is None:
                raise ValueError("field intake admission has no exact source unit")
            admission = field_admissions[index]
            action["semantic_purpose_verified"] = True
            action["_semantic_admission_receipt"] = build_field_intake_admission_receipt(
                thread_id=thread_id,
                session_id=session_id,
                submitted_turn_index=submitted_turn_index,
                transaction_hash=transaction_hash,
                admission_action_id=admission_action_ids[index],
                config_field=str(action.get("config_field") or ""),
                source_evidence=source,
                source_unit_id=str(source_unit.get("unit_id") or ""),
                source_unit_text=str(source_unit.get("source_text") or ""),
                source_quote=source,
                semantic_unit_hash=str(admission.get("semantic_unit_hash") or ""),
                reviewer_evidence_hash=str(admission.get("reviewer_evidence_hash") or ""),
            )
        if isinstance(action, dict) and str(action.get("type") or "") == "propose_config_values":
            if action.get("conflicts"):
                raise ValueError("conflicting configuration proposal cannot receive admission receipts")
            canonical = _canonicalize_proposal_values(action)
            action.clear()
            action.update(canonical)
            units = _source_units_for_action(payload, index)
            receipts: dict[str, Any] = {}
            for field, value in dict(action.get("config_values") or {}).items():
                source = _proposal_field_source(action, field, value, units)
                if source is None:
                    raise ValueError(f"configuration proposal lacks current-turn provenance: {field}")
                unit_id, unit_text, quote = source
                receipts[field] = build_proposal_field_receipt(
                    thread_id=thread_id,
                    session_id=session_id,
                    submitted_turn_index=submitted_turn_index,
                    transaction_hash=transaction_hash,
                    admission_action_id=admission_action_ids[index],
                    config_field=field,
                    canonical_value=value,
                    source_unit_id=unit_id,
                    source_unit_text=unit_text,
                    source_quote=quote,
                )
            action["_proposal_transaction_hashes"] = [transaction_hash]
            action["_proposal_field_receipts"] = receipts
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _manual_answer_has_literal_source(action: dict[str, Any], user_text: str) -> bool:
    """Prove that a normalized manual answer occurs in exact source evidence."""

    return bool(_manual_answer_literal_source(action, user_text))


def _manual_answer_literal_source(action: dict[str, Any], user_text: str) -> str:
    """Return the exact source slice for one already-normalized manual value."""

    answer = str(action.get("answer") or "").strip()
    quote = str(action.get("source_evidence") or "").strip()
    if not answer or not quote or quote not in str(user_text or ""):
        return ""
    match = re.search(
        rf"(?<!\w){re.escape(answer)}(?!\w)",
        quote,
        flags=re.IGNORECASE,
    )
    return match.group(0) if match else ""


def _requires_semantic_fulfillment_review(action: dict[str, Any]) -> bool:
    action_type = str(action.get("type") or "")
    spec = ACTION_BY_TYPE.get(action_type)
    if action_type in {"answer_opening_question", "analyze_evidence", "analyze_report"}:
        return True
    if action_type == "rpc_catalog_command":
        return True
    if action_type == "secondary_handoff_command":
        return str(action.get("handoff_command") or "") == "append_evidence"
    if action_type in {
        "answer_pending",
        "reset_session",
        "approve_preflight_smoke",
        "approve_final_benchmark",
    }:
        return True
    return bool(spec is not None and spec.lifetime == "durable")


def _semantic_fulfillment_prompt(*, review_kind: str = "both") -> str:
    if review_kind == "actions":
        output_contract = (
            "Audit only the supplied action-purpose rows. Return one JSON object only: "
            "{reviews:[{action_index:integer,supported:boolean,source_unit_reviews:[{source_text:string,"
            "role:'direct'|'support',support_relation:string}],reason:string}]}. "
            "Return no other keys and review every supplied action_index exactly once. "
        )
    elif review_kind == "units":
        output_contract = (
            "Audit only the supplied compound-unit rows. Return one JSON object only: "
            "{unit_reviews:[{unit_id:string,complete:boolean,missing_demand_quote:string,reason:string}]}. "
            "Return no other keys and review every supplied unit_id exactly once. "
        )
    elif review_kind == "contexts":
        output_contract = (
            "Audit only the supplied proposed context rows. Return one JSON object only: "
            "{context_reviews:[{unit_id:string,context_only:boolean,reason:string}]}. "
            "Return no other keys and review every supplied unit_id exactly once. "
        )
    else:
        output_contract = (
            "Audit high-risk AnyChain action-purpose mappings and compound-unit completeness. Return one JSON object only: "
            "{reviews:[{action_index:integer,supported:boolean,reason:string}],unit_reviews:[{unit_id:string,complete:boolean,missing_demand_quote:string,reason:string}]}. "
            "Review every supplied operation row and unit row exactly once. "
        )
    return (
        output_contract
        + "For context_reviews, context_only is true only when source_text is prose with no independent present demand and either (a) requests only an inevitable completion_effect explicitly declared by the supplied pending_question for the same pending answer, or (b) is explanatory context, an operation restatement, provenance, format scope, temporal scope, evidence-completeness scope, or a non-mutation constraint for exactly one related_operation, and that operation declares the matching allowed_support_relation. operation_restatement repeats the same requested effect or explains why that same effect is wanted, without adding another requested effect; it requires a direct_source_unit for the operation and cannot authorize an operation by itself. evidence_completeness means only that the source states which request, response, parameter, or documentation evidence is presently available or absent for the same submitted RPC operation; it cannot supply a second method, contradict the submitted operation, or waive required validation. The related operation must have a direct_source_unit that states the actual operation request. Such declared support adds no independent operation. A present answer, selection, correction, contradiction, concrete value, mutation, different navigation destination, evidence submission, or execution instruction is independent and therefore false. A statement that answers the supplied pending_question is not context. input_shape must be prose. planner_reason is untrusted and cannot establish the verdict. Missing or ambiguous intent is false. "
        + f"{FRAMED_REQUEST_SEMANTIC_POLICY}"
        "Operations are opaque, already-registered Harness operations. Internal operation names are intentionally absent because registration, lifecycle, ordering, and choose-versus-change selection are deterministic Harness responsibilities. Never infer or discuss an internal operation name and never reject a purpose on registry or lifecycle grounds. Decide only whether the exact source_units "
        "semantically and explicitly support the declared purpose and its supplied arguments. Workflow state "
        "and a pending question are context, never user evidence. For unit_reviews, ignore pending_question entirely: "
        "an explicit mutation or navigation may interrupt the old question, and the coordinator alone decides interruption, invalidation, and resume behavior. "
        "For a closed-enum argument, rejecting or excluding one value while multiple legal values remain does not select any one remaining value. Reject a purpose that supplies a specific remaining value from that evidence; a registered typed intake must collect the unresolved selection. "
        "For an action-purpose review with several source_units, return every exact source_units string once in source_unit_reviews. Mark at least one as direct. Mark another as support only when its exact support_relation appears in the supplied allowed_support_relations; otherwise the action is unsupported. Direct rows use an empty support_relation. A support unit may not hide an independent selection, mutation, consultation, contradiction, concrete value owned by another operation, or evidence demand. For a single source unit, source_unit_reviews may be empty. In particular, structured partial configuration directly supports a configuration-proposal purpose, and a same-turn instruction to ask for remaining required values after review is processing scope only when that relation is declared by the operation contract. "
        "Pending-question context is present only when a pending-answer purpose itself is being reviewed. A question asking to "
        "summarize, explain, compare, or report retained state is read-only and cannot support "
        "a durable mutation. An evidence-ingestion purpose is supported only when the "
        "source itself contributes an attributable RPC method-schema, request, parameter, "
        "response, or documentation fact; a request to discuss existing evidence is not new "
        "evidence. An explicit statement that a named/current RPC method has no parameters, "
        "describes one or more parameter meanings/types, or describes its response/result is new schema evidence. "
        "For unit_reviews, source_text is the exact unit under review. related_source_units contains only other units in the same turn that share at least one mapped operation; it is relationship context, not permission to ignore an independent demand. A source_text that is purely introductory, trailing, format, provenance, or temporal framing for those related units is complete when the shared mapped operations preserve the related request. If source_text contains its own selection, mutation, consultation, contradiction, correction, value, or evidence demand, that demand must still be represented by mapped_actions. Each mapped_actions row contains an opaque operation index, source-grounded arguments, and its authoritative declared_purpose. Judge whether the set of declared purposes preserves every explicit demand; do not require fields that a declared purpose intentionally collects in later typed questions. Review the proposed operations as one ordered transaction: an earlier purpose that selects a source-grounded wire method may establish the draft referenced by a later evidence-ingestion purpose, but pre-existing workflow state alone cannot. "
        "A compact registered-domain request may combine exactly one source-grounded registered value with adjacent operation framing that only names the workflow the selection is intended to enter. The declared value-selection purpose completely preserves that unit when the framing supplies no second value, question, consultation, navigation, analysis, contradiction, evidence contribution, or explicit immediate-execution authorization. Generic framing around a selection does not create a separate execution demand merely because it names the eventual workflow activity; later typed questions and approval still govern execution. "
        "Selecting or navigating to a group does not preserve an explicit request to alter one registered scalar field in that group. When no replacement value is supplied, completeness requires the registered typed field-intake purpose; when a source-grounded value is supplied, completeness requires the owning proposal or mutation purpose. "
        "A registered operation performs the effect stated by its authoritative declared_purpose. Never require operation_arguments to repeat, simulate, or prove that effect; arguments carry only source-selected values and evidence required by that operation. In particular, a lifecycle command may legitimately have only source evidence as its argument while its declared purpose defines the state transition. "
        "When an active evidence collection exists, an explicit source demand to pause or suspend that collection is independent from navigation or configuration and requires a purpose that pauses while preserving it. Navigation alone is incomplete for that demand. An explicit request to resume a paused collection likewise requires a resume purpose. "
        "A custom-RPC-entry purpose is supported when the source explicitly asks to start, add, supply, or configure a custom RPC method workflow. It intentionally carries no endpoint, method identity, or schema payload; requiring those facts at entry would skip later typed collection questions. A discussion-only question about whether custom RPC is possible does not support entry. "
        "A read-only consultation purpose is supported only when the source asks for an answer, explanation, comparison, status, preparation guidance, or similar information. It is not supported when the source explicitly requests only a selection, mutation, navigation, execution, or evidence-ingestion operation. A declarative reason attached to a pending-option selection does not become a consultation merely because it explains that selection; when it has no independent question or requested effect, it is support for the pending-answer purpose and a proposed consultation over that reason is unsupported. Distinct consultation subjects remain independent demands: a purpose that reports workflow configuration or pending context does not report whether a current or historical benchmark job exists, and a job-status purpose does not report retained workflow configuration. A compound unit asking for both is complete only when mapped purposes explicitly cover both subjects. "
        "A request to identify the cause of supplied error evidence and explain how to correct it belongs to the same evidence-analysis demand. It does not independently support a generic opening consultation unless it asks for a distinct product fact beyond that diagnosis or recovery guidance. A plan may render only one primary answer for one demand; a generic consultation that overlaps an admitted evidence diagnosis must be rejected rather than appended as a contradictory fallback. "
        + GROUP_NAVIGATION_SEMANTIC_POLICY
        + "When an action record has registry_incomplete_mutation_intake=true, its authoritative declared_purpose intentionally opens a later typed intake. Admit it when the source explicitly requests the initial selection or replacement described by that purpose but supplies no concrete value; do not require the value that the later question exists to collect. When such an action is the declared owner of an active pending option, selecting that option is sufficient source support for the intake purpose. "
        "When an action record has registry_incomplete_read_intake=true, its authoritative declared_purpose intentionally opens a later typed read-only collection because the source requests that registered analysis but has not supplied its payload yet. Admit that intake when the source explicitly requests the declared read operation; do not require the evidence, document, log, or other payload that the typed collection exists to collect. This rule does not authorize a different read subject, mutation, navigation, or execution request. "
        "A chain-candidate-intake purpose is supported when the source presents one or more tentative benchmark-chain candidates without committing to one. It intentionally preserves candidates for a later typed confirmation question and is not a chain mutation. "
        "A QPS-customization purpose requires an explicit request to alter, tune, override, or avoid defaults of one or more QPS profile values, even when concrete numbers arrive later. Merely visiting the QPS area without requesting a profile-value change is navigation. "
        "A wire-method-selection purpose requires an actual callable wire method, not merely a schema field named method or method_id. An endpoint-selection purpose requires an explicitly selected validation endpoint, not an example or documentation URL. A secondary-development evidence purpose likewise requires new protocol, endpoint, request, response, or official-document evidence. A pending-answer purpose must actually answer the supplied pending contract. Reset and execution "
        "approval require explicit authorization. A declared pending option plus adjacent prose that only explains why that same option was selected is one supported pending-answer purpose: the option selection is direct evidence and the reason may be explanatory_context or operation_restatement support. The reason is not an omitted demand unless it independently asks, changes, contradicts, navigates, or selects something else. "
        "For each unit_review, complete is true only when mapped operations collectively preserve every explicit selection, mutation, consultation, evidence request, and navigation demand in source_text; one mapped purpose may be valid while the set is still incomplete. When complete is false, missing_demand_quote must be the shortest non-empty exact substring of source_text that states one omitted independently actionable demand; otherwise it must be empty. A committed named benchmark chain requires a declared purpose that selects that exact chain. A tentative chain candidate may instead be completely preserved by a declared intake purpose that asks the user to resolve candidates, while a chain mentioned only in a support question needs no chain mutation or intake purpose. When a schema-evidence pending question identifies an existing catalog draft, deictic source text such as 'this method has no parameters' or 'it returns a hex value' contributes parameter/response evidence to that current draft and may support an evidence-ingestion purpose; it still cannot support a wire-method-selection purpose unless the source itself names the wire method. Do not repair, route, invent, rename, or classify an operation."
    )


def _prepare_untrusted_action_document(text: str) -> str:
    """Remove model-forged control receipts and assign Harness identities."""

    payload = _parse_json_object(text)
    for key in _ADMISSION_RECEIPT_KEYS:
        payload.pop(key, None)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    cleaned: list[Any] = []
    for raw in actions:
        if not isinstance(raw, dict):
            cleaned.append(raw)
            continue
        action = dict(raw)
        action.pop("action_id", None)
        for key in tuple(action):
            if key in TRUSTED_ACTION_METADATA_FIELDS or str(key).startswith("_"):
                action.pop(key, None)
        arguments = action.get("arguments")
        if isinstance(arguments, dict):
            action["arguments"] = {
                key: value
                for key, value in arguments.items()
                if key != "action_id"
                and key not in TRUSTED_ACTION_METADATA_FIELDS
                and not str(key).startswith("_")
            }
        cleaned.append(normalize_current_action_envelope(action))
    if isinstance(payload.get("actions"), list):
        payload["actions"] = cleaned
    units = payload.get("semantic_units")
    if isinstance(units, list):
        payload["semantic_units"] = [
            _canonicalize_semantic_scope_reference(unit)
            if isinstance(unit, dict)
            else unit
            for unit in units
        ]
    _ensure_admission_action_ids(payload)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _canonicalize_semantic_scope_reference(unit: dict[str, Any]) -> dict[str, Any]:
    """Resolve an exact exported scope-schema row to its registry identity."""

    normalized = dict(unit)
    scope = normalized.get("scope_constraint")
    if not isinstance(scope, dict):
        return normalized
    matches = [
        str(row["name"])
        for row in semantic_scope_schema()
        if scope == row
    ]
    if len(matches) == 1:
        normalized["scope_constraint"] = matches[0]
    return normalized


def _ensure_admission_action_ids(payload: dict[str, Any]) -> list[str]:
    """Return stable Harness-only ids aligned with the current action list."""

    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    existing = payload.get("admission_action_ids")
    ids = [str(value) for value in existing] if isinstance(existing, list) else []
    reserved = {
        str(row.get("admission_action_id") or "")
        for row in payload.get("admission_rejections", [])
        if isinstance(row, dict) and str(row.get("admission_action_id") or "")
    }
    if len(ids) > len(actions) or any(not value for value in ids) or len(ids) != len(set(ids)):
        ids = []
    for index in range(len(ids), len(actions)):
        canonical = json.dumps(actions[index], ensure_ascii=False, sort_keys=True, default=str)
        seed = f"{index}:{canonical}"
        candidate = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
        suffix = 0
        while candidate in ids or candidate in reserved:
            suffix += 1
            candidate = hashlib.sha256(f"{seed}:{suffix}".encode("utf-8")).hexdigest()[:20]
        ids.append(candidate)
    payload["admission_action_ids"] = ids
    return ids


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"type": "unknown", "reason": "model did not return valid JSON", "confidence": "low"}
    return payload if isinstance(payload, dict) else {"type": "unknown", "reason": "model returned non-object JSON", "confidence": "low"}


def _parse_action_queue(
    text: str,
    *,
    trusted_metadata: bool = False,
) -> dict[str, Any]:
    payload = _parse_json_object(text)
    actions_raw = payload.get("actions")
    if isinstance(actions_raw, dict):
        actions_raw = [actions_raw]
    if not isinstance(actions_raw, list):
        actions_raw = [dict(payload)]
    actions: list[dict[str, Any]] = []
    for raw in actions_raw:
        if not isinstance(raw, dict):
            continue
        try:
            action = validate_action_contract(raw, trusted_metadata=trusted_metadata)
        except ValueError as exc:
            actions.append({"type": "unknown", "reason": str(exc), "confidence": "low"})
            continue
        action_type = str(action["type"])
        action.setdefault("confidence", "low" if action_type == "unknown" else "medium")
        actions.append(action)
    if not actions:
        actions = [{"type": "unknown", "reason": "model returned no actions", "confidence": "low"}]
    return {
        "actions": actions,
        "clause_coverage": payload.get("clause_coverage") or [],
        "semantic_units": payload.get("semantic_units") or [],
        "pending_choice_contracts": payload.get("pending_choice_contracts") or [],
        "admission_rejections": payload.get("admission_rejections") or [],
        "reason": payload.get("reason") or payload.get("reasoning") or "",
    }
