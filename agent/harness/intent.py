"""LLM intent resolver for free-form Harness turns."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from ..llm.providers import provider_from_config
from ..llm.types import LLMTurnTimeoutError, LLMMessage, LLMRequest, ensure_turn_active
from ..onboarding.families import SUPPORTED_FAMILIES
from .action_registry import (
    ACTION_BY_TYPE,
    ACTION_SPECS,
    CONSULTATION_TOPIC_PURPOSES,
    CONSULTATION_TOPICS,
    SEMANTIC_SUPPORT_RELATIONS,
    TRUSTED_ACTION_METADATA_FIELDS,
    build_admission_transaction_hash,
    build_field_intake_admission_receipt,
    build_proposal_field_receipt,
    canonical_consultation_topic,
    lifecycle_rejected_action_indexes,
    normalize_action_relations,
    resolve_action_target_group,
    semantic_grounding_arguments,
    semantic_scope_accepts_action,
    semantic_scope_schema,
    validate_action_contract,
)
from .context import action_schema, build_action_resolver_prompt, group_schema, workflow_snapshot
from .domains.environment import (
    CONFIG_PROPOSAL_FIELDS,
    extract_structured_input_candidates,
    normalize_proposed_config_value,
)
from .input_values import (
    extract_rpc_method_identities,
    has_rpc_wire_evidence,
    target_mode_evidence_matches,
)
from .plan_coverage import PlanCoverageResult, TurnClause, segment_user_turn, validate_plan_coverage
from .questions import (
    answer_fits_pending,
    exact_answer,
    pending_contract_allows_semantic_scalar_normalization,
    pending_option_value_exists,
    typed_pending_value_candidates,
    value_satisfies_pending_contract,
)
from .semantic_compiler import (
    ImmutableSemanticPlan,
    WholePlanAdmission,
    freeze_semantic_plan,
    request_semantic_compilation,
    request_whole_plan_admission,
)
from .state import AgentGraphState
from agent.workflows.group_registry import (
    GROUP_SPEC_BY_NAME,
    USER_NAVIGABLE_GROUPS,
)

ALLOWED_GROUPS = list(USER_NAVIGABLE_GROUPS)

# Single source of truth for the supported adapter families (audit Finding C3):
# derive prompt/payload copies from `onboarding.families.SUPPORTED_FAMILIES`.
ADAPTER_FAMILIES = list(SUPPORTED_FAMILIES)
# The identity/hint resolvers additionally accept these two non-family answers.
ADAPTER_FAMILY_HINT_ENUM = "|".join(ADAPTER_FAMILIES + ["unsupported", "unknown"])

ALLOWED_ACTION_TYPES = [spec.action_type for spec in ACTION_SPECS]

_ADMISSION_RECEIPT_KEYS = frozenset({
    "pending_choice_contracts",
    "pending_answer_admissions",
    "pending_support_unit_ids",
    "chain_selection_admissions",
    "target_mode_selection_admissions",
    "consultation_admissions",
    "group_navigation_admissions",
    "field_reconfiguration_admissions",
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
    "becomes an owner mutation only when the source explicitly asks to alter, customize, override, or select a "
    "concrete field, value, or default that a registered owner action can apply. Naming the destination resource "
    "again, or expressing a general desire to revise it without a concrete setting or value, is not sufficient. "
    "Never manufacture a second owner mutation merely because navigation intentionally defers those questions. "
)


def resolve_action_queue(state: AgentGraphState, text: str) -> dict[str, Any]:
    """Compile and admit one immutable typed plan with a hard four-call cap."""

    try:
        provider = provider_from_config()
        request_payload = _action_queue_payload(state, text)
        clauses = tuple(
            TurnClause(
                str(item["clause_id"]),
                str(item["text"]),
                str(item.get("input_shape") or "prose"),
            )
            for item in request_payload["clauses"]
        )
        raw_response, compilation_errors = _compile_semantic_candidate(
            provider,
            system_prompt=_action_queue_prompt(),
            request_payload=request_payload,
        )
        candidate, validation = _prepare_bounded_semantic_candidate(
            raw_response,
            state,
            clauses,
            text,
            extra_errors=compilation_errors,
        )
        plan, admission, admission_errors = _review_bounded_semantic_candidate(
            provider,
            candidate,
            validation,
            state,
            clauses,
        )
        pending_contract_unresolved = bool(
            admission is not None
            and admission.valid
            and plan is not None
            and _admitted_plan_requires_pending_contract_adjudication(plan, admission, state)
        )
        if admission is not None and admission.valid and plan is not None and not pending_contract_unresolved:
            try:
                return _admitted_action_queue(plan, admission, state)
            except ValueError as exc:
                admission_errors = (*admission_errors, f"receipt attachment failed: {exc}")
        if pending_contract_unresolved:
            admission_errors = (
                *admission_errors,
                "active pending contract remains unresolved after whole-plan admission",
            )

        repair_payload = {
            "invalid_output": _parse_json_object(candidate),
            "validation_errors": list(dict.fromkeys([
                *validation.errors,
                *admission_errors,
            ])),
            "admission_result": admission.repair_context() if admission is not None else {},
            "action_schema": action_schema(),
            "original_request": request_payload,
        }
        active_pending_contract = bool(state.get("pending_question"))
        admitted_pending_answer = (
            _canonicalize_admitted_pending_argument(plan, admission, state)
            if pending_contract_unresolved and plan is not None and admission is not None
            else None
        )
        if admitted_pending_answer is not None:
            repaired_response, repair_errors = admitted_pending_answer, ()
        else:
            repaired_response, repair_errors = _compile_semantic_candidate(
                provider,
                system_prompt=(
                    _pending_contract_adjudication_prompt()
                    if active_pending_contract
                    else _action_plan_repair_prompt()
                ),
                request_payload=repair_payload,
            )
        repaired, repaired_validation = _prepare_bounded_semantic_candidate(
            repaired_response,
            state,
            clauses,
            text,
            extra_errors=repair_errors,
        )
        repaired_plan, final_admission, final_errors = _review_bounded_semantic_candidate(
            provider,
            repaired,
            repaired_validation,
            state,
            clauses,
        )
        final_pending_unresolved = bool(
            final_admission is not None
            and final_admission.valid
            and repaired_plan is not None
            and _admitted_plan_requires_pending_contract_adjudication(
                repaired_plan,
                final_admission,
                state,
            )
        )
        if (
            final_admission is not None
            and final_admission.valid
            and repaired_plan is not None
            and not final_pending_unresolved
        ):
            try:
                return _admitted_action_queue(repaired_plan, final_admission, state)
            except ValueError as exc:
                final_errors = (*final_errors, f"receipt attachment failed: {exc}")
        if final_pending_unresolved:
            final_errors = (
                *final_errors,
                "focused adjudication did not consume the active pending contract",
            )
        return _unresolved_action_queue(
            clauses,
            (*repaired_validation.errors, *final_errors),
        )
    except LLMTurnTimeoutError:
        raise
    except Exception as exc:
        return {
            "actions": [{"type": "unknown", "reason": f"action queue resolver failed: {type(exc).__name__}", "confidence": "low"}],
            "reason": "resolver failed",
        }


def _canonicalize_admitted_pending_argument(
    plan: ImmutableSemanticPlan,
    admission: WholePlanAdmission,
    state: AgentGraphState,
) -> str | None:
    """Convert one independently admitted pending-value argument to its owner."""

    pending = dict(state.get("pending_question") or {})
    if pending.get("manual_input_allowed") is not True:
        return None
    payload = plan.document()
    request = plan.request_payload()
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    records = {
        str(record.get("action_id") or ""): record
        for record in request.get("actions") or []
        if isinstance(record, Mapping)
    }
    candidates: list[tuple[int, Any, Mapping[str, Any]]] = []
    for index, row in enumerate(admission.action_verdicts):
        candidate_id = str(row.get("pending_answer_argument") or "")
        if not candidate_id or index >= len(actions) or not isinstance(actions[index], Mapping):
            continue
        record = records.get(str(row.get("action_id") or "")) or {}
        candidate = next(
            (
                item for item in record.get("pending_value_candidates") or []
                if isinstance(item, Mapping)
                and str(item.get("candidate_id") or "") == candidate_id
            ),
            None,
        )
        if candidate is None or not value_satisfies_pending_contract(candidate.get("value"), pending):
            raise ValueError("admitted pending argument does not satisfy the active contract")
        candidates.append((index, candidate.get("value"), row))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("whole-plan admission selected multiple pending answer arguments")
    index, pending_value, row = candidates[0]
    original = dict(actions[index])
    source = str(original.get("source_evidence") or "").strip()
    if not source:
        direct = next(
            (
                evidence for evidence in row.get("evidence") or []
                if isinstance(evidence, Mapping)
                and str(evidence.get("relation") or "") == "direct"
                and str(evidence.get("quote") or "").strip()
            ),
            {},
        )
        source = str(direct.get("quote") or "").strip()
    actions[index] = {
        "type": "answer_pending",
        "answer": pending_value,
        "source_evidence": source,
        "confidence": str(original.get("confidence") or "medium"),
    }
    payload["actions"] = actions
    payload.pop("admission_action_ids", None)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _compile_semantic_candidate(
    provider: Any,
    *,
    system_prompt: str,
    request_payload: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    """Run one compiler call and preserve malformed output as a repair reason."""

    try:
        return request_semantic_compilation(
            provider,
            system_prompt=system_prompt,
            request_payload=request_payload,
        ), ()
    except ValueError as exc:
        return "{}", (str(exc),)


def _prepare_bounded_semantic_candidate(
    raw_response: str,
    state: AgentGraphState,
    clauses: tuple[TurnClause, ...],
    user_text: str,
    *,
    extra_errors: tuple[str, ...] = (),
) -> tuple[str, PlanCoverageResult]:
    """Apply only deterministic ownership and registry policy before review."""

    try:
        candidate = _prepare_untrusted_action_document(raw_response)
        candidate = _apply_state_plan_policy(candidate, state)
        candidate = _reconcile_structured_candidate_ownership(candidate, clauses, state)
        candidate = _remove_empty_config_proposals(candidate)
        candidate = _normalize_prose_pending_field_proposal(
            candidate,
            state,
            clauses,
            user_text,
        )
        candidate = _materialize_absent_semantic_units(candidate, clauses)
        candidate = _canonicalize_candidate_config_proposals(candidate)
        candidate = _reset_candidate_action_ids(candidate)
        candidate = _canonicalize_pending_choice_actions(candidate, state)
        candidate = _mark_pending_owner_candidates(candidate, state)
        validation = _validate_action_document(candidate, clauses, state)
        validation = _validate_action_source_grounding(candidate, validation)
    except ValueError as exc:
        candidate = "{}"
        validation = _invalid_plan_coverage(clauses, (str(exc),))
    if extra_errors:
        validation = _merge_plan_errors(validation, extra_errors)
    return candidate, validation


def _materialize_absent_semantic_units(
    text: str,
    clauses: tuple[TurnClause, ...],
) -> str:
    """Restore missing structural units without choosing or changing actions.

    Some providers return valid typed actions but omit the required unit table.
    Mapping every authoritative clause to the immutable candidate lets the
    independent whole-plan reviewer reject wrong or incomplete semantics while
    preserving the two-call architecture.
    """

    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    units = payload.get("semantic_units")
    if not actions or (isinstance(units, list) and units):
        return text
    action_indexes = list(range(len(actions)))
    payload["semantic_units"] = [
        {
            "unit_id": f"structural-{clause.clause_id}",
            "clause_id": clause.clause_id,
            "source_text": clause.text,
            "disposition": "action",
            "action_indexes": action_indexes,
            "reason": (
                "Harness restored an omitted structural unit; semantic "
                "ownership remains subject to independent admission."
            ),
        }
        for clause in clauses
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _canonicalize_candidate_config_proposals(text: str) -> str:
    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    for index, action in enumerate(actions):
        if isinstance(action, dict) and str(action.get("type") or "") == "propose_config_values":
            actions[index] = _canonicalize_proposal_values(action)
    payload["actions"] = actions
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _canonicalize_pending_choice_actions(text: str, state: AgentGraphState) -> str:
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
        option = _matching_pending_option(action, state)
        manual_value = None if option else _matching_pending_manual_value(action, pending)
        if manual_value is not None:
            source = str(action.get("source_evidence") or "").strip()
            units = _source_units_for_action(payload, index)
            if not source or not any(
                source in str(unit.get("source_text") or "") for unit in units
            ):
                continue
            actions[index] = {
                "type": "answer_pending",
                "answer": manual_value,
                "source_evidence": source,
                "confidence": str(action.get("confidence") or "medium"),
            }
            continue
        if not option:
            continue
        action_type = str(action.get("type") or "")
        if action_type != "answer_pending":
            spec = ACTION_BY_TYPE.get(action_type)
            if spec is None or not spec.pending_option_admission:
                continue
        units = _source_units_for_action(payload, index)
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
            "answer": selected,
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
        answer = action.get("answer")
        if option or not value_satisfies_pending_contract(answer, dict(pending)):
            old_to_new[old_index] = len(retained)
            retained.append(action)
            continue
        identity = json.dumps(answer, ensure_ascii=False, sort_keys=True, default=str)
        prior = first_by_value.get(identity)
        if prior is None:
            prior = len(retained)
            first_by_value[identity] = prior
            retained.append(action)
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


def _mark_pending_owner_candidates(text: str, state: AgentGraphState) -> str:
    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    admitted = [
        index
        for index, action in enumerate(actions)
        if isinstance(action, dict) and _action_owns_pending_candidate(action, state)
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
    if str(action.get("type") or "") != "answer_pending":
        option = _matching_pending_option(action, state)
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        return bool(option and (spec is None or spec.pending_option_admission))
    if _declared_option_for_pending_answer(action, pending):
        return True
    answer = str(action.get("answer") or "").strip()
    source = str(action.get("source_evidence") or "")
    return bool(
        pending.get("manual_input_allowed") is True
        and value_satisfies_pending_contract(answer, pending)
        and _manual_answer_has_literal_source(action, source)
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
        source = str(action.get("source_evidence") or "").strip()
        if not source:
            continue
        units = _source_units_for_action(payload, index)
        if not any(source in str(unit.get("source_text") or "") for unit in units):
            errors.append(f"action {index} source_evidence is not an exact mapped-unit quote")
    return _merge_plan_errors(validation, tuple(errors)) if errors else validation


def _review_bounded_semantic_candidate(
    provider: Any,
    candidate: str,
    validation: PlanCoverageResult,
    state: AgentGraphState,
    clauses: tuple[TurnClause, ...],
) -> tuple[ImmutableSemanticPlan | None, WholePlanAdmission | None, tuple[str, ...]]:
    if not validation.valid:
        return None, None, tuple(validation.errors)
    try:
        plan = _freeze_bounded_semantic_plan(candidate, state, clauses)
    except ValueError as exc:
        return None, None, (str(exc),)
    admission = request_whole_plan_admission(
        provider,
        plan,
        semantic_policy=_semantic_fulfillment_prompt(),
        allowed_action_types=ALLOWED_ACTION_TYPES,
    )
    return plan, admission, admission.errors


def _admitted_plan_requires_pending_contract_adjudication(
    plan: ImmutableSemanticPlan,
    admission: WholePlanAdmission,
    state: AgentGraphState,
) -> bool:
    """Detect admitted work that bypasses an active typed question owner."""

    pending = dict(state.get("pending_question") or {})
    if not pending:
        return False
    payload = plan.document()
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    if payload.get("pending_choice_contracts") or any(
        isinstance(action, Mapping) and str(action.get("type") or "") == "answer_pending"
        for action in actions
    ):
        return False
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    unresolved = any(
        isinstance(unit, Mapping) and str(unit.get("disposition") or "") == "unresolved"
        for unit in units
    ) or any(
        str(row.get("verdict") or "") in {"unresolved", "omitted"}
        for row in admission.unit_verdicts
        if isinstance(row, Mapping)
    )
    if pending.get("options"):
        return unresolved or any(
            isinstance(action, Mapping) and str(action.get("type") or "") == "clarify_unresolved"
            for action in actions
        )
    if pending.get("manual_input_allowed") is not True:
        return False
    reviewer_selected_pending_owner = any(
        str(row.get("pending_answer_argument") or "")
        for row in admission.action_verdicts
        if isinstance(row, Mapping)
    )
    # The independent reviewer has already compared every typed candidate with
    # the active pending contract. An admitted mutation with no selected
    # pending argument is an interruption and must retain normal cross-group
    # routing instead of being forced to consume an unrelated question.
    return bool(reviewer_selected_pending_owner or unresolved or any(
        isinstance(action, Mapping) and str(action.get("type") or "") == "clarify_unresolved"
        for action in actions
    ))


def _freeze_bounded_semantic_plan(
    text: str,
    state: AgentGraphState,
    clauses: tuple[TurnClause, ...],
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
        pending_value_candidates = _pending_operation_value_candidates(
            operation_arguments,
            pending,
        )
        action_records.append({
            "action_id": action_ids[index],
            "action_index": index,
            "action": action,
            "registry_owner": spec.owner,
            "registry_effect": spec.effect,
            "registry_target_group": target_group,
            "registry_target_group_owner": target_spec.owner if target_spec is not None else "",
            "declared_purpose": _semantic_action_purpose(action, spec.purpose, state),
            "operation_arguments": operation_arguments,
            "allowed_support_relations": list(spec.semantic_support_relations),
            "required_value_grounding_arguments": list(
                semantic_grounding_arguments(action)
            ),
            "pending_value_candidates": pending_value_candidates,
            "unit_ids": [
                unit_ids[unit_index]
                for unit_index, owners in enumerate(unit_owner_indexes)
                if index in owners
            ],
        })
    unit_records = [
        {
            "unit_id": unit_ids[index],
            "unit_index": index,
            "unit": unit,
            "clause_id": str(unit.get("clause_id") or ""),
            "source_text": str(unit.get("source_text") or ""),
            "input_shape": clause_shapes.get(str(unit.get("clause_id") or ""), ""),
            "disposition": str(unit.get("disposition") or ""),
            "owner_action_ids": [action_ids[action_index] for action_index in unit_owner_indexes[index]],
        }
        for index, unit in enumerate(units)
        if isinstance(unit, dict)
    ]
    return freeze_semantic_plan(
        payload,
        action_records=action_records,
        unit_records=unit_records,
        review_context={
            "pending_question": state.get("pending_question") or {},
            "workflow_state": workflow_snapshot(state),
            "action_schema": action_schema(),
            "group_schema": group_schema(),
            "semantic_scope_schema": semantic_scope_schema(),
        },
    )


def _pending_operation_value_candidates(
    operation_arguments: Mapping[str, Any],
    pending: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expose typed operation values without assigning pending ownership."""

    if pending.get("manual_input_allowed") is not True:
        return []
    output: list[dict[str, Any]] = []
    for argument, value in operation_arguments.items():
        values: list[tuple[tuple[str, ...], Any]] = [((), value)]
        if isinstance(value, Mapping):
            values.extend(
                ((str(key),), nested)
                for key, nested in value.items()
            )
        for path, candidate in values:
            if not value_satisfies_pending_contract(candidate, dict(pending)):
                continue
            output.append({
                "candidate_id": f"candidate-{len(output)}",
                "argument": str(argument),
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
    pending_indexes: list[int] = []
    chain_indexes: list[int] = []
    target_indexes: list[int] = []
    consultation_indexes: list[int] = []
    navigation_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
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
        if _action_owns_pending_candidate(action, state) or _admitted_manual_answer_owns_pending(
            action,
            state,
            payload,
            index,
            row,
        ):
            pending_indexes.append(index)
        if action_type in {"choose_chain", "change_chain"}:
            chain_indexes.append(index)
        if action_type in {"choose_target_mode", "queue_workflow_goal"}:
            target_indexes.append(index)
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
    })
    admitted = _attach_semantic_admission_receipts(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        state,
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
        normalize_relations=False,
    )


def _admitted_manual_answer_owns_pending(
    action: Mapping[str, Any],
    state: AgentGraphState,
    payload: Mapping[str, Any],
    action_index: int,
    admission_row: Mapping[str, Any],
) -> bool:
    """Bind a semantically normalized value after independent admission."""

    pending = dict(state.get("pending_question") or {})
    if (
        str(action.get("type") or "") != "answer_pending"
        or pending.get("manual_input_allowed") is not True
        or not value_satisfies_pending_contract(action.get("answer"), pending)
    ):
        return False
    source = str(action.get("source_evidence") or "").strip()
    units = _source_units_for_action(payload, action_index)
    admitted_unit_ids = {
        str(unit_id)
        for unit_id in admission_row.get("unit_ids") or []
        if str(unit_id)
    }
    return bool(
        source
        and any(
            str(unit.get("unit_id") or "") in admitted_unit_ids
            and source in str(unit.get("source_text") or "")
            for unit in units
        )
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
) -> dict[str, Any]:
    unresolved = tuple(clause.text for clause in clauses)
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






















def _apply_state_plan_policy(text: str, state: AgentGraphState) -> str:
    """Remove actions that are impossible in the current product case.

    This is plan admission, not routing. It keeps semantic-unit coverage by
    remapping surviving action indexes and marks a unit unresolved only when
    every action that represented it was removed.
    """

    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    lifecycle_rejected = lifecycle_rejected_action_indexes(
        state,
        [action if isinstance(action, dict) else {} for action in actions],
    )
    if lifecycle_rejected:
        text, _changed = _remove_rejected_action_indexes(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            lifecycle_rejected,
            reason=(
                "operation is incompatible with the active target-mode lifecycle; "
                "answer the active typed question or explicitly change target mode first"
            ),
        )
        payload = _parse_json_object(text)
        actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    identity = state.get("chain_identity") or {}
    if identity.get("case") != "case3" or identity.get("adapter_family") != "unsupported":
        return text
    removed = {
        index
        for index, action in enumerate(actions)
        if isinstance(action, dict)
        and str(action.get("type") or "") in {"rpc_catalog_command", "rpc_workload_command"}
    }
    if not removed:
        return text
    index_map: dict[int, int] = {}
    retained: list[Any] = []
    for old_index, action in enumerate(actions):
        if old_index in removed:
            continue
        index_map[old_index] = len(retained)
        retained.append(action)
    payload["actions"] = retained
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        old_indexes = unit.get("action_indexes") if isinstance(unit.get("action_indexes"), list) else []
        new_indexes = [index_map[index] for index in old_indexes if index in index_map]
        unit["action_indexes"] = new_indexes
        if old_indexes and not new_indexes:
            unit["disposition"] = "unresolved"
            unit["reason"] = "RPC catalog actions are unavailable during an unsupported-family Case 3 handoff"
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _reconcile_structured_candidate_ownership(
    text: str,
    clauses: tuple[TurnClause, ...],
    state: AgentGraphState | None = None,
) -> str:
    """Complete syntax-owned facts after the model selects config intent.

    The model decides whether a structured clause is configuration at all.
    Once it maps that clause to ``propose_config_values``, the deterministic
    parser owns lossless transfer of recognized and unmapped assignments. An
    exact registered config assignment that the model tried to bind directly
    to the active manual field is also a configuration transaction, so this
    boundary converts it to the same review proposal. This keeps logs/examples
    outside the configuration path while preventing either silent field loss
    or review bypass.
    """

    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    if not actions or not units:
        return text

    changed = False
    pending = dict((state or {}).get("pending_question") or {})
    pending_field = str(pending.get("field") or "").strip().upper()
    structured_pending_values = []
    if pending.get("manual_input_allowed") is True and pending_field:
        for candidate_clause in clauses:
            if candidate_clause.input_shape != "structured":
                continue
            candidate = extract_structured_input_candidates(candidate_clause.text) or {}
            candidate_values = dict(candidate.get("config_values") or {})
            if pending_field in candidate_values:
                structured_pending_values.append(candidate_values[pending_field])
    for clause in clauses:
        if clause.input_shape != "structured":
            continue
        candidates = extract_structured_input_candidates(clause.text)
        if not candidates:
            continue
        clause_units = [
            unit
            for unit in units
            if isinstance(unit, dict) and str(unit.get("clause_id") or "") == clause.clause_id
        ]
        config_values = dict(candidates.get("config_values") or {})
        unmapped_values = dict(candidates.get("unmapped_values") or {})
        proposal_indexes = {
            index
            for unit in clause_units
            for index in (unit.get("action_indexes") if isinstance(unit.get("action_indexes"), list) else [])
            if isinstance(index, int)
            and 0 <= index < len(actions)
            and isinstance(actions[index], dict)
            and str(actions[index].get("type") or "") == "propose_config_values"
        }
        if (
            not proposal_indexes
            and pending.get("manual_input_allowed") is True
            and pending_field
            and pending_field in config_values
        ):
            direct_answer_indexes = {
                index
                for unit in clause_units
                for index in (
                    unit.get("action_indexes")
                    if isinstance(unit.get("action_indexes"), list)
                    else []
                )
                if isinstance(index, int)
                and 0 <= index < len(actions)
                and isinstance(actions[index], dict)
                and str(actions[index].get("type") or "") == "answer_pending"
            }
            if direct_answer_indexes:
                proposal_index = min(direct_answer_indexes)
                actions[proposal_index] = {
                    "type": "propose_config_values",
                    "source_format": str(candidates.get("source_format") or "mixed"),
                    "config_values": dict(config_values),
                    "unmapped_values": dict(unmapped_values),
                    "source_evidence": clause.text,
                    "confidence": "high",
                }
                for unit in clause_units:
                    indexes = (
                        unit.get("action_indexes")
                        if isinstance(unit.get("action_indexes"), list)
                        else []
                    )
                    if not any(index in direct_answer_indexes for index in indexes):
                        continue
                    unit["action_indexes"] = [
                        *[index for index in indexes if index not in direct_answer_indexes],
                        proposal_index,
                    ]
                    unit["disposition"] = "action"
                    unit["reason"] = "registered structured config assignment requires review"
                proposal_indexes = {proposal_index}
                changed = True
            elif len(structured_pending_values) == 1:
                pending_answer_indexes = {
                    index
                    for index, action in enumerate(actions)
                    if isinstance(action, dict)
                    and str(action.get("type") or "") == "answer_pending"
                    and str(
                        action.get("selected_value")
                        if action.get("selected_value") is not None
                        else action.get("answer") or ""
                    ).strip().strip("\"'")
                    == str(structured_pending_values[0]).strip().strip("\"'")
                }
                if len(pending_answer_indexes) == 1:
                    proposal_index = next(iter(pending_answer_indexes))
                    actions[proposal_index] = {
                        "type": "propose_config_values",
                        "source_format": str(candidates.get("source_format") or "mixed"),
                        "config_values": dict(config_values),
                        "unmapped_values": dict(unmapped_values),
                        "source_evidence": clause.text,
                        "confidence": "high",
                    }
                    for unit in clause_units:
                        unit["action_indexes"] = [proposal_index]
                        unit["disposition"] = "action"
                        unit["reason"] = "source-grounded structured config assignment requires review"
                    payload["pending_answer_admissions"] = [
                        index
                        for index in payload.get("pending_answer_admissions", [])
                        if index != proposal_index
                    ]
                    proposal_indexes = {proposal_index}
                    changed = True
        if not proposal_indexes:
            continue
        for index in proposal_indexes:
            action = actions[index]
            merged_config = dict(action.get("config_values") or {})
            merged_unmapped = dict(action.get("unmapped_values") or {})
            before = (dict(merged_config), dict(merged_unmapped))
            for key in config_values:
                for existing in tuple(merged_config):
                    if str(existing).strip().upper() == str(key).strip().upper():
                        merged_config.pop(existing, None)
            for key in unmapped_values:
                for existing in tuple(merged_unmapped):
                    if str(existing).strip().upper() == str(key).strip().upper():
                        merged_unmapped.pop(existing, None)
            merged_config.update(config_values)
            merged_unmapped.update(unmapped_values)
            action["config_values"] = merged_config
            action["unmapped_values"] = merged_unmapped
            changed = changed or before != (merged_config, merged_unmapped)

        proposal_index = min(proposal_indexes)
        for unit in clause_units:
            if str(unit.get("disposition") or "") != "unresolved":
                continue
            unit_candidates = extract_structured_input_candidates(str(unit.get("source_text") or ""))
            if not unit_candidates:
                continue
            if not (
                unit_candidates.get("config_values")
                or unit_candidates.get("unmapped_values")
            ):
                continue
            unit["disposition"] = "action"
            unit["action_indexes"] = [proposal_index]
            unit["reason"] = "deterministic structured assignment owned by propose_config_values"
            changed = True

        workflow_values = dict(candidates.get("workflow_values") or {})
        workflow_owner_indexes: set[int] = set()
        every_workflow_value_owned = True
        for key, value in workflow_values.items():
            assignment = f"{key}={value}"
            owners = {
                index
                for index, action in enumerate(actions)
                if isinstance(action, dict)
                and str(action.get("type") or "") != "propose_config_values"
                and assignment.casefold() in str(action.get("source_evidence") or "").casefold()
            }
            if not owners:
                every_workflow_value_owned = False
                break
            workflow_owner_indexes.update(owners)
        if every_workflow_value_owned:
            owner_indexes = sorted(proposal_indexes | workflow_owner_indexes)
            atomic_unit = {
                "unit_id": f"{clause.clause_id}-structured",
                "clause_id": clause.clause_id,
                "source_text": clause.text,
                "disposition": "action",
                "action_indexes": owner_indexes,
                "reason": "atomic structured configuration transaction",
            }
            normalized_units: list[dict[str, Any]] = []
            inserted = False
            for unit in units:
                if (
                    isinstance(unit, dict)
                    and str(unit.get("clause_id") or "") == clause.clause_id
                ):
                    if not inserted:
                        normalized_units.append(atomic_unit)
                        inserted = True
                    continue
                if isinstance(unit, dict):
                    normalized_units.append(unit)
            if not inserted:
                normalized_units.append(atomic_unit)
            payload["semantic_units"] = normalized_units
            units = payload["semantic_units"]
            changed = True

    if not changed:
        return text
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _normalize_prose_pending_field_proposal(
    text: str,
    state: AgentGraphState,
    clauses: tuple[TurnClause, ...],
    user_text: str,
) -> str:
    """Compile a prose-supplied pending scalar back to its finite owner."""

    pending = dict(state.get("pending_question") or {})
    pending_field = str(pending.get("field") or "").strip().upper()
    if (
        pending.get("manual_input_allowed") is not True
        or pending_field not in CONFIG_PROPOSAL_FIELDS
        or not clauses
        or any(clause.input_shape != "prose" for clause in clauses)
    ):
        return text

    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    proposal_indexes = [
        index
        for index, action in enumerate(actions)
        if isinstance(action, dict)
        and str(action.get("type") or "") == "propose_config_values"
    ]
    if len(proposal_indexes) != 1:
        return text
    proposal_index = proposal_indexes[0]
    proposal = actions[proposal_index]
    config_values = dict(proposal.get("config_values") or {})
    normalized_values = {
        str(key).strip().upper(): value for key, value in config_values.items()
    }
    residual_values = {
        key: value
        for key, value in normalized_values.items()
        if key != pending_field
        and not _proposal_value_matches_confirmed_config(state, key, value)
    }
    if (
        pending_field not in normalized_values
        or residual_values
        or proposal.get("unmapped_values")
        or proposal.get("conflicts")
    ):
        return text
    answer = str(normalized_values[pending_field]).strip()
    if (
        not answer
        or not value_satisfies_pending_contract(answer, pending)
        or re.search(
            rf"(?<!\w){re.escape(answer)}(?!\w)",
            str(user_text or ""),
            flags=re.IGNORECASE,
        ) is None
    ):
        return text
    actions[proposal_index] = {
        "type": "answer_pending",
        "answer": answer,
        "source_evidence": answer,
    }
    payload["actions"] = actions
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _proposal_value_matches_confirmed_config(
    state: AgentGraphState,
    key: str,
    proposed_value: Any,
) -> bool:
    """Return whether a proposal only reaffirms canonical confirmed state."""

    normalized_key, normalized_proposed = normalize_proposed_config_value(key, proposed_value)
    confirmed = dict(state.get("confirmed_config") or {})
    state_key = (
        "has_accounts_device"
        if normalized_key == "HAS_ACCOUNTS_DEVICE"
        else normalized_key
    )
    if state_key not in confirmed:
        return False
    _current_key, normalized_current = normalize_proposed_config_value(
        normalized_key,
        confirmed[state_key],
    )
    return normalized_proposed == normalized_current


def _remove_empty_config_proposals(text: str) -> str:
    """Remove proposal actions that own no mapped, unmapped, or conflicting fact."""

    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    empty_indexes = tuple(
        index
        for index, action in enumerate(actions)
        if isinstance(action, dict)
        and str(action.get("type") or "") == "propose_config_values"
        and not action.get("config_values")
        and not action.get("unmapped_values")
        and not action.get("conflicts")
    )
    if not empty_indexes:
        return text
    normalized, _ = _remove_action_indexes(text, empty_indexes)
    return normalized






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
    coverage = validate_plan_coverage(payload, clauses)
    if state is not None:
        action_errors.extend(_validate_pending_choice_contracts(payload, state))
    if not action_errors:
        return coverage
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
        return (
            f"Select the displayed pending option labelled {str(option.get('label') or '')!r} "
            f"with value {option.get('value')!r}; its declared effect is: {effect}"
        )
    if action_type == "answer_pending" and state:
        return "Supply a source-grounded answer that actually satisfies the active typed pending contract."
    if action_type == "answer_opening_question":
        topic = canonical_consultation_topic(action.get("topic"))
        subject = str(action.get("subject") or "").strip()
        subject_scope = f" for the exact subject {subject!r}" if subject else ""
        topic_purpose = CONSULTATION_TOPIC_PURPOSES.get(topic, "Answer the selected read-only topic.")
        return (
            f"Answer only the user's independent read-only consultation topic {topic!r}"
            f"{subject_scope}: {topic_purpose} This purpose does not answer a different consultation topic."
        )
    if action_type in {"choose_chain", "change_chain"}:
        candidates = [
            str(item).strip()
            for item in action.get("chain_candidates") or []
            if str(item).strip()
        ]
        if len(candidates) > 1:
            return "Ask the user to resolve the exact finite chain candidate set supplied in this source unit."
        if action_type == "change_chain":
            return "Select the exact source-supplied chain as the requested replacement for the current chain."
        return "Select the exact source-supplied chain as the benchmark target when no chain is confirmed."
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


def _declared_option_for_pending_answer(
    action: dict[str, Any],
    pending: dict[str, Any],
) -> dict[str, Any]:
    """Resolve one answer action against the exact typed option contract.

    The planner may express a declared option through ``answer`` without
    repeating it in ``selected_value``. When it supplies both representations,
    they must agree before the action can become an option anchor. A distinct
    source-grounded manual value is owned by complete-turn adjudication.
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
    for index, action in enumerate(actions):
        if isinstance(action, dict):
            action["_admission_action_id"] = admission_action_ids[index]
            action["_transaction_action_ids"] = list(admission_action_ids)
            action["_plan_transaction_hash"] = transaction_hash
        if isinstance(action, dict) and _requires_semantic_fulfillment_review(action):
            action["semantic_purpose_verified"] = True
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
























def _remove_rejected_action_indexes(
    text: str,
    rejected_indexes: tuple[int, ...],
    *,
    reason: str,
) -> tuple[str, bool]:
    """Remove rejected siblings while preserving semantic-unit ownership."""

    return _remove_action_indexes(
        text,
        rejected_indexes,
        rejection_reason=reason,
    )


def _remove_action_indexes(
    text: str,
    removed_indexes: tuple[int, ...],
    *,
    rejection_reason: str = "",
) -> tuple[str, bool]:
    """Compact actions and every index-bearing receipt after removal.

    ``rejection_reason`` is present only when an admission gate rejected the
    action. Transaction normalization uses the same index-safe compaction
    without creating false rejection evidence.
    """

    removed = set(removed_indexes)
    if not removed:
        return text, False
    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    admission_action_ids = _ensure_admission_action_ids(payload)
    removed_types = {
        index: str(actions[index].get("type") or "unknown")
        for index in removed
        if 0 <= index < len(actions) and isinstance(actions[index], dict)
    }
    old_to_new: dict[int, int] = {}
    retained: list[dict[str, Any]] = []
    for old_index, action in enumerate(actions):
        if old_index in removed:
            continue
        old_to_new[old_index] = len(retained)
        retained.append(action)
    payload["actions"] = retained
    payload["admission_action_ids"] = [
        admission_action_ids[index]
        for index in range(len(actions))
        if index in old_to_new
    ]
    for admission_key in (
        "pending_answer_admissions",
        "chain_selection_admissions",
        "target_mode_selection_admissions",
        "consultation_admissions",
    ):
        admissions = payload.get(admission_key)
        if not isinstance(admissions, list):
            continue
        payload[admission_key] = [
            old_to_new[index]
            for index in admissions
            if isinstance(index, int) and index in old_to_new
        ]
    navigation_admissions = payload.get("group_navigation_admissions")
    if isinstance(navigation_admissions, list):
        payload["group_navigation_admissions"] = [
            {**row, "action_index": old_to_new[int(row["action_index"])]}
            for row in navigation_admissions
            if isinstance(row, dict)
            and isinstance(row.get("action_index"), int)
            and int(row["action_index"]) in old_to_new
        ]
    field_admissions = payload.get("field_reconfiguration_admissions")
    if isinstance(field_admissions, list):
        payload["field_reconfiguration_admissions"] = [
            {**row, "action_index": old_to_new[int(row["action_index"])]}
            for row in field_admissions
            if isinstance(row, dict)
            and isinstance(row.get("action_index"), int)
            and int(row["action_index"]) in old_to_new
        ]
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        indexes = unit.get("action_indexes") if isinstance(unit.get("action_indexes"), list) else []
        unit["action_indexes"] = [old_to_new[index] for index in indexes if index in old_to_new]
        if not unit["action_indexes"] and str(unit.get("disposition") or "") == "action":
            unit["disposition"] = "unresolved"
            unit["reason"] = rejection_reason or "action removed during transaction normalization"
    if rejection_reason:
        payload.setdefault("admission_rejections", []).extend(
            {
                "admission_action_id": admission_action_ids[index],
                "stage_action_index": index,
                "action_type": removed_types.get(index, "unknown"),
                "reason": rejection_reason,
            }
            for index in sorted(removed)
            if 0 <= index < len(admission_action_ids)
        )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True), True


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
        + "For context_reviews, context_only is true only when source_text is prose with no independent present demand and either (a) requests only an inevitable completion_effect explicitly declared by the supplied pending_question for the same pending answer, or (b) is explanatory context, an operation restatement, provenance, format scope, temporal scope, evidence-completeness scope, or a non-mutation constraint for exactly one related_operation, and that operation declares the matching allowed_support_relation. operation_restatement repeats the same requested effect or explains why that same effect is wanted, without adding another requested effect; it requires a direct_source_unit for the operation and cannot authorize an operation by itself. evidence_completeness means only that the source states which request, response, parameter, or documentation evidence is presently available or absent for the same submitted RPC operation; it cannot supply a second method, contradict the submitted operation, or waive required validation. The related operation must have a direct_source_unit that states the actual operation request. Such declared support adds no independent operation. A present answer, question, selection, correction, contradiction, concrete value, mutation, different navigation destination, evidence submission, or execution instruction is independent and therefore false. A statement that answers the supplied pending_question is not context. input_shape must be prose. planner_reason is untrusted and cannot establish the verdict. Missing or ambiguous intent is false. "
        "Operations are opaque, already-registered Harness operations. Internal operation names are intentionally absent because registration, lifecycle, ordering, and choose-versus-change selection are deterministic Harness responsibilities. Never infer or discuss an internal operation name and never reject a purpose on registry or lifecycle grounds. Decide only whether the exact source_units "
        "semantically and explicitly support the declared purpose and its supplied arguments. Workflow state "
        "and a pending question are context, never user evidence. For unit_reviews, ignore pending_question entirely: "
        "an explicit mutation or navigation may interrupt the old question, and the coordinator alone decides interruption, invalidation, and resume behavior. "
        "For an action-purpose review with several source_units, return every exact source_units string once in source_unit_reviews. Mark at least one as direct. Mark another as support only when its exact support_relation appears in the supplied allowed_support_relations; otherwise the action is unsupported. Direct rows use an empty support_relation. A support unit may not hide an independent selection, mutation, consultation, contradiction, concrete value owned by another operation, or evidence demand. For a single source unit, source_unit_reviews may be empty. In particular, structured partial configuration directly supports a configuration-proposal purpose, and a same-turn instruction to ask for remaining required values after review is processing scope only when that relation is declared by the operation contract. "
        "Pending-question context is present only when a pending-answer purpose itself is being reviewed. A question asking to "
        "summarize, explain, compare, or report retained state is read-only and cannot support "
        "a durable mutation. An evidence-ingestion purpose is supported only when the "
        "source itself contributes an attributable RPC method-schema, request, parameter, "
        "response, or documentation fact; a request to discuss existing evidence is not new "
        "evidence. An explicit statement that a named/current RPC method has no parameters, "
        "describes one or more parameter meanings/types, or describes its response/result is new schema evidence. "
        "For unit_reviews, source_text is the exact unit under review. related_source_units contains only other units in the same turn that share at least one mapped operation; it is relationship context, not permission to ignore an independent demand. A source_text that is purely introductory, trailing, format, provenance, or temporal framing for those related units is complete when the shared mapped operations preserve the related request. If source_text contains its own selection, mutation, consultation, contradiction, correction, value, or evidence demand, that demand must still be represented by mapped_actions. Each mapped_actions row contains an opaque operation index, source-grounded arguments, and its authoritative declared_purpose. Judge whether the set of declared purposes preserves every explicit demand; do not require fields that a declared purpose intentionally collects in later typed questions. Review the proposed operations as one ordered transaction: an earlier purpose that selects a source-grounded wire method may establish the draft referenced by a later evidence-ingestion purpose, but pre-existing workflow state alone cannot. "
        "Selecting or navigating to a group does not preserve an explicit request to alter one registered scalar field in that group. When no replacement value is supplied, completeness requires the registered typed field-intake purpose; when a source-grounded value is supplied, completeness requires the owning proposal or mutation purpose. "
        "A registered operation performs the effect stated by its authoritative declared_purpose. Never require operation_arguments to repeat, simulate, or prove that effect; arguments carry only source-selected values and evidence required by that operation. In particular, a lifecycle command may legitimately have only source evidence as its argument while its declared purpose defines the state transition. "
        "When an active evidence collection exists, an explicit source demand to pause or suspend that collection is independent from navigation or configuration and requires a purpose that pauses while preserving it. Navigation alone is incomplete for that demand. An explicit request to resume a paused collection likewise requires a resume purpose. "
        "A custom-RPC-entry purpose is supported when the source explicitly asks to start, add, supply, or configure a custom RPC method workflow. It intentionally carries no endpoint, method identity, or schema payload; requiring those facts at entry would skip later typed collection questions. A discussion-only question about whether custom RPC is possible does not support entry. "
        "A read-only consultation purpose is supported only when the source asks for an answer, explanation, comparison, status, preparation guidance, or similar information. It is not supported when the source explicitly requests only a selection, mutation, navigation, execution, or evidence-ingestion operation. Distinct consultation subjects remain independent demands: a purpose that reports workflow configuration or pending context does not report whether a current or historical benchmark job exists, and a job-status purpose does not report retained workflow configuration. A compound unit asking for both is complete only when mapped purposes explicitly cover both subjects. "
        + GROUP_NAVIGATION_SEMANTIC_POLICY
        + "A target-mode-intake purpose is supported when the source has a benchmark or observation goal but leaves fake-node, real-node, or sync-observe unresolved, including explicit indecision between modes. It intentionally asks a later typed question and requires no selected mode in the source. "
        "A chain-candidate-intake purpose is supported when the source presents one or more tentative benchmark-chain candidates without committing to one. It intentionally preserves candidates for a later typed confirmation question and is not a chain mutation. "
        "A QPS-customization purpose requires an explicit request to alter, tune, override, or avoid defaults of one or more QPS profile values, even when concrete numbers arrive later. Merely visiting the QPS area without requesting a profile-value change is navigation. "
        "A wire-method-selection purpose requires an actual callable wire method, not merely a schema field named method or method_id. An endpoint-selection purpose requires an explicitly selected validation endpoint, not an example or documentation URL. A secondary-development evidence purpose likewise requires new protocol, endpoint, request, response, or official-document evidence. A pending-answer purpose must actually answer the supplied pending contract. Reset and execution "
        "approval require explicit authorization. For each unit_review, complete is true only when mapped operations collectively preserve every explicit selection, mutation, consultation, evidence request, and navigation demand in source_text; one mapped purpose may be valid while the set is still incomplete. When complete is false, missing_demand_quote must be the shortest non-empty exact substring of source_text that states one omitted independently actionable demand; otherwise it must be empty. A committed named benchmark chain requires a declared purpose that selects that exact chain. A tentative chain candidate may instead be completely preserved by a declared intake purpose that asks the user to resolve candidates, while a chain mentioned only in a support question needs no chain mutation or intake purpose. When a schema-evidence pending question identifies an existing catalog draft, deictic source text such as 'this method has no parameters' or 'it returns a hex value' contributes parameter/response evidence to that current draft and may support an evidence-ingestion purpose; it still cannot support a wire-method-selection purpose unless the source itself names the wire method. Do not repair, route, invent, rename, or classify an operation."
    )


def _action_plan_repair_prompt() -> str:
    return (
        "Repair one malformed AnyChain typed action-plan response. Return one JSON object only. "
        "Preserve the original user requests; do not add choices or values. Use only action types "
        "and fields declared by action_schema. Satisfy each action's required_arguments, omit every "
        "undeclared field, and place allowed arguments directly beside type. Return an actions list "
        "with optional document-level conflicts and reason. The original_request contains authoritative structural clauses. "
        "admission_rejections in invalid_output are authoritative control-boundary failures. Do not repeat a rejected operation for the same source unit. Preserve that unit with the registered owner-domain mutation or intake operation whose declared purpose matches the user's request; if none matches, leave it unresolved. In particular, a navigation operation rejected because the source requests a specific owned configuration change must be replaced by that group's mutation/intake operation, not rephrased as navigation. "
        "Preserve every semantic-unit source_text partition from invalid_output that already exactly covers its authoritative clause. When one invalid action must be split into multiple valid typed actions, keep that valid partition and update only its action_indexes to reference all replacement actions; do not repartition or reinterpret the user turn. "
        "Return semantic_units with ordered exact source_text anchors copied from every clause. "
        "The Harness derives character offsets; do not calculate them. Each row is "
        "{unit_id, clause_id, source_text, disposition:'action'|'context'|'unresolved', optional scope_constraint from semantic_scope_schema, "
        "action_indexes:[zero-based indexes], reason}. Split prose only when exact contiguous anchors cover every word. "
        "When conjunctions or framing make a lossless split uncertain, use one full-clause unit mapped to every preserving typed action. "
        "Keep structured clauses atomic. Anchors may omit only punctuation or whitespace between units; never omit prose. "
        "Every action source_evidence must be a short exact excerpt contained wholly inside one semantic unit mapped to that action. Never span, concatenate, or quote across neighboring units; map neighboring support units to the action separately. "
        "Introductory, framing, and trailing prose around a structured block must have its own semantic unit mapped to the block-consuming action, be context only when it contains no present operation and will pass independent context admission, or be explicitly unresolved when the relationship is unclear. Structured clauses can never be context. "
        "structured_candidates are deterministic syntax facts for their clause. When that clause is configuration/review input, preserve config_values and unmapped_values in propose_config_values, route workflow_values through their typed workflow actions, and map the atomic clause to every action needed to preserve it. A same-turn instruction to review or apply that partial configuration and continue asking for required values not supplied is processing scope of propose_config_values; map it to that proposal without adding resume_current_flow, bypassing review, or leaving it unresolved. "
        "Use only semantic_scope_schema constraints. consultation_only maps only read-only consultation actions. no_configuration_mutation may map read-only inspection and workflow navigation but not configuration changes. no_execution may map non-execution actions. Judge these constraints from action_schema.effect, never action lifetime. "
        "Never claim an action covers a URL, exact wire RPC method, or concrete fact unless that exact value is present in the mapped owning action."
        "When validation_errors report source anchors that do not cover a prose clause, repair that clause with exactly one semantic unit whose source_text is the complete authoritative clause text and whose action_indexes list every typed action that preserves the clause. "
        "When validation_errors report a missing exact wire RPC method, add the owning rpc_catalog_command set_method action with that exact method; citing the whole sentence as source_evidence on another action does not preserve it."
        "When validation rejects set_method because its method came only from workflow state while a schema-evidence question is active, replace it with rpc_catalog_command append_evidence and copy the exact user description into rpc_schema_evidence; do not require the user to repeat the already-active draft method."
        "When validation reports an incomplete semantic unit, preserve its existing valid actions and add every missing typed action explicitly required by that unit; do not replace the unit with a clarification when action_schema can represent the request."
        "When an active evidence collection exists and validation reports an omitted pause/suspend or resume demand, add the registered evidence-collection lifecycle action and retain every independent navigation or configuration action from the same turn."
        "When validation rejects request_qps_customization because the source only asks to visit or configure the QPS area before another area, replace it with change_group(qps_profile) and exact source_evidence; do not leave that representable navigation unresolved."
        "When a source selects or navigates to a configuration group and also asks to alter one registered scalar field without supplying a replacement value, preserve the field request with request_config_field_input using that exact field and source evidence. Do not copy the current, detected, default, or example value into propose_config_values."
        "When adjacent clauses reject the current mutually exclusive workflow and explicitly select a replacement, one choose_target_mode action for the replacement may preserve both clauses. Map both semantic units to that same action index; do not invent a cancellation action or leave the rejection unresolved."
        "Never add answer_pending merely because a pending question exists. Add it only when the exact source text actually answers that typed question contract. When validation rejects an operation as incompatible with the active target-mode lifecycle and the same source supplies a value for the active typed question, preserve that value with answer_pending; never retry the incompatible operation."
    )


def _pending_contract_adjudication_prompt() -> str:
    """Return the single bounded adjudication contract for an active question."""

    return (
        _action_plan_repair_prompt()
        + " The original_request contains one active typed pending question. Re-adjudicate the complete "
        "user turn once. If any semantic unit selects one declared option, represent that selection only "
        "as answer_pending with selected_value exactly equal to that option's declared value and "
        "source_evidence copied from the selecting semantic unit. If the question accepts manual input "
        "and a semantic unit supplies a value satisfying its typed validation, represent that value only "
        "as answer_pending with answer equal to the exact extracted value and source_evidence copied from "
        "the supplying semantic unit. pending_typed_candidates are syntax-only candidates derived from the "
        "question's declared validation; use one only when the source semantically selects it, and never "
        "guess among multiple candidates. Do not emit a domain owner mutation for the same answer. Preserve "
        "every unrelated, interrupting, or compound demand as its own registered action and preserve all "
        "semantic-unit mappings. If the active question is not resolved by the source, leave it unresolved "
        "rather than guessing. Return one complete plan; this is the only focused adjudication and the "
        "entire result remains subject to whole-plan admission."
    )


def _action_queue_prompt() -> str:
    return build_action_resolver_prompt() + GROUP_NAVIGATION_SEMANTIC_POLICY


def _action_queue_payload(state: AgentGraphState, text: str) -> dict[str, Any]:
    raw = str(text or "")
    non_empty_lines = [line for line in raw.splitlines() if line.strip()]
    clauses = segment_user_turn(raw)
    structured_candidates = []
    for clause in clauses:
        if clause.input_shape != "structured":
            continue
        candidates = extract_structured_input_candidates(clause.text)
        if candidates and (candidates.get("config_values") or candidates.get("workflow_values")):
            structured_candidates.append({
                "clause_id": clause.clause_id,
                **candidates,
            })
    pending = dict(state.get("pending_question") or {})
    return {
        "user_text": raw,
        "clauses": [clause.as_dict() for clause in clauses],
        "input_shape": {
            "multiline": len(non_empty_lines) > 1,
            "non_empty_line_count": len(non_empty_lines),
            "contains_json_delimiters": "{" in raw and "}" in raw,
        },
        "structured_candidates": structured_candidates,
        "pending_typed_candidates": list(
            typed_pending_value_candidates(raw, pending)
        ),
        "action_schema": action_schema(),
        "semantic_scope_schema": semantic_scope_schema(),
        "group_schema": group_schema(),
        "workflow_state": workflow_snapshot(state),
    }



def resolve_unknown_chain_identity(state: AgentGraphState, chain_text: str) -> dict[str, Any]:
    """Ask the configured model whether an unknown chain appears to exist."""

    try:
        provider = provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=_chain_identity_prompt()),
                    LLMMessage(
                        role="user",
                        content=json.dumps(_chain_identity_payload(state, chain_text), ensure_ascii=False, sort_keys=True),
                    ),
                ],
                temperature=0.0,
                max_tokens=900,
            )
        )
        payload = _parse_json_object(response.text)
    except LLMTurnTimeoutError:
        raise
    except Exception as exc:
        payload = {"chain_exists": None, "reason": f"chain identity resolver failed: {type(exc).__name__}", "confidence": "low"}
    payload.setdefault("chain_text", chain_text)
    return payload


def extract_chain_mention(state: AgentGraphState, text: str) -> dict[str, Any]:
    """Extract a chain/network mention from a turn that was routed elsewhere."""

    try:
        provider = provider_from_config()
        payload = _extract_chain_mention_with_provider(provider, state, text)
    except LLMTurnTimeoutError:
        raise
    except Exception as exc:
        payload = {"found": False, "reason": f"chain mention extraction failed: {type(exc).__name__}", "confidence": "low"}
    payload.setdefault("found", False)
    return payload


def _extract_chain_mention_with_provider(
    provider: Any,
    state: AgentGraphState,
    text: str,
) -> dict[str, Any]:
    response = provider.complete(
        LLMRequest(
            messages=[
                LLMMessage(role="system", content=_chain_mention_prompt()),
                LLMMessage(
                    role="user",
                    content=json.dumps(_chain_mention_payload(state, text), ensure_ascii=False, sort_keys=True),
                ),
            ],
            temperature=0.0,
            max_tokens=300,
        )
    )
    return _parse_json_object(response.text)






def extract_rpc_schema_from_evidence(state: AgentGraphState, evidence: str, *, method_hint: str = "") -> dict[str, Any]:
    """Extract a typed RPC schema draft from user-provided evidence.

    The draft is not trusted by itself. The Harness must still ask the user to
    confirm it and then probe the endpoint before accepting the method.
    """

    try:
        provider = provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=_rpc_schema_prompt()),
                    LLMMessage(
                        role="user",
                        content=json.dumps(
                            _rpc_schema_payload(state, evidence, method_hint=method_hint),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=1200,
            )
        )
        payload = _parse_json_object(response.text)
    except LLMTurnTimeoutError:
        raise
    except Exception as exc:
        payload = {"status": "failed", "reason": f"schema extraction failed: {type(exc).__name__}", "confidence": "low"}
    payload.setdefault("status", "draft")
    payload.setdefault("method", method_hint)
    return payload


def analyze_evidence_with_model(state: AgentGraphState, evidence: str, user_question: str) -> str:
    """Analyze saved evidence with product context, without authorizing actions.

    Evidence analysis is advisory. The model receives the workflow snapshot and
    framework boundaries, but its response cannot mutate state or execute a
    tool. When the provider is unavailable, return an explicit conservative
    fallback instead of inventing a diagnosis from keyword rules.
    """

    language = str(state.get("language") or "en")
    if not evidence.strip():
        return (
            "没有可分析的日志证据。请粘贴真实日志/错误栈，或指定 job_id。"
            if language.startswith("zh")
            else "No log evidence is available. Paste real logs/a stack trace, or name a job_id."
        )
    try:
        provider = provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "You analyze execution evidence for AnyChain Benchmark Agent. "
                            "Use only the supplied evidence and workflow context. Distinguish observed facts, likely causes, "
                            "and verification steps. Never claim a benchmark passed, an endpoint works, or a metric exists "
                            "without evidence. Do not propose state mutations or pretend to run tools. Answer in the requested "
                            "language while preserving commands, paths, environment variables, job ids, chain names, and RPC methods."
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=json.dumps(
                            {
                                "language": language,
                                "question": user_question,
                                "evidence": evidence,
                                "workflow_state": workflow_snapshot(state),
                                "framework_boundaries": {
                                    "supported_groups": list(DEFAULT_GROUP_ORDER),
                                    "supported_adapter_families": ADAPTER_FAMILIES,
                                },
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=1200,
            )
        )
        answer = str(response.text or "").strip()
        if answer:
            return answer
    except LLMTurnTimeoutError:
        raise
    except Exception as exc:
        error_type = type(exc).__name__
    else:
        error_type = "EmptyResponse"
    preview = "\n".join(evidence.splitlines()[:8])
    if language.startswith("zh"):
        return (
            f"当前无法调用配置模型完成证据分析（{error_type}）。我不会猜测根因。\n"
            "请确认模型配置后重试，或提供对应 job_id 以读取本地任务产物。\n"
            f"已保存证据预览：\n{preview}"
        )
    return (
        f"The configured model could not analyze this evidence ({error_type}); I will not guess the root cause.\n"
        "Verify the model configuration and retry, or provide the related job_id so local artifacts can be read.\n"
        f"Saved evidence preview:\n{preview}"
    )


def _chain_identity_prompt() -> str:
    return (
        "You resolve blockchain chain names for AnyChain Benchmark Agent. "
        "Return one JSON object only. Do not explain. "
        "The input chain is not one of the configured AnyChain templates. "
        "Decide whether it appears to be a real blockchain/network name, "
        "whether it is just a typo/partial alias for a known chain, and what adapter family it likely uses. "
        "Do not claim certainty if unsure. "
        f"Allowed adapter_family values: {', '.join(ADAPTER_FAMILIES + ['unsupported', 'unknown'])}. "
        "Schema: {chain_exists:boolean|null, canonical_chain_name:string, adapter_family:string, protocol_or_api:string, possible_known_chain:string, "
        "evidence_summary:string, confidence:'low'|'medium'|'high'}."
    )


def _chain_identity_payload(state: AgentGraphState, chain_text: str) -> dict[str, Any]:
    framework = state.get("framework_summary") or {}
    return {
        "unknown_chain_text": chain_text,
        "known_chains": framework.get("chains", [])[:120],
        "supported_adapter_families": ADAPTER_FAMILIES,
        "web_research": state.get("web_research") or {},
        "target_mode": state.get("target_mode") or "",
        "workflow_mode": state.get("workflow_mode") or "",
    }


def _chain_mention_prompt() -> str:
    return (
        "You extract blockchain or network names from one AnyChain user turn. "
        "Return one JSON object only. Do not explain. "
        "Extract only an explicit chain/network/product name that the user wants to benchmark or observe. "
        "Do not treat cloud regions, zones, instance types, disks, URLs, QPS values, or generic words as chain names. "
        "If no explicit chain/network is present, return found=false. "
        "Known examples include Solana, Ethereum, BNB, BSC, Bitcoin, Flow, Monad, Base, Polygon. "
        "Schema: {found:boolean, chain_text:string, reason:string, confidence:'low'|'medium'|'high'}."
    )


def _chain_mention_payload(state: AgentGraphState, text: str) -> dict[str, Any]:
    framework = state.get("framework_summary") or {}
    return {
        "user_text": text,
        "known_chains": framework.get("chains", [])[:120],
        "current_target_mode": state.get("target_mode") or "",
        "current_workflow_mode": state.get("workflow_mode") or "",
        "current_chain": (state.get("chain_identity") or {}).get("canonical") or "",
    }


def _rpc_schema_prompt() -> str:
    return (
        "You extract RPC method schemas for AnyChain Benchmark Agent. "
        "Return one JSON object only. Do not explain. "
        "The user may provide a curl command, JSON-RPC request, response sample, docs excerpt, URL text, or endpoint transcript. "
        "First classify the evidence kind and transport before extracting a method. "
        "Do not treat a URL, REST path, or REST documentation title as a JSON-RPC method name. "
        "Infer only what is supported by the evidence; mark unknown fields as unknown. "
        "Do not invent parameters. "
        "Schema: {status:'draft'|'failed', evidence_kind:'jsonrpc_request'|'rest_endpoint'|'rest_path'|'response_sample'|'docs_excerpt'|'method_name'|'mixed'|'unknown', "
        "transport:'jsonrpc'|'rest'|'unknown', method:string, endpoint_url:string, rest_path:string, http_method:string, "
        "params:[{index:number,name:string,json_type:string,semantic_type:string,encoding:string,meaning:string,example:any,required:boolean|null}], "
        "params_json:any, response_summary:string, response_fields:[{name:string,type:string,meaning:string}], "
        "auth_notes:string, rate_limit_notes:string, conflicts:[string], confidence:'low'|'medium'|'high', reason:string, "
        "evidence_summary:string}."
    )


def _rpc_schema_payload(state: AgentGraphState, evidence: str, *, method_hint: str) -> dict[str, Any]:
    identity = state.get("chain_identity") or {}
    return {
        "evidence": evidence,
        "method_hint": method_hint,
        "target_mode": state.get("target_mode") or "",
        "workflow_mode": state.get("workflow_mode") or "",
        "chain": identity.get("canonical") or identity.get("raw") or "",
        "adapter_family": identity.get("adapter_family") or "",
        "web_research": state.get("web_research") or {},
    }


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
        cleaned.append(action)
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
        return {"intent": "unknown", "reason": "model did not return valid JSON", "confidence": "low"}
    return payload if isinstance(payload, dict) else {"intent": "unknown", "reason": "model returned non-object JSON", "confidence": "low"}


def _parse_action_queue(
    text: str,
    *,
    trusted_metadata: bool = False,
    normalize_relations: bool = True,
) -> dict[str, Any]:
    payload = _parse_json_object(text)
    actions_raw = payload.get("actions")
    if isinstance(actions_raw, dict):
        actions_raw = [actions_raw]
    if not isinstance(actions_raw, list):
        action = dict(payload)
        if "type" not in action and "intent" in action:
            action["type"] = action.get("intent")
        actions_raw = [action]
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
    if normalize_relations:
        actions = normalize_action_relations(actions)
    return {
        "actions": actions,
        "clause_coverage": payload.get("clause_coverage") or [],
        "pending_choice_contracts": payload.get("pending_choice_contracts") or [],
        "reason": payload.get("reason") or payload.get("reasoning") or "",
    }
