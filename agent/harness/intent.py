"""LLM intent resolver for free-form Harness turns."""

from __future__ import annotations

import json
import re
from typing import Any

from ..llm.providers import provider_from_config
from ..llm.types import LLMTurnTimeoutError, LLMMessage, LLMRequest, ensure_turn_active
from ..onboarding.families import SUPPORTED_FAMILIES
from .action_registry import (
    ACTION_BY_TYPE,
    ACTION_SPECS,
    CONSULTATION_TOPICS,
    validate_action_contract,
)
from .context import action_schema, build_action_resolver_prompt, group_schema, workflow_snapshot
from .domains.environment import extract_structured_input_candidates
from .input_values import target_mode_evidence_matches
from .plan_coverage import PlanCoverageResult, TurnClause, segment_user_turn, validate_plan_coverage
from .questions import answer_fits_pending, exact_answer, pending_option_value_exists
from .state import DEFAULT_GROUP_ORDER, AgentGraphState

ALLOWED_GROUPS = list(DEFAULT_GROUP_ORDER)

# Single source of truth for the supported adapter families (audit Finding C3):
# derive prompt/payload copies from `onboarding.families.SUPPORTED_FAMILIES`.
ADAPTER_FAMILIES = list(SUPPORTED_FAMILIES)
# The identity/hint resolvers additionally accept these two non-family answers.
ADAPTER_FAMILY_HINT_ENUM = "|".join(ADAPTER_FAMILIES + ["unsupported", "unknown"])

ALLOWED_ACTION_TYPES = [spec.action_type for spec in ACTION_SPECS]


def resolve_action_queue(state: AgentGraphState, text: str) -> dict[str, Any]:
    """Return ordered typed action proposals for a free-form user turn."""

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
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=_action_queue_prompt()),
                    LLMMessage(role="user", content=json.dumps(request_payload, ensure_ascii=False, sort_keys=True)),
                ],
                temperature=0.0,
                max_tokens=1800,
            )
        )
        raw_response = str(response.text or "")
        raw_response = _recover_omitted_chain_selection(provider, raw_response, state)
        raw_response = _apply_state_plan_policy(raw_response, state)
        raw_response = _reconcile_structured_candidate_ownership(raw_response, clauses)
        raw_response, _ = _adjudicate_pending_action_ownership(provider, raw_response, state, text)
        raw_response, _ = _adjudicate_group_navigation_actions(provider, raw_response, state)
        raw_response, _ = _adjudicate_chain_selection_actions(provider, raw_response, state)
        raw_response, _ = _adjudicate_target_mode_actions(provider, raw_response)
        raw_response = _reconstruct_missing_semantic_units(
            provider,
            raw_response,
            clauses,
        )
        validation = _validate_action_document(raw_response, clauses, state)
        if validation.valid:
            validation = _validate_semantic_fulfillment(
                provider,
                raw_response,
                clauses,
                state,
            )
        raw_response, validation = _recover_and_validate_semantic_actions(
            provider,
            raw_response,
            clauses,
            state,
            text,
            validation,
        )
        if not validation.valid:
            ensure_turn_active()
            repair = provider.complete(
                LLMRequest(
                    messages=[
                        LLMMessage(role="system", content=_action_plan_repair_prompt()),
                        LLMMessage(
                            role="user",
                            content=json.dumps(
                                {
                                    "invalid_output": raw_response,
                                    "validation_errors": list(validation.errors),
                                    "action_schema": action_schema(),
                                    "original_request": request_payload,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        ),
                    ],
                    temperature=0.0,
                    max_tokens=1800,
                )
            )
            raw_response = str(repair.text or "")
            raw_response = _reconstruct_missing_semantic_units(
                provider,
                raw_response,
                clauses,
            )
            raw_response = _recover_omitted_chain_selection(provider, raw_response, state)
            raw_response = _apply_state_plan_policy(raw_response, state)
            raw_response = _reconcile_structured_candidate_ownership(raw_response, clauses)
            raw_response, _ = _adjudicate_pending_action_ownership(provider, raw_response, state, text)
            raw_response, _ = _adjudicate_group_navigation_actions(provider, raw_response, state)
            raw_response, _ = _adjudicate_chain_selection_actions(provider, raw_response, state)
            raw_response, _ = _adjudicate_target_mode_actions(provider, raw_response)
            validation = _validate_action_document(raw_response, clauses, state)
            if validation.valid:
                validation = _validate_semantic_fulfillment(
                    provider,
                    raw_response,
                    clauses,
                    state,
                )
            raw_response, validation = _recover_and_validate_semantic_actions(
                provider,
                raw_response,
                clauses,
                state,
                text,
                validation,
            )
        if not validation.valid:
            unresolved = validation.unresolved_clauses or tuple(clause.text for clause in clauses)
            return {
                "actions": [{
                    "type": "clarify_unresolved",
                    "clauses": list(unresolved),
                    "reason": "; ".join(validation.errors) or "incomplete clause coverage",
                    "confidence": "high",
                }],
                "reason": "resolver returned an incomplete clause plan",
                "coverage_errors": list(validation.errors),
                "unresolved_clauses": list(unresolved),
            }
        raw_response = _attach_semantic_admission_receipts(raw_response)
        return _parse_action_queue(raw_response, trusted_metadata=True)
    except LLMTurnTimeoutError:
        raise
    except Exception as exc:
        return {
            "actions": [{"type": "unknown", "reason": f"action queue resolver failed: {type(exc).__name__}", "confidence": "low"}],
            "reason": "resolver failed",
        }


def _recover_and_validate_semantic_actions(
    provider: Any,
    plan_text: str,
    clauses: tuple[TurnClause, ...],
    state: AgentGraphState,
    user_text: str,
    validation: PlanCoverageResult,
) -> tuple[str, PlanCoverageResult]:
    """Converge one plan document through the bounded control recovery gate."""

    if validation.valid:
        return plan_text, validation
    plan_text, decomposed = _decompose_unresolved_semantic_units(
        provider,
        plan_text,
        clauses,
    )
    if decomposed:
        validation = _validate_action_document(plan_text, clauses, state)
    recovered, changed = _recover_registry_bounded_semantic_actions(
        provider,
        plan_text,
        clauses,
        state,
        user_text,
        validation,
    )
    if not changed:
        return plan_text, validation
    recovered, _ = _adjudicate_pending_action_ownership(provider, recovered, state, user_text)
    recovered, _ = _adjudicate_group_navigation_actions(provider, recovered, state)
    recovered, _ = _adjudicate_chain_selection_actions(provider, recovered, state)
    recovered, _ = _adjudicate_target_mode_actions(provider, recovered)
    recovered_validation = _validate_action_document(recovered, clauses, state)
    if recovered_validation.valid:
        recovered_validation = _validate_semantic_fulfillment(
            provider,
            recovered,
            clauses,
            state,
        )
    return recovered, recovered_validation


def _decompose_unresolved_semantic_units(
    provider: Any,
    plan_text: str,
    clauses: tuple[TurnClause, ...],
) -> tuple[str, bool]:
    """Partition compound unresolved prose before owner-domain recovery.

    This stage identifies independently actionable source spans only. It has no
    action schema and cannot select a group, action, value, or workflow path.
    Every proposed split must remain a complete, ordered, exact partition of
    the original unresolved unit before the registry-bounded recovery stage can
    see it.
    """

    payload = _parse_json_object(plan_text)
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    candidates = [
        unit
        for unit in units
        if isinstance(unit, dict)
        and str(unit.get("disposition") or "") == "unresolved"
        and not (unit.get("action_indexes") if isinstance(unit.get("action_indexes"), list) else [])
        and str(unit.get("source_text") or "").strip()
    ]
    if not candidates:
        return plan_text, False

    replacements: dict[str, list[dict[str, Any]]] = {}
    for _attempt in range(2):
        replacements = _request_validated_semantic_partitions(provider, candidates)
        if replacements:
            break
    if not replacements:
        return plan_text, False

    expanded: list[dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        expanded.extend(replacements.get(str(unit.get("unit_id") or ""), [unit]))
    candidate = dict(payload)
    candidate["semantic_units"] = expanded
    coverage = validate_plan_coverage(candidate, clauses)
    if coverage.errors:
        return plan_text, False
    return json.dumps(candidate, ensure_ascii=False, sort_keys=True), True


def _request_validated_semantic_partitions(
    provider: Any,
    candidates: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Request one partition candidate and admit only exact source proofs."""

    ensure_turn_active()
    response = provider.complete(LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "Partition unresolved AnyChain prose into independently actionable semantic source units. "
                    "Return JSON only: {partitions:[{unit_id:string,source_units:[string],reason:string}]}. "
                    "Review every supplied unit exactly once. Split only when the source contains multiple "
                    "independent present requests, selections, mutations, questions, navigation demands, evidence "
                    "submissions, or execution instructions. Keep one full source unit when it expresses one demand "
                    "or cannot be split safely. Every source_units entry must be a non-empty exact contiguous excerpt "
                    "of source_text, in source order. Together the excerpts must cover all words; punctuation and "
                    "whitespace between adjacent excerpts may remain between anchors. Do not classify, route, answer, "
                    "infer defaults, name internal actions, or invent text."
                ),
            ),
            LLMMessage(
                role="user",
                content=json.dumps({
                    "units": [
                        {
                            "unit_id": str(unit.get("unit_id") or ""),
                            "source_text": str(unit.get("source_text") or ""),
                        }
                        for unit in candidates
                    ],
                }, ensure_ascii=False, sort_keys=True),
            ),
        ],
        temperature=0.0,
        max_tokens=900,
    ))
    result = _parse_json_object(response.text)
    partitions = result.get("partitions") if isinstance(result.get("partitions"), list) else []
    by_unit = {
        str(row.get("unit_id") or ""): row
        for row in partitions
        if isinstance(row, dict) and str(row.get("unit_id") or "")
    }

    replacements: dict[str, list[dict[str, Any]]] = {}
    for unit in candidates:
        unit_id = str(unit.get("unit_id") or "")
        source = str(unit.get("source_text") or "")
        row = by_unit.get(unit_id) or {}
        anchors = row.get("source_units") if isinstance(row.get("source_units"), list) else []
        anchors = [str(anchor) for anchor in anchors if str(anchor)]
        if len(anchors) < 2:
            continue
        local_clause = TurnClause("unresolved-source", source, "prose")
        local_units = [
            {
                "unit_id": f"{unit_id}.part-{index}",
                "clause_id": local_clause.clause_id,
                "source_text": anchor,
                "disposition": "unresolved",
                "action_indexes": [],
                "reason": "independent demand awaiting registry-bounded recovery",
            }
            for index, anchor in enumerate(anchors, start=1)
        ]
        partition_check = validate_plan_coverage(
            {"actions": [], "semantic_units": local_units},
            (local_clause,),
        )
        if partition_check.errors:
            continue
        replacements[unit_id] = [
            {**partition_unit, "clause_id": str(unit.get("clause_id") or "")}
            for partition_unit in local_units
        ]
    return replacements


def _apply_state_plan_policy(text: str, state: AgentGraphState) -> str:
    """Remove actions that are impossible in the current product case.

    This is plan admission, not routing. It keeps semantic-unit coverage by
    remapping surviving action indexes and marks a unit unresolved only when
    every action that represented it was removed.
    """

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
) -> str:
    """Complete syntax-owned facts after the model selects config intent.

    The model decides whether a structured clause is configuration at all.
    Once it maps that clause to ``propose_config_values``, the deterministic
    parser owns lossless transfer of recognized and unmapped assignments. This
    keeps logs/examples outside the configuration path while preventing model
    variance from dropping one key inside an admitted config transaction.
    """

    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    if not actions or not units:
        return text

    changed = False
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
        proposal_indexes = {
            index
            for unit in clause_units
            for index in (unit.get("action_indexes") if isinstance(unit.get("action_indexes"), list) else [])
            if isinstance(index, int)
            and 0 <= index < len(actions)
            and isinstance(actions[index], dict)
            and str(actions[index].get("type") or "") == "propose_config_values"
        }
        if not proposal_indexes:
            continue

        config_values = dict(candidates.get("config_values") or {})
        unmapped_values = dict(candidates.get("unmapped_values") or {})
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


def _reconstruct_missing_semantic_units(
    provider: Any,
    text: str,
    clauses: tuple[TurnClause, ...],
) -> str:
    """Rebuild only omitted coverage metadata around an immutable action list."""

    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    units = payload.get("semantic_units")
    if not actions:
        if isinstance(units, list) and units:
            return text
        payload["actions"] = []
        payload["semantic_units"] = [
            {
                "unit_id": f"{clause.clause_id}-unresolved",
                "clause_id": clause.clause_id,
                "source_text": clause.text,
                "disposition": "unresolved",
                "action_indexes": [],
                "reason": "authoritative source clause omitted by the planner",
            }
            for clause in clauses
        ]
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if isinstance(units, list) and units:
        coverage = validate_plan_coverage(payload, clauses)
        reconstructable_prefixes = (
            "semantic unit has no unit_id",
            "semantic unit has an empty source anchor",
            "missing semantic units for ",
        )
        if not any(
            error.startswith(reconstructable_prefixes)
            for error in coverage.errors
        ):
            return text
    request_payload = {
        "actions": actions,
        "clauses": [clause.as_dict() for clause in clauses],
    }
    for _attempt in range(2):
        ensure_turn_active()
        response = provider.complete(LLMRequest(
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "Reconstruct only the missing semantic_units for one immutable AnyChain action plan. "
                        "Return one JSON object only: {semantic_units:[{unit_id:string,clause_id:string,"
                        "source_text:string,disposition:'action'|'context'|'unresolved',action_indexes:[integer],reason:string,"
                        "optional scope_constraint:'consultation_only'}]}. Do not return actions. Do not add, remove, "
                        "reorder, rename, reinterpret, or repair any supplied action. Every clause must be covered by "
                        "ordered exact source_text anchors. When conjunctions make a lossless split uncertain, use "
                        "one full-clause unit mapped to every supplied action that preserves part of it. Keep structured "
                        "clauses atomic. Use context only for prose with no present request, answer, question, selection, "
                        "mutation, navigation, or execution instruction; context has no action indexes and structured "
                        "input can never be context. Use unresolved with no action indexes only when the immutable actions "
                        "cannot preserve an explicit demand. Do not calculate character offsets."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=json.dumps(request_payload, ensure_ascii=False, sort_keys=True),
                ),
            ],
            temperature=0.0,
            max_tokens=1000,
        ))
        reconstructed = _parse_json_object(response.text)
        recovered_units = reconstructed.get("semantic_units")
        if not isinstance(recovered_units, list) or not recovered_units:
            continue
        candidate = dict(payload)
        candidate["semantic_units"] = recovered_units
        if not any(
            error.startswith("missing semantic units for ")
            for error in validate_plan_coverage(candidate, clauses).errors
        ):
            return json.dumps(candidate, ensure_ascii=False, sort_keys=True)
    return text


def _validate_action_document(
    text: str,
    clauses: tuple[TurnClause, ...],
    state: AgentGraphState | None = None,
):
    payload = _parse_json_object(text)
    if not isinstance(payload.get("actions"), list):
        return validate_plan_coverage({}, clauses)
    action_errors: list[str] = []
    for index, raw in enumerate(payload["actions"]):
        if not isinstance(raw, dict):
            action_errors.append(f"action {index} is not an object")
            continue
        try:
            validate_action_contract(raw)
        except ValueError as exc:
            action_errors.append(f"action {index} contract invalid: {exc}")
            continue
        if str(raw.get("type") or "") != "answer_pending" or state is None:
            continue
        pending = dict(state.get("pending_question") or {})
        if not pending:
            action_errors.append(f"action {index} answer_pending has no active question")
            continue
        if not pending.get("options"):
            continue
        selected = raw.get("selected_value")
        if pending_option_value_exists(selected, pending):
            continue
        answer = str(raw.get("answer") or "").strip()
        if pending.get("manual_input_allowed") is True and answer_fits_pending(answer, pending):
            continue
        matches, _ = exact_answer(answer, pending)
        if not matches:
            action_errors.append(
                f"action {index} answer_pending does not select an exact declared option "
                f"for pending question {pending.get('id')!r}"
            )
    coverage = validate_plan_coverage(payload, clauses)
    if not action_errors:
        return coverage
    return PlanCoverageResult(
        valid=False,
        errors=tuple([*action_errors, *coverage.errors]),
        unresolved_clauses=coverage.unresolved_clauses,
    )


def _validate_semantic_fulfillment(
    provider: Any,
    text: str,
    clauses: tuple[TurnClause, ...],
    state: AgentGraphState,
) -> PlanCoverageResult:
    """Fail closed when a high-risk action does not fulfil its source unit.

    Structural coverage proves that source text was accounted for, but cannot
    prove that a durable action is the operation the user requested. This
    risk-gated adjudication belongs to plan admission; it neither routes nor
    mutates product state.
    """

    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    chain_admissions = {
        int(index)
        for index in payload.get("chain_selection_admissions", [])
        if isinstance(index, int) and 0 <= index < len(actions)
    }
    target_mode_admissions = {
        int(index)
        for index in payload.get("target_mode_selection_admissions", [])
        if isinstance(index, int) and 0 <= index < len(actions)
    }
    pending_admissions = {
        int(index)
        for index in payload.get("pending_answer_admissions", [])
        if isinstance(index, int) and 0 <= index < len(actions)
    }
    risky_indexes = [
        index
        for index, action in enumerate(actions)
        if isinstance(action, dict)
        and _requires_semantic_fulfillment_review(action)
        and index not in chain_admissions
        and index not in target_mode_admissions
        and index not in pending_admissions
    ]
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    context_units = [
        unit
        for unit in units
        if isinstance(unit, dict) and str(unit.get("disposition") or "") == "context"
    ]
    action_units_per_clause: dict[str, int] = {}
    for unit in units:
        if not isinstance(unit, dict) or str(unit.get("disposition") or "") != "action":
            continue
        clause_id = str(unit.get("clause_id") or "")
        action_units_per_clause[clause_id] = action_units_per_clause.get(clause_id, 0) + 1
    mutation_units: list[dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        indexes = unit.get("action_indexes") if isinstance(unit.get("action_indexes"), list) else []
        mapped = [
            actions[index]
            for index in indexes
            if isinstance(index, int) and 0 <= index < len(actions) and isinstance(actions[index], dict)
        ]
        if len(mapped) == 1 and str(mapped[0].get("type") or "") == "change_group":
            # Navigation is complete at this boundary; the destination group
            # owns its subsequent field intake.
            continue
        if (
            action_units_per_clause.get(str(unit.get("clause_id") or ""), 0) == 1
            and any(
                (ACTION_BY_TYPE.get(str(action.get("type") or "")) is not None)
                and ACTION_BY_TYPE[str(action.get("type") or "")].lifetime == "durable"
                for action in mapped
            )
        ):
            mutation_units.append(unit)
    if not risky_indexes and not mutation_units and not context_units:
        return validate_plan_coverage(payload, clauses)

    reviews_input: list[dict[str, Any]] = []
    for index in risky_indexes:
        action = actions[index]
        action_type = str(action.get("type") or "")
        spec = ACTION_BY_TYPE[action_type]
        source_units = [
            str(unit.get("source_text") or "")
            for unit in units
            if isinstance(unit, dict)
            and index in (unit.get("action_indexes") if isinstance(unit.get("action_indexes"), list) else [])
        ]
        reviews_input.append({
            "action_index": index,
            "operation_arguments": _semantic_operation_arguments(action),
            "declared_purpose": _semantic_action_purpose(action, spec.purpose, state),
            "source_units": source_units,
        })
    unit_reviews_input: list[dict[str, Any]] = []
    for unit in mutation_units:
        indexes = [
            index
            for index in unit.get("action_indexes") or []
            if isinstance(index, int) and 0 <= index < len(actions)
        ]
        index_set = set(indexes)
        unit_id = str(unit.get("unit_id") or "")
        unit_reviews_input.append({
            "unit_id": unit_id,
            "source_text": str(unit.get("source_text") or ""),
            "related_source_units": [
                {
                    "unit_id": str(related.get("unit_id") or ""),
                    "source_text": str(related.get("source_text") or ""),
                    "input_shape": str(related.get("input_shape") or "prose"),
                }
                for related in units
                if isinstance(related, dict)
                and str(related.get("unit_id") or "") != unit_id
                and index_set.intersection(
                    index
                    for index in (
                        related.get("action_indexes")
                        if isinstance(related.get("action_indexes"), list)
                        else []
                    )
                    if isinstance(index, int)
                )
            ],
            "mapped_actions": [
                {
                    "operation_index": index,
                    "operation_arguments": _semantic_operation_arguments(actions[index]),
                    "declared_purpose": _semantic_action_purpose(
                        actions[index],
                        ACTION_BY_TYPE[str(actions[index].get("type") or "")].purpose,
                        state,
                    ),
                }
                for index in indexes
                if str(actions[index].get("type") or "") in ACTION_BY_TYPE
            ],
        })

    reviewed_pending: dict[str, Any] = {}
    if context_units or any(
        str(actions[index].get("type") or "") == "answer_pending"
        or _matching_pending_option(actions[index], state)
        for index in risky_indexes
    ):
        pending = dict(state.get("pending_question") or {})
        reviewed_pending = {
            "id": str(pending.get("id") or ""),
            "field": str(pending.get("field") or ""),
            "kind": str(pending.get("kind") or ""),
            "options": [
                {
                    "id": str(option.get("id") or ""),
                    "label": str(option.get("label") or ""),
                    "value": option.get("value"),
                }
                for option in pending.get("options") or []
                if isinstance(option, dict)
            ],
        }

    result = _request_semantic_fulfillment_review(
        provider,
        reviews=reviews_input,
        unit_reviews=unit_reviews_input,
        context_reviews=[{
            "unit_id": str(unit.get("unit_id") or ""),
            "source_text": str(unit.get("source_text") or ""),
            "input_shape": next(
                (
                    clause.input_shape
                    for clause in clauses
                    if clause.clause_id == str(unit.get("clause_id") or "")
                ),
                "",
            ),
            "planner_reason": str(unit.get("reason") or ""),
        } for unit in context_units],
        pending_question=reviewed_pending,
    )
    rows = result.get("reviews") if isinstance(result.get("reviews"), list) else []
    rows = _adjudicate_rejected_action_reviews(
        provider,
        action_reviews=reviews_input,
        first_rows=rows,
    )
    by_index = {
        row.get("action_index"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("action_index"), int)
    }
    errors: list[str] = []
    for index in risky_indexes:
        row = by_index.get(index)
        if not row or not isinstance(row.get("supported"), bool):
            errors.append(f"semantic fulfilment review missing for action {index}")
            continue
        if not row["supported"]:
            reason = str(row.get("reason") or "declared purpose is not supported by its source unit").strip()
            errors.append(f"action {index} semantic fulfilment failed: {reason}")
    unit_rows = result.get("unit_reviews") if isinstance(result.get("unit_reviews"), list) else []
    unit_rows = _adjudicate_incomplete_unit_reviews(
        provider,
        unit_reviews=unit_reviews_input,
        first_rows=unit_rows,
    )
    by_unit_id = {
        str(row.get("unit_id") or ""): row
        for row in unit_rows
        if isinstance(row, dict) and str(row.get("unit_id") or "")
    }
    for review in unit_reviews_input:
        unit_id = str(review.get("unit_id") or "")
        row = by_unit_id.get(unit_id)
        if not row or not isinstance(row.get("complete"), bool):
            errors.append(f"semantic unit fulfilment review missing for {unit_id or '<missing>'}")
            continue
        if not row["complete"]:
            reason = str(row.get("reason") or "mapped actions omit an explicit request").strip()
            errors.append(f"semantic unit {unit_id} fulfilment failed: {reason}")
    context_rows = result.get("context_reviews") if isinstance(result.get("context_reviews"), list) else []
    context_by_unit_id = {
        str(row.get("unit_id") or ""): row
        for row in context_rows
        if isinstance(row, dict) and str(row.get("unit_id") or "")
    }
    for unit in context_units:
        unit_id = str(unit.get("unit_id") or "")
        row = context_by_unit_id.get(unit_id)
        if not row or not isinstance(row.get("context_only"), bool):
            errors.append(f"context admission review missing for {unit_id or '<missing>'}")
            continue
        if not row["context_only"]:
            reason = str(row.get("reason") or "source contains a current actionable demand").strip()
            errors.append(f"context semantic unit {unit_id} admission failed: {reason}")
    if errors:
        return PlanCoverageResult(
            valid=False,
            errors=tuple(errors),
            unresolved_clauses=(),
            rejected_action_indexes=tuple(
                index
                for index in risky_indexes
                if not by_index.get(index) or by_index[index].get("supported") is not True
            ),
            incomplete_unit_ids=tuple(
                str(review.get("unit_id") or "")
                for review in unit_reviews_input
                if not by_unit_id.get(str(review.get("unit_id") or ""))
                or by_unit_id[str(review.get("unit_id") or "")].get("complete") is not True
            ),
        )
    coverage = validate_plan_coverage(payload, clauses)
    return PlanCoverageResult(
        valid=coverage.valid,
        errors=coverage.errors,
        unresolved_clauses=coverage.unresolved_clauses,
    )


def _request_semantic_fulfillment_review(
    provider: Any,
    *,
    reviews: list[dict[str, Any]],
    unit_reviews: list[dict[str, Any]],
    context_reviews: list[dict[str, Any]] | None = None,
    pending_question: dict[str, Any],
) -> dict[str, Any]:
    """Run independent, bounded audits for actions and compound units.

    These are separate admission gates. Combining them lets an unsolicited or
    oversized result for one contract invalidate otherwise valid evidence from
    the other contract.
    """

    result: dict[str, Any] = {"reviews": [], "unit_reviews": [], "context_reviews": []}
    if reviews:
        result["reviews"] = _request_bounded_semantic_rows(
            provider,
            review_kind="actions",
            rows=reviews,
            pending_question=pending_question,
        )

    if unit_reviews:
        result["unit_reviews"] = _request_bounded_semantic_rows(
            provider,
            review_kind="units",
            rows=unit_reviews,
            pending_question={},
        )
    if context_reviews:
        result["context_reviews"] = _request_bounded_semantic_rows(
            provider,
            review_kind="contexts",
            rows=context_reviews,
            pending_question=pending_question,
        )
    return result


def _request_bounded_semantic_rows(
    provider: Any,
    *,
    review_kind: str,
    rows: list[dict[str, Any]],
    pending_question: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return schema-valid rows, retrying only missing IDs once."""

    is_action_review = review_kind == "actions"
    is_context_review = review_kind == "contexts"
    payload_key = "reviews" if is_action_review else ("context_reviews" if is_context_review else "unit_reviews")
    id_key = "action_index" if is_action_review else "unit_id"
    verdict_key = "supported" if is_action_review else ("context_only" if is_context_review else "complete")
    requested = {row.get(id_key): row for row in rows}
    accepted: dict[Any, dict[str, Any]] = {}

    for _attempt in range(2):
        missing = [key for key in requested if key not in accepted]
        if not missing:
            break
        ensure_turn_active()
        request_rows = [requested[key] for key in missing]
        request_payload: dict[str, Any] = {payload_key: request_rows}
        if is_action_review or is_context_review:
            request_payload["pending_question"] = pending_question
        response = provider.complete(LLMRequest(
            messages=[
                LLMMessage(
                    role="system",
                    content=_semantic_fulfillment_prompt(review_kind=review_kind),
                ),
                LLMMessage(
                    role="user",
                    content=json.dumps(request_payload, ensure_ascii=False, sort_keys=True),
                ),
            ],
            temperature=0.0,
            max_tokens=800,
        ))
        payload = _parse_json_object(response.text)
        returned = payload.get(payload_key) if isinstance(payload.get(payload_key), list) else []
        counts: dict[Any, int] = {}
        candidates: dict[Any, dict[str, Any]] = {}
        for item in returned:
            if not isinstance(item, dict):
                continue
            row_id = item.get(id_key)
            if row_id not in requested or not isinstance(item.get(verdict_key), bool):
                continue
            counts[row_id] = counts.get(row_id, 0) + 1
            candidates[row_id] = item
        for row_id, count in counts.items():
            if count == 1:
                accepted[row_id] = candidates[row_id]

    return [accepted[row.get(id_key)] for row in rows if row.get(id_key) in accepted]


def _adjudicate_incomplete_unit_reviews(
    provider: Any,
    *,
    unit_reviews: list[dict[str, Any]],
    first_rows: list[Any],
) -> list[Any]:
    """Recheck negative unit verdicts under an exact-quote rejection contract."""

    review_by_id = {
        str(item.get("unit_id") or ""): item
        for item in unit_reviews
        if isinstance(item, dict) and str(item.get("unit_id") or "")
    }
    negative_rows = [
        row
        for row in first_rows
        if isinstance(row, dict)
        and row.get("complete") is False
        and str(row.get("unit_id") or "") in review_by_id
    ]
    if not negative_rows:
        return first_rows

    response = provider.complete(LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "Adjudicate only the supplied negative compound-unit verdicts. Return JSON only: "
                    "{unit_reviews:[{unit_id:string,complete:boolean,missing_demand_quote:string,reason:string}]}. "
                    "Review every row exactly once using source_text, related_source_units, mapped_actions, and the "
                    "same completeness contract. If mapped_actions collectively preserve every independently "
                    "actionable demand, complete must be true and missing_demand_quote must be empty. If incomplete, "
                    "missing_demand_quote must be the shortest non-empty exact substring of source_text that states "
                    "one omitted selection, mutation, consultation, navigation, correction, value, or evidence "
                    "demand. Framing, provenance, formatting, and an explicit instruction not to invent unspecified "
                    "values are not omitted demands when mapped operations preserve the related request and supplied "
                    "values. Do not invent, repair, rename, or add operations."
                ),
            ),
            LLMMessage(role="user", content=json.dumps({
                "unit_reviews": [review_by_id[str(row.get("unit_id") or "")] for row in negative_rows],
                "invalid_first_verdicts": negative_rows,
            }, ensure_ascii=False, sort_keys=True)),
        ],
        temperature=0.0,
        max_tokens=600,
    ))
    result = _parse_json_object(response.text)
    adjudicated = result.get("unit_reviews") if isinstance(result.get("unit_reviews"), list) else []
    by_id = {
        str(row.get("unit_id") or ""): row
        for row in adjudicated
        if isinstance(row, dict) and isinstance(row.get("complete"), bool)
    }
    replacements: dict[str, dict[str, Any]] = {}
    for first in negative_rows:
        unit_id = str(first.get("unit_id") or "")
        row = by_id.get(unit_id)
        if not row:
            continue
        quote = str(row.get("missing_demand_quote") or "")
        source = str(review_by_id[unit_id].get("source_text") or "")
        if row["complete"] is True or (quote and quote in source):
            replacements[unit_id] = row
    return [
        replacements.get(str(row.get("unit_id") or ""), row)
        if isinstance(row, dict)
        else row
        for row in first_rows
    ]


def _adjudicate_rejected_action_reviews(
    provider: Any,
    *,
    action_reviews: list[dict[str, Any]],
    first_rows: list[Any],
) -> list[Any]:
    """Recheck negative purpose verdicts without changing proposed actions."""

    review_by_index = {
        int(item["action_index"]): item
        for item in action_reviews
        if isinstance(item, dict) and isinstance(item.get("action_index"), int)
    }
    negative_rows = [
        row
        for row in first_rows
        if isinstance(row, dict)
        and row.get("supported") is False
        and isinstance(row.get("action_index"), int)
        and int(row["action_index"]) in review_by_index
    ]
    if not negative_rows:
        return first_rows

    response = provider.complete(LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "Adjudicate only the supplied negative action-purpose verdicts. Return JSON only: "
                    "{reviews:[{action_index:integer,supported:boolean,reason:string}]}. Review every row exactly "
                    "once. Operations are immutable opaque registered actions. Decide only whether source_units "
                    "semantically and explicitly support declared_purpose with the supplied operation_arguments. "
                    "Apply the declared purpose literally, including distinctions it states between navigation, "
                    "consultation, intake, and mutation. Do not infer a stronger operation from a general word when "
                    "the source supplies no value, default change, or other specific mutation required by the "
                    "declared purpose. Equally, do not accept topical discussion, help, uncertainty, or workflow "
                    "state as evidence for an unsupported operation. Never repair, replace, rename, reorder, or add "
                    "an operation."
                ),
            ),
            LLMMessage(role="user", content=json.dumps({
                "reviews": [review_by_index[int(row["action_index"])] for row in negative_rows],
                "negative_first_verdicts": negative_rows,
            }, ensure_ascii=False, sort_keys=True)),
        ],
        temperature=0.0,
        max_tokens=450,
    ))
    result = _parse_json_object(response.text)
    adjudicated = result.get("reviews") if isinstance(result.get("reviews"), list) else []
    counts: dict[int, int] = {}
    candidates: dict[int, dict[str, Any]] = {}
    for row in adjudicated:
        if not isinstance(row, dict) or not isinstance(row.get("action_index"), int):
            continue
        index = int(row["action_index"])
        if index not in review_by_index or not isinstance(row.get("supported"), bool):
            continue
        counts[index] = counts.get(index, 0) + 1
        candidates[index] = row
    replacements = {
        index: candidates[index]
        for index, count in counts.items()
        if count == 1
    }
    return [
        replacements.get(int(row.get("action_index")), row)
        if isinstance(row, dict) and isinstance(row.get("action_index"), int)
        else row
        for row in first_rows
    ]


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
        selected = action.get("selected_value")
        return next((item for item in options if item.get("value") == selected), {})
    for option in options:
        declared = option.get("action")
        if not isinstance(declared, dict) or str(declared.get("type") or "") != action_type:
            continue
        if all(
            key == "type" or action.get(key) == value
            for key, value in declared.items()
        ):
            return option
    return {}


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
    }
    return {
        key: value
        for key, value in action.items()
        if key not in internal_fields
    }


def _attach_semantic_admission_receipts(text: str) -> str:
    """Mark actions whose purpose passed the final semantic admission review.

    This helper is called only after structural and semantic validation have
    both succeeded. The marker is trusted runtime metadata, so model-produced
    documents cannot supply it through the normal action contract.
    """

    payload = _parse_json_object(text)
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
    for index, action in enumerate(actions):
        if isinstance(action, dict) and _requires_semantic_fulfillment_review(action):
            action["semantic_purpose_verified"] = True
        if isinstance(action, dict) and index in pending_admissions:
            action["pending_option_semantic_verified"] = True
        if isinstance(action, dict) and index in chain_admissions:
            action["chain_selection_semantic_verified"] = True
        if isinstance(action, dict) and index in target_mode_admissions:
            action["target_mode_semantic_verified"] = True
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _adjudicate_target_mode_actions(provider: Any, text: str) -> tuple[str, bool]:
    """Retain explicit workflow selections and stage unresolved mode intake.

    This adjudicator is deliberately separate from general action-purpose
    review. It receives no workflow state from which it could infer a default.
    """

    payload = _parse_json_object(text)
    payload.pop("target_mode_selection_admissions", None)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    semantic_units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    authoritative_sources = {
        index: "\n".join(
            part
            for part in (
                str(action.get("source_evidence") or ""),
                *(
                    str(unit.get("source_text") or "")
                    for unit in semantic_units
                    if isinstance(unit, dict)
                    and index in (
                        unit.get("action_indexes")
                        if isinstance(unit.get("action_indexes"), list)
                        else []
                    )
                ),
            )
            if part
        )
        for index, action in enumerate(actions)
        if isinstance(action, dict)
    }
    review_indexes = [
        index
        for index, action in enumerate(actions)
        if isinstance(action, dict)
        and str(action.get("type") or "") in {"choose_target_mode", "queue_workflow_goal"}
        and not target_mode_evidence_matches(
            action.get("target_mode"),
            action.get("source_evidence"),
            authoritative_sources.get(index, ""),
        )
    ]
    if not review_indexes:
        return text, False
    response = provider.complete(LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "Verify AnyChain target-mode selection facts. Return JSON only: "
                    "{reviews:[{action_index:integer,explicit:boolean,evidence_quote:string,"
                    "unresolved_selection:boolean,unresolved_quote:string,reason:string}]}. "
                    "Review every row once and judge both facts independently. explicit=true only when "
                    "source_evidence itself selects the proposed "
                    "fake-node, real-node, or sync-observe workflow, including a clear natural-language equivalent. "
                    "unresolved_selection=true only when source_evidence has a benchmark or synchronization-observation "
                    "goal but explicitly leaves the target mode undecided, including uncertainty between named modes. "
                    "A generic request to test/benchmark a chain, select RPC/QPS/monitoring settings, run quickly, "
                    "avoid risk, or receive a recommendation does not select a target mode. Do not infer a default "
                    "from missing endpoints or other workflow state. Quotes must be the shortest exact excerpts that "
                    "express their facts; use an empty quote for each false fact."
                ),
            ),
            LLMMessage(role="user", content=json.dumps({
                "reviews": [
                    {
                        "action_index": index,
                        "action_type": actions[index].get("type"),
                        "proposed_target_mode": actions[index].get("target_mode"),
                        "source_evidence": actions[index].get("source_evidence"),
                        "semantic_source_units": [
                            str(unit.get("source_text") or "")
                            for unit in semantic_units
                            if isinstance(unit, dict)
                            and index in (
                                unit.get("action_indexes")
                                if isinstance(unit.get("action_indexes"), list)
                                else []
                            )
                        ],
                    }
                    for index in review_indexes
                ],
            }, ensure_ascii=False, sort_keys=True)),
        ],
        temperature=0.0,
        max_tokens=350,
    ))
    result = _parse_json_object(response.text)
    rows = result.get("reviews") if isinstance(result.get("reviews"), list) else []
    by_index = {
        row.get("action_index"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("action_index"), int)
    }
    rejected: dict[int, str] = {}
    actions = list(actions)
    changed = False
    admitted_indexes: set[int] = set()
    for index in review_indexes:
        row = by_index.get(index)
        action_type = str(actions[index].get("type") or "")
        source = authoritative_sources.get(index, "")
        quote = str((row or {}).get("evidence_quote") or "").strip()
        unresolved_quote = str((row or {}).get("unresolved_quote") or "").strip()
        explicit = bool(row and row.get("explicit") is True and quote and quote in source)
        unresolved = bool(
            row
            and row.get("unresolved_selection") is True
            and unresolved_quote
            and unresolved_quote in source
        )
        if explicit and not unresolved:
            admitted_indexes.add(index)
            continue
        if unresolved and action_type == "choose_target_mode":
            actions[index] = {
                "type": "request_target_mode_selection",
                "source_evidence": unresolved_quote,
            }
            changed = True
            continue
        rejected[index] = str(
            (row or {}).get("reason") or "no explicit or unresolved target-mode selection in source evidence"
        )
    payload["actions"] = actions
    if rejected:
        repaired, removed = _remove_rejected_action_indexes(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            tuple(rejected),
            reason="; ".join(sorted(set(rejected.values()))),
        )
        return repaired, changed or removed
    if changed:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True), True
    if admitted_indexes:
        payload["target_mode_selection_admissions"] = sorted(admitted_indexes)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True), False
    return text, False


def _adjudicate_pending_answer_actions(
    provider: Any,
    text: str,
    state: AgentGraphState,
    user_text: str,
) -> tuple[str, bool]:
    """Keep semantic menu answers from impersonating exact option matches."""

    pending = dict(state.get("pending_question") or {})
    options = [item for item in pending.get("options") or [] if isinstance(item, dict)]
    if not options:
        return text, False
    exact_match, _value = exact_answer(user_text, pending)
    if exact_match:
        return text, False

    payload = _parse_json_object(text)
    payload.pop("pending_answer_admissions", None)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    review_indexes = [
        index
        for index, action in enumerate(actions)
        if isinstance(action, dict) and str(action.get("type") or "") == "answer_pending"
        and not (
            pending.get("manual_input_allowed") is True
            and not pending_option_value_exists(action.get("selected_value"), pending)
            and answer_fits_pending(str(action.get("answer") or ""), pending)
        )
    ]
    if not review_indexes:
        return text, False

    reviews: list[dict[str, Any]] = []
    invalid_indexes: dict[int, str] = {}
    admitted_indexes: set[int] = set()
    transformed_indexes: set[int] = set()
    option_action_types = {
        str((item.get("action") or {}).get("type") or "")
        for item in options
        if isinstance(item.get("action"), dict)
    }
    target_mode_menu = (
        "choose_target_mode" in option_action_types
        and option_action_types <= {"choose_target_mode", "answer_opening_question"}
    )
    for index in review_indexes:
        selected = actions[index].get("selected_value")
        option = next((item for item in options if item.get("value") == selected), None)
        if option is None and not target_mode_menu:
            invalid_indexes[index] = "the proposed value is not a declared pending option"
            continue
        reviews.append({
            "action_index": index,
            "question": str(pending.get("prompt") or ""),
            "selected_option_label": str((option or {}).get("label") or ""),
            "selected_option_value": selected,
            "allows_unresolved_target_mode": target_mode_menu,
            "source_units": [
                str(unit.get("source_text") or "")
                for unit in units
                if isinstance(unit, dict)
                and index in (
                    unit.get("action_indexes")
                    if isinstance(unit.get("action_indexes"), list)
                    else []
                )
            ],
        })
    if reviews:
        result = _request_pending_answer_reviews(provider, reviews)
        rows = result.get("reviews") if isinstance(result.get("reviews"), list) else []
        rows = _adjudicate_rejected_pending_answer_reviews(
            provider,
            reviews=reviews,
            first_rows=rows,
        )
        by_index = {
            row.get("action_index"): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("action_index"), int)
        }
        for review in reviews:
            index = int(review["action_index"])
            row = by_index.get(index) or {}
            decision = str(row.get("decision") or "").strip()
            quote = str(row.get("evidence_quote") or "").strip()
            sources = review.get("source_units") or []
            unresolved = bool(
                review.get("allows_unresolved_target_mode") is True
                and decision == "unresolved_target_mode"
                and quote
                and quote in user_text
            )
            if unresolved:
                actions[index] = {
                    "type": "request_target_mode_selection",
                    "source_evidence": quote,
                }
                transformed_indexes.add(index)
                continue
            if (
                decision != "select_option"
                or not quote
                or quote not in user_text
            ):
                invalid_indexes[index] = str(
                    row.get("reason") or "source does not select the proposed pending option"
                )
            else:
                option = next(
                    (
                        item
                        for item in options
                        if item.get("value") == actions[index].get("selected_value")
                    ),
                    None,
                )
                declared = dict((option or {}).get("action") or {})
                if str(declared.get("type") or "") in {"", "answer_pending"}:
                    admitted_indexes.add(index)
                    continue
                materialized = _materialize_pending_option_action(declared, quote)
                if materialized is None:
                    invalid_indexes[index] = "the registered pending option effect is not a valid typed action"
                    continue
                actions[index] = materialized
                transformed_indexes.add(index)
    payload["actions"] = actions
    if admitted_indexes:
        payload["pending_answer_admissions"] = sorted(admitted_indexes)
    if not invalid_indexes:
        if admitted_indexes or transformed_indexes:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True), True
        return text, False
    return _remove_rejected_action_indexes(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        tuple(sorted(invalid_indexes)),
        reason="; ".join(sorted(set(invalid_indexes.values()))),
    )


def _materialize_pending_option_action(
    declared: dict[str, Any],
    source_quote: str,
) -> dict[str, Any] | None:
    """Compile one admitted semantic option into its registered typed effect."""

    action_type = str(declared.get("type") or "")
    spec = ACTION_BY_TYPE.get(action_type)
    if spec is None or action_type == "answer_pending":
        return None
    candidate = dict(declared)
    if "source_evidence" in spec.allowed_arguments:
        candidate["source_evidence"] = source_quote
    if "mutation_explicit" in spec.allowed_arguments and "mutation_explicit" not in candidate:
        candidate["mutation_explicit"] = True
    if "target_mode_explicit" in spec.allowed_arguments and "target_mode_explicit" not in candidate:
        candidate["target_mode_explicit"] = True
    try:
        validate_action_contract(candidate)
    except ValueError:
        return None
    return candidate


def _reconcile_pending_owner_mutations(
    provider: Any,
    text: str,
    state: AgentGraphState,
    user_text: str,
) -> tuple[str, bool]:
    """Separate a pending-option answer from a same-group owner mutation.

    A model may restate an already selected owner value while the user is
    actually accepting or rejecting the derived values displayed by the
    active question. Only durable mutations owned by that exact pending group
    are reviewed here. The reviewer can bind one declared option or preserve
    the mutation; it cannot invent a third operation.
    """

    pending = dict(state.get("pending_question") or {})
    options = [item for item in pending.get("options") or [] if isinstance(item, dict)]
    pending_group = str(pending.get("group") or "")
    accepted_types = {str(item) for item in pending.get("accepted_action_types") or []}
    if not options or not pending_group:
        return text, False

    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    reviews: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("type") or "")
        spec = ACTION_BY_TYPE.get(action_type)
        if (
            spec is None
            or action_type in accepted_types
            or spec.lifetime != "durable"
            or not spec.mutation_dimension
            or spec.target_group != pending_group
        ):
            continue
        source_units = [
            str(unit.get("source_text") or "")
            for unit in units
            if isinstance(unit, dict)
            and index in (
                unit.get("action_indexes")
                if isinstance(unit.get("action_indexes"), list)
                else []
            )
            and str(unit.get("source_text") or "")
        ]
        source = " ".join(source_units).strip() or str(action.get("source_evidence") or "").strip()
        if not source:
            continue
        reviews.append({
            "action_index": index,
            "question": str(pending.get("prompt") or ""),
            "declared_options": [
                {"label": item.get("label"), "value": item.get("value")}
                for item in options
            ],
            "proposed_owner_mutation": action,
            "source_text": source,
        })
    if not reviews:
        return text, False

    result = _request_pending_owner_reviews(provider, reviews)
    rows = result.get("reviews") if isinstance(result.get("reviews"), list) else []
    by_index = {
        row.get("action_index"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("action_index"), int)
    }
    changed = False
    rejected_indexes: set[int] = set()
    for review in reviews:
        index = int(review["action_index"])
        row = by_index.get(index) or {}
        decision = str(row.get("decision") or "")
        quote = str(row.get("evidence_quote") or "").strip()
        if decision == "keep_owner_mutation" and quote and quote in user_text:
            continue
        if decision != "select_pending_option":
            rejected_indexes.add(index)
            continue
        selected = row.get("selected_option_value")
        if not pending_option_value_exists(selected, pending) or not quote or quote not in user_text:
            rejected_indexes.add(index)
            continue
        actions[index] = {
            "type": "answer_pending",
            "answer": selected,
            "selected_value": selected,
            "source_evidence": quote,
        }
        changed = True
    if rejected_indexes:
        payload["actions"] = actions
        return _remove_rejected_action_indexes(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            tuple(sorted(rejected_indexes)),
            reason="the source neither answered the active pending contract nor requested the proposed owner mutation",
        )
    if not changed:
        return text, False
    payload["actions"] = actions
    return json.dumps(payload, ensure_ascii=False, sort_keys=True), True


def _bind_active_pending_config_value(
    text: str,
    state: AgentGraphState,
    user_text: str,
) -> str:
    """Bind a proposed value to the exact active field contract.

    Configuration proposals normally require user review. When the proposal
    contains the field that the active question is already asking for, that
    value is a direct answer rather than a second configuration transaction.
    Any other proposed fields remain in their original review action.
    """

    pending = dict(state.get("pending_question") or {})
    field = str(pending.get("field") or "").strip()
    if pending.get("manual_input_allowed") is not True or not field:
        return text

    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    changed = False
    for index, action in enumerate(tuple(actions)):
        if not isinstance(action, dict) or str(action.get("type") or "") != "propose_config_values":
            continue
        config_values = dict(action.get("config_values") or {})
        matching_keys = [key for key in config_values if str(key).strip().casefold() == field.casefold()]
        if len(matching_keys) != 1:
            continue
        key = matching_keys[0]
        value = config_values[key]
        answer = str(value).strip()
        if not answer_fits_pending(answer, pending):
            continue

        remainder = dict(action)
        remainder_values = dict(config_values)
        remainder_values.pop(key, None)
        remainder["config_values"] = remainder_values
        actions[index] = {
            "type": "answer_pending",
            "answer": answer,
            "selected_value": None,
            "source_evidence": str(user_text or "").strip(),
        }
        if remainder_values or remainder.get("unmapped_values"):
            remainder_index = len(actions)
            actions.append(remainder)
            for unit in units:
                if not isinstance(unit, dict):
                    continue
                indexes = unit.get("action_indexes")
                if isinstance(indexes, list) and index in indexes and remainder_index not in indexes:
                    indexes.append(remainder_index)
        changed = True

    if not changed:
        return text
    payload["actions"] = actions
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _adjudicate_pending_action_ownership(
    provider: Any,
    text: str,
    state: AgentGraphState,
    user_text: str,
) -> tuple[str, bool]:
    """Apply the single pending-contract arbitration pipeline."""

    current = _bind_active_pending_config_value(text, state, user_text)
    changed = current != text
    current, step_changed = _adjudicate_manual_pending_answers(provider, current, state, user_text)
    changed = changed or step_changed
    current, step_changed = _reconcile_pending_owner_mutations(provider, current, state, user_text)
    changed = changed or step_changed
    current, step_changed = _adjudicate_pending_answer_actions(provider, current, state, user_text)
    return current, changed or step_changed


def _adjudicate_manual_pending_answers(
    provider: Any,
    text: str,
    state: AgentGraphState,
    user_text: str,
) -> tuple[str, bool]:
    """Admit only direct answers to an active manual-input contract."""

    pending = dict(state.get("pending_question") or {})
    if pending.get("manual_input_allowed") is not True:
        return text, False
    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    candidate_indexes = [
        index
        for index, action in enumerate(actions)
        if isinstance(action, dict) and str(action.get("type") or "") == "answer_pending"
        and not pending_option_value_exists(action.get("selected_value"), pending)
        and (
            not pending.get("options")
            or answer_fits_pending(str(action.get("answer") or ""), pending)
        )
    ]
    directly_grounded = {
        index
        for index in candidate_indexes
        if _manual_answer_has_literal_source(actions[index], user_text)
    }
    review_indexes = [index for index in candidate_indexes if index not in directly_grounded]
    if not review_indexes and not directly_grounded:
        return text, False

    if not review_indexes:
        payload["pending_answer_admissions"] = sorted(directly_grounded)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True), False

    reviews = []
    for index in review_indexes:
        sources = [
            str(unit.get("source_text") or "")
            for unit in units
            if isinstance(unit, dict)
            and index in (
                unit.get("action_indexes")
                if isinstance(unit.get("action_indexes"), list)
                else []
            )
            and str(unit.get("source_text") or "")
        ]
        reviews.append({
            "action_index": index,
            "question": str(pending.get("prompt") or ""),
            "field": str(pending.get("field") or ""),
            "kind": str(pending.get("kind") or ""),
            "proposed_answer": actions[index].get("answer"),
            "source_units": sources,
        })

    result = _request_manual_pending_reviews(provider, reviews)
    rows = result.get("reviews") if isinstance(result.get("reviews"), list) else []
    by_index = {
        row.get("action_index"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("action_index"), int)
    }
    admitted: set[int] = set(directly_grounded)
    rejected: set[int] = set()
    changed = False
    for review in reviews:
        index = int(review["action_index"])
        row = by_index.get(index) or {}
        decision = str(row.get("decision") or "")
        quote = str(row.get("evidence_quote") or "").strip()
        if decision == "direct_answer" and quote and quote in user_text:
            admitted.add(index)
            continue
        if decision == "generic_resume" and quote and quote in user_text:
            actions[index] = {
                "type": "resume_current_flow",
                "source_evidence": quote,
            }
            changed = True
            continue
        rejected.add(index)

    payload["actions"] = actions
    if admitted:
        payload["pending_answer_admissions"] = sorted(admitted)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if rejected:
        return _remove_rejected_action_indexes(
            serialized,
            tuple(sorted(rejected)),
            reason="the source does not directly answer the active manual pending contract",
        )
    if admitted or changed:
        return serialized, changed
    return text, False


def _manual_answer_has_literal_source(action: dict[str, Any], user_text: str) -> bool:
    """Prove that a normalized manual answer occurs in exact source evidence."""

    answer = str(action.get("answer") or "").strip()
    quote = str(action.get("source_evidence") or "").strip()
    if not answer or not quote or quote not in str(user_text or ""):
        return False
    return re.search(
        rf"(?<!\w){re.escape(answer)}(?!\w)",
        quote,
        flags=re.IGNORECASE,
    ) is not None


def _request_manual_pending_reviews(provider: Any, reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Return one finite decision for every semantic manual-field answer."""

    system = (
        "Return JSON only: {reviews:[{action_index:integer,"
        "decision:'direct_answer'|'generic_resume'|'reject',evidence_quote:string,reason:string}]}. "
        "Review every row exactly once. direct_answer means source_units directly supply one value requested by the "
        "displayed question and field; ordinary prose around that value is allowed, but navigation, a question, a "
        "correction, a comparison, or a value for another field is not. generic_resume means the source explicitly "
        "asks to resume or return to the current benchmark configuration without naming a different destination or "
        "supplying the requested field value. reject means neither. Never infer a value from workflow state, defaults, "
        "or the proposed_answer. For direct_answer and generic_resume, evidence_quote must be the shortest exact excerpt "
        "from source_units proving the decision. reject uses an empty quote."
    )
    invalid = ""
    for attempt in range(2):
        body: dict[str, Any] = {"reviews": reviews}
        if attempt:
            body["invalid_previous_output"] = invalid
            body["repair_instruction"] = "Return a complete replacement following the finite schema."
        response = provider.complete(LLMRequest(
            messages=[
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=json.dumps(body, ensure_ascii=False, sort_keys=True)),
            ],
            temperature=0.0,
            max_tokens=450,
        ))
        invalid = str(response.text or "")
        result = _parse_json_object(invalid)
        rows = result.get("reviews") if isinstance(result.get("reviews"), list) else []
        by_index = {
            row.get("action_index"): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("action_index"), int)
        }
        complete = True
        for review in reviews:
            row = by_index.get(review["action_index"])
            decision = str((row or {}).get("decision") or "")
            quote = str((row or {}).get("evidence_quote") or "").strip()
            if decision not in {"direct_answer", "generic_resume", "reject"}:
                complete = False
                break
            if decision == "reject" and quote:
                complete = False
                break
            if decision != "reject" and not quote:
                complete = False
                break
        if complete:
            return result
    return {"reviews": []}


def _request_pending_owner_reviews(provider: Any, reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Return one finite ownership decision for each same-group mutation."""

    system = (
        "Return JSON only: {reviews:[{action_index:integer,"
        "decision:'select_pending_option'|'keep_owner_mutation'|'reject',"
        "selected_option_value:any,evidence_quote:string,reason:string}]}. Include every key exactly once and "
        "review every row exactly once. select_pending_option means the source explicitly accepts or rejects one "
        "declared option in the displayed question, even if it also restates the already selected owner value. "
        "keep_owner_mutation means the source explicitly requests the proposed new owner mutation instead of "
        "answering the displayed question. reject means neither interpretation is source-grounded. Never infer a "
        "default from workflow state. For select_pending_option, selected_option_value must equal one declared "
        "option value and evidence_quote must be the shortest exact source_text excerpt proving that choice. For "
        "the other decisions selected_option_value must be null; reject uses an empty quote."
    )
    invalid = ""
    for attempt in range(2):
        body: dict[str, Any] = {"reviews": reviews}
        if attempt:
            body["invalid_previous_output"] = invalid
            body["repair_instruction"] = "Return a complete replacement that follows the finite schema."
        response = provider.complete(LLMRequest(
            messages=[
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=json.dumps(body, ensure_ascii=False, sort_keys=True)),
            ],
            temperature=0.0,
            max_tokens=500,
        ))
        invalid = str(response.text or "")
        result = _parse_json_object(invalid)
        rows = result.get("reviews") if isinstance(result.get("reviews"), list) else []
        by_index = {
            row.get("action_index"): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("action_index"), int)
        }
        complete = True
        for review in reviews:
            row = by_index.get(review["action_index"])
            decision = str((row or {}).get("decision") or "")
            if decision not in {"select_pending_option", "keep_owner_mutation", "reject"}:
                complete = False
                break
            if decision == "select_pending_option":
                values = [item.get("value") for item in review["declared_options"]]
                if row.get("selected_option_value") not in values or not str(row.get("evidence_quote") or "").strip():
                    complete = False
                    break
            elif decision == "keep_owner_mutation":
                if row.get("selected_option_value") is not None or not str(row.get("evidence_quote") or "").strip():
                    complete = False
                    break
            elif row.get("selected_option_value") is not None or str(row.get("evidence_quote") or "").strip():
                complete = False
                break
        if complete:
            return result
    return {"reviews": []}


def _request_pending_answer_reviews(
    provider: Any,
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return one mutually exclusive decision for every pending option review."""

    schema = (
        "Return JSON only: {reviews:[{action_index:integer,"
        "decision:'select_option'|'unresolved_target_mode'|'reject',"
        "evidence_quote:string,reason:string}]}. Include each key exactly once in every row. "
    )
    rules = (
        "Review every supplied row exactly once. Use select_option only when source_units explicitly select, "
        "request, or answer the displayed option in the context of the displayed question. Use "
        "unresolved_target_mode only when allows_unresolved_target_mode=true and the source has a benchmark or "
        "synchronization-observation goal but explicitly leaves the displayed target mode undecided. Otherwise use "
        "reject. Topical relation is insufficient. A greeting, identity question, help request, explanation request, "
        "confusion, or other read-only detour does not select an option. Never infer a choice from workflow state or "
        "defaults. select_option and unresolved_target_mode require the shortest exact excerpt from source_units in "
        "evidence_quote; reject uses an empty evidence_quote."
    )
    invalid_output = ""
    for attempt in range(2):
        messages = [LLMMessage(role="system", content=schema + rules)]
        payload: dict[str, Any] = {"reviews": reviews}
        if attempt:
            payload["invalid_previous_output"] = invalid_output
            payload["repair_instruction"] = (
                "The previous output did not satisfy the finite decision schema. Re-evaluate the original rows and "
                "return a complete replacement document; do not copy malformed rows."
            )
        messages.append(
            LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True))
        )
        response = provider.complete(LLMRequest(messages=messages, temperature=0.0, max_tokens=350))
        invalid_output = str(response.text or "")
        result = _parse_json_object(invalid_output)
        rows = result.get("reviews") if isinstance(result.get("reviews"), list) else []
        by_index = {
            row.get("action_index"): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("action_index"), int)
        }
        structurally_complete = True
        for review in reviews:
            row = by_index.get(review["action_index"])
            decision = str((row or {}).get("decision") or "")
            quote = str((row or {}).get("evidence_quote") or "").strip()
            if decision not in {"select_option", "unresolved_target_mode", "reject"}:
                structurally_complete = False
                break
            if decision in {"select_option", "unresolved_target_mode"} and not quote:
                structurally_complete = False
                break
        if structurally_complete:
            return result
    return {"reviews": []}


def _adjudicate_rejected_pending_answer_reviews(
    provider: Any,
    *,
    reviews: list[dict[str, Any]],
    first_rows: list[Any],
) -> list[Any]:
    """Recheck semantic option false negatives without weakening admission.

    Menu IDs and labels are input conveniences, not mandatory user language.
    A second bounded reviewer may reverse only a source-quoted rejection for
    the exact option already proposed by the planner. It cannot choose another
    value or use workflow state as evidence.
    """

    review_by_index = {
        int(item["action_index"]): item
        for item in reviews
        if isinstance(item, dict) and isinstance(item.get("action_index"), int)
    }
    negative_rows = [
        row
        for row in first_rows
        if isinstance(row, dict)
        and row.get("decision") == "reject"
        and isinstance(row.get("action_index"), int)
        and int(row["action_index"]) in review_by_index
    ]
    if not negative_rows:
        return first_rows

    response = provider.complete(LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "Adjudicate only the supplied negative pending-option verdicts. Return JSON only: "
                    "{reviews:[{action_index:integer,decision:'select_option'|'reject',"
                    "evidence_quote:string,reason:string}]}. Review every row exactly once. Compare the complete "
                    "source_units with the displayed question and the exact proposed option label and value. "
                    "select_option means the source semantically answers the question by accepting, rejecting, "
                    "choosing, or requesting that proposed option. Natural-language answers are valid and do not "
                    "need to repeat the option ID, number, label, Y, or N. Topical discussion, help, explanation, "
                    "confusion, or a question about the option is reject. Never select a different value and never "
                    "infer from workflow state or defaults. select_option requires the shortest non-empty exact "
                    "substring of source_units that proves the choice; reject requires an empty evidence_quote."
                ),
            ),
            LLMMessage(role="user", content=json.dumps({
                "reviews": [review_by_index[int(row["action_index"])] for row in negative_rows],
                "negative_first_verdicts": negative_rows,
            }, ensure_ascii=False, sort_keys=True)),
        ],
        temperature=0.0,
        max_tokens=450,
    ))
    result = _parse_json_object(response.text)
    adjudicated = result.get("reviews") if isinstance(result.get("reviews"), list) else []
    replacements: dict[int, dict[str, Any]] = {}
    counts: dict[int, int] = {}
    candidates: dict[int, dict[str, Any]] = {}
    for row in adjudicated:
        if not isinstance(row, dict) or not isinstance(row.get("action_index"), int):
            continue
        index = int(row["action_index"])
        if index not in review_by_index:
            continue
        decision = str(row.get("decision") or "")
        quote = str(row.get("evidence_quote") or "").strip()
        source = "\n".join(str(item) for item in review_by_index[index].get("source_units") or [])
        if decision == "select_option" and quote and quote in source:
            pass
        elif decision == "reject" and not quote:
            pass
        else:
            continue
        counts[index] = counts.get(index, 0) + 1
        candidates[index] = row
    for index, count in counts.items():
        if count == 1:
            replacements[index] = candidates[index]
    return [
        replacements.get(int(row.get("action_index")), row)
        if isinstance(row, dict) and isinstance(row.get("action_index"), int)
        else row
        for row in first_rows
    ]


def _adjudicate_group_navigation_actions(
    provider: Any,
    text: str,
    state: AgentGraphState,
) -> tuple[str, bool]:
    """Normalize group navigation and owner intake at one admission boundary.

    The reviewer deliberately receives no workflow destination or pending-field
    details beyond whether a typed pending question exists. It can verify the
    proposed destination, but it cannot invent one from current state.
    """

    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    review_indexes: list[int] = []
    review_specs: dict[int, Any] = {}
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("type") or "")
        spec = ACTION_BY_TYPE.get(action_type)
        if action_type == "change_group" or (spec and spec.requires_specific_change):
            review_indexes.append(index)
            review_specs[index] = spec
    if not review_indexes:
        return text, False

    has_pending_question = bool(state.get("pending_question"))
    review_payload = [
        {
            "action_index": index,
            "proposed_group": (
                actions[index].get("group")
                or getattr(review_specs.get(index), "target_group", "")
            ),
            "source_evidence": actions[index].get("source_evidence"),
        }
        for index in review_indexes
    ]
    response = provider.complete(LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "Adjudicate AnyChain workflow-group navigation. Return JSON only: "
                    "{reviews:[{action_index:integer,destination_named:boolean,destination_quote:string,"
                    "generic_resume:boolean,resume_quote:string,specific_change_requested:boolean,"
                    "specific_change_quote:string,reason:string}]}. Review every row exactly once. "
                    "destination_named is true only when source_evidence itself identifies the proposed destination "
                    "configuration area through its registered name, field, question, or a clear natural-language "
                    "equivalent. A precise product field such as benchmark mode may identify its registered group. "
                    "Words that name the overall product activity, such as benchmark, test, configuration, setup, "
                    "settings, or workflow, never identify a subgroup by themselves; they require a subgroup-specific "
                    "modifier such as QPS/profile, RPC/workload, disk/storage, chain, endpoint, or observability. "
                    "generic_resume is true only when the source asks to resume/return to benchmark or configuration "
                    "work without identifying one exact registered area. Therefore a source that names QPS settings, "
                    "RPC workload, observability, disk, chain, or another exact area is destination_named=true and "
                    "generic_resume=false. specific_change_requested is true when the source explicitly asks to alter, "
                    "customize, override, replace, enable, disable, add, or remove a value/default owned by the proposed "
                    "group, rather than merely visit that group. An explicit instruction to visit/configure a group "
                    "without changing its values/defaults is not a specific change. Do not infer a destination or change "
                    "from defaults, ordering, state, or the existence "
                    "of a pending question. Each true fact requires the shortest exact source excerpt in its matching "
                    "quote field; use an empty quote for a false fact."
                ),
            ),
            LLMMessage(role="user", content=json.dumps({
                "has_pending_question": has_pending_question,
                "registered_groups": [
                    {
                        "name": row["name"],
                        "category": row["category"],
                        "fields": row["fields"],
                        "questions": row["questions"],
                    }
                    for row in group_schema()
                ],
                "reviews": review_payload,
            }, ensure_ascii=False, sort_keys=True)),
        ],
        temperature=0.0,
        max_tokens=500,
    ))
    result = _parse_json_object(response.text)
    rows = result.get("reviews") if isinstance(result.get("reviews"), list) else []
    row_counts: dict[int, int] = {}
    row_candidates: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("action_index"), int):
            continue
        index = int(row["action_index"])
        if index not in review_specs:
            continue
        row_counts[index] = row_counts.get(index, 0) + 1
        row_candidates[index] = row
    by_index = {
        index: row_candidates[index]
        for index, count in row_counts.items()
        if count == 1
    }
    by_index = _adjudicate_positive_specific_group_changes(
        provider,
        reviews=review_payload,
        first_rows=by_index,
    )

    rejected: dict[int, str] = {}
    changed = False
    for index in review_indexes:
        row = by_index.get(index) or {}
        action_type = str(actions[index].get("type") or "")
        spec = review_specs.get(index)
        proposed_group = str(
            actions[index].get("group") or getattr(spec, "target_group", "") or ""
        )
        source = str(actions[index].get("source_evidence") or "")
        destination_quote = str(row.get("destination_quote") or "").strip()
        resume_quote = str(row.get("resume_quote") or "").strip()
        specific_change_quote = str(row.get("specific_change_quote") or "").strip()
        reason = str(row.get("reason") or "group navigation was not explicit in source evidence")
        destination_named = bool(
            row.get("destination_named") is True
            and destination_quote
            and destination_quote in source
        )
        generic_resume = bool(
            row.get("generic_resume") is True
            and resume_quote
            and resume_quote in source
        )
        specific_change = bool(
            row.get("specific_change_requested") is True
            and specific_change_quote
            and specific_change_quote in source
        )
        if specific_change and destination_named:
            if action_type != "change_group":
                continue
            replacement = _resolve_owned_group_mutation(
                provider,
                proposed_group,
                source,
            )
            if replacement is not None:
                actions[index] = replacement
                changed = True
                continue
            rejected[index] = "source requests a specific owned configuration change that was not safely resolved"
            continue
        if destination_named:
            if action_type != "change_group":
                replacement = {
                    "type": "change_group",
                    "group": proposed_group,
                    "navigation_explicit": True,
                    "source_evidence": destination_quote,
                }
                try:
                    actions[index] = validate_action_contract(replacement)
                except ValueError:
                    rejected[index] = "registered owner intake could not normalize to group navigation"
                    continue
                changed = True
            continue
        if generic_resume and has_pending_question:
            actions[index] = {
                "type": "resume_current_flow",
                "source_evidence": resume_quote,
            }
            changed = True
            continue
        rejected[index] = reason

    if rejected:
        text, removed = _remove_rejected_action_indexes(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            tuple(rejected),
            reason="; ".join(sorted(set(rejected.values()))),
        )
        return text, changed or removed
    if changed:
        payload["actions"] = actions
        return json.dumps(payload, ensure_ascii=False, sort_keys=True), True
    return text, False


def _adjudicate_positive_specific_group_changes(
    provider: Any,
    *,
    reviews: list[dict[str, Any]],
    first_rows: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Require independent support before group navigation becomes mutation."""

    review_by_index = {
        int(item["action_index"]): item
        for item in reviews
        if isinstance(item.get("action_index"), int)
    }
    positive_indexes = [
        index
        for index, row in first_rows.items()
        if row.get("specific_change_requested") is True
    ]
    if not positive_indexes:
        return first_rows

    response = provider.complete(LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "Independently adjudicate only positive specific-change verdicts for registered AnyChain "
                    "group control. Return JSON only: {reviews:[{action_index:integer,"
                    "specific_change_requested:boolean,specific_change_quote:string,reason:string}]}. Review "
                    "every row exactly once. The proposed operation and group are immutable. True means the "
                    "source itself explicitly asks to alter, tune, customize, override, replace, enable, disable, "
                    "add, or remove one or more values/defaults owned by that proposed group. The new concrete "
                    "values or numbers may be collected in later turns and are not required in this source. Merely asking to visit, "
                    "configure, handle, or do that group first is false when no value/default change is requested. "
                    "Do not infer a change from workflow state, ordering, defaults, or the proposed action type. "
                    "True requires the shortest non-empty exact source excerpt proving the change; false requires "
                    "an empty quote. Never repair or replace an operation."
                ),
            ),
            LLMMessage(role="user", content=json.dumps({
                "reviews": [review_by_index[index] for index in positive_indexes],
            }, ensure_ascii=False, sort_keys=True)),
        ],
        temperature=0.0,
        max_tokens=350,
    ))
    result = _parse_json_object(response.text)
    rows = result.get("reviews") if isinstance(result.get("reviews"), list) else []
    counts: dict[int, int] = {}
    candidates: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("action_index"), int):
            continue
        index = int(row["action_index"])
        if index not in positive_indexes or not isinstance(row.get("specific_change_requested"), bool):
            continue
        quote = str(row.get("specific_change_quote") or "").strip()
        source = str(review_by_index[index].get("source_evidence") or "")
        if row["specific_change_requested"] is True:
            if not quote or quote not in source:
                continue
        elif quote:
            continue
        counts[index] = counts.get(index, 0) + 1
        candidates[index] = row

    output = dict(first_rows)
    for index in positive_indexes:
        if counts.get(index) != 1:
            output.pop(index, None)
            continue
        second = candidates[index]
        if second["specific_change_requested"] is False:
            updated = dict(output[index])
            updated["specific_change_requested"] = False
            updated["specific_change_quote"] = ""
            updated["reason"] = str(second.get("reason") or updated.get("reason") or "")
            output[index] = updated
    return output


def _recover_registry_bounded_semantic_actions(
    provider: Any,
    plan_text: str,
    clauses: tuple[TurnClause, ...],
    state: AgentGraphState,
    user_text: str,
    validation: PlanCoverageResult | None = None,
) -> tuple[str, bool]:
    """Recover rejected semantic units through registry-bounded safe actions.

    The primary planner can occasionally express an explicit group detour as an
    invalid consultation topic, leave it unresolved, or assign a safe
    turn-local purpose incorrectly. This admission-stage reviewer sees only
    those structurally unrepresented semantic units. Its output is bounded by
    the group, consultation, and action registries; all recovered actions still
    pass the normal navigation and semantic gates.
    """

    payload = _parse_json_object(plan_text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    if not units:
        return plan_text, False

    validation = validation or PlanCoverageResult(False, (), ())
    invalid_indexes: set[int] = set(validation.rejected_action_indexes)
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            invalid_indexes.add(index)
            continue
        try:
            validate_action_contract(action)
        except ValueError:
            invalid_indexes.add(index)

    candidates: list[dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        indexes = unit.get("action_indexes") if isinstance(unit.get("action_indexes"), list) else []
        rejected = [index for index in indexes if index in invalid_indexes]
        if (
            str(unit.get("disposition") or "") != "unresolved"
            and not rejected
            and str(unit.get("unit_id") or "") not in validation.incomplete_unit_ids
        ):
            continue
        source = str(unit.get("source_text") or "")
        if not source:
            continue
        candidates.append({
            "unit_id": str(unit.get("unit_id") or ""),
            "source_text": source,
            "invalid_actions": [
                actions[index]
                for index in rejected
                if 0 <= index < len(actions) and isinstance(actions[index], dict)
            ],
        })
    if not candidates:
        return plan_text, False

    recoverable_turn_local_actions = [
        {
            "type": spec.action_type,
            "purpose": spec.purpose,
            "source_argument": spec.semantic_recovery_source_argument,
        }
        for spec in ACTION_SPECS
        if spec.lifetime == "turn_local"
        and spec.semantic_recovery_source_argument
        and spec.semantic_recovery_source_argument in spec.allowed_arguments
    ]
    recoverable_turn_local_by_type = {
        row["type"]: row
        for row in recoverable_turn_local_actions
    }

    sibling_navigations = [
        {
            "action_index": index,
            "group": str(action.get("group") or ""),
            "source_units": [
                str(unit.get("source_text") or "")
                for unit in units
                if isinstance(unit, dict)
                and index in (
                    unit.get("action_indexes")
                    if isinstance(unit.get("action_indexes"), list)
                    else []
                )
            ],
        }
        for index, action in enumerate(actions)
        if index not in invalid_indexes
        and isinstance(action, dict)
        and str(action.get("type") or "") == "change_group"
        and str(action.get("group") or "")
    ]

    response = provider.complete(LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "Adjudicate only structurally unrepresented AnyChain group-control intent. Return JSON only: "
                    "{decisions:[{unit_id:string,disposition:'navigation'|'shared_navigation'|'generic_resume'|'consultation'|"
                    "'owner_mutation'|'explicit_target_mode'|'unresolved_target_mode'|'turn_local_action'|"
                    "'context'|'not_group_control',group:string,"
                    "existing_action_index:integer|null,"
                    "target_mode:'fake-node'|'real-node'|'sync-observe'|'',"
                    "consultation_topic:string,turn_local_action_type:string,evidence_quote:string,"
                    "reason:string}]}. Review every supplied unit. Return one decision for a unit with one semantic "
                    "demand. When one unsplit unit contains multiple independent present demands, return one decision "
                    "per demand with the same unit_id and a distinct shortest exact evidence_quote; this is required "
                    "even when a prior partition stage did not split the prose. Multiple decisions for one unit may "
                    "combine owner_mutation with unresolved_target_mode or one explicit_target_mode, but must not mix "
                    "context, ambiguity, consultation, navigation, resume, or turn-local handling with configuration "
                    "mutations. Use navigation only when source_text "
                    "explicitly asks to visit, return to, or configure one exact registered group without supplying a "
                    "more specific value change. Use owner_mutation only when it explicitly asks to change/customize a "
                    "value owned by one exact group. Use generic_resume only when it asks to resume the current workflow "
                    "without naming a group. Use consultation only for a read-only question and select exactly one "
                    "allowed consultation topic. Use unresolved_target_mode only when source_text expresses a concrete "
                    "benchmark or synchronization-observation goal but explicitly leaves fake-node, real-node, and "
                    "sync-observe undecided; this disposition requests the typed target-mode menu and never selects a "
                    "default. Use explicit_target_mode only when source_text explicitly selects exactly one registered "
                    "target mode; set target_mode to that exact registered value. A generic benchmark goal, risk "
                    "preference, or missing endpoint does not select a target mode. A request for a recommendation "
                    "without a concrete execution goal is consultation, not "
                    "unresolved_target_mode. Use shared_navigation only when source_text is temporal/defer/resume framing "
                    "for exactly one available_sibling_navigation already supported by another source unit in this same "
                    "turn; set existing_action_index to that row's exact action_index. It does not create a destination "
                    "or action. Never use it for unrelated, contradictory, ambiguous, or independently actionable text. "
                    "Use turn_local_action only when source_text directly fulfils exactly one listed "
                    "recoverable_turn_local_action purpose; set turn_local_action_type to that exact registered type. "
                    "The complete user turn will be supplied only through its declared source_argument. Do not select "
                    "a durable configuration or execution action. Use not_group_control for ambiguity or unrelated "
                    "content that may still require clarification. Use context only for prose that is background, "
                    "provenance, or a tentative future possibility and contains no present request, answer, question, "
                    "selection, correction, contradiction, mutation, navigation, evidence submission, or execution "
                    "instruction. Context creates no action and will be independently audited after recovery. Never infer a "
                    "destination from workflow order, current state, defaults, or a pending question. group must be an "
                    "exact registered name for navigation/owner_mutation and empty otherwise. consultation_topic must be "
                    "an exact allowed topic for consultation and empty otherwise. target_mode must be empty unless "
                    "disposition is explicit_target_mode. turn_local_action_type must be empty unless disposition is "
                    "turn_local_action. existing_action_index must be null unless disposition is "
                    "shared_navigation. evidence_quote must be the shortest "
                    "exact excerpt from source_text that proves the disposition; use an empty quote only for "
                    "not_group_control. Context requires a non-empty exact evidence quote."
                ),
            ),
            LLMMessage(role="user", content=json.dumps({
                "has_pending_question": bool(state.get("pending_question")),
                "registered_groups": [
                    {
                        "name": row["name"],
                        "category": row["category"],
                        "fields": row["fields"],
                        "questions": row["questions"],
                    }
                    for row in group_schema()
                ],
                "allowed_consultation_topics": sorted(CONSULTATION_TOPICS),
                "recoverable_turn_local_actions": recoverable_turn_local_actions,
                "available_sibling_navigations": sibling_navigations,
                "units": candidates,
            }, ensure_ascii=False, sort_keys=True)),
        ],
        temperature=0.0,
        max_tokens=900,
    ))
    result = _parse_json_object(response.text)
    decisions = result.get("decisions") if isinstance(result.get("decisions"), list) else []
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for row in decisions:
        if not isinstance(row, dict) or not str(row.get("unit_id") or ""):
            continue
        by_unit.setdefault(str(row["unit_id"]), []).append(row)
    known_groups = {str(row["name"]) for row in group_schema()}
    sibling_navigation_indexes = {
        int(row["action_index"])
        for row in sibling_navigations
    }
    replacements: dict[str, list[dict[str, Any]]] = {}
    shared_navigation_indexes: dict[str, int] = {}
    context_unit_ids: set[str] = set()
    for candidate in candidates:
        unit_id = candidate["unit_id"]
        source = candidate["source_text"]
        unit_decisions = by_unit.get(unit_id) or []
        if not unit_decisions:
            return plan_text, False
        dispositions = {str(row.get("disposition") or "") for row in unit_decisions}
        if len(unit_decisions) > 1 and dispositions.difference({
            "owner_mutation",
            "explicit_target_mode",
            "unresolved_target_mode",
        }):
            return plan_text, False
        unit_replacements: list[dict[str, Any]] = []
        seen_replacements: set[str] = set()
        for decision in unit_decisions:
            disposition = str(decision.get("disposition") or "")
            quote = str(decision.get("evidence_quote") or "").strip()
            group = str(decision.get("group") or "").strip()
            topic = str(decision.get("consultation_topic") or "").strip()
            target_mode = str(decision.get("target_mode") or "").strip()
            turn_local_action_type = str(decision.get("turn_local_action_type") or "").strip()
            existing_action_index = decision.get("existing_action_index")
            if not quote or quote not in source:
                return plan_text, False
            if (
                disposition == "shared_navigation"
                and isinstance(existing_action_index, int)
                and existing_action_index in sibling_navigation_indexes
            ):
                shared_navigation_indexes[unit_id] = existing_action_index
                continue
            if disposition == "context":
                context_unit_ids.add(unit_id)
                continue
            if disposition == "navigation" and group in known_groups:
                replacement = {
                    "type": "change_group",
                    "group": group,
                    "navigation_explicit": True,
                    "source_evidence": user_text,
                }
            elif disposition == "generic_resume" and state.get("pending_question"):
                replacement = {
                    "type": "resume_current_flow",
                    "source_evidence": quote,
                }
            elif disposition == "consultation" and topic in CONSULTATION_TOPICS:
                replacement = {
                    "type": "answer_opening_question",
                    "topic": topic,
                    "source_evidence": user_text,
                }
            elif disposition == "turn_local_action" and turn_local_action_type in recoverable_turn_local_by_type:
                recovery_contract = recoverable_turn_local_by_type[turn_local_action_type]
                replacement = {
                    "type": turn_local_action_type,
                    str(recovery_contract["source_argument"]): user_text,
                }
            elif disposition == "unresolved_target_mode":
                replacement = {
                    "type": "request_target_mode_selection",
                    "source_evidence": quote,
                }
            elif disposition == "explicit_target_mode" and target_mode in {"fake-node", "real-node", "sync-observe"}:
                replacement = {
                    "type": "choose_target_mode",
                    "target_mode": target_mode,
                    "target_mode_explicit": True,
                    "source_evidence": quote,
                }
            elif disposition == "owner_mutation" and group in known_groups:
                replacement = _resolve_owned_group_mutation(provider, group, quote)
                if replacement is None:
                    return plan_text, False
            else:
                return plan_text, False
            try:
                replacement = validate_action_contract(replacement)
            except ValueError:
                return plan_text, False
            replacement_key = json.dumps(replacement, ensure_ascii=False, sort_keys=True)
            if replacement_key in seen_replacements:
                continue
            seen_replacements.add(replacement_key)
            unit_replacements.append(replacement)
        if unit_replacements:
            replacements[unit_id] = unit_replacements

    recovered_target_modes = {
        str(action.get("target_mode") or "")
        for unit_actions in replacements.values()
        for action in unit_actions
        if str(action.get("type") or "") == "choose_target_mode"
    }
    recovered_unresolved_target_mode = any(
        str(action.get("type") or "") == "request_target_mode_selection"
        for unit_actions in replacements.values()
        for action in unit_actions
    )
    if len(recovered_target_modes) > 1 or (
        recovered_target_modes and recovered_unresolved_target_mode
    ):
        return plan_text, False

    filtered_text, _ = _remove_rejected_action_indexes(
        plan_text,
        tuple(sorted(invalid_indexes)),
        reason="structurally invalid operation-purpose mapping",
    )
    recovered = _parse_json_object(filtered_text)
    recovered_actions = recovered.get("actions") if isinstance(recovered.get("actions"), list) else []
    recovered_units = recovered.get("semantic_units") if isinstance(recovered.get("semantic_units"), list) else []
    retained_index_map = {
        old_index: new_index
        for new_index, old_index in enumerate(
            index for index in range(len(actions)) if index not in invalid_indexes
        )
    }
    shareable_action_types = {
        "change_group",
        "answer_opening_question",
        "request_target_mode_selection",
        "choose_target_mode",
        *recoverable_turn_local_by_type,
    }
    shared_indexes: dict[tuple[str, str], int] = {
        (
            str(action.get("type") or ""),
            str(action.get("group") or action.get("topic") or action.get("target_mode") or ""),
        ): index
        for index, action in enumerate(recovered_actions)
        if isinstance(action, dict)
        and str(action.get("type") or "") in shareable_action_types
    }
    for unit in recovered_units:
        if not isinstance(unit, dict):
            continue
        unit_id = str(unit.get("unit_id") or "")
        shared_old_index = shared_navigation_indexes.get(unit_id)
        if shared_old_index is not None:
            shared_new_index = retained_index_map.get(shared_old_index)
            if shared_new_index is None or shared_new_index >= len(recovered_actions):
                return plan_text, False
            shared_action = recovered_actions[shared_new_index]
            if (
                not isinstance(shared_action, dict)
                or str(shared_action.get("type") or "") != "change_group"
            ):
                return plan_text, False
            indexes = unit.get("action_indexes") if isinstance(unit.get("action_indexes"), list) else []
            unit["action_indexes"] = [*indexes, shared_new_index]
            unit["disposition"] = "action"
            unit["reason"] = "registry-bounded shared navigation recovery"
            continue
        if unit_id in context_unit_ids:
            unit["action_indexes"] = []
            unit["disposition"] = "context"
            unit["reason"] = "registry-bounded non-action context recovery"
            continue
        unit_replacements = replacements.get(unit_id)
        if not unit_replacements:
            continue
        recovered_indexes: list[int] = []
        for replacement in unit_replacements:
            action_type = str(replacement.get("type") or "")
            merge_key = (
                action_type,
                str(replacement.get("group") or replacement.get("topic") or replacement.get("target_mode") or ""),
            )
            if action_type in shareable_action_types and merge_key in shared_indexes:
                action_index = shared_indexes[merge_key]
            else:
                action_index = len(recovered_actions)
                recovered_actions.append(replacement)
                if action_type in shareable_action_types:
                    shared_indexes[merge_key] = action_index
            recovered_indexes.append(action_index)
        indexes = unit.get("action_indexes") if isinstance(unit.get("action_indexes"), list) else []
        unit["action_indexes"] = list(dict.fromkeys([*indexes, *recovered_indexes]))
        unit["disposition"] = "action"
        unit["reason"] = "registry-bounded group-control recovery"
    recovered["actions"] = recovered_actions
    return json.dumps(recovered, ensure_ascii=False, sort_keys=True), True


def _resolve_owned_group_mutation(
    provider: Any,
    group: str,
    source_text: str,
) -> dict[str, Any] | None:
    """Resolve one rejected navigation only within its registered domain."""

    group_row = next((row for row in group_schema() if row.get("name") == group), None)
    if not group_row:
        return None
    owner = str(group_row.get("owner") or "")
    candidates = [
        row
        for row in action_schema()
        if str(row.get("owner") or "") == owner
        and str(row.get("target_group") or "") == group
        and str(row.get("type") or "") not in {"change_group", "answer_pending"}
    ]
    if not candidates:
        return None
    response = provider.complete(LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "Resolve one explicit AnyChain configuration change within one already-selected owner domain. "
                    "Return JSON only: {action:object|null,reason:string}. Use exactly one candidate action type and "
                    "only its declared arguments. Select an action only when source_text explicitly supports its "
                    "purpose and every required concrete value; use the domain's intake/request action when the user "
                    "requests customization but has not supplied all values. Never return navigation, never infer a "
                    "default, and never invent a value. Return action=null when no candidate safely represents the source."
                ),
            ),
            LLMMessage(role="user", content=json.dumps({
                "group": group,
                "owner": owner,
                "source_text": source_text,
                "candidate_actions": candidates,
            }, ensure_ascii=False, sort_keys=True)),
        ],
        temperature=0.0,
        max_tokens=450,
    ))
    result = _parse_json_object(response.text)
    action = result.get("action")
    if not isinstance(action, dict):
        return None
    action = dict(action)
    action.setdefault("source_evidence", source_text)
    action.setdefault("confidence", "medium")
    try:
        return validate_action_contract(action)
    except ValueError:
        return None


def _adjudicate_chain_selection_actions(
    provider: Any,
    text: str,
    state: AgentGraphState | None = None,
) -> tuple[str, bool]:
    """Retain committed chain mutations and stage uncertain candidates."""

    payload = _parse_json_object(text)
    payload.pop("chain_selection_admissions", None)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    semantic_units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    review_indexes = [
        index
        for index, action in enumerate(actions)
        if isinstance(action, dict)
        and str(action.get("type") or "") in {"choose_chain", "change_chain"}
    ]
    if not review_indexes:
        return text, False
    pending = dict((state or {}).get("pending_question") or {})
    active_chain_intake = bool(
        pending.get("group") == "chain_identity"
        and pending.get("manual_input_allowed") is True
        and str(pending.get("kind") or "") == "chain"
    )
    response = provider.complete(LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "Verify explicit AnyChain chain-selection mutations. Return JSON only: "
                    "{reviews:[{action_index:integer,committed_selection:boolean,commitment_quote:string,"
                    "uncertain_candidate:boolean,uncertainty_quote:string,informational_only:boolean,"
                    "information_quote:string,finite_candidate_request:boolean,candidate_quote:string,reason:string}]}. "
                    "Review every row exactly once and judge the four facts "
                    "independently. committed_selection means the source commits to testing, using, selecting, or "
                    "changing to the proposed chain. When active_chain_intake=true, a direct answer that assigns or "
                    "provides exactly one proposed chain name is also a committed selection; the active typed question "
                    "already supplies the selection context, so the answer does not need to repeat a test/use verb. "
                    "uncertain_candidate means a chain is framed as a possible or "
                    "hypothetical future benchmark target without commitment. informational_only means the chain is "
                    "only an example/comparison or the subject of a capability/support/existence/protocol question. "
                    "finite_candidate_request means the user explicitly "
                    "asks the Agent to resolve one benchmark target from the supplied finite chain candidate set. "
                    "Do not infer selection from workflow state, known aliases, defaults, or a recommendation. Every "
                    "true fact requires the shortest exact source excerpt in its matching quote; false facts use an "
                    "empty quote."
                ),
            ),
            LLMMessage(role="user", content=json.dumps({
                "reviews": [
                    {
                        "action_index": index,
                        "action_type": actions[index].get("type"),
                        "proposed_chain": actions[index].get("chain_text"),
                        "proposed_candidates": actions[index].get("chain_candidates"),
                        "active_chain_intake": active_chain_intake,
                        "source_evidence": actions[index].get("source_evidence"),
                        "semantic_source_units": [
                            str(unit.get("source_text") or "")
                            for unit in semantic_units
                            if isinstance(unit, dict)
                            and index in (
                                unit.get("action_indexes")
                                if isinstance(unit.get("action_indexes"), list)
                                else []
                            )
                        ],
                    }
                    for index in review_indexes
                ],
            }, ensure_ascii=False, sort_keys=True)),
        ],
        temperature=0.0,
        max_tokens=400,
    ))
    result = _parse_json_object(response.text)
    rows = result.get("reviews") if isinstance(result.get("reviews"), list) else []
    by_index = {
        row.get("action_index"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("action_index"), int)
    }
    rejected: dict[int, str] = {}
    admitted_indexes: set[int] = set()
    changed = False
    for index in review_indexes:
        row = by_index.get(index) or {}
        source_parts = [str(actions[index].get("source_evidence") or "")]
        source_parts.extend(
            str(unit.get("source_text") or "")
            for unit in semantic_units
            if isinstance(unit, dict)
            and index in (
                unit.get("action_indexes")
                if isinstance(unit.get("action_indexes"), list)
                else []
            )
        )
        source = "\n".join(part for part in source_parts if part)
        commitment_quote = str(row.get("commitment_quote") or "").strip()
        uncertainty_quote = str(row.get("uncertainty_quote") or "").strip()
        information_quote = str(row.get("information_quote") or "").strip()
        candidate_quote = str(row.get("candidate_quote") or "").strip()
        committed = bool(
            row.get("committed_selection") is True
            and commitment_quote
            and commitment_quote in source
        )
        uncertain = bool(
            row.get("uncertain_candidate") is True
            and uncertainty_quote
            and uncertainty_quote in source
        )
        informational = bool(
            row.get("informational_only") is True
            and information_quote
            and information_quote in source
        )
        finite_candidates = bool(
            row.get("finite_candidate_request") is True
            and candidate_quote
            and candidate_quote in source
            and len(actions[index].get("chain_candidates") or []) > 1
        )
        if finite_candidates or (committed and not uncertain and not informational):
            admitted_indexes.add(index)
            continue
        if uncertain:
            actions[index] = {
                "type": "request_chain_selection",
                "chain_candidates": list(actions[index].get("chain_candidates") or []),
                "source_evidence": uncertainty_quote,
            }
            changed = True
            continue
        rejected[index] = str(row.get("reason") or "no explicit chain-selection commitment")
    if rejected:
        text, removed = _remove_rejected_action_indexes(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            tuple(rejected),
            reason="; ".join(sorted(set(rejected.values()))),
        )
        return text, changed or removed
    if changed:
        payload["actions"] = actions
        return json.dumps(payload, ensure_ascii=False, sort_keys=True), True
    if admitted_indexes:
        payload["chain_selection_admissions"] = sorted(admitted_indexes)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True), False
    return text, False


def _remove_rejected_action_indexes(
    text: str,
    rejected_indexes: tuple[int, ...],
    *,
    reason: str,
) -> tuple[str, bool]:
    """Remove rejected siblings while preserving semantic-unit ownership."""

    rejected = set(rejected_indexes)
    if not rejected:
        return text, False
    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    rejected_types = {
        index: str(actions[index].get("type") or "unknown")
        for index in rejected
        if 0 <= index < len(actions) and isinstance(actions[index], dict)
    }
    old_to_new: dict[int, int] = {}
    retained: list[dict[str, Any]] = []
    for old_index, action in enumerate(actions):
        if old_index in rejected:
            continue
        old_to_new[old_index] = len(retained)
        retained.append(action)
    payload["actions"] = retained
    for admission_key in (
        "pending_answer_admissions",
        "chain_selection_admissions",
        "target_mode_selection_admissions",
    ):
        admissions = payload.get(admission_key)
        if not isinstance(admissions, list):
            continue
        payload[admission_key] = [
            old_to_new[index]
            for index in admissions
            if isinstance(index, int) and index in old_to_new
        ]
    units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        indexes = unit.get("action_indexes") if isinstance(unit.get("action_indexes"), list) else []
        unit["action_indexes"] = [old_to_new[index] for index in indexes if index in old_to_new]
        if not unit["action_indexes"] and str(unit.get("disposition") or "") == "action":
            unit["disposition"] = "unresolved"
            unit["reason"] = reason
    payload.setdefault("admission_rejections", []).extend(
        {
            "action_index": index,
            "action_type": rejected_types.get(index, "unknown"),
            "reason": reason,
        }
        for index in sorted(rejected)
    )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True), True


def _requires_semantic_fulfillment_review(action: dict[str, Any]) -> bool:
    action_type = str(action.get("type") or "")
    spec = ACTION_BY_TYPE.get(action_type)
    if action_type == "answer_opening_question":
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
            "{reviews:[{action_index:integer,supported:boolean,reason:string}]}. "
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
        + "For context_reviews, context_only is true only when source_text is prose containing background, provenance, or a tentative future possibility and contains no present request, answer, question, selection, correction, contradiction, mutation, navigation, evidence submission, or execution instruction. A statement that answers the supplied pending_question is not context. input_shape must be prose. planner_reason is untrusted and cannot establish the verdict. Missing or ambiguous intent is false. "
        "Operations are opaque, already-registered Harness operations. Internal operation names are intentionally absent because registration, lifecycle, ordering, and choose-versus-change selection are deterministic Harness responsibilities. Never infer or discuss an internal operation name and never reject a purpose on registry or lifecycle grounds. Decide only whether the exact source_units "
        "semantically and explicitly support the declared purpose and its supplied arguments. Workflow state "
        "and a pending question are context, never user evidence. For unit_reviews, ignore pending_question entirely: "
        "an explicit mutation or navigation may interrupt the old question, and the coordinator alone decides interruption, invalidation, and resume behavior. "
        "For an action-purpose review with several source_units, the action is supported when at least one unit directly supports the declared purpose and every other mapped unit is purely provenance, format, temporal ordering, or processing scope for that same operation. Such scope units do not have to restate the operation's verb. They may not hide an independent selection, mutation, consultation, contradiction, concrete value owned by another operation, or evidence demand. In particular, structured partial configuration directly supports a configuration-proposal purpose, and a same-turn instruction to ask for remaining required values after review is processing scope for that proposal rather than a second operation. "
        "Pending-question context is present only when a pending-answer purpose itself is being reviewed. A question asking to "
        "summarize, explain, compare, or report retained state is read-only and cannot support "
        "a durable mutation. An evidence-ingestion purpose is supported only when the "
        "source itself contributes an attributable RPC method-schema, request, parameter, "
        "response, or documentation fact; a request to discuss existing evidence is not new "
        "evidence. An explicit statement that a named/current RPC method has no parameters, "
        "describes one or more parameter meanings/types, or describes its response/result is new schema evidence. "
        "For unit_reviews, source_text is the exact unit under review. related_source_units contains only other units in the same turn that share at least one mapped operation; it is relationship context, not permission to ignore an independent demand. A source_text that is purely introductory, trailing, format, provenance, or temporal framing for those related units is complete when the shared mapped operations preserve the related request. If source_text contains its own selection, mutation, consultation, contradiction, correction, value, or evidence demand, that demand must still be represented by mapped_actions. Each mapped_actions row contains an opaque operation index, source-grounded arguments, and its authoritative declared_purpose. Judge whether the set of declared purposes preserves every explicit demand; do not require fields that a declared purpose intentionally collects in later typed questions. Review the proposed operations as one ordered transaction: an earlier purpose that selects a source-grounded wire method may establish the draft referenced by a later evidence-ingestion purpose, but pre-existing workflow state alone cannot. "
        "A custom-RPC-entry purpose is supported when the source explicitly asks to start, add, supply, or configure a custom RPC method workflow. It intentionally carries no endpoint, method identity, or schema payload; requiring those facts at entry would skip later typed collection questions. A discussion-only question about whether custom RPC is possible does not support entry. "
        "A read-only consultation purpose is supported only when the source asks for an answer, explanation, comparison, status, preparation guidance, or similar information. It is not supported when the source explicitly requests only a selection, mutation, navigation, execution, or evidence-ingestion operation. "
        "A group-navigation purpose is supported when the source asks to visit, return to, or configure a named area without making a more specific mutation request. It intentionally does not carry destination fields. It is not a substitute for an explicit request to adjust or customize QPS, RPC workload, endpoint, observability, or another owned setting. "
        "A target-mode-intake purpose is supported when the source has a benchmark or observation goal but leaves fake-node, real-node, or sync-observe unresolved, including explicit indecision between modes. It intentionally asks a later typed question and requires no selected mode in the source. "
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
        "{unit_id, clause_id, source_text, disposition:'action'|'context'|'unresolved', optional scope_constraint:'consultation_only', "
        "action_indexes:[zero-based indexes], reason}. Split prose only when exact contiguous anchors cover every word. "
        "When conjunctions or framing make a lossless split uncertain, use one full-clause unit mapped to every preserving typed action. "
        "Keep structured clauses atomic. Anchors may omit only punctuation or whitespace between units; never omit prose. "
        "Introductory, framing, and trailing prose around a structured block must have its own semantic unit mapped to the block-consuming action, be context only when it contains no present operation and will pass independent context admission, or be explicitly unresolved when the relationship is unclear. Structured clauses can never be context. "
        "structured_candidates are deterministic syntax facts for their clause. When that clause is configuration/review input, preserve config_values and unmapped_values in propose_config_values, route workflow_values through their typed workflow actions, and map the atomic clause to every action needed to preserve it. A same-turn instruction to review or apply that partial configuration and continue asking for required values not supplied is processing scope of propose_config_values; map it to that proposal without adding resume_current_flow, bypassing review, or leaving it unresolved. "
        "An explicit consultation-only, not-starting-yet, or no-change unit sets scope_constraint='consultation_only' and maps to all read-only consultation action indexes it scopes; it is not unresolved. "
        "Never claim an action covers a URL, exact wire RPC method, or concrete fact unless that exact value is present in the mapped owning action."
        "When validation_errors report source anchors that do not cover a prose clause, repair that clause with exactly one semantic unit whose source_text is the complete authoritative clause text and whose action_indexes list every typed action that preserves the clause. "
        "When validation_errors report a missing exact wire RPC method, add the owning rpc_catalog_command set_method action with that exact method; citing the whole sentence as source_evidence on another action does not preserve it."
        "When validation rejects set_method because its method came only from workflow state while a schema-evidence question is active, replace it with rpc_catalog_command append_evidence and copy the exact user description into rpc_schema_evidence; do not require the user to repeat the already-active draft method."
        "When validation reports an incomplete semantic unit, preserve its existing valid actions and add every missing typed action explicitly required by that unit; do not replace the unit with a clarification when action_schema can represent the request."
        "When validation rejects request_qps_customization because the source only asks to visit or configure the QPS area before another area, replace it with change_group(qps_profile) and exact source_evidence; do not leave that representable navigation unresolved."
        "When adjacent clauses reject the current mutually exclusive workflow and explicitly select a replacement, one choose_target_mode action for the replacement may preserve both clauses. Map both semantic units to that same action index; do not invent a cancellation action or leave the rejection unresolved."
        "Never add answer_pending merely because a pending question exists. Add it only when the exact source text actually answers that typed question contract."
    )


def _action_queue_prompt() -> str:
    return build_action_resolver_prompt()


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
    return {
        "user_text": raw,
        "clauses": [clause.as_dict() for clause in clauses],
        "input_shape": {
            "multiline": len(non_empty_lines) > 1,
            "non_empty_line_count": len(non_empty_lines),
            "contains_json_delimiters": "{" in raw and "}" in raw,
        },
        "structured_candidates": structured_candidates,
        "action_schema": action_schema(),
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


def _recover_omitted_chain_selection(
    provider: Any,
    text: str,
    state: AgentGraphState,
) -> str:
    """Recover a source-grounded benchmark chain omitted by a compound plan."""

    payload = _parse_json_object(text)
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    if not actions or (state.get("chain_identity") or {}).get("canonical"):
        return text
    if any(
        isinstance(action, dict)
        and str(action.get("type") or "") in {"choose_chain", "change_chain"}
        for action in actions
    ):
        return text
    mode_indexes = _target_mode_selection_action_indexes(actions, state)
    if not mode_indexes:
        return text
    for mode_index in mode_indexes:
        evidence = str(actions[mode_index].get("source_evidence") or "").strip()
        mention = _extract_chain_mention_with_provider(provider, state, evidence)
        chain_text = str(mention.get("chain_text") or "").strip()
        confidence = str(mention.get("confidence") or "low").strip().lower()
        if (
            mention.get("found") is not True
            or confidence not in {"medium", "high"}
            or not chain_text
            or chain_text.casefold() not in evidence.casefold()
        ):
            continue
        chain_action_index = len(actions)
        actions.append({
            "type": "choose_chain",
            "chain_text": chain_text,
            "source_evidence": evidence,
            "confidence": confidence,
            "reason": str(mention.get("reason") or "specialized chain-target extraction"),
        })
        units = payload.get("semantic_units") if isinstance(payload.get("semantic_units"), list) else []
        for unit in units:
            if not isinstance(unit, dict):
                continue
            indexes = unit.get("action_indexes") if isinstance(unit.get("action_indexes"), list) else []
            if mode_index not in indexes:
                continue
            source = str(unit.get("source_text") or "")
            if chain_text.casefold() not in source.casefold():
                continue
            unit["action_indexes"] = [*indexes, chain_action_index]
            break
        payload["actions"] = actions
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return text


def _target_mode_selection_action_indexes(
    actions: list[Any],
    state: AgentGraphState,
) -> list[int]:
    """Locate operations whose declared contract selects a target mode."""

    pending = state.get("pending_question") or {}
    options = pending.get("options") if isinstance(pending.get("options"), list) else []
    option_actions = {
        str(option.get("value") or ""): option.get("action")
        for option in options
        if isinstance(option, dict)
        and isinstance(option.get("action"), dict)
    }
    indexes: list[int] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or not str(action.get("source_evidence") or "").strip():
            continue
        action_type = str(action.get("type") or "")
        if action_type == "choose_target_mode":
            indexes.append(index)
            continue
        if action_type != "answer_pending":
            continue
        selected = str(action.get("selected_value") or action.get("answer") or "")
        declared = option_actions.get(selected)
        if isinstance(declared, dict) and str(declared.get("type") or "") == "choose_target_mode":
            indexes.append(index)
    return indexes


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


def _parse_action_queue(text: str, *, trusted_metadata: bool = False) -> dict[str, Any]:
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
    return {
        "actions": actions,
        "clause_coverage": payload.get("clause_coverage") or [],
        "reason": payload.get("reason") or payload.get("reasoning") or "",
    }
