"""Semantic action admission authority for the AnyChain Harness."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .action_registry import (
    ACTION_BY_TYPE,
    lifecycle_rejected_action_indexes,
    normalize_action_relations,
    validate_action_contract,
    validate_action_transaction_contract,
    validate_field_intake_admission_receipt,
    validate_proposal_field_receipts,
)
from .domains.chain_rpc_support import is_existing_family_lifecycle
from .input_values import normalize_target_mode, target_mode_evidence_matches
from .invariants import StateInvariantError
from .questions import (
    action_for_value,
    exact_answer as contract_exact_answer,
    pending_option_value_exists as _pending_option_value_exists,
    value_satisfies_pending_contract as _value_satisfies_pending_contract,
)
from .routing import chain_identity_confirmed
from .state import AgentGraphState
from agent.workflows.group_registry import invalidation_targets


def reconcile_admission_coverage(
    receipt: Mapping[str, Any],
    actions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconcile semantic coverage after rejected proposals are removed."""

    reconciled = deepcopy(dict(receipt))
    action_ids_by_unit: dict[str, list[str]] = {}
    for action in actions:
        action_type = str(action.get("type") or "")
        if action_type == "unknown" or action_type not in ACTION_BY_TYPE:
            continue
        action_id = str(action.get("action_id") or "")
        if not action_id:
            continue
        for unit_id in action.get("_source_unit_ids") or []:
            normalized_unit_id = str(unit_id or "")
            if normalized_unit_id:
                action_ids_by_unit.setdefault(normalized_unit_id, []).append(action_id)

    unresolved: list[str] = []
    semantic_units: list[dict[str, Any]] = []
    for raw_unit in reconciled.get("semantic_units") or []:
        unit = deepcopy(dict(raw_unit))
        unit_id = str(unit.get("unit_id") or "")
        if (
            unit_id
            and str(unit.get("disposition") or "") == "action"
            and unit_id not in action_ids_by_unit
        ):
            unit["proposed_action_indexes"] = list(unit.get("action_indexes") or [])
            unit["action_indexes"] = []
            unit["disposition"] = "unresolved"
        if str(unit.get("disposition") or "") == "unresolved" and unit_id:
            unresolved.append(unit_id)
        semantic_units.append(unit)

    omission_checks: list[dict[str, Any]] = []
    for raw_check in reconciled.get("sibling_omission_checks") or []:
        check = deepcopy(dict(raw_check))
        unit_id = str(check.get("unit_id") or "")
        action_ids = list(dict.fromkeys(action_ids_by_unit.get(unit_id, [])))
        check["action_ids"] = action_ids
        if str(check.get("disposition") or "") == "action" and not action_ids:
            check["proposed_action_indexes"] = list(check.get("action_indexes") or [])
            check["action_indexes"] = []
            check["disposition"] = "unresolved"
            check["verdict"] = "unresolved"
        omission_checks.append(check)

    reconciled["semantic_units"] = semantic_units
    reconciled["unresolved_units"] = list(dict.fromkeys(unresolved))
    reconciled["sibling_omission_checks"] = omission_checks
    return reconciled


def _normalized_action_queue(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_actions = payload.get("actions") if isinstance(payload, dict) else None
    if isinstance(raw_actions, dict):
        raw_actions = [raw_actions]
    if not isinstance(raw_actions, list):
        return []
    output: list[dict[str, Any]] = []
    current_actions = [raw for raw in raw_actions[:12] if isinstance(raw, dict)]
    for raw in current_actions:
        try:
            # The resolver is the sole producer of trusted admission receipts.
            # Model documents are validated before those receipts are attached.
            action = validate_action_contract(raw, trusted_metadata=True)
        except ValueError as exc:
            action = {"type": "unknown", "reason": str(exc), "confidence": "low"}
        action_type = str(action.get("type") or action.get("intent") or "unknown").strip()
        action["type"] = action_type
        action["confidence"] = str(action.get("confidence") or "medium").strip().lower()
        output.append(action)
    return output


def _canonical_pending_choice_matches(
    state: AgentGraphState,
    action: Mapping[str, Any],
) -> bool:
    """Verify one admitted semantic choice against its immutable contract."""

    pending = dict(state.get("pending_question") or {})
    selected = action.get("selected_value")
    action_id = str(action.get("_admission_action_id") or "")
    matches = []
    for row in (state.get("turn_context") or {}).get("pending_choice_contracts") or []:
        if not isinstance(row, Mapping):
            continue
        question = row.get("question") if isinstance(row.get("question"), Mapping) else {}
        option = row.get("option") if isinstance(row.get("option"), Mapping) else {}
        provenance = row.get("semantic_units") if isinstance(row.get("semantic_units"), list) else []
        if (
            str(question.get("id") or "") == str(pending.get("id") or "")
            and str(question.get("group") or "") == str(pending.get("group") or "")
            and option.get("selected_value") == selected
            and str(row.get("admission_action_id") or "") == action_id
            and provenance
        ):
            matches.append(row)
    return len(matches) == 1


def _bypasses_canonical_pending_choice(
    state: AgentGraphState,
    action: Mapping[str, Any],
) -> bool:
    """Reject a declared option owner that bypassed canonical admission."""

    if str(action.get("type") or "") == "answer_pending":
        return False
    pending = dict(state.get("pending_question") or {})
    for option in pending.get("options") or []:
        declared = option.get("action") if isinstance(option, Mapping) else None
        if not isinstance(declared, Mapping):
            continue
        if str(declared.get("type") or "") != str(action.get("type") or ""):
            continue
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if spec is None or not spec.pending_option_admission:
            continue
        if all(key == "type" or action.get(key) == value for key, value in declared.items()):
            return True
    return False


def _validate_action_plan(state: AgentGraphState, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate typed plan semantics without reinterpreting user language."""
    if not actions:
        return actions
    prepared = normalize_action_relations([dict(item) for item in actions])
    try:
        validate_action_transaction_contract(prepared)
    except ValueError:
        # The semantic resolver normally repairs this before admission. Keep a
        # fail-closed boundary for stale checkpoints or invalid integrations:
        # clarification is atomic, so no sibling may mutate state.
        prepared = [
            item
            for item in prepared
            if str(item.get("type") or "") == "clarify_unresolved"
        ]
    prepared = [
        item
        for item in prepared
        if not _bypasses_canonical_pending_choice(state, item)
    ]
    for item in prepared:
        if item.get("pending_option_semantic_verified") is True and _canonical_pending_choice_matches(state, item):
            item["selection_contract_verified"] = True
    prepared = _resolve_pending_answer_invalidation_conflicts(state, prepared)
    lifecycle_rejected = lifecycle_rejected_action_indexes(state, prepared)
    if lifecycle_rejected:
        rejected_types = ", ".join(
            str(prepared[index].get("type") or "unknown")
            for index in lifecycle_rejected
        )
        raise StateInvariantError(
            f"actions incompatible with current typed lifecycle state: {rejected_types}"
        )
    identity = state.get("chain_identity") or {}
    if identity.get("case") == "case3" and identity.get("adapter_family") == "unsupported":
        # An unsupported-family handoff cannot also mutate the RPC catalog.
        # The user must explicitly leave Case 3 by changing chain/family first.
        changes_case = any(
            str(item.get("type") or "") in {"choose_chain", "change_chain", "choose_adapter_family"}
            for item in prepared
        )
        if not changes_case:
            prepared = [
                item for item in prepared
                if str(item.get("type") or "") not in {"rpc_catalog_command", "rpc_workload_command"}
            ]
    if any(str(item.get("type") or "") != "unknown" for item in prepared):
        # ``unknown`` is the whole-turn fallback, not an executable sibling.
        # A provider may emit one invalid proposal and other valid actions in
        # the same plan; retaining the normalized fallback would block those
        # valid actions and surface a false "no safe action" response.
        prepared = [item for item in prepared if str(item.get("type") or "") != "unknown"]
    prepared = _drop_conflicting_answer_actions(state, prepared)
    if state.get("target_mode"):
        # A free-standing model request cannot reopen replacement of a
        # confirmed mode. Declared menu choices have already been rebound to
        # answer_pending; explicit user navigation uses change_group.
        prepared = [
            item
            for item in prepared
            if str(item.get("type") or "") != "request_target_mode_selection"
        ]
    admitted: list[dict[str, Any]] = []
    current_answer_checked = False
    for item in prepared:
        if str(item.get("type") or "") != "answer_pending":
            admitted.append(item)
            continue
        # A user turn can answer only the question that existed when the turn
        # began. Future decisions must be represented by domain actions, not by
        # speculative answer_pending entries for questions that do not exist.
        if not current_answer_checked:
            current_answer_checked = True
            if _action_answers_pending_contract(state, item):
                admitted.append(item)
            continue
        continue
    prepared = admitted
    pending_id = str((state.get("pending_question") or {}).get("id") or "")
    # A model plan cannot answer a domain question that does not exist yet.
    # Protocol selection is intentionally a separate user-confirmed decision
    # after an unknown-chain identity choice. Exact menu answers are applied
    # locally and never need this speculative domain action.
    if pending_id not in {"adapter_family_confirm", "custom_rpc_adapter_family_confirm"}:
        prepared = [
            item for item in prepared
            if str(item.get("type") or "") != "choose_adapter_family"
        ]
    pending_answers = [
        item for item in prepared
        if str(item.get("type") or "") == "answer_pending"
    ]
    if pending_answers:
        # One turn may express the current answer both as answer_pending and as
        # its declared domain effect. Drop only that exact duplicate effect.
        # Another operation owned by the same action type (for example a
        # custom-RPC method supplied beside an endpoint answer) is independent
        # durable work and must survive the newly created question barrier.
        prepared = [
            item for item in prepared
            if str(item.get("type") or "") == "answer_pending"
            or not any(
                _action_duplicates_pending_answer_effect(state, item, answer)
                for answer in pending_answers
            )
        ]
    prepared = [
        item
        for item in prepared
        if str(item.get("type") or "") != "choose_target_mode"
        or (
            bool(normalize_target_mode(item.get("target_mode")))
            and (
                item.get("selection_contract_verified") is True
                or item.get("target_mode_semantic_verified") is True
                or target_mode_evidence_matches(
                    item.get("target_mode"),
                    item.get("source_evidence"),
                    state.get("last_user_input"),
                )
            )
        )
    ]
    prepared = [
        item
        for item in prepared
        if str(item.get("type") or "") != "queue_workflow_goal"
        or (
            bool(normalize_target_mode(item.get("target_mode")))
            and bool(str(item.get("goal") or "").strip())
            and bool(str(item.get("source_evidence") or "").strip())
            and str(item.get("source_evidence") or "").strip() in str(state.get("last_user_input") or "")
        )
    ]
    normalized_partial_actions: list[dict[str, Any]] = []
    for item in prepared:
        if str(item.get("type") or "") == "set_qps_override" and not isinstance(item.get("qps_overrides"), dict):
            item = {
                **item,
                "type": "request_qps_customization",
                "qps_fields": item.get("qps_fields") or [],
            }
            item.pop("qps_overrides", None)
        normalized_partial_actions.append(item)
    prepared = normalized_partial_actions
    prepared = [
        item
        for item in prepared
        if str(item.get("type") or "") != "change_group"
        or item.get("selection_contract_verified") is True
        or (
            item.get("navigation_explicit") is True
            and bool(str(item.get("source_evidence") or "").strip())
            and str(item.get("source_evidence") or "").strip() in str(state.get("last_user_input") or "")
        )
    ]
    extension_consultation = any(
        str(item.get("type") or "") == "answer_opening_question"
        and str(item.get("topic") or "").strip().lower() == "extension"
        for item in prepared
    )
    if extension_consultation:
        prepared = [
            item
            for item in prepared
            if str(item.get("type") or "") != "rpc_catalog_command"
            or str(item.get("catalog_command") or "") != "enter"
        ]
    mutation_types = {
        "choose_chain", "change_chain", "set_rpc_mode", "set_qps_mode", "request_qps_customization",
        "set_qps_override", "set_observability", "rpc_catalog_command", "rpc_workload_command",
    }
    has_benchmark_mutation = any(str(item.get("type") or "") in mutation_types for item in prepared)
    chooses_mode = any(
        str(item.get("type") or "") == "choose_target_mode"
        or (
            str(item.get("type") or "") == "answer_pending"
            and _pending_answer_declared_action_type(state, item) == "choose_target_mode"
        )
        for item in prepared
    )
    requests_mode = any(str(item.get("type") or "") == "request_target_mode_selection" for item in prepared)
    if has_benchmark_mutation and not state.get("target_mode") and not chooses_mode and not requests_mode:
        prepared.append({
            "type": "request_target_mode_selection",
            "source_evidence": str(state.get("last_user_input") or ""),
            "confidence": "high",
            "reason": "benchmark actions require an explicit target mode",
        })
    prepared = _drop_redundant_group_navigation(state, prepared)
    return _ensure_action_prerequisites(state, prepared)


def _action_duplicates_pending_answer_effect(
    state: AgentGraphState,
    action: Mapping[str, Any],
    pending_answer: Mapping[str, Any],
) -> bool:
    """Match a sibling action to the current answer's declared exact effect."""

    pending = dict(state.get("pending_question") or {})
    selected = pending_answer.get("selected_value")
    declared = action_for_value(pending, selected)
    if not declared and pending.get("manual_input_allowed") is True:
        manual = pending.get("manual_action")
        if isinstance(manual, Mapping):
            declared = {
                str(key): value
                for key, value in manual.items()
                if str(key) not in {"value_argument", "use_complete_turn"}
            }
            value_argument = str(manual.get("value_argument") or "").strip()
            answer_value = (
                selected
                if selected not in (None, "")
                else pending_answer.get("answer")
            )
            if value_argument and answer_value not in (None, ""):
                declared[value_argument] = answer_value
    if not declared or str(action.get("type") or "") != str(declared.get("type") or ""):
        return False
    effect_fields = {
        key: value
        for key, value in declared.items()
        if key != "type"
        and key != "source_evidence"
        and not key.endswith("_explicit")
        and key not in {"selection_contract_verified", "semantic_purpose_verified"}
    }
    if not effect_fields:
        return False
    return all(
        action.get(key) == value
        for key, value in effect_fields.items()
    )


def _action_satisfies_pending_manual_effect(
    pending: Mapping[str, Any],
    action: Mapping[str, Any],
) -> bool:
    """Return whether one typed action supplies the pending manual contract."""

    manual = pending.get("manual_action")
    if not isinstance(manual, Mapping):
        return False
    if str(action.get("type") or "") != str(manual.get("type") or ""):
        return False
    value_argument = str(manual.get("value_argument") or "").strip()
    fixed = {
        str(key): value
        for key, value in manual.items()
        if str(key) not in {"type", "value_argument", "use_complete_turn"}
    }
    if any(action.get(key) != value for key, value in fixed.items()):
        return False
    return bool(value_argument and action.get(value_argument) not in (None, ""))


def _resolve_pending_answer_invalidation_conflicts(
    state: AgentGraphState,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve pending answers against invalidating sibling mutations."""

    pending_group = str((state.get("pending_question") or {}).get("group") or "").strip()
    if not pending_group:
        return actions
    invalidating: list[dict[str, Any]] = []
    for action in actions:
        if str(action.get("type") or "") == "answer_pending":
            continue
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if spec is None or not spec.mutation_dimension:
            continue
        source_group = str(spec.target_group or spec.mutation_dimension).strip()
        if pending_group in set(invalidation_targets(source_group)):
            invalidating.append(action)
    if invalidating:
        return [
            item for item in actions
            if str(item.get("type") or "") != "answer_pending"
        ]
    return actions


def _drop_redundant_group_navigation(state: AgentGraphState, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove navigation when a concrete action already owns that destination."""

    concrete_groups = {
        "chain_identity": {"choose_chain", "change_chain", "request_chain_selection"},
        "workload_rpc": {"set_rpc_mode", "rpc_catalog_command", "rpc_workload_command", "use_default_workload", "configure_workload_weights"},
        "qps_profile": {"set_qps_mode", "request_qps_customization", "set_qps_override"},
        "observability": {"set_observability"},
        "sync_observe": {
            "set_sync_observe_source",
            "clear_sync_observe_source",
            "set_sync_observe_options",
        },
    }
    action_types = {str(item.get("type") or "") for item in actions}
    output: list[dict[str, Any]] = []
    for item in actions:
        if str(item.get("type") or "") == "change_group":
            group = str(item.get("group") or "")
            if action_types & concrete_groups.get(group, set()):
                continue
            if "answer_pending" in action_types and group == str(state.get("active_group") or ""):
                continue
        output.append(item)
    return output


def _ensure_action_prerequisites(state: AgentGraphState, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add typed prerequisite questions without inventing user choices."""

    output = list(actions)
    action_types = {str(item.get("type") or "") for item in output}
    chain_required = {"set_rpc_mode", "rpc_workload_command", "use_default_workload", "configure_workload_weights"}
    chain_confirmed = chain_identity_confirmed(state)
    identity = state.get("chain_identity") or {}
    catalog_continues_case2 = is_existing_family_lifecycle(identity)
    needs_chain_prerequisite = bool(action_types & chain_required) or (
        "rpc_catalog_command" in action_types and not catalog_continues_case2
    )
    chain_planned = bool(action_types & {"choose_chain", "change_chain", "request_chain_selection"})
    if needs_chain_prerequisite and not chain_confirmed and not chain_planned:
        output.insert(0, {
            "type": "change_group",
            "group": "workload_rpc",
            "navigation_explicit": True,
            "source_evidence": str(state.get("last_user_input") or ""),
            "confidence": "high",
            "reason": "workload actions require confirmed chain identity",
        })
    return output


def _action_answers_pending_contract(state: AgentGraphState, action: dict[str, Any]) -> bool:
    """Reject model answers that bypass the active typed question contract."""

    pending = state.get("pending_question") or {}
    if not pending:
        return False
    if str(pending.get("kind") or "") in {"numbered_choice", "yes_no"}:
        selected = action.get("selected_value")
        if not _pending_option_value_exists(selected, pending):
            answer = str(action.get("answer") or "").strip()
            evidence = str(action.get("source_evidence") or "").strip()
            user_text = str(state.get("last_user_input") or "")
            return bool(
                pending.get("manual_input_allowed") is True
                and answer
                and evidence
                and evidence in user_text
                and answer in evidence
                and _value_satisfies_pending_contract(answer, pending)
                and action.get("semantic_purpose_verified") is True
            )
        if not (
            action.get("selection_contract_verified") is True
            and _canonical_pending_choice_matches(state, action)
        ):
            return False
        return True
    selected = action.get("selected_value")
    if isinstance(selected, str) and not selected.strip():
        selected = None
    raw_answer = selected if selected is not None else action.get("answer")
    answer = str(raw_answer or "").strip()
    evidence = str(action.get("source_evidence") or "").strip()
    user_text = str(state.get("last_user_input") or "")
    declared_option = _pending_option_value_exists(selected, pending)
    if not evidence or evidence not in user_text:
        return False
    if (
        not declared_option
        and answer not in evidence
        and action.get("semantic_purpose_verified") is not True
    ):
        return False
    reserved_identifiers = {
        str(pending.get("id") or "").strip().casefold(),
        str(pending.get("field") or "").strip().casefold(),
        str(pending.get("group") or "").strip().casefold(),
    }
    if answer.casefold() in reserved_identifiers:
        return False
    return bool(
        raw_answer not in (None, "")
        and (
            declared_option
            or _value_satisfies_pending_contract(raw_answer, pending)
        )
    )


def _pending_answer_declared_action_type(state: AgentGraphState, action: dict[str, Any]) -> str:
    pending = state.get("pending_question") or {}
    selected = action.get("selected_value")
    declared = action_for_value(pending, selected)
    return str(declared.get("type") or "")


def _drop_conflicting_answer_actions(state: AgentGraphState, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user_text = str(state.get("last_user_input") or "")
    pending = dict(state.get("pending_question") or {})
    evidenced_consultation_topics = {
        str(item.get("topic") or "").strip().lower()
        for item in actions
        if str(item.get("type") or "") == "answer_opening_question"
        and bool(str(item.get("source_evidence") or "").strip())
        and str(item.get("source_evidence") or "").strip() in user_text
    }
    if evidenced_consultation_topics:
        actions = [
            item
            for item in actions
            if not (
                str(item.get("type") or "") == "answer_opening_question"
                and str(item.get("topic") or "").strip().lower() in evidenced_consultation_topics
                and not (
                    bool(str(item.get("source_evidence") or "").strip())
                    and str(item.get("source_evidence") or "").strip() in user_text
                )
            )
        ]
        raw_selects_pending, _selected_value = contract_exact_answer(user_text, pending)
        if pending and not raw_selects_pending:
            # A read-only consultation is an overlay on the active workflow.
            # The model may not turn that prose into a menu value and thereby
            # consume the pending contract. Exact option selection is owned by
            # the raw-input contract matcher above.
            actions = [
                item
                for item in actions
                if str(item.get("type") or "") != "answer_pending"
                or item.get("pending_option_semantic_verified") is True
            ]
    requests_target_mode = any(
        str(item.get("type") or "") == "request_target_mode_selection"
        for item in actions
    )
    if requests_target_mode and str((state.get("pending_question") or {}).get("id") or "") == "opening_next_action":
        actions = [
            item
            for item in actions
            if str(item.get("type") or "") != "answer_pending"
        ]
    if any(str(item.get("type") or "") == "change_group" for item in actions):
        actions = [item for item in actions if str(item.get("type") or "") != "go_back"]
    if any(str(item.get("type") or "") == "analyze_report" for item in actions):
        actions = [
            item
            for item in actions
            if not (
                str(item.get("type") or "") in {"ask_capabilities", "answer_opening_question"}
                and str(item.get("topic") or "capabilities").strip().lower()
                in {"capabilities", "current_job", "job_status", "execution_status", "evidence_help", "log_help"}
            )
        ]
    consultation_topics = {
        str(item.get("topic") or "").strip().lower()
        for item in actions
        if str(item.get("type") or "") == "answer_opening_question"
    }
    if "workload_config" in consultation_topics:
        workload_will_be_rendered_by_mutation = any(
            str(item.get("type") or "") in {
                "set_rpc_mode",
                "use_default_workload",
                "configure_workload_weights",
                "rpc_catalog_command",
                "rpc_workload_command",
            }
            for item in actions
        )
        actions = [
            item
            for item in actions
            if not (
                str(item.get("type") or "") == "answer_opening_question"
                and (
                    str(item.get("topic") or "").strip().lower()
                    in {"current_config", "config_explanation"}
                    or (
                        workload_will_be_rendered_by_mutation
                        and str(item.get("topic") or "").strip().lower() == "workload_config"
                    )
                )
            )
        ]
        consultation_topics = {
            str(item.get("topic") or "").strip().lower()
            for item in actions
            if str(item.get("type") or "") == "answer_opening_question"
        }
    if "current_config" in consultation_topics:
        # The current-config renderer owns current context and includes the
        # computed next blocker. These two topics are proven output subsets,
        # unlike requirements/workflow questions which must remain independent.
        actions = [
            item
            for item in actions
            if not (
                str(item.get("type") or "") == "answer_opening_question"
                and str(item.get("topic") or "").strip().lower() in {"current_context", "next_action"}
            )
        ]
    topics = {
        str(item.get("topic") or "").strip().lower()
        for item in actions
        if str(item.get("type") or "") == "answer_opening_question"
    }
    specific_chain_consultation = any(
        str(item.get("type") or "") == "answer_opening_question"
        and str(item.get("topic") or "").strip().lower() == "supported_chains"
        and bool(str(item.get("subject") or "").strip())
        for item in actions
    )
    if specific_chain_consultation:
        actions = [
            item
            for item in actions
            if not (
                str(item.get("type") or "") in {"ask_capabilities", "answer_opening_question"}
                and str(item.get("topic") or "capabilities").strip().lower() in {"identity", "capabilities", "agent_capability"}
            )
        ]
        topics = {
            str(item.get("topic") or "").strip().lower()
            for item in actions
            if str(item.get("type") or "") == "answer_opening_question"
        }
    if not (topics & {"performance_benchmark_guidance", "mode_comparison"}):
        return actions
    pruned: list[dict[str, Any]] = []
    for item in actions:
        action_type = str(item.get("type") or "").strip()
        topic = str(item.get("topic") or "").strip().lower()
        target_mode = normalize_target_mode(item.get("target_mode"))
        if action_type == "answer_opening_question" and topic in {"recommendation", "recommend_start"}:
            continue
        if "performance_benchmark_guidance" in topics and action_type == "choose_target_mode" and target_mode == "fake-node":
            continue
        pruned.append(item)
    return pruned


def _has_meaningful_queue(actions: list[dict[str, Any]]) -> bool:
    return any(
        action_type in ACTION_BY_TYPE and action_type != "unknown"
        for item in actions
        if (action_type := str(item.get("type") or "").strip())
    )


def _validate_admission_transaction(
    state: AgentGraphState,
    actions: list[dict[str, Any]],
    *,
    current_submission: bool,
) -> None:
    """Validate Harness receipts against their session and ordered transaction."""

    thread_id = str(state.get("thread_id") or "default")
    session_id = str((state.get("session") or {}).get("id") or thread_id)
    turn_index = int(state.get("turn_index") or 0)
    receipt_actions = [
        action for action in actions
        if isinstance(action, dict) and action.get("_admission_action_id")
    ]
    if receipt_actions:
        declared_orders = {
            tuple(str(value) for value in action.get("_transaction_action_ids") or [])
            for action in receipt_actions
        }
        if len(declared_orders) != 1:
            raise StateInvariantError("admission transaction order metadata is inconsistent")
        declared_order = next(iter(declared_orders))
        actual_order = tuple(str(action.get("_admission_action_id") or "") for action in receipt_actions)
        if actual_order != tuple(value for value in declared_order if value in set(actual_order)):
            raise StateInvariantError("admission transaction action order mismatch")
    for action in actions:
        action_type = str(action.get("type") or "")
        if action_type == "request_config_field_input":
            validate_field_intake_admission_receipt(
                action,
                thread_id=thread_id,
                session_id=session_id,
            )
            receipt = action.get("_semantic_admission_receipt") or {}
            if current_submission and int(receipt.get("submitted_turn_index") or 0) != turn_index:
                raise StateInvariantError("field intake receipt belongs to another turn")
        elif action_type == "propose_config_values":
            validate_proposal_field_receipts(
                action,
                thread_id=thread_id,
                session_id=session_id,
                submitted_turn_index=turn_index if current_submission else None,
            )
