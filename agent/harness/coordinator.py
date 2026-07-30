"""Deterministic group workflow engine for the LangGraph Harness."""

from __future__ import annotations

import json
import hashlib
import re
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
from typing import Any, Mapping

from agent.runners.job_manager import verify_job_receipt
from agent.utils.redaction import redact

from .state import AgentGraphState, PendingQuestion, RESET_PRESERVED_KEYS, new_state
from .secret_refs import (
    discard_draft_secret_references,
    discard_secret_reference,
    discard_state_secret_references,
    materialize_state_secret_references,
    missing_state_secret_bindings,
    reconcile_state_secret_bindings,
    redact_secret_references,
    register_state_secret_bindings,
    resolve_secret_reference,
    restore_secret_reference,
    secret_value_hash,
    store_secret_reference,
)
from .input_identity import user_input_hash
from . import hierarchical_planner
from .bounded_semantic_lane import (
    compile_bounded_semantic_value,
    review_bounded_semantic_plan,
)
from .oracle import (
    compute_next_action,
)
from .routing import (
    group_readiness,
    navigation_prerequisite as _navigation_prerequisite,
    next_group_and_reason,
    option_return_policy as _option_return_policy,
)
from .turns import adjudicate_turn
from .plan_coverage import segment_user_turn
from .action_registry import (
    ACTION_BY_TYPE,
    ACTION_METADATA_FIELDS,
    TRUSTED_ACTION_METADATA_FIELDS,
    action_crosses_pending_barrier,
    action_is_turn_local,
    action_merge_key,
    assign_action_ids,
    lifecycle_rejected_action_indexes,
    merge_semantic_actions,
    resolve_action_target_group,
)
from .contracts import (
    ActionEnvelope,
    ActionProposal,
    CheckpointCommand,
    FailureDescriptor,
    FieldReconfigurationCommand,
    HandlerResult,
    NavigationCommand,
    RecoveryCommand,
    PendingDomainResult,
    SideEffectIntent,
    SideEffectReceipt,
    SemanticDraftCommand,
    TurnReceipt,
    WorkflowGoalCommand,
    StateDelta,
    action_envelope_from_dict,
    action_envelope_to_dict,
    handler_result_from_dict,
    handler_result_to_dict,
    failure_descriptor_to_dict,
    pending_domain_result_from_dict,
    pending_domain_result_to_dict,
    side_effect_intent_to_dict,
    side_effect_receipt_to_dict,
    turn_receipt_to_dict,
    response_fragment_from_dict,
    response_fragment_to_dict,
    ResponseFragment,
)
from .control_receipts import (
    EXECUTION_APPROVAL_CONTRACTS,
    execution_intent_projection,
    execution_side_effect_receipt_id,
    execution_side_effect_projection,
    validate_coordinator_control_receipt,
    validate_domain_control_receipt,
)
from .domains.analysis import (
    JOB_ID_RE,
    is_evidence_completion_command,
    prompt_evidence_collection_waiting,
    should_start_evidence_collection,
)
from .domains.execution import reconcile_execution_state
from .domains.recovery import question_for_recovery
from .failures import (
    domain_blocker_failure_record,
    failure_record_response_fragment,
    failure_response_fragment,
    model_provider_failure_record,
)
from .domains.registry import GROUP_OWNER
from .domains.runtime import DOMAIN_RUNTIME, DomainRuntime
from .invariants import StateInvariantError, apply_state_delta, validate_state
from .transitions import (
    field_confirmation_revision,
    mark_group_reconfigured,
    mark_group_reconfiguring,
)
from .questions import (
    action_for_value,
    exact_answer as contract_exact_answer,
    exact_option_answer as contract_exact_option_answer,
    manual_action_for_value,
    pending_option_value_exists as _pending_option_value_exists,
    validate_pending_question_contract,
    value_satisfies_pending_contract as _value_satisfies_pending_contract,
    choice_question,
    manual_question,
    question_text,
    semantic_draft_question_template,
    secret_reentry_question_template,
    with_pending_question_behavior,
    with_pending_question_created_turn,
)
from .semantic_drafts import (
    build_semantic_draft_finalization_receipt,
    cancel_semantic_plan_draft,
    mark_semantic_plan_draft_stale,
    missing_semantic_draft_secret_bindings,
    resolve_semantic_draft_atom,
    reopen_previous_semantic_draft_atom,
    semantic_draft_question_binding,
    semantic_draft_staleness_reasons,
    validate_semantic_plan_draft,
)
from .response import (
    finalize_turn_response as _finalize_turn_response,
    reset_turn_response,
)
from .response_catalog import render_fragment, semantic_hash
from .queue import (
    action_can_run_while_pending as _action_can_run_while_pending,
    action_requirements as _action_requirements,
    order_action_queue as _order_action_queue,
    state_has_capability as _state_has_capability,
)
from .admission import (
    _action_satisfies_pending_manual_effect,
    _has_meaningful_queue,
    _normalized_action_queue,
    validate_action_plan,
    _validate_admission_transaction,
    reconcile_admission_coverage,
)
from .input_values import normalize_target_mode
from agent.workflows.group_registry import (
    GROUP_ORDER,
    GROUP_SPEC_BY_NAME,
    USER_NAVIGABLE_GROUPS,
    group_for_field,
    group_registry_contract_hash,
    is_user_navigable_group,
    reconfiguration_question_for_field,
)

ALLOWED_GROUPS = frozenset(USER_NAVIGABLE_GROUPS)
_ADMISSION_METADATA_KEYS = (
    "_semantic_admission_receipt",
    "_semantic_consensus_receipt",
    "_replacement_intake_receipt",
    "_proposal_field_receipts",
    "_proposal_transaction_hashes",
    "_admission_action_id",
    "_transaction_action_ids",
    "_plan_transaction_hash",
    "_merged_origin_texts",
    "_semantic_draft_finalization_receipt",
    "_semantic_secret_bindings",
)


def _failure(
    code: str,
    *,
    arguments: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
    retryable: bool = False,
) -> FailureDescriptor:
    return FailureDescriptor(
        code=code,
        arguments=dict(arguments or {}),
        payload=dict(payload or {}),
        source=__name__,
        retryable=retryable,
    )


def _fragment(
    message_id: str,
    *,
    kind: str = "message",
    arguments: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
) -> ResponseFragment:
    return ResponseFragment(
        kind=kind,  # type: ignore[arg-type]
        message_id=message_id,
        arguments=dict(arguments or {}),
        payload=dict(payload or {}),
        source=__name__,
    )


def _replace_response_fragments(
    state: AgentGraphState,
    *fragments: ResponseFragment,
) -> None:
    state["response_fragments"] = [
        response_fragment_to_dict(fragment) for fragment in fragments
    ]


def _append_response_fragments(
    state: AgentGraphState,
    *fragments: ResponseFragment,
) -> None:
    current = list(state.get("response_fragments") or [])
    current.extend(response_fragment_to_dict(fragment) for fragment in fragments)
    state["response_fragments"] = current


def _response_fragment_manifest(state: AgentGraphState) -> list[dict[str, str]]:
    language = str(state.get("language") or "en")
    manifest: list[dict[str, str]] = []
    for raw in state.get("response_fragments") or ():
        if not isinstance(raw, Mapping):
            raise StateInvariantError("response fragment state entry is not a mapping")
        rendered = render_fragment(response_fragment_from_dict(raw), language)
        manifest.append(
            {
                "semantic_hash": rendered.semantic_hash,
                "render_hash": rendered.render_hash,
                "message_id": rendered.message_id,
                "role": rendered.kind,
            }
        )
    return manifest


def apply_coordinator_action(state: AgentGraphState, action: ActionProposal) -> HandlerResult:
    """Apply navigation actions owned by the single workflow coordinator."""

    if action.action_type == "resume_current_flow":
        pending = dict(state.get("pending_question") or {})
        if not pending:
            return HandlerResult(
                blocker=_failure(
                    "harness.failure.coordinator.no_active_question",
                    arguments={"operation": "resume_current_flow"},
                )
            )
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            pending_question=pending,
            completion="unchanged",
            stop_after_response=True,
        )
    if action.action_type == "reenter_secret_reference":
        try:
            binding = {
                "scope_id": str(action.arguments.get("scope_id") or ""),
                "owner_kind": str(action.arguments.get("owner_kind") or ""),
                "owner_id": str(action.arguments.get("owner_id") or ""),
                "owner_revision": int(
                    action.arguments.get("owner_revision") or 0
                ),
                "reference": str(action.arguments.get("reference") or ""),
                "atom_id": str(action.arguments.get("atom_id") or ""),
                "value_hash": str(action.arguments.get("value_hash") or ""),
            }
            pending_binding = dict(
                (state.get("pending_question") or {}).get(
                    "secret_reentry_binding"
                )
                or {}
            )
            if binding != pending_binding:
                raise ValueError("secret re-entry binding is not active")
            if binding["owner_kind"] == "semantic_draft":
                draft = validate_semantic_plan_draft(
                    state.get("semantic_plan_draft") or {}
                )
                expected_bindings = [
                    {
                        "scope_id": str(item.get("draft_id") or ""),
                        "owner_kind": "semantic_draft",
                        "owner_id": str(draft.get("draft_id") or ""),
                        "owner_revision": int(draft.get("revision") or 0),
                        "reference": str(item.get("reference") or ""),
                        "atom_id": str(item.get("atom_id") or ""),
                        "value_hash": str(item.get("value_hash") or ""),
                    }
                    for item in missing_semantic_draft_secret_bindings(draft)
                ]
            elif binding["owner_kind"] == "durable_state":
                expected_bindings = [
                    {
                        **dict(item),
                        "owner_kind": "durable_state",
                        "owner_id": str(item.get("reference") or ""),
                        "owner_revision": 0,
                    }
                    for item in missing_state_secret_bindings(state)
                ]
            else:
                expected_bindings = []
            if binding not in expected_bindings:
                raise ValueError("secret re-entry binding is not active")
            restore_secret_reference(
                str(action.arguments.get("secret_value") or ""),
                reference=binding["reference"],
                draft_id=binding["scope_id"],
                atom_id=binding["atom_id"],
                expected_hash=binding["value_hash"],
            )
        except (TypeError, ValueError):
            return HandlerResult(
                blocker=_failure(
                    "harness.failure.coordinator.semantic_draft_unavailable",
                    arguments={"operation": action.action_type},
                )
            )
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            clear_pending=True,
            completion="completed",
        )
    if action.action_type == "resolve_semantic_draft_atom":
        try:
            draft = validate_semantic_plan_draft(
                state.get("semantic_plan_draft") or {}
            )
            raw_resolution = str(action.arguments.get("resolution") or "").strip()
            safe_resolution = str(redact(raw_resolution)).strip()
            if not safe_resolution:
                raise ValueError("semantic draft resolution cannot be empty")
            draft_id = str(action.arguments.get("draft_id") or "")
            revision = int(action.arguments.get("revision") or 0)
            atom_id = str(action.arguments.get("atom_id") or "")
            if (
                draft_id != str(draft.get("draft_id") or "")
                or revision != int(draft.get("revision") or 0)
                or atom_id != str(draft.get("active_atom_id") or "")
            ):
                raise ValueError("semantic draft resolution binding mismatch")
            stale_reasons = semantic_draft_staleness_reasons(draft, state)
            if stale_reasons:
                return HandlerResult(
                    consumed_action_ids=(action.action_id,),
                    semantic_draft_command=SemanticDraftCommand(
                        operation="invalidate",
                        draft_id=draft_id,
                        revision=revision,
                        reason="|".join(stale_reasons),
                    ),
                    clear_pending=True,
                    response_fragments=(_fragment(
                        "harness.failure.coordinator.semantic_draft_unavailable",
                        kind="error",
                        arguments={"operation": action.action_type},
                    ),),
                    completion="blocked",
                    stop_after_response=True,
                )
            resolution_ref = ""
            resolution_hash = secret_value_hash(raw_resolution)
            if safe_resolution != raw_resolution:
                resolution_ref, resolution_hash = store_secret_reference(
                    raw_resolution,
                    draft_id=draft_id,
                    atom_id=atom_id,
                )
            command = SemanticDraftCommand(
                operation="resolve",
                draft_id=draft_id,
                revision=revision,
                atom_id=atom_id,
                resolution=safe_resolution,
                resolution_hash=resolution_hash,
                resolution_ref=resolution_ref,
            )
            resolved = resolve_semantic_draft_atom(
                draft,
                draft_id=command.draft_id,
                revision=command.revision,
                atom_id=command.atom_id,
                resolution=command.resolution,
                resolution_hash=command.resolution_hash,
                resolution_ref=command.resolution_ref,
            )
        except (TypeError, ValueError):
            return HandlerResult(
                blocker=_failure(
                    "harness.failure.coordinator.semantic_draft_unavailable",
                    arguments={"operation": action.action_type},
                )
            )
        next_question = (
            _semantic_draft_question(resolved)
            if resolved.get("status") == "awaiting_clarification"
            else None
        )
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            semantic_draft_command=command,
            clear_pending=True,
            pending_question=next_question,
            next_group=str((next_question or {}).get("group") or ""),
            completion=(
                "blocked" if next_question is not None else "completed"
            ),
            stop_after_response=next_question is not None,
        )
    if action.action_type == "previous_semantic_draft_atom":
        try:
            draft = validate_semantic_plan_draft(
                state.get("semantic_plan_draft") or {}
            )
            command = SemanticDraftCommand(
                operation="previous",
                draft_id=str(action.arguments.get("draft_id") or ""),
                revision=int(action.arguments.get("revision") or 0),
                atom_id=str(action.arguments.get("atom_id") or ""),
            )
            if (
                command.draft_id != str(draft.get("draft_id") or "")
                or command.revision != int(draft.get("revision") or 0)
            ):
                raise ValueError("semantic draft previous binding mismatch")
            stale_reasons = semantic_draft_staleness_reasons(draft, state)
            if stale_reasons:
                return HandlerResult(
                    consumed_action_ids=(action.action_id,),
                    semantic_draft_command=SemanticDraftCommand(
                        operation="invalidate",
                        draft_id=command.draft_id,
                        revision=command.revision,
                        reason="|".join(stale_reasons),
                    ),
                    clear_pending=True,
                    response_fragments=(_fragment(
                        "harness.failure.coordinator.semantic_draft_unavailable",
                        kind="error",
                        arguments={"operation": action.action_type},
                    ),),
                    completion="blocked",
                    stop_after_response=True,
                )
            previous = reopen_previous_semantic_draft_atom(
                draft,
                draft_id=command.draft_id,
                revision=command.revision,
                atom_id=command.atom_id,
            )
        except (TypeError, ValueError):
            return HandlerResult(
                blocker=_failure(
                    "harness.failure.coordinator.semantic_draft_unavailable",
                    arguments={"operation": action.action_type},
                )
            )
        next_question = _semantic_draft_question(previous)
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            semantic_draft_command=command,
            clear_pending=True,
            pending_question=next_question,
            next_group=str(next_question.get("group") or ""),
            completion="blocked",
            stop_after_response=True,
        )
    if action.action_type == "cancel_semantic_draft":
        try:
            draft = validate_semantic_plan_draft(
                state.get("semantic_plan_draft") or {}
            )
            command = SemanticDraftCommand(
                operation="cancel",
                draft_id=str(action.arguments.get("draft_id") or ""),
                revision=int(action.arguments.get("revision") or 0),
                reason=str(action.arguments.get("reason") or "cancelled"),
            )
            if (
                command.draft_id != str(draft.get("draft_id") or "")
                or command.revision != int(draft.get("revision") or 0)
            ):
                raise ValueError("semantic draft cancellation binding mismatch")
            stale_reasons = semantic_draft_staleness_reasons(draft, state)
            if stale_reasons:
                return HandlerResult(
                    consumed_action_ids=(action.action_id,),
                    semantic_draft_command=SemanticDraftCommand(
                        operation="invalidate",
                        draft_id=command.draft_id,
                        revision=command.revision,
                        reason="|".join(stale_reasons),
                    ),
                    clear_pending=True,
                    response_fragments=(_fragment(
                        "harness.failure.coordinator.semantic_draft_unavailable",
                        kind="error",
                        arguments={"operation": action.action_type},
                    ),),
                    completion="blocked",
                    stop_after_response=True,
                )
        except (TypeError, ValueError):
            return HandlerResult(
                blocker=_failure(
                    "harness.failure.coordinator.semantic_draft_unavailable",
                    arguments={"operation": action.action_type},
                )
            )
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            semantic_draft_command=command,
            clear_pending=True,
            completion="completed",
        )
    if action.action_type == "request_config_field_input":
        field = str(action.arguments.get("config_field") or "").strip()
        group = group_for_field(field)
        question_id = reconfiguration_question_for_field(field)
        if not group or not question_id:
            return HandlerResult(
                blocker=_failure(
                    "harness.failure.coordinator.field_not_registered",
                    arguments={"field": field or "<missing>"},
                )
            )
        runtime = DOMAIN_RUNTIME.get(GROUP_OWNER.get(group, ""))
        question = (
            runtime.field_question_factory(state, group, field)
            if runtime and runtime.field_question_factory
            else None
        )
        owner_spec = GROUP_SPEC_BY_NAME.get(group)
        if (
            not question
            or not owner_spec
            or str(question.get("group") or "") != group
            or str(question.get("id") or "") not in owner_spec.questions
            or str(question.get("field") or "") not in owner_spec.fields
        ):
            return HandlerResult(
                blocker=_failure(
                    "harness.failure.coordinator.field_question_unavailable",
                    arguments={"field": field or "<missing>"},
                )
            )
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            pending_question=question,
            next_group=group,
            field_reconfiguration_command=FieldReconfigurationCommand(
                group=group,
                config_field=field,
                prerequisite_question_id=(
                    str(question.get("id") or "")
                    if str(question.get("field") or "") != field
                    else ""
                ),
            ),
            completion="blocked",
            stop_after_response=True,
        )
    if action.action_type == "change_group":
        group = str(action.arguments.get("group") or "").strip()
        if group not in ALLOWED_GROUPS or not is_user_navigable_group(group):
            return HandlerResult(
                blocker=_failure(
                    "harness.failure.coordinator.group_not_navigable",
                    arguments={"group": group or "<missing>"},
                )
            )
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            navigation_command=NavigationCommand(
                operation="change_group",
                target_group=group,
                origin_group=str(action.arguments.get("queue_origin_group") or "").strip(),
            ),
            completion="completed",
            stop_after_response=True,
        )
    if action.action_type == "go_back":
        draft = dict(state.get("semantic_plan_draft") or {})
        if draft.get("status") == "awaiting_clarification":
            return HandlerResult(
                blocker=_failure(
                    "harness.failure.coordinator.semantic_draft_unavailable",
                    arguments={"operation": action.action_type},
                )
            )
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            navigation_command=NavigationCommand(operation="go_back"),
            completion="completed",
            stop_after_response=True,
        )
    if action.action_type == "queue_workflow_goal":
        target_mode = normalize_target_mode(action.arguments.get("target_mode"))
        goal = str(action.arguments.get("goal") or "").strip()
        source = str(action.arguments.get("source_evidence") or "").strip()
        if not target_mode or not goal or not source:
            return HandlerResult(
                blocker=_failure(
                    "harness.failure.coordinator.workflow_goal_invalid",
                    payload={
                        "target_mode_present": bool(target_mode),
                        "goal_present": bool(goal),
                        "source_evidence_present": bool(source),
                    },
                )
            )
        goals = [dict(item) for item in state.get("workflow_goals") or [] if isinstance(item, dict)]
        candidate = {"target_mode": target_mode, "goal": goal, "source_evidence": source}
        already_queued = any(
            item.get("target_mode") == target_mode and item.get("goal") == goal
            for item in goals
        )
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            workflow_goal_command=(
                None
                if already_queued
                else WorkflowGoalCommand(operation="enqueue", goal=candidate)
            ),
            response_fragments=(
                _fragment(
                    "harness.response.workflow_goal_saved",
                    kind="status",
                    arguments={"target_mode": target_mode, "goal": goal},
                ),
            ),
            completion="completed",
        )
    if action.action_type in {"activate_next_workflow_goal", "discard_next_workflow_goal"}:
        goals = [dict(item) for item in state.get("workflow_goals") or [] if isinstance(item, dict)]
        if not goals:
            return HandlerResult(
                blocker=_failure(
                    "harness.failure.coordinator.workflow_goal_missing"
                )
            )
        goal = goals.pop(0)
        if action.action_type == "discard_next_workflow_goal":
            return HandlerResult(
                consumed_action_ids=(action.action_id,),
                workflow_goal_command=WorkflowGoalCommand(operation="remove_first"),
                response_fragments=(
                    _fragment(
                        "harness.response.workflow_goal_removed",
                        kind="status",
                    ),
                ),
                completion="completed",
            )
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            workflow_goal_command=WorkflowGoalCommand(operation="remove_first"),
            followup_actions=({
                "type": "choose_target_mode",
                "target_mode": str(goal.get("target_mode") or ""),
                "target_mode_explicit": True,
                "source_evidence": str(goal.get("source_evidence") or "").strip(),
                "selection_contract_verified": True,
                "confidence": "high",
            },),
            response_fragments=(
                _fragment(
                    "harness.response.workflow_goal_activated",
                    kind="status",
                    arguments={
                        "goal": str(
                            goal.get("goal") or goal.get("target_mode") or ""
                        )
                    },
                ),
            ),
            completion="completed",
        )
    if action.action_type == "answer_pending":
        return HandlerResult(
            blocker=_failure(
                "harness.failure.coordinator.pending_dispatch_required"
            )
        )
    return HandlerResult(
        blocker=_failure(
            "harness.failure.coordinator.unsupported_action",
            arguments={"action_type": action.action_type},
        )
    )

COORDINATOR_RUNTIME = DomainRuntime(apply_action=apply_coordinator_action)


def _semantic_draft_question(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Build the only question allowed to resolve the active draft atom."""

    validated = validate_semantic_plan_draft(draft)
    atom_id = str(validated.get("active_atom_id") or "")
    atom = next(
        (
            dict(item)
            for item in validated.get("unresolved_atoms") or ()
            if isinstance(item, Mapping)
            and str(item.get("atom_id") or "") == atom_id
        ),
        {},
    )
    if not atom:
        raise StateInvariantError("semantic draft has no active atom record")
    group = str(atom.get("group") or "opening")
    binding = semantic_draft_question_binding(validated)
    template = semantic_draft_question_template()
    atom_ids = [
        str(item.get("atom_id") or "")
        for item in validated.get("unresolved_atoms") or ()
        if isinstance(item, Mapping)
    ]
    active_index = atom_ids.index(atom_id)
    options: list[dict[str, Any]] = []
    if active_index > 0:
        previous_atom_id = atom_ids[active_index - 1]
        previous_spec = next(
            item
            for item in template["options"]
            if item["id"] == "previous"
        )
        options.append({
            "id": previous_spec["id"],
            "value": previous_spec["value"],
            "label": question_text(
                previous_spec["label_message_id"]
            ),
            "action": {
                "type": previous_spec["action_type"],
                "draft_id": binding["draft_id"],
                "revision": binding["revision"],
                "atom_id": previous_atom_id,
            },
            "expected_patch": {},
            "return_policy": previous_spec["return_policy"],
        })
    cancel_spec = next(
        item
        for item in template["options"]
        if item["id"] == "cancel"
    )
    options.append({
        "id": cancel_spec["id"],
        "value": cancel_spec["value"],
        "label": question_text(
            cancel_spec["label_message_id"]
        ),
        "action": {
            "type": cancel_spec["action_type"],
            "draft_id": binding["draft_id"],
            "revision": binding["revision"],
            "reason": "user_cancelled",
        },
        "expected_patch": {},
        "return_policy": cancel_spec["return_policy"],
    })
    return choice_question(
        group,
        f"semantic_draft:{binding['draft_id']}:{binding['revision']}:{atom_id}",
        question_text(
            template["prompt_message_id"],
            source=str(atom.get("source_text") or atom.get("source_path") or ""),
        ),
        owner="coordinator",
        field=template["field"],
        kind=template["kind"],
        options=options,
        manual_input_allowed=True,
        accepted_action_types=(template["manual_action_type"],),
        manual_action={
            "type": template["manual_action_type"],
            "draft_id": binding["draft_id"],
            "revision": binding["revision"],
            "atom_id": binding["atom_id"],
            "value_argument": template["manual_value_argument"],
        },
        queue_barrier=template["queue_barrier"],
        barrier_policy=template["barrier_policy"],
        validation=template["validation"],
        semantic_draft_binding=binding,
    )


def _secret_reentry_question(
    group: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one signed re-entry contract for lost process-local material."""

    exact_binding = {
        "scope_id": str(binding.get("scope_id") or ""),
        "owner_kind": str(binding.get("owner_kind") or ""),
        "owner_id": str(binding.get("owner_id") or ""),
        "owner_revision": int(binding.get("owner_revision") or 0),
        "reference": str(binding.get("reference") or ""),
        "atom_id": str(binding.get("atom_id") or ""),
        "value_hash": str(binding.get("value_hash") or ""),
    }
    template = secret_reentry_question_template()
    return manual_question(
        str(group or "opening"),
        (
            "secret_reentry:"
            f"{exact_binding['owner_kind']}:{exact_binding['owner_id']}:"
            f"{exact_binding['atom_id']}"
        ),
        question_text(template["prompt_message_id"]),
        owner="coordinator",
        field=template["field"],
        kind=template["kind"],
        accepted_action_types=(template["manual_action_type"],),
        manual_action={
            "type": template["manual_action_type"],
            **exact_binding,
            "value_argument": template["manual_value_argument"],
        },
        queue_barrier=template["queue_barrier"],
        barrier_policy=template["barrier_policy"],
        validation=template["validation"],
        secret_reentry_binding=exact_binding,
    )


def _set_turn_phase(state: AgentGraphState, phase: str, reason: str = "") -> AgentGraphState:
    control = dict(state.get("control") or {})
    control["phase"] = phase
    if reason:
        control["reason"] = reason
    state["control"] = control
    return state


def _record_admitted_action(
    state: AgentGraphState,
    action: Mapping[str, Any],
    *,
    source: str,
) -> None:
    """Record only actions admitted for the current user turn.

    ``completed_actions`` is execution history for the current graph pass and
    can be replaced by reset-style transitions.  Live evidence therefore uses
    this turn-scoped admission ledger as the authoritative control-plane fact.
    """

    action_type = str(
        action.get("type")
        or action.get("action_type")
        or ""
    ).strip()
    if not action_type:
        return
    turn = state.setdefault("turn_context", {})
    admitted = turn.setdefault("admitted_actions", [])
    action_id = str(action.get("action_id") or "").strip()
    identity = (action_type, action_id)
    if any(
        (str(item.get("type") or ""), str(item.get("action_id") or "")) == identity
        for item in admitted
        if isinstance(item, Mapping)
    ):
        return
    admitted_action = {
        "type": action_type,
        "action_id": action_id,
        "source": source,
    }
    spec = ACTION_BY_TYPE.get(action_type)
    if spec is not None:
        admitted_action["owner"] = spec.owner
        admitted_action["effect"] = spec.effect
        admitted_action["group"] = resolve_action_target_group(action)
    envelope_arguments = action.get("arguments")
    if (
        str(action.get("action_type") or "")
        and isinstance(envelope_arguments, Mapping)
    ):
        argument_payload = {
            str(key): value
            for key, value in envelope_arguments.items()
            if str(key)
        }
    else:
        argument_payload = {
            str(key): value
            for key, value in action.items()
            if key not in ACTION_METADATA_FIELDS
            and key not in TRUSTED_ACTION_METADATA_FIELDS
            and not str(key).startswith("_")
        }
    admitted_action["argument_names"] = sorted(argument_payload)
    admitted_action["arguments_hash"] = hashlib.sha256(
        json.dumps(
            argument_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    admitted_action["argument_value_hashes"] = {
        key: hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        for key, value in sorted(argument_payload.items())
    }
    admitted_action["source_unit_ids"] = [
        str(item)
        for item in (
            action.get("source_unit_ids")
            or action.get("_source_unit_ids")
            or ()
        )
        if str(item)
    ]
    source_evidence = str(action.get("source_evidence") or "")
    admitted_action["source_hash"] = hashlib.sha256(
        source_evidence.encode("utf-8")
    ).hexdigest()
    admission_receipt = action.get("_semantic_admission_receipt")
    if isinstance(admission_receipt, Mapping):
        admitted_action["admission_receipt_id"] = str(
            admission_receipt.get("receipt_id") or ""
        )
    if action_type == "change_group":
        target_group = str(
            action.get("group")
            or (action.get("arguments") or {}).get("group")
            or ""
        ).strip()
        if target_group:
            admitted_action["group"] = target_group
    admitted.append(admitted_action)
    receipt = dict(state.get("turn_receipt") or {})
    if not str(receipt.get("turn_id") or ""):
        return
    admitted_ids = list(receipt.get("admitted_action_ids") or [])
    semantic_order = list(receipt.get("semantic_order") or [])
    owners = dict(receipt.get("owner_bindings") or {})
    action_units = {
        str(key): list(value)
        for key, value in dict(receipt.get("action_unit_bindings") or {}).items()
    }
    unit_actions = {
        str(key): list(value)
        for key, value in dict(receipt.get("unit_action_bindings") or {}).items()
    }
    if action_id and action_id not in admitted_ids:
        admitted_ids.append(action_id)
        semantic_order.append(action_id)
    if action_id and spec is not None:
        owners[action_id] = (
            GROUP_OWNER.get(
                str((state.get("pending_question") or {}).get("group") or ""),
                spec.owner,
            )
            if action_type == "answer_pending"
            else spec.owner
        )
    source_unit_ids = [
        str(item)
        for item in (
            action.get("source_unit_ids")
            or action.get("_source_unit_ids")
            or ()
        )
        if str(item)
    ]
    if action_id and source_unit_ids:
        action_units[action_id] = list(dict.fromkeys(source_unit_ids))
        for unit_id in source_unit_ids:
            bindings = unit_actions.setdefault(unit_id, [])
            if action_id not in bindings:
                bindings.append(action_id)
    receipt["admitted_action_ids"] = admitted_ids
    receipt["semantic_order"] = semantic_order
    receipt["owner_bindings"] = owners
    receipt["action_unit_bindings"] = action_units
    receipt["unit_action_bindings"] = unit_actions
    receipt["status"] = "planned"
    state["turn_receipt"] = receipt


def _receipt_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _invalidate_semantic_draft(
    state: AgentGraphState,
    draft: Mapping[str, Any],
    *,
    reasons: tuple[str, ...],
) -> None:
    state["semantic_plan_draft"] = mark_semantic_plan_draft_stale(
        draft,
        reasons=reasons,
    )
    discard_draft_secret_references(draft)


def _append_control_receipt(
    state: AgentGraphState,
    receipt_type: str,
    payload: Mapping[str, Any],
) -> None:
    """Append one secret-free observation emitted by the current owner."""

    body = {
        "receipt_type": str(receipt_type),
        "turn_index": int(state.get("turn_index") or 0),
        **deepcopy(dict(payload)),
    }
    body["receipt_id"] = _receipt_hash(body)
    turn_context = dict(state.get("turn_context") or {})
    receipts = [
        dict(item)
        for item in turn_context.get("control_receipts") or ()
        if isinstance(item, Mapping)
    ]
    if not any(item.get("receipt_id") == body["receipt_id"] for item in receipts):
        receipts.append(body)
    turn_context["control_receipts"] = receipts
    state["turn_context"] = turn_context


def _record_pending_resolution(
    state: AgentGraphState,
    pending: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    input_text: str,
    resolution_path: str,
) -> None:
    """Record one accepted pending answer independently of its entry path."""

    selected = action.get("selected_value")
    if selected is None:
        selected = action.get("answer")
    if selected is None:
        manual = pending.get("manual_action")
        if (
            isinstance(manual, Mapping)
            and str(action.get("type") or "")
            == str(manual.get("type") or "")
        ):
            value_argument = str(
                manual.get("value_argument") or ""
            ).strip()
            if value_argument:
                selected = action.get(value_argument)
    option = next(
        (
            item
            for item in pending.get("options") or ()
            if isinstance(item, Mapping) and item.get("value") == selected
        ),
        {},
    )
    _append_control_receipt(
        state,
        "pending_resolution",
        {
            "pending_id": str(pending.get("id") or ""),
            "pending_group": str(pending.get("group") or ""),
            "pending_contract_hash": _receipt_hash(pending),
            "resolution_path": resolution_path,
            "selected_option_id": str(
                option.get("id") or option.get("value") or ""
            ),
            "selected_value_hash": _receipt_hash(selected),
            "resolved_action_id": str(action.get("action_id") or ""),
            "input_hash": user_input_hash(input_text),
            "normalizer": str(
                (pending.get("validation") or {}).get("normalization")
                or (
                    "exact_contract"
                    if resolution_path == "exact_contract"
                    else "declared_value_type"
                )
            ),
            "verdict": "accepted",
        },
    )


def _execution_authorization_receipt(
    state: AgentGraphState,
    action: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the canonical user approval before any external side effect."""

    approval_action_type = str(
        action.get("type") or action.get("action_type") or ""
    )
    approval_contracts: dict[str, set[str]] = {}
    for question_id, action_type in EXECUTION_APPROVAL_CONTRACTS.items():
        approval_contracts.setdefault(action_type, set()).add(question_id)
    if approval_action_type not in approval_contracts:
        raise StateInvariantError("action is not a registered execution approval")
    turn_index = int(state.get("turn_index") or 0)
    context = dict(state.get("turn_context") or {})
    pending_receipts = [
        dict(item)
        for item in context.get("control_receipts") or ()
        if isinstance(item, Mapping)
        and item.get("receipt_type") == "pending_resolution"
        and item.get("pending_id") in approval_contracts[approval_action_type]
        and item.get("verdict") == "accepted"
    ]
    valid_pending = []
    for receipt in pending_receipts:
        valid, _reason = validate_coordinator_control_receipt(
            receipt,
            turn_index=turn_index,
        )
        if valid:
            valid_pending.append(receipt)
    if len(valid_pending) != 1:
        raise StateInvariantError(
            "execution approval requires one canonical pending-resolution receipt"
        )
    pending_receipt = valid_pending[0]
    answer_actions = [
        dict(item)
        for item in state.get("completed_actions") or ()
        if isinstance(item, Mapping)
        and str(item.get("action_id") or "")
        == str(pending_receipt.get("resolved_action_id") or "")
        and str(item.get("type") or item.get("action_type") or "")
        == "answer_pending"
    ]
    answer_value = (
        answer_actions[0].get("selected_value")
        if answer_actions and "selected_value" in answer_actions[0]
        else answer_actions[0].get("answer")
        if answer_actions
        else None
    )
    if (
        len(answer_actions) != 1
        or answer_value is not True
        or pending_receipt.get("selected_value_hash") != _receipt_hash(True)
    ):
        raise StateInvariantError(
            "execution approval requires one canonical affirmative answer"
        )
    return pending_receipt


def _record_execution_approval(
    state: AgentGraphState,
    action: Mapping[str, Any],
) -> None:
    """Bind one committed execution approval to its answer, plan, and revision."""

    approval_action_type = str(
        action.get("type") or action.get("action_type") or ""
    )
    if approval_action_type not in set(EXECUTION_APPROVAL_CONTRACTS.values()):
        return
    pending_receipt = _execution_authorization_receipt(state, action)
    context = dict(state.get("turn_context") or {})
    revision = dict(context.get("repository_revision") or {})
    plan = dict(state.get("plan") or {})
    job_id = str((state.get("job") or {}).get("job_id") or "")
    execution_request_id = str(
        (state.get("preflight") or {}).get("execution_request_id") or ""
    )
    if not plan or not job_id:
        raise StateInvariantError(
            "execution approval completed without a plan or submitted job"
        )
    if not revision or not execution_request_id:
        raise StateInvariantError(
            "submitted execution approval is missing revision or request evidence"
        )
    intent = dict(state.get("side_effect_intent") or {})
    side_effect_receipt = dict(state.get("side_effect_receipt") or {})
    execution_receipts = dict(
        (state.get("job") or {}).get("execution_receipts") or {}
    )
    submission_receipt = dict(
        execution_receipts.get("submission_attempt")
        or execution_receipts.get("submission")
        or {}
    )
    expected_action_id = str(action.get("action_id") or action.get("id") or "")
    expected_request_fingerprint = _receipt_hash(intent.get("request") or {})
    approved_plan_hash = _receipt_hash(plan)
    if (
        not intent
        or intent.get("status") != "succeeded"
        or intent.get("action_id") != expected_action_id
        or intent.get("operation") != approval_action_type
        or intent.get("execution_request_id") != execution_request_id
        or intent.get("idempotency_key") != f"harness:{execution_request_id}"
        or intent.get("request_fingerprint") != expected_request_fingerprint
        or not side_effect_receipt
        or side_effect_receipt.get("status") != "succeeded"
        or side_effect_receipt.get("intent_id") != intent.get("intent_id")
        or side_effect_receipt.get("action_id") != expected_action_id
        or side_effect_receipt.get("idempotency_key")
        != intent.get("idempotency_key")
        or side_effect_receipt.get("job_id") != job_id
        or side_effect_receipt.get("receipt_id")
        != execution_side_effect_receipt_id(side_effect_receipt)
        or not verify_job_receipt(submission_receipt)
        or submission_receipt.get("job_id") != job_id
        or submission_receipt.get("approved_plan_hash") != approved_plan_hash
    ):
        raise StateInvariantError(
            "execution approval lacks a canonical side-effect submission chain"
        )
    _append_control_receipt(
        state,
        "execution_approval",
        {
            "approval_question_id": str(pending_receipt["pending_id"]),
            "pending_resolution_receipt_id": str(pending_receipt["receipt_id"]),
            "answer_action_id": str(pending_receipt["resolved_action_id"]),
            "approval_action_id": expected_action_id,
            "approval_action_type": approval_action_type,
            "execution_request_id": execution_request_id,
            "side_effect_intent_id": str(intent["intent_id"]),
            "side_effect_intent_hash": _receipt_hash(
                execution_intent_projection(intent)
            ),
            "side_effect_receipt_id": str(side_effect_receipt["receipt_id"]),
            "side_effect_receipt_hash": _receipt_hash(
                execution_side_effect_projection(side_effect_receipt)
            ),
            "idempotency_key_hash": _receipt_hash(intent["idempotency_key"]),
            "request_fingerprint": str(intent["request_fingerprint"]),
            "job_submission_receipt_id": str(
                submission_receipt["receipt_id"]
            ),
            "job_submission_receipt_hash": _receipt_hash(
                submission_receipt
            ),
            "approved_plan_hash": approved_plan_hash,
            "repository_revision": revision,
            "plan_hash": _receipt_hash(plan),
            "workflow_type": str(state.get("workflow_mode") or ""),
            "target_mode": str(state.get("target_mode") or ""),
            "job_id": job_id,
        },
    )


def _append_domain_control_receipts(
    state: AgentGraphState,
    receipts_to_commit: tuple[Mapping[str, Any], ...],
    *,
    owner: str,
) -> None:
    """Commit only original, owner-validated domain observation receipts."""

    turn_index = int(state.get("turn_index") or 0)
    turn_context = dict(state.get("turn_context") or {})
    receipts = [
        dict(item)
        for item in turn_context.get("control_receipts") or ()
        if isinstance(item, Mapping)
    ]
    known_ids = {
        str(item.get("receipt_id") or "")
        for item in receipts
        if str(item.get("receipt_id") or "")
    }
    for raw in receipts_to_commit:
        receipt = deepcopy(dict(raw))
        valid, reason = validate_domain_control_receipt(
            receipt,
            handler_owner=owner,
            turn_index=turn_index,
        )
        if not valid:
            receipt_identity = "/".join(
                value
                for value in (
                    str(receipt.get("receipt_type") or ""),
                    str(receipt.get("command") or ""),
                )
                if value
            )
            raise StateInvariantError(
                "invalid domain control receipt"
                f" ({receipt_identity or 'unknown'}): {reason}"
            )
        receipt_id = str(receipt.get("receipt_id") or "")
        if receipt_id not in known_ids:
            receipts.append(receipt)
            known_ids.add(receipt_id)
    turn_context["control_receipts"] = receipts
    state["turn_context"] = turn_context


def _record_semantic_plan_receipt(
    state: AgentGraphState,
    semantic_units: list[Mapping[str, Any]],
    pending_choice_contracts: list[Mapping[str, Any]],
) -> None:
    """Persist lossless unit coverage before any admitted action can execute."""

    receipt = dict(state.get("turn_receipt") or {})
    if not receipt:
        return
    new_units = [
        deepcopy(dict(unit))
        for unit in semantic_units
        if str(unit.get("unit_id") or "")
    ]
    normalized_units = new_units
    if receipt.get("admitted_action_ids"):
        normalized_units = [
            deepcopy(dict(unit))
            for unit in receipt.get("semantic_units") or []
            if isinstance(unit, Mapping) and str(unit.get("unit_id") or "")
        ]
        index_by_id = {
            str(unit["unit_id"]): index
            for index, unit in enumerate(normalized_units)
        }
        for unit in new_units:
            unit_id = str(unit["unit_id"])
            prior = index_by_id.get(unit_id)
            if prior is None:
                index_by_id[unit_id] = len(normalized_units)
                normalized_units.append(unit)
            else:
                normalized_units[prior] = unit
    unresolved = [
        str(unit.get("unit_id") or "")
        for unit in normalized_units
        if str(unit.get("disposition") or "") == "unresolved"
    ]
    omission_checks: list[dict[str, Any]] = []
    for unit in normalized_units:
        unit_id = str(unit.get("unit_id") or "")
        disposition = str(unit.get("disposition") or "")
        indexes = [
            int(index)
            for index in unit.get("action_indexes") or []
            if isinstance(index, int) and not isinstance(index, bool)
        ]
        verdict = (
            "covered"
            if disposition == "action" and indexes
            else "context"
            if disposition == "context" and not indexes
            else "unresolved"
            if disposition == "unresolved" and not indexes
            else "invalid"
        )
        omission_checks.append({
            "unit_id": unit_id,
            "disposition": disposition,
            "action_indexes": indexes,
            "verdict": verdict,
        })
    pending_verdicts = [
        deepcopy(dict(row))
        for row in receipt.get("pending_candidate_verdicts") or []
        if isinstance(row, Mapping)
    ]
    pending_verdicts.extend([
        {
            "action_index": contract.get("action_index"),
            "admission_action_id": str(
                contract.get("admission_action_id") or ""
            ),
            "candidate_value": deepcopy(contract.get("candidate_value")),
            "semantic_unit_ids": [
                str(unit.get("unit_id") or "")
                for unit in contract.get("semantic_units") or []
                if isinstance(unit, Mapping)
                and str(unit.get("unit_id") or "")
            ],
            "verdict": "admitted",
        }
        for contract in pending_choice_contracts
        if isinstance(contract, Mapping)
    ])
    receipt["semantic_units"] = normalized_units
    receipt["unresolved_units"] = unresolved
    receipt["pending_candidate_verdicts"] = pending_verdicts
    receipt["sibling_omission_checks"] = omission_checks
    state["turn_receipt"] = receipt


def prepare_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Create one immutable turn snapshot before any routing or mutation."""

    state = _copy_state(state)
    state["action_queue"] = _durable_actions(state.get("action_queue") or [])
    resumed_phase = _inflight_transition_phase(state)
    if resumed_phase:
        validate_state(state)
        return _set_turn_phase(
            state,
            resumed_phase,
            "inflight_transition_resumed",
        )
    state = _apply_handler_result(state, reconcile_execution_state(state), owner="execution")
    try:
        reconcile_state_secret_bindings(state)
    except ValueError as exc:
        raise StateInvariantError(str(exc)) from exc
    pending = dict(state.get("pending_question") or {})
    pending_owner = str(pending.get("group") or "").strip()
    if (
        pending_owner
        and not pending.get("semantic_draft_binding")
        and not pending.get("secret_reentry_binding")
    ):
        state["active_group"] = pending_owner
    state["turn_index"] = int(state.get("turn_index") or 0) + 1
    text = str(state.get("last_user_input") or "").strip()
    clauses = segment_user_turn(text)
    clause_shapes = {str(clause.input_shape or "prose") for clause in clauses}
    input_shape = (
        "structured"
        if clause_shapes == {"structured"}
        else "mixed"
        if "structured" in clause_shapes
        else "prose"
    )
    state["input_shape"] = input_shape
    reset_turn_response(state)
    state["current_action"] = {}
    state["completed_actions"] = []
    state["action_errors"] = []
    turn = adjudicate_turn(state, text)
    state["turn_context"] = {
        "id": int(state.get("turn_index") or 0),
        "kind": str(turn.kind),
        "text": text,
        "input_shape": input_shape,
        "origin_group": str(state.get("active_group") or ""),
        "pending_snapshot": dict(state.get("pending_question") or {}),
        "admitted_actions": [],
    }
    turn_id = (
        f"{state.get('thread_id') or 'default'}:"
        f"{int(state.get('turn_index') or 0)}"
    )
    state["turn_receipt"] = turn_receipt_to_dict(
        TurnReceipt(
            turn_id=turn_id,
            input_hash=user_input_hash(text),
            language=str(state.get("language") or "en"),
            input_shape=input_shape,
            clauses=tuple(clause.as_dict() for clause in clauses),
            pending_before=deepcopy(state.get("pending_question") or {}),
        )
    )
    state["semantic_planning"] = {}
    state["proposed_actions"] = []
    missing_durable_secrets = missing_state_secret_bindings(state)
    if missing_durable_secrets and not (
        state.get("pending_question") or {}
    ).get("secret_reentry_binding"):
        binding = missing_durable_secrets[0]
        _install_pending_question(
            state,
            _secret_reentry_question(
                str(state.get("active_group") or "opening"),
                {
                    **binding,
                    "owner_kind": "durable_state",
                    "owner_id": binding["reference"],
                    "owner_revision": 0,
                },
            ),
            activate_group=False,
        )
        if state.get("action_queue"):
            _set_pending_queue_resume(state, enabled=True)
        return _set_turn_phase(
            state,
            "compose",
            "durable_secret_reentry_required",
        )
    return _set_turn_phase(state, "adjudicate")


def _inflight_transition_phase(state: AgentGraphState) -> str:
    """Return the first incomplete checkpointed action phase, if any."""

    selected = dict(state.get("selected_action") or {})
    if not selected:
        return ""
    if state.get("pending_domain_result"):
        return "commit"
    intent = dict(state.get("side_effect_intent") or {})
    receipt = dict(state.get("side_effect_receipt") or {})
    if receipt:
        return "commit_receipt"
    if not intent:
        return "execute"
    status = str(intent.get("status") or "")
    if status == "prepared":
        return "invoke_effect"
    if status == "invoking":
        return "perform_effect"
    return ""


def adjudicate_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Handle only deterministic contracts; free text proceeds to planning."""

    text = str((state.get("turn_context") or {}).get("text") or "")
    runtime_action = dict(
        (state.get("turn_context") or {}).get("runtime_action") or {}
    )
    if runtime_action:
        return _admit_deterministic_action(
            state,
            runtime_action,
            source="runtime_command",
        )
    turn_kind = str((state.get("turn_context") or {}).get("kind") or "free_text")
    collecting = state.get("evidence_collection") or {}

    if turn_kind == "evidence_continuation":
        if not text:
            state = _apply_handler_result(state, prompt_evidence_collection_waiting(state, collecting), owner="analysis")
            return _set_turn_phase(state, "compose", "evidence_waiting")
        if is_evidence_completion_command(text):
            return _admit_deterministic_action(
                state,
                {
                    "type": "finish_evidence_collection",
                    "source_evidence": text,
                    "confidence": "high",
                },
                source="evidence_transport",
            )
        if "\n" in text or "\r" in text:
            return _admit_deterministic_action(
                state,
                {
                    "type": "append_evidence_collection",
                    "evidence": text,
                    "source_evidence": text,
                    "confidence": "high",
                },
                source="evidence_transport",
            )
        return _set_turn_phase(state, "plan", "typed_evidence_collection_turn")

    if turn_kind == "empty":
        return _set_turn_phase(state, "fallback", "empty_turn")

    pending = state.get("pending_question") or {}
    # Evidence framing is a terminal transport concern: collect a complete
    # multiline block before asking the semantic planner to classify it. The
    # completed block re-enters the normal semantic owner as one turn.
    if pending and str(pending.get("kind") or "") == "evidence" and should_start_evidence_collection(text):
        return _admit_deterministic_action(
            state,
            {
                "type": "start_evidence_collection",
                "evidence": text,
                "source_evidence": text,
                "confidence": "high",
            },
            source="evidence_transport",
        )

    matched, selected_value = (
        contract_exact_option_answer(text, pending)
        if pending
        else (False, None)
    )
    if matched:
        resume_queue = pending.get("resume_action_queue") is True
        action = {
            "type": "answer_pending",
            "answer": text,
            "source_evidence": text,
            "confidence": "high",
            "selection_contract_verified": True,
            "selected_value": selected_value,
        }
        action = assign_action_ids(
            f"{state.get('thread_id') or 'default'}:{int(state.get('turn_index') or 0)}",
            text,
            [action],
        )[0]
        _record_pending_resolution(
            state,
            pending,
            action,
            input_text=text,
            resolution_path="exact_contract",
        )
        source_clauses = [
            dict(clause)
            for clause in (state.get("turn_receipt") or {}).get("clauses") or []
            if isinstance(clause, Mapping)
            and str(clause.get("clause_id") or "")
        ]
        source_unit_ids = [
            str(clause.get("clause_id") or "")
            for clause in source_clauses
        ]
        action["_source_unit_ids"] = source_unit_ids
        _record_semantic_plan_receipt(
            state,
            [
                {
                    "unit_id": str(clause.get("clause_id") or ""),
                    "clause_id": str(clause.get("clause_id") or ""),
                    "source_text": str(clause.get("text") or ""),
                    "start": 0,
                    "end": len(str(clause.get("text") or "")),
                    "disposition": "action",
                    "action_indexes": [0],
                }
                for clause in source_clauses
            ],
            [],
        )
        scope = (
            f"{state.get('thread_id') or 'default'}:"
            f"{int(state.get('turn_index') or 0)}"
        )
        action = action_envelope_to_dict(
            _build_action_envelope(
                state,
                action,
                semantic_order=0,
                submitted_turn_index=int(state.get("turn_index") or 0),
                origin_group=str(state.get("active_group") or ""),
                origin_text=text,
                plan_scope=scope,
            )
        )
        existing = (
            [dict(item) for item in state.get("action_queue") or [] if isinstance(item, Mapping)]
            if resume_queue
            else []
        )
        state["action_queue"] = [action, *existing]
        _record_admitted_action(
            state,
            action,
            source="pending_question_contract",
        )
        return _set_turn_phase(state, "execute", "pending_answer_admitted")

    manual_matched, manual_value = (
        contract_exact_answer(text, pending)
        if pending.get("structured_input_owner") is True
        else (False, None)
    )
    if manual_matched:
        manual_action = manual_action_for_value(pending, manual_value)
        if manual_action:
            spec = ACTION_BY_TYPE.get(str(manual_action.get("type") or ""))
            if (
                spec is not None
                and "source_evidence" in spec.allowed_arguments
                and not str(manual_action.get("source_evidence") or "").strip()
            ):
                manual_action["source_evidence"] = text
            return _admit_deterministic_action(
                state,
                manual_action,
                source="pending_question_manual_contract",
            )

    return _set_turn_phase(state, "plan", "semantic_input")


def _admit_deterministic_action(
    state: AgentGraphState,
    action: Mapping[str, Any],
    *,
    source: str,
) -> AgentGraphState:
    """Admit one exact local contract through the normal graph lifecycle."""

    text = str((state.get("turn_context") or {}).get("text") or "")
    scope = f"{state.get('thread_id') or 'default'}:{int(state.get('turn_index') or 0)}"
    admitted = assign_action_ids(scope, text, [dict(action)])[0]
    pending = dict(state.get("pending_question") or {})
    if source == "pending_question_manual_contract" and pending:
        _record_pending_resolution(
            state,
            pending,
            admitted,
            input_text=text,
            resolution_path="typed_manual_value",
        )
    source_clauses = [
        dict(clause)
        for clause in (state.get("turn_receipt") or {}).get("clauses") or []
        if isinstance(clause, Mapping)
        and str(clause.get("clause_id") or "")
    ]
    source_unit_ids = [
        str(clause.get("clause_id") or "")
        for clause in source_clauses
    ]
    admitted["_source_unit_ids"] = source_unit_ids
    _record_semantic_plan_receipt(
        state,
        [
            {
                "unit_id": str(clause.get("clause_id") or ""),
                "clause_id": str(clause.get("clause_id") or ""),
                "source_text": str(clause.get("text") or ""),
                "start": 0,
                "end": len(str(clause.get("text") or "")),
                "disposition": "action",
                "action_indexes": [0],
            }
            for clause in source_clauses
        ],
        [],
    )
    admitted = action_envelope_to_dict(
        _build_action_envelope(
            state,
            admitted,
            semantic_order=0,
            submitted_turn_index=int(state.get("turn_index") or 0),
            origin_group=str(state.get("active_group") or ""),
            origin_text=text,
            plan_scope=scope,
        )
    )
    existing = [
        dict(item)
        for item in state.get("action_queue") or []
        if isinstance(item, Mapping)
    ]
    state["action_queue"] = [admitted, *existing]
    _record_admitted_action(state, admitted, source=source)
    return _set_turn_phase(state, "execute", "deterministic_action_admitted")


def partition_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Persist Stage A's semantic partition and owner schedule."""

    draft = dict(state.get("semantic_plan_draft") or {})
    finalizing_draft = draft.get("status") == "ready_for_review"
    active_draft = draft.get("status") in {
        "awaiting_clarification",
        "ready_for_review",
    }
    text = (
        str(draft.get("original_input") or "")
        if finalizing_draft
        else str((state.get("turn_context") or {}).get("text") or "")
    )
    document = (
        None
        if active_draft
        else compile_bounded_semantic_value(state, text)
    )
    planning_lane = (
        "semantic_draft_finalization"
        if finalizing_draft
        else "bounded_semantic_value"
    )
    if document is None:
        document = hierarchical_planner.begin_semantic_partition(state, text)
        if not finalizing_draft:
            planning_lane = "hierarchical"
    state["semantic_planning"] = document
    _append_control_receipt(
        state,
        "semantic_partition",
        {
            "planning_lane": planning_lane,
            "status": str(document.get("status") or ""),
            "unit_count": int(document.get("unit_count") or 0),
            "owner_count": int(document.get("owner_count") or 0),
            "stage_a_calls": int(document.get("stage_a_calls") or 0),
            "errors_hash": _receipt_hash(
                document.get("errors")
                or document.get("owner_failures")
                or []
            ),
        },
    )
    next_phase = (
        "compile_owner"
        if document.get("status") == "compile_owner"
        else "review_plan"
    )
    return _set_turn_phase(state, next_phase, "semantic_partition_persisted")


def compile_owner_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Compile and checkpoint one owner document per graph transition."""

    before = dict(state.get("semantic_planning") or {})
    cursor = int(before.get("owner_cursor") or 0)
    requests = [
        dict(item)
        for item in before.get("owner_requests") or []
        if isinstance(item, Mapping)
    ]
    owner = (
        str(requests[cursor].get("owner") or "")
        if cursor < len(requests)
        else ""
    )
    document = hierarchical_planner.compile_next_owner(state, before)
    state["semantic_planning"] = document
    _append_control_receipt(
        state,
        "owner_compilation",
        {
            "owner": owner,
            "cursor_before": cursor,
            "cursor_after": int(document.get("owner_cursor") or 0),
            "status": str(document.get("status") or ""),
            "document_hash": _receipt_hash(
                (document.get("owner_documents") or {}).get(owner) or {}
            ),
            "errors_hash": _receipt_hash(document.get("errors") or []),
        },
    )
    next_phase = (
        "compile_owner"
        if document.get("status") == "compile_owner"
        else "review_plan"
    )
    return _set_turn_phase(state, next_phase, "owner_compilation_persisted")


def review_plan_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Run whole-plan admission only after all owner documents are persisted."""

    document = dict(state.get("semantic_planning") or {})
    try:
        queue = review_bounded_semantic_plan(state, document)
    except ValueError as exc:
        raise StateInvariantError(str(exc)) from exc
    if queue is None:
        queue = hierarchical_planner.review_semantic_plan(state, document)
    state["semantic_planning"] = {
        **document,
        "status": "reviewed",
        "result_hash": _receipt_hash(queue),
    }
    _append_control_receipt(
        state,
        "whole_plan_review",
        {
            "status": "reviewed",
            "result_hash": _receipt_hash(queue),
            "admission_calls": int(
                (queue.get("planner_metrics") or {}).get("admission_calls") or 0
            ),
        },
    )
    return _consume_planner_queue(state, queue)


def _consume_planner_queue(
    state: AgentGraphState,
    queue: Mapping[str, Any],
) -> AgentGraphState:
    """Project one reviewed semantic queue into graph admission state."""

    text = str((state.get("turn_context") or {}).get("text") or "")
    _append_control_receipt(
        state,
        "semantic_planner",
        {
            "input_hash": user_input_hash(text),
            "pending_contract_hash": _receipt_hash(
                state.get("pending_question") or {}
            ),
            "resolver_invoked": True,
            "result_reason_hash": hashlib.sha256(
                str(queue.get("reason") or "").encode("utf-8")
            ).hexdigest(),
            "planned_action_types": list(dict.fromkeys(
                str(item.get("type") or "")
                for item in queue.get("actions") or ()
                if isinstance(item, Mapping) and str(item.get("type") or "")
            )),
            "semantic_units": [
                {
                    "unit_id": str(item.get("unit_id") or ""),
                    "disposition": str(item.get("disposition") or ""),
                }
                for item in queue.get("semantic_units") or ()
                if isinstance(item, Mapping) and str(item.get("unit_id") or "")
            ],
            "planner_metrics": {
                str(key): value
                for key, value in dict(queue.get("planner_metrics") or {}).items()
                if isinstance(value, (int, float, bool))
            },
        },
    )
    semantic_draft = queue.get("semantic_draft")
    if isinstance(semantic_draft, Mapping):
        draft = validate_semantic_plan_draft(semantic_draft)
        question = _semantic_draft_question(draft)
        state["semantic_plan_draft"] = draft
        state.setdefault("turn_context", {})["semantic_units"] = [
            dict(row)
            for row in queue.get("semantic_units") or []
            if isinstance(row, Mapping)
        ]
        _record_semantic_plan_receipt(
            state,
            state["turn_context"]["semantic_units"],
            [],
        )
        if str((state.get("turn_receipt") or {}).get("turn_id") or ""):
            state["turn_receipt"] = reconcile_admission_coverage(
                state.get("turn_receipt") or {},
                [],
            )
        _install_pending_question(
            state,
            question,
            activate_group=False,
        )
        return _set_turn_phase(
            state,
            "compose",
            "semantic_draft_awaiting_clarification",
        )
    existing_draft = dict(state.get("semantic_plan_draft") or {})
    if (
        existing_draft.get("status") == "ready_for_review"
        and any(
            isinstance(item, Mapping)
            and str(item.get("type") or "") == "clarify_unresolved"
            for item in queue.get("actions") or ()
        )
    ):
        _invalidate_semantic_draft(
            state,
            existing_draft,
            reasons=("finalization_still_unresolved",),
        )
    if str(queue.get("reason") or "") == "resolver failed":
        if existing_draft.get("status") == "ready_for_review":
            _invalidate_semantic_draft(
                state,
                existing_draft,
                reasons=("finalization_provider_failed",),
            )
        record = model_provider_failure_record("resolver_failed")
        state["failure_recovery"] = {"status": "pending", "record": record}
        _replace_response_fragments(
            state,
            _fragment(
                "harness.failure.coordinator.model_provider_unavailable",
                kind="error",
                arguments={"error_type": "resolver_failed"},
                payload={"failure_id": str(record.get("failure_id") or "")},
            ),
        )
        return _set_turn_phase(state, "compose", "planner_failed")
    actions = _normalized_action_queue(dict(queue))
    state.setdefault("turn_context", {})["pending_choice_contracts"] = [
        dict(row)
        for row in queue.get("pending_choice_contracts") or []
        if isinstance(row, Mapping)
    ]
    state["turn_context"]["semantic_units"] = [
        dict(row)
        for row in queue.get("semantic_units") or []
        if isinstance(row, Mapping)
    ]
    for index, action in enumerate(actions):
        action["_source_unit_ids"] = [
            str(unit.get("unit_id") or "")
            for unit in state["turn_context"]["semantic_units"]
            if index in (
                unit.get("action_indexes")
                if isinstance(unit.get("action_indexes"), list)
                else []
            )
            and str(unit.get("unit_id") or "")
        ]
    _record_semantic_plan_receipt(
        state,
        state["turn_context"]["semantic_units"],
        state["turn_context"]["pending_choice_contracts"],
    )
    state["proposed_actions"] = actions
    return _set_turn_phase(state, "admit", "planner_completed")


def admit_turn_step(state: AgentGraphState) -> AgentGraphState:
    """Admit every reviewed action to the graph-owned execution queue."""

    text = str((state.get("turn_context") or {}).get("text") or "")
    admission = validate_action_plan(
        state,
        list(state.get("proposed_actions") or []),
    )
    if admission.status == "rejected":
        draft = dict(state.get("semantic_plan_draft") or {})
        rejection_codes = {item.code for item in admission.rejections}
        if (
            draft.get("status") == "awaiting_clarification"
            and "pending_answer_invalidated" in rejection_codes
            and (state.get("pending_question") or {}).get(
                "semantic_draft_binding"
            )
        ):
            _invalidate_semantic_draft(
                state,
                draft,
                reasons=("incompatible_sibling_mutation",),
            )
            state["pending_question"] = {}
            state["semantic_planning"] = {}
            state["proposed_actions"] = []
            return _set_turn_phase(
                state,
                "plan",
                "semantic_draft_invalidated_by_complete_turn",
            )
        if draft.get("status") == "ready_for_review":
            _invalidate_semantic_draft(
                state,
                draft,
                reasons=("final_admission_rejected",),
            )
        state["action_errors"] = [
            {
                "code": rejection.code,
                "message": rejection.message,
                "action_indexes": list(rejection.action_indexes),
            }
            for rejection in admission.rejections
        ]
        state["turn_receipt"] = reconcile_admission_coverage(
            state.get("turn_receipt") or {},
            [],
        )
        if state.get("pending_question"):
            return _set_turn_phase(
                state,
                "compose",
                "semantic_plan_rejected_pending_preserved",
            )
        return _set_turn_phase(state, "fallback", "semantic_plan_rejected")
    actions = [dict(item) for item in admission.actions]
    _validate_admission_transaction(state, actions, current_submission=True)
    scope = f"{state.get('thread_id') or 'default'}:{int(state.get('turn_index') or 0)}"
    actions = assign_action_ids(scope, text, actions)
    for index, action in enumerate(actions):
        action["_plan_scope"] = scope
        action["_plan_index"] = index
        action["_submitted_turn_index"] = int(state.get("turn_index") or 0)
        action["_origin_text"] = text
        action["_source_unit_ids"] = list(dict.fromkeys(
            str(unit_id)
            for unit_id in action.get("_source_unit_ids") or []
            if str(unit_id)
        ))
    pending_before_admission = dict(state.get("pending_question") or {})
    for action in actions:
        if str(action.get("type") or "") == "answer_pending":
            _record_pending_resolution(
                state,
                pending_before_admission,
                action,
                input_text=text,
                resolution_path=(
                    "exact_contract"
                    if action.get("selection_contract_verified") is True
                    else "typed_manual_value"
                ),
            )
    receipt = reconcile_admission_coverage(
        state.get("turn_receipt") or {},
        actions,
    )
    pending_verdicts = [
        dict(row)
        for row in receipt.get("pending_candidate_verdicts") or []
        if isinstance(row, Mapping)
    ]
    for verdict in pending_verdicts:
        index = verdict.get("action_index")
        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < len(actions)
        ):
            verdict["action_id"] = str(actions[index].get("action_id") or "")
    omission_checks = [
        dict(row)
        for row in receipt.get("sibling_omission_checks") or []
        if isinstance(row, Mapping)
    ]
    for check in omission_checks:
        unit_id = str(check.get("unit_id") or "")
        check["action_ids"] = [
            str(action.get("action_id") or "")
            for action in actions
            if unit_id
            and unit_id in {
                str(item)
                for item in action.get("_source_unit_ids") or []
            }
        ]
    receipt["pending_candidate_verdicts"] = pending_verdicts
    receipt["sibling_omission_checks"] = omission_checks
    state["turn_receipt"] = receipt
    origin_group = str(state.get("active_group") or "")
    turn_local_actions = [item for item in actions if action_is_turn_local(item)]
    durable_actions = [item for item in actions if not action_is_turn_local(item)]
    draft = dict(state.get("semantic_plan_draft") or {})
    if not _has_meaningful_queue(actions):
        if draft.get("status") == "ready_for_review":
            _invalidate_semantic_draft(
                state,
                draft,
                reasons=("finalization_produced_no_actions",),
            )
        pending = state.get("pending_question") or {}
        if pending:
            _replace_response_fragments(
                state,
                _fragment(
                    "harness.response.pending_answer_not_admitted",
                    kind="warning",
                    payload={
                        "pending_question_id": str(pending.get("id") or "")
                    },
                ),
            )
            return _set_turn_phase(state, "compose", "semantic_input_not_admitted")
        return _set_turn_phase(state, "fallback", "no_admitted_actions")
    pending = state.get("pending_question") or {}
    existing_queue = [
        _admission_action_from_envelope(item)
        for item in state.get("action_queue") or []
        if isinstance(item, dict)
    ]
    accepted_types = {
        str(item).strip()
        for item in pending.get("accepted_action_types") or []
        if str(item).strip()
    }
    answers_active_semantic_contract = bool(
        accepted_types
        and any(str(item.get("type") or "") in accepted_types for item in durable_actions)
    )
    if answers_active_semantic_contract and not pending.get("resume_action_queue"):
        # The question contract is the sole authority for retaining deferred
        # work. Without resume_action_queue, accepting the current semantic
        # answer supersedes every pre-existing queue entry regardless of
        # whether a legacy checkpoint happens to contain plan-like metadata.
        existing_queue = []
    finalizing_ready_draft = bool(
        draft.get("status") == "ready_for_review"
        and all(
            str(item.get("type") or "")
            != "reenter_secret_reference"
            for item in actions
        )
    )
    if finalizing_ready_draft:
        if any(
            (ACTION_BY_TYPE.get(str(item.get("type") or "")) is not None)
            and ACTION_BY_TYPE[str(item.get("type") or "")].effect == "execution"
            for item in actions
        ):
            _invalidate_semantic_draft(
                state,
                draft,
                reasons=("external_execution_requires_fresh_authorization",),
            )
            state["proposed_actions"] = []
            state["action_errors"] = [{
                "code": "semantic_draft_external_execution_rejected",
                "message": (
                    "external execution requires a fresh authorization turn"
                ),
                "action_indexes": [
                    index
                    for index, item in enumerate(actions)
                    if (
                        ACTION_BY_TYPE.get(str(item.get("type") or "")) is not None
                        and ACTION_BY_TYPE[
                            str(item.get("type") or "")
                        ].effect == "execution"
                    )
                ],
            }]
            return _set_turn_phase(
                state,
                "fallback",
                "semantic_draft_external_execution_rejected",
            )
        for action in actions:
            secret_bindings = _semantic_secret_bindings(draft, action)
            if secret_bindings:
                action["_semantic_secret_bindings"] = secret_bindings
        finalization_receipt = build_semantic_draft_finalization_receipt(
            draft,
            actions,
        )
        for action in actions:
            action["_semantic_draft_finalization_receipt"] = deepcopy(
                finalization_receipt
            )
        _validate_admission_transaction(
            state,
            actions,
            current_submission=True,
        )
    else:
        for action in actions:
            secret_bindings = _turn_secret_bindings(state, action)
            if secret_bindings:
                action["_semantic_secret_bindings"] = secret_bindings
    for action in actions:
        _record_admitted_action(state, action, source="semantic_plan")
    ordered_actions = _order_action_queue(state, [
        *turn_local_actions,
        *_merge_durable_action_queue(existing_queue, durable_actions),
    ])
    ordered_queue = _serialize_admitted_actions(
        state,
        ordered_actions,
        default_origin_text=text,
        default_origin_group=origin_group,
        default_plan_scope=scope,
    )
    register_state_secret_bindings(
        state,
        [
            binding
            for action in actions
            for binding in action.get("_semantic_secret_bindings") or ()
            if isinstance(binding, Mapping)
        ],
    )
    secret_candidate = {
        **state,
        "action_queue": ordered_queue,
    }
    try:
        reconcile_state_secret_bindings(secret_candidate)
    except ValueError as exc:
        raise StateInvariantError(str(exc)) from exc
    state["secret_bindings"] = list(
        secret_candidate.get("secret_bindings") or []
    )
    queue_deferred_by_pending_contract = bool(
        pending
        and ordered_queue
        and not any(
            _action_can_run_while_pending(state, dict(item))
            for item in ordered_actions
        )
    )
    if queue_deferred_by_pending_contract:
        _set_pending_queue_resume(state, enabled=True)
    _discard_unreferenced_action_secrets(
        state.get("action_queue") or [],
        ordered_queue,
        state,
    )
    state["action_queue"] = ordered_queue
    state["completed_actions"] = []
    state["action_errors"] = []
    if finalizing_ready_draft:
        state.setdefault("audit_events", []).append({
            "event": "semantic_draft_finalized",
            **deepcopy(finalization_receipt),
        })
        state.setdefault("turn_context", {})[
            "semantic_draft_finalization_receipt"
        ] = deepcopy(finalization_receipt)
        retained_secret_refs = {
            str(binding.get("reference") or "")
            for action in actions
            for binding in action.get("_semantic_secret_bindings") or ()
            if isinstance(binding, Mapping)
            and str(binding.get("reference") or "")
        }
        for binding in draft.get("source_secret_bindings") or ():
            if not isinstance(binding, Mapping):
                continue
            reference = str(binding.get("reference") or "")
            if reference and reference not in retained_secret_refs:
                discard_secret_reference(reference)
        for atom in draft.get("unresolved_atoms") or ():
            if not isinstance(atom, Mapping):
                continue
            reference = str(atom.get("resolution_ref") or "")
            if reference and reference not in retained_secret_refs:
                discard_secret_reference(reference)
        state["semantic_plan_draft"] = {}
    if pending.get("queue_barrier") is True and queue_deferred_by_pending_contract:
        return _set_turn_phase(state, "compose", "pending_contract_barrier")
    if state.get("action_queue"):
        return _set_turn_phase(state, "execute", "actions_admitted")
    if state.get("pending_question"):
        return _set_turn_phase(state, "compose", "pending_contract_preserved")
    return _set_turn_phase(state, "fallback", "no_durable_actions")


def select_action_step(state: AgentGraphState) -> AgentGraphState:
    """Select one durable action without applying domain state."""

    state = _copy_state(state)
    queue = [
        dict(item)
        for item in state.get("action_queue") or []
        if isinstance(item, Mapping)
    ]
    state["action_queue"] = queue
    state["selected_action"] = {}
    state["current_action"] = {}
    if not queue:
        if state.get("pending_question"):
            return _set_turn_phase(state, "compose", "queue_blocked_or_complete")
        return _set_turn_phase(state, "fallback", "queue_complete")
    pending = dict(state.get("pending_question") or {})
    if pending:
        eligible_index = next(
            (
                index
                for index, queued_action in enumerate(queue)
                if _action_can_run_while_pending(
                    state,
                    _queue_action(queued_action),
                )
            ),
            None,
        )
        if eligible_index is None:
            return _set_turn_phase(state, "compose", "pending_barrier_blocks_queue")
        if eligible_index:
            action = queue.pop(eligible_index)
            queue.insert(0, action)
            state["action_queue"] = queue
        queued_envelope = dict(queue[0])
    else:
        queued_envelope = dict(queue[0])
    envelope = action_envelope_from_dict(queued_envelope)
    action = _queue_action(queued_envelope)
    finalization_receipt = dict(
        envelope.admission_metadata.get(
            "semantic_draft_finalization_receipt"
        )
        or {}
    )
    try:
        _validate_admission_transaction(
            state,
            [action],
            current_submission=False,
        )
    except (StateInvariantError, TypeError, ValueError) as exc:
        state["action_queue"] = queue[1:]
        state.setdefault("action_errors", []).append({
            "action": action,
            "error": "durable_admission_metadata_invalid",
            "detail": str(exc),
        })
        return _set_turn_phase(
            state,
            "execute",
            "durable_admission_metadata_rejected",
        )
    target_group = resolve_action_target_group(action)
    prerequisite = ""
    if not finalization_receipt:
        prerequisite = next(
            (
                capability
                for capability in sorted(_action_requirements(state, action))
                if not _state_has_capability(state, capability)
                and capability in ALLOWED_GROUPS
            ),
            "",
        )
    if prerequisite:
        control = dict(state.get("control") or {})
        control["deferred_group"] = target_group
        state["control"] = control
        state = _activate_group_question(state, prerequisite)
        if state.get("pending_question"):
            _set_pending_queue_resume(state, enabled=True)
        return _set_turn_phase(
            state,
            "compose",
            "action_waiting_for_registered_prerequisite",
        )
    if lifecycle_rejected_action_indexes(state, [action]):
        state["action_queue"] = queue[1:]
        state.setdefault("action_errors", []).append({
            "action": action,
            "error": "lifecycle_inapplicable",
        })
        return _set_turn_phase(state, "execute", "lifecycle_action_rejected")
    action_id = str(action.get("action_id") or "")
    if action_id and action_id in set(state.get("applied_action_ids") or []):
        state["action_queue"] = queue[1:]
        return _set_turn_phase(state, "execute", "already_applied_action_skipped")
    spec = ACTION_BY_TYPE.get(envelope.action_type)
    if spec is None:
        state["action_queue"] = queue[1:]
        state.setdefault("action_errors", []).append({
            "action": action,
            "error": "unregistered_action",
        })
        return _set_turn_phase(state, "execute", "unregistered_action_rejected")
    _record_admitted_action(state, action, source="durable_queue")
    state["selected_action"] = action_envelope_to_dict(
        replace(envelope, status="selected")
    )
    state["current_action"] = action
    control = dict(state.get("control") or {})
    control["selected_owner"] = envelope.owner
    state["control"] = control
    validate_state(state)
    return _set_turn_phase(state, "route_owner", "action_selected")


def _build_action_envelope(
    state: AgentGraphState,
    action: Mapping[str, Any],
    *,
    semantic_order: int | None = None,
    submitted_turn_index: int | None = None,
    origin_group: str | None = None,
    origin_text: str | None = None,
    plan_scope: str | None = None,
) -> ActionEnvelope:
    """Separate validated action arguments from Harness-owned provenance."""

    action_type = str(action.get("type") or "")
    spec = ACTION_BY_TYPE.get(action_type)
    if spec is None:
        raise StateInvariantError(f"cannot envelope unregistered action: {action_type}")
    arguments = {
        str(key): deepcopy(value)
        for key, value in action.items()
        if key not in {"type", "action_id", "confidence", "reason"}
        and not str(key).startswith("_")
    }
    exact_origin_text = str(
        origin_text
        if origin_text is not None
        else action.get("_origin_text")
        or (state.get("turn_context") or {}).get("text")
        or ""
    )
    source_evidence = str(
        arguments.get("source_evidence")
        or exact_origin_text
        or ""
    )
    receipt = action.get("_semantic_admission_receipt")
    source_unit_ids = tuple(
        str(item)
        for item in (
            action.get("_source_unit_ids")
            or (
                (receipt or {}).get("source_unit_ids")
                if isinstance(receipt, Mapping)
                else ()
            )
        )
        or ()
    )
    effect_kind = (
        "external"
        if spec.effect == "execution"
        else "read_only"
        if spec.effect == "read_only"
        else "pure"
    )
    action_id = str(action.get("action_id") or "")
    pending = state.get("pending_question") or {}
    owner = spec.owner
    if action_type == "answer_pending":
        owner = str(pending.get("owner") or "")
        if not owner:
            raise StateInvariantError(
                "answer_pending requires an explicitly owned pending contract"
            )
    return ActionEnvelope(
        action_id=action_id,
        action_type=action_type,
        owner=owner,
        target_group=spec.target_group,
        arguments=arguments,
        confidence=str(action.get("confidence") or "medium"),  # type: ignore[arg-type]
        reason=str(action.get("reason") or ""),
        source_unit_ids=source_unit_ids,
        source_evidence_hash=hashlib.sha256(source_evidence.encode("utf-8")).hexdigest(),
        semantic_order=int(
            semantic_order
            if semantic_order is not None
            else action.get("_plan_index")
            or 0
        ),
        execution_order=len(state.get("completed_actions") or []),
        submitted_turn_index=int(
            submitted_turn_index
            if submitted_turn_index is not None
            else action.get("_submitted_turn_index")
            or state.get("turn_index")
            or 0
        ),
        origin_group=str(
            origin_group
            if origin_group is not None
            else action.get("_queue_origin_group")
            or (state.get("turn_context") or {}).get("origin_group")
            or ""
        ),
        origin_text=exact_origin_text,
        plan_scope=str(
            plan_scope
            if plan_scope is not None
            else action.get("_plan_scope")
            or ""
        ),
        admission_metadata={
            key.removeprefix("_"): deepcopy(action[key])
            for key in _ADMISSION_METADATA_KEYS
            if key in action
        },
        effect_kind=effect_kind,  # type: ignore[arg-type]
        idempotency_key=(
            f"{state.get('thread_id') or 'default'}:"
            f"{int(state.get('turn_index') or 0)}:{action_id}"
        ),
    )


def _action_from_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    envelope = action_envelope_from_dict(payload)
    return {
        "type": envelope.action_type,
        "action_id": envelope.action_id,
        "confidence": envelope.confidence,
        "reason": envelope.reason,
        **deepcopy(dict(envelope.arguments)),
    }


def _admission_action_from_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct a proposal only at registry and admission boundaries."""

    envelope = action_envelope_from_dict(payload)
    action = _action_from_envelope(payload)
    for key, value in envelope.admission_metadata.items():
        action[f"_{key}"] = deepcopy(value)
    action["_origin_text"] = envelope.origin_text
    action["_queue_origin_group"] = envelope.origin_group
    action["_submitted_turn_index"] = envelope.submitted_turn_index
    action["_plan_scope"] = envelope.plan_scope
    action["_plan_index"] = envelope.semantic_order
    return action


def _is_serialized_action_envelope(payload: Mapping[str, Any]) -> bool:
    return bool(
        str(payload.get("action_type") or "")
        and str(payload.get("owner") or "")
        and isinstance(payload.get("arguments"), Mapping)
    )


def _queue_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not _is_serialized_action_envelope(payload):
        raise StateInvariantError("durable action queue contains a raw proposal")
    return _admission_action_from_envelope(payload)


def _serialize_admitted_actions(
    state: AgentGraphState,
    actions: list[dict[str, Any]],
    *,
    default_origin_text: str,
    default_origin_group: str,
    default_plan_scope: str,
) -> list[dict[str, Any]]:
    """Create the only durable queue representation after admission."""

    serialized: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        envelope = _build_action_envelope(
            state,
            action,
            semantic_order=int(action.get("_plan_index") or index),
            submitted_turn_index=int(
                action.get("_submitted_turn_index")
                or state.get("turn_index")
                or 0
            ),
            origin_group=str(
                action.get("_queue_origin_group")
                or default_origin_group
            ),
            origin_text=str(
                redact_secret_references(
                    action.get("_origin_text")
                    or default_origin_text
                )
            ),
            plan_scope=str(
                action.get("_plan_scope")
                or default_plan_scope
            ),
        )
        serialized.append(action_envelope_to_dict(envelope))
    return serialized


def _secret_reference_paths(
    value: Any,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], str]]:
    if isinstance(value, str):
        return [
            (path, match.group(0))
            for match in re.finditer(
                r"semantic-secret:[A-Za-z0-9_-]+",
                value,
            )
        ]
    if isinstance(value, Mapping):
        return [
            row
            for key, item in value.items()
            for row in _secret_reference_paths(item, (*path, str(key)))
        ]
    if isinstance(value, list):
        return [
            row
            for index, item in enumerate(value)
            for row in _secret_reference_paths(item, (*path, index))
        ]
    return []


def _semantic_secret_bindings(
    draft: Mapping[str, Any],
    action: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bindings_by_reference = {
        str(atom.get("resolution_ref") or ""): dict(atom)
        for atom in draft.get("unresolved_atoms") or ()
        if isinstance(atom, Mapping)
        and str(atom.get("resolution_ref") or "")
    }
    bindings_by_reference.update({
        str(binding.get("reference") or ""): {
            "atom_id": str(binding.get("atom_id") or ""),
            "resolution_hash": str(binding.get("value_hash") or ""),
        }
        for binding in draft.get("source_secret_bindings") or ()
        if isinstance(binding, Mapping)
        and str(binding.get("reference") or "")
    })
    bindings: list[dict[str, Any]] = []
    business_action = {
        key: value
        for key, value in action.items()
        if not str(key).startswith("_")
        and key not in {"action_id", "confidence", "reason"}
    }
    for path, reference in _secret_reference_paths(business_action):
        atom = bindings_by_reference.get(reference)
        if atom is None:
            raise StateInvariantError(
                "semantic action contains an unbound secret reference"
            )
        bindings.append({
            "path": list(path),
            "reference": reference,
            "draft_id": str(draft.get("draft_id") or ""),
            "atom_id": str(atom.get("atom_id") or ""),
            "value_hash": str(atom.get("resolution_hash") or ""),
        })
    return bindings


def _turn_secret_bindings(
    state: Mapping[str, Any],
    action: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind ingress references used by one admitted action to exact paths."""

    available = {
        str(binding.get("reference") or ""): dict(binding)
        for binding in (
            state.get("turn_context") or {}
        ).get("input_secret_bindings") or ()
        if isinstance(binding, Mapping)
        and str(binding.get("reference") or "")
    }
    bindings: list[dict[str, Any]] = []
    business_action = {
        key: value
        for key, value in action.items()
        if not str(key).startswith("_")
        and key not in {"action_id", "confidence", "reason"}
    }
    for path, reference in _secret_reference_paths(business_action):
        source = available.get(reference)
        if source is None:
            raise StateInvariantError(
                "admitted action contains an unbound ingress secret reference"
            )
        bindings.append({
            "path": list(path),
            "reference": reference,
            "draft_id": str(source.get("draft_id") or ""),
            "atom_id": str(source.get("atom_id") or ""),
            "value_hash": str(source.get("value_hash") or ""),
        })
    return bindings


def _value_at_path(value: Any, path: list[str | int]) -> Any:
    current = value
    for key in path:
        if isinstance(key, int) and isinstance(current, list):
            current = current[key]
        elif isinstance(key, str) and isinstance(current, Mapping):
            current = current[key]
        else:
            raise StateInvariantError("semantic secret binding path is invalid")
    return current


def _replace_value_at_path(
    value: Any,
    path: list[str | int],
    replacement: str,
) -> None:
    if not path:
        raise StateInvariantError("semantic secret binding path cannot be empty")
    parent = value
    for key in path[:-1]:
        if isinstance(key, int) and isinstance(parent, list):
            parent = parent[key]
        elif isinstance(key, str) and isinstance(parent, dict):
            parent = parent[key]
        else:
            raise StateInvariantError("semantic secret binding path is invalid")
    key = path[-1]
    if isinstance(key, int) and isinstance(parent, list):
        parent[key] = replacement
    elif isinstance(key, str) and isinstance(parent, dict):
        parent[key] = replacement
    else:
        raise StateInvariantError("semantic secret binding path is invalid")


def _materialize_semantic_secret_bindings(
    action: dict[str, Any],
    envelope: ActionEnvelope,
) -> dict[str, Any]:
    """Resolve admitted secret refs only for the selected domain invocation."""

    bindings = envelope.admission_metadata.get("semantic_secret_bindings") or ()
    for raw in bindings:
        if not isinstance(raw, Mapping):
            raise StateInvariantError("semantic secret binding is invalid")
        path = [
            int(item) if isinstance(item, int) and not isinstance(item, bool)
            else str(item)
            for item in raw.get("path") or ()
        ]
        reference = str(raw.get("reference") or "")
        current = _value_at_path(action, path)
        if not isinstance(current, str) or reference not in current:
            raise StateInvariantError("semantic secret reference binding changed")
        value = resolve_secret_reference(
            reference,
            draft_id=str(raw.get("draft_id") or ""),
            atom_id=str(raw.get("atom_id") or ""),
            expected_hash=str(raw.get("value_hash") or ""),
        )
        if value is None:
            raise StateInvariantError("semantic secret reference is unavailable")
        if path != ["source_evidence"]:
            _replace_value_at_path(
                action,
                path,
                current.replace(reference, value),
            )
    return action


def _project_secret_values_to_references(
    value: Any,
    envelope: ActionEnvelope,
) -> Any:
    """Remove materialized secret values before a domain result is durable."""

    replacements: dict[str, str] = {}
    for raw in (
        envelope.admission_metadata.get("semantic_secret_bindings") or ()
    ):
        if not isinstance(raw, Mapping):
            raise StateInvariantError("semantic secret binding is invalid")
        reference = str(raw.get("reference") or "")
        secret = resolve_secret_reference(
            reference,
            draft_id=str(raw.get("draft_id") or ""),
            atom_id=str(raw.get("atom_id") or ""),
            expected_hash=str(raw.get("value_hash") or ""),
        )
        if secret is None:
            raise StateInvariantError("semantic secret reference is unavailable")
        replacements[secret] = reference

    def project(item: Any) -> Any:
        if isinstance(item, str):
            result = item
            for secret, reference in replacements.items():
                result = result.replace(secret, reference)
            return result
        if isinstance(item, Mapping):
            return {key: project(child) for key, child in item.items()}
        if isinstance(item, list):
            return [project(child) for child in item]
        if isinstance(item, tuple):
            return tuple(project(child) for child in item)
        return item

    return project(value)


def _project_handler_result_secrets(
    result: HandlerResult,
    envelope: ActionEnvelope,
) -> HandlerResult:
    payload = handler_result_to_dict(result)
    return handler_result_from_dict(
        _project_secret_values_to_references(payload, envelope)
    )


def _secret_references_in_envelopes(
    envelopes: Any,
) -> set[str]:
    references: set[str] = set()
    for payload in envelopes or ():
        if not isinstance(payload, Mapping) or not _is_serialized_action_envelope(
            payload
        ):
            continue
        envelope = action_envelope_from_dict(payload)
        for binding in (
            envelope.admission_metadata.get("semantic_secret_bindings") or ()
        ):
            if isinstance(binding, Mapping):
                reference = str(binding.get("reference") or "")
                if reference:
                    references.add(reference)
    return references


def _discard_unreferenced_action_secrets(
    removed: Any,
    retained: Any,
    state: Mapping[str, Any] | None = None,
) -> None:
    retained_refs = _secret_references_in_envelopes(retained)
    retained_refs.update(
        str(item.get("reference") or "")
        for item in (state or {}).get("secret_bindings") or ()
        if isinstance(item, Mapping)
    )
    for reference in _secret_references_in_envelopes(removed) - retained_refs:
        discard_secret_reference(reference)


def _domain_action_from_envelope(
    state: AgentGraphState,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the owner input from trusted envelope provenance."""

    envelope = action_envelope_from_dict(payload)
    action = _materialize_semantic_secret_bindings(
        _action_from_envelope(payload),
        envelope,
    )
    state_secret_identity_fields = (
        frozenset({
            "source_evidence",
            "scope_id",
            "owner_kind",
            "owner_id",
            "owner_revision",
            "reference",
            "atom_id",
            "value_hash",
        })
        if envelope.action_type == "reenter_secret_reference"
        else frozenset({"source_evidence"})
    )
    try:
        action = materialize_state_secret_references(
            action,
            state,
            skip_keys=state_secret_identity_fields,
        )
    except ValueError as exc:
        raise StateInvariantError(str(exc)) from exc
    origin_text = envelope.origin_text
    if envelope.action_type == "propose_config_values":
        action["source_text"] = origin_text
    if envelope.owner == "chain_rpc":
        action["origin_text"] = origin_text
    if envelope.owner == "coordinator":
        action["queue_origin_group"] = envelope.origin_group
    if envelope.action_type == "analyze_report" and not action.get("job_id"):
        requested_job = JOB_ID_RE.search(origin_text)
        if requested_job:
            action["job_id"] = requested_job.group(0)
    return action


def _queue_has_eligible_action(state: AgentGraphState) -> bool:
    queue = [
        dict(item)
        for item in state.get("action_queue") or []
        if isinstance(item, Mapping)
    ]
    if not queue:
        return False
    if not state.get("pending_question"):
        return True
    return any(
        _action_can_run_while_pending(state, _queue_action(action))
        for action in queue
    )


def execute_selected_owner_step(
    state: AgentGraphState,
    *,
    expected_owner: str,
) -> AgentGraphState:
    """Prepare one owner result or run an explicitly unmigrated lifecycle."""

    selected_envelope = dict(state.get("selected_action") or {})
    envelope = (
        action_envelope_from_dict(selected_envelope)
        if selected_envelope
        else None
    )
    selected = _action_from_envelope(selected_envelope) if selected_envelope else {}
    spec = ACTION_BY_TYPE.get(str(selected.get("type") or ""))
    if not selected or spec is None or envelope is None or envelope.owner != expected_owner:
        raise StateInvariantError(
            f"selected action owner mismatch: {expected_owner}/"
            f"{str(selected.get('type') or '<missing>')}"
        )
    if spec.effect == "execution":
        return prepare_execution_intent_step(
            state,
            expected_owner=expected_owner,
        )
    return prepare_selected_owner_result_step(
        state,
        expected_owner=expected_owner,
    )


def prepare_execution_intent_step(
    state: AgentGraphState,
    *,
    expected_owner: str,
) -> AgentGraphState:
    """Persist one external execution authorization before invocation."""

    candidate = _copy_state(state)
    envelope = action_envelope_from_dict(candidate.get("selected_action") or {})
    if envelope.owner != "execution" or expected_owner != "execution":
        raise StateInvariantError("only the execution owner may prepare a side effect")
    _execution_authorization_receipt(
        candidate,
        _action_from_envelope(candidate.get("selected_action") or {}),
    )
    request_id = str(
        (candidate.get("preflight") or {}).get("execution_request_id")
        or envelope.action_id
    )
    idempotency_key = f"harness:{request_id}"
    turn_id = (
        f"{candidate.get('thread_id') or 'default'}:"
        f"{int(candidate.get('turn_index') or 0)}"
    )
    request = {
        "action": action_envelope_to_dict(
            replace(
                envelope,
                idempotency_key=idempotency_key,
                status="prepared",
            )
        ),
        "workflow_mode": str(candidate.get("workflow_mode") or ""),
        "target_mode": str(candidate.get("target_mode") or ""),
        "plan_file": str(candidate.get("plan_file") or ""),
    }
    request_fingerprint = hashlib.sha256(
        json.dumps(
            request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    intent_id = hashlib.sha256(
        f"{turn_id}:{envelope.action_id}:{idempotency_key}".encode("utf-8")
    ).hexdigest()
    candidate.setdefault("preflight", {})["execution_request_id"] = request_id
    candidate["side_effect_intent"] = side_effect_intent_to_dict(
        SideEffectIntent(
            intent_id=intent_id,
            turn_id=turn_id,
            action_id=envelope.action_id,
            operation=envelope.action_type,
            execution_request_id=request_id,
            idempotency_key=idempotency_key,
            request=request,
            request_fingerprint=request_fingerprint,
            expected_receipt_kind="execution_handler_result",
        )
    )
    candidate["side_effect_receipt"] = {}
    validate_state(candidate)
    return _set_turn_phase(candidate, "invoke_effect", "execution_intent_committed")


def mark_side_effect_invoking_step(state: AgentGraphState) -> AgentGraphState:
    """Checkpoint the execution attempt before entering the application service."""

    candidate = _copy_state(state)
    intent = dict(candidate.get("side_effect_intent") or {})
    if not intent:
        raise StateInvariantError("side-effect invocation has no durable intent")
    if intent.get("status") not in {"prepared", "invoking"}:
        raise StateInvariantError(
            f"side-effect intent cannot be invoked from status {intent.get('status')}"
        )
    intent["status"] = "invoking"
    intent["attempt_count"] = int(intent.get("attempt_count") or 0) + 1
    candidate["side_effect_intent"] = intent
    validate_state(candidate)
    return _set_turn_phase(candidate, "perform_effect", "execution_attempt_checkpointed")


def invoke_idempotent_side_effect_step(state: AgentGraphState) -> AgentGraphState:
    """Invoke one external execution operation under its persisted identity."""

    candidate = _copy_state(state)
    intent = dict(candidate.get("side_effect_intent") or {})
    selected = dict(candidate.get("selected_action") or {})
    if not intent or not selected:
        raise StateInvariantError("external invocation is missing intent or selected action")
    envelope = action_envelope_from_dict(selected)
    if intent.get("action_id") != envelope.action_id:
        raise StateInvariantError("side-effect intent does not match the selected action")
    action = _domain_action_from_envelope(candidate, selected)
    _execution_authorization_receipt(candidate, action)
    result = DOMAIN_RUNTIME["execution"].apply_action(
        deepcopy(candidate),
        _action_proposal(action, envelope.confidence),
    )
    result = _project_handler_result_secrets(result, envelope)
    serialized_result = handler_result_to_dict(result)
    job_id = ""
    for write in result.delta.writes:
        if write.path == ("job", "job_id"):
            job_id = str(write.value or "")
            break
    status = "blocked" if result.blocker or result.completion == "blocked" else "succeeded"
    observed_receipt = side_effect_receipt_to_dict(
        SideEffectReceipt(
            receipt_id="",
            intent_id=str(intent.get("intent_id") or ""),
            action_id=envelope.action_id,
            status=status,  # type: ignore[arg-type]
            idempotency_key=str(intent.get("idempotency_key") or ""),
            result={"handler_result": serialized_result},
            job_id=job_id,
            failure_code=result.blocker.code if result.blocker else "",
            retryable=False,
        )
    )
    observed_receipt["receipt_id"] = execution_side_effect_receipt_id(
        observed_receipt
    )
    candidate["side_effect_receipt"] = observed_receipt
    validate_state(candidate)
    return _set_turn_phase(candidate, "commit_receipt", "execution_receipt_observed")


def commit_side_effect_receipt_step(state: AgentGraphState) -> AgentGraphState:
    """Admit a persisted external receipt to the common action commit path."""

    candidate = _copy_state(state)
    intent = dict(candidate.get("side_effect_intent") or {})
    receipt = dict(candidate.get("side_effect_receipt") or {})
    if not intent or not receipt:
        raise StateInvariantError("execution receipt commit requires intent and receipt")
    if receipt.get("intent_id") != intent.get("intent_id"):
        raise StateInvariantError("execution receipt does not match its intent")
    envelope = action_envelope_from_dict(candidate.get("selected_action") or {})
    handler_payload = dict((receipt.get("result") or {}).get("handler_result") or {})
    result = handler_result_from_dict(handler_payload)
    intent["status"] = str(receipt.get("status") or "failed")
    candidate["side_effect_intent"] = intent
    candidate = _set_turn_phase(candidate, "commit", "execution_receipt_committed")
    candidate["pending_domain_result"] = pending_domain_result_to_dict(
        PendingDomainResult(
            action=replace(envelope, status="prepared"),
            result=result,
            prepared_state_hash=_prepared_state_hash(candidate),
        )
    )
    validate_state(candidate)
    return candidate


def prepare_selected_owner_result_step(
    state: AgentGraphState,
    *,
    expected_owner: str,
) -> AgentGraphState:
    """Run one side-effect-free domain handler and persist its typed result."""

    candidate = _copy_state(state)
    envelope = action_envelope_from_dict(candidate.get("selected_action") or {})
    if envelope.owner != expected_owner:
        raise StateInvariantError(
            f"prepared action owner mismatch: {expected_owner}/{envelope.owner}"
        )
    if envelope.effect_kind == "external":
        raise StateInvariantError(
            "external action reached pure owner preparation without a side-effect intent"
        )
    action = _domain_action_from_envelope(
        candidate,
        candidate["selected_action"],
    )
    if envelope.action_type == "answer_pending":
        result = _prepare_pending_answer_result(
            candidate,
            action,
            envelope,
        )
        result = _project_handler_result_secrets(result, envelope)
        candidate = _set_turn_phase(candidate, "commit", "pending_result_prepared")
        candidate["pending_domain_result"] = pending_domain_result_to_dict(
            PendingDomainResult(
                action=replace(envelope, status="prepared"),
                result=result,
                prepared_state_hash=_prepared_state_hash(candidate),
            )
        )
        validate_state(candidate)
        return candidate
    runtime = (
        COORDINATOR_RUNTIME
        if expected_owner == "coordinator"
        else DOMAIN_RUNTIME.get(expected_owner)
    )
    if runtime is None:
        raise StateInvariantError(f"no runtime is registered for owner: {expected_owner}")
    handler_state = deepcopy(candidate)
    result = runtime.apply_action(
        handler_state,
        _action_proposal(action, envelope.confidence),
    )
    result = _project_handler_result_secrets(result, envelope)
    spec = ACTION_BY_TYPE.get(envelope.action_type)
    if spec is not None and spec.lifetime == "turn_local":
        allowed_roots = set(spec.turn_local_result_roots)
        overlay_question = (
            result.pending_question
            if not candidate.get("pending_question") and result.pending_question is not None
            else None
        )
        turn_local_delta = StateDelta(
            writes=tuple(
                write
                for write in result.delta.writes
                if write.path and write.path[0] in allowed_roots
            ),
            deletes=tuple(
                path
                for path in result.delta.deletes
                if path and path[0] in allowed_roots
            ),
        )
        result = replace(
            result,
            delta=turn_local_delta,
            next_group=(
                str(
                    (
                        asdict(overlay_question)
                        if is_dataclass(overlay_question)
                        else dict(overlay_question or {})
                    ).get("group")
                    or ""
                )
                if overlay_question is not None
                else ""
            ),
            pending_question=overlay_question,
            clear_pending=False,
            navigation_command=None,
            checkpoint_command=None,
            invalidated_groups=(),
            reconfigured_groups=(),
            invalidated_fields=(),
            followup_actions=(),
        )
    candidate = _set_turn_phase(candidate, "commit", "domain_result_prepared")
    candidate["pending_domain_result"] = pending_domain_result_to_dict(
        PendingDomainResult(
            action=replace(envelope, status="prepared"),
            result=result,
            prepared_state_hash=_prepared_state_hash(candidate),
        )
    )
    validate_state(candidate)
    return candidate


def _prepare_pending_answer_result(
    state: AgentGraphState,
    action: dict[str, Any],
    envelope: ActionEnvelope,
) -> HandlerResult:
    """Translate one admitted answer into a domain result or one follow-up."""

    pending = dict(state.get("pending_question") or {})
    if not pending:
        return HandlerResult(
            blocker=_failure(
                "harness.failure.coordinator.no_active_question",
                arguments={"operation": "answer_pending"},
            )
        )
    raw_answer = action.get("answer", "")
    selected = action.get("selected_value")
    if isinstance(selected, str) and not selected.strip():
        selected = None
    choice_question = str(pending.get("kind") or "") in {
        "numbered_choice",
        "yes_no",
    }
    interpreted: Any = selected if selected is not None else raw_answer
    if not choice_question and (
        interpreted is None
        or (isinstance(interpreted, str) and not interpreted.strip())
    ):
        return HandlerResult(
            blocker=_failure(
                "harness.failure.coordinator.pending_answer_empty",
                arguments={
                    "question_id": str(pending.get("id") or "<missing>")
                },
            )
        )
    if not choice_question and not (
        _pending_option_value_exists(selected, pending)
        or _value_satisfies_pending_contract(interpreted, pending)
    ):
        return HandlerResult(
            blocker=_failure(
                "harness.failure.coordinator.pending_value_invalid",
                arguments={
                    "question_id": str(pending.get("id") or "<missing>")
                },
            )
        )
    manual_choice_value = bool(
        choice_question
        and pending.get("manual_input_allowed") is True
        and not _pending_option_value_exists(selected, pending)
        and _value_satisfies_pending_contract(
            selected if selected is not None else str(raw_answer),
            pending,
        )
    )
    if (
        choice_question
        and not _pending_option_value_exists(selected, pending)
        and not manual_choice_value
    ):
        return HandlerResult(
            blocker=_failure(
                "harness.failure.coordinator.pending_option_unmapped",
                arguments={
                    "question_id": str(pending.get("id") or "<missing>")
                },
            )
        )
    value = selected if _pending_option_value_exists(selected, pending) else interpreted
    declared_action = action_for_value(pending, value)
    if declared_action and str(declared_action.get("type") or "") != "answer_pending":
        followup = dict(declared_action)
        spec = ACTION_BY_TYPE.get(str(followup.get("type") or ""))
        if (
            spec is not None
            and "source_evidence" in spec.allowed_arguments
            and not str(followup.get("source_evidence") or "").strip()
        ):
            followup["source_evidence"] = str(raw_answer or "").strip()
        followup["confidence"] = "high"
        followup["selection_contract_verified"] = True
        policy = _option_return_policy(pending, value)
        return HandlerResult(
            consumed_action_ids=(envelope.action_id,),
            clear_pending=policy != "stay",
            pending_question=pending if policy == "stay" else None,
            followup_actions=(followup,),
            completion="completed",
        )
    declared_manual = (
        dict(pending.get("manual_action") or {})
        if isinstance(pending.get("manual_action"), Mapping)
        else {}
    )
    if (
        declared_manual
        and str(declared_manual.get("type") or "") != "answer_pending"
        and (
            isinstance(pending.get("semantic_draft_binding"), Mapping)
            or isinstance(pending.get("secret_reentry_binding"), Mapping)
        )
    ):
        value_argument = str(declared_manual.pop("value_argument", "") or "")
        followup = {
            key: item
            for key, item in declared_manual.items()
            if key != "type"
        }
        followup["type"] = str(declared_manual.get("type") or "")
        followup[value_argument] = value
        spec = ACTION_BY_TYPE.get(followup["type"])
        if (
            spec is not None
            and "source_evidence" in spec.allowed_arguments
            and not str(followup.get("source_evidence") or "").strip()
        ):
            followup["source_evidence"] = str(raw_answer or value or "").strip()
        followup["confidence"] = "high"
        followup["selection_contract_verified"] = True
        return HandlerResult(
            consumed_action_ids=(envelope.action_id,),
            clear_pending=not bool(spec is not None and spec.preserve_pending),
            followup_actions=(followup,),
            completion="completed",
        )
    question_id = str(pending.get("id") or "")
    group = str(pending.get("group") or "")
    pending_owner = str(
        pending.get("owner")
        or GROUP_OWNER.get(group, "")
    )
    runtime = DOMAIN_RUNTIME.get(pending_owner)
    if runtime is None or runtime.apply_answer is None:
        return HandlerResult(
            blocker=_failure(
                "harness.failure.coordinator.answer_handler_missing",
                arguments={
                    "owner": pending_owner or "<missing>",
                    "group": group or "<missing>",
                    "question_id": question_id or "<missing>",
                },
            )
        )
    return replace(
        runtime.apply_answer(
            deepcopy(state),
            pending,
            value,
            str(
                (state.get("turn_context") or {}).get("text")
                or state.get("last_user_input")
                or raw_answer
                or ""
            ),
        ),
        consumed_action_ids=(envelope.action_id,),
    )


def commit_selected_action_step(state: AgentGraphState) -> AgentGraphState:
    """Commit exactly one prepared owner result and disposition its action."""

    candidate = _copy_state(state)
    prepared_payload = dict(candidate.get("pending_domain_result") or {})
    if not prepared_payload:
        raise StateInvariantError("commit node has no pending domain result")
    prepared = pending_domain_result_from_dict(prepared_payload)
    envelope = prepared.action
    if prepared.prepared_state_hash != _prepared_state_hash(candidate):
        raise StateInvariantError("prepared domain result state fingerprint changed before commit")
    queue = [
        dict(item)
        for item in candidate.get("action_queue") or []
        if isinstance(item, Mapping)
    ]
    if not queue or str(queue[0].get("action_id") or "") != envelope.action_id:
        raise StateInvariantError("prepared action is not the admitted queue head")
    action = _action_from_envelope(action_envelope_to_dict(envelope))
    before_pending = dict(candidate.get("pending_question") or {})
    before_pending_id = str(before_pending.get("id") or "")
    spec = ACTION_BY_TYPE.get(envelope.action_type)
    candidate["action_queue"] = (
        queue if prepared.result.blocker else queue[1:]
    )
    if envelope.action_type == "answer_pending" and not prepared.result.blocker:
        _discard_superseded_queue_actions(candidate, before_pending)
    candidate["pending_domain_result"] = {}
    candidate["selected_action"] = {}
    candidate["current_action"] = action
    control = dict(candidate.get("control") or {})
    control.pop("selected_owner", None)
    candidate["control"] = control
    if (
        envelope.action_type != "answer_pending"
        and not (spec is not None and spec.lifetime == "turn_local")
        and envelope.action_type
        in {
            str(item).strip()
            for item in before_pending.get("accepted_action_types") or []
            if str(item).strip()
        }
    ):
        candidate["pending_question"] = {}
    finalization_receipt = dict(
        envelope.admission_metadata.get(
            "semantic_draft_finalization_receipt"
        )
        or {}
    )
    if finalization_receipt and prepared.result.blocker:
        raise StateInvariantError(
            "semantic draft finalization transaction blocked before publish: "
            f"{prepared.result.blocker.code}"
        )
    if finalization_receipt and not prepared.result.blocker:
        candidate.setdefault("audit_events", []).append({
            "event": "semantic_draft_finalization_action_applied",
            "draft_id": str(finalization_receipt.get("draft_id") or ""),
            "receipt_hash": str(finalization_receipt.get("receipt_hash") or ""),
            "admission_transaction_hash": str(
                finalization_receipt.get("admission_transaction_hash") or ""
            ),
            "action_id": envelope.action_id,
        })
    committed = _apply_handler_result(
        candidate,
        prepared.result,
        owner=envelope.owner,
    )
    if prepared.result.suppress_pending_render:
        turn_context = dict(committed.get("turn_context") or {})
        turn_context["suppress_pending_render"] = True
        committed["turn_context"] = turn_context
    register_state_secret_bindings(
        committed,
        envelope.admission_metadata.get("semantic_secret_bindings") or (),
    )
    try:
        reconcile_state_secret_bindings(committed)
    except ValueError as exc:
        raise StateInvariantError(str(exc)) from exc
    remaining_finalized_actions = []
    if finalization_receipt:
        remaining_finalized_actions = [
            item
            for item in committed.get("action_queue") or ()
            if isinstance(item, Mapping)
            and str(
                (
                    (item.get("admission_metadata") or {}).get(
                        "semantic_draft_finalization_receipt"
                    )
                    or {}
                ).get("receipt_hash")
                or ""
            )
            == str(finalization_receipt.get("receipt_hash") or "")
        ]
    if finalization_receipt and remaining_finalized_actions:
        # A finalized pure plan is one Product Head transaction. Questions
        # produced by intermediate domain handlers are deferred until every
        # action succeeds, so a later blocker can roll back the whole attempt.
        committed["pending_question"] = {}
        control = dict(committed.get("control") or {})
        control.pop("deferred_group", None)
        committed["control"] = control
    if (
        not remaining_finalized_actions
        and prepared.result.completion == "in_progress"
        and prepared.result.next_group
        and not committed.get("pending_question")
    ):
        active_group = str(prepared.result.next_group)
        internal_group_progression = bool(
            active_group
            and active_group
            in {
                str(envelope.target_group or ""),
                str(before_pending.get("group") or ""),
                str(candidate.get("active_group") or ""),
            }
        )
        prerequisite = (
            _navigation_prerequisite(committed, active_group)
            if active_group and not internal_group_progression
            else ""
        )
        if prerequisite:
            control = dict(committed.get("control") or {})
            control["deferred_group"] = active_group
            committed["control"] = control
            _record_group_transition(committed, prerequisite)
            active_group = prerequisite
        next_question = _question_for_group(committed, active_group) if active_group else None
        if next_question:
            next_question["same_turn_navigation_allowed"] = (
                prepared.result.completion != "blocked"
            )
            _install_pending_question(committed, next_question)
    if committed.get("pending_question") and committed.get("action_queue"):
        _set_pending_queue_resume(committed, enabled=True)
    defer_after_answer = False
    if envelope.action_type == "answer_pending":
        committed = _resume_field_reconfiguration_after_prerequisite(
            committed,
            answered_question=before_pending,
        )
        if (
            before_pending_id
            and str((committed.get("pending_question") or {}).get("id") or "")
            == before_pending_id
            and prepared.result.completion == "blocked"
        ):
            committed["pending_question"] = with_pending_question_created_turn(
                committed["pending_question"],
                created_turn_index=int(committed.get("turn_index") or 0),
            )
        if (
            prepared.result.completion == "blocked"
            and committed.get("pending_question")
            and committed.get("action_queue")
        ):
            _set_pending_queue_resume(committed, enabled=True)
            defer_after_answer = True
        elif committed.get("pending_question") and committed.get("action_queue"):
            _set_pending_queue_resume(committed, enabled=True)
            if (
                not _queue_has_eligible_action(committed)
            ):
                defer_after_answer = True
    if envelope.action_type == "reset_session" and queue[1:]:
        committed["action_queue"] = queue[1:]
    after_pending_id = str((committed.get("pending_question") or {}).get("id") or "")
    if (
        spec is not None
        and spec.preserve_pending
        and before_pending_id
        and after_pending_id != before_pending_id
        and not _action_satisfies_pending_manual_effect(before_pending, action)
    ):
        _push_interruption_frame(
            committed,
            before_pending,
            reason=f"{envelope.action_type}_overlay",
        )
    rejected = bool(prepared.result.blocker)
    if not rejected:
        if spec is None or spec.lifetime != "turn_local":
            completed = list(committed.get("completed_actions") or [])
            completed.append(redact_secret_references(action))
            committed["completed_actions"] = completed[-20:]
        if envelope.action_id:
            applied = list(committed.get("applied_action_ids") or [])
            if envelope.action_id not in applied:
                applied.append(envelope.action_id)
            committed["applied_action_ids"] = applied[-200:]
        receipt = dict(committed.get("turn_receipt") or {})
        execution_order = list(receipt.get("execution_order") or [])
        if envelope.action_id and envelope.action_id not in execution_order:
            execution_order.append(envelope.action_id)
        receipt["execution_order"] = execution_order
        receipt["status"] = "executing"
        committed["turn_receipt"] = receipt
        _record_execution_approval(committed, action)
        if spec is not None and spec.lifetime == "turn_local":
            result_count = len(prepared.result.response_fragments)
            if result_count:
                turn_context = dict(committed.get("turn_context") or {})
                turn_context["turn_local_result_count"] = int(
                    turn_context.get("turn_local_result_count") or 0
                ) + result_count
                committed["turn_context"] = turn_context
    committed["current_action"] = {}
    try:
        reconcile_state_secret_bindings(committed)
    except ValueError as exc:
        raise StateInvariantError(str(exc)) from exc
    if not rejected:
        _discard_unreferenced_action_secrets(
            queue,
            committed.get("action_queue") or [],
            committed,
        )
    draft_finalization = _prepare_ready_semantic_draft_finalization(committed)
    if draft_finalization:
        validate_state(committed)
        return _set_turn_phase(
            committed,
            str(draft_finalization["phase"]),
            str(draft_finalization["reason"]),
        )
    if (
        spec is not None
        and spec.lifetime == "turn_local"
        and not committed.get("action_queue")
    ):
        validate_state(committed)
        return _set_turn_phase(
            committed,
            "compose",
            "turn_local_result_completed",
        )
    if defer_after_answer:
        validate_state(committed)
        return _set_turn_phase(
            committed,
            "compose",
            "new_pending_contract_defers_remaining_queue",
        )
    queue_can_continue = _queue_has_eligible_action(committed)
    if rejected or (
        prepared.result.stop_after_response
        and not committed.get("action_queue")
    ):
        return _set_turn_phase(committed, "compose", "action_commit_stopped")
    if queue_can_continue:
        validate_state(committed)
        return _set_turn_phase(committed, "execute", "action_committed_queue_continues")
    if committed.get("pending_question"):
        validate_state(committed)
        return _set_turn_phase(committed, "compose", "action_committed_pending")
    validate_state(committed)
    return _set_turn_phase(committed, "fallback", "action_committed_queue_complete")


def _prepare_ready_semantic_draft_finalization(
    state: AgentGraphState,
) -> dict[str, str] | None:
    """Prepare a ready draft for full recompilation without applying candidates."""

    ready_draft = dict(state.get("semantic_plan_draft") or {})
    if ready_draft.get("status") != "ready_for_review":
        return None
    stale_reasons = semantic_draft_staleness_reasons(ready_draft, state)
    if stale_reasons:
        _invalidate_semantic_draft(
            state,
            ready_draft,
            reasons=stale_reasons,
        )
        state["pending_question"] = {}
        return {
            "phase": "fallback",
            "reason": "semantic_draft_precondition_changed",
        }
    missing_secret_bindings = missing_semantic_draft_secret_bindings(
        ready_draft
    )
    if missing_secret_bindings:
        missing = missing_secret_bindings[0]
        question = _secret_reentry_question(
            str(ready_draft.get("source_active_group") or "opening"),
            {
                "scope_id": str(missing.get("draft_id") or ""),
                "owner_kind": "semantic_draft",
                "owner_id": str(ready_draft.get("draft_id") or ""),
                "owner_revision": int(ready_draft.get("revision") or 0),
                "reference": str(missing.get("reference") or ""),
                "atom_id": str(missing.get("atom_id") or ""),
                "value_hash": str(missing.get("value_hash") or ""),
            },
        )
        _install_pending_question(state, question)
        turn_context = dict(state.get("turn_context") or {})
        turn_context.pop("semantic_draft_finalization", None)
        state["turn_context"] = turn_context
        return {
            "phase": "compose",
            "reason": "semantic_draft_secret_reentry_required",
        }
    state["pending_question"] = dict(
        ready_draft.get("source_pending_question") or {}
    )
    state["active_group"] = str(
        ready_draft.get("source_active_group") or "opening"
    )
    turn_context = dict(state.get("turn_context") or {})
    turn_context.update({
        "text": str(ready_draft.get("original_input") or ""),
        "semantic_draft_finalization": {
            "draft_id": str(ready_draft.get("draft_id") or ""),
            "revision": int(ready_draft.get("revision") or 0),
        },
    })
    clarification_receipt = dict(state.get("turn_receipt") or {})
    if clarification_receipt:
        turn_context["semantic_draft_clarification_receipt"] = (
            clarification_receipt
        )
    replay_receipt = turn_receipt_to_dict(TurnReceipt(
        turn_id=(
            f"semantic-draft:{ready_draft.get('draft_id')}:"
            f"{int(ready_draft.get('revision') or 0)}"
        ),
        input_hash=str(ready_draft.get("original_input_hash") or ""),
        language=str(state.get("language") or "en"),
        input_shape="semantic_draft_finalization",
        clauses=tuple(
            dict(item)
            for item in ready_draft.get("source_clauses") or ()
            if isinstance(item, Mapping)
        ),
        pending_before=dict(
            ready_draft.get("source_pending_question") or {}
        ),
    ))
    turn_context["semantic_draft_replay_receipt_id"] = str(
        replay_receipt.get("turn_id") or ""
    )
    state["turn_context"] = turn_context
    state["turn_receipt"] = replay_receipt
    state["semantic_planning"] = {}
    state["proposed_actions"] = []
    return {
        "phase": "plan",
        "reason": "semantic_draft_ready_for_finalization",
    }


def _prepared_state_hash(state: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in state.items()
        if key not in {
            "discovery",
            "framework_summary",
            "web_research",
            "pending_domain_result",
        }
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fallback_turn_step(state: AgentGraphState) -> AgentGraphState:
    selected_group, reason = next_group_and_reason(state)
    readiness = {}
    for group in GROUP_ORDER:
        fact = group_readiness(state, group)
        readiness[group] = {
            "ready": bool(fact.ready),
            "continuation": bool(fact.continuation),
            "reason_hash": hashlib.sha256(
                str(fact.reason or "").encode("utf-8")
            ).hexdigest(),
        }
    state = _ask_next_blocking_question(state)
    _append_control_receipt(
        state,
        "fallback_selection",
        {
            "selected_group": selected_group,
            "reason_hash": hashlib.sha256(
                str(reason or "").encode("utf-8")
            ).hexdigest(),
            "group_readiness": readiness,
            "pending_after_id": str(
                (state.get("pending_question") or {}).get("id") or ""
            ),
        },
    )
    return _set_turn_phase(state, "compose", "fallback_resolved")


def compose_turn_step(state: AgentGraphState) -> AgentGraphState:
    state = _finalize_turn_response(state)
    receipt = dict(state.get("turn_receipt") or {})
    if receipt:
        receipt["pending_after"] = deepcopy(state.get("pending_question") or {})
        receipt["response_count"] = len(
            (state.get("turn_context") or {}).get("response_manifest") or ()
        )
        receipt["status"] = (
            "blocked"
            if state.get("pending_question") or state.get("failure_recovery")
            else "completed"
        )
        state["turn_receipt"] = receipt
    pending = dict(state.get("pending_question") or {})
    _append_control_receipt(
        state,
        "response_composition",
        {
            "language": str(state.get("language") or ""),
            "active_group": str(state.get("active_group") or ""),
            "source_action_ids": [
                str(item)
                for item in (state.get("turn_receipt") or {}).get(
                    "execution_order"
                )
                or ()
                if str(item)
            ],
            "pending_contract_hash": _receipt_hash(pending),
            "terminal_response_hash": str(
                (state.get("turn_context") or {}).get(
                    "terminal_response_hash"
                )
                or ""
            ),
            "terminal_semantic_hash": str(
                (state.get("turn_context") or {}).get(
                    "terminal_semantic_hash"
                )
                or ""
            ),
            "fragments": [
                dict(fragment)
                for fragment in (state.get("turn_context") or {}).get(
                    "response_manifest"
                )
                or ()
                if isinstance(fragment, Mapping)
            ],
        },
    )
    turn_context = dict(state.get("turn_context") or {})
    turn_context.pop("input_secret_bindings", None)
    state["turn_context"] = turn_context
    return _set_turn_phase(state, "end", "response_composed")


def _copy_state(state: AgentGraphState) -> AgentGraphState:
    output: AgentGraphState = dict(state)
    output.setdefault("action_queue", [])
    output.setdefault("current_action", {})
    output.setdefault("completed_actions", [])
    output.setdefault("action_errors", [])
    output.setdefault("group_states", {})
    output.setdefault("group_history", [])
    output.setdefault("confirmed_config", {})
    output.setdefault("inferred_config", {})
    output.setdefault("invalidated_groups", [])
    output.setdefault("interruption_stack", [])
    output.setdefault("chain_identity", {})
    output.setdefault("workload", {})
    output.setdefault("custom_rpc", {})
    output.setdefault("endpoint_evidence", {})
    output.setdefault("fixture_evidence", {})
    output.setdefault("sync_observe", {})
    output.setdefault("observability", {})
    output.setdefault("evidence_buffer", [])
    output.setdefault("evidence_collection", {})
    output.setdefault("failure_recovery", {})
    output.setdefault("final_benchmark", {})
    output.setdefault("audit_events", [])
    output.setdefault("resume_context", {})
    output.setdefault("workflow_goals", [])
    return output



def _merge_durable_action_queue(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge a new user turn with deferred actions from earlier turns.

    Incoming actions run first because they may answer the current barrier or
    revise an earlier request. A new action supersedes an older action with the
    same semantic identity or mutation dimension. Reset intentionally discards
    deferred work from the prior workflow.
    """

    existing = [
        (
            _admission_action_from_envelope(item)
            if _is_serialized_action_envelope(item)
            else dict(item)
        )
        for item in _durable_actions(existing)
    ]
    incoming = [
        (
            _admission_action_from_envelope(item)
            if _is_serialized_action_envelope(item)
            else dict(item)
        )
        for item in _durable_actions(incoming)
    ]
    if any(str(item.get("type") or "") == "reset_session" for item in incoming):
        return incoming
    merged_incoming = [dict(item) for item in incoming]
    incoming_index = {
        action_merge_key(item): index
        for index, item in enumerate(merged_incoming)
    }
    incoming_dimensions = {
        spec.mutation_dimension
        for item in incoming
        if (spec := ACTION_BY_TYPE.get(str(item.get("type") or ""))) is not None
        and spec.mutation_dimension
    }
    retained: list[dict[str, Any]] = []
    for item in existing:
        spec = ACTION_BY_TYPE.get(str(item.get("type") or ""))
        merge_key = action_merge_key(item)
        if merge_key in incoming_index:
            index = incoming_index[merge_key]
            merged_incoming[index] = merge_semantic_actions(item, merged_incoming[index])
            continue
        if spec is not None and spec.mutation_dimension in incoming_dimensions:
            continue
        retained.append(item)
    return merged_incoming + retained


def _durable_actions(actions: Any) -> list[dict[str, Any]]:
    """Drop turn-local actions from current and legacy durable queues."""

    return [
        dict(item)
        for item in actions or []
        if (
            isinstance(item, dict)
            and not action_is_turn_local(
                _action_from_envelope(item)
                if _is_serialized_action_envelope(item)
                else item
            )
        )
    ]


def _discard_superseded_queue_actions(state: AgentGraphState, question: PendingQuestion) -> None:
    """Drop queued actions whose decision was resolved by this question.

    Interruption questions declare this relationship in their typed contract;
    the coordinator only enforces it and does not infer domain intent.
    """

    superseded = {
        str(action_type).strip()
        for action_type in question.get("supersedes_action_types") or []
        if str(action_type).strip()
    }
    if not superseded:
        return
    state["action_queue"] = [
        action
        for action in state.get("action_queue") or []
        if str(_queue_action(action).get("type") or "").strip() not in superseded
    ]


def _next_queue_action_crosses_pending_barrier(state: AgentGraphState) -> bool:
    queue = state.get("action_queue") or []
    if not queue:
        return False
    envelope = action_envelope_from_dict(queue[0])
    action = _action_from_envelope(queue[0])
    if not action_crosses_pending_barrier(action):
        return False
    pending = state.get("pending_question") or {}
    if not pending:
        return True
    # Navigation may detour from a question that was already visible when the
    # user submitted this turn. A separately admitted explicit navigation in
    # the same semantic transaction may also suspend a question created by an
    # earlier sibling action; the interruption stack, not the barrier, owns
    # resumption. Other actions cannot bypass a newly created contract.
    submitted_turn = envelope.submitted_turn_index
    created_turn = int(pending.get("created_turn_index") or 0)
    if submitted_turn and created_turn < submitted_turn:
        return True
    return bool(
        submitted_turn
        and created_turn == submitted_turn
        and str(action.get("type") or "") == "change_group"
        and action.get("group_navigation_semantic_verified") is True
        and str(action.get("source_evidence") or "").strip()
    )


def _push_interruption_frame(state: AgentGraphState, pending: PendingQuestion, *, reason: str) -> None:
    """Persist one typed return point without copying stale prompt text."""

    frame = {
        "group": str(pending.get("group") or state.get("active_group") or ""),
        "owner": str(
            pending.get("owner")
            or GROUP_OWNER.get(str(pending.get("group") or ""), "")
        ),
        "question_id": str(pending.get("id") or ""),
        "reason": reason,
    }
    field = str(pending.get("field") or "").strip()
    if field:
        frame["field"] = field
    reconfiguration_target = str(
        pending.get("reconfiguration_target_field") or ""
    ).strip()
    if reconfiguration_target:
        frame["reconfiguration_target_field"] = reconfiguration_target
        frame["reconfiguration_baseline_revision"] = field_confirmation_revision(
            state,
            reconfiguration_target,
            group=str(frame.get("group") or ""),
        )
    if not frame["group"] or not frame["question_id"]:
        return
    stack = list(state.get("interruption_stack") or [])
    frame_identity = (
        frame["group"],
        frame["owner"],
        frame["question_id"],
        str(frame.get("field") or ""),
        str(frame.get("reconfiguration_target_field") or ""),
        int(frame.get("reconfiguration_baseline_revision") or 0),
    )
    top_identity = (
        str((stack[-1] if stack else {}).get("group") or ""),
        str((stack[-1] if stack else {}).get("owner") or ""),
        str((stack[-1] if stack else {}).get("question_id") or ""),
        str((stack[-1] if stack else {}).get("field") or ""),
        str(
            (stack[-1] if stack else {}).get("reconfiguration_target_field")
            or ""
        ),
        int(
            (stack[-1] if stack else {}).get(
                "reconfiguration_baseline_revision"
            )
            or 0
        ),
    )
    if top_identity != frame_identity:
        stack.append(frame)
    state["interruption_stack"] = stack[-40:]


def _resume_suspended_question(state: AgentGraphState) -> PendingQuestion | None:
    """Restore the newest still-relevant typed question after a detour.

    Frames store identity only.  The owning domain reconstructs the question
    from current state, so stale prompts cannot revive after invalidation.
    """

    # A detour owns control until its current domain has no blocking question.
    # Restoring an older frame first would abandon a partially configured group
    # and make the next user answer bind to stale work.
    stack = list(state.get("interruption_stack") or [])
    active_group = str(state.get("active_group") or "").strip()
    if active_group and active_group in ALLOWED_GROUPS:
        active_question = _question_for_group(state, active_group)
        if active_question:
            # Questions derived from one admitted transaction retain canonical
            # dependency order. An explicit user detour is different: its
            # active group owns control until that group is complete.
            active_order = GROUP_ORDER.index(active_group) if active_group in GROUP_ORDER else len(GROUP_ORDER)
            for frame_index in range(len(stack) - 1, -1, -1):
                frame = stack[frame_index]
                if str((frame or {}).get("reason") or "") != "same_turn_action_queue":
                    continue
                group = str((frame or {}).get("group") or "").strip()
                question_id = str((frame or {}).get("question_id") or "").strip()
                if group not in GROUP_ORDER or GROUP_ORDER.index(group) >= active_order:
                    continue
                question = _reconstruct_question(state, frame)
                if not question or str(question.get("id") or "") != question_id:
                    continue
                del stack[frame_index]
                state["interruption_stack"] = stack
                _record_group_transition(state, group, record_history=False)
                return question
            return None

    return _pop_interruption_question(state)


def _pop_interruption_question(state: AgentGraphState) -> PendingQuestion | None:
    """Consume the newest reconstructable interruption frame."""

    stack = list(state.get("interruption_stack") or [])
    while stack:
        frame = stack.pop()
        group = str((frame or {}).get("group") or "").strip()
        question_id = str((frame or {}).get("question_id") or "").strip()
        if not group or group not in ALLOWED_GROUPS:
            continue
        question = _reconstruct_question(state, frame)
        if not question or str(question.get("id") or "") != question_id:
            continue
        state["interruption_stack"] = stack
        _record_group_transition(state, group, record_history=False)
        return question
    state["interruption_stack"] = []
    return None


def _apply_handler_result(
    state: AgentGraphState,
    result: HandlerResult,
    *,
    owner: str,
    control_state: AgentGraphState | None = None,
) -> AgentGraphState:
    """Atomically validate and commit one domain result.

    Domain handlers run against isolated copies. A blocker therefore rejects
    every proposed domain and control mutation; only its response is appended.
    A business failure expressed as ``completion='blocked'`` without a blocker
    still commits the owner's failure delta.
    """

    previous_pending = dict(state.get("pending_question") or {})
    navigation_command = result.navigation_command
    navigation_binding = navigation_command
    if result.blocker:
        candidate: AgentGraphState = deepcopy(state)
        _append_domain_control_receipts(
            candidate,
            result.control_receipts,
            owner=owner,
        )
        current_action = dict(candidate.get("current_action") or {})
        if current_action and owner not in {"coordinator", "orientation", "recovery"}:
            retained_state = {
                "affected_group": str(
                    (ACTION_BY_TYPE.get(str(current_action.get("type") or "")) or ACTION_BY_TYPE["unknown"]).target_group
                    or candidate.get("active_group")
                    or (candidate.get("pending_question") or {}).get("group")
                    or "opening"
                ),
                "active_group": str(candidate.get("active_group") or ""),
                "pending_question": deepcopy(candidate.get("pending_question") or {}),
                "confirmed_config": deepcopy(candidate.get("confirmed_config") or {}),
                "durable_queue": _durable_actions(candidate.get("action_queue") or []),
            }
            record = domain_blocker_failure_record(
                current_action,
                owner=owner,
                validation_detail=result.blocker,
                retained_state=retained_state,
            )
            candidate["failure_recovery"] = {"status": "pending", "record": record}
            candidate.setdefault("action_errors", []).append({
                "action": record["action_identity"],
                "error": "domain_blocked",
                "failure_id": record["failure_id"],
            })
            blocker_fragment = failure_record_response_fragment(record)
        else:
            blocker_fragment = failure_response_fragment(result.blocker)
        _append_response_fragments(
            candidate,
            blocker_fragment,
        )
        _append_control_receipt(
            candidate,
            "domain_commit",
            {
                "owner": owner,
                "cause_kind": "validation_rejection",
                "completion": "rejected",
                "group_registry_contract_hash": group_registry_contract_hash(),
                "blocker_semantic_hash": semantic_hash(
                    failure_descriptor_to_dict(result.blocker)
                ),
                "pending_before_hash": _receipt_hash(previous_pending),
                "pending_after_hash": _receipt_hash(
                    candidate.get("pending_question") or {}
                ),
                "consumed_action_ids": [],
                "invalidated_groups": [],
                "invalidated_fields": [],
                "response_fragments": _response_fragment_manifest(candidate),
            },
        )
        validate_state(candidate)
        return candidate

    navigation_state: AgentGraphState | None = None
    navigation_followups: tuple[Mapping[str, Any], ...] = ()
    if result.navigation_command is not None:
        navigation_state, navigation_followups = _commit_navigation_command(
            state,
            result.navigation_command,
        )
        navigation_origin = (
            result.navigation_command.origin_group
            or str(state.get("active_group") or "")
            or str((state.get("pending_question") or {}).get("group") or "")
            or "opening"
        )
        navigation_target = (
            result.navigation_command.target_group
            or str(navigation_state.get("active_group") or "")
            or str(
                (navigation_state.get("pending_question") or {}).get("group")
                or ""
            )
            or navigation_origin
        )
        navigation_binding = replace(
            result.navigation_command,
            origin_group=navigation_origin,
            target_group=navigation_target,
        )
        result = replace(
            result,
            navigation_command=None,
            followup_actions=(
                *navigation_followups,
                *result.followup_actions,
            ),
        )
    if result.checkpoint_command is not None:
        candidate = _apply_checkpoint_command(state, result.checkpoint_command)
    elif navigation_state is not None:
        candidate = navigation_state
    elif owner == "coordinator" and control_state is not None:
        candidate = deepcopy(control_state)
    else:
        candidate = deepcopy(state)
    if result.field_reconfiguration_command is not None:
        command = result.field_reconfiguration_command
        control = dict(candidate.get("control") or {})
        if command.prerequisite_question_id:
            control["field_reconfiguration_continuation"] = {
                "group": command.group,
                "config_field": command.config_field,
                "prerequisite_question_id": command.prerequisite_question_id,
            }
        else:
            control.pop("field_reconfiguration_continuation", None)
        candidate["control"] = control
    if result.workflow_goal_command is not None:
        command = result.workflow_goal_command
        goals = [
            dict(item)
            for item in candidate.get("workflow_goals") or []
            if isinstance(item, dict)
        ]
        if command.operation == "enqueue":
            goal = dict(command.goal)
            if not any(
                item.get("target_mode") == goal.get("target_mode")
                and item.get("goal") == goal.get("goal")
                for item in goals
            ):
                goals.append(goal)
        elif command.operation == "remove_first":
            if goals:
                goals.pop(0)
        else:
            raise StateInvariantError(
                f"unsupported workflow goal command: {command.operation}"
            )
        candidate["workflow_goals"] = goals
    if result.semantic_draft_command is not None:
        command = result.semantic_draft_command
        current_draft = validate_semantic_plan_draft(
            candidate.get("semantic_plan_draft") or {}
        )
        if command.operation == "resolve":
            candidate["semantic_plan_draft"] = resolve_semantic_draft_atom(
                current_draft,
                draft_id=command.draft_id,
                revision=command.revision,
                atom_id=command.atom_id,
                resolution=command.resolution,
                resolution_hash=command.resolution_hash,
                resolution_ref=command.resolution_ref,
            )
        elif command.operation == "cancel":
            if (
                str(current_draft.get("draft_id") or "") != command.draft_id
                or int(current_draft.get("revision") or 0) != command.revision
            ):
                raise StateInvariantError("semantic draft cancellation binding mismatch")
            candidate["semantic_plan_draft"] = cancel_semantic_plan_draft(
                current_draft,
                reason=command.reason,
            )
            discard_draft_secret_references(current_draft)
        elif command.operation == "previous":
            reopened = reopen_previous_semantic_draft_atom(
                current_draft,
                draft_id=command.draft_id,
                revision=command.revision,
                atom_id=command.atom_id,
            )
            retained_refs = {
                str(atom.get("resolution_ref") or "")
                for atom in reopened.get("unresolved_atoms") or ()
                if isinstance(atom, Mapping)
            }
            for atom in current_draft.get("unresolved_atoms") or ():
                if not isinstance(atom, Mapping):
                    continue
                reference = str(atom.get("resolution_ref") or "")
                if reference and reference not in retained_refs:
                    discard_secret_reference(reference)
            candidate["semantic_plan_draft"] = reopened
        elif command.operation == "invalidate":
            if (
                str(current_draft.get("draft_id") or "") != command.draft_id
                or int(current_draft.get("revision") or 0) != command.revision
            ):
                raise StateInvariantError("semantic draft invalidation binding mismatch")
            candidate["semantic_plan_draft"] = mark_semantic_plan_draft_stale(
                current_draft,
                reasons=tuple(
                    item
                    for item in command.reason.split("|")
                    if item
                ),
            )
            discard_draft_secret_references(current_draft)
        else:
            raise StateInvariantError(
                f"unsupported semantic draft command: {command.operation}"
            )
    candidate = apply_state_delta(candidate, result.delta, owner=owner)

    recovery_pending: dict[str, Any] | None = None
    if result.recovery_command is not None:
        recovery_pending = _apply_recovery_command(candidate, result.recovery_command)

    if result.invalidated_groups:
        _apply_group_invalidation_state(candidate, result.invalidated_groups)
        invalidated = set(candidate.get("invalidated_groups") or [])
        invalidated.update(result.invalidated_groups)
        candidate["invalidated_groups"] = sorted(invalidated)
        group_states = candidate.setdefault("group_states", {})
        for group in result.invalidated_groups:
            group_states.setdefault(group, {})["status"] = "invalidated"
    for group in result.reconfigured_groups:
        mark_group_reconfigured(candidate, group)
    if result.invalidated_fields:
        confirmed = candidate.setdefault("confirmed_config", {})
        inferred = candidate.setdefault("inferred_config", {})
        for field in result.invalidated_fields:
            confirmed.pop(field, None)
            inferred.pop(field, None)
    if result.evidence:
        candidate.setdefault("audit_events", []).extend(dict(item) for item in result.evidence)
    if result.action_errors:
        candidate.setdefault("action_errors", []).extend(dict(item) for item in result.action_errors)

    if result.clear_pending:
        candidate["pending_question"] = {}
    effective_next_group = "failure_recovery" if recovery_pending else result.next_group
    if effective_next_group:
        _record_group_transition(candidate, effective_next_group)
    _append_response_fragments(candidate, *result.response_fragments)
    effective_pending = recovery_pending if recovery_pending is not None else result.pending_question
    if effective_pending is not None:
        pending = asdict(effective_pending) if is_dataclass(effective_pending) else dict(effective_pending)
        if pending != previous_pending:
            pending["same_turn_navigation_allowed"] = (
                result.completion != "blocked"
            )
            _install_pending_question(candidate, pending)
    pending_owner = str((candidate.get("pending_question") or {}).get("group") or "").strip()
    pending_is_semantic_draft = bool(
        (candidate.get("pending_question") or {}).get("semantic_draft_binding")
    )
    if pending_owner and not pending_is_semantic_draft:
        candidate["active_group"] = pending_owner
        mark_group_reconfiguring(candidate, pending_owner)
    elif result.completion == "completed":
        completed_group = str(candidate.get("active_group") or "").strip()
        if completed_group and completed_group in set(candidate.get("invalidated_groups") or []):
            mark_group_reconfigured(candidate, completed_group)
    if result.consumed_action_ids:
        applied = list(candidate.get("applied_action_ids") or [])
        for action_id in result.consumed_action_ids:
            if action_id and action_id not in applied:
                applied.append(action_id)
        candidate["applied_action_ids"] = applied[-200:]
    if result.followup_actions:
        origin_text = str((candidate.get("turn_context") or {}).get("text") or candidate.get("last_user_input") or "")
        scope = (
            f"{candidate.get('thread_id') or 'default'}:"
            f"{int(candidate.get('turn_index') or 0)}:followup"
        )
        followups = assign_action_ids(
            scope,
            origin_text,
            [dict(item) for item in result.followup_actions],
        )
        for index, action in enumerate(followups):
            action["_origin_text"] = origin_text
            action["_queue_origin_group"] = str(candidate.get("active_group") or "")
            action["_submitted_turn_index"] = int(candidate.get("turn_index") or 0)
            action["_plan_scope"] = scope
            action["_plan_index"] = index
        ordered_followups = _order_action_queue(
            candidate,
            [
                *followups,
                *[
                    _admission_action_from_envelope(item)
                    for item in candidate.get("action_queue") or []
                ],
            ],
        )
        candidate["action_queue"] = _serialize_admitted_actions(
            candidate,
            ordered_followups,
            default_origin_text=origin_text,
            default_origin_group=str(candidate.get("active_group") or ""),
            default_plan_scope=scope,
        )
    if candidate.get("pending_question") and not candidate.get("action_queue"):
        _set_pending_queue_resume(candidate, enabled=False)

    candidate = _activate_ready_deferred_group(candidate)
    _append_domain_control_receipts(
        candidate,
        result.control_receipts,
        owner=owner,
    )
    delta_paths = [
        {
            "operation": "write",
            "path": ".".join(write.path),
            "value_hash": _receipt_hash(write.value),
        }
        for write in result.delta.writes
    ]
    delta_paths.extend(
        {
            "operation": "delete",
            "path": ".".join(path),
            "value_hash": "",
        }
        for path in result.delta.deletes
    )
    before_group_states = dict(state.get("group_states") or {})
    after_group_states = dict(candidate.get("group_states") or {})
    group_state_transitions = []
    for group in sorted(set(before_group_states) | set(after_group_states)):
        before_status = str(
            (before_group_states.get(group) or {}).get("status") or ""
        )
        after_status = str(
            (after_group_states.get(group) or {}).get("status") or ""
        )
        if before_status != after_status:
            group_state_transitions.append({
                "group": group,
                "before": before_status,
                "after": after_status,
            })
    _append_control_receipt(
        candidate,
        "domain_commit",
        {
            "owner": owner,
            "cause_kind": (
                "admitted_action"
                if result.consumed_action_ids
                else "system_reconcile"
            ),
            "completion": str(result.completion or ""),
            "group_registry_contract_hash": group_registry_contract_hash(),
            "consumed_action_ids": [
                str(item) for item in result.consumed_action_ids if str(item)
            ],
            "invalidated_groups": [
                str(item) for item in result.invalidated_groups if str(item)
            ],
            "invalidated_fields": [
                str(item) for item in result.invalidated_fields if str(item)
            ],
            "reconfigured_groups": [
                str(item) for item in result.reconfigured_groups if str(item)
            ],
            "group_state_transitions": group_state_transitions,
            "material_delta": delta_paths,
            "navigation_operation": str(
                navigation_binding.operation if navigation_binding else ""
            ),
            "navigation_origin_group": str(
                navigation_binding.origin_group if navigation_binding else ""
            ),
            "navigation_target_group": str(
                navigation_binding.target_group if navigation_binding else ""
            ),
            "pending_before_hash": _receipt_hash(previous_pending),
            "pending_after_hash": _receipt_hash(
                candidate.get("pending_question") or {}
            ),
            "pending_after_id": str(
                (candidate.get("pending_question") or {}).get("id") or ""
            ),
            "response_fragments": _response_fragment_manifest(candidate),
        },
    )

    validate_state(candidate)
    return candidate


def _commit_navigation_command(
    state: AgentGraphState,
    command: NavigationCommand,
) -> tuple[AgentGraphState, tuple[Mapping[str, Any], ...]]:
    """Commit one typed navigation transaction at the control boundary."""

    candidate = deepcopy(state)
    followups: tuple[Mapping[str, Any], ...] = ()
    if command.operation == "change_group":
        target_group = command.target_group
        interrupted_pending = deepcopy(candidate.get("pending_question") or {})
        pending_group = str(interrupted_pending.get("group") or "").strip()
        if (
            interrupted_pending
            and target_group != pending_group
            and _reconstruct_question(candidate, interrupted_pending) is not None
        ):
            _push_interruption_frame(
                candidate,
                interrupted_pending,
                reason="explicit_navigation",
            )
        prerequisite = _navigation_prerequisite(candidate, target_group)
        if prerequisite:
            control = dict(candidate.get("control") or {})
            control["deferred_group"] = target_group
            candidate["control"] = control
            return _activate_group_question(candidate, prerequisite), followups
        if (
            target_group == command.origin_group
            and _active_group_has_blocking_question(candidate)
        ):
            return candidate, followups
        if _queue_has_followup_for_group(candidate, target_group):
            _record_group_transition(candidate, target_group)
            candidate["pending_question"] = {}
            return candidate, followups
        if (
            target_group == "chain_identity"
            and (candidate.get("chain_identity") or {}).get("canonical")
        ):
            candidate["pending_question"] = {}
            followups = (
                {
                    "type": "request_chain_selection",
                    "confidence": "high",
                },
            )
            return candidate, followups
        return (
            _activate_group_question(
                candidate,
                target_group,
                reconfigure=True,
            ),
            followups,
        )

    control = dict(candidate.get("control") or {})
    control.pop("field_reconfiguration_continuation", None)
    candidate["control"] = control
    resume_group = ""
    pending = dict(candidate.get("pending_question") or {})
    pending_owner = str(
        pending.get("owner")
        or GROUP_OWNER.get(str(pending.get("group") or ""), "")
    )
    runtime = DOMAIN_RUNTIME.get(pending_owner)
    if pending and runtime is not None and runtime.cancel_question is not None:
        cancellation = runtime.cancel_question(deepcopy(candidate), pending)
        resume_group = str(cancellation.navigation_resume_group or "").strip()
        candidate = _apply_handler_result(
            candidate,
            replace(
                cancellation,
                followup_actions=(),
                stop_after_response=False,
            ),
            owner=pending_owner,
        )
        followups = tuple(cancellation.followup_actions)
    interrupted_question = _pop_interruption_question(candidate)
    if interrupted_question:
        _install_pending_question(candidate, interrupted_question)
        return candidate, followups
    if resume_group:
        _discard_cancelled_origin_from_history(candidate)
    domain_resume = bool(resume_group)
    previous_group = resume_group or _pop_previous_group(candidate)
    if previous_group:
        candidate = _activate_group_question(
            candidate,
            previous_group,
            record_history=False,
            reconfigure=not domain_resume,
        )
    else:
        candidate["pending_question"] = {}
        _replace_response_fragments(
            candidate,
            _fragment(
                "harness.response.no_previous_group",
            ),
        )
    return candidate, followups


def _activate_ready_deferred_group(state: AgentGraphState) -> AgentGraphState:
    """Resume an explicit group detour before registry fallback can take over."""

    if state.get("pending_question") or state.get("action_queue"):
        return state
    control = dict(state.get("control") or {})
    deferred_group = str(control.get("deferred_group") or "").strip()
    if not deferred_group or deferred_group not in ALLOWED_GROUPS:
        return state
    prerequisite = _navigation_prerequisite(state, deferred_group)
    if prerequisite:
        if prerequisite != str(state.get("active_group") or ""):
            return _activate_group_question(state, prerequisite)
        return state
    control.pop("deferred_group", None)
    state["control"] = control
    return _activate_group_question(state, deferred_group)


def _apply_recovery_command(
    state: AgentGraphState,
    command: RecoveryCommand,
) -> dict[str, Any] | None:
    """Commit an execution failure signal at the control-plane boundary."""

    if command.operation == "activate":
        state["failure_recovery"] = {
            "status": "pending",
            "record": dict(command.record),
        }
        return question_for_recovery(state, "failure_recovery")
    recovery = state.get("failure_recovery") or {}
    if recovery.get("status") == "correcting":
        recovery["status"] = "resolved"
        recovery["validation_receipt"] = command.validation_receipt
    return None


def _apply_group_invalidation_state(
    state: AgentGraphState,
    groups: tuple[str, ...],
) -> None:
    """Apply cross-domain invalidation only at the coordinator commit boundary."""

    invalidated = set(groups)
    if "preflight_smoke_execution" in invalidated:
        execution_decision = str(
            (state.get("preflight") or {}).get("decision") or ""
        ).strip()
        state["plan"] = {}
        state["plan_file"] = ""
        state["preflight"] = (
            {
                "approved": False,
                "decision": "declined",
                "status": "declined",
            }
            if execution_decision == "declined"
            else {}
        )
        state["smoke"] = {}
        state["final_benchmark"] = {}
    if "job_monitoring" in invalidated:
        state["job"] = {}
    if "workload_rpc" in invalidated:
        state["workload"] = {}
        state["rpc_mode"] = ""
    if "target_samples_fixtures" in invalidated:
        state["fixture_evidence"] = {}
    if "qps_profile" in invalidated:
        state["qps_profile"] = {}
    if "sync_observe" in invalidated:
        state["sync_observe"] = {}
    if "endpoint_process" in invalidated:
        evidence = state.setdefault("endpoint_evidence", {})
        evidence.pop("local_rpc_url_ready", None)
        evidence.pop("sync_rpc_url_ready", None)


def _apply_checkpoint_command(state: AgentGraphState, command: CheckpointCommand) -> AgentGraphState:
    language = str(state.get("language") or "en")
    session = state.get("session") or {}
    candidate = new_state(
        str(state.get("thread_id") or "default"),
        language=language,
        session_purpose=str(session.get("purpose") or "user"),
    )
    for key in RESET_PRESERVED_KEYS:
        if key in state:
            candidate[key] = deepcopy(state[key])  # type: ignore[literal-required]
    # A workflow reset clears durable configuration, not the control-plane
    # identity of the turn that requested it.  Retaining this snapshot keeps
    # the reset auditable through the same admission/evidence path as every
    # other pending-question action.
    candidate["turn_index"] = int(state.get("turn_index") or 0)
    candidate["last_user_input"] = str(state.get("last_user_input") or "")
    candidate["turn_context"] = deepcopy(dict(state.get("turn_context") or {}))
    candidate["turn_receipt"] = deepcopy(dict(state.get("turn_receipt") or {}))
    reset_events = [{"event": "workflow_reset"}]
    draft = dict(state.get("semantic_plan_draft") or {})
    discard_state_secret_references(state)
    if draft:
        discard_draft_secret_references(draft)
        reset_events.append({
            "event": "semantic_draft_reset",
            "draft_id": str(draft.get("draft_id") or ""),
            "draft_revision": int(draft.get("revision") or 0),
        })
    candidate["audit_events"] = [
        *list(state.get("audit_events") or []),
        *reset_events,
    ]
    if command.command == "retain_safe":
        candidate["confirmed_config"] = deepcopy(dict(command.confirmed_config))
    return candidate


def _action_proposal(action: dict[str, Any], confidence: str) -> ActionProposal:
    return ActionProposal(
        action_id=str(action.get("action_id") or ""),
        action_type=str(action.get("type") or "unknown"),
        arguments={
            key: value
            for key, value in action.items()
            if key not in {"action_id", "type", "confidence", "reason"} and not key.startswith("_")
        },
        confidence=confidence if confidence in {"low", "medium", "high"} else "medium",  # type: ignore[arg-type]
        reason=str(action.get("reason") or ""),
    )


def _queue_has_followup_for_group(state: AgentGraphState, group: str) -> bool:
    action_types = {
        str(_queue_action(item).get("type") or "").strip()
        for item in state.get("action_queue") or []
        if isinstance(item, dict)
    }
    if group == "qps_profile":
        return bool(action_types & {"set_qps_mode", "request_qps_customization", "set_qps_override"})
    if group == "observability":
        return "set_observability" in action_types
    if group == "workload_rpc":
        return "set_rpc_mode" in action_types
    if group == "sync_observe":
        return bool(
            action_types
            & {
                "set_sync_observe_source",
                "clear_sync_observe_source",
                "set_sync_observe_options",
            }
        )
    return False


def _active_group_has_blocking_question(state: AgentGraphState) -> bool:
    active_group = str(state.get("active_group") or "").strip()
    if not active_group:
        return False
    return bool(_question_for_group(state, active_group))


def _ask_next_blocking_question(state: AgentGraphState) -> AgentGraphState:
    interrupted_question = _resume_suspended_question(state)
    if interrupted_question:
        _install_pending_question(state, interrupted_question)
        return state
    control = dict(state.get("control") or {})
    deferred_group = str(control.get("deferred_group") or "").strip()
    if deferred_group and deferred_group in ALLOWED_GROUPS:
        prerequisite = _navigation_prerequisite(state, deferred_group)
        if not prerequisite:
            control.pop("deferred_group", None)
            state["control"] = control
            return _activate_group_question(state, deferred_group)
        if prerequisite != str(state.get("active_group") or ""):
            return _activate_group_question(state, prerequisite)
    active_group = str(state.get("active_group") or "").strip()
    if active_group and active_group not in {"opening", "job_monitoring", "error_evidence_analysis", "report_artifact_analysis"}:
        active_question = _question_for_group(state, active_group)
        if active_question:
            _install_pending_question(state, active_question)
            if state.get("action_queue"):
                _set_pending_queue_resume(state, enabled=True)
            return state
    # A completed handler relinquishes its active group. Shared fallback then
    # computes the earliest relevant incomplete group from validated state.
    group = _next_group(state)
    _record_group_transition(state, group)
    question = _question_for_group(state, group)
    if question:
        _install_pending_question(state, question)
        if state.get("action_queue"):
            _set_pending_queue_resume(state, enabled=True)
    else:
        next_action = compute_next_action(state)
        state["pending_question"] = {}
        _append_response_fragments(
            state,
            _fragment(
                "harness.response.no_blocking_configuration",
                kind="status",
                arguments={
                    "next_group": str(
                        next_action.next_blocking_group or "job_monitoring"
                    ),
                    "execution_status": str(
                        next_action.execution_status or "not_requested"
                    ),
                },
            ),
        )
    return state


def _reconfiguration_question(state: AgentGraphState, group: str) -> PendingQuestion | None:
    """Ask a domain's first owned question without discarding confirmed state.

    The projection is used only to construct the question. The real state and
    its evidence remain intact until the owning domain applies the user's
    replacement value and dependency invalidation.
    """

    spec = GROUP_SPEC_BY_NAME.get(group)
    if spec is None or not spec.fields:
        return None
    projected = deepcopy(state)
    confirmed = dict(projected.get("confirmed_config") or {})
    for field in spec.fields:
        confirmed.pop(field, None)
        projected.pop(field, None)
    projected["confirmed_config"] = confirmed
    projected.setdefault("group_states", {}).setdefault(group, {})["status"] = "reconfiguring"
    return _question_for_group(projected, group)


def _activate_group_question(
    state: AgentGraphState,
    group: str,
    *,
    record_history: bool = True,
    reconfigure: bool = False,
) -> AgentGraphState:
    _record_group_transition(state, group, record_history=record_history)
    state.setdefault("group_states", {}).setdefault(group, {})["status"] = "in_progress"
    # An explicit navigation to chain identity means "choose or change the
    # chain" even when the current chain is already confirmed. Keep the old
    # value until the user's replacement is confirmed; the chain owner then
    # applies dependency invalidation atomically.
    question = _question_for_group(state, group)
    if reconfigure and not question:
        question = _reconfiguration_question(state, group)
    if question:
        _install_pending_question(state, question)
    else:
        state["pending_question"] = {}
        _replace_response_fragments(
            state,
            _fragment(
                "harness.response.group_has_no_blocker",
                kind="status",
                arguments={"group": group},
            ),
        )
        return state
    return state


def _record_group_transition(state: AgentGraphState, next_group: str, *, record_history: bool = True) -> None:
    current = str(state.get("active_group") or "").strip()
    if record_history and current and current != next_group:
        history = list(state.get("group_history") or [])
        if not history or history[-1] != current:
            history.append(current)
        state["group_history"] = history[-20:]
    state["active_group"] = next_group
    mark_group_reconfiguring(state, next_group)


def _install_pending_question(
    state: AgentGraphState,
    question: PendingQuestion,
    *,
    activate_group: bool = True,
) -> None:
    """Install one blocking contract and enter its owner's repair lifecycle."""

    pending = validate_pending_question_contract(dict(question))
    pending = with_pending_question_created_turn(
        pending,
        created_turn_index=int(state.get("turn_index") or 0),
    )
    turn_context = dict(state.get("turn_context") or {})
    installed = [
        dict(item)
        for item in turn_context.get("installed_questions") or []
        if isinstance(item, dict)
    ]
    pending_hash = _receipt_hash(pending)
    if not any(_receipt_hash(item) == pending_hash for item in installed):
        installed.append(dict(pending))
    turn_context["installed_questions"] = installed
    state["turn_context"] = turn_context
    state["pending_question"] = pending
    group = str(pending.get("group") or "").strip()
    if (
        group
        and activate_group
        and not pending.get("semantic_draft_binding")
        and not pending.get("secret_reentry_binding")
    ):
        state["active_group"] = group
        mark_group_reconfiguring(state, group)


def _set_pending_queue_resume(
    state: AgentGraphState,
    *,
    enabled: bool,
) -> None:
    """Transition queue-resume behavior through the signed question contract."""

    pending = dict(state.get("pending_question") or {})
    if not pending:
        return
    if (
        pending.get("semantic_draft_binding")
        or pending.get("secret_reentry_binding")
    ):
        state["pending_question"] = with_pending_question_behavior(
            pending,
            resume_action_queue=enabled,
        )
    elif enabled:
        state["pending_question"]["resume_action_queue"] = True
    else:
        state["pending_question"].pop("resume_action_queue", None)


def _pop_previous_group(state: AgentGraphState) -> str:
    history = list(state.get("group_history") or [])
    current = str(state.get("active_group") or "").strip()
    while history:
        candidate = str(history.pop() or "").strip()
        if candidate and candidate != current and candidate in ALLOWED_GROUPS:
            state["group_history"] = history
            return candidate
    state["group_history"] = history
    return ""


def _discard_cancelled_origin_from_history(state: AgentGraphState) -> None:
    """Consume the abandoned group's history entry exactly once."""

    history = list(state.get("group_history") or [])
    origin = str(state.get("active_group") or "").strip()
    if history and str(history[-1] or "").strip() == origin:
        history.pop()
    state["group_history"] = history


def _next_group(state: AgentGraphState) -> str:
    """Return the next blocking group name.

    Delegates to `routing.next_group_and_reason`, the single shared
    implementation also used by `oracle.py`'s status/explanation text, so
    the two can never disagree about what group is next (see architecture
    audit Finding B1).
    """

    group, _reason = next_group_and_reason(state)
    return group


def _question_for_group(state: AgentGraphState, group: str) -> PendingQuestion | None:
    owner = GROUP_OWNER.get(group, "")
    runtime = DOMAIN_RUNTIME.get(owner)
    return runtime.question_factory(state, group) if runtime and runtime.question_factory else None


def _reconstruct_question(
    state: AgentGraphState,
    identity: Mapping[str, Any],
) -> PendingQuestion | None:
    """Rebuild one suspended typed question from current authoritative state."""

    group = str(identity.get("group") or "").strip()
    question_id = str(identity.get("question_id") or identity.get("id") or "").strip()
    if not group or not question_id:
        return None
    owner = str(identity.get("owner") or GROUP_OWNER.get(group, "")).strip()
    runtime = DOMAIN_RUNTIME.get(owner)
    if runtime and runtime.question_reconstructor:
        question = runtime.question_reconstructor(state, dict(identity))
        if question and str(question.get("id") or "") == question_id:
            return question
    field = str(identity.get("field") or "").strip()
    reconfiguration_target = str(
        identity.get("reconfiguration_target_field") or ""
    ).strip()
    if field and reconfiguration_target == field:
        baseline_revision = int(
            identity.get("reconfiguration_baseline_revision") or 0
        )
        current_revision = field_confirmation_revision(
            state,
            field,
            group=group,
        )
        if current_revision > baseline_revision:
            return None
        runtime = DOMAIN_RUNTIME.get(owner)
        question = (
            runtime.field_question_factory(state, group, field)
            if runtime and runtime.field_question_factory
            else None
        )
        if question and str(question.get("id") or "") == question_id:
            return question
    question = _question_for_group(state, group)
    if question and str(question.get("id") or "") == question_id:
        return question
    return None

def _resume_field_reconfiguration_after_prerequisite(
    state: AgentGraphState,
    *,
    answered_question: PendingQuestion,
) -> AgentGraphState:
    """Consume one durable field-edit continuation after its prerequisite."""

    control = dict(state.get("control") or {})
    continuation = control.get("field_reconfiguration_continuation")
    if not isinstance(continuation, dict):
        return state
    if str(answered_question.get("id") or "") != str(
        continuation.get("prerequisite_question_id") or ""
    ):
        return state
    group = str(continuation.get("group") or "").strip()
    field = str(continuation.get("config_field") or "").strip()
    if group_for_field(field) != group:
        control.pop("field_reconfiguration_continuation", None)
        state["control"] = control
        return state
    runtime = DOMAIN_RUNTIME.get(GROUP_OWNER.get(group, ""))
    question = (
        runtime.field_question_factory(state, group, field)
        if runtime and runtime.field_question_factory
        else None
    )
    owner_spec = GROUP_SPEC_BY_NAME.get(group)
    if (
        not question
        or not owner_spec
        or str(question.get("group") or "") != group
        or str(question.get("id") or "") not in owner_spec.questions
        or str(question.get("field") or "") not in owner_spec.fields
    ):
        return state
    if str(question.get("field") or "") == field:
        control.pop("field_reconfiguration_continuation", None)
    else:
        continuation["prerequisite_question_id"] = str(question.get("id") or "")
        control["field_reconfiguration_continuation"] = continuation
    state["control"] = control
    _record_group_transition(state, group)
    _install_pending_question(state, question)
    return state
