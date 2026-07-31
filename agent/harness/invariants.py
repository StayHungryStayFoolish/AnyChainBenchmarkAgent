"""Global state invariants enforced after every Harness transition."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping

from ..llm.types import (
    MAX_PROVIDER_ATTEMPT_RECORDS,
    validate_provider_attempt_record,
)

from agent.workflows.group_registry import GROUPS
from .contracts import DOMAIN_CONTROL_ROOTS, StateDelta, StatePath
from .domains.registry import GROUP_OWNER
from .state import AgentGraphState
from .input_values import normalize_target_mode
from .questions import validate_pending_question_contract
from .semantic_drafts import (
    semantic_action_secret_binding_hash,
    semantic_draft_question_binding,
    semantic_draft_uses_current_authority,
    validate_semantic_draft_finalization_receipt_body,
    validate_semantic_plan_draft,
)


class StateInvariantError(RuntimeError):
    pass


def _registry_owner_paths() -> dict[str, set[StatePath]]:
    paths: dict[str, set[StatePath]] = {}
    top_level = set(AgentGraphState.__optional_keys__) | set(AgentGraphState.__required_keys__)
    for spec in GROUPS:
        owned = paths.setdefault(spec.owner, set())
        for field in spec.fields:
            owned.add(("confirmed_config", field))
            if field in top_level:
                owned.add((field,))
    return paths


_REGISTRY_OWNER_PATHS = _registry_owner_paths()
_INVALIDATION_PATHS: set[StatePath] = {
    ("preflight",),
    ("plan",),
    ("plan_file",),
    ("smoke",),
    ("final_benchmark",),
    ("job",),
}


OWNER_PATH_POLICY: dict[str, frozenset[StatePath]] = {
    "orientation": frozenset(
        _REGISTRY_OWNER_PATHS.get("orientation", set())
        | {
            ("language",),
            ("checkpoint_recovery",),
            ("resume_context",),
        }
    ),
    "environment": frozenset(
        _REGISTRY_OWNER_PATHS.get("environment", set())
        | {
            ("inferred_config",),
            ("endpoint_evidence", "proposed_values"),
            # Structured review is one typed environment action. These exact
            # fields are shared proposal inputs, not workflow-control grants.
            ("confirmed_config", "CHAIN_REST_URL"),
            ("confirmed_config", "CHAIN_INDEXER_URL"),
            ("confirmed_config", "CHAIN_SIDECAR_URL"),
            ("confirmed_config", "CHAIN_EVM_RPC_URL"),
            ("confirmed_config", "CHAIN_JSON_RPC_URL"),
            ("confirmed_config", "CHAIN_MIRROR_URL"),
            ("confirmed_config", "RPC_API_KEY"),
        }
    ),
    "chain_rpc": frozenset(
        _REGISTRY_OWNER_PATHS.get("chain_rpc", set())
        | {("target_mode_change_candidate",)}
    ),
    "performance": frozenset(
        _REGISTRY_OWNER_PATHS.get("performance", set())
    ),
    "sync_observe": frozenset(
        _REGISTRY_OWNER_PATHS.get("sync_observe", set())
    ),
    "execution": frozenset(
        _REGISTRY_OWNER_PATHS.get("execution", set())
        | _INVALIDATION_PATHS
    ),
    "recovery": frozenset(
        _REGISTRY_OWNER_PATHS.get("recovery", set())
        | {
            ("failure_recovery",),
        }
    ),
    "analysis": frozenset(
        _REGISTRY_OWNER_PATHS.get("analysis", set())
    ),
    "coordinator": frozenset(),
}


def validate_delta_owner(delta: StateDelta, owner: str) -> None:
    """Reject every delta path not assigned to the registry-supplied owner."""

    if owner not in OWNER_PATH_POLICY:
        raise StateInvariantError(f"unknown delta owner: {owner or '<missing>'}")
    allowed = OWNER_PATH_POLICY[owner]
    for path in [*(write.path for write in delta.writes), *delta.deletes]:
        if not path:
            raise StateInvariantError("state delta contains an empty path")
        if path[0] in DOMAIN_CONTROL_ROOTS or path[0].startswith("_"):
            raise StateInvariantError(
                f"domain {owner} cannot mutate coordinator path: {'.'.join(path)}"
            )
        if not any(path[: len(prefix)] == prefix for prefix in allowed):
            raise StateInvariantError(
                f"domain {owner} does not own state path: {'.'.join(path)}"
            )


def apply_state_delta(
    state: AgentGraphState,
    delta: StateDelta,
    *,
    owner: str,
) -> AgentGraphState:
    """Apply one owner-validated delta to an isolated candidate state."""

    validate_delta_owner(delta, owner)
    candidate: AgentGraphState = deepcopy(state)
    for write in delta.writes:
        _write_path(candidate, write.path, deepcopy(write.value))
    for path in delta.deletes:
        _delete_path(candidate, path)
    return candidate


def _write_path(state: dict[str, Any], path: StatePath, value: Any) -> None:
    current: dict[str, Any] = state
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[path[-1]] = value


def _delete_path(state: dict[str, Any], path: StatePath) -> None:
    current: Any = state
    for part in path[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(path[-1], None)


def _validate_provider_attempt_evidence(evidence: Any) -> None:
    if not isinstance(evidence, (list, tuple)):
        raise StateInvariantError(
            "provider attempt evidence must be a list"
        )
    if len(evidence) > MAX_PROVIDER_ATTEMPT_RECORDS:
        raise StateInvariantError(
            "provider attempt evidence exceeds the per-turn limit"
        )
    for record in evidence:
        if not isinstance(record, Mapping):
            raise StateInvariantError(
                "provider attempt evidence shape is invalid"
            )
        try:
            validate_provider_attempt_record(record)
        except ValueError as exc:
            raise StateInvariantError(
                "provider attempt evidence semantics are invalid"
            ) from exc


def _validate_semantic_finalization_action_sets(
    state: AgentGraphState,
) -> None:
    """Prove that an in-flight finalized draft retains its complete action set."""

    session_id = str(
        (state.get("session") or {}).get("id")
        or state.get("thread_id")
        or ""
    )
    queued_ids: set[str] = set()
    queue_receipts: list[dict[str, Any]] = []
    for envelope in [
        *(
            item
            for item in state.get("action_queue") or ()
            if isinstance(item, Mapping)
        ),
        *(
            [state.get("selected_action")]
            if isinstance(state.get("selected_action"), Mapping)
            and state.get("selected_action")
            else []
        ),
    ]:
        action_id = str(envelope.get("action_id") or "")
        if action_id:
            queued_ids.add(action_id)
        receipt = dict(
            (envelope.get("admission_metadata") or {}).get(
                "semantic_draft_finalization_receipt"
            )
            or {}
        )
        if receipt:
            expected_binding_hash = str(
                (receipt.get("secret_binding_hashes") or {}).get(action_id)
                or ""
            )
            observed_binding_hash = semantic_action_secret_binding_hash({
                "_semantic_secret_bindings": (
                    envelope.get("admission_metadata") or {}
                ).get("semantic_secret_bindings")
                or (),
            })
            if expected_binding_hash != observed_binding_hash:
                raise StateInvariantError(
                    "semantic draft finalization secret binding changed"
                )
            queue_receipts.append(receipt)

    audit_receipts = [
        {
            key: value
            for key, value in item.items()
            if key != "event"
        }
        for item in state.get("audit_events") or ()
        if isinstance(item, Mapping)
        and str(item.get("event") or "") == "semantic_draft_finalized"
    ]
    receipts = [*queue_receipts, *audit_receipts]
    by_transaction: dict[tuple[str, str], dict[str, Any]] = {}
    for receipt in receipts:
        try:
            receipt = validate_semantic_draft_finalization_receipt_body(
                receipt,
                session_id=session_id,
            )
        except ValueError as exc:
            raise StateInvariantError(str(exc)) from exc
        key = (
            str(receipt.get("draft_id") or ""),
            str(receipt.get("admission_transaction_hash") or ""),
        )
        if not all(key):
            raise StateInvariantError(
                "semantic draft finalization receipt identity is incomplete"
            )
        existing = by_transaction.get(key)
        if existing and existing != receipt:
            raise StateInvariantError(
                "semantic draft finalization receipts are inconsistent"
            )
        by_transaction[key] = receipt

    applied_by_transaction: dict[tuple[str, str], set[str]] = {}
    for item in state.get("audit_events") or ():
        if (
            not isinstance(item, Mapping)
            or str(item.get("event") or "")
            != "semantic_draft_finalization_action_applied"
        ):
            continue
        key = (
            str(item.get("draft_id") or ""),
            str(item.get("admission_transaction_hash") or ""),
        )
        receipt = by_transaction.get(key)
        if receipt is None or str(item.get("receipt_hash") or "") != str(
            receipt.get("receipt_hash") or ""
        ):
            raise StateInvariantError(
                "semantic draft finalization application receipt is invalid"
            )
        action_id = str(item.get("action_id") or "")
        if action_id not in {
            str(value)
            for value in receipt.get("final_action_ids") or ()
        }:
            raise StateInvariantError(
                "semantic draft finalization applied an unknown action"
            )
        applied_by_transaction.setdefault(key, set()).add(action_id)

    for receipt in by_transaction.values():
        key = (
            str(receipt.get("draft_id") or ""),
            str(receipt.get("admission_transaction_hash") or ""),
        )
        expected_ids = {
            str(item)
            for item in receipt.get("final_action_ids") or ()
            if str(item)
        }
        accounted_ids = queued_ids | applied_by_transaction.get(key, set())
        if not expected_ids <= accounted_ids:
            raise StateInvariantError(
                "semantic draft finalization action set is incomplete"
            )


def validate_state(state: AgentGraphState) -> None:
    from .secret_refs import (
        raw_secret_paths_in_state,
        validate_state_secret_bindings,
    )

    try:
        validate_state_secret_bindings(state)
        raw_secret_paths = raw_secret_paths_in_state(state)
        if raw_secret_paths:
            raise ValueError(
                "durable state contains unprojected secret material: "
                + ", ".join(raw_secret_paths)
            )
    except ValueError as exc:
        raise StateInvariantError(str(exc)) from exc
    declared_roots = (
        set(AgentGraphState.__optional_keys__)
        | set(AgentGraphState.__required_keys__)
    )
    undeclared_roots = sorted(set(state) - declared_roots)
    if undeclared_roots:
        raise StateInvariantError(
            "state contains undeclared top-level roots: "
            + ", ".join(undeclared_roots)
        )
    turn_context = state.get("turn_context") or {}
    _validate_provider_attempt_evidence(
        turn_context["provider_attempt_evidence"]
        if "provider_attempt_evidence" in turn_context
        else ()
    )

    planning = state.get("semantic_planning") or {}
    if planning:
        if int(planning.get("contract_version") or 0) != 1:
            raise StateInvariantError(
                "semantic planning document has an unsupported contract version"
            )
        status = str(planning.get("status") or "")
        if status not in {
            "partition",
            "compile_owner",
            "review_plan",
            "reviewed",
            "failed",
        }:
            raise StateInvariantError(
                f"semantic planning document has invalid status: {status!r}"
            )
        requests = planning.get("owner_requests") or []
        if not isinstance(requests, list):
            raise StateInvariantError(
                "semantic planning owner requests must be a list"
            )
        cursor = int(planning.get("owner_cursor") or 0)
        if cursor < 0 or cursor > len(requests):
            raise StateInvariantError(
                "semantic planning owner cursor is outside its schedule"
            )
        owners: list[str] = []
        for request in requests:
            if not isinstance(request, Mapping):
                raise StateInvariantError(
                    "semantic planning owner request must be an object"
                )
            owner = str(request.get("owner") or "")
            if owner not in OWNER_PATH_POLICY:
                raise StateInvariantError(
                    f"semantic planning request has unknown owner: {owner!r}"
                )
            if owner in owners:
                raise StateInvariantError(
                    f"semantic planning schedule repeats owner: {owner}"
                )
            owners.append(owner)
        documents = planning.get("owner_documents") or {}
        if not isinstance(documents, Mapping):
            raise StateInvariantError(
                "semantic planning owner documents must be an object"
            )
        expected_documents = set(owners[:cursor])
        if set(documents) != expected_documents:
            raise StateInvariantError(
                "semantic planning documents do not match the compiled cursor"
            )
        if status == "review_plan" and cursor != len(requests):
            raise StateInvariantError(
                "semantic planning reached review before every owner compiled"
            )

    draft = state.get("semantic_plan_draft") or {}
    if draft:
        try:
            validate_semantic_plan_draft(
                draft,
                validate_current_authority=semantic_draft_uses_current_authority(
                    draft
                ),
            )
        except ValueError as exc:
            raise StateInvariantError(str(exc)) from exc

    active_group = str(state.get("active_group") or "opening")
    if active_group not in GROUP_OWNER:
        raise StateInvariantError(f"unknown active group: {active_group}")

    pending = state.get("pending_question") or {}
    if (
        draft
        and draft.get("status") == "awaiting_clarification"
        and dict(pending.get("semantic_draft_binding") or {})
        != semantic_draft_question_binding(draft)
    ):
        raise StateInvariantError(
            "awaiting semantic draft has no matching pending question"
        )
    if pending:
        draft_binding = dict(pending.get("semantic_draft_binding") or {})
        secret_reentry_binding = dict(
            pending.get("secret_reentry_binding") or {}
        )
        if draft_binding or secret_reentry_binding:
            try:
                validate_pending_question_contract(dict(pending))
            except (TypeError, ValueError) as exc:
                raise StateInvariantError(str(exc)) from exc
        if draft_binding:
            if not draft or draft.get("status") != "awaiting_clarification":
                raise StateInvariantError(
                    "semantic draft question has no awaiting draft"
                )
            expected_binding = semantic_draft_question_binding(draft)
            if draft_binding != expected_binding:
                raise StateInvariantError(
                    "semantic draft question binding does not match its draft"
                )
        if secret_reentry_binding:
            owner_kind = str(secret_reentry_binding.get("owner_kind") or "")
            if owner_kind == "semantic_draft":
                if not draft or draft.get("status") != "ready_for_review":
                    raise StateInvariantError(
                        "semantic secret re-entry has no ready draft"
                    )
                durable_bindings = {
                    (
                        str(item.get("reference") or ""),
                        str(item.get("atom_id") or ""),
                        str(item.get("value_hash") or ""),
                    )
                    for item in (
                        list(draft.get("source_secret_bindings") or ())
                        + [
                            {
                                "reference": atom.get("resolution_ref"),
                                "atom_id": atom.get("atom_id"),
                                "value_hash": atom.get("resolution_hash"),
                            }
                            for atom in draft.get("unresolved_atoms") or ()
                            if isinstance(atom, Mapping)
                            and str(atom.get("resolution_ref") or "")
                        ]
                    )
                    if isinstance(item, Mapping)
                }
                valid_binding = bool(
                    str(secret_reentry_binding.get("scope_id") or "")
                    == str(draft.get("draft_id") or "")
                    and str(secret_reentry_binding.get("owner_id") or "")
                    == str(draft.get("draft_id") or "")
                    and int(
                        secret_reentry_binding.get("owner_revision") or 0
                    )
                    == int(draft.get("revision") or 0)
                    and (
                        str(secret_reentry_binding.get("reference") or ""),
                        str(secret_reentry_binding.get("atom_id") or ""),
                        str(secret_reentry_binding.get("value_hash") or ""),
                    )
                    in durable_bindings
                )
            else:
                binding_tuple = (
                    str(secret_reentry_binding.get("reference") or ""),
                    str(secret_reentry_binding.get("scope_id") or ""),
                    str(secret_reentry_binding.get("atom_id") or ""),
                    str(secret_reentry_binding.get("value_hash") or ""),
                )
                valid_binding = bool(
                    owner_kind == "durable_state"
                    and str(secret_reentry_binding.get("owner_id") or "")
                    == binding_tuple[0]
                    and int(
                        secret_reentry_binding.get("owner_revision") or 0
                    )
                    == 0
                    and binding_tuple
                    in {
                        (
                            str(item.get("reference") or ""),
                            str(item.get("scope_id") or ""),
                            str(item.get("atom_id") or ""),
                            str(item.get("value_hash") or ""),
                        )
                        for item in state.get("secret_bindings") or ()
                        if isinstance(item, Mapping)
                    }
                )
            if not valid_binding:
                raise StateInvariantError(
                    "secret re-entry binding does not match its owner"
                )
        pending_group = str(pending.get("group") or "")
        pending_owner = str(pending.get("owner") or "")
        if pending_group not in GROUP_OWNER:
            raise StateInvariantError(f"pending question has unknown group: {pending_group}")
        if not pending_owner:
            raise StateInvariantError("pending question is missing its explicit owner")
        if pending_owner not in {*GROUP_OWNER.values(), "coordinator"}:
            raise StateInvariantError(
                f"pending question has unknown owner: {pending_owner!r}"
            )
        if (
            (draft_binding or secret_reentry_binding)
            and pending_owner != "coordinator"
        ):
            raise StateInvariantError(
                "semantic draft question requires coordinator ownership"
            )
        if (
            pending_group != active_group
            and not draft_binding
            and not secret_reentry_binding
        ):
            raise StateInvariantError(
                f"pending question owner {pending_group} differs from active group {active_group}"
            )
        invalidated = set(state.get("invalidated_groups") or [])
        group_state = (state.get("group_states") or {}).get(pending_group) or {}
        if (
            not draft_binding
            and pending_group in invalidated
            and str(group_state.get("status") or "").lower() not in {
            "in_progress",
            "reconfiguring",
            }
        ):
            raise StateInvariantError(f"pending question belongs to invalidated group: {pending_group}")
        if (
            not draft_binding
            and str(group_state.get("status") or "").lower()
            in {"complete", "completed"}
        ):
            raise StateInvariantError(f"pending question belongs to completed group: {pending_group}")
        for capability in pending.get("requires_capabilities") or []:
            if str(capability) == "chain_identity":
                identity = state.get("chain_identity") or {}
                if not str(identity.get("canonical") or identity.get("raw") or "").strip():
                    raise StateInvariantError("pending question requires chain identity")

    queue = state.get("action_queue") or []
    for item in queue:
        if not isinstance(item, dict):
            raise StateInvariantError("action queue item must be an object")
        if (
            not str(item.get("action_id") or "")
            or not str(item.get("action_type") or "")
            or not str(item.get("owner") or "")
            or not isinstance(item.get("arguments"), dict)
            or str(item.get("status") or "") != "admitted"
        ):
            raise StateInvariantError(
                "action queue contains a raw or incomplete action envelope"
            )
        private_keys = sorted(
            str(key)
            for key in item
            if str(key).startswith("_")
        )
        if private_keys:
            raise StateInvariantError(
                "action queue envelope contains private metadata: "
                + ", ".join(private_keys)
            )
    action_ids = [str(item.get("action_id") or "") for item in queue if isinstance(item, dict)]
    non_empty_ids = [item for item in action_ids if item]
    if len(non_empty_ids) != len(set(non_empty_ids)):
        raise StateInvariantError("action queue contains duplicate action ids")
    _validate_semantic_finalization_action_sets(state)

    selected = state.get("selected_action") or {}
    if selected:
        selected_id = str(selected.get("action_id") or "")
        selected_type = str(selected.get("action_type") or "")
        selected_owner = str(selected.get("owner") or "")
        if (
            not selected_id
            or not selected_type
            or not selected_owner
            or str(selected.get("status") or "") != "selected"
        ):
            raise StateInvariantError("selected action envelope is incomplete")
        if not queue or str((queue[0] or {}).get("action_id") or "") != selected_id:
            raise StateInvariantError("selected action is not the admitted queue head")
        current = state.get("current_action") or {}
        if str(current.get("action_id") or "") != selected_id:
            raise StateInvariantError("selected action and current action differ")
        control_owner = str((state.get("control") or {}).get("selected_owner") or "")
        if control_owner != selected_owner:
            raise StateInvariantError("selected action owner differs from graph route owner")
    elif (state.get("control") or {}).get("selected_owner"):
        raise StateInvariantError("graph route owner exists without a selected action")

    prepared = state.get("pending_domain_result") or {}
    if prepared:
        prepared_action = dict(prepared.get("action") or {})
        if not selected:
            raise StateInvariantError("pending domain result exists without a selected action")
        if str(prepared_action.get("action_id") or "") != str(selected.get("action_id") or ""):
            raise StateInvariantError("pending domain result belongs to another action")

    intent = state.get("side_effect_intent") or {}
    receipt = state.get("side_effect_receipt") or {}
    if receipt and not intent:
        raise StateInvariantError("side-effect receipt exists without its intent")
    if intent and receipt and str(receipt.get("intent_id") or "") != str(intent.get("intent_id") or ""):
        raise StateInvariantError("side-effect receipt belongs to another intent")

    turn_receipt = state.get("turn_receipt") or {}
    if turn_receipt:
        if not str(turn_receipt.get("turn_id") or ""):
            raise StateInvariantError("turn receipt has no turn identity")
        if not str(turn_receipt.get("input_hash") or ""):
            raise StateInvariantError("turn receipt has no input hash")
        submitted_input_hash = str(
            turn_receipt.get("submitted_input_hash")
            or turn_receipt.get("input_hash")
            or ""
        )
        if re.fullmatch(r"[0-9a-f]{64}", submitted_input_hash) is None:
            raise StateInvariantError(
                "turn receipt has no submitted input identity"
            )
        admitted_ids = [
            str(item)
            for item in turn_receipt.get("admitted_action_ids") or []
            if str(item)
        ]
        semantic_order = [
            str(item)
            for item in turn_receipt.get("semantic_order") or []
            if str(item)
        ]
        execution_order = [
            str(item)
            for item in turn_receipt.get("execution_order") or []
            if str(item)
        ]
        if len(admitted_ids) != len(set(admitted_ids)):
            raise StateInvariantError("turn receipt contains duplicate admitted actions")
        if semantic_order != admitted_ids:
            raise StateInvariantError("turn receipt semantic order differs from admission order")
        if any(action_id not in admitted_ids for action_id in execution_order):
            raise StateInvariantError("turn receipt executed an unadmitted action")
        semantic_units = [
            dict(item)
            for item in turn_receipt.get("semantic_units") or []
            if isinstance(item, Mapping)
        ]
        unit_ids = [
            str(item.get("unit_id") or "")
            for item in semantic_units
            if str(item.get("unit_id") or "")
        ]
        if len(unit_ids) != len(set(unit_ids)):
            raise StateInvariantError("turn receipt contains duplicate semantic units")
        action_units = {
            str(action_id): [str(unit_id) for unit_id in unit_id_values]
            for action_id, unit_id_values in dict(
                turn_receipt.get("action_unit_bindings") or {}
            ).items()
        }
        unit_actions = {
            str(unit_id): [str(action_id) for action_id in action_id_values]
            for unit_id, action_id_values in dict(
                turn_receipt.get("unit_action_bindings") or {}
            ).items()
        }
        if any(action_id not in admitted_ids for action_id in action_units):
            raise StateInvariantError(
                "turn receipt binds semantic units to an unadmitted action"
            )
        if any(unit_id not in unit_ids for unit_id in unit_actions):
            raise StateInvariantError(
                "turn receipt binds actions to an unknown semantic unit"
            )
        for action_id, bound_units in action_units.items():
            for unit_id in bound_units:
                if action_id not in unit_actions.get(unit_id, []):
                    raise StateInvariantError(
                        "turn receipt action/unit bindings are not bidirectional"
                    )
        omission_checks = [
            dict(item)
            for item in turn_receipt.get("sibling_omission_checks") or []
            if isinstance(item, Mapping)
        ]
        if any(str(item.get("verdict") or "") == "invalid" for item in omission_checks):
            raise StateInvariantError(
                "turn receipt contains an invalid sibling omission verdict"
            )
        if str(turn_receipt.get("status") or "") in {
            "planned",
            "executing",
            "completed",
            "blocked",
            "failed",
        }:
            unresolved = {
                str(item)
                for item in turn_receipt.get("unresolved_units") or []
                if str(item)
            }
            for unit in semantic_units:
                unit_id = str(unit.get("unit_id") or "")
                disposition = str(unit.get("disposition") or "")
                bindings = unit_actions.get(unit_id, [])
                if disposition == "action" and not bindings:
                    raise StateInvariantError(
                        f"turn receipt omitted admitted semantic unit: {unit_id}"
                    )
                if disposition == "unresolved" and (
                    unit_id not in unresolved or bindings
                ):
                    raise StateInvariantError(
                        f"turn receipt unresolved unit is inconsistent: {unit_id}"
                    )
                if disposition == "context" and bindings:
                    raise StateInvariantError(
                        f"turn receipt context unit owns an action: {unit_id}"
                    )

    history = state.get("group_history") or []
    unknown_history = [group for group in history if group not in GROUP_OWNER]
    if unknown_history:
        raise StateInvariantError(f"group history contains unknown groups: {unknown_history}")

    workflow_goals = state.get("workflow_goals") or []
    if not isinstance(workflow_goals, list):
        raise StateInvariantError("workflow goals must be a list")
    goal_keys: set[tuple[str, str]] = set()
    for item in workflow_goals:
        if not isinstance(item, dict):
            raise StateInvariantError("workflow goal must be an object")
        target_mode = normalize_target_mode(item.get("target_mode"))
        goal = str(item.get("goal") or "").strip()
        source = str(item.get("source_evidence") or "").strip()
        if not target_mode or not goal or not source:
            raise StateInvariantError("workflow goal requires canonical target_mode, goal, and source_evidence")
        key = (target_mode, goal)
        if key in goal_keys:
            raise StateInvariantError("workflow goals contain a duplicate target and goal")
        goal_keys.add(key)

    if str(state.get("workflow_mode") or "") == "sync_observe" and any(
        (state.get("rpc_mode"), state.get("workload"), state.get("custom_rpc"), state.get("fixture_evidence"), state.get("qps_profile"))
    ):
        raise StateInvariantError("sync-observe state contains RPC benchmark workload or QPS configuration")

    identity = state.get("chain_identity") or {}
    pending_id = str((state.get("pending_question") or {}).get("id") or "")
    case3_statuses = {
        "unsupported_family_handoff",
        "case3_collecting_evidence",
        "case3_needs_evidence",
    }
    case3_active = str(identity.get("status") or "") in case3_statuses or pending_id in {
        "case3_protocol_evidence",
        "case3_evidence_input",
        "case3_evidence_next",
    }
    if case3_active:
        handoff = state.get("secondary_handoff") or {}
        if str(identity.get("case") or "") != "case3":
            raise StateInvariantError("Case 3 evidence state requires case3 chain identity ownership")
        if str(identity.get("adapter_family") or "") != "unsupported":
            raise StateInvariantError("Case 3 evidence state requires an unsupported adapter family")
        if str(handoff.get("status") or "") != "collecting_evidence":
            raise StateInvariantError("Case 3 evidence state requires an active collecting handoff")
        if str(handoff.get("kind") or "") != "case3_protocol_adapter_implementation":
            raise StateInvariantError("Case 3 evidence state requires the protocol-adapter handoff owner")


def verify_expected_patch(state: AgentGraphState, expected: dict[str, Any]) -> None:
    """Fail an option execution whose declared postcondition did not occur."""

    for dotted_key, expected_value in expected.items():
        current: Any = state
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                raise StateInvariantError(f"missing option postcondition: {dotted_key}")
            current = current[part]
        if current != expected_value:
            raise StateInvariantError(
                f"option postcondition mismatch for {dotted_key}: expected {expected_value!r}, got {current!r}"
            )
