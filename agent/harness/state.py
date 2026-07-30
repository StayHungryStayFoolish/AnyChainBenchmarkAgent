"""Typed graph state for the AnyChain Agent Harness."""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from typing import Any, Literal, Mapping, TypedDict

from agent.workflows.group_registry import GROUP_ORDER
WorkflowMode = Literal["rpc_benchmark", "sync_observe", ""]
TargetMode = Literal["fake-node", "real-node", "sync-observe", ""]


class SecretBinding(TypedDict):
    reference: str
    scope_id: str
    atom_id: str
    value_hash: str


class PendingQuestion(TypedDict, total=False):
    contract_version: int
    id: str
    group: str
    owner: str
    subgroup: str
    kind: str
    prompt: str
    options: list[dict[str, Any]]
    field: str
    state_patch_on_valid: dict[str, Any]
    validation: dict[str, Any]
    next_on_valid: str
    next_on_invalid: str
    manual_input_allowed: bool
    value_domain: str
    structured_input_owner: bool
    structured_config_key: str
    candidate_bindings: list[dict[str, Any]]
    accepted_action_types: list[str]
    manual_action: dict[str, Any]
    requires_capabilities: list[str]
    resume_action_queue: bool
    supersedes_action_types: list[str]
    reconfiguration_target_field: str
    created_turn_index: int
    queue_barrier: bool
    barrier_policy: str
    semantic_draft_binding: dict[str, Any]
    secret_reentry_binding: dict[str, Any]
    sensitive_input: bool
    semantic_draft_question_hash: str


class AgentGraphState(TypedDict, total=False):
    schema_version: int
    thread_id: str
    session: dict[str, Any]
    language: str
    last_user_input: str
    turn_index: int
    turn_context: dict[str, Any]
    control: dict[str, Any]
    semantic_planning: dict[str, Any]
    semantic_plan_draft: dict[str, Any]
    secret_bindings: list[SecretBinding]
    proposed_actions: list[dict[str, Any]]
    input_shape: str
    active_group: str
    discovery: dict[str, Any]
    framework_summary: dict[str, Any]
    web_research: dict[str, Any]
    pending_question: PendingQuestion
    action_queue: list[dict[str, Any]]
    selected_action: dict[str, Any]
    pending_domain_result: dict[str, Any]
    side_effect_intent: dict[str, Any]
    side_effect_receipt: dict[str, Any]
    turn_receipt: dict[str, Any]
    current_action: dict[str, Any]
    completed_actions: list[dict[str, Any]]
    applied_action_ids: list[str]
    action_errors: list[dict[str, Any]]
    group_states: dict[str, dict[str, Any]]
    group_history: list[str]
    confirmed_config: dict[str, Any]
    inferred_config: dict[str, Any]
    invalidated_groups: list[str]
    interruption_stack: list[dict[str, Any]]
    chain_identity: dict[str, Any]
    target_mode: TargetMode
    target_mode_change_candidate: str
    workflow_mode: WorkflowMode
    rpc_mode: str
    workload: dict[str, Any]
    custom_rpc: dict[str, Any]
    qps_profile: dict[str, Any]
    endpoint_evidence: dict[str, Any]
    fixture_evidence: dict[str, Any]
    sync_observe: dict[str, Any]
    observability: dict[str, Any]
    advanced_tuning: dict[str, Any]
    secondary_handoff: dict[str, Any]
    preflight: dict[str, Any]
    plan: dict[str, Any]
    plan_file: str
    smoke: dict[str, Any]
    final_benchmark: dict[str, Any]
    job: dict[str, Any]
    report_context: dict[str, Any]
    failure_recovery: dict[str, Any]
    evidence_buffer: list[dict[str, Any]]
    evidence_collection: dict[str, Any]
    response_fragments: list[dict[str, Any]]
    visible_response: list[str]
    audit_events: list[dict[str, Any]]
    checkpoint_recovery: dict[str, Any]
    resume_context: dict[str, Any]
    workflow_goals: list[dict[str, Any]]


# Durable environment ownership excludes invocation-only discovery. Discovery
# is refreshed for every graph invocation and is deliberately cleared before
# checkpointing; retention checks must only claim the state that can persist.
DURABLE_ENVIRONMENT_STATE_ROOTS = frozenset({
    "inferred_config",
    "confirmed_config",
})
INVOCATION_ENVIRONMENT_STATE_ROOTS = frozenset({"discovery"})

# These roots are the state-schema authority for values that may own opaque
# secret references. Secret lifecycle code derives ownership and migration
# checks from this contract instead of maintaining a parallel root list.
DURABLE_SECRET_CAPABLE_STATE_ROOTS = frozenset({
    "action_queue",
    "selected_action",
    "current_action",
    "pending_domain_result",
    "side_effect_intent",
    "side_effect_receipt",
    "group_states",
    "inferred_config",
    "confirmed_config",
    "chain_identity",
    "endpoint_evidence",
    "custom_rpc",
    "workload",
    "fixture_evidence",
    "sync_observe",
    "observability",
    "preflight",
    "plan",
    "smoke",
    "final_benchmark",
    "job",
    "report_context",
    "resume_context",
    "workflow_goals",
    "evidence_buffer",
    "evidence_collection",
    "failure_recovery",
})
DURABLE_SECRET_CAPABLE_LEAF_KEYS = frozenset({
    "RPC_API_KEY",
    "api_key",
    "authorization",
    "password",
    "token",
})
# These values are generated by the Harness control plane and cannot carry
# user-authored payload. Their text may legitimately include a sensitive field
# name as part of a question or action identity.
DURABLE_SECRET_CONTROL_METADATA_LEAF_KEYS = frozenset({
    "_plan_scope",
    "atom_id",
    "draft_id",
    "idempotency_key",
    "owner_id",
    "plan_scope",
    "reference",
    "scope_id",
    "value_hash",
})


INVOCATION_CONTEXT_DEFAULTS: dict[str, Any] = {
    "discovery": {},
    "framework_summary": {},
    "web_research": {},
}


def project_checkpoint_state(state: Mapping[str, Any]) -> AgentGraphState:
    """Return the only state shape that may cross the checkpoint boundary."""

    projected: AgentGraphState = deepcopy(dict(state))
    for key, value in INVOCATION_CONTEXT_DEFAULTS.items():
        projected[key] = deepcopy(value)  # type: ignore[literal-required]
    return projected


RESET_PRESERVED_KEYS = (
    "job",
    "report_context",
)


DEFAULT_GROUP_ORDER = list(GROUP_ORDER)

STATE_SCHEMA_VERSION = 23


_V13_IMPLICIT_QUEUE_RESUME_QUESTION_IDS = frozenset({
    "inferred_config_review",
    "target_mode_change_confirm",
    "chain_change_confirm",
    "chain_ambiguity_confirm",
    "unknown_chain_identity_confirm",
})


class UnsupportedStateVersion(RuntimeError):
    pass
"""Compatibility view of the authoritative GroupSpec order."""


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_session_metadata(
    state: AgentGraphState,
    thread_id: str,
    session_purpose: str = "user",
    *,
    touch: bool = True,
) -> AgentGraphState:
    """Attach runtime session metadata without changing benchmark workflow state."""

    now = _utc_timestamp()
    session = dict(state.get("session") or {})
    session.setdefault("id", thread_id)
    session.setdefault("purpose", session_purpose or "user")
    session.setdefault("created_at", now)
    if touch or not session.get("updated_at"):
        session["updated_at"] = now
    state["session"] = session
    state["thread_id"] = thread_id
    return state


def migrate_state(
    raw_state: dict[str, Any],
    *,
    thread_id: str,
    language: str,
    session_purpose: str,
) -> AgentGraphState:
    """Upgrade one checkpoint through the explicit version boundary."""

    raw_version = int(raw_state.get("schema_version") or 1)
    if raw_version > STATE_SCHEMA_VERSION:
        raise UnsupportedStateVersion(
            f"checkpoint schema {raw_version} is newer than supported schema {STATE_SCHEMA_VERSION}"
        )
    if raw_version < 12:
        return _quarantine_pre_v12_state(
            raw_state,
            thread_id=thread_id,
            language=language,
            session_purpose=session_purpose,
            raw_version=raw_version,
        )
    if (
        raw_version == 20
        and _has_incomplete_v20_semantic_finalization(raw_state)
    ):
        return _quarantine_partial_semantic_finalization(
            raw_state,
            thread_id=thread_id,
            language=language,
            session_purpose=session_purpose,
            raw_version=raw_version,
        )
    if raw_version in {21, 22}:
        from .secret_refs import (
            raw_secret_paths_in_state,
            state_owned_secret_references,
        )

        if (
            raw_secret_paths_in_state(raw_state)
            or (
                raw_version == 21
                and state_owned_secret_references(raw_state)
            )
            or (
                raw_version == 22
                and (
                    state_owned_secret_references(raw_state)
                    or raw_state.get("secret_bindings")
                )
            )
        ):
            return _quarantine_unbound_secret_reference_state(
                raw_state,
                thread_id=thread_id,
                language=language,
                session_purpose=session_purpose,
                raw_version=raw_version,
            )
    fresh = new_state(thread_id, language=language, session_purpose=session_purpose)
    for key, value in raw_state.items():
        if key in fresh or key in AgentGraphState.__optional_keys__:
            fresh[key] = value  # type: ignore[literal-required]
    for key, value in INVOCATION_CONTEXT_DEFAULTS.items():
        fresh[key] = value.copy() if isinstance(value, dict) else value  # type: ignore[literal-required]
    if raw_version == 12:
        _migrate_v12_mode_exclusive_state(fresh)
    if raw_version <= 13:
        _migrate_v13_pending_queue_ownership(fresh)
    if raw_version <= 14:
        fresh["response_fragments"] = []
    if raw_version <= 15:
        _migrate_v15_pending_question_owner(fresh)
    if raw_version <= 16:
        fresh["semantic_planning"] = {}
    if raw_version <= 17:
        _migrate_v18_response_contract(fresh)
    if raw_version <= 18:
        _migrate_v19_semantic_draft_contract(fresh)
    if raw_version <= 19:
        _migrate_v20_product_head_bound_draft(fresh)
    if raw_version <= 20:
        _migrate_v21_semantic_evidence_contract(fresh)
    if raw_version <= 22:
        fresh["secret_bindings"] = []
    if raw_version < STATE_SCHEMA_VERSION:
        _quarantine_legacy_inflight_control_state(
            fresh,
            raw_version=raw_version,
        )
    _invalidate_draft_on_contract_authority_drift(fresh)
    # Version migrations first separate compatible durable work from state
    # owned by a retired mode. Only then may an invalid pending contract
    # quarantine the queue that still depends on it.
    _quarantine_unsupported_pending_actions(fresh)
    if raw_version < STATE_SCHEMA_VERSION:
        fresh.setdefault("audit_events", []).append({
            "event": "checkpoint_schema_migrated",
            "from_schema_version": raw_version,
            "to_schema_version": STATE_SCHEMA_VERSION,
            "in_flight_transition_retained": False,
        })
    fresh["schema_version"] = STATE_SCHEMA_VERSION
    return ensure_session_metadata(fresh, thread_id, session_purpose, touch=False)


def _quarantine_legacy_inflight_control_state(
    state: AgentGraphState,
    *,
    raw_version: int,
) -> None:
    """Retain durable configuration, never executable legacy control state."""

    quarantined = {
        "action_queue": len(state.get("action_queue") or ()),
        "selected_action": bool(state.get("selected_action")),
        "current_action": bool(state.get("current_action")),
        "pending_domain_result": bool(state.get("pending_domain_result")),
        "pending_question": bool(state.get("pending_question")),
        "semantic_plan_draft": bool(state.get("semantic_plan_draft")),
        "side_effect_intent": bool(state.get("side_effect_intent")),
        "side_effect_receipt": bool(state.get("side_effect_receipt")),
    }
    state["action_queue"] = []
    state["selected_action"] = {}
    state["current_action"] = {}
    state["pending_domain_result"] = {}
    state["pending_question"] = {}
    state["proposed_actions"] = []
    state["semantic_planning"] = {}
    state["semantic_plan_draft"] = {}
    state["side_effect_intent"] = {}
    state["side_effect_receipt"] = {}
    state["turn_receipt"] = {}
    state["secret_bindings"] = []
    control = dict(state.get("control") or {})
    control.pop("selected_owner", None)
    control.pop("phase", None)
    state["control"] = control
    state["checkpoint_recovery"] = {
        "status": "quarantined",
        "error_type": "LegacyInflightControlState",
        "from_schema_version": raw_version,
        "durable_configuration_retained": True,
    }
    state.setdefault("audit_events", []).append({
        "event": "checkpoint_legacy_inflight_quarantined",
        "from_schema_version": raw_version,
        "quarantined": quarantined,
    })


def _quarantine_unbound_secret_reference_state(
    raw_state: Mapping[str, Any],
    *,
    thread_id: str,
    language: str,
    session_purpose: str,
    raw_version: int,
) -> AgentGraphState:
    from .secret_refs import (
        raw_secret_paths_in_state,
        state_owned_secret_references,
    )

    state = new_state(
        thread_id,
        language=language,
        session_purpose=session_purpose,
    )
    for key in RESET_PRESERVED_KEYS:
        state[key] = deepcopy(raw_state.get(key) or {})  # type: ignore[literal-required]
    references = sorted(state_owned_secret_references(raw_state))
    raw_paths = sorted(raw_secret_paths_in_state(raw_state))
    state["checkpoint_recovery"] = {
        "status": "quarantined",
        "error_type": "UnsafeSecretCheckpoint",
        "from_schema_version": raw_version,
        "safe_confirmed_config": {},
        "requires_reconfirmation": [],
    }
    state["audit_events"] = [{
        "event": "checkpoint_unbound_secret_references_quarantined",
        "from_schema_version": raw_version,
        "to_schema_version": STATE_SCHEMA_VERSION,
        "reference_hashes": [
            hashlib.sha256(reference.encode("utf-8")).hexdigest()
            for reference in references
        ],
        "raw_path_hashes": [
            hashlib.sha256(path.encode("utf-8")).hexdigest()
            for path in raw_paths
        ],
    }]
    return state


def _has_incomplete_v20_semantic_finalization(
    state: Mapping[str, Any],
) -> bool:
    finalized: dict[tuple[str, str, str], set[str]] = {}
    applied: dict[tuple[str, str, str], set[str]] = {}
    saw_finalization_event = False
    malformed_event = False
    for item in state.get("audit_events") or ():
        if not isinstance(item, Mapping):
            continue
        event = str(item.get("event") or "")
        if event not in {
            "semantic_draft_finalized",
            "semantic_draft_finalization_action_applied",
        }:
            continue
        saw_finalization_event = True
        key = (
            str(item.get("draft_id") or ""),
            str(item.get("admission_transaction_hash") or ""),
            str(item.get("receipt_hash") or ""),
        )
        if not all(key):
            malformed_event = True
            continue
        if event == "semantic_draft_finalized":
            finalized[key] = {
                str(action_id)
                for action_id in item.get("final_action_ids") or ()
                if str(action_id)
            }
        else:
            action_id = str(item.get("action_id") or "")
            if not action_id:
                malformed_event = True
            else:
                applied.setdefault(key, set()).add(action_id)
    if not saw_finalization_event:
        return False
    if malformed_event or set(applied) - set(finalized):
        return True
    return any(
        not expected or not expected.issubset(applied.get(key, set()))
        for key, expected in finalized.items()
    )


def _quarantine_partial_semantic_finalization(
    raw_state: Mapping[str, Any],
    *,
    thread_id: str,
    language: str,
    session_purpose: str,
    raw_version: int,
) -> AgentGraphState:
    state = new_state(
        thread_id,
        language=language,
        session_purpose=session_purpose,
    )
    state["checkpoint_recovery"] = {
        "status": "quarantined",
        "error_type": "PartialSemanticFinalizationCheckpoint",
        "from_schema_version": raw_version,
        "safe_confirmed_config": {},
        "requires_reconfirmation": [],
    }
    state["audit_events"] = [{
        "event": "checkpoint_partial_semantic_finalization_quarantined",
        "from_schema_version": raw_version,
        "to_schema_version": STATE_SCHEMA_VERSION,
        "transactions": semantic_finalization_transaction_summaries(raw_state),
        "source_audit_hash": _semantic_finalization_audit_hash(raw_state),
    }]
    return state


def _semantic_finalization_audit_hash(state: Mapping[str, Any]) -> str:
    events = [
        dict(item)
        for item in state.get("audit_events") or ()
        if isinstance(item, Mapping)
        and str(item.get("event") or "") in {
            "semantic_draft_finalized",
            "semantic_draft_finalization_action_applied",
        }
    ]
    return hashlib.sha256(
        json.dumps(
            events,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def semantic_finalization_transaction_summaries(
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return non-secret audit identities for finalized semantic transactions."""

    finalized: dict[tuple[str, str, str], set[str]] = {}
    applied: dict[tuple[str, str, str], set[str]] = {}
    for item in state.get("audit_events") or ():
        if not isinstance(item, Mapping):
            continue
        event = str(item.get("event") or "")
        if event not in {
            "semantic_draft_finalized",
            "semantic_draft_finalization_action_applied",
        }:
            continue
        key = (
            str(item.get("draft_id") or ""),
            str(item.get("admission_transaction_hash") or ""),
            str(item.get("receipt_hash") or ""),
        )
        if event == "semantic_draft_finalized":
            finalized[key] = {
                str(action_id)
                for action_id in item.get("final_action_ids") or ()
                if str(action_id)
            }
        else:
            action_id = str(item.get("action_id") or "")
            if action_id:
                applied.setdefault(key, set()).add(action_id)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(finalized) | set(applied)):
        expected = finalized.get(key, set())
        observed = applied.get(key, set())
        rows.append({
            "draft_id": key[0],
            "admission_transaction_hash": key[1],
            "receipt_hash": key[2],
            "final_action_ids": sorted(expected),
            "applied_action_ids": sorted(observed),
            "missing_action_ids": sorted(expected - observed),
            "complete": bool(expected and expected <= observed),
        })
    return rows


def has_incomplete_semantic_finalization(state: Mapping[str, Any]) -> bool:
    """Return whether durable audit declares a transaction not fully applied."""

    rows = semantic_finalization_transaction_summaries(state)
    return any(not row["complete"] for row in rows)


def quarantine_inflight_semantic_finalization(
    raw_state: Mapping[str, Any],
    *,
    thread_id: str,
    language: str,
    session_purpose: str,
    error_type: str,
) -> AgentGraphState:
    """Discard partial business state while preserving a typed audit summary."""

    state = new_state(
        thread_id,
        language=language,
        session_purpose=session_purpose,
    )
    for key in RESET_PRESERVED_KEYS:
        state[key] = deepcopy(raw_state.get(key) or {})  # type: ignore[literal-required]
    state["checkpoint_recovery"] = {
        "status": "quarantined",
        "error_type": str(error_type),
        "from_schema_version": int(
            raw_state.get("schema_version") or STATE_SCHEMA_VERSION
        ),
        "safe_confirmed_config": {},
        "requires_reconfirmation": [],
    }
    state["audit_events"] = [{
        "event": "semantic_finalization_transaction_quarantined",
        "transactions": semantic_finalization_transaction_summaries(raw_state),
        "source_audit_hash": _semantic_finalization_audit_hash(raw_state),
    }]
    return state


def _migrate_v18_response_contract(state: AgentGraphState) -> None:
    """Discard turn-local text and manifests owned by the retired response schema."""

    state["response_fragments"] = []
    state["visible_response"] = []
    turn_context = dict(state.get("turn_context") or {})
    for key in (
        "response_manifest",
        "terminal_response_hash",
        "terminal_semantic_hash",
    ):
        turn_context.pop(key, None)
    state["turn_context"] = turn_context


def _migrate_v19_semantic_draft_contract(state: AgentGraphState) -> None:
    """Fail closed across the first durable semantic-draft boundary."""

    draft = dict(state.get("semantic_plan_draft") or {})
    if draft:
        state.setdefault("audit_events", []).append({
            "event": "semantic_draft_migration_invalidated",
            "draft_id": str(draft.get("draft_id") or ""),
            "draft_revision": int(draft.get("revision") or 0),
            "from_schema_version": 18,
            "to_schema_version": STATE_SCHEMA_VERSION,
        })
    state["semantic_planning"] = {}
    state["semantic_plan_draft"] = {}
    pending = dict(state.get("pending_question") or {})
    if pending.get("semantic_draft_binding"):
        state["pending_question"] = {}


def _migrate_v20_product_head_bound_draft(state: AgentGraphState) -> None:
    """Invalidate v19 drafts that lack Product Head and question authority."""

    draft = dict(state.get("semantic_plan_draft") or {})
    if draft:
        state.setdefault("audit_events", []).append({
            "event": "semantic_draft_migration_invalidated",
            "draft_id": str(draft.get("draft_id") or ""),
            "draft_revision": int(draft.get("revision") or 0),
            "from_schema_version": 19,
            "to_schema_version": STATE_SCHEMA_VERSION,
        })
    state["semantic_planning"] = {}
    state["semantic_plan_draft"] = {}
    pending = dict(state.get("pending_question") or {})
    if pending.get("semantic_draft_binding"):
        state["pending_question"] = {}


def _migrate_v21_semantic_evidence_contract(state: AgentGraphState) -> None:
    """Invalidate drafts that predate secret refs and full question binding."""

    from .secret_refs import discard_draft_secret_references

    draft = dict(state.get("semantic_plan_draft") or {})
    if draft:
        discard_draft_secret_references(draft)
        state.setdefault("audit_events", []).append({
            "event": "semantic_draft_migration_invalidated",
            "draft_id": str(draft.get("draft_id") or ""),
            "draft_revision": int(draft.get("revision") or 0),
            "from_schema_version": 20,
            "to_schema_version": STATE_SCHEMA_VERSION,
        })
    state["semantic_planning"] = {}
    state["semantic_plan_draft"] = {}
    pending = dict(state.get("pending_question") or {})
    if pending.get("semantic_draft_binding"):
        state["pending_question"] = {}
    semantic_action_types = {
        "resolve_semantic_draft_atom",
        "previous_semantic_draft_atom",
        "cancel_semantic_draft",
    }

    def is_incompatible_action(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return (
            str(value.get("action_type") or value.get("type") or "")
            in semantic_action_types
            or bool(
                (value.get("admission_metadata") or {}).get(
                    "semantic_draft_finalization_receipt"
                )
            )
            or bool(value.get("_semantic_draft_finalization_receipt"))
        )

    queue = list(state.get("action_queue") or [])
    retained_queue = [
        item for item in queue if not is_incompatible_action(item)
    ]
    removed_in_flight = len(retained_queue) != len(queue)
    state["action_queue"] = retained_queue
    state["proposed_actions"] = [
        item
        for item in state.get("proposed_actions") or []
        if not is_incompatible_action(item)
    ]
    audit_events = list(state.get("audit_events") or [])
    completed_audit = [
        {
            "event": "legacy_semantic_finalization_preserved",
            "from_schema_version": 20,
            **row,
        }
        for row in semantic_finalization_transaction_summaries(state)
        if row["complete"]
    ]
    retained_events = [
        item
        for item in audit_events
        if not (
            isinstance(item, Mapping)
            and str(item.get("event") or "") in {
                "semantic_draft_finalized",
                "semantic_draft_finalization_action_applied",
            }
        )
    ]
    retained_events.extend(completed_audit)
    if len(retained_events) != len(state.get("audit_events") or []):
        state["audit_events"] = retained_events
        removed_in_flight = True
    turn_context = dict(state.get("turn_context") or {})
    if turn_context.pop("semantic_draft_finalization_receipt", None):
        state["turn_context"] = turn_context
        removed_in_flight = True
    if is_incompatible_action(state.get("selected_action")):
        state["selected_action"] = {}
        removed_in_flight = True
    if is_incompatible_action(state.get("current_action")):
        state["current_action"] = {}
        removed_in_flight = True
    pending_result = dict(state.get("pending_domain_result") or {})
    pending_handler_result = pending_result.get("result")
    if (
        (
            isinstance(pending_handler_result, Mapping)
            and pending_handler_result.get("semantic_draft_command")
        )
        or is_incompatible_action(pending_result.get("action"))
    ):
        state["pending_domain_result"] = {}
        removed_in_flight = True
    if removed_in_flight:
        state["turn_receipt"] = {}
        control = dict(state.get("control") or {})
        control.pop("selected_owner", None)
        control.pop("phase", None)
        state["control"] = control


def _invalidate_draft_on_contract_authority_drift(
    state: AgentGraphState,
) -> None:
    """Fail closed when a persisted draft outlives a contract authority."""

    from .semantic_drafts import (
        mark_semantic_plan_draft_stale,
        semantic_draft_uses_current_authority,
    )

    draft = dict(state.get("semantic_plan_draft") or {})
    if (
        not draft
        or str(draft.get("status") or "") in {"stale", "cancelled"}
        or semantic_draft_uses_current_authority(draft)
    ):
        return
    state["semantic_plan_draft"] = mark_semantic_plan_draft_stale(
        draft,
        reasons=("contract_authority_changed_during_recovery",),
    )
    from .secret_refs import discard_draft_secret_references

    discard_draft_secret_references(draft)
    pending = dict(state.get("pending_question") or {})
    if pending.get("semantic_draft_binding"):
        state["pending_question"] = {}
    state.setdefault("audit_events", []).append({
        "event": "semantic_draft_authority_invalidated",
        "draft_id": str(draft.get("draft_id") or ""),
        "draft_revision": int(draft.get("revision") or 0),
    })


_PRE_V12_RECONFIRM_FIELDS = frozenset({
    "CLOUD_PROVIDER",
    "CLOUD_REGION",
    "CLOUD_ZONE",
    "MACHINE_TYPE",
    "CPU_CORES",
    "MEMORY_GIB",
    "LEDGER_DEVICE",
    "DATA_VOL_TYPE",
    "DATA_VOL_SIZE",
    "DATA_VOL_MAX_IOPS",
    "DATA_VOL_MAX_THROUGHPUT",
    "ACCOUNTS_DEVICE",
    "ACCOUNTS_VOL_TYPE",
    "ACCOUNTS_VOL_SIZE",
    "ACCOUNTS_VOL_MAX_IOPS",
    "ACCOUNTS_VOL_MAX_THROUGHPUT",
    "NETWORK_INTERFACE",
    "NETWORK_MAX_BANDWIDTH_GBPS",
})


def _quarantine_pre_v12_state(
    raw_state: Mapping[str, Any],
    *,
    thread_id: str,
    language: str,
    session_purpose: str,
    raw_version: int,
) -> AgentGraphState:
    """Retain only reviewable metadata from unsupported old checkpoints."""

    state = new_state(
        thread_id,
        language=language,
        session_purpose=session_purpose,
    )
    old_confirmed = dict(raw_state.get("confirmed_config") or {})
    safe_values = {
        field: deepcopy(value)
        for field, value in old_confirmed.items()
        if field in _PRE_V12_RECONFIRM_FIELDS and value not in (None, "")
    }
    if safe_values:
        state["inferred_config"] = {
            "pending_review": {
                "config_values": safe_values,
                "unmapped_values": {},
                "source_format": "checkpoint_migration",
            }
        }
    state["checkpoint_recovery"] = {
        "status": "quarantined",
        "error_type": "UnsupportedLegacyCheckpoint",
        "from_schema_version": raw_version,
        "safe_confirmed_config": safe_values,
        "requires_reconfirmation": sorted(safe_values),
    }
    state["audit_events"] = [{
        "event": "checkpoint_legacy_quarantined",
        "from_schema_version": raw_version,
        "to_schema_version": STATE_SCHEMA_VERSION,
        "retained_fields": sorted(safe_values),
    }]
    return state


def _migrate_v13_pending_queue_ownership(state: AgentGraphState) -> None:
    """Materialize v13's implicit queue policy into the typed question contract.

    Schema v13 let a coordinator-side question-ID allowlist retain deferred
    actions even when the persisted question omitted ``resume_action_queue``.
    Schema v14 retires that runtime authority. This migration preserves the
    old checkpoint's effective behavior once, at the checkpoint boundary.
    """

    pending = state.get("pending_question")
    queue = state.get("action_queue") or []
    if (
        not isinstance(pending, dict)
        or not pending
        or not queue
        or str(pending.get("id") or "") not in _V13_IMPLICIT_QUEUE_RESUME_QUESTION_IDS
    ):
        return
    pending["resume_action_queue"] = True
    state.setdefault("audit_events", []).append({
        "event": "checkpoint_v13_queue_ownership_migrated",
        "question_id": str(pending.get("id") or ""),
        "retained_action_count": len(queue),
        "ownership": "pending_question.resume_action_queue",
    })


def _migrate_v15_pending_question_owner(state: AgentGraphState) -> None:
    """Materialize the retired implicit group-to-owner relation once."""

    from .domains.registry import GROUP_OWNER

    migrated: list[tuple[str, str]] = []
    custom_rpc_questions = {
        "custom_rpc_adapter_family_confirm",
        "custom_rpc_endpoint",
        "custom_rpc_method",
        "custom_rpc_schema_evidence",
        "custom_rpc_schema_confirm",
        "custom_rpc_parameter_confirm",
        "custom_rpc_response_confirm",
        "custom_rpc_probe_confirm",
        "custom_rpc_continue",
        "custom_rpc_scope",
        "custom_rpc_single_method",
        "custom_rpc_weights",
        "custom_rpc_fixture_choice",
    }
    new_chain_questions = {
        "new_chain_endpoint",
        "new_chain_method",
        "new_chain_schema_evidence",
        "new_chain_schema_confirm",
        "new_chain_parameter_confirm",
        "new_chain_response_confirm",
        "new_chain_probe_confirm",
        "new_chain_method_continue",
        "new_chain_workload_scope",
        "new_chain_single_method",
        "new_chain_custom_weights",
        "new_chain_runtime_choice",
    }

    def migrate_question(question: dict[str, Any], location: str) -> None:
        question_id = str(question.get("id") or "")
        group = str(question.get("group") or "")
        if not str(question.get("owner") or ""):
            owner = (
                "environment"
                if question_id == "inferred_config_review"
                else GROUP_OWNER.get(group, "")
            )
            if owner:
                question["owner"] = owner
                migrated.append((location, question_id))
        if not question.get("domain_context"):
            rpc_case = (
                "custom_rpc"
                if question_id in custom_rpc_questions
                else "new_chain"
                if question_id in new_chain_questions
                else ""
            )
            question["domain_context"] = (
                {"rpc_case": rpc_case} if rpc_case else {}
            )

    pending = state.get("pending_question")
    if isinstance(pending, dict) and pending:
        migrate_question(pending, "top_level")
    resume_context = state.get("resume_context")
    if isinstance(resume_context, dict):
        resumed = resume_context.get("pending_question")
        if isinstance(resumed, dict) and resumed:
            migrate_question(resumed, "resume_context")
    for location, question_id in migrated:
        state.setdefault("audit_events", []).append({
            "event": "checkpoint_v15_pending_owner_migrated",
            "location": location,
            "question_id": question_id,
        })


def _quarantine_unsupported_pending_actions(state: AgentGraphState) -> None:
    """Drop persisted questions that fail the current complete contract."""

    from .questions import validate_pending_question_contract

    candidates: list[tuple[str, dict[str, Any], str]] = []
    pending = state.get("pending_question")
    if isinstance(pending, dict) and pending:
        candidates.append(("pending_question", pending, "top_level"))
    resume_context = state.get("resume_context")
    if isinstance(resume_context, dict):
        resumed = resume_context.get("pending_question")
        if isinstance(resumed, dict) and resumed:
            candidates.append(("pending_question", resumed, "resume_context"))
    for key, candidate, location in candidates:
        try:
            validate_pending_question_contract(candidate)
        except (TypeError, ValueError) as exc:
            contract_error = str(exc)
        else:
            continue
        if location == "top_level":
            state[key] = {}
            quarantined_queue = len(state.get("action_queue") or [])
            state["action_queue"] = []
            state["selected_action"] = {}
            state["checkpoint_recovery"] = {
                "status": "quarantined",
                "error_type": "PendingQuestionContractError",
                "safe_confirmed_config": dict(
                    state.get("confirmed_config") or {}
                ),
            }
        else:
            updated = dict(state.get("resume_context") or {})
            updated[key] = {}
            state["resume_context"] = updated
        state.setdefault("audit_events", []).append({
            "event": "checkpoint_pending_actions_quarantined",
            "location": location,
            "question_id": str(candidate.get("id") or ""),
            "contract_error": contract_error,
            **(
                {"quarantined_action_count": quarantined_queue}
                if location == "top_level"
                else {}
            ),
        })


def _migrate_v12_mode_exclusive_state(state: AgentGraphState) -> None:
    """Remove v12 RPC-benchmark state from a sync-observe checkpoint."""

    if str(state.get("workflow_mode") or "") != "sync_observe":
        return
    incompatible_groups = {
        "workload_rpc",
        "target_samples_fixtures",
        "qps_profile",
    }
    changed = any((
        state.get("rpc_mode"),
        state.get("workload"),
        state.get("custom_rpc"),
        state.get("fixture_evidence"),
        state.get("qps_profile"),
    ))
    state["rpc_mode"] = ""
    state["workload"] = {}
    state["custom_rpc"] = {}
    state["fixture_evidence"] = {}
    state["qps_profile"] = {}
    queue = list(state.get("action_queue") or [])
    incompatible_actions = {
        "set_rpc_mode",
        "set_qps_mode",
        "set_qps_override",
        "rpc_catalog_command",
        "rpc_workload_command",
        "use_default_workload",
        "configure_workload_weights",
    }
    filtered = [
        item
        for item in queue
        if str((item or {}).get("action_type") or "") not in incompatible_actions
    ]
    changed = changed or len(filtered) != len(queue)
    state["action_queue"] = filtered
    pending_group = str((state.get("pending_question") or {}).get("group") or "")
    active_group = str(state.get("active_group") or "")
    if pending_group in incompatible_groups or active_group in incompatible_groups:
        state["pending_question"] = {}
        state["active_group"] = "opening"
        changed = True
    if changed:
        state.setdefault("audit_events", []).append({
            "event": "checkpoint_v12_mode_exclusivity_migrated",
            "workflow_mode": "sync_observe",
        })


def new_state(thread_id: str, language: str = "en", session_purpose: str = "user") -> AgentGraphState:
    now = _utc_timestamp()
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "thread_id": thread_id,
        "session": {
            "id": thread_id,
            "purpose": session_purpose or "user",
            "created_at": now,
            "updated_at": now,
        },
        "language": language,
        "last_user_input": "",
        "turn_index": 0,
        "turn_context": {},
        "control": {},
        "semantic_planning": {},
        "semantic_plan_draft": {},
        "secret_bindings": [],
        "proposed_actions": [],
        "input_shape": "",
        "active_group": "opening",
        "discovery": {},
        "framework_summary": {},
        "web_research": {},
        "pending_question": {},
        "action_queue": [],
        "selected_action": {},
        "pending_domain_result": {},
        "side_effect_intent": {},
        "side_effect_receipt": {},
        "turn_receipt": {},
        "current_action": {},
        "completed_actions": [],
        "applied_action_ids": [],
        "action_errors": [],
        "group_states": {},
        "group_history": [],
        "confirmed_config": {},
        "inferred_config": {},
        "invalidated_groups": [],
        "interruption_stack": [],
        "chain_identity": {},
        "target_mode": "",
        "target_mode_change_candidate": "",
        "workflow_mode": "",
        "rpc_mode": "",
        "workload": {},
        "custom_rpc": {},
        "qps_profile": {},
        "endpoint_evidence": {},
        "fixture_evidence": {},
        "sync_observe": {},
        "observability": {},
        "advanced_tuning": {},
        "secondary_handoff": {},
        "preflight": {},
        "plan": {},
        "plan_file": "",
        "smoke": {},
        "final_benchmark": {},
        "job": {},
        "report_context": {},
        "failure_recovery": {},
        "evidence_buffer": [],
        "evidence_collection": {},
        "response_fragments": [],
        "visible_response": [],
        "audit_events": [],
        "checkpoint_recovery": {},
        "resume_context": {},
        "workflow_goals": [],
    }
