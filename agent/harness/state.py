"""Typed graph state for the AnyChain Agent Harness."""

from __future__ import annotations

import hashlib
import time
from copy import deepcopy
from typing import Any, Literal, Mapping, TypedDict

from agent.workflows.group_registry import GROUP_ORDER
WorkflowMode = Literal["rpc_benchmark", "sync_observe", ""]
TargetMode = Literal["fake-node", "real-node", "sync-observe", ""]


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
    created_turn_index: int
    queue_barrier: bool
    barrier_policy: str


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

STATE_SCHEMA_VERSION = 18


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
    fresh = new_state(thread_id, language=language, session_purpose=session_purpose)
    for key, value in raw_state.items():
        if key in fresh or key in AgentGraphState.__optional_keys__:
            fresh[key] = value  # type: ignore[literal-required]
    for key, value in INVOCATION_CONTEXT_DEFAULTS.items():
        fresh[key] = value.copy() if isinstance(value, dict) else value  # type: ignore[literal-required]
    if raw_version == 12:
        _migrate_v12_action_queue_contract(fresh)
        _migrate_v12_mode_exclusive_state(fresh)
        fresh["selected_action"] = {}
        fresh["pending_domain_result"] = {}
        fresh["side_effect_intent"] = {}
        fresh["side_effect_receipt"] = {}
        fresh["turn_receipt"] = {}
        control = dict(fresh.get("control") or {})
        control.pop("selected_owner", None)
        control.pop("phase", None)
        fresh["control"] = control
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


def _migrate_v12_action_queue_contract(state: AgentGraphState) -> None:
    """Compile every pre-v12 raw queue into admitted ActionEnvelopes once."""

    from .action_registry import (
        ACTION_BY_TYPE,
        assign_action_ids,
    )
    from .checkpoint_migrations import compile_v12_custom_rpc_action
    from .contracts import ActionEnvelope, action_envelope_to_dict
    from .domains.registry import GROUP_OWNER

    queue = list(state.get("action_queue") or [])
    if not queue or all(
        isinstance(item, dict)
        and str(item.get("action_type") or "")
        and str(item.get("owner") or "")
        and isinstance(item.get("arguments"), Mapping)
        for item in queue
    ):
        return
    migrated: list[dict[str, Any]] = []
    rejected = 0
    for item in queue:
        if not isinstance(item, dict):
            rejected += 1
            continue
        if (
            str(item.get("action_type") or "")
            and str(item.get("owner") or "")
            and isinstance(item.get("arguments"), Mapping)
        ):
            migrated.append(item)
            continue
        action_type = str(item.get("type") or "")
        if action_type != "start_custom_rpc":
            if action_type in ACTION_BY_TYPE:
                migrated.append(item)
                continue
            rejected += 1
            continue
        source = str(item.get("source_evidence") or "").strip()
        exact_values = [
            str(item.get(key) or "").strip()
            for key in ("rpc_endpoint", "rpc_method", "rpc_schema_evidence")
            if item.get(key) not in (None, "")
        ]
        if not source or any(value not in source for value in exact_values):
            rejected += 1
            continue
        migrated.extend(compile_v12_custom_rpc_action(item))
    if any(not str(item.get("action_id") or "") for item in migrated):
        migration_scope = (
            f"checkpoint:{state.get('thread_id') or 'default'}:"
            f"{int(state.get('turn_index') or 0)}"
        )
        migrated = assign_action_ids(
            migration_scope,
            "checkpoint action queue migration",
            migrated,
        )
    migration_scope = (
        f"checkpoint:{state.get('thread_id') or 'default'}:"
        f"{int(state.get('turn_index') or 0)}"
    )
    enveloped: list[dict[str, Any]] = []
    pending = state.get("pending_question") or {}
    pending_group = str(pending.get("group") or "")
    pending_owner = str(pending.get("owner") or GROUP_OWNER.get(pending_group, ""))
    for index, action in enumerate(migrated):
        if str(action.get("action_type") or ""):
            enveloped.append(action)
            continue
        action_type = str(action.get("type") or "")
        spec = ACTION_BY_TYPE[action_type]
        owner = (
            pending_owner or spec.owner
            if action_type == "answer_pending"
            else spec.owner
        )
        origin_text = str(
            action.get("_origin_text")
            or action.get("source_evidence")
            or ""
        )
        arguments = {
            str(key): deepcopy(value)
            for key, value in action.items()
            if key not in {"type", "action_id", "confidence", "reason"}
            and not str(key).startswith("_")
        }
        effect_kind = (
            "external"
            if spec.effect == "execution"
            else "read_only"
            if spec.effect == "read_only"
            else "pure"
        )
        enveloped.append(action_envelope_to_dict(ActionEnvelope(
            action_id=str(action.get("action_id") or ""),
            action_type=action_type,
            owner=owner,
            target_group=spec.target_group,
            arguments=arguments,
            confidence=str(action.get("confidence") or "medium"),  # type: ignore[arg-type]
            reason=str(action.get("reason") or ""),
            source_evidence_hash=hashlib.sha256(origin_text.encode("utf-8")).hexdigest(),
            semantic_order=int(action.get("_plan_index") or index),
            submitted_turn_index=int(
                action.get("_submitted_turn_index")
                or state.get("turn_index")
                or 0
            ),
            origin_group=str(action.get("_queue_origin_group") or ""),
            origin_text=origin_text,
            plan_scope=str(action.get("_plan_scope") or migration_scope),
            admission_metadata={
                str(key).removeprefix("_"): deepcopy(value)
                for key, value in action.items()
                if str(key).startswith("_")
                and str(key) not in {
                    "_origin_text",
                    "_queue_origin_group",
                    "_submitted_turn_index",
                    "_plan_scope",
                    "_plan_index",
                }
            },
            effect_kind=effect_kind,  # type: ignore[arg-type]
            idempotency_key=(
                f"{state.get('thread_id') or 'default'}:"
                f"{int(state.get('turn_index') or 0)}:"
                f"{str(action.get('action_id') or '')}"
            ),
        )))
    state["action_queue"] = enveloped
    if enveloped and state.get("pending_question"):
        state["pending_question"]["resume_action_queue"] = True
    state.setdefault("audit_events", []).append({
        "event": "checkpoint_action_queue_enveloped",
        "before": len(queue),
        "after": len(enveloped),
        "rejected_untrusted": rejected,
    })


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
