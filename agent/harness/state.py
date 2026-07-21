"""Typed graph state for the AnyChain Agent Harness."""

from __future__ import annotations

import time
from pathlib import Path
from copy import deepcopy
from typing import Any, Literal, Mapping, TypedDict

from agent.workflows.group_registry import GROUP_ORDER
WorkflowMode = Literal["rpc_benchmark", "sync_observe", ""]
TargetMode = Literal["fake-node", "real-node", "sync-observe", ""]


class PendingQuestion(TypedDict, total=False):
    contract_version: int
    id: str
    group: str
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
    structured_input_owner: bool
    accepted_action_types: list[str]
    manual_action: dict[str, Any]
    requires_capabilities: list[str]
    resume_action_queue: bool
    supersedes_action_types: list[str]
    created_turn_index: int
    queue_barrier: bool


class AgentGraphState(TypedDict, total=False):
    schema_version: int
    thread_id: str
    session: dict[str, Any]
    language: str
    last_user_input: str
    turn_index: int
    turn_context: dict[str, Any]
    control: dict[str, Any]
    proposed_actions: list[dict[str, Any]]
    input_shape: str
    active_intent: str
    active_group: str
    active_subgroup: str
    discovery: dict[str, Any]
    framework_summary: dict[str, Any]
    web_research: dict[str, Any]
    pending_question: PendingQuestion
    action_queue: list[dict[str, Any]]
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
    visible_response: list[str]
    audit_events: list[dict[str, Any]]
    checkpoint_recovery: dict[str, Any]
    resume_context: dict[str, Any]
    workflow_goals: list[dict[str, Any]]


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

STATE_SCHEMA_VERSION = 9


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
    """Upgrade a persisted checkpoint without inventing workflow progress."""

    raw_version = int(raw_state.get("schema_version") or 1)
    if raw_version > STATE_SCHEMA_VERSION:
        raise UnsupportedStateVersion(
            f"checkpoint schema {raw_version} is newer than supported schema {STATE_SCHEMA_VERSION}"
        )
    fresh = new_state(thread_id, language=language, session_purpose=session_purpose)
    for key, value in raw_state.items():
        if key in fresh or key in AgentGraphState.__optional_keys__:
            fresh[key] = value  # type: ignore[literal-required]
    for key, value in INVOCATION_CONTEXT_DEFAULTS.items():
        fresh[key] = value.copy() if isinstance(value, dict) else value  # type: ignore[literal-required]
    _recover_legacy_prepared_plan(raw_state, fresh)
    from .domains.rpc_catalog import migrate_legacy_catalog

    migrate_legacy_catalog(fresh)
    _migrate_legacy_action_queue(fresh)
    fresh["schema_version"] = STATE_SCHEMA_VERSION
    _normalize_mode_exclusive_state(fresh)
    _normalize_pending_question_contract(fresh)
    return ensure_session_metadata(fresh, thread_id, session_purpose, touch=False)


def _migrate_legacy_action_queue(state: AgentGraphState) -> None:
    """Compile retired durable actions once while loading old checkpoints."""

    from .action_registry import compile_legacy_custom_rpc_action

    queue = list(state.get("action_queue") or [])
    migrated = [
        compiled
        for item in queue
        if isinstance(item, dict)
        for compiled in compile_legacy_custom_rpc_action(item)
    ]
    if migrated == queue:
        return
    state["action_queue"] = migrated
    state.setdefault("audit_events", []).append({
        "event": "legacy_custom_rpc_actions_migrated",
        "before": len(queue),
        "after": len(migrated),
    })


def _recover_legacy_prepared_plan(raw_state: dict[str, Any], state: AgentGraphState) -> None:
    """Recover the immutable final plan omitted by checkpoint schemas before v6."""

    if state.get("plan_file"):
        return
    smoke_result = ((raw_state.get("smoke") or {}).get("result") or {}).get("data") or {}
    source = str(smoke_result.get("source_plan_file") or "").strip()
    if not source:
        smoke_file = Path(str(smoke_result.get("smoke_plan_file") or ""))
        suffix = "_real_node_smoke.json"
        if smoke_file.name.endswith(suffix):
            source = str(Path(__file__).resolve().parents[2] / ".agent" / "prepared" / f"{smoke_file.name[:-len(suffix)]}.json")
    source_path = Path(source) if source else Path()
    if not source or not source_path.is_file():
        return
    state["plan_file"] = str(source_path)
    try:
        import json

        state["plan"] = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state["plan"] = {}
    state.setdefault("audit_events", []).append({"event": "legacy_prepared_plan_recovered"})


def _normalize_pending_question_contract(state: AgentGraphState) -> None:
    """Upgrade persisted generic manual questions to the current type contract."""

    pending = state.get("pending_question") or {}
    if str(pending.get("kind") or "") != "manual_value" or pending.get("validation"):
        return
    pending["validation"] = {"value_type": "scalar_token", "max_length": 180}
    state["pending_question"] = pending
    state.setdefault("audit_events", []).append({
        "event": "checkpoint_question_contract_upgraded",
        "question_id": str(pending.get("id") or ""),
    })


def _normalize_mode_exclusive_state(state: AgentGraphState) -> None:
    """Remove checkpoint values that cannot exist in the selected workflow."""

    if str(state.get("workflow_mode") or "") != "sync_observe":
        return
    incompatible_groups = {"workload_rpc", "target_samples_fixtures", "qps_profile"}
    changed = any((state.get("rpc_mode"), state.get("workload"), state.get("custom_rpc"), state.get("fixture_evidence"), state.get("qps_profile")))
    state["rpc_mode"] = ""
    state["workload"] = {}
    state["custom_rpc"] = {}
    state["fixture_evidence"] = {}
    state["qps_profile"] = {}
    queue = list(state.get("action_queue") or [])
    filtered = [
        item for item in queue
        if str((item or {}).get("type") or "") not in {
            "set_rpc_mode", "set_qps_mode", "set_qps_override", "start_custom_rpc",
            "rpc_catalog_command", "rpc_workload_command",
            "use_default_workload", "configure_workload_weights",
        }
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
        state.setdefault("audit_events", []).append({"event": "checkpoint_mode_exclusivity_repaired", "workflow_mode": "sync_observe"})


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
        "proposed_actions": [],
        "input_shape": "",
        "active_intent": "",
        "active_group": "opening",
        "active_subgroup": "",
        "discovery": {},
        "framework_summary": {},
        "web_research": {},
        "pending_question": {},
        "action_queue": [],
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
        "visible_response": [],
        "audit_events": [],
        "checkpoint_recovery": {},
        "resume_context": {},
        "workflow_goals": [],
    }
