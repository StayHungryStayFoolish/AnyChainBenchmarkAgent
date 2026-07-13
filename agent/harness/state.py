"""Typed graph state for the AnyChain Agent Harness."""

from __future__ import annotations

import time
from typing import Any, Literal, TypedDict


WorkflowMode = Literal["rpc_benchmark", "sync_observe", ""]
TargetMode = Literal["fake-node", "real-node", "sync-observe", ""]


class PendingQuestion(TypedDict, total=False):
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


class AgentGraphState(TypedDict, total=False):
    thread_id: str
    session: dict[str, Any]
    language: str
    last_user_input: str
    input_shape: str
    active_intent: str
    active_group: str
    active_subgroup: str
    discovery: dict[str, Any]
    framework_summary: dict[str, Any]
    web_research: dict[str, Any]
    latest_job_id: str
    pending_question: PendingQuestion
    action_queue: list[dict[str, Any]]
    current_action: dict[str, Any]
    completed_actions: list[dict[str, Any]]
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
    smoke: dict[str, Any]
    job: dict[str, Any]
    report_context: dict[str, Any]
    evidence_buffer: list[dict[str, Any]]
    evidence_collection: dict[str, Any]
    visible_response: list[str]
    audit_events: list[dict[str, Any]]


RESET_PRESERVED_KEYS = (
    "discovery",
    "framework_summary",
    "web_research",
    "latest_job_id",
    "job",
    "report_context",
)


DEFAULT_GROUP_ORDER = [
    "opening",
    "target_mode",
    "chain_identity",
    "provider_deployment",
    "ledger_disk",
    "accounts_disk",
    "network",
    "endpoint_process",
    "chain_auxiliary_endpoints",
    "workload_rpc",
    "target_samples_fixtures",
    "qps_profile",
    "sync_observe",
    "observability",
    "advanced_tuning",
    "preflight_smoke_execution",
    "job_monitoring",
    "error_evidence_analysis",
    "report_artifact_analysis",
]
"""Canonical group order/membership for the AnyChain Agent Harness.

This is the single source of truth (architecture audit Finding A).
`agent/harness/intent.py`'s `ALLOWED_GROUPS` imports this list directly
instead of retyping it, and `agent/workflows/group_registry.py` must keep
its `GROUP_ORDER` consistent with this list. `hardware_discovery` was
removed: it was declared here but never implemented in `groups.py` and had
no product specification anywhere in the docs (unlike `advanced_tuning` and
`chain_auxiliary_endpoints`, which are documented in
`docs/en/anychain-agent-ai-work-gate.md` and are now implemented)."""


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_session_metadata(state: AgentGraphState, thread_id: str, session_purpose: str = "user") -> AgentGraphState:
    """Attach runtime session metadata without changing benchmark workflow state."""

    now = _utc_timestamp()
    session = dict(state.get("session") or {})
    session.setdefault("id", thread_id)
    session.setdefault("purpose", session_purpose or "user")
    session.setdefault("created_at", now)
    session["updated_at"] = now
    state["session"] = session
    state["thread_id"] = thread_id
    return state


def new_state(thread_id: str, language: str = "en", session_purpose: str = "user") -> AgentGraphState:
    now = _utc_timestamp()
    return {
        "thread_id": thread_id,
        "session": {
            "id": thread_id,
            "purpose": session_purpose or "user",
            "created_at": now,
            "updated_at": now,
        },
        "language": language,
        "last_user_input": "",
        "input_shape": "",
        "active_intent": "",
        "active_group": "opening",
        "active_subgroup": "",
        "discovery": {},
        "framework_summary": {},
        "web_research": {},
        "latest_job_id": "",
        "pending_question": {},
        "action_queue": [],
        "current_action": {},
        "completed_actions": [],
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
        "smoke": {},
        "job": {},
        "report_context": {},
        "evidence_buffer": [],
        "evidence_collection": {},
        "visible_response": [],
        "audit_events": [],
    }
