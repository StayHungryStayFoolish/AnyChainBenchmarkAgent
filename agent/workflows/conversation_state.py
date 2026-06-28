"""File-backed workflow state for the ADK product Agent.

This module stores structured conversation facts. It intentionally does not
parse natural language, classify intent, or route business workflows. ADK and
the configured model infer intent and entities, then update this state through
typed patches. Validators decide whether execution can continue.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_SESSION_ID = "terminal-session"
DEFAULT_STATE_ROOT = Path(".agent/sessions")

ALLOWED_PATCH_KEYS = {
    "language",
    "active_intent",
    "active_workflow",
    "workflow_step",
    "target_mode",
    "chain",
    "rpc_mode",
    "rpc_methods",
    "mixed_weights",
    "custom_rpc",
    "confirmed_config",
    "inferred_config",
    "missing_fields",
    "pending_question",
    "last_user_change",
    "latest_plan_file",
    "latest_job_id",
    "history_summary",
}


def default_workflow_state(session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
    now = _now()
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "language": "en",
        "active_intent": "",
        "active_workflow": "",
        "workflow_step": "",
        "target_mode": "",
        "chain": "",
        "rpc_mode": "",
        "rpc_methods": [],
        "mixed_weights": {},
        "custom_rpc": [],
        "confirmed_config": {},
        "inferred_config": {},
        "missing_fields": [],
        "pending_question": {},
        "last_user_change": "",
        "latest_plan_file": "",
        "latest_job_id": "",
        "history_summary": "",
        "history": [],
        "revision": 0,
        "created_at": now,
        "updated_at": now,
    }


def workflow_state_path(
    session_id: str = DEFAULT_SESSION_ID,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> Path:
    safe_session = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)
    return Path(state_root) / safe_session / "conversation_state.json"


def load_workflow_state(
    session_id: str = DEFAULT_SESSION_ID,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    path = workflow_state_path(session_id=session_id, state_root=state_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_workflow_state(session_id=session_id)
    return _normalize_state(payload, session_id=session_id)


def save_workflow_state(
    state: dict[str, Any],
    session_id: str | None = None,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> Path:
    normalized = _normalize_state(state, session_id=session_id or state.get("session_id") or DEFAULT_SESSION_ID)
    path = workflow_state_path(session_id=normalized["session_id"], state_root=state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    return path


def update_workflow_state(
    patch: dict[str, Any],
    *,
    reason: str = "",
    session_id: str = DEFAULT_SESSION_ID,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    """Apply a structured patch and persist state.

    The patch must be explicit fields inferred by ADK. Unknown keys are ignored
    and returned as warnings so they cannot silently become runtime behavior.
    """
    current = load_workflow_state(session_id=session_id, state_root=state_root)
    clean_patch = {key: deepcopy(value) for key, value in (patch or {}).items() if key in ALLOWED_PATCH_KEYS}
    ignored = sorted(set((patch or {}).keys()) - ALLOWED_PATCH_KEYS)
    _push_history(current)

    for key, value in clean_patch.items():
        if key in {"confirmed_config", "inferred_config", "mixed_weights"}:
            current[key] = _merge_dict(current.get(key), value)
        elif key in {"rpc_methods", "custom_rpc", "missing_fields"}:
            current[key] = list(value or [])
        elif key == "pending_question":
            current[key] = dict(value or {})
        else:
            current[key] = value if value is not None else ""

    current["revision"] = int(current.get("revision") or 0) + 1
    current["updated_at"] = _now()
    if reason:
        current["last_update_reason"] = reason
    path = save_workflow_state(current, session_id=session_id, state_root=state_root)
    return {
        "state": current,
        "state_file": str(path),
        "ignored_keys": ignored,
    }


def revert_workflow_state(
    steps: int = 1,
    *,
    reason: str = "",
    session_id: str = DEFAULT_SESSION_ID,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    """Restore an earlier structured state snapshot.

    This supports user corrections such as "go back" or "I gave the wrong
    disk". ADK should still explain what changed and re-run validators before
    execution.
    """
    current = load_workflow_state(session_id=session_id, state_root=state_root)
    history = list(current.get("history") or [])
    count = max(1, int(steps or 1))
    if not history:
        path = save_workflow_state(current, session_id=session_id, state_root=state_root)
        return {
            "state": current,
            "state_file": str(path),
            "reverted": False,
            "message": "no previous workflow state snapshot",
        }
    selected = history[-count] if len(history) >= count else history[0]
    restored = _normalize_state(selected, session_id=session_id)
    restored["history"] = history[: max(0, len(history) - count)]
    restored["revision"] = int(current.get("revision") or 0) + 1
    restored["updated_at"] = _now()
    if reason:
        restored["last_update_reason"] = reason
    path = save_workflow_state(restored, session_id=session_id, state_root=state_root)
    return {
        "state": restored,
        "state_file": str(path),
        "reverted": True,
        "message": f"reverted {min(count, len(history))} workflow state snapshot(s)",
    }


def reset_workflow_state(
    *,
    reason: str = "",
    session_id: str = DEFAULT_SESSION_ID,
    state_root: str | Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    state = default_workflow_state(session_id=session_id)
    if reason:
        state["last_update_reason"] = reason
    path = save_workflow_state(state, session_id=session_id, state_root=state_root)
    return {
        "state": state,
        "state_file": str(path),
    }


def _normalize_state(payload: dict[str, Any], session_id: str) -> dict[str, Any]:
    state = default_workflow_state(session_id=session_id)
    for key in state:
        if key in payload:
            state[key] = deepcopy(payload[key])
    state["schema_version"] = SCHEMA_VERSION
    state["session_id"] = str(payload.get("session_id") or session_id or DEFAULT_SESSION_ID)
    state["revision"] = int(state.get("revision") or 0)
    state["confirmed_config"] = dict(state.get("confirmed_config") or {})
    state["inferred_config"] = dict(state.get("inferred_config") or {})
    state["mixed_weights"] = dict(state.get("mixed_weights") or {})
    state["pending_question"] = dict(state.get("pending_question") or {})
    state["rpc_methods"] = list(state.get("rpc_methods") or [])
    state["custom_rpc"] = list(state.get("custom_rpc") or [])
    state["missing_fields"] = list(state.get("missing_fields") or [])
    state["history"] = list(state.get("history") or [])
    return state


def _merge_dict(existing: Any, incoming: Any) -> dict[str, Any]:
    merged = dict(existing or {})
    for key, value in dict(incoming or {}).items():
        if value in {None, ""}:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _push_history(state: dict[str, Any]) -> None:
    snapshot = deepcopy(state)
    snapshot.pop("history", None)
    history = list(state.get("history") or [])
    history.append(snapshot)
    state["history"] = history[-20:]
