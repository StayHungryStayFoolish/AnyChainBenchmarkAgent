"""ADK tools for structured workflow state.

These tools give ADK a persistent place to store confirmed facts and workflow
progress. They do not classify natural language or route business intent.
"""

from __future__ import annotations

from typing import Any

from workflows.conversation_state import (
    load_workflow_state as _load_workflow_state,
    revert_workflow_state as _revert_workflow_state,
    reset_workflow_state as _reset_workflow_state,
    update_workflow_state as _update_workflow_state,
)

from .read_only import _tool_result


def load_workflow_state(session_id: str = "terminal-session") -> dict[str, Any]:
    """Load the current structured conversation/workflow state."""
    return _tool_result(
        data=_load_workflow_state(session_id=session_id or "terminal-session"),
        next_actions=["continue current workflow", "ask pending question", "validate confirmed config"],
    )


def update_workflow_state(
    patch: dict | None = None,
    reason: str = "",
    session_id: str = "terminal-session",
) -> dict[str, Any]:
    """Persist structured facts inferred by ADK from the conversation.

    Pass explicit fields only. Do not pass raw natural language as a parsing
    shortcut.
    """
    payload = _update_workflow_state(
        patch or {},
        reason=reason,
        session_id=session_id or "terminal-session",
    )
    return _tool_result(
        data=payload,
        evidence_paths=[payload.get("state_file", "")],
        warnings=[f"ignored unsupported state key: {key}" for key in payload.get("ignored_keys", [])],
        next_actions=["validate current workflow step", "ask one blocking question", "prepare benchmark run"],
    )


def reset_workflow_state(reason: str = "", session_id: str = "terminal-session") -> dict[str, Any]:
    """Reset structured workflow state when the user explicitly starts over."""
    payload = _reset_workflow_state(reason=reason, session_id=session_id or "terminal-session")
    return _tool_result(
        data=payload,
        evidence_paths=[payload.get("state_file", "")],
        next_actions=["start a new workflow"],
    )


def revert_workflow_state(
    steps: int = 1,
    reason: str = "",
    session_id: str = "terminal-session",
) -> dict[str, Any]:
    """Revert workflow state when the user asks to go back or correct prior input."""
    payload = _revert_workflow_state(
        steps=steps,
        reason=reason,
        session_id=session_id or "terminal-session",
    )
    return _tool_result(
        data=payload,
        evidence_paths=[payload.get("state_file", "")],
        next_actions=["explain reverted values", "re-run validators", "ask the corrected blocking question"],
    )


def get_workflow_state_tools() -> list:
    """Return ADK workflow-state tool callables."""
    return [
        load_workflow_state,
        update_workflow_state,
        revert_workflow_state,
        reset_workflow_state,
    ]
