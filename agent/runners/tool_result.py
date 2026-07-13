"""Shared structured-result envelope for tool-shaped call results.

Lives here (not in `agent/adk_app/`) so both the ADK tool bridge and the
LangGraph Harness can build the same `{status, data, evidence_paths,
warnings, next_actions}` shape without the Harness importing from the ADK
bridge layer.
"""

from __future__ import annotations

from typing import Any


def tool_result(
    data: dict[str, Any],
    status: str = "ok",
    evidence_paths: list[str] | None = None,
    warnings: list[str] | None = None,
    next_actions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "data": data,
        "evidence_paths": [path for path in (evidence_paths or []) if path],
        "warnings": [warning for warning in (warnings or []) if warning],
        "next_actions": next_actions or [],
        "requires_user_confirmation": False,
    }
