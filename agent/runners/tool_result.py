"""Shared structured-result envelope for tool-shaped call results.

Used by the LangGraph Harness (`agent/harness/nodes/execution.py`) and the
CLI tool-dispatch surface (`agent/tools/executor.py`) alike, so both build the
same `{status, data, evidence_paths, warnings, next_actions}` shape from one
definition.
"""

from __future__ import annotations

from typing import Any


def tool_result(
    data: dict[str, Any],
    status: str = "ok",
    evidence_paths: list[str] | None = None,
    warnings: list[str] | None = None,
    next_actions: list[str] | None = None,
    requires_user_confirmation: bool = False,
) -> dict[str, Any]:
    return {
        "status": status,
        "data": data,
        "evidence_paths": [path for path in (evidence_paths or []) if path],
        "warnings": [warning for warning in (warnings or []) if warning],
        "next_actions": next_actions or [],
        "requires_user_confirmation": requires_user_confirmation,
    }
