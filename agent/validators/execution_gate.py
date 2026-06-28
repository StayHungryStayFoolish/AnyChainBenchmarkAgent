"""Execution gate validators for smoke and real benchmark actions."""

from __future__ import annotations

from typing import Any


def validate_execution_gate(
    plan: dict[str, Any] | None,
    preflight: dict[str, Any] | None = None,
    smoke: dict[str, Any] | None = None,
    approved: bool = False,
    real_execution: bool = False,
) -> dict[str, Any]:
    """Return whether the Agent may run smoke or real benchmark execution."""
    blockers = []
    if not plan:
        blockers.append("plan is required")
    if preflight is None:
        blockers.append("preflight is required")
    elif not preflight.get("passed", False):
        blockers.append("preflight must pass")
    if real_execution and not smoke:
        blockers.append("smoke validation is required before real execution")
    if real_execution and smoke and smoke.get("status") not in {"completed", "ok", "passed"}:
        blockers.append("smoke validation must complete successfully")
    if not approved:
        blockers.append("explicit user approval is required")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "real_execution": real_execution,
        "requires_user_confirmation": not approved,
    }
