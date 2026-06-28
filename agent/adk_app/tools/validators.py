"""ADK tool wrappers around deterministic AnyChain validators."""

from __future__ import annotations

from typing import Any

from validators.chain_template import validate_chain_template as _validate_chain_template
from validators.config_contract import build_missing_config_questions as _build_missing_config_questions
from validators.config_contract import validate_required_config as _validate_required_config
from validators.execution_gate import validate_execution_gate as _validate_execution_gate
from validators.onboarding_gate import build_onboarding_handoff as _build_onboarding_handoff
from validators.rpc_workload import default_workload as _default_workload
from validators.rpc_workload import validate_rpc_workload as _validate_rpc_workload

from .read_only import _tool_result


def validate_required_config(target_mode: str = "", confirmed_config: dict | None = None) -> dict[str, Any]:
    """Validate required fake-node or real-node runtime configuration."""
    return _tool_result(data=_validate_required_config(target_mode or None, confirmed_config or {}))


def build_missing_config_questions(
    target_mode: str = "",
    confirmed_config: dict | None = None,
    discovery: dict | None = None,
) -> dict[str, Any]:
    """Build precise configuration questions from current confirmed values."""
    return _tool_result(data=_build_missing_config_questions(target_mode or None, confirmed_config or {}, discovery or {}))


def validate_rpc_workload(
    chain: str,
    rpc_mode: str,
    methods: list[str] | None = None,
    mixed_weights: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Validate single/custom/mixed RPC workload choices."""
    payload = _validate_rpc_workload(chain, rpc_mode, methods or [], mixed_weights or {})
    return _tool_result(
        status="ok" if payload["ready"] else "blocked",
        data=payload,
        warnings=payload.get("errors", []) + payload.get("warnings", []),
        next_actions=["ask for corrected RPC workload"] if not payload["ready"] else ["confirm TARGET_* samples"],
    )


def load_default_workload(chain: str) -> dict[str, Any]:
    """Load chain-template single and mixed workload defaults."""
    return _tool_result(data=_default_workload(chain), next_actions=["ask user whether to use defaults or customize"])


def validate_chain_template(chain: str) -> dict[str, Any]:
    """Validate the selected chain template and workload metadata."""
    payload = _validate_chain_template(chain)
    return _tool_result(
        status="ok" if payload["ready"] else "blocked",
        data=payload,
        warnings=payload.get("errors", []),
        next_actions=["continue benchmark setup"] if payload["ready"] else ["generate onboarding handoff"],
    )


def validate_execution_gate(
    plan: dict | None = None,
    preflight: dict | None = None,
    smoke: dict | None = None,
    approved: bool = False,
    real_execution: bool = False,
) -> dict[str, Any]:
    """Validate approval, preflight, and smoke gates before execution."""
    payload = _validate_execution_gate(plan, preflight, smoke, approved, real_execution)
    return _tool_result(
        status="ok" if payload["ready"] else "blocked",
        data=payload,
        warnings=payload.get("blockers", []),
        next_actions=["execute approved action"] if payload["ready"] else ["ask for missing gate or approval"],
    )


def build_onboarding_handoff(
    chain: str,
    family: str,
    methods: list[str] | None = None,
    evidence: dict | None = None,
) -> dict[str, Any]:
    """Build an evidence-aware chain/RPC onboarding handoff."""
    payload = _build_onboarding_handoff(chain, family, methods or [], evidence or {})
    return _tool_result(
        status="ok" if payload["ready_for_coding"] else "needs_evidence",
        data=payload,
        warnings=[f"missing evidence: {item}" for item in payload.get("missing_evidence", [])],
        next_actions=["collect missing evidence", "draft chain template", "run validation commands"],
    )


def get_validator_tools() -> list:
    """Return deterministic validator tool callables for ADK agents."""
    return [
        validate_required_config,
        build_missing_config_questions,
        validate_rpc_workload,
        load_default_workload,
        validate_chain_template,
        validate_execution_gate,
        build_onboarding_handoff,
    ]
