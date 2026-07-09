"""ADK tool wrappers around deterministic AnyChain validators."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from ...validators.chain_template import validate_chain_template as _validate_chain_template
    from ...validators.config_contract import build_missing_config_questions as _build_missing_config_questions
    from ...validators.config_contract import validate_required_config as _validate_required_config
    from ...validators.execution_gate import validate_execution_gate as _validate_execution_gate
    from ...validators.onboarding_gate import build_onboarding_handoff as _build_onboarding_handoff
    from ...validators.endpoint_probe import validate_rpc_endpoint as _validate_rpc_endpoint
    from ...validators.rpc_workload import default_workload as _default_workload
    from ...validators.rpc_workload import validate_rpc_workload as _validate_rpc_workload
except ImportError:  # script execution with agent/ on sys.path
    from validators.chain_template import validate_chain_template as _validate_chain_template
    from validators.config_contract import build_missing_config_questions as _build_missing_config_questions
    from validators.config_contract import validate_required_config as _validate_required_config
    from validators.execution_gate import validate_execution_gate as _validate_execution_gate
    from validators.onboarding_gate import build_onboarding_handoff as _build_onboarding_handoff
    from validators.endpoint_probe import validate_rpc_endpoint as _validate_rpc_endpoint
    from validators.rpc_workload import default_workload as _default_workload
    from validators.rpc_workload import validate_rpc_workload as _validate_rpc_workload

from .read_only import _tool_result

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_COVERAGE = REPO_ROOT / "tools" / "fake-node" / "check_fixture_coverage.py"
FIXTURE_AUTHENTICITY = REPO_ROOT / "tools" / "fake-node" / "validate_fixture_authenticity.py"


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


def validate_rpc_endpoint(
    chain: str,
    endpoint: str,
    methods: list[str] | None = None,
    address: str = "",
    timeout: float = 3.0,
    adapter_family: str = "",
    method_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Probe a user-provided endpoint and selected RPC methods before trust."""
    payload = _validate_rpc_endpoint(
        chain,
        endpoint,
        methods or [],
        address,
        timeout,
        adapter_family=adapter_family,
        method_params=method_params or {},
    )
    return _tool_result(
        status="ok" if payload["ready"] else "blocked",
        data=payload,
        warnings=payload.get("warnings", []) + payload.get("blockers", []),
        next_actions=["continue endpoint-dependent workflow"] if payload["ready"] else ["ask for corrected endpoint or method samples"],
    )


def validate_fake_node_fixture_coverage(
    chains: str = "all",
    modes: str = "single,mixed",
    strict: bool = True,
) -> dict[str, Any]:
    """Validate fake-node fixture coverage through the canonical fake-node checker."""
    command = ["python3", str(FIXTURE_COVERAGE), "--chains", chains or "all", "--modes", modes or "single,mixed", "--json"]
    if strict:
        command.append("--strict")
    payload = _run_json_tool(command, allow_failure=True)
    incomplete = _fixture_payload_incomplete(payload)
    return _tool_result(
        status="blocked" if incomplete else "ok",
        data=payload,
        warnings=_fixture_warnings(payload),
        next_actions=["record missing or placeholder fixtures"] if incomplete else ["continue fake-node smoke gate"],
    )


def validate_fake_node_fixture_authenticity(
    modes: str = "single,mixed",
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate that fixture files have matching recorded request/response evidence."""
    command = ["python3", str(FIXTURE_AUTHENTICITY), "--modes", modes or "single,mixed", "--json"]
    if allow_incomplete:
        command.append("--allow-incomplete")
    payload = _run_json_tool(command, allow_failure=True)
    incomplete = bool(payload.get("incomplete"))
    return _tool_result(
        status="blocked" if incomplete else "ok",
        data=payload,
        warnings=[f"{key}: {value}" for key, value in sorted((payload.get("statuses") or {}).items()) if key != "real-recorded"],
        next_actions=["record real request/response evidence"] if incomplete else ["continue fake-node smoke gate"],
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
        next_actions=[
            "collect missing evidence",
            "prepare in-chat draft content only; do not claim any file was written",
            "run validation commands after a reviewed file exists",
        ],
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
        validate_rpc_endpoint,
        validate_fake_node_fixture_coverage,
        validate_fake_node_fixture_authenticity,
        build_onboarding_handoff,
    ]


def _run_json_tool(command: list[str], allow_failure: bool = False) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        payload = {}
    payload.setdefault("command", " ".join(command))
    payload["exit_code"] = completed.returncode
    if completed.stderr:
        payload["stderr"] = completed.stderr.strip()
    if completed.returncode and not allow_failure:
        raise RuntimeError(f"{command[1]} failed with exit_code={completed.returncode}: {completed.stderr}")
    return payload


def _fixture_payload_incomplete(payload: dict[str, Any]) -> bool:
    if payload.get("exit_code"):
        return True
    statuses = payload.get("statuses") or {}
    return any(key != "ok" and count for key, count in statuses.items())


def _fixture_warnings(payload: dict[str, Any]) -> list[str]:
    statuses = payload.get("statuses") or {}
    warnings = [f"{key}: {value}" for key, value in sorted(statuses.items()) if key != "ok"]
    if payload.get("exit_code"):
        warnings.append(f"fixture checker exit_code={payload['exit_code']}")
    return warnings
