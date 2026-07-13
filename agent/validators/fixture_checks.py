"""Fake-node fixture coverage/authenticity checks, shared by the CLI tool-call surface."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from ..runners.tool_result import tool_result as _tool_result
except ImportError:  # script execution with agent/ on sys.path
    from runners.tool_result import tool_result as _tool_result

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_COVERAGE = REPO_ROOT / "tools" / "fake-node" / "check_fixture_coverage.py"
FIXTURE_AUTHENTICITY = REPO_ROOT / "tools" / "fake-node" / "validate_fixture_authenticity.py"


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
    incomplete = bool(payload.get("exit_code")) or bool(payload.get("incomplete"))
    warnings = [f"{key}: {value}" for key, value in sorted((payload.get("statuses") or {}).items()) if key != "real-recorded"]
    if payload.get("exit_code"):
        warnings.append(f"fixture checker exit_code={payload['exit_code']}")
    return _tool_result(
        status="blocked" if incomplete else "ok",
        data=payload,
        warnings=warnings,
        next_actions=["record real request/response evidence"] if incomplete else ["continue fake-node smoke gate"],
    )


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
