"""Command safety guardrails for Agent execution."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any

from agent.runners.execution_scenarios import scenario_by_id, workflow_type_from_plan


ALLOWED_BENCHMARK_COMMANDS = {
    "./blockchain_node_benchmark.sh",
    "blockchain_node_benchmark.sh",
}


def build_benchmark_command(command: list[str]) -> list[str]:
    """Return the concrete command used to execute a benchmark plan."""
    if not command:
        return command
    if command[0] not in ALLOWED_BENCHMARK_COMMANDS:
        return command
    bash = find_compatible_bash()
    if not bash:
        return command
    return [bash, *command]


def find_compatible_bash() -> str:
    """Find a Bash 4+ executable for benchmark scripts."""
    candidates: list[str] = []
    override = os.environ.get("ANYCHAIN_BASH", "").strip()
    if override:
        candidates.append(override)
    for candidate in (
        shutil.which("bash") or "",
        "/usr/local/bin/bash",
        "/bin/bash",
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if _bash_major_version(candidate) >= 4:
            return candidate
    return ""


def validate_execution_plan(plan: dict[str, Any], approved: bool = False) -> list[str]:
    errors: list[str] = []
    command = plan.get("execution", {}).get("command", [])
    if not command:
        return ["execution.command is empty"]

    executable = command[0]
    if executable not in ALLOWED_BENCHMARK_COMMANDS:
        errors.append(f"command is not allowlisted for Agent execution: {executable}")
    elif not find_compatible_bash():
        errors.append(
            "benchmark execution requires the supported Docker/Linux environment with Bash 4+; "
            "set ANYCHAIN_BASH only when pointing at a compatible Linux bash"
        )

    if "plan_execution" in plan.get("approval_checkpoints", []) and not approved:
        errors.append("plan_execution approval is required")

    if "stress_execution" in plan.get("approval_checkpoints", []) and not approved:
        errors.append("stress_execution approval is required")

    if "dependency_install" in plan.get("approval_checkpoints", []) and not approved:
        errors.append("dependency_install approval is required")

    provenance = plan.get("execution_provenance")
    if isinstance(provenance, dict) and provenance.get("operation"):
        try:
            scenario = scenario_by_id(str(provenance.get("scenario_id") or ""))
        except ValueError as exc:
            errors.append(str(exc))
        else:
            operation = str(provenance.get("operation") or "")
            workflow = workflow_type_from_plan(plan)
            if operation != scenario.operation:
                errors.append("execution provenance operation does not match its scenario")
            if workflow != scenario.workflow_type:
                errors.append("execution workflow does not match its scenario")
            for token in scenario.required_command_tokens:
                if token not in command:
                    errors.append(f"execution command is missing required token: {token}")
            for token in scenario.forbidden_command_tokens:
                if token in command:
                    errors.append(f"execution command contains forbidden token: {token}")

    return errors


def _bash_major_version(path: str) -> int:
    try:
        completed = subprocess.run(
            [path, "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except Exception:
        return 0
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    match = re.search(r"version\s+(\d+)\.", first_line, re.IGNORECASE)
    if not match:
        return 0
    return int(match.group(1))
