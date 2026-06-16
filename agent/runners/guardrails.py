"""Command safety guardrails for Agent execution."""

from __future__ import annotations

from typing import Any


ALLOWED_BENCHMARK_COMMANDS = {
    "./blockchain_node_benchmark.sh",
    "blockchain_node_benchmark.sh",
}


def validate_execution_plan(plan: dict[str, Any], approved: bool = False) -> list[str]:
    errors: list[str] = []
    command = plan.get("execution", {}).get("command", [])
    if not command:
        return ["execution.command is empty"]

    executable = command[0]
    if executable not in ALLOWED_BENCHMARK_COMMANDS:
        errors.append(f"command is not allowlisted for Agent execution: {executable}")

    if "plan_execution" in plan.get("approval_checkpoints", []) and not approved:
        errors.append("plan_execution approval is required")

    if "stress_execution" in plan.get("approval_checkpoints", []) and not approved:
        errors.append("stress_execution approval is required")

    if "dependency_install" in plan.get("approval_checkpoints", []) and not approved:
        errors.append("dependency_install approval is required")

    return errors
