"""Benchmark/Agent dependency install and audit, shared by the CLI tool-call surface.

Ungated core logic — the caller (``agent/tools/executor.py``) is responsible
for requiring explicit user/platform approval before calling
``install_dependencies``, matching the pattern already used by
``agent/runners/benchmark_pipeline.py`` (ungated core, gate lives at the
tool-dispatch layer).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agent.runners.tool_result import tool_result as _tool_result

REPO_ROOT = Path(__file__).resolve().parents[2]


def audit_dependencies() -> dict[str, Any]:
    """Run the dependency installer in audit-only mode without changing the host."""
    benchmark_command = ["bash", "scripts/install_deps.sh", "--check"]
    agent_command = ["bash", "scripts/install_agent_deps.sh", "--check"]
    benchmark = subprocess.run(
        benchmark_command,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    agent = subprocess.run(
        agent_command,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    status = "ok" if benchmark.returncode == 0 and agent.returncode == 0 else "needs_dependencies"
    return _tool_result(
        status=status,
        data={
            "benchmark": {
                "command": benchmark_command,
                "exit_code": benchmark.returncode,
                "output": benchmark.stdout[-12000:],
            },
            "agent_runtime": {
                "command": agent_command,
                "exit_code": agent.returncode,
                "output": agent.stdout[-12000:],
            },
        },
        warnings=[] if status == "ok" else ["dependency check found missing requirements"],
        next_actions=["review missing dependencies", "ask approval before install_dependencies"],
    )


def install_dependencies(
    no_sudo: bool = True,
    include_vegeta: bool = True,
    include_agent_runtime: bool = False,
    include_gcloud: bool = False,
    adk_venv: str = ".venv-adk",
    allow_system_python: bool = False,
) -> dict[str, Any]:
    """Install benchmark (and optionally Agent runtime) dependencies.

    Caller must already have obtained explicit approval; this function has no
    confirmation gate of its own.
    """
    benchmark_command = ["bash", "scripts/install_deps.sh", "--yes"]
    if no_sudo:
        benchmark_command.append("--no-sudo")
    if not include_vegeta:
        benchmark_command.append("--no-vegeta")
    if allow_system_python:
        benchmark_command.append("--system-python")
    benchmark = subprocess.run(
        benchmark_command,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    agent = None
    if include_agent_runtime or include_gcloud:
        agent_command = ["bash", "scripts/install_agent_deps.sh", "--yes", "--adk-venv", adk_venv]
        if no_sudo:
            agent_command.append("--no-sudo")
        if include_gcloud:
            agent_command.append("--with-gcloud")
        agent = subprocess.run(
            agent_command,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    exit_codes = [benchmark.returncode]
    if agent is not None:
        exit_codes.append(agent.returncode)
    ok = all(code == 0 for code in exit_codes)
    return _tool_result(
        status="ok" if ok else "failed",
        data={
            "benchmark": {
                "command": benchmark_command,
                "exit_code": benchmark.returncode,
                "output": benchmark.stdout[-12000:],
            },
            "agent_runtime": (
                {
                    "command": agent.args,
                    "exit_code": agent.returncode,
                    "output": agent.stdout[-12000:],
                }
                if agent is not None
                else {"skipped": True}
            ),
        },
        warnings=[] if ok else ["dependency installation did not complete successfully"],
        next_actions=["run audit_dependencies", "run prepare_benchmark_run"],
    )
