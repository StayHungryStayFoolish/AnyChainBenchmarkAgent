"""Confirmation-gated ADK action tool wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import subprocess

try:
    from ...runners.benchmark_pipeline import _job_terminal_commands, _job_user_next_actions, _nested_job
    from ...runners.benchmark_pipeline import run_fake_node_smoke_benchmark as _run_fake_node_smoke_benchmark_core
    from ...runners.benchmark_pipeline import submit_benchmark_job as _submit_benchmark_job_core
    from ...runners.job_manager import resume_job as _resume_job
    from ...runners.tool_result import tool_result as _tool_result
except ImportError:  # script execution with agent/ on sys.path
    from runners.benchmark_pipeline import _job_terminal_commands, _job_user_next_actions, _nested_job
    from runners.benchmark_pipeline import run_fake_node_smoke_benchmark as _run_fake_node_smoke_benchmark_core
    from runners.benchmark_pipeline import submit_benchmark_job as _submit_benchmark_job_core
    from runners.job_manager import resume_job as _resume_job
    from runners.tool_result import tool_result as _tool_result


def run_fake_node_smoke_benchmark(
    plan_file: str,
    jobs_dir: str = ".agent/jobs",
    approved: bool = False,
) -> dict[str, Any]:
    """Run the real benchmark engine in quick fake-node mode after approval.

    It executes the benchmark entry script with ``--quick --fake-node`` and
    injects job-local output directories so smoke data does not overwrite the
    user's normal benchmark result tree.
    """
    if not approved:
        return _confirmation_required(
            action="run_fake_node_smoke_benchmark",
            summary="Run real quick benchmark traffic against fake-node with isolated output paths.",
            next_actions=["ask user for explicit yes/no confirmation"],
        )
    return _run_fake_node_smoke_benchmark_core(plan_file, jobs_dir=jobs_dir)


def run_quick_assumed_fake_node_smoke(
    source_prompt: str = "",
    chain: str = "solana",
    rpc_mode: str = "single",
    jobs_dir: str = ".agent/jobs",
    approved: bool = False,
) -> dict[str, Any]:
    """Run the approved quick fake-node smoke path with smoke-only assumptions.

    This tool exists to keep the common "just verify the Agent/framework can
    run" path deterministic. It does not create real benchmark approval and the
    generated plan is marked ``assumed_for_smoke`` so execution gates block it
    from being promoted to real-node testing.
    """
    if not approved:
        return _confirmation_required(
            action="run_quick_assumed_fake_node_smoke",
            summary="Run quick fake-node smoke with explicit smoke-only assumed values.",
            next_actions=["ask user for explicit yes/no confirmation"],
        )

    from .planning import prepare_benchmark_run

    prepared = prepare_benchmark_run(
        source_prompt=source_prompt,
        chain=(chain or "solana").strip().lower(),
        goal="smoke",
        rpc_mode=(rpc_mode or "single").strip().lower(),
        use_fake_node=True,
        confirmations=[
            "benchmark_mode_confirmed",
            "qps_profile_confirmed",
            "observability_choice_confirmed",
            "chain_template_reviewed",
            "rpc_workload_confirmed",
            "rpc_workload_confirmation",
            "rpc_param_samples_confirmed",
            "rpc_param_samples_confirmation",
            "custom_rpc_method_review",
            "advanced_config_review",
            "disk_inventory_confirmation",
            "ledger_device_confirmation",
            "has_accounts_device",
        ],
        assumed_for_smoke=True,
    )
    data = prepared.get("data", {})
    preflight = data.get("preflight", {})
    if not preflight.get("passed"):
        return _tool_result(
            status="blocked",
            data={
                "prepared": data,
                "preflight": preflight,
            },
            evidence_paths=prepared.get("evidence_paths", []),
            warnings=prepared.get("warnings", []) + preflight.get("blockers", []),
            next_actions=["report exact preflight blockers", "ask user for missing values"],
        )

    smoke = run_fake_node_smoke_benchmark(
        str(data.get("plan_file", "")),
        jobs_dir=jobs_dir,
        approved=True,
    )
    merged_evidence = []
    merged_evidence.extend(prepared.get("evidence_paths", []))
    merged_evidence.extend(smoke.get("evidence_paths", []))
    return _tool_result(
        status=smoke.get("status", "ok"),
        data={
            "prepared": data,
            "smoke": smoke.get("data", {}),
            "assumed_for_smoke": True,
            "terminal_commands": _job_terminal_commands(_nested_job(smoke)),
        },
        evidence_paths=[item for item in merged_evidence if item],
        warnings=prepared.get("warnings", []) + smoke.get("warnings", []),
        next_actions=_job_user_next_actions(_nested_job(smoke)),
    )


def submit_benchmark_job(
    plan_file: str,
    jobs_dir: str = ".agent/jobs",
    detached: bool = True,
    approved: bool = False,
) -> dict[str, Any]:
    """Submit a real benchmark job after explicit user confirmation."""
    if not approved:
        return _confirmation_required(
            action="submit_benchmark_job",
            summary="Submit a real benchmark job. This can generate load against the target node.",
            next_actions=["ask user for explicit yes/no confirmation"],
        )
    # The generated plan controls foreground/detached mode. The detached
    # argument is kept in the tool schema so the Agent can explain the default.
    _ = detached
    return _submit_benchmark_job_core(plan_file, jobs_dir=jobs_dir)


def install_dependencies(
    approved: bool = False,
    no_sudo: bool = True,
    include_vegeta: bool = True,
    include_agent_runtime: bool = False,
    include_gcloud: bool = False,
    adk_venv: str = ".venv-adk",
    allow_system_python: bool = False,
) -> dict[str, Any]:
    """Install benchmark dependencies after explicit user approval.

    Normal Agent usage installs the ADK runtime once before launch, then lets
    this tool install the benchmark engine dependencies. Google ADK/gcloud setup
    is repeated only when explicitly requested.
    """
    if not approved:
        return _confirmation_required(
            action="install_dependencies",
            summary="Install or update local benchmark and Agent runtime dependencies.",
            next_actions=["ask user for explicit yes/no confirmation"],
        )
    repo = Path(__file__).resolve().parents[3]
    benchmark_command = ["bash", "scripts/install_deps.sh", "--yes"]
    if no_sudo:
        benchmark_command.append("--no-sudo")
    if not include_vegeta:
        benchmark_command.append("--no-vegeta")
    if allow_system_python:
        benchmark_command.append("--system-python")
    benchmark = subprocess.run(
        benchmark_command,
        cwd=str(repo),
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
            cwd=str(repo),
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
                    "command": agent.args if agent is not None else [],
                    "exit_code": agent.returncode if agent is not None else 0,
                    "output": agent.stdout[-12000:] if agent is not None else "",
                }
                if include_agent_runtime
                else {"skipped": True}
            ),
        },
        warnings=[] if ok else ["dependency installation did not complete successfully"],
        next_actions=["run audit_dependencies", "run prepare_benchmark_run"],
    )


def resume_job(job_id: str, jobs_dir: str = ".agent/jobs") -> dict[str, Any]:
    """Resume the file-backed AnyChain job context after terminal/session restart."""
    payload = _resume_job(job_id, jobs_dir=jobs_dir)
    return _tool_result(
        data=payload,
        evidence_paths=[payload.get("runtime_env_file", ""), payload.get("artifact_index", "")],
        next_actions=payload.get("next_actions", []),
    )


def stop_job(job_id: str, jobs_dir: str = ".agent/jobs", approved: bool = False) -> dict[str, Any]:
    """Request a stop action for a running job after confirmation.

    Stop is intentionally conservative in this phase; process termination is
    implemented only after job lifecycle tests cover the target runner modes.
    """
    if not approved:
        return _confirmation_required(
            action="stop_job",
            summary="Stop a running benchmark job.",
            next_actions=["ask user for explicit yes/no confirmation"],
        )
    return _tool_result(
        status="not_implemented",
        data={"job_id": job_id, "jobs_dir": jobs_dir},
        warnings=["stop_job requires runner-specific termination support before it can be enabled"],
        next_actions=["inspect job_status", "tail_job_log"],
    )


def get_action_tools() -> list:
    """Return confirmation-gated action ADK tool callables."""
    return [
        run_fake_node_smoke_benchmark,
        run_quick_assumed_fake_node_smoke,
        submit_benchmark_job,
        install_dependencies,
        resume_job,
        stop_job,
    ]


def _confirmation_required(action: str, summary: str, next_actions: list[str]) -> dict[str, Any]:
    return {
        "status": "needs_confirmation",
        "data": {"action": action, "summary": summary},
        "evidence_paths": [],
        "warnings": [],
        "next_actions": next_actions,
        "requires_user_confirmation": True,
    }
