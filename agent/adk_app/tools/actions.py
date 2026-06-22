"""Confirmation-gated ADK action tool wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import subprocess

from runners.job_manager import resume_job as _resume_job
from runners.job_manager import submit_job as _submit_job
from planners.strategy_planner import write_json

from .read_only import _tool_result


def run_smoke(plan_file: str, jobs_dir: str = ".agent/jobs", approved: bool = False) -> dict[str, Any]:
    """Run a lifecycle-only mock smoke job after user confirmation."""
    if not approved:
        return _confirmation_required(
            action="run_smoke",
            summary="Run a mock smoke job to validate lifecycle and artifacts before real execution.",
            next_actions=["ask user for yes/no confirmation"],
        )
    job = _submit_job(plan_file, jobs_dir=jobs_dir, mock=True, approved=True)
    return _tool_result(
        data={"job": job},
        evidence_paths=[job.get("runtime_env_file", ""), job.get("artifact_index", "")],
        next_actions=["analyze_artifacts", "ask approval for real benchmark"],
    )


def run_fake_node_smoke_benchmark(
    plan_file: str,
    jobs_dir: str = ".agent/jobs",
    approved: bool = False,
) -> dict[str, Any]:
    """Run the real benchmark engine in quick fake-node mode after approval.

    This is different from ``run_smoke``. It executes the benchmark entry
    script with ``--quick --fake-node`` and injects job-local output directories
    so smoke data does not overwrite the user's normal benchmark result tree.
    """
    if not approved:
        return _confirmation_required(
            action="run_fake_node_smoke_benchmark",
            summary="Run real quick benchmark traffic against fake-node with isolated output paths.",
            next_actions=["ask user for explicit yes/no confirmation"],
        )
    plan_path = Path(plan_file)
    if not plan_path.is_file():
        return _tool_result(
            status="blocked",
            data={"plan_file": plan_file},
            warnings=[f"plan file not found: {plan_file}"],
            next_actions=["prepare_benchmark_run", "write plan file"],
        )

    smoke_root = Path(jobs_dir) / "fake_node_smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    smoke_plan = _fake_node_smoke_plan(plan_path, smoke_root)
    smoke_plan_file = smoke_root / f"{smoke_plan['plan_id']}.json"
    write_json(smoke_plan_file, smoke_plan)

    job = _submit_job(smoke_plan_file, jobs_dir=jobs_dir, mock=False, approved=True)
    evidence = [
        job.get("runtime_env_file", ""),
        job.get("artifact_index", ""),
        str(smoke_root / "benchmark-data"),
    ]
    return _tool_result(
        status="ok" if job.get("status") in {"completed", "running"} else "failed",
        data={"job": job, "smoke_plan_file": str(smoke_plan_file), "isolated_output_root": str(smoke_root)},
        evidence_paths=evidence,
        warnings=[job.get("error", "")] if job.get("error") else [],
        next_actions=["job_status", "tail_job_log", "analyze_artifacts", "ask approval for submit_benchmark_job"],
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
    if not Path(plan_file).is_file():
        return _tool_result(
            status="blocked",
            data={"plan_file": plan_file},
            warnings=[f"plan file not found: {plan_file}"],
            next_actions=["generate_benchmark_plan", "write plan file"],
        )
    # The generated plan controls foreground/detached mode. The detached
    # argument is kept in the tool schema so the Agent can explain the default.
    _ = detached
    job = _submit_job(plan_file, jobs_dir=jobs_dir, mock=False, approved=True)
    return _tool_result(
        data={"job": job},
        evidence_paths=[job.get("runtime_env_file", ""), job.get("artifact_index", "")],
        next_actions=["job_status", "tail_job_log", "analyze_artifacts"],
    )


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
    return [run_smoke, run_fake_node_smoke_benchmark, submit_benchmark_job, install_dependencies, resume_job, stop_job]


def _confirmation_required(action: str, summary: str, next_actions: list[str]) -> dict[str, Any]:
    return {
        "status": "needs_confirmation",
        "data": {"action": action, "summary": summary},
        "evidence_paths": [],
        "warnings": [],
        "next_actions": next_actions,
        "requires_user_confirmation": True,
    }


def _fake_node_smoke_plan(plan_file: Path, smoke_root: Path) -> dict[str, Any]:
    import json

    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan = dict(plan)
    plan["plan_id"] = f"{plan.get('plan_id', 'plan')}_fake_node_smoke"
    plan["strategy"] = "smoke"
    plan["goal"] = "smoke"
    plan["use_fake_node"] = True
    plan["required_inputs"] = [
        item for item in plan.get("required_inputs", [])
        if item not in _FAKE_NODE_IGNORED_REQUIREMENTS
    ]
    plan["required_questions"] = [
        item for item in plan.get("required_questions", [])
        if item.get("id") not in _FAKE_NODE_IGNORED_REQUIREMENTS and item.get("id") != "ledger_device_confirmation"
    ]
    checklist = dict(plan.get("configuration_checklist", {}))
    if checklist:
        checklist["missing_blockers"] = [
            item for item in checklist.get("missing_blockers", [])
            if item not in _FAKE_NODE_IGNORED_REQUIREMENTS
        ]
        checklist["summary"] = "fake-node smoke configuration uses isolated job-local output and has no real-node blocker requirements."
        plan["configuration_checklist"] = checklist
    command = ["./blockchain_node_benchmark.sh", "--quick", f"--{plan.get('rpc_mode', 'single')}", "--fake-node"]
    execution = dict(plan.get("execution", {}))
    env = dict(execution.get("environment", {}))
    env.update({
        "BLOCKCHAIN_NODE": plan.get("chain", env.get("BLOCKCHAIN_NODE", "")),
        "RPC_MODE": plan.get("rpc_mode", env.get("RPC_MODE", "single")),
        "LOCAL_RPC_URL": env.get("LOCAL_RPC_URL", ""),
        "QUICK_INITIAL_QPS": "1",
        "QUICK_MAX_QPS": "1",
        "QUICK_QPS_STEP": "1",
        "QUICK_DURATION": "10",
        "BLOCKCHAIN_BENCHMARK_DATA_DIR": str(smoke_root / "benchmark-data"),
        "MEMORY_SHARE_DIR": str(smoke_root / "memory"),
    })
    execution.update({
        "command": command,
        "environment": env,
        "runner_mode": "foreground",
    })
    plan["execution"] = execution
    materialized = dict(plan.get("materialized_config", {}))
    materialized.update({
        "BLOCKCHAIN_BENCHMARK_DATA_DIR": str(smoke_root / "benchmark-data"),
        "MEMORY_SHARE_DIR": str(smoke_root / "memory"),
    })
    plan["materialized_config"] = materialized
    artifacts = dict(plan.get("artifacts", {}))
    artifacts.update({
        "fake_node_smoke_output_root": str(smoke_root / "benchmark-data"),
        "fake_node_smoke_memory_dir": str(smoke_root / "memory"),
    })
    plan["artifacts"] = artifacts
    return plan


_FAKE_NODE_IGNORED_REQUIREMENTS = {
    "local_rpc_url",
    "blockchain_process_names",
    "ledger_device",
    "data_vol_type",
    "data_vol_size",
    "data_vol_max_iops",
    "data_vol_max_throughput",
    "network_interface",
    "network_max_bandwidth_gbps",
}
