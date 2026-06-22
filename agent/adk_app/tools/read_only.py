"""Read-only ADK tool wrappers for AnyChain deterministic capabilities."""

from __future__ import annotations

from typing import Any
import subprocess

from analyzers.artifact_qa import answer_artifact_question as _answer_artifact_question
from analyzers.bottleneck_rules import diagnose_artifacts as _diagnose_artifacts
from analyzers.result_analyzer import analyze_job
from diagnostics.doctor import run_doctor as _run_doctor
from discovery.environment import discover_environment as _discover_environment
from knowledge.framework_capabilities import load_framework_capabilities as _load_framework_capabilities
from knowledge.loader import load_knowledge_provider, provider_status
from runners.job_manager import get_job, list_jobs, resume_job, tail_job_log as _tail_job_log


def discover_environment() -> dict[str, Any]:
    """Inspect the local host without changing it.

    Use first for real-node benchmarks or when the user asks whether the current
    environment is ready. It detects cloud/deployment hints, disks, network,
    Kubernetes context, and dependency availability.
    """
    return _tool_result(
        data=_discover_environment(),
        next_actions=["confirm inferred values", "run_doctor", "generate_benchmark_plan"],
    )


def run_doctor() -> dict[str, Any]:
    """Summarize readiness, missing dependencies, LLM auth, KB config, and capabilities.

    Use this when the user asks to check setup, diagnose why the Agent cannot
    run, or confirm what must be configured before a benchmark.
    """
    report = _run_doctor()
    return _tool_result(
        data=report,
        warnings=report.get("warnings", []),
        next_actions=report.get("next_actions", []),
    )


def audit_dependencies() -> dict[str, Any]:
    """Run the dependency installer in audit-only mode without changing the host."""
    benchmark_command = ["bash", "scripts/install_deps.sh", "--check"]
    agent_command = ["bash", "scripts/install_agent_deps.sh", "--check"]
    benchmark = subprocess.run(
        benchmark_command,
        cwd=_repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    agent = subprocess.run(
        agent_command,
        cwd=_repo_root(),
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


def load_framework_capabilities() -> dict[str, Any]:
    """Load current chain, RPC, adapter-family, and fake-node support from repo state.

    Use this before answering support questions. Do not rely on memory when the
    user asks how many chains, which RPC methods, or whether fake-node fixtures
    exist.
    """
    capabilities = _load_framework_capabilities()
    return _tool_result(
        data=capabilities,
        next_actions=["answer framework question", "run gap_analysis", "draft onboarding plan"],
    )


def list_supported_chains() -> dict[str, Any]:
    """Return supported chain names from current chain templates."""
    capabilities = _load_framework_capabilities()
    chain_rows = capabilities.get("chains", [])
    chains = sorted(row.get("chain", "") for row in chain_rows if row.get("chain"))
    return _tool_result(
        data={"chain_count": len(chains), "chains": chains},
        next_actions=["list_rpc_methods", "generate_benchmark_plan"],
    )


def list_rpc_methods(chain: str) -> dict[str, Any]:
    """Return single, mixed, and weighted RPC methods for one supported chain."""
    capabilities = _load_framework_capabilities()
    chain_data = next(
        (row for row in capabilities.get("chains", []) if row.get("chain") == chain),
        None,
    )
    if not chain_data:
        return _tool_result(
            status="not_found",
            data={"chain": chain, "methods": []},
            warnings=[f"unsupported chain: {chain}"],
            next_actions=["draft_chain_template", "gap_analysis"],
        )
    methods = {
        "single": chain_data.get("single", ""),
        "methods": chain_data.get("methods", []),
        "mixed_methods": chain_data.get("mixed_methods", []),
        "mixed_weighted": chain_data.get("mixed_weighted", []),
    }
    return _tool_result(
        data={"chain": chain, "methods": methods},
        next_actions=["design workload", "validate fake-node fixture coverage"],
    )


def knowledge_search(query: str, chain: str | None = None) -> dict[str, Any]:
    """Search the optional enterprise Knowledge Base adapter.

    Use this only when the configured KB can add private endpoint, workload,
    incident, or customer-specific knowledge. If disabled, fall back to local
    framework capability tools.
    """
    status = provider_status()
    if not status["enabled"] or status["error"]:
        return _tool_result(
            status="disabled",
            data={"provider_status": status, "results": []},
            warnings=[status["error"]] if status["error"] else [],
            next_actions=["answer from local framework capabilities"],
        )
    provider = load_knowledge_provider()
    data: dict[str, Any] = {
        "provider_status": status,
        "results": provider.search(query),
    }
    if chain:
        data["rpc_methods"] = provider.get_rpc_methods(chain)
    return _tool_result(data=data, next_actions=["ground answer with KB results"])


def latest_job(jobs_dir: str = ".agent/jobs") -> dict[str, Any]:
    """Return the latest known benchmark job from the durable job store."""
    jobs = list_jobs(jobs_dir=jobs_dir, limit=1)
    if not jobs:
        return _tool_result(status="not_found", data={"jobs": []}, next_actions=["create benchmark plan"])
    return _tool_result(data={"job": jobs[0]}, next_actions=["job_status", "tail_job_log", "analyze_artifacts"])


def job_status(job_id: str, jobs_dir: str = ".agent/jobs") -> dict[str, Any]:
    """Read current job metadata, runtime.env path, artifact index, and lifecycle status."""
    job = get_job(job_id, jobs_dir=jobs_dir)
    return _tool_result(
        data={"job": job, "resume": resume_job(job_id, jobs_dir=jobs_dir)},
        evidence_paths=_job_evidence_paths(job),
        next_actions=["tail_job_log", "analyze_artifacts", "resume_job"],
    )


def tail_job_log(job_id: str, jobs_dir: str = ".agent/jobs", lines: int = 80) -> dict[str, Any]:
    """Read recent benchmark log lines for a running or completed job."""
    payload = _tail_job_log(job_id, jobs_dir=jobs_dir, lines=lines)
    return _tool_result(
        data=payload,
        evidence_paths=[payload["log_file"]] if payload.get("log_file") else [],
        next_actions=["job_status", "analyze_artifacts"],
    )


def analyze_artifacts(job_id: str, jobs_dir: str = ".agent/jobs") -> dict[str, Any]:
    """Analyze generated benchmark artifacts and summarize evidence-backed results."""
    job = get_job(job_id, jobs_dir=jobs_dir)
    analysis = analyze_job(job)
    return _tool_result(
        data=analysis,
        evidence_paths=list(analysis.get("evidence", {}).values()) if isinstance(analysis.get("evidence"), dict) else [],
        warnings=analysis.get("warnings", []),
        next_actions=analysis.get("recommendations", []),
    )


def answer_artifact_question(
    question: str,
    job_id: str = "",
    jobs_dir: str = ".agent/jobs",
    artifact_index: str = "",
) -> dict[str, Any]:
    """Answer a user question using benchmark artifacts and deterministic diagnostics."""
    job = get_job(job_id, jobs_dir=jobs_dir) if job_id else None
    answer = _answer_artifact_question(question, job=job, artifact_index=artifact_index or None)
    return _tool_result(
        data=answer,
        evidence_paths=answer.get("evidence_paths", []),
        next_actions=["analyze_artifacts", "diagnose_artifacts"],
    )


def diagnose_artifacts(
    job_id: str = "",
    jobs_dir: str = ".agent/jobs",
    artifact_index: str = "",
) -> dict[str, Any]:
    """Run deterministic bottleneck and chart diagnostics against benchmark artifacts."""
    job = get_job(job_id, jobs_dir=jobs_dir) if job_id else None
    diagnostics = _diagnose_artifacts(job=job, artifact_index=artifact_index or None)
    return _tool_result(
        data=diagnostics,
        evidence_paths=diagnostics.get("evidence_paths", []),
        next_actions=["explain bottleneck findings", "review HTML report"],
    )


def get_read_only_tools() -> list:
    """Return read-only ADK tool callables."""
    return [
        discover_environment,
        run_doctor,
        audit_dependencies,
        load_framework_capabilities,
        list_supported_chains,
        list_rpc_methods,
        knowledge_search,
        latest_job,
        job_status,
        tail_job_log,
        analyze_artifacts,
        answer_artifact_question,
        diagnose_artifacts,
    ]


def _tool_result(
    data: dict[str, Any],
    status: str = "ok",
    evidence_paths: list[str] | None = None,
    warnings: list[str] | None = None,
    next_actions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "data": data,
        "evidence_paths": [path for path in (evidence_paths or []) if path],
        "warnings": [warning for warning in (warnings or []) if warning],
        "next_actions": next_actions or [],
        "requires_user_confirmation": False,
    }


def _job_evidence_paths(job: dict[str, Any]) -> list[str]:
    paths = [
        job.get("plan_file", ""),
        job.get("runtime_env_file", ""),
        job.get("artifact_index", ""),
    ]
    artifacts = job.get("artifacts", {})
    if isinstance(artifacts, dict):
        paths.extend(str(value) for value in artifacts.values() if isinstance(value, str))
    return [path for path in paths if path]


def _repo_root() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[3])
