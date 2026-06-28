"""ADK session-state adapter backed by AnyChain runtime artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runners.job_manager import list_jobs, resume_job


def load_startup_state(
    jobs_dir: str | Path = ".agent/jobs",
) -> dict[str, Any]:
    """Load safe job state for ADK tools and runtime startup.

    The returned state is intentionally file-backed so long-running benchmark
    jobs can outlive the terminal session or ADK process.
    """
    latest = _latest_job_state(jobs_dir)
    return {
        "jobs_dir": str(jobs_dir),
        "latest_job": latest,
        "resume_available": bool(latest),
        "next_actions": _startup_next_actions(latest),
    }


def preserved_state_for_adk(state: dict[str, Any]) -> dict[str, Any]:
    """Extract job-critical values that should be rehydrated into ADK state."""
    preserved: dict[str, Any] = {}
    latest = state.get("latest_job") or {}
    if latest:
        preserved.setdefault("job_id", latest.get("job_id", ""))
        preserved.setdefault("job_status", latest.get("status", ""))
        preserved.setdefault("runtime_env_file", latest.get("runtime_env_file", ""))
        preserved.setdefault("artifact_index", latest.get("artifact_index", ""))
        preserved.setdefault("plan_file", latest.get("plan_file", ""))
    return {key: value for key, value in preserved.items() if value not in ("", None)}


def _latest_job_state(jobs_dir: str | Path) -> dict[str, Any]:
    jobs = list_jobs(jobs_dir=jobs_dir, limit=1)
    if not jobs:
        return {}
    job_id = jobs[0]["job_id"]
    try:
        return resume_job(job_id, jobs_dir=jobs_dir)
    except Exception:
        return jobs[0]


def _startup_next_actions(latest_job: dict[str, Any]) -> list[str]:
    if not latest_job:
        return ["ask for benchmark goal", "run environment discovery"]
    status = latest_job.get("status", "unknown")
    if status == "running":
        return ["show job status", "tail job logs", "wait for completion"]
    if status == "completed":
        return ["analyze latest job", "show report evidence", "start a new benchmark"]
    if status == "failed":
        return ["tail job logs", "inspect runtime.env", "generate retry plan"]
    return ["show job status", "ask user for next action"]
