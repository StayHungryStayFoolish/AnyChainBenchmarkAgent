"""REPL startup state: safe, file-backed job state for session/process restarts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.harness.failures import failure_record_from_job
from agent.runners.job_manager import DEFAULT_JOBS_DIR, get_job, list_jobs, resume_job


def load_startup_state(
    jobs_dir: str | Path = DEFAULT_JOBS_DIR,
) -> dict[str, Any]:
    """Load safe job state for terminal startup.

    The returned state is intentionally file-backed so long-running benchmark
    jobs can outlive the terminal session or process restart.
    """
    latest = _latest_job_state(jobs_dir)
    return {
        "jobs_dir": str(jobs_dir),
        "latest_job": latest,
        "resume_available": bool(latest),
        "next_actions": _startup_next_actions(latest),
    }


def _latest_job_state(jobs_dir: str | Path) -> dict[str, Any]:
    jobs = list_jobs(jobs_dir=jobs_dir, limit=1)
    if not jobs:
        return {}
    job_id = jobs[0]["job_id"]
    try:
        summary = resume_job(job_id, jobs_dir=jobs_dir)
        if summary.get("status") in {"failed", "partial"}:
            job = get_job(job_id, jobs_dir=jobs_dir)
            record = failure_record_from_job(job)
            summary["failure_record"] = record
            summary["next_actions"] = list(record.get("allowed_actions") or [])
        return summary
    except Exception:
        return jobs[0]


def _startup_next_actions(latest_job: dict[str, Any]) -> list[str]:
    if not latest_job:
        return ["describe benchmark goal", "confirm environment inference"]
    job_id = str(latest_job.get("job_id", "") or "").strip()
    status = latest_job.get("status", "unknown")
    logs = f"logs {job_id}" if job_id else "logs"
    follow = f"follow {job_id}" if job_id else "follow"
    if status == "running":
        return ["status", logs, follow]
    if status == "completed":
        return ["ask: analyze latest job", "ask: show report evidence", "start a new benchmark"]
    if status in {"failed", "partial"}:
        actions = list(latest_job.get("next_actions") or [])
        return actions or [logs, "ask: inspect failure evidence"]
    return ["status", "ask for next action"]
