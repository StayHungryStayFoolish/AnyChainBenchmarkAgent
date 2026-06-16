"""File-backed benchmark job lifecycle for Agent runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from runners.artifacts import write_artifact_index
from runners.guardrails import validate_execution_plan
from runners.materialize import load_runtime_env_file, materialize_runtime_env

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOBS_DIR = REPO_ROOT / ".agent" / "jobs"


def submit_job(
    plan_file: str | Path,
    jobs_dir: str | Path = DEFAULT_JOBS_DIR,
    mock: bool = False,
    approved: bool = False,
) -> dict[str, Any]:
    plan_file = Path(plan_file).resolve()
    plan = _read_json(plan_file)
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job_id = f"job_{int(time.time())}"
    run_dir = jobs_dir / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    copied_plan = run_dir / "plan.json"
    shutil.copy2(plan_file, copied_plan)

    job = _job(job_id, plan["plan_id"], "pending", copied_plan, run_dir)
    runtime_env_file = materialize_runtime_env(plan, run_dir)
    job["runtime_env_file"] = runtime_env_file
    job["artifacts"]["runtime_env_file"] = runtime_env_file
    _write_json(run_dir / "job.json", job)

    guardrail_errors = validate_execution_plan(plan, approved=approved or mock)
    if guardrail_errors:
        job["status"] = "failed"
        job["updated_at"] = _now()
        job["error"] = "; ".join(guardrail_errors)
        job["artifact_index"] = write_artifact_index(run_dir, job, plan)
        _write_json(run_dir / "job.json", job)
        return job

    if mock:
        job["status"] = "completed"
        job["updated_at"] = _now()
        job["artifacts"] = {
            "mode": "mock",
            "html_report": "",
            "archive_dir": "",
            "summary_json": "",
            "runtime_env_file": runtime_env_file,
        }
        job["artifact_index"] = write_artifact_index(run_dir, job, plan)
        job["analysis"] = {
            "summary": f"Mock lifecycle completed for {plan.get('chain', '<unknown>')} using {plan.get('strategy', '<unknown>')} strategy.",
            "recommendations": ["Run a fake-node smoke job next to validate the full execution plane."],
        }
        _write_json(run_dir / "job.json", job)
        return job

    job["status"] = "running"
    job["updated_at"] = _now()
    _write_json(run_dir / "job.json", job)

    try:
        env = load_runtime_env_file(runtime_env_file)
        completed = subprocess.run(
            plan["execution"]["command"],
            cwd=plan["execution"].get("working_dir", str(REPO_ROOT)),
            env={**os.environ, **env},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (run_dir / "benchmark.log").write_text(completed.stdout, encoding="utf-8")
        job["status"] = "completed" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            job["error"] = f"benchmark command exited with {completed.returncode}"
    except Exception as exc:  # pragma: no cover - defensive lifecycle guard
        job["status"] = "failed"
        job["error"] = str(exc)

    job["updated_at"] = _now()
    job["artifact_index"] = write_artifact_index(run_dir, job, plan)
    _write_json(run_dir / "job.json", job)
    return job


def get_job(job_id: str, jobs_dir: str | Path = DEFAULT_JOBS_DIR) -> dict[str, Any]:
    job_file = Path(jobs_dir) / job_id / "job.json"
    if not job_file.is_file():
        raise FileNotFoundError(f"job not found: {job_id}")
    return _read_json(job_file)


def _job(job_id: str, plan_id: str, status: str, plan_file: Path, run_dir: Path) -> dict[str, Any]:
    timestamp = _now()
    return {
        "job_id": job_id,
        "plan_id": plan_id,
        "status": status,
        "created_at": timestamp,
        "updated_at": timestamp,
        "plan_file": str(plan_file),
        "run_dir": str(run_dir),
        "artifacts": {},
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
