"""File-backed benchmark job lifecycle for Agent runs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import request as urlrequest

from agent.runners.artifacts import write_artifact_index
from agent.runners.guardrails import build_benchmark_command, validate_execution_plan
from agent.runners.materialize import benchmark_subprocess_env, load_runtime_env_file, materialize_runtime_env
from agent.runners.result_status import classify_benchmark_result

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOBS_DIR = Path(
    os.environ.get("ANYCHAIN_AGENT_JOBS_DIR") or REPO_ROOT / ".agent" / "jobs"
).resolve()


def submit_job(
    plan_file: str | Path,
    jobs_dir: str | Path = DEFAULT_JOBS_DIR,
    mock: bool = False,
    approved: bool = False,
    execution_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the sole job-owned execution-plan file and run the job."""

    plan_file = Path(plan_file).resolve()
    source_plan = _read_json(plan_file)
    plan = dict(execution_plan) if execution_plan is not None else source_plan
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    execution_key = str((plan.get("execution") or {}).get("idempotency_key") or "").strip()
    lock_path = jobs_dir / ".submission.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = _find_job_by_execution_key(jobs_dir, execution_key) if execution_key else None
        if existing:
            result = dict(existing)
            result["submission_reused"] = True
            return result

        job_id = _new_job_id()
        run_dir = jobs_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        copied_plan = run_dir / "plan.json"

        provenance = plan.get("execution_provenance")
        if isinstance(provenance, dict):
            provenance = dict(provenance)
            provenance.update({"job_id": job_id, "job_plan_file": str(copied_plan.resolve())})
            plan["execution_provenance"] = provenance
        _write_json(copied_plan, plan)

        job = _job(job_id, plan["plan_id"], "pending", copied_plan, run_dir)
        job["execution_key"] = execution_key
        if isinstance(provenance, dict):
            job["approved_plan_file"] = str(provenance.get("approved_plan_file") or "")
            job["execution_plan_file"] = str(copied_plan.resolve())
        runtime_env_file = materialize_runtime_env(plan, run_dir)
        job["runtime_env_file"] = runtime_env_file
        job["artifacts"]["runtime_env_file"] = runtime_env_file
        _write_json(run_dir / "job.json", job)

    guardrail_errors = [] if mock else validate_execution_plan(plan, approved=approved)
    if guardrail_errors:
        job["status"] = "failed"
        job["updated_at"] = _now()
        job["error"] = "; ".join(guardrail_errors)
        job["artifact_index"] = write_artifact_index(run_dir, job, plan)
        _write_json(run_dir / "job.json", job)
        _notify_status(job)
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
        _notify_status(job)
        return job

    runner_mode = plan.get("execution", {}).get("runner_mode", "foreground")
    job["runner_mode"] = runner_mode
    job["status"] = "running"
    job["updated_at"] = _now()
    _write_json(run_dir / "job.json", job)

    if runner_mode == "detached":
        worker_args = [
            sys.executable,
            "-m",
            "agent.runners.job_worker",
            "--job-file",
            str(run_dir / "job.json"),
        ]
        worker = subprocess.Popen(
            worker_args,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            start_new_session=True,
        )
        job["worker_pid"] = worker.pid
        job["updated_at"] = _now()
        _write_json(run_dir / "job.json", job)
        return job

    try:
        env = load_runtime_env_file(runtime_env_file)
        completed = subprocess.run(
            build_benchmark_command(plan["execution"]["command"]),
            cwd=plan["execution"].get("working_dir", str(REPO_ROOT)),
            env=benchmark_subprocess_env(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (run_dir / "benchmark.log").write_text(completed.stdout, encoding="utf-8")
        job["artifacts"].update(_discover_completed_artifacts(plan, env))
        result = classify_benchmark_result(plan, completed.returncode, job["artifacts"])
        job["status"] = result["status"]
        job["result_validation"] = result
        job["exit_code"] = completed.returncode
        if result["failures"]:
            job["error"] = "; ".join(result["failures"])
    except Exception as exc:  # pragma: no cover - defensive lifecycle guard
        job["status"] = "failed"
        job["error"] = str(exc)

    job["updated_at"] = _now()
    job["artifact_index"] = write_artifact_index(run_dir, job, plan)
    _write_json(run_dir / "job.json", job)
    _notify_status(job)
    return job


def get_job(job_id: str, jobs_dir: str | Path = DEFAULT_JOBS_DIR) -> dict[str, Any]:
    job_file = Path(jobs_dir) / job_id / "job.json"
    if not job_file.is_file():
        raise FileNotFoundError(f"job not found: {job_id}")
    return _legacy_result_view(_read_json(job_file))


def list_jobs(jobs_dir: str | Path = DEFAULT_JOBS_DIR, limit: int = 20) -> list[dict[str, Any]]:
    root = Path(jobs_dir)
    if not root.is_dir():
        return []
    jobs = []
    for job_file in sorted(root.glob("job_*/job.json"), reverse=True):
        try:
            job = _legacy_result_view(_read_json(job_file))
        except Exception:
            continue
        jobs.append({
            "job_id": job.get("job_id", job_file.parent.name),
            "status": job.get("status", "unknown"),
            "created_at": job.get("created_at", ""),
            "updated_at": job.get("updated_at", ""),
            "run_dir": job.get("run_dir", str(job_file.parent)),
            "artifact_index": job.get("artifact_index", ""),
        })
        if len(jobs) >= limit:
            break
    return jobs


def tail_job_log(job_id: str, jobs_dir: str | Path = DEFAULT_JOBS_DIR, lines: int = 80) -> dict[str, Any]:
    job = get_job(job_id, jobs_dir=jobs_dir)
    log_file = Path(job["run_dir"]) / "benchmark.log"
    if not log_file.is_file():
        return {"job_id": job_id, "log_file": str(log_file), "exists": False, "lines": []}
    content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "job_id": job_id,
        "log_file": str(log_file),
        "exists": True,
        "lines": content[-max(1, lines):],
    }


def resume_job(job_id: str, jobs_dir: str | Path = DEFAULT_JOBS_DIR) -> dict[str, Any]:
    job = get_job(job_id, jobs_dir=jobs_dir)
    next_actions = ["status", "analyze", "artifact-qa"]
    if job.get("status") == "running":
        next_actions = ["status", "logs", "wait for completion"]
    if job.get("status") in {"failed", "partial"}:
        next_actions = ["status", "logs", "analyze"]
    return {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "run_dir": job.get("run_dir", ""),
        "plan_file": job.get("plan_file", ""),
        "runtime_env_file": job.get("runtime_env_file", ""),
        "artifact_index": job.get("artifact_index", ""),
        "runner_mode": job.get("runner_mode", ""),
        "worker_pid": job.get("worker_pid", ""),
        "next_actions": next_actions,
    }


def migrate_legacy_job_result(
    job_id: str,
    jobs_dir: str | Path = DEFAULT_JOBS_DIR,
) -> dict[str, Any]:
    """Explicitly persist the evidence-based status fields for one legacy job."""

    job_file = Path(jobs_dir) / job_id / "job.json"
    if not job_file.is_file():
        raise FileNotFoundError(f"job not found: {job_id}")
    original = _read_json(job_file)
    migrated = _legacy_result_view(dict(original))
    if migrated != original:
        original_bytes = job_file.read_bytes()
        backup_file = job_file.with_name("job.json.pre-result-migration-v1.bak")
        if not backup_file.exists():
            backup_file.write_bytes(original_bytes)
        elif backup_file.read_bytes() != original_bytes:
            raise RuntimeError(f"legacy migration backup conflicts with current source: {backup_file}")
        migrated["migration_provenance"] = {
            "operation": "legacy_result_migration",
            "version": 1,
            "migrated_at": _now(),
            "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "backup_file": str(backup_file),
        }
        _atomic_write_json(job_file, migrated)
    return migrated


def _job(job_id: str, plan_id: str, status: str, plan_file: Path, run_dir: Path) -> dict[str, Any]:
    timestamp = _now()
    return {
        "job_id": job_id,
        "plan_id": plan_id,
        "status": status,
        "created_at": timestamp,
        "updated_at": timestamp,
        "plan_file": str(plan_file.resolve()),
        "run_dir": str(run_dir.resolve()),
        "artifacts": {},
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_json(temporary, payload)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_job_id() -> str:
    return f"job_{time.strftime('%Y%m%d%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"


def _find_job_by_execution_key(jobs_dir: Path, execution_key: str) -> dict[str, Any] | None:
    for job_file in sorted(jobs_dir.glob("job_*/job.json"), reverse=True):
        try:
            job = _read_json(job_file)
        except (OSError, json.JSONDecodeError):
            continue
        if str(job.get("execution_key") or "") == execution_key:
            return job
    return None


def _legacy_result_view(job: dict[str, Any]) -> dict[str, Any]:
    """Derive evidence-based legacy status without changing historical files."""

    if job.get("result_validation") or (job.get("artifacts") or {}).get("mode") == "mock":
        return job
    if job.get("status") not in {"completed", "failed"}:
        return job
    plan_file = Path(str(job.get("plan_file") or ""))
    if not plan_file.is_file():
        return job
    try:
        plan = _read_json(plan_file)
        raw_exit_code = job.get("exit_code")
        inferred_exit_code = (
            int(raw_exit_code)
            if isinstance(raw_exit_code, int)
            else (0 if job.get("status") == "completed" else 1)
        )
        result = classify_benchmark_result(plan, inferred_exit_code, dict(job.get("artifacts") or {}))
    except (OSError, ValueError, json.JSONDecodeError):
        return job
    original = str(job.get("status") or "")
    job["result_validation"] = result
    job["status"] = result["status"]
    job["legacy_status"] = original
    if not isinstance(job.get("exit_code"), int):
        job["legacy_exit_code_inferred"] = inferred_exit_code
    if result["failures"]:
        job["error"] = "; ".join(result["failures"])
    return job


def _notify_status(job: dict[str, Any]) -> None:
    webhook = os.environ.get("AGENT_NOTIFY_WEBHOOK_URL", "").strip()
    if not webhook:
        return
    events = {
        item.strip()
        for item in os.environ.get("AGENT_NOTIFY_ON", "completed,failed").split(",")
        if item.strip()
    }
    if job.get("status") not in events:
        return
    payload = json.dumps({
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "updated_at": job.get("updated_at"),
        "run_dir": job.get("run_dir"),
        "artifact_index": job.get("artifact_index"),
        "error": job.get("error", ""),
    }).encode("utf-8")
    req = urlrequest.Request(
        webhook,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=5):  # nosec B310 - user-configured notification endpoint
            pass
    except Exception:
        return


def _discover_completed_artifacts(plan: dict[str, Any], env: dict[str, str]) -> dict[str, str]:
    """Find benchmark artifacts after a successful foreground engine run.

    The benchmark entrypoint archives ``current`` at the end of a successful
    run, so Agent evidence must point at the latest archive instead of stale
    plan-relative ``current/...`` paths.
    """
    data_dir = Path(env.get("BLOCKCHAIN_BENCHMARK_DATA_DIR") or "")
    if not data_dir:
        return {}
    if not data_dir.is_absolute():
        data_dir = (Path(plan.get("execution", {}).get("working_dir", REPO_ROOT)) / data_dir).resolve()
    archive_dir = _latest_archive_dir(data_dir)
    root = archive_dir or data_dir / "current"
    artifacts: dict[str, str] = {}
    if archive_dir:
        artifacts["archive_dir"] = str(archive_dir)
        summary = archive_dir / "test_summary.json"
        if summary.is_file():
            artifacts["summary_json"] = str(summary)
    html = _latest_file(root / "reports", "performance_report_*.html")
    if html:
        artifacts["html_report"] = str(html)
    html_en = _latest_file(root / "reports", "performance_report_en_*.html")
    if html_en:
        artifacts["html_report_en"] = str(html_en)
    html_zh = _latest_file(root / "reports", "performance_report_zh_*.html")
    if html_zh:
        artifacts["html_report_zh"] = str(html_zh)
    sync_timeline = root / "reports" / "sync_execution_timeline.png"
    if sync_timeline.is_file():
        artifacts["sync_timeline_chart"] = str(sync_timeline)
    performance = _latest_file(root / "logs", "performance_*.csv")
    if performance:
        artifacts["performance_csv"] = str(performance)
    proxy = root / "logs" / "proxy_method.csv"
    if proxy.is_file():
        artifacts["proxy_method_csv"] = str(proxy)
    sync_health = _latest_file(root / "logs", "block_height_monitor_*.csv")
    if sync_health:
        artifacts["sync_health_csv"] = str(sync_health)
    vegeta = _latest_file(root / "vegeta_results", "vegeta_*.json")
    if vegeta:
        artifacts["vegeta_json"] = str(vegeta)
    return artifacts


def _latest_archive_dir(data_dir: Path) -> Path | None:
    archives = [path for path in (data_dir / "archives").glob("run_*") if path.is_dir()]
    if not archives:
        return None
    return max(archives, key=lambda path: path.stat().st_mtime)


def _latest_file(root: Path, pattern: str) -> Path | None:
    if not root.is_dir():
        return None
    matches = [path for path in root.glob(pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)
