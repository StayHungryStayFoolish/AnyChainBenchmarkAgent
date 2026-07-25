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
from agent.runners.execution_scenarios import (
    SYNC_OBSERVE_WORKFLOW,
    scenario_for_operation,
    workflow_type_from_plan,
)
from agent.runners.guardrails import build_benchmark_command, validate_execution_plan
from agent.runners.materialize import benchmark_subprocess_env, load_runtime_env_file, materialize_runtime_env
from agent.runners.result_status import classify_benchmark_result
from agent.utils.redaction import redact

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOBS_DIR = Path(
    os.environ.get("ANYCHAIN_AGENT_JOBS_DIR") or REPO_ROOT / ".agent" / "jobs"
).resolve()
JOB_RECEIPT_VERSION = 1
_JOB_RECEIPT_FIELDS = {
    "job_submission": frozenset(
        {
            "receipt_type",
            "receipt_version",
            "owner",
            "job_id",
            "operation",
            "scenario_id",
            "workflow_type",
            "execution_key_hash",
            "approved_plan_hash",
            "execution_plan_hash",
            "command_hash",
            "disposition",
            "matching_job_count",
            "load_generator",
            "vegeta_allowed",
            "receipt_id",
        }
    ),
    "job_read": frozenset(
        {
            "receipt_type",
            "receipt_version",
            "owner",
            "job_id",
            "source_sha256",
            "persisted_status",
            "observed_status",
            "status_source",
            "workflow_type",
            "artifact_keys",
            "vegeta_artifact_present",
            "submission_receipt_id",
            "receipt_id",
        }
    ),
}


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
        matching_jobs = _jobs_by_execution_key(jobs_dir, execution_key) if execution_key else []
        existing = matching_jobs[0] if matching_jobs else None
        if existing:
            result = dict(existing)
            result["submission_reused"] = True
            persisted_plan = _read_json(
                Path(str(result.get("plan_file") or ""))
            )
            receipts = dict(result.get("execution_receipts") or {})
            receipts["submission_attempt"] = _submission_receipt(
                plan=persisted_plan,
                job_id=str(result.get("job_id") or ""),
                execution_key=execution_key,
                disposition="reused",
                matching_job_count=len(matching_jobs),
            )
            result["execution_receipts"] = receipts
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
        job["execution_receipts"] = {
            "submission": _submission_receipt(
                plan=plan,
                job_id=job_id,
                execution_key=execution_key,
                disposition="created",
                matching_job_count=1,
            )
        }
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
    source_bytes = job_file.read_bytes()
    persisted = json.loads(source_bytes)
    persisted_status = str(persisted.get("status") or "unknown")
    job = _legacy_result_view(dict(persisted))
    plan = _read_job_plan(job)
    receipts = dict(job.get("execution_receipts") or {})
    receipts["last_read"] = _job_read_receipt(
        job=job,
        plan=plan,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        persisted_status=persisted_status,
    )
    job["execution_receipts"] = receipts
    return job


def list_jobs(jobs_dir: str | Path = DEFAULT_JOBS_DIR, limit: int = 20) -> list[dict[str, Any]]:
    root = Path(jobs_dir)
    if not root.is_dir():
        return []
    jobs = []
    for job_file in sorted(root.glob("job_*/job.json"), reverse=True):
        try:
            job = get_job(job_file.parent.name, jobs_dir=root)
        except Exception:
            continue
        jobs.append({
            "job_id": job.get("job_id", job_file.parent.name),
            "status": job.get("status", "unknown"),
            "created_at": job.get("created_at", ""),
            "updated_at": job.get("updated_at", ""),
            "run_dir": job.get("run_dir", str(job_file.parent)),
            "artifact_index": job.get("artifact_index", ""),
            "execution_receipts": {"last_read": _last_read_receipt(job)},
        })
        if len(jobs) >= limit:
            break
    return jobs


def tail_job_log(job_id: str, jobs_dir: str | Path = DEFAULT_JOBS_DIR, lines: int = 80) -> dict[str, Any]:
    job = get_job(job_id, jobs_dir=jobs_dir)
    log_file = Path(job["run_dir"]) / "benchmark.log"
    if not log_file.is_file():
        return {
            "job_id": job_id,
            "log_file": str(log_file),
            "exists": False,
            "lines": [],
            "job_read_receipt": _last_read_receipt(job),
        }
    content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "job_id": job_id,
        "log_file": str(log_file),
        "exists": True,
        "lines": content[-max(1, lines):],
        "job_read_receipt": _last_read_receipt(job),
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
        "job_read_receipt": _last_read_receipt(job),
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
    target = Path(path)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_job_id() -> str:
    return f"job_{time.strftime('%Y%m%d%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"


def _jobs_by_execution_key(jobs_dir: Path, execution_key: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for job_file in sorted(jobs_dir.glob("job_*/job.json"), reverse=True):
        try:
            job = _read_json(job_file)
        except (OSError, json.JSONDecodeError):
            continue
        if str(job.get("execution_key") or "") == execution_key:
            matches.append(job)
    return matches


def verify_job_receipt(receipt: dict[str, Any]) -> bool:
    """Validate a manager-issued receipt without consulting model output."""

    if not isinstance(receipt, dict):
        return False
    receipt_type = receipt.get("receipt_type")
    expected_fields = _JOB_RECEIPT_FIELDS.get(str(receipt_type or ""))
    if (
        expected_fields is None
        or set(receipt) != set(expected_fields)
        or receipt.get("owner") != "job_manager"
        or receipt.get("receipt_version") != JOB_RECEIPT_VERSION
        or not str(receipt.get("job_id") or "")
    ):
        return False
    if receipt_type == "job_submission":
        if (
            receipt.get("disposition") not in {"created", "reused"}
            or not isinstance(receipt.get("matching_job_count"), int)
            or isinstance(receipt.get("matching_job_count"), bool)
            or int(receipt["matching_job_count"]) < 1
            or not _is_sha256(receipt.get("execution_key_hash"))
            or not _is_sha256(receipt.get("approved_plan_hash"))
            or not _is_sha256(receipt.get("execution_plan_hash"))
            or not _is_sha256(receipt.get("command_hash"))
            or receipt.get("load_generator") not in {"none", "vegeta"}
            or not isinstance(receipt.get("vegeta_allowed"), bool)
        ):
            return False
        workflow_type = str(receipt.get("workflow_type") or "")
        if workflow_type == SYNC_OBSERVE_WORKFLOW:
            if receipt.get("load_generator") != "none" or receipt.get("vegeta_allowed"):
                return False
        elif receipt.get("load_generator") != "vegeta" or not receipt.get(
            "vegeta_allowed"
        ):
            return False
    if receipt_type == "job_read":
        if (
            not _is_sha256(receipt.get("source_sha256"))
            or not str(receipt.get("persisted_status") or "")
            or not str(receipt.get("observed_status") or "")
            or receipt.get("status_source")
            not in {"persisted_job", "evidence_derived_legacy"}
            or not isinstance(receipt.get("artifact_keys"), list)
            or any(not isinstance(item, str) for item in receipt.get("artifact_keys") or ())
            or not isinstance(receipt.get("vegeta_artifact_present"), bool)
            or (
                receipt.get("submission_receipt_id")
                and not _is_sha256(receipt.get("submission_receipt_id"))
            )
        ):
            return False
    receipt_id = str(receipt.get("receipt_id") or "")
    if not _is_sha256(receipt_id):
        return False
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    return receipt_id == _evidence_hash(unsigned)


def _submission_receipt(
    *,
    plan: dict[str, Any],
    job_id: str,
    execution_key: str,
    disposition: str,
    matching_job_count: int,
) -> dict[str, Any]:
    provenance = dict(plan.get("execution_provenance") or {})
    operation = str(provenance.get("operation") or "")
    workflow_type = workflow_type_from_plan(plan)
    scenario_id = ""
    if operation:
        try:
            scenario_id = scenario_for_operation(operation, plan).scenario_id
        except ValueError:
            scenario_id = ""
    body = {
        "receipt_type": "job_submission",
        "receipt_version": JOB_RECEIPT_VERSION,
        "owner": "job_manager",
        "job_id": job_id,
        "operation": operation,
        "scenario_id": scenario_id,
        "workflow_type": workflow_type,
        "execution_key_hash": _evidence_hash(execution_key),
        "approved_plan_hash": str(
            provenance.get("approved_plan_hash")
            or _evidence_hash(plan)
        ),
        "execution_plan_hash": _evidence_hash(plan),
        "command_hash": _evidence_hash(
            redact((plan.get("execution") or {}).get("command") or [])
        ),
        "disposition": disposition,
        "matching_job_count": int(matching_job_count),
        "load_generator": "none" if workflow_type == SYNC_OBSERVE_WORKFLOW else "vegeta",
        "vegeta_allowed": workflow_type != SYNC_OBSERVE_WORKFLOW,
    }
    body["receipt_id"] = _evidence_hash(body)
    return body


def _job_read_receipt(
    *,
    job: dict[str, Any],
    plan: dict[str, Any],
    source_sha256: str,
    persisted_status: str,
) -> dict[str, Any]:
    observed_status = str(job.get("status") or "unknown")
    workflow_type = workflow_type_from_plan(plan) if plan else ""
    artifacts = job.get("artifacts") if isinstance(job.get("artifacts"), dict) else {}
    submission = (job.get("execution_receipts") or {}).get("submission")
    body = {
        "receipt_type": "job_read",
        "receipt_version": JOB_RECEIPT_VERSION,
        "owner": "job_manager",
        "job_id": str(job.get("job_id") or ""),
        "source_sha256": source_sha256,
        "persisted_status": persisted_status,
        "observed_status": observed_status,
        "status_source": (
            "evidence_derived_legacy"
            if observed_status != persisted_status
            else "persisted_job"
        ),
        "workflow_type": workflow_type,
        "artifact_keys": sorted(str(key) for key in artifacts),
        "vegeta_artifact_present": bool(artifacts.get("vegeta_json")),
        "submission_receipt_id": (
            str(submission.get("receipt_id") or "")
            if isinstance(submission, dict)
            else ""
        ),
    }
    body["receipt_id"] = _evidence_hash(body)
    return body


def _read_job_plan(job: dict[str, Any]) -> dict[str, Any]:
    plan_file = Path(str(job.get("plan_file") or ""))
    if not plan_file.is_file():
        return {}
    try:
        return _read_json(plan_file)
    except (OSError, json.JSONDecodeError):
        return {}


def _last_read_receipt(job: dict[str, Any]) -> dict[str, Any]:
    receipts = job.get("execution_receipts")
    if not isinstance(receipts, dict):
        return {}
    receipt = receipts.get("last_read")
    return dict(receipt) if isinstance(receipt, dict) else {}


def _evidence_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


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
