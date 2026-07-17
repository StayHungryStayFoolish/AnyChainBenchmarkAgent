"""Truthful business-result classification for benchmark jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def classify_benchmark_result(
    plan: dict[str, Any],
    returncode: int,
    artifacts: dict[str, str],
) -> dict[str, Any]:
    """Classify a finished process using benchmark evidence, not exit code alone."""

    execution_failures: list[str] = []
    artifact_failures: list[str] = []
    failure_facts: list[dict[str, Any]] = []
    if returncode != 0:
        execution_failures.append(f"benchmark command exited with {returncode}")
        failure_facts.append({
            "code": "WORKLOAD_PROCESS_FAILED",
            "source": "workload",
            "returncode": returncode,
            "detail": f"benchmark command exited with {returncode}",
        })

    workflow = str(plan.get("workflow_type") or plan.get("run_mode") or "rpc_benchmark").replace("-", "_")
    if workflow == "sync_observe":
        _require_artifacts(artifacts, ("performance_csv", "html_report"), artifact_failures, failure_facts)
    else:
        _require_artifacts(
            artifacts,
            ("archive_dir", "summary_json", "performance_csv", "html_report", "proxy_method_csv", "vegeta_json"),
            artifact_failures,
            failure_facts,
        )
        vegeta_path = str(artifacts.get("vegeta_json") or "")
        if vegeta_path:
            _validate_vegeta(Path(vegeta_path), execution_failures, failure_facts)

    status = "failed" if execution_failures else "partial" if artifact_failures else "completed"
    failures = execution_failures + artifact_failures

    return {
        "status": status,
        "passed": status == "completed",
        "executed": True,
        "workload_passed": not execution_failures,
        "artifacts_complete": not artifact_failures,
        "execution_failures": execution_failures,
        "artifact_failures": artifact_failures,
        "failure_facts": failure_facts,
        "failures": failures,
        "process_exit_code": returncode,
    }


def _require_artifacts(
    artifacts: dict[str, str],
    names: tuple[str, ...],
    failures: list[str],
    facts: list[dict[str, Any]],
) -> None:
    for name in names:
        raw = str(artifacts.get(name) or "")
        if not raw or not _artifact_path(raw).exists():
            failures.append(f"required artifact missing: {name}")
            facts.append({
                "code": "ARTIFACT_INCOMPLETE",
                "source": "artifact",
                "artifact": name,
                "detail": f"required artifact missing: {name}",
            })


def _validate_vegeta(path: Path, failures: list[str], facts: list[dict[str, Any]]) -> None:
    path = _artifact_path(str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid vegeta result: {type(exc).__name__}")
        facts.append({
            "code": "ARTIFACT_INCOMPLETE",
            "source": "artifact",
            "artifact": "vegeta_json",
            "detail": f"invalid vegeta result: {type(exc).__name__}",
        })
        return
    requests = int(payload.get("requests") or 0)
    success = float(payload.get("success") or 0)
    if requests <= 0:
        failures.append("vegeta executed zero requests")
        facts.append({
            "code": "WORKLOAD_NO_REQUESTS",
            "source": "workload",
            "requests": requests,
            "detail": "vegeta executed zero requests",
        })
    if success <= 0:
        codes = payload.get("status_codes") or {}
        failures.append(f"vegeta recorded zero successful requests; status_codes={codes}")
        facts.append({
            "code": "WORKLOAD_ZERO_SUCCESS",
            "source": "workload",
            "requests": requests,
            "success": success,
            "status_codes": codes,
            "detail": f"vegeta recorded zero successful requests; status_codes={codes}",
        })


def _artifact_path(raw: str) -> Path:
    """Resolve persisted container paths when inspecting the shared repository."""

    path = Path(raw)
    if path.exists() or not path.is_absolute():
        return path
    try:
        relative = path.relative_to("/workspace")
    except ValueError:
        return path
    return REPO_ROOT / relative
