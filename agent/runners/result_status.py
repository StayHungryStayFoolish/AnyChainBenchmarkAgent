"""Truthful business-result classification for benchmark jobs."""

from __future__ import annotations

import json
import csv
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
        _require_artifacts(
            artifacts,
            (
                "summary_json",
                "performance_csv",
                "html_report_en",
                "html_report_zh",
                "sync_timeline_chart",
                "sync_health_csv",
            ),
            artifact_failures,
            failure_facts,
        )
        for forbidden in ("vegeta_json", "proxy_method_csv"):
            if str(artifacts.get(forbidden) or ""):
                artifact_failures.append(f"sync-observe produced forbidden artifact: {forbidden}")
                failure_facts.append({
                    "code": "WORKFLOW_ARTIFACT_CONTAMINATION",
                    "source": "artifact",
                    "artifact": forbidden,
                    "detail": f"sync-observe produced forbidden artifact: {forbidden}",
                })
        performance_csv = str(artifacts.get("performance_csv") or "")
        if performance_csv:
            _validate_sync_observe_csv(
                _artifact_path(performance_csv),
                artifact_failures,
                failure_facts,
            )
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


def _validate_sync_observe_csv(
    path: Path,
    failures: list[str],
    facts: list[dict[str, Any]],
) -> None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        rows = []
        detail = f"invalid sync-observe performance CSV: {type(exc).__name__}"
        failures.append(detail)
        facts.append({
            "code": "SYNC_OBSERVE_DATA_INVALID",
            "source": "artifact",
            "artifact": "performance_csv",
            "detail": detail,
        })
        return
    required = {
        "timestamp",
        "cpu_usage",
        "mem_usage",
        "net_total_mbps",
        "local_block_height",
        "sync_status",
        "execution_mgas_per_sec",
        "execution_metric_source",
        "execution_metric_status",
        "current_qps",
        "qps_data_available",
    }
    columns = set(rows[0]) if rows else set()
    missing = sorted(required - columns)
    if len(rows) < 2 or missing:
        detail = (
            "sync-observe performance CSV requires at least two rows and columns: "
            + ", ".join(sorted(required))
        )
        failures.append(detail)
        facts.append({
            "code": "SYNC_OBSERVE_DATA_INCOMPLETE",
            "source": "artifact",
            "artifact": "performance_csv",
            "detail": detail,
            "observed_rows": len(rows),
            "missing_columns": missing,
        })
        return
    if not any(column.endswith("_total_iops") for column in columns) or not any(
        column.endswith("_avg_await") for column in columns
    ):
        detail = "sync-observe performance CSV has no disk IOPS/latency evidence"
        failures.append(detail)
        facts.append({
            "code": "SYNC_OBSERVE_DATA_INCOMPLETE",
            "source": "artifact",
            "artifact": "performance_csv",
            "detail": detail,
        })
        return
    for row in rows:
        if str(row.get("current_qps") or "").strip() not in {"0", "0.0", "0.00"}:
            detail = "sync-observe reported a non-zero QPS workload"
            failures.append(detail)
            facts.append({
                "code": "SYNC_OBSERVE_RPC_WORKLOAD_DETECTED",
                "source": "artifact",
                "artifact": "performance_csv",
                "detail": detail,
            })
            return
        if str(row.get("qps_data_available") or "").strip().lower() not in {"false", "0"}:
            detail = "sync-observe incorrectly reports QPS data as available"
            failures.append(detail)
            facts.append({
                "code": "SYNC_OBSERVE_RPC_WORKLOAD_DETECTED",
                "source": "artifact",
                "artifact": "performance_csv",
                "detail": detail,
            })
            return
        status = str(row.get("execution_metric_status") or "").strip().lower()
        source = str(row.get("execution_metric_source") or "").strip()
        if status not in {"available", "unavailable"} or not source:
            detail = "sync-observe MGas provenance is ambiguous"
            failures.append(detail)
            facts.append({
                "code": "SYNC_OBSERVE_MGAS_PROVENANCE_INVALID",
                "source": "artifact",
                "artifact": "performance_csv",
                "detail": detail,
            })
            return


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
