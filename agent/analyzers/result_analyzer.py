"""Analyze Agent job artifacts."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def analyze_job(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("analysis"):
        grade, grade_reason = _grade_job(job, _evidence_from_artifacts(job.get("artifacts", {})))
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "grade": grade,
            "grade_reason": grade_reason,
            "summary": job["analysis"],
            "artifacts": job.get("artifacts", {}),
            "evidence": _evidence_from_artifacts(job.get("artifacts", {})),
            "recommendations": job["analysis"].get("recommendations", []),
        }

    latest_summary = _latest_archive_summary()
    if latest_summary:
        summary = _read_json(latest_summary)
        grade = "WARNING" if summary.get("bottleneck_detected") else "PASS"
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "grade": grade,
            "grade_reason": "Bottleneck detected." if summary.get("bottleneck_detected") else "Archive summary is available.",
            "summary": {
                "run_id": summary.get("run_id"),
                "benchmark_mode": summary.get("benchmark_mode"),
                "max_stable_qps": summary.get("max_successful_qps"),
                "bottleneck_detected": summary.get("bottleneck_detected"),
                "bottleneck_types": summary.get("bottleneck_types", []),
            },
            "artifacts": {"summary_json": str(latest_summary)},
            "evidence": {"archive_summary": str(latest_summary)},
            "recommendations": _recommendations(summary),
        }

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "grade": "INCONCLUSIVE",
        "grade_reason": "No archive summary or evidence artifacts were found.",
        "summary": {"message": "No archive summary found yet."},
        "artifacts": job.get("artifacts", {}),
        "evidence": _evidence_from_artifacts(job.get("artifacts", {})),
        "recommendations": ["Run a benchmark job or point the Agent to an archived run."],
    }


def _latest_archive_summary() -> Path | None:
    matches = glob.glob(str(REPO_ROOT / "archives" / "*" / "test_summary.json"))
    if not matches:
        return None
    return Path(max(matches, key=lambda item: Path(item).stat().st_mtime))


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _recommendations(summary: dict[str, Any]) -> list[str]:
    if summary.get("bottleneck_detected"):
        return ["Confirm the detected bottleneck with a focused follow-up run before changing hardware."]
    return ["Use a ramp strategy if the next goal is to find maximum stable QPS."]


def _evidence_from_artifacts(artifacts: dict[str, Any]) -> dict[str, str]:
    mapping = {
        "html_report": "html_report",
        "summary_json": "archive_summary",
        "proxy_method_csv": "proxy_method_csv",
        "performance_csv": "performance_csv",
        "sync_health_csv": "sync_health_csv",
        "runtime_env_file": "runtime_env_file",
        "prometheus_url": "prometheus_url",
        "grafana_url": "grafana_url",
        "exporter_url": "exporter_url",
    }
    return {
        evidence_key: str(artifacts[artifact_key])
        for artifact_key, evidence_key in mapping.items()
        if artifacts.get(artifact_key)
    }


def _grade_job(job: dict[str, Any], evidence: dict[str, str]) -> tuple[str, str]:
    if job.get("status") == "failed":
        return "FAIL", job.get("error", "Benchmark job failed.")
    if job.get("artifacts", {}).get("mode") == "mock":
        return "WARNING", "Mock lifecycle completed; no real benchmark evidence was produced."
    if not evidence:
        return "INCONCLUSIVE", "No evidence artifacts were found."
    return "PASS", "Job completed with evidence artifacts."
