"""Artifact index helpers for Agent jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from planners.strategy_planner import write_json


def build_artifact_index(job: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    artifacts = job.get("artifacts", {})
    plan_artifacts = plan.get("artifacts", {})
    return {
        "job_id": job["job_id"],
        "plan_id": job["plan_id"],
        "status": job["status"],
        "run_dir": job.get("run_dir", ""),
        "evidence": {
            "html_report": artifacts.get("html_report", ""),
            "archive_summary": artifacts.get("summary_json", ""),
            "proxy_method_csv": artifacts.get("proxy_method_csv", plan_artifacts.get("proxy_method_csv", "")),
            "performance_csv": artifacts.get("performance_csv", plan_artifacts.get("performance_latest_csv", "")),
            "sync_health_csv": artifacts.get("sync_health_csv", ""),
            "runtime_env_file": artifacts.get("runtime_env_file", job.get("runtime_env_file", "")),
            "prometheus_url": artifacts.get("prometheus_url", ""),
            "grafana_url": artifacts.get("grafana_url", ""),
            "exporter_url": artifacts.get("exporter_url", ""),
        },
    }


def write_artifact_index(run_dir: str | Path, job: dict[str, Any], plan: dict[str, Any]) -> str:
    path = Path(run_dir) / "artifact_index.json"
    write_json(path, build_artifact_index(job, plan))
    return str(path)
