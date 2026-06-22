"""Execution artifact diagnosis for smoke and benchmark jobs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def diagnose_execution_artifacts(job: dict[str, Any] | None = None, artifact_index: str | Path | None = None) -> dict[str, Any]:
    index = _load_artifact_index(job, artifact_index)
    evidence = index.get("evidence", {})
    proxy = _path(evidence.get("proxy_method_csv", ""))
    performance = _path(evidence.get("performance_csv", ""))
    html = _path(evidence.get("html_report", ""))
    benchmark_log = _path(Path(index.get("run_dir", "")) / "benchmark.log" if index.get("run_dir") else "")

    proxy_rows = _csv_rows(proxy)
    performance_rows = _csv_rows(performance)
    html_exists = bool(html and html.is_file())
    log_exists = bool(benchmark_log and benchmark_log.is_file())

    if proxy_rows > 0 and performance_rows < 0:
        conclusion = "traffic_ok_monitor_sample_missing"
        summary = "RPC traffic evidence exists, but performance_latest.csv is missing."
    elif proxy_rows > 0 and performance_rows == 0:
        conclusion = "traffic_ok_monitor_sample_empty"
        summary = "RPC traffic evidence exists, but performance CSV has no data rows."
    elif proxy_rows > 0 and html_exists:
        conclusion = "traffic_ok_report_ok"
        summary = "RPC traffic and report evidence exist."
    elif proxy_rows > 0:
        conclusion = "traffic_ok_report_unknown"
        summary = "RPC traffic evidence exists; report evidence is missing or unavailable."
    elif proxy_rows == 0:
        conclusion = "traffic_failed_proxy_empty"
        summary = "proxy_method.csv exists but has no request rows."
    else:
        conclusion = "unknown_needs_log_review"
        summary = "No proxy_method.csv evidence was available; inspect benchmark logs."

    return {
        "conclusion": conclusion,
        "summary": summary,
        "signals": {
            "proxy_method_rows": proxy_rows,
            "performance_rows": performance_rows,
            "html_report_exists": html_exists,
            "benchmark_log_exists": log_exists,
        },
        "evidence_paths": {
            "proxy_method_csv": str(proxy) if proxy else "",
            "performance_csv": str(performance) if performance else "",
            "html_report": str(html) if html else "",
            "benchmark_log": str(benchmark_log) if benchmark_log else "",
        },
    }


def _load_artifact_index(job: dict[str, Any] | None, artifact_index: str | Path | None) -> dict[str, Any]:
    if artifact_index and Path(artifact_index).is_file():
        return json.loads(Path(artifact_index).read_text(encoding="utf-8"))
    if job and job.get("artifact_index") and Path(job["artifact_index"]).is_file():
        return json.loads(Path(job["artifact_index"]).read_text(encoding="utf-8"))
    return {"run_dir": job.get("run_dir", "") if job else "", "evidence": job.get("artifacts", {}) if job else {}}


def _path(value: object) -> Path | None:
    if not value:
        return None
    text = str(value)
    if text.startswith("<") or text.startswith("http://") or text.startswith("https://"):
        return None
    return Path(text)


def _csv_rows(path: Path | None) -> int:
    if not path or not path.is_file():
        return -1
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)
    except Exception:
        return -1
