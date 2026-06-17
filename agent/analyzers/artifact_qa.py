"""Artifact-aware Q&A for completed or mock Agent jobs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from analyzers.chart_explainer import explain_charts, format_chart_explanation


def answer_artifact_question(question: str, job: dict[str, Any] | None = None, artifact_index: str | Path | None = None) -> dict[str, Any]:
    lowered = question.lower()
    if not any(token in lowered for token in ("artifact", "report", "chart", "csv", "empty", "bottleneck", "evidence", "图表", "报告", "瓶颈", "为空")):
        return {
            "intent": "framework_question",
            "answer": "This question does not appear to require artifact inspection.",
            "confidence": 0.2,
            "sources": [],
        }
    index = _load_artifact_index(job, artifact_index)
    evidence = index.get("evidence", {})
    findings = []
    sources = []

    for key, value in evidence.items():
        if not value or value.startswith("<"):
            findings.append(f"{key}: not available")
            continue
        if value.startswith("http://") or value.startswith("https://"):
            findings.append(f"{key}: endpoint configured at {value}")
            sources.append({"path": value, "line": 0, "text": key})
            continue
        path = Path(value)
        sources.append({"path": str(path), "line": 0, "text": key})
        if not path.exists():
            findings.append(f"{key}: missing file {path}")
        elif path.suffix == ".csv":
            findings.append(f"{key}: {_csv_summary(path)}")
        elif path.suffix == ".json":
            findings.append(f"{key}: {_json_summary(path)}")
        elif path.suffix == ".html":
            findings.append(f"{key}: html report exists ({path.stat().st_size} bytes)")
        else:
            findings.append(f"{key}: file exists ({path.stat().st_size} bytes)")

    if not evidence:
        findings.append("No artifact evidence was registered for this job.")
    chart_explanation = explain_charts(evidence)
    answer = (
        "Artifact inspection summary:\n"
        + "\n".join(f"- {item}" for item in findings)
        + "\n\n"
        + format_chart_explanation(chart_explanation)
    )
    return {
        "intent": "framework_question",
        "answer": answer,
        "confidence": 0.85 if evidence else 0.35,
        "sources": sources,
        "artifact_index": index,
        "chart_explanation": chart_explanation,
    }


def _load_artifact_index(job: dict[str, Any] | None, artifact_index: str | Path | None) -> dict[str, Any]:
    if artifact_index:
        path = Path(artifact_index)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    if job and job.get("artifact_index"):
        path = Path(job["artifact_index"])
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    if job:
        return {"job_id": job.get("job_id", ""), "status": job.get("status", ""), "evidence": job.get("artifacts", {})}
    return {"evidence": {}}


def _csv_summary(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        if rows == 0:
            return f"CSV exists but has no data rows ({len(header)} columns)"
        return f"CSV has {rows} data rows and {len(header)} columns"
    except Exception as exc:
        return f"CSV could not be read: {exc}"


def _json_summary(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            keys = ", ".join(list(payload.keys())[:8])
            return f"JSON object exists with keys: {keys}"
        if isinstance(payload, list):
            return f"JSON list exists with {len(payload)} items"
        return "JSON exists"
    except Exception as exc:
        return f"JSON could not be read: {exc}"
