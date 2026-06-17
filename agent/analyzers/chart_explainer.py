"""Explain report charts from available artifact evidence."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


CHART_DEPENDENCIES = {
    "performance_overview": {
        "label": "Performance overview",
        "needs": ["performance_csv"],
        "columns": ("cpu", "memory", "iops", "throughput", "util"),
        "meaning": "Shows CPU, memory, disk IOPS/throughput, and disk utilization over time.",
    },
    "cpu_disk_correlation": {
        "label": "CPU-disk correlation",
        "needs": ["performance_csv"],
        "columns": ("cpu", "iowait", "await", "aqu", "util"),
        "meaning": "Helps decide whether CPU pressure is actually disk wait or queueing.",
    },
    "disk_thresholds": {
        "label": "Disk threshold charts",
        "needs": ["performance_csv"],
        "columns": ("util", "await", "iops", "throughput"),
        "meaning": "Compares observed disk utilization and latency against configured baselines.",
    },
    "per_method_attribution": {
        "label": "Per-method RPC attribution",
        "needs": ["proxy_method_csv"],
        "columns": ("method", "status", "latency", "p50", "p90", "p99"),
        "meaning": "Shows success/failure counts and latency percentiles by configured workload RPC method.",
    },
    "sync_health": {
        "label": "Block-height / sync-health",
        "needs": ["sync_health_csv"],
        "columns": ("height", "lag", "diff", "health", "sync"),
        "meaning": "Shows whether the node stayed close enough to chain tip or reported healthy sync state.",
    },
    "monitoring_overhead": {
        "label": "Monitoring overhead",
        "needs": ["performance_csv"],
        "columns": ("monitor", "overhead"),
        "meaning": "Shows whether the benchmark framework itself consumed meaningful CPU or memory.",
    },
}


def explain_charts(evidence: dict[str, str]) -> dict[str, Any]:
    rows = []
    for chart_id, spec in CHART_DEPENDENCIES.items():
        missing_files = []
        row_counts = []
        matching_columns: set[str] = set()
        for evidence_key in spec["needs"]:
            value = evidence.get(evidence_key, "")
            if not value or value.startswith("<"):
                missing_files.append(evidence_key)
                continue
            path = Path(value)
            if not path.is_file():
                missing_files.append(evidence_key)
                continue
            row_count, header = _csv_shape(path) if path.suffix == ".csv" else (1, [])
            row_counts.append(row_count)
            matching_columns.update(_matching_columns(header, spec["columns"]))
        if missing_files:
            status = "missing_input"
            reason = "Missing required artifact(s): " + ", ".join(missing_files)
        elif row_counts and max(row_counts) == 0:
            status = "empty_input"
            reason = "Required CSV exists but has no data rows."
        elif not matching_columns:
            status = "partial_or_unavailable"
            reason = "Input exists, but expected chart columns were not found."
        else:
            status = "available"
            reason = "Input data and expected columns are available."
        rows.append({
            "chart_id": chart_id,
            "label": spec["label"],
            "status": status,
            "meaning": spec["meaning"],
            "reason": reason,
            "matched_columns": sorted(matching_columns),
        })
    return {
        "charts": rows,
        "summary": _summary(rows),
    }


def format_chart_explanation(explanation: dict[str, Any]) -> str:
    lines = ["Report chart explanation:", f"- summary: {explanation['summary']}"]
    for chart in explanation["charts"]:
        lines.append(
            f"- {chart['label']}: {chart['status']} - {chart['reason']} "
            f"Meaning: {chart['meaning']}"
        )
    return "\n".join(lines)


def _csv_shape(path: Path) -> tuple[int, list[str]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        return rows, header
    except Exception:
        return 0, []


def _matching_columns(header: list[str], expected_tokens: tuple[str, ...]) -> set[str]:
    matches = set()
    lowered = [column.lower() for column in header]
    for original, column in zip(header, lowered):
        if any(token in column for token in expected_tokens):
            matches.add(original)
    return matches


def _summary(rows: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
