"""Rule-based bottleneck diagnostics from benchmark artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def diagnose_artifacts(job: dict[str, Any] | None = None, artifact_index: str | Path | None = None) -> dict[str, Any]:
    index = _load_artifact_index(job, artifact_index)
    evidence = index.get("evidence", {})
    performance = _read_numeric_csv(evidence.get("performance_csv", ""))
    proxy = _read_csv(evidence.get("proxy_method_csv", ""))
    sync = _read_csv(evidence.get("sync_health_csv", ""))

    findings: list[dict[str, Any]] = []
    findings.extend(_diagnose_system(performance))
    findings.extend(_diagnose_proxy(proxy))
    findings.extend(_diagnose_sync(sync))

    if not findings:
        findings.append({
            "severity": "info",
            "category": "evidence",
            "message": "No bottleneck rules fired. Required artifacts may be missing or too sparse.",
            "evidence": {
                "performance_rows": len(performance["rows"]),
                "proxy_rows": len(proxy["rows"]),
                "sync_rows": len(sync["rows"]),
            },
        })
    return {
        "summary": _summary(findings),
        "findings": findings,
        "artifact_index": index,
    }


def format_diagnostics(payload: dict[str, Any]) -> str:
    lines = ["Bottleneck diagnostics:", f"- summary: {payload['summary']}"]
    for item in payload["findings"]:
        lines.append(f"- [{item['severity']}] {item['category']}: {item['message']}")
    return "\n".join(lines)


def _diagnose_system(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows = table["rows"]
    if not rows:
        return []
    cpu = _series(table, ("cpu_usage", "cpu_percent", "cpu_util"))
    iowait = _series(table, ("iowait", "io_wait", "cpu_iowait"))
    await_ms = _series(table, ("await", "avg_await", "disk_await", "data_avg_await", "accounts_avg_await"))
    util = _series(table, ("util", "disk_util", "data_util", "accounts_util"))
    iops = _series(table, ("iops", "total_iops", "read_iops", "write_iops"))
    throughput = _series(table, ("throughput", "mbps", "mib", "total_throughput"))

    findings: list[dict[str, Any]] = []
    cpu_avg = _avg(cpu)
    iowait_avg = _avg(iowait)
    await_avg = _avg(await_ms)
    util_avg = _avg(util)

    if cpu_avg >= 85 and iowait_avg >= 10 and (await_avg >= 20 or util_avg >= 80):
        findings.append(_finding(
            "critical",
            "disk_latency",
            "CPU is high, but IO wait and disk latency/utilization indicate a disk bottleneck rather than pure CPU saturation.",
            cpu_avg=cpu_avg,
            iowait_avg=iowait_avg,
            await_ms_avg=await_avg,
            disk_util_avg=util_avg,
        ))
    elif cpu_avg >= 85 and iowait_avg < 8 and util_avg < 75:
        findings.append(_finding(
            "warning",
            "cpu",
            "CPU utilization is high while IO wait and disk utilization are low; the node may be CPU-bound.",
            cpu_avg=cpu_avg,
            iowait_avg=iowait_avg,
            disk_util_avg=util_avg,
        ))
    if util_avg >= 90 and await_avg >= 20:
        findings.append(_finding(
            "critical",
            "disk_queueing",
            "Disk utilization and average await are both high; queueing latency is likely affecting RPC latency.",
            disk_util_avg=util_avg,
            await_ms_avg=await_avg,
        ))
    if _avg(iops) > 0 and util_avg >= 85:
        findings.append(_finding(
            "warning",
            "disk_iops",
            "Disk IOPS are active while utilization is high. Compare this against the configured disk IOPS baseline.",
            iops_avg=_avg(iops),
            disk_util_avg=util_avg,
        ))
    if _avg(throughput) > 0 and util_avg >= 85:
        findings.append(_finding(
            "warning",
            "disk_throughput",
            "Disk throughput is active while utilization is high. Compare this against the configured throughput baseline.",
            throughput_avg=_avg(throughput),
            disk_util_avg=util_avg,
        ))
    return findings


def _diagnose_proxy(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows = table["rows"]
    if not rows:
        return []
    method_stats: dict[str, dict[str, float]] = {}
    for row in rows:
        method = str(row.get("method") or row.get("method_name") or row.get("rpc_method") or "unknown")
        status_text = str(row.get("status") or row.get("status_code") or row.get("success") or "").lower()
        latency = _to_float(row.get("latency_ms") or row.get("duration_ms") or row.get("p99_ms") or row.get("latency"))
        stats = method_stats.setdefault(method, {"total": 0, "fail": 0, "latency_sum": 0, "latency_count": 0})
        stats["total"] += 1
        if status_text in {"false", "failed", "error"} or status_text.startswith(("4", "5")):
            stats["fail"] += 1
        if latency is not None:
            stats["latency_sum"] += latency
            stats["latency_count"] += 1
    findings = []
    for method, stats in sorted(method_stats.items()):
        total = max(1, stats["total"])
        fail_rate = stats["fail"] / total
        latency_avg = stats["latency_sum"] / stats["latency_count"] if stats["latency_count"] else 0
        if fail_rate >= 0.05:
            findings.append(_finding(
                "critical" if fail_rate >= 0.2 else "warning",
                "rpc_errors",
                f"RPC method {method} has an elevated failure rate.",
                method=method,
                failure_rate=round(fail_rate, 4),
                total_requests=int(stats["total"]),
            ))
        if latency_avg >= 1000:
            findings.append(_finding(
                "warning",
                "rpc_latency",
                f"RPC method {method} has high average response latency.",
                method=method,
                latency_ms_avg=round(latency_avg, 2),
            ))
    return findings


def _diagnose_sync(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows = table["rows"]
    if not rows:
        return []
    bad = 0
    lag_values = []
    for row in rows:
        status = str(row.get("health") or row.get("sync_status") or row.get("status") or "").lower()
        lag = _to_float(row.get("height_diff") or row.get("lag_value") or row.get("sync_lag"))
        if status in {"warning", "unhealthy", "behind", "false", "failed"}:
            bad += 1
        if lag is not None:
            lag_values.append(lag)
    findings = []
    if bad:
        findings.append(_finding(
            "warning",
            "sync_health",
            "Sync-health reported warning or unhealthy samples.",
            unhealthy_samples=bad,
            total_samples=len(rows),
        ))
    if lag_values and max(lag_values) > 0:
        findings.append(_finding(
            "info",
            "sync_lag",
            "Sync lag was observed. Compare max lag with BLOCK_HEIGHT_DIFF_THRESHOLD and duration with BLOCK_HEIGHT_TIME_THRESHOLD.",
            max_lag=max(lag_values),
            avg_lag=round(mean(lag_values), 2),
        ))
    return findings


def _load_artifact_index(job: dict[str, Any] | None, artifact_index: str | Path | None) -> dict[str, Any]:
    if artifact_index and Path(artifact_index).is_file():
        return json.loads(Path(artifact_index).read_text(encoding="utf-8"))
    if job and job.get("artifact_index") and Path(job["artifact_index"]).is_file():
        return json.loads(Path(job["artifact_index"]).read_text(encoding="utf-8"))
    if job:
        return {"job_id": job.get("job_id", ""), "status": job.get("status", ""), "evidence": job.get("artifacts", {})}
    return {"evidence": {}}


def _read_csv(path_value: str) -> dict[str, Any]:
    if not path_value or path_value.startswith("<") or path_value.startswith("http"):
        return {"header": [], "rows": []}
    path = Path(path_value)
    if not path.is_file():
        return {"header": [], "rows": []}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        return {"header": reader.fieldnames or [], "rows": list(reader)}


def _read_numeric_csv(path_value: str) -> dict[str, Any]:
    table = _read_csv(path_value)
    rows = []
    for row in table["rows"]:
        rows.append({key: _to_float(value) for key, value in row.items()})
    return {"header": table["header"], "rows": rows}


def _series(table: dict[str, Any], tokens: tuple[str, ...]) -> list[float]:
    columns = _matching_columns(table.get("header", []), tokens)
    values: list[float] = []
    for row in table["rows"]:
        for key in columns:
            value = row.get(key)
            if value is None:
                continue
            values.append(value)
    return values


def _matching_columns(header: list[str], tokens: tuple[str, ...]) -> list[str]:
    lowered_tokens = tuple(token.lower() for token in tokens)
    exact = [column for column in header if column.lower() in lowered_tokens]
    if exact:
        return exact

    suffix = []
    for column in header:
        lowered = column.lower()
        for token in lowered_tokens:
            if lowered.endswith(f"_{token}"):
                suffix.append(column)
                break
    if suffix:
        return suffix

    substring = []
    for column in header:
        lowered = column.lower()
        for token in lowered_tokens:
            if token in lowered:
                substring.append(column)
                break
    return substring


def _avg(values: list[float]) -> float:
    return round(mean(values), 2) if values else 0.0


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finding(severity: str, category: str, message: str, **evidence: Any) -> dict[str, Any]:
    return {
        "severity": severity,
        "category": category,
        "message": message,
        "evidence": evidence,
    }


def _summary(findings: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for item in findings:
        counts[item["severity"]] = counts.get(item["severity"], 0) + 1
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
