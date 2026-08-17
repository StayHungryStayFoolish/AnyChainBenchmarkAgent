#!/usr/bin/env python3
"""Read-only Prometheus exporter for benchmark runtime artifacts.

This exporter intentionally does not query blockchain RPC endpoints and does
not write benchmark state. It reads the JSON/CSV files already produced by the
monitoring stack and exposes a bounded Prometheus text-format snapshot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_DIR = Path(os.environ.get("MEMORY_SHARE_DIR", "/dev/shm/blockchain-node-benchmark"))
DEFAULT_LOGS_DIR = Path(os.environ.get("LOGS_DIR", PROJECT_ROOT / "data" / "current" / "logs"))


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def to_float(value: Any) -> float | None:
    if value is None or value == "" or value == "null":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def metric_name(raw: str) -> str:
    return "blockchain_benchmark_" + raw


def escape_label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def labels_text(labels: dict[str, Any]) -> str:
    clean = {k: v for k, v in labels.items() if v is not None and str(v) != ""}
    if not clean:
        return ""
    body = ",".join(f'{key}="{escape_label(value)}"' for key, value in sorted(clean.items()))
    return "{" + body + "}"


class PrometheusBuilder:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.seen_help: set[str] = set()

    def gauge(self, name: str, help_text: str, value: Any, labels: dict[str, Any] | None = None) -> None:
        number = to_float(value)
        if number is None:
            return
        full_name = metric_name(name)
        if full_name not in self.seen_help:
            self.lines.append(f"# HELP {full_name} {help_text}")
            self.lines.append(f"# TYPE {full_name} gauge")
            self.seen_help.add(full_name)
        self.lines.append(f"{full_name}{labels_text(labels or {})} {number:g}")

    def counter(self, name: str, help_text: str, value: Any, labels: dict[str, Any] | None = None) -> None:
        number = to_float(value)
        if number is None:
            return
        full_name = metric_name(name)
        if full_name not in self.seen_help:
            self.lines.append(f"# HELP {full_name} {help_text}")
            self.lines.append(f"# TYPE {full_name} counter")
            self.seen_help.add(full_name)
        self.lines.append(f"{full_name}{labels_text(labels or {})} {number:g}")

    def render(self) -> str:
        return "\n".join(self.lines) + "\n"


def load_workload_methods(chain: str, rpc_mode: str, config_dir: Path) -> set[str]:
    config_file = config_dir / f"{chain}.json"
    try:
        with config_file.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return set()

    methods = config.get("rpc_methods", {})
    mode = rpc_mode.lower()
    if mode == "mixed":
        weighted = methods.get("mixed_weighted") or []
        if isinstance(weighted, list):
            names = {str(row.get("method", "")).strip() for row in weighted if isinstance(row, dict)}
            names.discard("")
            if names:
                return names
        mixed = methods.get("mixed", "")
        if isinstance(mixed, str):
            return {part.strip() for part in mixed.split(",") if part.strip()}
    selected = methods.get(mode, methods.get("single", ""))
    if isinstance(selected, str):
        if "," in selected:
            return {part.strip() for part in selected.split(",") if part.strip()}
        return {selected.strip()} if selected.strip() else set()
    return set()


def status_to_number(sync_status: str, local_health: Any, data_loss: Any) -> int:
    status = str(sync_status or "unknown").lower()
    if status in {"healthy", "ok", "synced"} and str(local_health) != "0" and str(data_loss) != "1":
        return 1
    if status in {"unhealthy", "behind", "stale", "error"} or str(local_health) == "0" or str(data_loss) == "1":
        return -1
    return 0


def status_class(status_code: Any) -> str:
    try:
        code = int(float(status_code))
    except (TypeError, ValueError):
        return "unknown"
    if code <= 0:
        return "unknown"
    return f"{code // 100}xx"


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def collect_proxy_method_metrics(proxy_csv: Path, allowed_methods: set[str], max_rows: int) -> dict[tuple[str, str], dict[str, Any]]:
    metrics: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"requests": 0, "errors": 0, "latencies": []}
    )
    if not proxy_csv.exists():
        return metrics

    try:
        with proxy_csv.open("r", encoding="utf-8", newline="") as fh:
            rows = csv.DictReader(fh)
            for idx, row in enumerate(rows):
                if idx >= max_rows:
                    break
                method = (row.get("method_name") or "").strip()
                if not method:
                    continue
                if allowed_methods and method not in allowed_methods:
                    continue
                klass = status_class(row.get("status_code"))
                key = (method, klass)
                bucket = metrics[key]
                bucket["requests"] += 1
                code = to_float(row.get("status_code"))
                if code is None or code >= 400:
                    bucket["errors"] += 1
                latency = to_float(row.get("latency_ms"))
                if latency is not None:
                    bucket["latencies"].append(latency)
    except OSError:
        return defaultdict(lambda: {"requests": 0, "errors": 0, "latencies": []})
    return metrics


def read_latest_csv_row(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = csv.DictReader(fh)
            latest: dict[str, Any] = {}
            for row in rows:
                if row:
                    latest = dict(row)
            return latest
    except OSError:
        return {}


def emit_latest_csv_metrics(builder: PrometheusBuilder, latest_row: dict[str, Any], labels: dict[str, Any]) -> None:
    if not latest_row:
        return

    for field, metric, help_text in (
        ("cpu_iowait", "cpu_iowait_percent", "CPU iowait percent from performance_latest.csv."),
        ("execution_mgas_per_sec", "execution_mgas_per_sec", "Node execution throughput in million gas per second when exposed by client metrics."),
        ("execution_gas_per_sec", "execution_gas_per_sec", "Node execution throughput in gas per second when exposed by client metrics."),
        ("node_process_cpu_pct", "node_process_cpu_percent", "Observed blockchain node process CPU percent."),
        ("node_thread_count", "node_thread_count", "Observed blockchain node thread count."),
        ("node_hottest_thread_cpu_pct", "node_hottest_thread_cpu_percent", "CPU percent of the hottest observed node thread."),
        ("node_hottest_thread_core", "node_hottest_thread_core", "CPU core id running the hottest observed node thread."),
        ("node_hottest_core_cpu_pct", "node_hottest_core_cpu_percent", "CPU percent of the hottest core used by the node process."),
        ("node_cpu_concentration_top1_pct", "node_cpu_concentration_top1_percent", "Node CPU concentration in the hottest thread/core."),
        ("node_cpu_concentration_top5_pct", "node_cpu_concentration_top5_percent", "Node CPU concentration across the top five threads/cores."),
        ("node_process_rss_mib", "node_process_rss_mebibytes", "Observed blockchain node process resident memory in MiB."),
        ("node_process_memory_pct", "node_process_memory_percent", "Observed blockchain node process resident memory as a percentage of host memory."),
        ("local_block_height", "local_block_height", "Local node block height from performance_latest.csv."),
        ("mainnet_block_height", "mainnet_block_height", "Reference/mainnet block height from performance_latest.csv."),
        ("block_height_diff", "block_height_diff_csv", "Reference minus local block height difference from performance_latest.csv."),
        ("freshness_gap_seconds", "sync_freshness_gap_seconds_csv", "Local progress freshness gap from performance_latest.csv."),
    ):
        builder.gauge(metric, help_text, latest_row.get(field), labels)

    execution_labels = {
        **labels,
        "source": latest_row.get("execution_metric_source", "unknown"),
        "status": latest_row.get("execution_metric_status", "unknown"),
    }
    builder.gauge(
        "execution_metric_available",
        "Whether node execution/MGas metrics were available in the latest CSV row.",
        1 if to_float(latest_row.get("execution_mgas_per_sec")) is not None or to_float(latest_row.get("execution_gas_per_sec")) is not None else 0,
        execution_labels,
    )

    client_labels = {
        **labels,
        "client_profile": latest_row.get("client_metric_profile", "none"),
        "quality": latest_row.get("client_metric_quality", "unavailable"),
    }
    for field, metric, help_text in (
        ("client_block_insert_ms_p50", "client_block_insert_milliseconds_p50", "Client-native block insertion duration P50 in milliseconds."),
        ("client_import_mgas_per_sec_p50", "client_import_mgas_per_sec_p50", "Client-native imported-block execution throughput P50 in million gas per second."),
        ("client_block_tx_count", "client_block_transactions", "Transactions in the latest client-native imported-block sample."),
        ("client_block_gas_used", "client_block_gas_used", "Gas used by the latest client-native imported-block sample."),
        ("client_head_block", "client_head_block", "Client-native imported head block height."),
        ("client_justified_block", "client_justified_block", "Client-native justified block height."),
        ("client_finalized_block", "client_finalized_block", "Client-native finalized block height."),
    ):
        builder.gauge(metric, help_text, latest_row.get(field), client_labels)
    builder.gauge(
        "client_import_observation_count",
        "Client-native execution observations represented by the latest resetting summary scrape.",
        latest_row.get("client_import_observation_count"),
        client_labels,
    )
    builder.counter(
        "client_inserted_blocks_total",
        "Client-native inserted-block observations represented by the latest summary.",
        latest_row.get("client_inserted_blocks_count"),
        client_labels,
    )
    builder.gauge(
        "client_metric_profile_available",
        "Whether a client-native metric profile produced at least partial data in the latest CSV row.",
        1 if str(latest_row.get("client_metric_quality") or "").lower() in {"complete", "partial"} else 0,
        client_labels,
    )

    node_labels = {
        **labels,
        "status": latest_row.get("node_cpu_status", "unknown"),
        "pid": latest_row.get("node_process_pid", ""),
        "hottest_thread": latest_row.get("node_hottest_thread_name", ""),
    }
    builder.gauge(
        "node_cpu_attribution_available",
        "Whether node process/thread CPU attribution was available in the latest CSV row.",
        1 if to_float(latest_row.get("node_process_cpu_pct")) is not None else 0,
        node_labels,
    )


def build_metrics(
    memory_dir: Path,
    logs_dir: Path,
    chain: str,
    rpc_mode: str,
    session: str,
    config_dir: Path,
    max_proxy_rows: int,
    include_session_label: bool,
) -> str:
    builder = PrometheusBuilder()
    labels = {"chain": chain, "rpc_mode": rpc_mode}
    if include_session_label:
        labels["session"] = session

    latest = read_json(memory_dir / "latest_metrics.json")
    sync_cache = read_json(memory_dir / "block_height_monitor_cache.json")
    bottleneck = read_json(memory_dir / "bottleneck_status.json")
    qps_status = read_json(memory_dir / "qps_status.json")
    latest_csv = read_latest_csv_row(logs_dir / "performance_latest.csv")

    builder.gauge("exporter_up", "Exporter scrape succeeded.", 1, labels)
    builder.gauge("artifact_latest_metrics_present", "Whether latest_metrics.json is readable.", 1 if latest else 0, labels)
    builder.gauge("artifact_sync_cache_present", "Whether block_height_monitor_cache.json is readable.", 1 if sync_cache else 0, labels)
    builder.gauge("artifact_performance_csv_present", "Whether performance_latest.csv is readable.", 1 if latest_csv else 0, labels)

    for field, metric, help_text in (
        ("cpu_usage", "cpu_usage_percent", "CPU usage percent from latest metrics."),
        ("memory_usage", "memory_usage_percent", "Memory usage percent from latest metrics."),
        ("disk_util", "disk_util_percent", "Disk utilization percent from latest metrics."),
        ("disk_latency", "disk_latency_ms", "Disk average latency in milliseconds from latest metrics."),
        ("network_util", "network_util_percent", "Network utilization percent from latest metrics."),
        ("error_rate", "error_rate_percent", "RPC error rate percent from latest metrics."),
    ):
        builder.gauge(metric, help_text, latest.get(field), labels)

    qps_value = bottleneck.get("current_qps", qps_status.get("max_successful_qps"))
    builder.gauge("qps_current", "Current or last known benchmark QPS.", qps_value, labels)

    sync_labels = {
        **labels,
        "sync_mode": sync_cache.get("sync_mode", "unknown"),
        "sync_status": sync_cache.get("sync_status", "unknown"),
    }
    builder.gauge(
        "sync_health_status",
        "Sync health encoded as 1 healthy, 0 unknown, -1 unhealthy.",
        status_to_number(sync_cache.get("sync_status"), sync_cache.get("local_health"), sync_cache.get("data_loss")),
        sync_labels,
    )
    for field, metric, help_text in (
        ("local_block_height", "local_block_height", "Local node block height from sync cache."),
        ("mainnet_block_height", "mainnet_block_height", "Reference/mainnet block height from sync cache."),
        ("block_height_diff", "block_height_diff", "Target minus local block height difference."),
        ("lag_value", "sync_lag_value", "Reported sync lag value."),
        ("freshness_gap_seconds", "sync_freshness_gap_seconds", "Local progress freshness gap in seconds."),
    ):
        builder.gauge(metric, help_text, sync_cache.get(field), sync_labels)

    emit_latest_csv_metrics(builder, latest_csv, labels)

    detected = str(bottleneck.get("bottleneck_detected", "false")).lower() == "true"
    bottleneck_types = bottleneck.get("bottleneck_types") if isinstance(bottleneck.get("bottleneck_types"), list) else []
    if not bottleneck_types:
        bottleneck_types = ["none"]
    for btype in bottleneck_types:
        b_labels = {**labels, "type": str(btype).lower(), "status": bottleneck.get("status", "unknown")}
        builder.gauge("bottleneck_active", "Whether a bottleneck type is currently active.", 1 if detected and btype != "none" else 0, b_labels)

    allowed_methods = load_workload_methods(chain, rpc_mode, config_dir)
    proxy_metrics = collect_proxy_method_metrics(logs_dir / "proxy_method.csv", allowed_methods, max_proxy_rows)
    for (method, klass), data in sorted(proxy_metrics.items()):
        m_labels = {**labels, "method": method, "status_class": klass}
        builder.counter("rpc_method_requests_total", "Proxy-observed workload RPC method request count.", data["requests"], m_labels)
        builder.counter("rpc_method_errors_total", "Proxy-observed workload RPC method error count.", data["errors"], m_labels)
        latency_values = data["latencies"]
        builder.counter("rpc_method_latency_ms_sum", "Sum of proxy-observed workload RPC latencies in milliseconds.", sum(latency_values), m_labels)
        builder.counter("rpc_method_latency_ms_count", "Count of proxy-observed workload RPC latencies.", len(latency_values), m_labels)
        builder.gauge("rpc_method_latency_p99_ms", "Proxy-observed workload RPC p99 latency in milliseconds.", quantile(latency_values, 0.99), m_labels)

    builder.gauge("scrape_timestamp_seconds", "Unix timestamp for this exporter scrape.", int(time.time()), labels)
    return builder.render()


def parse_listen(value: str) -> tuple[str, int]:
    if ":" not in value:
        return value, 9108
    host, port_text = value.rsplit(":", 1)
    return host or "0.0.0.0", int(port_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Prometheus exporter for benchmark runtime artifacts.")
    parser.add_argument("--listen", default=os.environ.get("PROMETHEUS_EXPORTER_LISTEN", "0.0.0.0:9108"))
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "config" / "chains")
    parser.add_argument("--chain", default=os.environ.get("BLOCKCHAIN_NODE", "unknown"))
    parser.add_argument("--rpc-mode", default=os.environ.get("RPC_MODE", "single"))
    parser.add_argument("--session", default=os.environ.get("SESSION_TIMESTAMP", "unknown"))
    parser.add_argument("--max-proxy-rows", type=int, default=int(os.environ.get("PROMETHEUS_EXPORTER_MAX_PROXY_ROWS", "20000")))
    parser.add_argument("--include-session-label", action="store_true")
    parser.add_argument("--once", action="store_true", help="Print one metrics snapshot and exit.")
    args = parser.parse_args()

    def render() -> bytes:
        return build_metrics(
            args.memory_dir,
            args.logs_dir,
            args.chain,
            args.rpc_mode,
            args.session,
            args.config_dir,
            args.max_proxy_rows,
            args.include_session_label,
        ).encode("utf-8")

    if args.once:
        print(render().decode("utf-8"), end="")
        return 0

    host, port = parse_listen(args.listen)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming
            if self.path not in {"/metrics", "/"}:
                self.send_response(404)
                self.end_headers()
                return
            body = render()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Prometheus exporter listening on http://{host}:{port}/metrics")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
