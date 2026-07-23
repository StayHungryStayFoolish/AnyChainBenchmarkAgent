#!/usr/bin/env python3
"""Collect optional execution-throughput and node CPU hotspot fields.

The collector is intentionally fail-soft. It never blocks the benchmark when a
node Prometheus endpoint, node process, or Linux /proc detail is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


EXECUTION_HEADER = [
    "execution_mgas_per_sec",
    "execution_gas_per_sec",
    "execution_metric_source",
    "execution_metric_status",
]

NODE_CPU_HEADER = [
    "node_process_pid",
    "node_process_cpu_pct",
    "node_thread_count",
    "node_hottest_thread_tid",
    "node_hottest_thread_name",
    "node_hottest_thread_cpu_pct",
    "node_hottest_thread_core",
    "node_top_threads_cpu_pct",
    "node_hottest_core_id",
    "node_hottest_core_cpu_pct",
    "node_top_cores_cpu_pct",
    "node_cpu_concentration_top1_pct",
    "node_cpu_concentration_top5_pct",
    "node_cpu_status",
]

DIRECT_MGAS_METRICS = [
    "chain_mgasps",
    "chain_insert_mgasps",
    "reth_consensus_engine_beacon_block_insert_mgasps",
    "nethermind_mgas_per_sec",
    "chain_tip_mgas_per_sec",
    "exec_mgas_sec",
    "exec_task_mgas_sec",
]

GAS_PER_SEC_METRICS = [
    "reth_sync_execution_gas_per_second",
]


def _csv_safe(value: object, max_len: int = 120) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ").replace(",", ";").strip()
    return text[:max_len] if len(text) > max_len else text


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "0"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0"


def _metric_value(metrics_text: str, metric_name: str) -> Optional[float]:
    pattern = re.compile(
        rf"^{re.escape(metric_name)}\s*(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)\s*$"
    )
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line.strip())
        if not match:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def collect_execution_data() -> List[str]:
    url = (
        os.getenv("NODE_PROMETHEUS_METRICS_URL")
        or os.getenv("BLOCKCHAIN_NODE_METRICS_URL")
        or os.getenv("EXECUTION_METRICS_URL")
        or ""
    ).strip()
    if not url:
        return ["0", "0", "unconfigured", "unavailable"]

    timeout = float(os.getenv("NODE_PROMETHEUS_TIMEOUT_SECONDS", "2") or "2")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            metrics_text = response.read().decode("utf-8", errors="replace")
    except Exception:
        return ["0", "0", "prometheus_fetch_error", "error"]

    for name in DIRECT_MGAS_METRICS:
        value = _metric_value(metrics_text, name)
        if value is not None:
            return [_fmt(value), _fmt(value * 1_000_000), name, "available"]

    for name in GAS_PER_SEC_METRICS:
        value = _metric_value(metrics_text, name)
        if value is not None:
            return [_fmt(value / 1_000_000), _fmt(value), name, "available"]

    return ["0", "0", "no_supported_metric", "unavailable"]


def _proc_root() -> Path:
    return Path(os.getenv("PROC_ROOT", "/proc"))


def _state_path() -> Path:
    base = os.getenv("MEMORY_SHARE_DIR", "/tmp")
    return Path(os.getenv("NODE_CPU_STATE_FILE", str(Path(base) / "node_cpu_collector_state.json")))


def _find_pid() -> Optional[int]:
    explicit = os.getenv("NODE_PROCESS_PID", "").strip()
    if explicit.isdigit():
        return int(explicit)

    names = [name.strip() for name in os.getenv("BLOCKCHAIN_PROCESS_NAMES_STR", "").split() if name.strip()]
    if not names:
        return None
    proc = _proc_root()
    own_pid = os.getpid()
    candidates: List[Tuple[int, str]] = []
    for entry in proc.iterdir() if proc.exists() else []:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == own_pid:
            continue
        try:
            comm = (entry / "comm").read_text(encoding="utf-8", errors="ignore").strip()
            raw_cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [
            token.decode("utf-8", errors="ignore")
            for token in raw_cmdline.split(b"\x00")
            if token
        ]
        tokens = {comm, Path(comm).name}
        tokens.update(Path(arg).name for arg in argv if arg)
        tokens.update(arg for arg in argv if arg.endswith(".service"))
        if any(name in tokens for name in names):
            candidates.append((pid, comm))
    if not candidates:
        return None
    return sorted(candidates)[0][0]


def _read_total_cpu_jiffies(proc: Path) -> Tuple[int, Dict[str, Tuple[int, int]]]:
    total = 0
    cores: Dict[str, Tuple[int, int]] = {}
    try:
        lines = (proc / "stat").read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return 0, cores
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "cpu":
            values = [int(float(v)) for v in parts[1:]]
            total = sum(values)
        elif re.match(r"^cpu\d+$", parts[0]):
            values = [int(float(v)) for v in parts[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            core_total = sum(values)
            busy = core_total - idle
            cores[parts[0][3:]] = (busy, core_total)
    return total, cores


def _parse_thread_stat(path: Path) -> Optional[Tuple[int, int, int]]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if ") " not in raw:
        return None
    rest = raw.split(") ", 1)[1].split()
    try:
        utime = int(rest[11])
        stime = int(rest[12])
        processor = int(rest[36]) if len(rest) > 36 else -1
    except (IndexError, ValueError):
        return None
    return utime + stime, utime, processor


def _read_threads(pid: int, proc: Path) -> Dict[str, Dict[str, object]]:
    task_dir = proc / str(pid) / "task"
    threads: Dict[str, Dict[str, object]] = {}
    if not task_dir.exists():
        return threads
    for task in task_dir.iterdir():
        if not task.name.isdigit():
            continue
        parsed = _parse_thread_stat(task / "stat")
        if parsed is None:
            continue
        jiffies, _utime, processor = parsed
        try:
            name = (task / "comm").read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            name = "unknown"
        threads[task.name] = {
            "jiffies": jiffies,
            "name": name or "unknown",
            "processor": processor,
        }
    return threads


def _load_state(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, state: Dict[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def collect_node_cpu_data() -> List[str]:
    proc = _proc_root()
    pid = _find_pid()
    if pid is None:
        return ["0", "0", "0", "0", "unavailable", "0", "-1", "", "-1", "0", "", "0", "0", "unavailable"]

    total_jiffies, cores = _read_total_cpu_jiffies(proc)
    threads = _read_threads(pid, proc)
    state_file = _state_path()
    previous = _load_state(state_file)
    now_state = {
        "timestamp": time.time(),
        "pid": pid,
        "total_jiffies": total_jiffies,
        "threads": threads,
        "cores": {k: {"busy": v[0], "total": v[1]} for k, v in cores.items()},
    }
    _save_state(state_file, now_state)

    prev_total = int(previous.get("total_jiffies", 0) or 0)
    if not previous or int(previous.get("pid", 0) or 0) != pid or total_jiffies <= prev_total:
        return [str(pid), "0", str(len(threads)), "0", "warmup", "0", "-1", "", "-1", "0", "", "0", "0", "warmup"]

    ncpu = os.cpu_count() or 1
    total_delta = max(total_jiffies - prev_total, 1)
    prev_threads = previous.get("threads", {}) if isinstance(previous.get("threads"), dict) else {}

    thread_rows: List[Tuple[str, str, float, int]] = []
    process_delta = 0
    for tid, info in threads.items():
        prev_info = prev_threads.get(tid, {}) if isinstance(prev_threads, dict) else {}
        prev_jiffies = int(prev_info.get("jiffies", 0) or 0)
        delta = max(int(info.get("jiffies", 0) or 0) - prev_jiffies, 0)
        process_delta += delta
        pct = delta * ncpu * 100.0 / total_delta
        thread_rows.append((tid, str(info.get("name", "unknown")), pct, int(info.get("processor", -1) or -1)))

    process_cpu_pct = process_delta * ncpu * 100.0 / total_delta
    thread_rows.sort(key=lambda row: row[2], reverse=True)
    top_threads = thread_rows[:5]
    hottest = top_threads[0] if top_threads else ("0", "unavailable", 0.0, -1)
    top5_cpu = sum(row[2] for row in top_threads)

    prev_cores = previous.get("cores", {}) if isinstance(previous.get("cores"), dict) else {}
    core_rows: List[Tuple[str, float]] = []
    for core_id, (busy, core_total) in cores.items():
        prev_core = prev_cores.get(core_id, {}) if isinstance(prev_cores, dict) else {}
        prev_busy = int(prev_core.get("busy", 0) or 0)
        prev_core_total = int(prev_core.get("total", 0) or 0)
        busy_delta = max(busy - prev_busy, 0)
        core_delta = max(core_total - prev_core_total, 1)
        core_rows.append((core_id, busy_delta * 100.0 / core_delta))
    core_rows.sort(key=lambda row: row[1], reverse=True)
    top_cores = core_rows[:5]
    hottest_core = top_cores[0] if top_cores else ("-1", 0.0)

    top_threads_desc = "|".join(
        f"{tid}:{_csv_safe(name, 30)}:{_fmt(cpu)}:{core}" for tid, name, cpu, core in top_threads
    )
    top_cores_desc = "|".join(f"{core}:{_fmt(cpu)}" for core, cpu in top_cores)
    concentration_top1 = (hottest[2] / process_cpu_pct * 100.0) if process_cpu_pct > 0 else 0.0
    concentration_top5 = (top5_cpu / process_cpu_pct * 100.0) if process_cpu_pct > 0 else 0.0

    return [
        str(pid),
        _fmt(process_cpu_pct),
        str(len(threads)),
        hottest[0],
        _csv_safe(hottest[1], 40),
        _fmt(hottest[2]),
        str(hottest[3]),
        _csv_safe(top_threads_desc, 300),
        hottest_core[0],
        _fmt(hottest_core[1]),
        _csv_safe(top_cores_desc, 200),
        _fmt(concentration_top1),
        _fmt(concentration_top5),
        "available",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-header", action="store_true")
    parser.add_argument("--execution-data", action="store_true")
    parser.add_argument("--node-cpu-header", action="store_true")
    parser.add_argument("--node-cpu-data", action="store_true")
    args = parser.parse_args()

    if args.execution_header:
        print(",".join(EXECUTION_HEADER))
    elif args.execution_data:
        print(",".join(_csv_safe(v) for v in collect_execution_data()))
    elif args.node_cpu_header:
        print(",".join(NODE_CPU_HEADER))
    elif args.node_cpu_data:
        print(",".join(_csv_safe(v) for v in collect_node_cpu_data()))
    else:
        parser.error("one of --execution-header/--execution-data/--node-cpu-header/--node-cpu-data is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
