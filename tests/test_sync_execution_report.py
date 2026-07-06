#!/usr/bin/env python3
"""Verify the report renders sync execution metrics and chart from CSV data."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from visualization.report_generator import ReportGenerator


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        csv_path = root / "performance_latest.csv"
        reports_dir = root / "reports"
        logs_dir = root / "logs"
        reports_dir.mkdir()
        logs_dir.mkdir()

        df = pd.DataFrame(
            {
                "timestamp": ["2026-07-06 00:00:00", "2026-07-06 00:00:05", "2026-07-06 00:00:10"],
                "cpu_usage": [20.0, 40.0, 60.0],
                "mem_usage": [50.0, 51.0, 52.0],
                "local_block_height": [100, 110, 125],
                "mainnet_block_height": [130, 132, 135],
                "block_height_diff": [30, 22, 10],
                "sync_mode": ["absolute_gap", "absolute_gap", "absolute_gap"],
                "sync_status": ["syncing", "syncing", "healthy"],
                "execution_mgas_per_sec": [10.0, 20.0, 15.0],
                "execution_gas_per_sec": [10000000.0, 20000000.0, 15000000.0],
                "execution_metric_source": ["chain_insert_mgasps"] * 3,
                "execution_metric_status": ["available"] * 3,
                "node_process_pid": [1234, 1234, 1234],
                "node_process_cpu_pct": [100.0, 220.0, 180.0],
                "node_thread_count": [12, 12, 12],
                "node_hottest_thread_tid": [1235, 1235, 1235],
                "node_hottest_thread_name": ["geth", "geth", "geth"],
                "node_hottest_thread_cpu_pct": [80.0, 95.0, 75.0],
                "node_hottest_thread_core": [2, 2, 3],
                "node_top_threads_cpu_pct": ["1235:geth:80:2", "1235:geth:95:2", "1235:geth:75:3"],
                "node_hottest_core_id": [2, 2, 3],
                "node_hottest_core_cpu_pct": [90.0, 99.0, 82.0],
                "node_top_cores_cpu_pct": ["2:90", "2:99", "3:82"],
                "node_cpu_concentration_top1_pct": [80.0, 43.0, 42.0],
                "node_cpu_concentration_top5_pct": [95.0, 90.0, 88.0],
                "node_cpu_status": ["available"] * 3,
                "data_nvme0n1_avg_await": [1.2, 2.5, 1.7],
                "data_nvme0n1_util": [20.0, 45.0, 35.0],
                "data_nvme0n1_total_iops": [1000, 1200, 1100],
                "data_nvme0n1_total_throughput_mibs": [100.0, 140.0, 120.0],
                "net_rx_mbps": [10.0, 20.0, 15.0],
                "net_tx_mbps": [4.0, 6.0, 5.0],
                "current_qps": [1, 1, 1],
                "rpc_latency_ms": [10, 12, 11],
                "cloud_provider": ["other", "other", "other"],
            }
        )
        df.to_csv(csv_path, index=False)

        os.environ["REPORTS_DIR"] = str(reports_dir)
        os.environ["LOGS_DIR"] = str(logs_dir)
        os.environ["SESSION_TIMESTAMP"] = "sync_execution_test"
        os.environ["SYNC_OBSERVE_MODE"] = "true"

        output = ReportGenerator(str(csv_path), language="en").generate_html_report()
        assert output, "report generation returned no output path"
        html = Path(output).read_text(encoding="utf-8")
        assert "Node Sync Execution Analysis" in html
        assert "MGas/s min / avg / peak" in html
        assert "chain_insert_mgasps" in html
        assert (reports_dir / "sync_execution_timeline.png").exists()

        df["execution_mgas_per_sec"] = [0.0, 0.0, 0.0]
        df["execution_gas_per_sec"] = [0.0, 0.0, 0.0]
        df["execution_metric_source"] = ["no_supported_metric"] * 3
        df["execution_metric_status"] = ["unavailable"] * 3
        df.to_csv(csv_path, index=False)

        output = ReportGenerator(str(csv_path), language="en").generate_html_report()
        html = Path(output).read_text(encoding="utf-8")
        assert "MGas/s min / avg / peak</td><td>N/A / N/A / N/A" in html
        assert "no_supported_metric / unavailable" in html

        output = ReportGenerator(str(csv_path), language="zh").generate_html_report()
        html = Path(output).read_text(encoding="utf-8")
        assert "节点同步执行分析" in html
        assert "Node Sync Execution Analysis" not in html

        os.environ["SYNC_OBSERVE_MODE"] = "false"
        df["execution_metric_status"] = ["unavailable"] * 3
        df.to_csv(csv_path, index=False)
        output = ReportGenerator(str(csv_path), language="en").generate_html_report()
        html = Path(output).read_text(encoding="utf-8")
        assert "Node Sync Execution Analysis" not in html

    print("✅ sync execution report rendering passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
