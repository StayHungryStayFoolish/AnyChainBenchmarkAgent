#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory_dir = root / "memory"
        logs_dir = root / "logs"
        memory_dir.mkdir()
        logs_dir.mkdir()

        write_json(
            memory_dir / "latest_metrics.json",
            {
                "timestamp": "2026-06-11 12:00:00",
                "cpu_usage": 12.5,
                "memory_usage": 34.5,
                "disk_util": 45.5,
                "disk_latency": 6.7,
                "network_util": 8.9,
                "error_rate": 0,
            },
        )
        write_json(
            memory_dir / "block_height_monitor_cache.json",
            {
                "timestamp": "2026-06-11T12:00:00Z",
                "timestamp_ms": 1781150400000,
                "sync_mode": "reported_lag",
                "sync_status": "healthy",
                "local_health": 1,
                "data_loss": 0,
                "block_height_diff": None,
                "lag_value": 0,
                "freshness_gap_seconds": None,
            },
        )
        write_json(
            memory_dir / "bottleneck_status.json",
            {
                "status": "monitoring",
                "bottleneck_detected": False,
                "bottleneck_types": [],
                "current_qps": 100,
            },
        )

        (logs_dir / "proxy_method.csv").write_text(
            "\n".join(
                [
                    "timestamp_ns,method_name,protocol,request_id,batch_idx,status_code,latency_ms,upstream,client_addr",
                    "1,getAccountInfo,json_rpc,1,0,200,10,http://127.0.0.1:8899,127.0.0.1:1111",
                    "2,getAccountInfo,json_rpc,2,0,500,30,http://127.0.0.1:8899,127.0.0.1:1111",
                    "3,getHealth,json_rpc,3,0,200,5,http://127.0.0.1:8899,127.0.0.1:1111",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (logs_dir / "performance_latest.csv").write_text(
            "\n".join(
                [
                    (
                        "timestamp,cpu_iowait,local_block_height,mainnet_block_height,block_height_diff,"
                        "freshness_gap_seconds,execution_mgas_per_sec,execution_gas_per_sec,"
                        "execution_metric_source,execution_metric_status,node_process_pid,"
                        "node_process_cpu_pct,node_thread_count,node_hottest_thread_name,"
                        "node_hottest_thread_cpu_pct,node_hottest_thread_core,"
                        "node_hottest_core_cpu_pct,node_cpu_concentration_top1_pct,"
                        "node_cpu_concentration_top5_pct,node_cpu_status"
                    ),
                    "2026-06-11 12:00:00,1.5,100,105,5,2.5,12.25,12250000,client_metrics,ok,4242,188.5,64,geth,91.2,3,98.7,48.1,80.4,ok",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "monitoring" / "prometheus_exporter.py"),
                "--once",
                "--memory-dir",
                str(memory_dir),
                "--logs-dir",
                str(logs_dir),
                "--chain",
                "solana",
                "--rpc-mode",
                "mixed",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        output = result.stdout

        assert "blockchain_benchmark_exporter_up" in output
        assert 'blockchain_benchmark_cpu_usage_percent{chain="solana",rpc_mode="mixed"} 12.5' in output
        assert (
            'blockchain_benchmark_sync_health_status{chain="solana",rpc_mode="mixed",sync_mode="reported_lag",sync_status="healthy"} 1'
            in output
        )
        assert (
            'blockchain_benchmark_rpc_method_requests_total{chain="solana",method="getAccountInfo",rpc_mode="mixed",status_class="2xx"} 1'
            in output
        )
        assert (
            'blockchain_benchmark_rpc_method_errors_total{chain="solana",method="getAccountInfo",rpc_mode="mixed",status_class="5xx"} 1'
            in output
        )
        assert 'blockchain_benchmark_artifact_performance_csv_present{chain="solana",rpc_mode="mixed"} 1' in output
        assert 'blockchain_benchmark_execution_mgas_per_sec{chain="solana",rpc_mode="mixed"} 12.25' in output
        assert 'blockchain_benchmark_execution_gas_per_sec{chain="solana",rpc_mode="mixed"} 1.225e+07' in output
        assert (
            'blockchain_benchmark_execution_metric_available{chain="solana",rpc_mode="mixed",source="client_metrics",status="ok"} 1'
            in output
        )
        assert 'blockchain_benchmark_node_process_cpu_percent{chain="solana",rpc_mode="mixed"} 188.5' in output
        assert 'blockchain_benchmark_node_hottest_thread_cpu_percent{chain="solana",rpc_mode="mixed"} 91.2' in output
        assert 'blockchain_benchmark_cpu_iowait_percent{chain="solana",rpc_mode="mixed"} 1.5' in output
        assert 'blockchain_benchmark_local_block_height{chain="solana",rpc_mode="mixed"} 100' in output
        assert 'blockchain_benchmark_mainnet_block_height{chain="solana",rpc_mode="mixed"} 105' in output
        assert "method=\"getHealth\"" not in output

    print("✅ Prometheus exporter synthetic metrics test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
