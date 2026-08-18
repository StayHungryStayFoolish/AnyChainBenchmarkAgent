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
                    "1,eth_getBalance,json_rpc,1,0,200,10,http://127.0.0.1:8545,127.0.0.1:1111",
                    "2,eth_getBalance,json_rpc,2,0,500,30,http://127.0.0.1:8545,127.0.0.1:1111",
                    "3,eth_accounts,json_rpc,3,0,200,5,http://127.0.0.1:8545,127.0.0.1:1111",
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
                        "execution_metric_source,execution_metric_status,client_metric_profile,"
                        "client_block_insert_ms_p50,client_import_mgas_per_sec_p50,"
                        "client_import_observation_count,client_block_tx_count,client_block_gas_used,"
                        "client_head_block,client_justified_block,client_finalized_block,"
                        "client_inserted_blocks_count,client_metric_quality,node_process_pid,"
                        "node_process_cpu_pct,node_thread_count,node_hottest_thread_name,"
                        "node_hottest_thread_cpu_pct,node_hottest_thread_core,"
                        "node_hottest_core_cpu_pct,node_cpu_concentration_top1_pct,"
                        "node_cpu_concentration_top5_pct,node_process_rss_mib,"
                        "node_process_memory_pct,node_cpu_status"
                    ),
                    "2026-06-11 12:00:00,1.5,100,105,5,2.5,567.23,567230000,chain_mgasps,available,"
                    "bsc_v1_7,348.19,567.23,42,1769,140500000,5000,4999,4998,44,complete,"
                    "4242,188.5,64,geth,91.2,3,98.7,48.1,80.4,4096,50,ok",
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
                "bsc",
                "--rpc-mode",
                "mixed",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        output = result.stdout

        assert "blockchain_benchmark_exporter_up" in output
        assert 'blockchain_benchmark_cpu_usage_percent{chain="bsc",rpc_mode="mixed"} 12.5' in output
        assert (
            'blockchain_benchmark_sync_health_status{chain="bsc",rpc_mode="mixed",sync_mode="reported_lag",sync_status="healthy"} 1'
            in output
        )
        assert (
            'blockchain_benchmark_rpc_method_requests_total{chain="bsc",method="eth_getBalance",rpc_mode="mixed",status_class="2xx"} 1'
            in output
        )
        assert (
            'blockchain_benchmark_rpc_method_errors_total{chain="bsc",method="eth_getBalance",rpc_mode="mixed",status_class="5xx"} 1'
            in output
        )
        assert 'blockchain_benchmark_artifact_performance_csv_present{chain="bsc",rpc_mode="mixed"} 1' in output
        assert 'blockchain_benchmark_execution_mgas_per_sec{chain="bsc",rpc_mode="mixed"} 567.23' in output
        assert 'blockchain_benchmark_execution_gas_per_sec{chain="bsc",rpc_mode="mixed"} 5.6723e+08' in output
        assert (
            'blockchain_benchmark_execution_metric_available{chain="bsc",rpc_mode="mixed",source="chain_mgasps",status="available"} 1'
            in output
        )
        assert 'blockchain_benchmark_node_process_cpu_percent{chain="bsc",rpc_mode="mixed"} 188.5' in output
        assert 'blockchain_benchmark_node_hottest_thread_cpu_percent{chain="bsc",rpc_mode="mixed"} 91.2' in output
        assert 'blockchain_benchmark_node_process_rss_mebibytes{chain="bsc",rpc_mode="mixed"} 4096' in output
        assert 'blockchain_benchmark_cpu_iowait_percent{chain="bsc",rpc_mode="mixed"} 1.5' in output
        assert 'blockchain_benchmark_local_block_height{chain="bsc",rpc_mode="mixed"} 100' in output
        assert 'blockchain_benchmark_mainnet_block_height{chain="bsc",rpc_mode="mixed"} 105' in output
        assert 'blockchain_benchmark_client_block_insert_milliseconds_p50{chain="bsc",client_profile="bsc_v1_7",quality="complete",rpc_mode="mixed"} 348.19' in output
        assert 'blockchain_benchmark_client_block_transactions{chain="bsc",client_profile="bsc_v1_7",quality="complete",rpc_mode="mixed"} 1769' in output
        assert 'blockchain_benchmark_client_block_gas_used{chain="bsc",client_profile="bsc_v1_7",quality="complete",rpc_mode="mixed"} 1.405e+08' in output
        assert 'blockchain_benchmark_client_head_block{chain="bsc",client_profile="bsc_v1_7",quality="complete",rpc_mode="mixed"} 5000' in output
        assert 'blockchain_benchmark_client_import_observation_count{chain="bsc",client_profile="bsc_v1_7",quality="complete",rpc_mode="mixed"} 42' in output
        assert "blockchain_benchmark_client_import_observations_total" not in output
        assert "method=\"eth_accounts\"" not in output

    print("✅ Prometheus exporter synthetic metrics test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
