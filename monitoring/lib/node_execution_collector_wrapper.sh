#!/usr/bin/env bash
# =====================================================================
# Node Execution Collector Wrapper for Unified Monitor
# =====================================================================
# Provides optional execution-throughput and node CPU hotspot CSV segments.
# Missing Prometheus/process data is represented as explicit unavailable
# values; the benchmark must never fail only because these optional metrics are
# unavailable.
# =====================================================================

EXECUTION_PLACEHOLDER_HEADER="execution_mgas_per_sec,execution_gas_per_sec,execution_metric_source,execution_metric_status"
EXECUTION_PLACEHOLDER_DATA="0,0,unavailable,unavailable"

NODE_CPU_PLACEHOLDER_HEADER="node_process_pid,node_process_cpu_pct,node_thread_count,node_hottest_thread_tid,node_hottest_thread_name,node_hottest_thread_cpu_pct,node_hottest_thread_core,node_top_threads_cpu_pct,node_hottest_core_id,node_hottest_core_cpu_pct,node_top_cores_cpu_pct,node_cpu_concentration_top1_pct,node_cpu_concentration_top5_pct,node_cpu_status"
NODE_CPU_PLACEHOLDER_DATA="0,0,0,0,unavailable,0,-1,,-1,0,,0,0,unavailable"

resolve_node_execution_collector_path() {
    if [[ -n "${NODE_EXECUTION_COLLECTOR_PATH:-}" ]]; then
        echo "$NODE_EXECUTION_COLLECTOR_PATH"
        return 0
    fi

    local module_dir
    module_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    echo "${module_dir}/node_execution_collector.py"
}

get_execution_header() {
    local collector
    collector="$(resolve_node_execution_collector_path)"
    if [[ ! -f "$collector" ]]; then
        echo "$EXECUTION_PLACEHOLDER_HEADER"
        return 0
    fi
    python3 "$collector" --execution-header 2>/dev/null || echo "$EXECUTION_PLACEHOLDER_HEADER"
}

get_execution_data() {
    local collector
    collector="$(resolve_node_execution_collector_path)"
    if [[ ! -f "$collector" ]]; then
        echo "$EXECUTION_PLACEHOLDER_DATA"
        return 0
    fi
    python3 "$collector" --execution-data 2>/dev/null || echo "$EXECUTION_PLACEHOLDER_DATA"
}

get_node_cpu_header() {
    local collector
    collector="$(resolve_node_execution_collector_path)"
    if [[ ! -f "$collector" ]]; then
        echo "$NODE_CPU_PLACEHOLDER_HEADER"
        return 0
    fi
    python3 "$collector" --node-cpu-header 2>/dev/null || echo "$NODE_CPU_PLACEHOLDER_HEADER"
}

get_node_cpu_data() {
    local collector
    collector="$(resolve_node_execution_collector_path)"
    if [[ ! -f "$collector" ]]; then
        echo "$NODE_CPU_PLACEHOLDER_DATA"
        return 0
    fi
    python3 "$collector" --node-cpu-data 2>/dev/null || echo "$NODE_CPU_PLACEHOLDER_DATA"
}
