#!/usr/bin/env bash
# =====================================================================
# Unified Performance CSV Data Line Builder
# =====================================================================
# Assembles one performance CSV data row in the same order as
# unified_monitor.sh::generate_csv_header().
# =====================================================================

sanitize_csv_short_field() {
    local value="$1"
    local max_len="${2:-20}"
    echo "$value" | tr -d '\n\r' | head -c "$max_len"
}

build_performance_data_line() {
    local ena_enabled="$1"
    local timestamp="$2"
    local cpu_data="$3"
    local memory_data="$4"
    local device_data="$5"
    local network_data="$6"
    local ena_data="$7"
    local overhead_data="$8"
    local block_height_data="$9"
    local current_qps="${10}"
    local rpc_latency_ms="${11}"
    local qps_data_available="${12}"
    local execution_data="${13}"
    local node_cpu_data="${14}"
    local cgroup_data="${15}"
    local cloud_provider_val="${16}"

    current_qps=$(sanitize_csv_short_field "$current_qps" 20)
    rpc_latency_ms=$(sanitize_csv_short_field "$rpc_latency_ms" 20)
    qps_data_available=$(sanitize_csv_short_field "$qps_data_available" 10)

    if [[ "$ena_enabled" == "true" ]]; then
        echo "$timestamp,$cpu_data,$memory_data,$device_data,$network_data,$ena_data,$overhead_data,$block_height_data,$current_qps,$rpc_latency_ms,$qps_data_available,$execution_data,$node_cpu_data,$cgroup_data,$cloud_provider_val"
    else
        echo "$timestamp,$cpu_data,$memory_data,$device_data,$network_data,$overhead_data,$block_height_data,$current_qps,$rpc_latency_ms,$qps_data_available,$execution_data,$node_cpu_data,$cgroup_data,$cloud_provider_val"
    fi
}
