#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source monitoring/lib/performance_data_line_builder.sh

timestamp="2026-06-11 12:00:00"
cpu_data="1,2,3,4,5,6"
memory_data="7,8,9"
device_data="10,11"
network_data="eth0,12,13"
ena_data="14,15"
overhead_data="16,17"
block_height_data="18,19,20,1,1,0,absolute_gap,healthy,0,block,0,null"
execution_data="31,31000000,chain_insert_mgasps,available"
node_cpu_data="1234,250,12,1235,geth,95,7,1235:geth:95:7,7,99,7:99,38,80,available"
cgroup_data="21,22"
cloud_provider="aws"

non_ena_line="$(build_performance_data_line \
    false "$timestamp" "$cpu_data" "$memory_data" "$device_data" "$network_data" \
    "$ena_data" "$overhead_data" "$block_height_data" "30"$'\n' "40"$'\r' "true"$'\n' \
    "$execution_data" "$node_cpu_data" "$cgroup_data" "$cloud_provider")"

expected_non_ena="$timestamp,$cpu_data,$memory_data,$device_data,$network_data,$overhead_data,$block_height_data,30,40,true,$execution_data,$node_cpu_data,$cgroup_data,$cloud_provider"
[[ "$non_ena_line" == "$expected_non_ena" ]] || {
    echo "Non-ENA line mismatch"
    echo "expected: $expected_non_ena"
    echo "actual:   $non_ena_line"
    exit 1
}

ena_line="$(build_performance_data_line \
    true "$timestamp" "$cpu_data" "$memory_data" "$device_data" "$network_data" \
    "$ena_data" "$overhead_data" "$block_height_data" "30" "40" "true" \
    "$execution_data" "$node_cpu_data" "$cgroup_data" "$cloud_provider")"

expected_ena="$timestamp,$cpu_data,$memory_data,$device_data,$network_data,$ena_data,$overhead_data,$block_height_data,30,40,true,$execution_data,$node_cpu_data,$cgroup_data,$cloud_provider"
[[ "$ena_line" == "$expected_ena" ]] || {
    echo "ENA line mismatch"
    echo "expected: $expected_ena"
    echo "actual:   $ena_line"
    exit 1
}

echo "✅ performance_data_line_builder preserves CSV field order"
