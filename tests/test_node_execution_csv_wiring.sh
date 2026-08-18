#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

source monitoring/lib/node_execution_collector_wrapper.sh

execution_header="$(get_execution_header)"
node_cpu_header="$(get_node_cpu_header)"

[[ "$(echo "$execution_header" | awk -F, '{print NF}')" -eq 15 ]] || fail "execution header must have 15 fields"
[[ "$(echo "$node_cpu_header" | awk -F, '{print NF}')" -eq 16 ]] || fail "node CPU header must have 16 fields"

grep -q 'node_execution_collector_wrapper.sh' monitoring/unified_monitor.sh \
    || fail "unified_monitor does not source node_execution_collector_wrapper.sh"
grep -q 'execution_header=$(get_execution_header)' monitoring/unified_monitor.sh \
    || fail "generate_csv_header does not include execution_header"
grep -q 'node_cpu_header=$(get_node_cpu_header)' monitoring/unified_monitor.sh \
    || fail "generate_csv_header does not include node_cpu_header"
grep -q 'execution_data=$(get_execution_data)' monitoring/unified_monitor.sh \
    || fail "log_performance_data does not collect execution_data"
grep -q 'node_cpu_data=$(get_node_cpu_data)' monitoring/unified_monitor.sh \
    || fail "log_performance_data does not collect node_cpu_data"

source monitoring/lib/performance_data_line_builder.sh
line="$(build_performance_data_line false \
    "2026-07-06 00:00:00" \
    "1,2,3,4,5,6" \
    "7,8,9" \
    "10,11" \
    "eth0,12,13" \
    "" \
    "14,15" \
    "16,17,18,1,1,0,absolute_gap,healthy,0,block,0,null" \
    "1" \
    "2" \
    "true" \
    "3,3000000,chain_insert_mgasps,available,none,,,,,,,,,,unsupported" \
    "123,4,5,124,geth,6,7,124:geth:6:7,7,8,7:8,9,10,2048,25,available" \
    "21,22" \
    "other")"

expected_cols=$((1 + 6 + 3 + 2 + 3 + 2 + 12 + 3 + 15 + 16 + 2 + 1))
actual_cols="$(echo "$line" | awk -F, '{print NF}')"
[[ "$actual_cols" -eq "$expected_cols" ]] || fail "data line field count mismatch: expected $expected_cols got $actual_cols"

echo "✅ node execution CSV wiring passed"
