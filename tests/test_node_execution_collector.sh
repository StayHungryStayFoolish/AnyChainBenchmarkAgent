#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COLLECTOR="monitoring/node_execution_collector.py"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

count_fields() {
    awk -F, '{print NF}'
}

execution_header="$(python3 "$COLLECTOR" --execution-header)"
execution_data_unconfigured="$(NODE_PROMETHEUS_METRICS_URL= python3 "$COLLECTOR" --execution-data)"
[[ "$(echo "$execution_header" | count_fields)" -eq 4 ]]
[[ "$(echo "$execution_data_unconfigured" | count_fields)" -eq 4 ]]
[[ "$execution_data_unconfigured" == *"unavailable"* ]]

cat > "$TMP_DIR/metrics.prom" <<'PROM'
# HELP chain_insert_mgasps Imported MGas per second
chain_insert_mgasps 12.5
PROM
execution_data="$(NODE_PROMETHEUS_METRICS_URL="file://$TMP_DIR/metrics.prom" python3 "$COLLECTOR" --execution-data)"
[[ "$execution_data" == "12.50,12500000.00,chain_insert_mgasps,available" ]]

cat > "$TMP_DIR/gas.prom" <<'PROM'
reth_sync_execution_gas_per_second 42000000
PROM
gas_data="$(NODE_PROMETHEUS_METRICS_URL="file://$TMP_DIR/gas.prom" python3 "$COLLECTOR" --execution-data)"
[[ "$gas_data" == "42.00,42000000.00,reth_sync_execution_gas_per_second,available" ]]

node_header="$(python3 "$COLLECTOR" --node-cpu-header)"
[[ "$(echo "$node_header" | count_fields)" -eq 14 ]]

state_file="$TMP_DIR/node_cpu_state.json"
node_data_1="$(NODE_PROCESS_PID="$$" NODE_CPU_STATE_FILE="$state_file" python3 "$COLLECTOR" --node-cpu-data)"
node_data_2="$(NODE_PROCESS_PID="$$" NODE_CPU_STATE_FILE="$state_file" python3 "$COLLECTOR" --node-cpu-data)"
[[ "$(echo "$node_data_1" | count_fields)" -eq 14 ]]
[[ "$(echo "$node_data_2" | count_fields)" -eq 14 ]]

fake_proc="$TMP_DIR/proc"
mkdir -p "$fake_proc/101" "$fake_proc/102"
cat > "$fake_proc/101/comm" <<'EOF_COMM'
tail
EOF_COMM
printf 'tail\0-F\0/tmp/blockchain-node-benchmark/logs/performance_latest.csv\0' > "$fake_proc/101/cmdline"
cat > "$fake_proc/102/comm" <<'EOF_COMM'
agave-validator
EOF_COMM
printf '/usr/bin/agave-validator\0--identity\0validator.json\0' > "$fake_proc/102/cmdline"

no_false_match="$(PROC_ROOT="$fake_proc" BLOCKCHAIN_PROCESS_NAMES_STR="blockchain" NODE_CPU_STATE_FILE="$TMP_DIR/no_false_match.json" python3 "$COLLECTOR" --node-cpu-data)"
[[ "$no_false_match" == *"unavailable" ]]

exact_match="$(PROC_ROOT="$fake_proc" BLOCKCHAIN_PROCESS_NAMES_STR="agave-validator" NODE_CPU_STATE_FILE="$TMP_DIR/exact_match.json" python3 "$COLLECTOR" --node-cpu-data)"
[[ "$exact_match" == *"unavailable" ]] || [[ "$(echo "$exact_match" | cut -d, -f1)" == "102" ]]

echo "✅ node execution collector contract passed"
