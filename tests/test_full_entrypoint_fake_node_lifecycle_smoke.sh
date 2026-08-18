#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEST_ROOT="$(mktemp -d)"
RUN_LOG="$TEST_ROOT/full_entrypoint.log"
SESSION_ID="$(date +%Y%m%d_%H%M%S)_smoke"

cleanup() {
    set +e
    if [[ -n "${TMP_DIR:-}" ]]; then
        rm -f "$TMP_DIR/qps_test_status" "$TMP_DIR/qps_test_status.tmp"
    fi
    pkill -f "$REPO_ROOT/monitoring/monitoring_coordinator.sh start" 2>/dev/null || true
    pkill -f "$REPO_ROOT/monitoring/unified_monitor.sh" 2>/dev/null || true
    pkill -f "$REPO_ROOT/monitoring/block_height_monitor.sh" 2>/dev/null || true
    pkill -f "$REPO_ROOT/tools/disk_bottleneck_detector.sh -b" 2>/dev/null || true
    pkill -f "$REPO_ROOT/tools/proxy/cmd/proxy" 2>/dev/null || true
    pkill -f "$REPO_ROOT/tools/fake-node" 2>/dev/null || true
    rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*"
    echo "--- full entrypoint log tail ---"
    tail -120 "$RUN_LOG" 2>/dev/null || true
    echo "--- generated tree ---"
    find "${DATA_DIR:-$TEST_ROOT}" -maxdepth 4 -type f 2>/dev/null | sort || true
    exit 1
}

assert_file() {
    local path="$1"
    local description="$2"
    [[ -s "$path" ]] || fail "$description missing or empty: $path"
    echo "✅ $description"
}

assert_csv_data() {
    local path="$1"
    local description="$2"
    [[ -s "$path" ]] || fail "$description missing: $path"
    local rows
    rows="$(wc -l < "$path")"
    [[ "$rows" -gt 1 ]] || fail "$description has no data rows: $path"
    echo "✅ $description ($rows lines)"
}

assert_no_live_pids_from_file() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] || return 0
    while IFS= read -r pid; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        if kill -0 "$pid" 2>/dev/null; then
            fail "monitor pid still alive after entrypoint exit: $pid"
        fi
    done < "$pid_file"
    echo "✅ monitor PID registry has no live PIDs"
}

export BLOCKCHAIN_BENCHMARK_DATA_DIR="$TEST_ROOT/result"
export SESSION_TIMESTAMP="$SESSION_ID"
export BLOCKCHAIN_NODE="solana"
export RPC_MODE="mixed"
export QUICK_INITIAL_QPS=1
export QUICK_MAX_QPS=1
export QUICK_QPS_STEP=1
export QUICK_DURATION=3
export QPS_WARMUP_DURATION=0
export QPS_COOLDOWN=0
export BLOCK_HEIGHT_CURL_TIMEOUT=1
export ENA_MONITOR_ENABLED=false
export FORCE_CONFIG_RELOAD=true

set +u
# shellcheck source=/dev/null
source config/config_loader.sh >"$TEST_ROOT/config_loader.log" 2>&1
set -u

rm -rf "$DATA_DIR" "$MEMORY_SHARE_DIR"
mkdir -p "$LOGS_DIR" "$REPORTS_DIR" "$VEGETA_RESULTS_DIR" "$TMP_DIR" "$ARCHIVES_DIR" "$MEMORY_SHARE_DIR" "$NODE_HEALTH_CACHE_DIR"

cat > "$ACCOUNTS_OUTPUT_FILE" <<'EOF'
11111111111111111111111111111111
TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
SysvarRent111111111111111111111111111111111
ComputeBudget111111111111111111111111111111
EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v
EOF

if ! timeout 120s env -u SHELLOPTS \
    BLOCKCHAIN_BENCHMARK_DATA_DIR="$BLOCKCHAIN_BENCHMARK_DATA_DIR" \
    SESSION_TIMESTAMP="$SESSION_TIMESTAMP" \
    BLOCKCHAIN_NODE="$BLOCKCHAIN_NODE" \
    RPC_MODE="$RPC_MODE" \
    QUICK_INITIAL_QPS="$QUICK_INITIAL_QPS" \
    QUICK_MAX_QPS="$QUICK_MAX_QPS" \
    QUICK_QPS_STEP="$QUICK_QPS_STEP" \
    QUICK_DURATION="$QUICK_DURATION" \
    QPS_WARMUP_DURATION="$QPS_WARMUP_DURATION" \
    QPS_COOLDOWN="$QPS_COOLDOWN" \
    BLOCK_HEIGHT_CURL_TIMEOUT="$BLOCK_HEIGHT_CURL_TIMEOUT" \
    ENA_MONITOR_ENABLED="$ENA_MONITOR_ENABLED" \
    FORCE_CONFIG_RELOAD="$FORCE_CONFIG_RELOAD" \
    bash ./blockchain_node_benchmark.sh \
        --quick \
        --mixed \
        --fake-node \
        --initial-qps 1 \
        --max-qps 1 \
        --step-qps 1 \
        --duration 3 >"$RUN_LOG" 2>&1; then
    fail "full fake-node entrypoint smoke failed"
fi

archive_dir="$(find "$ARCHIVES_DIR" -maxdepth 1 -type d -name 'run_*' | sort | tail -1)"
[[ -n "$archive_dir" ]] || fail "archive run directory not generated"

[[ ! -e "$TMP_DIR/qps_test_status" ]] || fail "lifecycle marker still exists after entrypoint exit"
[[ ! -e "$archive_dir/tmp/qps_test_status" ]] || fail "archived lifecycle marker should not exist"
echo "✅ lifecycle marker removed"

assert_no_live_pids_from_file "$archive_dir/tmp/monitor_pids.txt"

archived_proxy_method_csv="$archive_dir/logs/proxy_method.csv"
archived_performance_csv="$archive_dir/logs/performance_${SESSION_TIMESTAMP}.csv"
archived_block_height_csv="$archive_dir/logs/block_height_monitor_${SESSION_TIMESTAMP}.csv"

assert_csv_data "$archived_proxy_method_csv" "archived proxy method CSV"
assert_csv_data "$archived_performance_csv" "archived performance CSV"
assert_csv_data "$archived_block_height_csv" "archived block-height monitor CSV"

vegeta_json="$(find "$archive_dir/vegeta_results" -maxdepth 1 -name 'vegeta_1qps_*.json' -type f | head -1)"
[[ -n "$vegeta_json" ]] || fail "vegeta JSON result not generated"
assert_file "$vegeta_json" "archived vegeta JSON result"

requests="$(jq -r '.requests // 0' "$vegeta_json")"
success_200="$(jq -r '.status_codes."200" // 0' "$vegeta_json")"
[[ "$requests" -gt 0 ]] || fail "vegeta JSON has no requests"
[[ "$success_200" -gt 0 ]] || fail "vegeta JSON has no HTTP 200 responses"
echo "✅ vegeta produced requests=$requests http_200=$success_200"

html_report="$(find "$archive_dir/reports" -maxdepth 1 -name 'performance_report_*.html' -type f | head -1)"
[[ -n "$html_report" ]] || fail "HTML performance report not generated"
assert_file "$html_report" "archived HTML performance report"

for subdir in logs reports vegeta_results stats; do
    [[ -d "$archive_dir/$subdir" ]] || fail "archive missing $subdir directory: $archive_dir"
done
assert_file "$archive_dir/test_summary.json" "archive summary"
summary_start_time="$(jq -r '.start_time // ""' "$archive_dir/test_summary.json")"
summary_end_time="$(jq -r '.end_time // ""' "$archive_dir/test_summary.json")"
[[ -n "$summary_start_time" ]] || fail "archive summary has no observed start_time"
[[ -n "$summary_end_time" ]] || fail "archive summary has no observed end_time"
echo "✅ archive directory structure"

bash tools/audit_monitoring_lifecycle.sh >/dev/null
echo "✅ lifecycle static audit"

echo "✅ full entrypoint fake-node lifecycle smoke test passed"
