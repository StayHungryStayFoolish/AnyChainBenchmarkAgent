#!/usr/bin/env bash
# =====================================================================
# Monitoring Lifecycle Static Audit
# =====================================================================
# Read-only checks for monitor ownership, PID registry, lifecycle marker, and
# known fallback paths. This intentionally avoids starting processes.
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

failures=0
warnings=0

log_ok() { echo "✅ $*"; }
log_warn() { warnings=$((warnings + 1)); echo "⚠️  $*" >&2; }
log_fail() { failures=$((failures + 1)); echo "❌ $*" >&2; }

require_pattern() {
    local file="$1"
    local pattern="$2"
    local label="$3"

    if grep -qE "$pattern" "$PROJECT_ROOT/$file"; then
        log_ok "$label"
    else
        log_fail "$label missing in $file"
        echo "   pattern: $pattern" >&2
    fi
}

require_absent_pattern() {
    local file="$1"
    local pattern="$2"
    local label="$3"

    if grep -qE "$pattern" "$PROJECT_ROOT/$file"; then
        log_fail "$label found in $file"
        echo "   pattern: $pattern" >&2
    else
        log_ok "$label"
    fi
}

check_entry_lifecycle() {
    local file="blockchain_node_benchmark.sh"

    require_pattern "$file" '^trap cleanup_framework EXIT INT TERM' "entry has one cleanup trap"
    require_pattern "$file" '^start_monitoring_system\(\)' "entry defines start_monitoring_system"
    require_pattern "$file" '^stop_monitoring_system\(\)' "entry defines stop_monitoring_system"
    require_pattern "$file" 'qps_test_status\.tmp' "entry creates lifecycle marker atomically"
    require_pattern "$file" 'mv "\$TMP_DIR/qps_test_status\.tmp" "\$TMP_DIR/qps_test_status"' "entry publishes lifecycle marker"
    require_pattern "$file" 'rm -f "\$TMP_DIR/qps_test_status"' "entry removes lifecycle marker after QPS"
    require_pattern "$file" 'monitoring/monitoring_coordinator\.sh" start &' "entry starts monitoring coordinator"
    require_pattern "$file" 'MONITOR_PIDS_FILE="\$\{TMP_DIR\}/monitor_pids\.txt"' "entry exports monitor PID registry"
    require_pattern "$file" 'MONITOR_STATUS_FILE="\$\{TMP_DIR\}/monitoring_status\.json"' "entry exports monitor status registry"
}

check_coordinator_lifecycle() {
    local file="monitoring/monitoring_coordinator.sh"

    require_pattern "$file" '^declare -A MONITOR_TASKS=' "coordinator declares monitor task registry"
    for task in unified block_height network disk_bottleneck iostat; do
        require_pattern "$file" "\\[\"$task\"\\]=" "coordinator registers $task"
    done
    require_absent_pattern "$file" '\["ena_network"\]=' "coordinator no longer registers legacy ena_network task"

    require_pattern "$file" '^init_coordinator\(\)' "coordinator defines init_coordinator"
    require_pattern "$file" '^start_monitor\(\)' "coordinator defines start_monitor"
    require_pattern "$file" '^stop_monitor\(\)' "coordinator defines stop_monitor"
    require_pattern "$file" '^start_all_monitors\(\)' "coordinator defines start_all_monitors"
    require_pattern "$file" '^stop_all_monitors\(\)' "coordinator defines stop_all_monitors"
    require_pattern "$file" '^cleanup_coordinator\(\)' "coordinator defines cleanup_coordinator"
    require_pattern "$file" '^find_monitor_pids\(\)' "coordinator defines filtered monitor PID discovery"
    require_pattern "$file" 'bash -lc' "coordinator excludes shell launcher false positives"
    require_pattern "$file" 'bash -n' "coordinator excludes syntax-check false positives"
    require_pattern "$file" '^trap cleanup_coordinator EXIT INT TERM' "coordinator owns cleanup trap"
    require_pattern "$file" 'echo "\$monitor_name:\$pid" >> "\$MONITOR_PIDS_FILE"' "coordinator records child PIDs"
    require_pattern "$file" 'while true; do' "coordinator stays alive during QPS lifecycle"
    require_pattern "$file" '\[\[ -f "\$TMP_DIR/qps_test_status" \]\]' "coordinator follows lifecycle marker"
    require_pattern "$file" 'BLOCK_HEIGHT_CACHE_FILE' "coordinator cleanup uses registered block-height cache path"

    local registered
    registered="$(grep -oE '\["[^"]+"\]=' "$PROJECT_ROOT/$file" | sed 's/\["//;s/"\]=//' | sort -u)"
    local referenced
    referenced="$(grep -oE 'start_monitor "[^"]+"' "$PROJECT_ROOT/$file" \
        | sed 's/start_monitor "//;s/"//' \
        | grep -v '^\$' \
        | sort -u || true)"
    local invalid
    invalid="$(comm -13 <(printf '%s\n' "$registered") <(printf '%s\n' "$referenced") || true)"
    if [[ -n "$invalid" ]]; then
        log_fail "coordinator start_monitor references only registered monitor names"
        echo "$invalid" | sed 's/^/   invalid: /' >&2
    else
        log_ok "coordinator start_monitor references only registered monitor names"
    fi
}

check_monitor_contracts() {
    require_pattern "monitoring/unified_monitor.sh" 'while \[\[ -f "\$TMP_DIR/qps_test_status" \]\]; do' "unified monitor follows lifecycle marker"
    require_pattern "monitoring/unified_monitor.sh" 'PERFORMANCE_LATEST_CSV' "unified monitor writes registered latest performance path"
    require_pattern "monitoring/unified_monitor.sh" 'next_sample_count' "unified monitor updates registered sample counter helper"

    require_pattern "monitoring/block_height_monitor.sh" 'while \[\[ -f "\$TMP_DIR/qps_test_status" \]\]; do' "block-height monitor follows lifecycle marker"
    require_pattern "monitoring/block_height_monitor.sh" 'BLOCK_HEIGHT_CACHE_FILE' "block-height monitor uses registered cache path"
    require_absent_pattern "monitoring/block_height_monitor.sh" 'rm -f "\$MEMORY_SHARE_DIR"/block_height_monitor_cache\.json' "block-height monitor does not hardcode cache cleanup"

    require_pattern "tools/disk_bottleneck_detector.sh" 'PERFORMANCE_LATEST_CSV' "disk bottleneck detector reads registered latest performance path"
    require_pattern "tools/disk_bottleneck_detector.sh" 'while \[\[ -f "\$TMP_DIR/qps_test_status" \]\]; do' "disk bottleneck detector polls lifecycle marker without a persistent tail pipeline"
    require_pattern "monitoring/iostat_collector.sh" '^stop_iostat_collectors\(\)' "iostat collector owns an explicit cleanup operation"
    require_pattern "monitoring/unified_monitor.sh" 'stop_iostat_collectors' "unified monitor invokes module-owned iostat cleanup"

    require_pattern "core/master_qps_executor.sh" 'disk_bottleneck_detector\.sh\.\*-b' "QPS executor checks coordinator disk detector before fallback"
    require_pattern "core/master_qps_executor.sh" 'MONITOR_PIDS_FILE' "QPS executor records fallback monitor PID"
}

echo "🔎 Monitoring lifecycle static audit"
check_entry_lifecycle
check_coordinator_lifecycle
check_monitor_contracts

echo ""
echo "Lifecycle audit summary: failures=$failures warnings=$warnings"
if [[ "$failures" -gt 0 ]]; then
    exit 1
fi
exit 0
