#!/bin/bash

# =====================================================================
# Blockchain Node QPS Testing Framework Master Controller - Pure QPS Testing Engine
# Master QPS Executor - Core QPS Testing Engine Only
# =====================================================================

# Load shared functions and configurations
QPS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${QPS_SCRIPT_DIR}/common_functions.sh"
source "${QPS_SCRIPT_DIR}/../config/config_loader.sh"
# CSV Schema Registry for provider-aware reader column resolution.
source "${QPS_SCRIPT_DIR}/../config/csv_schema_registry.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../utils/unified_logger.sh"

# Initialize unified logger
init_logger "master_qps_executor" $LOG_LEVEL "${LOGS_DIR}/master_qps_executor.log"

# Redirect stdout to log file (keep stderr for log_* functions)
exec 1> >(tee -a "${LOGS_DIR}/master_qps_executor.log")

source "${QPS_SCRIPT_DIR}/../utils/error_handler.sh"

# Setup error handling
setup_error_handling "$(basename "${BASH_SOURCE[0]}")" "QPS Testing Engine"
log_script_start "$(basename "$0")"

# Validate required environment variables
if [[ -z "${MONITOR_PIDS_FILE:-}" ]]; then
    log_warn "⚠️  MONITOR_PIDS_FILE environment variable not set, using default value"
    export MONITOR_PIDS_FILE="${TMP_DIR}/monitor_pids.txt"
fi

# Global variables
readonly PROGRAM_NAME="Blockchain Node QPS Benchmark Engine"
readonly VERSION="v2.1"
readonly BENCHMARK_MODES=("quick" "standard" "intensive")
readonly RPC_MODES=("single" "mixed")

# Benchmark parameters - directly use values from user_config.sh
# Note: All default values come from user_config.sh to ensure configuration consistency
BENCHMARK_MODE=""
RPC_MODE="single"
INITIAL_QPS=$QUICK_INITIAL_QPS    # From user_config.sh: QUICK_INITIAL_QPS=1000
MAX_QPS=$QUICK_MAX_QPS           # From user_config.sh: QUICK_MAX_QPS=3000
STEP_QPS=$QUICK_QPS_STEP         # From user_config.sh: QUICK_QPS_STEP=500
DURATION=""
CUSTOM_PARAMS=false

# Bottleneck detection status
BOTTLENECK_DETECTED=false
BOTTLENECK_COUNT=0
LAST_SUCCESSFUL_QPS=0

# Display help information
show_help() {
    cat << EOF
🚀 $PROGRAM_NAME $VERSION

📋 Usage:
    $0 [test mode] [RPC mode] [custom parameters]

🎯 Benchmark Modes:
    --quick     Quick benchmark test
    --standard  Standard benchmark test
    --intensive Intensive benchmark test (automatic bottleneck detection)

🔗 RPC Modes:
    --single    Single RPC method test (default: getAccountInfo)
    --mixed     Mixed RPC method test (multiple methods combined)

⚙️ Custom Parameters:
    --initial-qps NUM    Initial QPS (default: $QUICK_INITIAL_QPS)
    --max-qps NUM        Maximum QPS (default: depends on test mode)
    --step-qps NUM       QPS step size (default: $QUICK_QPS_STEP)
    --duration NUM       Duration per level (seconds)

📊 Other Options:
    --status    Display current test status
    --version   Display version information
    --help      Display this help information

📖 Examples:
    $0 --intensive --mixed
    $0 --quick --single --initial-qps 500 --max-qps 2000
    $0 --standard --mixed --duration 300

EOF
}

# Display version information
show_version() {
    echo "$PROGRAM_NAME $VERSION"
}

# Display test status
show_status() {
    echo "📊 QPS Testing Engine Status"
    echo "=================="
    
    # Check if vegeta is available
    if command -v vegeta >/dev/null 2>&1; then
        echo "✅ Vegeta: $(vegeta --version 2>&1 | head -1)"
    else
        echo "❌ Vegeta: Not installed (required v12.13.0+ — run: ./blockchain_node_benchmark.sh --install-vegeta)"
    fi
    
    # Check target files
    if [[ -f "$SINGLE_METHOD_TARGETS_FILE" ]]; then
        echo "✅ Single method target file: $(wc -l < "$SINGLE_METHOD_TARGETS_FILE") targets"
    else
        echo "❌ Single method target file: Does not exist"
    fi
    
    if [[ -f "$MIXED_METHOD_TARGETS_FILE" ]]; then
        echo "✅ Mixed method target file: $(wc -l < "$MIXED_METHOD_TARGETS_FILE") targets"
    else
        echo "❌ Mixed method target file: Does not exist"
    fi
    
    # Check RPC connection
    echo "🔗 RPC Connection Test:"
    if get_block_height "$LOCAL_RPC_URL" >/dev/null 2>&1; then
        echo "✅ Local RPC: $LOCAL_RPC_URL"
    else
        echo "❌ Local RPC: $LOCAL_RPC_URL (Connection failed)"
    fi
    
    # Check monitoring status
    if pgrep -f "monitoring.*coordinator" > /dev/null; then
        echo "✅ Monitoring system: Running"
    else
        echo "⚠️ Monitoring system: Not running"
    fi
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                BENCHMARK_MODE="quick"
                INITIAL_QPS=$QUICK_INITIAL_QPS
                MAX_QPS=$QUICK_MAX_QPS
                STEP_QPS=$QUICK_QPS_STEP
                DURATION=$QUICK_DURATION
                shift
                ;;
            --standard)
                BENCHMARK_MODE="standard"
                INITIAL_QPS=$STANDARD_INITIAL_QPS
                MAX_QPS=$STANDARD_MAX_QPS
                STEP_QPS=$STANDARD_QPS_STEP
                DURATION=$STANDARD_DURATION
                shift
                ;;
            --intensive)
                BENCHMARK_MODE="intensive"
                INITIAL_QPS=$INTENSIVE_INITIAL_QPS
                MAX_QPS=$INTENSIVE_MAX_QPS
                STEP_QPS=$INTENSIVE_QPS_STEP
                DURATION=$INTENSIVE_DURATION
                shift
                ;;
            --single)
                RPC_MODE="single"
                shift
                ;;
            --mixed)
                RPC_MODE="mixed"
                shift
                ;;
            --fake-node|--no-proxy)
                # These two arguments are consumed by the main entrypoint parse_rpc_mode_args
                # (--fake-node→FAKE_NODE_MODE, --no-proxy→SKIP_RPC_PROXY),
                # but the main entrypoint forwards the original argument list to this executor.
                # Ignore them here so they do not hit the default hard-fail branch.
                shift
                ;;
            --initial-qps)
                INITIAL_QPS="$2"
                CUSTOM_PARAMS=true
                shift 2
                ;;
            --max-qps)
                MAX_QPS="$2"
                CUSTOM_PARAMS=true
                shift 2
                ;;
            --step-qps)
                STEP_QPS="$2"
                CUSTOM_PARAMS=true
                shift 2
                ;;
            --duration)
                DURATION="$2"
                CUSTOM_PARAMS=true
                shift 2
                ;;
            --status)
                show_status
                exit 0
                ;;
            --version)
                show_version
                exit 0
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "❌ Unknown parameter: $1"
                echo "💡 Use --help to view help information"
                exit 1
                ;;
        esac
    done
    
    # Set default benchmark mode
    if [[ -z "$BENCHMARK_MODE" ]]; then
        BENCHMARK_MODE="quick"
        INITIAL_QPS=$QUICK_INITIAL_QPS
        MAX_QPS=$QUICK_MAX_QPS
        STEP_QPS=$QUICK_QPS_STEP
        DURATION=$QUICK_DURATION
    fi
}

# Display benchmark configuration
show_benchmark_config() {
    echo "⚙️ QPS Benchmark Configuration"
    echo "=================="
    echo "Benchmark mode: $BENCHMARK_MODE"
    echo "RPC mode:      $RPC_MODE"
    echo "Initial QPS:   $INITIAL_QPS"
    echo "Maximum QPS:   $MAX_QPS"
    echo "QPS step:      $STEP_QPS"
    echo "Duration:      ${DURATION} seconds"
    echo "Local RPC:     $LOCAL_RPC_URL"
    echo ""
}

# Pre-check system environment
pre_check() {
    echo "🔍 Performing pre-check..."
    
    # Check vegeta; benchmark modes require it.
    if ! command -v vegeta >/dev/null 2>&1; then
        echo "❌ Error: vegeta not installed (required: v12.13.0+)"
        echo "💡 Recommended installation (verified version: v12.13.0):"
        echo "   curl -sLO https://github.com/tsenart/vegeta/releases/download/v12.13.0/vegeta_12.13.0_linux_amd64.tar.gz"
        echo "   tar -xzf vegeta_12.13.0_linux_amd64.tar.gz && sudo mv vegeta /usr/local/bin/ && chmod +x /usr/local/bin/vegeta"
        echo "   # Or add it to PATH:  export PATH=\"\$HOME/.local/bin:\$PATH\"   (add this to ~/.bashrc to make it permanent)"
        echo "💡 Or install it automatically: ./blockchain_node_benchmark.sh --install-vegeta"
        return 1
    fi
    
    # Check target file
    local targets_file
    if [[ "$RPC_MODE" == "mixed" ]]; then
        targets_file="$MIXED_METHOD_TARGETS_FILE"
    else
        targets_file="$SINGLE_METHOD_TARGETS_FILE"
    fi
    
    if [[ ! -f "$targets_file" ]]; then
        echo "❌ Error: Target file does not exist: $targets_file"
        echo "💡 Please ensure vegeta target file is generated"
        return 1
    fi
    
    # Check RPC connection
    if ! get_block_height "$LOCAL_RPC_URL" >/dev/null 2>&1; then
        echo "❌ Error: Cannot connect to RPC endpoint: $LOCAL_RPC_URL"
        return 1
    fi
    
    echo "✅ Pre-check passed"
    return 0
}

# Read provider-aware fields from monitoring JSON through the registry.
#   provider comes from the JSON cloud_provider field; do not re-detect at analysis time.
#   logical: disk_iops_provider_adjusted | disk_throughput_provider_adjusted
#   prefix:  data_<device> | accounts_<device>
# jq // 0 is used only when the resolved field is absent.
_mqe_provider_field() {
    local json="$1" logical="$2" prefix="$3"
    local provider
    provider=$(echo "$json" | jq -r '.cloud_provider // "other"' 2>/dev/null || echo "other")
    case "$provider" in aws|gcp|other) : ;; *) provider="other" ;; esac
    local col=""
    if declare -F csv_registry_resolve >/dev/null 2>&1; then
        col="$(csv_registry_resolve "$logical" "$provider" "$prefix" 2>/dev/null || echo "")"
    fi
    if [[ -z "$col" ]]; then
        echo "0"; return
    fi
    echo "$json" | jq -r ".\"${col}\" // 0" 2>/dev/null || echo "0"
}
# Check bottleneck status
check_bottleneck_during_test() {
    local current_qps=$1
    
    # Read latest monitoring data
    local latest_data=$(get_latest_monitoring_data)
    if [[ -z "$latest_data" ]]; then
        return 0  # No monitoring data, continue testing
    fi
    
    local bottleneck_found=false
    local bottleneck_reasons=()
    local bottleneck_severity="low"
    
    # Check CPU bottleneck
    local cpu_usage=$(echo "$latest_data" | jq -r '.cpu_usage // 0' 2>/dev/null || echo "0")
    if (( $(awk "BEGIN {print ($cpu_usage > $BOTTLENECK_CPU_THRESHOLD) ? 1 : 0}") )); then
        bottleneck_found=true
        local severity="Moderate"
        if (( $(awk "BEGIN {print ($cpu_usage > 95) ? 1 : 0}") )); then
            severity="Severe"
            bottleneck_severity="high"
        fi
        bottleneck_reasons+=("CPU usage: ${cpu_usage}% > ${BOTTLENECK_CPU_THRESHOLD}% ($severity)")
    fi
    
    # Check memory bottleneck
    local mem_usage=$(echo "$latest_data" | jq -r '.memory_usage // 0' 2>/dev/null || echo "0")
    if (( $(awk "BEGIN {print ($mem_usage > $BOTTLENECK_MEMORY_THRESHOLD) ? 1 : 0}") )); then
        bottleneck_found=true
        local severity="Moderate"
        if (( $(awk "BEGIN {print ($mem_usage > 95) ? 1 : 0}") )); then
            severity="Severe"
            bottleneck_severity="high"
        fi
        bottleneck_reasons+=("Memory usage: ${mem_usage}% > ${BOTTLENECK_MEMORY_THRESHOLD}% ($severity)")
    fi
    
    # Check DATA device disk-standardized IOPS bottleneck
    local data_adjusted_iops=$(_mqe_provider_field "$latest_data" disk_iops_provider_adjusted "data_${LEDGER_DEVICE}")
    local data_provisioned_iops=${DATA_VOL_MAX_IOPS:-30000}
    local data_iops_util=$(awk "BEGIN {printf \"%.2f\", ($data_adjusted_iops / $data_provisioned_iops) * 100}")
    
    if (( $(awk "BEGIN {print ($data_iops_util > $BOTTLENECK_DISK_IOPS_THRESHOLD) ? 1 : 0}") )); then
        bottleneck_found=true
        bottleneck_reasons+=("DATA Disk IOPS: ${data_adjusted_iops}/${data_provisioned_iops} (${data_iops_util}%)")
    fi
    
    # Check DATA device disk-standardized Throughput bottleneck
    local data_adjusted_throughput=$(_mqe_provider_field "$latest_data" disk_throughput_provider_adjusted "data_${LEDGER_DEVICE}")
    local data_provisioned_throughput=${DATA_VOL_MAX_THROUGHPUT:-4000}
    local data_throughput_util=$(awk "BEGIN {printf \"%.2f\", ($data_adjusted_throughput / $data_provisioned_throughput) * 100}")
    
    if (( $(awk "BEGIN {print ($data_throughput_util > $BOTTLENECK_DISK_THROUGHPUT_THRESHOLD) ? 1 : 0}") )); then
        bottleneck_found=true
        bottleneck_reasons+=("DATA Disk Throughput: ${data_adjusted_throughput}/${data_provisioned_throughput} MiB/s (${data_throughput_util}%)")
    fi
    
    # Check ACCOUNTS device (if configured)
    if [[ -n "${ACCOUNTS_DEVICE:-}" && -n "${ACCOUNTS_VOL_TYPE:-}" && -n "${ACCOUNTS_VOL_MAX_IOPS:-}" ]]; then
        # ACCOUNTS device disk-standardized IOPS bottleneck
        local accounts_adjusted_iops=$(_mqe_provider_field "$latest_data" disk_iops_provider_adjusted "accounts_${ACCOUNTS_DEVICE}")
        local accounts_provisioned_iops=${ACCOUNTS_VOL_MAX_IOPS:-30000}
        local accounts_iops_util=$(awk "BEGIN {printf \"%.2f\", ($accounts_adjusted_iops / $accounts_provisioned_iops) * 100}")
        
        if (( $(awk "BEGIN {print ($accounts_iops_util > $BOTTLENECK_DISK_IOPS_THRESHOLD) ? 1 : 0}") )); then
            bottleneck_found=true
            bottleneck_reasons+=("ACCOUNTS Disk IOPS: ${accounts_adjusted_iops}/${accounts_provisioned_iops} (${accounts_iops_util}%)")
        fi
        
        # ACCOUNTS device disk-standardized Throughput bottleneck
        local accounts_adjusted_throughput=$(_mqe_provider_field "$latest_data" disk_throughput_provider_adjusted "accounts_${ACCOUNTS_DEVICE}")
        local accounts_provisioned_throughput=${ACCOUNTS_VOL_MAX_THROUGHPUT:-4000}
        local accounts_throughput_util=$(awk "BEGIN {printf \"%.2f\", ($accounts_adjusted_throughput / $accounts_provisioned_throughput) * 100}")
        
        if (( $(awk "BEGIN {print ($accounts_throughput_util > $BOTTLENECK_DISK_THROUGHPUT_THRESHOLD) ? 1 : 0}") )); then
            bottleneck_found=true
            bottleneck_reasons+=("ACCOUNTS Disk Throughput: ${accounts_adjusted_throughput}/${accounts_provisioned_throughput} MiB/s (${accounts_throughput_util}%)")
        fi
    fi
    
    # Check Disk latency bottleneck
    local disk_latency=$(echo "$latest_data" | jq -r '.disk_latency // 0' 2>/dev/null || echo "0")
    if (( $(awk "BEGIN {print ($disk_latency > $BOTTLENECK_DISK_LATENCY_THRESHOLD) ? 1 : 0}") )); then
        bottleneck_found=true
        bottleneck_severity="high"
        bottleneck_reasons+=("Disk latency: ${disk_latency}ms > ${BOTTLENECK_DISK_LATENCY_THRESHOLD}ms (Severe)")
    fi
    
    # Check network bottleneck
    local network_util=$(echo "$latest_data" | jq -r '.network_util // 0' 2>/dev/null || echo "0")
    if (( $(awk "BEGIN {print ($network_util > $BOTTLENECK_NETWORK_THRESHOLD) ? 1 : 0}") )); then
        bottleneck_found=true
        bottleneck_reasons+=("Network utilization: ${network_util}% > ${BOTTLENECK_NETWORK_THRESHOLD}%")
    fi
    
    # Check error rate bottleneck
    local error_rate=$(echo "$latest_data" | jq -r '.error_rate // 0' 2>/dev/null || echo "0")
    if (( $(awk "BEGIN {print ($error_rate > $BOTTLENECK_ERROR_RATE_THRESHOLD) ? 1 : 0}") )); then
        bottleneck_found=true
        bottleneck_severity="high"
        bottleneck_reasons+=("Error rate: ${error_rate}% > ${BOTTLENECK_ERROR_RATE_THRESHOLD}% (Severe)")
    fi
    
    # Resource bottleneck detection: increment counter
    if [[ "$bottleneck_found" == "true" ]]; then
        BOTTLENECK_COUNT=$((BOTTLENECK_COUNT + 1))
        echo "⚠️ Bottleneck detected ($BOTTLENECK_COUNT/${BOTTLENECK_CONSECUTIVE_COUNT}): ${bottleneck_reasons[*]}"
    fi
    
    # Call bottleneck_detector for comprehensive judgment (including node health check) regardless of resource bottleneck detection
    if ! trigger_immediate_bottleneck_analysis "$current_qps" "$bottleneck_severity" "${bottleneck_reasons[*]}"; then
        # bottleneck_detector returns 1: false positive or normal
        if [[ "$bottleneck_found" == "true" ]]; then
            # Scenario A: Resource bottleneck + Node healthy → False positive, reset counter
            echo "✅ bottleneck_detector determined as false positive (resource bottleneck but node healthy), resetting BOTTLENECK_COUNT"
            BOTTLENECK_COUNT=0
        fi
        # Scenario D: No resource bottleneck + Node healthy → Normal, no action
        return 0  # Continue testing
    fi
    
    # bottleneck_detector returns 0: Confirmed as true bottleneck
    if [[ "$bottleneck_found" == "false" ]]; then
        # Need to distinguish between Scenario A-RPC and Scenario C
        # Read bottleneck type saved by detector
        local is_rpc_bottleneck=false
        local bottleneck_status_file="${BOTTLENECK_STATUS_FILE:-${MEMORY_SHARE_DIR}/bottleneck_status.json}"
        
        if [[ -f "$bottleneck_status_file" ]]; then
            local bottleneck_types=$(jq -r '.bottleneck_types[]' "$bottleneck_status_file" 2>/dev/null || echo "")
            if echo "$bottleneck_types" | grep -qE "RPC_Success_Rate|RPC_Latency|RPC_Connection|error_rate"; then
                is_rpc_bottleneck=true
            fi
        fi
        
        if [[ "$is_rpc_bottleneck" == "true" ]]; then
            # Scenario A-RPC: RPC performance bottleneck + Node healthy → True bottleneck, accumulate count
            BOTTLENECK_COUNT=$((BOTTLENECK_COUNT + 1))
            echo "🚨 RPC performance bottleneck detected (mandatory condition) ($BOTTLENECK_COUNT/${BOTTLENECK_CONSECUTIVE_COUNT})"
            # Continue to subsequent logic to check if consecutive count reached
        else
            # Scenario C: No resource bottleneck + Node persistently unhealthy → Node failure
            echo "🚨 bottleneck_detector detected node persistently unhealthy (no resource bottleneck)"
            BOTTLENECK_DETECTED=true
            save_bottleneck_context "$current_qps" "Node_Unhealthy" "high"
            return 1  # Stop testing
        fi
    fi
    
    # Scenario B: Resource bottleneck + Node unhealthy → True bottleneck, continue accumulating count
    if [[ $BOTTLENECK_COUNT -ge $BOTTLENECK_CONSECUTIVE_COUNT ]]; then
        # Confirmed as true bottleneck after 3 consecutive detections, stop testing
        echo "🚨 True bottleneck detected ${BOTTLENECK_CONSECUTIVE_COUNT} consecutive times, stopping test"
        BOTTLENECK_DETECTED=true
        save_bottleneck_context "$current_qps" "${bottleneck_reasons[*]}" "$bottleneck_severity"
        return 1  # Stop testing
    fi
    
    # Consecutive count not reached, continue testing
    return 0
}

# Get latest monitoring data - enhanced version
get_latest_monitoring_data() {
    local monitoring_data="{}"
    
    # Try to read latest data from multiple data sources
    local data_sources=(
        "${LATEST_METRICS_FILE:-${MEMORY_SHARE_DIR}/latest_metrics.json}"
        "${UNIFIED_METRICS_FILE:-${MEMORY_SHARE_DIR}/unified_metrics.json}"
        "${PERFORMANCE_LATEST_CSV:-${LOGS_DIR}/performance_latest.csv}"
    )
    
    for source in "${data_sources[@]}"; do
        if [[ -f "$source" ]]; then
            case "$source" in
                *.json)
                    # JSON format data
                    local json_data=$(cat "$source" 2>/dev/null)
                    if [[ -n "$json_data" && "$json_data" != "{}" ]]; then
                        monitoring_data="$json_data"
                        break
                    fi
                    ;;
                *.csv)
                    # CSV format data, convert to JSON
                    monitoring_data=$(convert_csv_to_json "$source")
                    if [[ -n "$monitoring_data" && "$monitoring_data" != "{}" ]]; then
                        break
                    fi
                    ;;
            esac
        fi
    done
    
    # If no data found, try to get real-time data
    if [[ "$monitoring_data" == "{}" ]]; then
        monitoring_data=$(get_realtime_metrics)
    fi
    
    echo "$monitoring_data"
}

# Convert CSV data to JSON format
convert_csv_to_json() {
    local csv_file="$1"
    
    if [[ ! -f "$csv_file" ]]; then
        echo "{}"
        return
    fi
    
    # Read last line of CSV data
    local last_line=$(tail -n 1 "$csv_file" 2>/dev/null)
    if [[ -z "$last_line" ]]; then
        echo "{}"
        return
    fi
    
    # Simplified CSV to JSON conversion
    local json_data=$(python3 -c "
import sys, csv, json
try:
    with open('$csv_file', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            print(json.dumps(rows[-1]))
        else:
            print('{}')
except:
    print('{}')
" 2>/dev/null)
    
    echo "${json_data:-{}}"
}

# Get real-time metrics
get_realtime_metrics() {
    # Real-time metrics collection in Linux environment
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' 2>/dev/null || echo "0")
    local mem_usage=$(free | awk '/^Mem:/ {if($2>0) printf "%.1f", $3/$2 * 100; else print "0"}' 2>/dev/null || echo "0")
    
    # Build JSON
    local metrics=$(cat << EOF
{
    "timestamp": "$(date -Iseconds)",
    "cpu_usage": $cpu_usage,
    "memory_usage": $mem_usage,
    "disk_util": 0,
    "disk_latency": 0,
    "network_util": 0,
    "error_rate": 0
}
EOF
)
    
    echo "$metrics"
}

# Trigger immediate bottleneck analysis
trigger_immediate_bottleneck_analysis() {
    local qps=$1
    local severity=$2
    local reasons="$3"
    
    echo "🚨 Triggering bottleneck analysis, QPS: $qps, Severity: $severity"
    
    # Call bottleneck detector for real-time analysis and capture return value
    local bottleneck_detector_result=1  # Default: no bottleneck detected
    
    if [[ -f "${QPS_SCRIPT_DIR}/../monitoring/bottleneck_detector.sh" ]]; then
        echo "🔍 Performing real-time bottleneck analysis..."
        
        # Get latest performance data file
        local performance_csv="${PERFORMANCE_LATEST_CSV:-${LOGS_DIR}/performance_latest.csv}"
        # Get current QPS vegeta test result file
        local vegeta_result="${VEGETA_RESULTS_DIR}/vegeta_${qps}qps_${SESSION_TIMESTAMP}.json"
        
        if [[ -f "$performance_csv" ]]; then
            # Capture bottleneck_detector.sh return value, pass vegeta result file path
            if "${QPS_SCRIPT_DIR}/../monitoring/bottleneck_detector.sh" \
                detect "$qps" "$performance_csv" "$vegeta_result"; then
                # Return 0 = true bottleneck detected (resource bottleneck + node unhealthy or node persistently unhealthy)
                bottleneck_detector_result=0
                echo "🚨 bottleneck_detector confirmed as true bottleneck"
            else
                # Return 1 = false positive (resource bottleneck + node healthy) or normal
                bottleneck_detector_result=1
                echo "✅ bottleneck_detector determined as false positive or normal"
            fi
            
            # Wait for bottleneck detection to complete before continuing
            sleep 1
        else
            echo "⚠️  Performance data file does not exist, skipping bottleneck analysis: $performance_csv"
        fi
    fi
    
    # Check if already started via monitoring_coordinator.sh
    if pgrep -f "disk_bottleneck_detector.sh.*-b" >/dev/null 2>&1; then
        echo "💾 Disk bottleneck detector already started via monitoring coordinator, skipping duplicate start"
    else
        # Call Disk bottleneck detector
        if [[ -f "${QPS_SCRIPT_DIR}/../tools/disk_bottleneck_detector.sh" ]]; then
            echo "💾 Performing Disk bottleneck analysis..."
            "${QPS_SCRIPT_DIR}/../tools/disk_bottleneck_detector.sh" \
                --background &
            local disk_analysis_pid=$!
            echo "📊 Disk bottleneck analysis process started (PID: $disk_analysis_pid)"
            
            # Record PID to unified monitoring PID file
            echo "disk_analysis:$disk_analysis_pid" >> "$MONITOR_PIDS_FILE"
        fi
    fi
    
    # Log bottleneck event
    log_bottleneck_event "$qps" "$severity" "$reasons"
    
    # Return bottleneck_detector judgment result
    return $bottleneck_detector_result
}

# Log bottleneck event
log_bottleneck_event() {
    local qps=$1
    local severity=$2
    local reasons="$3"
    
    local event_data=$(cat << EOF
{
    "event_type": "bottleneck_detected",
    "timestamp": "$(date -Iseconds)",
    "qps": $qps,
    "severity": "$severity",
    "reasons": "$reasons",
    "benchmark_mode": "$BENCHMARK_MODE",
    "rpc_mode": "$RPC_MODE"
}
EOF
)
    
    # Save to event log
    local event_log="${LOGS_DIR}/bottleneck_events.jsonl"
    log_info "$event_data"
    
    echo "📝 Bottleneck event logged to: $(basename "$event_log")"
}

# Save bottleneck context - enhanced version
save_bottleneck_context() {
    local qps=$1
    local reasons="$2"
    local severity="${3:-medium}"
    
    # Get detailed system status
    local system_context=$(get_detailed_system_context)
    
    local bottleneck_data=$(cat << EOF
{
    "bottleneck_detected": true,
    "detection_time": "$(date -Iseconds)",
    "max_successful_qps": $LAST_SUCCESSFUL_QPS,
    "bottleneck_qps": $qps,
    "bottleneck_reasons": "$reasons",
    "severity": "$severity",
    "benchmark_mode": "$BENCHMARK_MODE",
    "rpc_mode": "$RPC_MODE",
    "consecutive_detections": $BOTTLENECK_COUNT,
    "system_context": $system_context,
    "analysis_window": {
        "start_time": "$(date -d "-${BOTTLENECK_ANALYSIS_WINDOW} seconds" -Iseconds)",
        "end_time": "$(date -Iseconds)",
        "window_seconds": $BOTTLENECK_ANALYSIS_WINDOW
    }
}
EOF
)
    
    echo "$bottleneck_data" > "$QPS_STATUS_FILE"
    echo "📊 Detailed bottleneck information saved to: $QPS_STATUS_FILE"
    
    # Also save to dedicated bottleneck analysis file
    local bottleneck_analysis_file="${LOGS_DIR}/bottleneck_analysis_${SESSION_TIMESTAMP}.json"
    echo "$bottleneck_data" > "$bottleneck_analysis_file"
    echo "🔍 Bottleneck analysis file: $(basename "$bottleneck_analysis_file")"
}

# Get detailed system context
get_detailed_system_context() {
    # Reuse bottleneck detection data acquisition mechanism
    local latest_data=$(get_latest_monitoring_data)
    
    # Extract fields from monitoring data
    local cpu_usage=$(echo "$latest_data" | jq -r '.cpu_usage // 0' 2>/dev/null || echo "0")
    local mem_usage=$(echo "$latest_data" | jq -r '.memory_usage // 0' 2>/dev/null || echo "0")
    local disk_util=$(echo "$latest_data" | jq -r '.disk_util // 0' 2>/dev/null || echo "0")
    local disk_latency=$(echo "$latest_data" | jq -r '.disk_latency // 0' 2>/dev/null || echo "0")
    local network_util=$(echo "$latest_data" | jq -r '.network_util // 0' 2>/dev/null || echo "0")
    
    # Get system static information
    local cpu_count=$(nproc 2>/dev/null || echo "1")
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | xargs 2>/dev/null || echo "0.0 0.0 0.0")
    local mem_total=$(free -g 2>/dev/null | awk '/^Mem:/ {print $2}' || echo "0")
    local mem_available=$(free -g 2>/dev/null | awk '/^Mem:/ {print $7}' || echo "0")
    
    # Get disk-standardized IOPS and Throughput (consistent with bottleneck detection)
    local disk_adjusted_iops=0
    local disk_adjusted_throughput=0
    if [[ -n "$LEDGER_DEVICE" ]]; then
        disk_adjusted_iops=$(_mqe_provider_field "$latest_data" disk_iops_provider_adjusted "data_${LEDGER_DEVICE}")
        disk_adjusted_throughput=$(_mqe_provider_field "$latest_data" disk_throughput_provider_adjusted "data_${LEDGER_DEVICE}")
    fi
    
    # Build JSON
    local context=$(cat << EOF
{
    "cpu_info": {
        "usage": $cpu_usage,
        "load_avg": "$load_avg",
        "core_count": $cpu_count
    },
    "memory_info": {
        "usage_percent": $mem_usage,
        "available_gb": $mem_available,
        "total_gb": $mem_total
    },
    "disk_info": {
        "disk_util": $disk_util,
        "disk_latency": $disk_latency,
        "iops": $disk_adjusted_iops,
        "throughput_mibs": $disk_adjusted_throughput
    },
    "network_info": {
        "utilization": $network_util,
        "connections": 0
    }
}
EOF
)
    
    echo "$context"
}

# Generate bottleneck recommendations
generate_bottleneck_recommendations() {
    local severity="$1"
    local reasons="$2"
    
    local recommendations='[]'
    
    # Generate recommendations based on bottleneck reasons
    if echo "$reasons" | grep -q "CPU"; then
        recommendations=$(echo "$recommendations" | jq '. + ["Consider upgrading to EC2 instance type with higher CPU performance", "Optimize application CPU usage efficiency", "Check for CPU-intensive processes"]')
    fi
    
    if echo "$reasons" | grep -qi "Memory"; then
        recommendations=$(echo "$recommendations" | jq '. + ["Consider upgrading to EC2 instance type with more memory", "Optimize memory usage patterns", "Check for memory leaks"]')
    fi
    
    if echo "$reasons" | grep -q "Disk"; then
        recommendations=$(echo "$recommendations" | jq '. + ["Consider upgrading Disk volume type to io2", "Increase Disk IOPS configuration", "Optimize I/O access patterns"]')
    fi
    
    if echo "$reasons" | grep -qi "Network"; then
        recommendations=$(echo "$recommendations" | jq '. + ["Consider upgrading to EC2 instance with higher network performance", "Optimize network I/O patterns", "Check network configuration"]')
    fi
    
    # Add recommendations based on severity
    if [[ "$severity" == "high" ]]; then
        recommendations=$(echo "$recommendations" | jq '. + ["Stop testing immediately to avoid system instability", "Perform detailed performance analysis", "Consider system architecture optimization"]')
    fi
    
    echo "$recommendations"
}

# Execute single QPS level test
execute_single_qps_test() {
    local qps=$1
    local duration=$2
    local targets_file=$3
    
    # Update QPS status file with current QPS value (required for all modes)
    if [[ -f "$TMP_DIR/qps_test_status" ]]; then
        echo "running qps:$qps" > "$TMP_DIR/qps_test_status.tmp"
        mv "$TMP_DIR/qps_test_status.tmp" "$TMP_DIR/qps_test_status"
    fi
    
    echo "🚀 Executing QPS test: ${qps} QPS, duration ${duration} seconds"
    
    # Build vegeta command
    local vegeta_cmd="vegeta attack -format=json -targets=$targets_file -rate=$qps -duration=${duration}s"
    local result_file="${VEGETA_RESULTS_DIR}/vegeta_${qps}qps_${SESSION_TIMESTAMP}.json"
    
    # Execute vegeta test
    echo "📊 Executing command: $vegeta_cmd"
    
    # First save attack output to temporary file
    local attack_output="${TMP_DIR}/vegeta_attack_${qps}qps_${SESSION_TIMESTAMP}.bin"
    if $vegeta_cmd > "$attack_output" 2>/dev/null; then
        # Generate JSON report (maintain existing functionality)
        vegeta report -type=json < "$attack_output" > "$result_file" 2>/dev/null
        
        # Generate TXT report for analyzer use
        local txt_report_file="${REPORTS_DIR}/vegeta_${qps}qps_${SESSION_TIMESTAMP}.txt"
        vegeta report -type=text < "$attack_output" > "$txt_report_file" 2>/dev/null
        
        # Clean up temporary file
        rm -f "$attack_output"
        
        echo "✅ QPS test completed, results saved to: $(basename "$result_file")"
        echo "📄 Text report generated: $(basename "$txt_report_file")"
        
        # Parse test results
        local total_requests=$(jq -r '.requests' "$result_file" 2>/dev/null || echo "1")
        local success_requests=$(jq -r '.status_codes."200" // 0' "$result_file" 2>/dev/null || echo "0")
        local success_rate=$(awk "BEGIN {printf \"%.0f\", $success_requests * 100 / $total_requests}" 2>/dev/null || echo "0")
        local avg_latency=$(jq -r '.latencies.mean' "$result_file" 2>/dev/null || echo "0")
        
        # Convert latency unit (nanoseconds to milliseconds)
        local avg_latency_ms=$(awk "BEGIN {printf \"%.2f\", $avg_latency / 1000000}" 2>/dev/null || echo "0")
        
        echo "📈 Test results: Success rate ${success_rate}%, Average latency ${avg_latency_ms}ms"
        
        # Check if test was successful
        local success_rate_num=$(awk "BEGIN {printf \"%.0f\", $success_rate * 100}" 2>/dev/null || echo "0")
        local avg_latency_num=$(awk "BEGIN {printf \"%.2f\", $avg_latency_ms}" 2>/dev/null || echo "0")
        
        if (( $(awk "BEGIN {print ($success_rate_num >= $SUCCESS_RATE_THRESHOLD) ? 1 : 0}") )) && \
           (( $(awk "BEGIN {print ($avg_latency_num <= $MAX_LATENCY_THRESHOLD) ? 1 : 0}") )); then
            LAST_SUCCESSFUL_QPS=$qps
            return 0
        else
            echo "⚠️ Test quality below threshold: Success rate ${success_rate}% (required>${SUCCESS_RATE_THRESHOLD}%), Latency ${avg_latency_ms}ms (required<${MAX_LATENCY_THRESHOLD}ms)"
            return 1
        fi
    else
        echo "❌ QPS test execution failed"
        return 1
    fi
}

# Execute QPS test main logic
execute_qps_test() {
    echo "🚀 Starting QPS test execution..."
    
    local test_start_time=${SESSION_TIMESTAMP}
    
    # Select target file
    local targets_file
    if [[ "$RPC_MODE" == "mixed" ]]; then
        targets_file="$MIXED_METHOD_TARGETS_FILE"
    else
        targets_file="$SINGLE_METHOD_TARGETS_FILE"
    fi
    
    echo "🎯 Using target file: $(basename "$targets_file")"
    echo "📊 Target count: $(wc -l < "$targets_file")"
    
    # Initialize test status
    BOTTLENECK_DETECTED=false
    BOTTLENECK_COUNT=0
    LAST_SUCCESSFUL_QPS=0
    
    # Initialize bottleneck detector for intensive mode
    if [[ "$BENCHMARK_MODE" == "intensive" && "$INTENSIVE_AUTO_STOP" == "true" ]]; then
        echo "🔍 Initializing bottleneck detector (intensive test mode)..."
        if [[ -f "${QPS_SCRIPT_DIR}/../monitoring/bottleneck_detector.sh" ]]; then
            "${QPS_SCRIPT_DIR}/../monitoring/bottleneck_detector.sh" init
            if [[ $? -eq 0 ]]; then
                echo "✅ Bottleneck detector initialized successfully"
            else
                echo "⚠️  Bottleneck detector initialization failed, but testing will continue"
            fi
        else
            echo "⚠️  Bottleneck detector script does not exist, skipping initialization"
        fi
        echo ""
    fi
    
    # QPS test loop
    local current_qps=$INITIAL_QPS
    local test_count=0
    
    while [[ $current_qps -le $MAX_QPS ]]; do
        test_count=$((test_count + 1))
        echo ""
        echo "📋 Test round $test_count: QPS = $current_qps"
        
        # Warmup phase
        if [[ $QPS_WARMUP_DURATION -gt 0 ]]; then
            echo "🔥 Warmup phase: ${QPS_WARMUP_DURATION} seconds"
            sleep $QPS_WARMUP_DURATION
        fi
        
        # Execute single QPS level test
        if execute_single_qps_test "$current_qps" "$DURATION" "$targets_file"; then
            echo "✅ QPS $current_qps benchmark test successful"
        else
            echo "❌ QPS $current_qps benchmark test failed"
            
            # Stop if not intensive benchmark mode when test fails
            if [[ "$BENCHMARK_MODE" != "intensive" ]]; then
                echo "🛑 Test failed in non-intensive benchmark mode, stopping test"
                break
            fi
        fi
        
        # Check bottleneck in intensive benchmark mode
        if [[ "$BENCHMARK_MODE" == "intensive" && "$INTENSIVE_AUTO_STOP" == "true" ]]; then
            if ! check_bottleneck_during_test "$current_qps"; then
                echo "🚨 Performance bottleneck detected, stopping benchmark test"
                echo "🏆 Maximum successful QPS: $LAST_SUCCESSFUL_QPS"
                break
            fi
        fi
        
        # Cooldown time
        if [[ $QPS_COOLDOWN -gt 0 ]]; then
            echo "❄️ Cooldown time: ${QPS_COOLDOWN} seconds"
            sleep $QPS_COOLDOWN
        fi
        
        # Increase QPS
        current_qps=$((current_qps + STEP_QPS))
    done
    
    echo ""
    echo "🎉 QPS test completed"
    echo "📊 Test rounds: $test_count"
    echo "🏆 Maximum successful QPS: $LAST_SUCCESSFUL_QPS"
    
    if [[ "$BOTTLENECK_DETECTED" == "true" ]]; then
        echo "🚨 Performance bottleneck detected, detailed information saved"
    else
        # Write status file on normal completion
        cat > "$QPS_STATUS_FILE" << EOF
{
    "status": "completed",
    "max_successful_qps": $LAST_SUCCESSFUL_QPS,
    "bottleneck_detected": false,
    "bottleneck_qps": 0,
    "completion_time": "$(date -Iseconds)",
    "test_duration": $DURATION,
    "benchmark_mode": "$BENCHMARK_MODE",
    "rpc_mode": "$RPC_MODE"
}
EOF
        echo "📊 QPS status saved to: $QPS_STATUS_FILE"
    fi
    
    return 0
}

# Main function
main() {

    # Parse arguments
    parse_arguments "$@"
    
    # Display configuration
    show_benchmark_config
    
    # Pre-check
    if ! pre_check; then
        echo "❌ Pre-check failed"
        return 1
    fi
    
    # Execute QPS test
    if execute_qps_test; then
        echo "🎉 QPS test execution successful"
        return 0
    else
        echo "❌ QPS test execution failed"
        return 1
    fi
}

# Execute main function
main "$@"
