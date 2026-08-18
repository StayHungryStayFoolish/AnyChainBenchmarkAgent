#!/bin/bash
# =====================================================================
# Unified Monitor - Unified Time Management (Unified Logger Version)
# =====================================================================
# Single monitoring entry point, avoid multiple scripts calling iostat/mpstat repeatedly
# Unified time format, support complete performance metrics monitoring
# Use unified logger
# =====================================================================

# Strict error handling - but allow safe use in interactive environments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Use strict mode when script is executed directly
    set -euo pipefail
else
    # Use relaxed mode when sourced to avoid exiting shell
    set -uo pipefail
fi

source "$(dirname "${BASH_SOURCE[0]}")/../config/config_loader.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../utils/unified_logger.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../utils/disk_converter.sh"

# Initialize unified logger
init_logger "unified_monitor" $LOG_LEVEL "${LOGS_DIR}/unified_monitor.log"

source "$(dirname "${BASH_SOURCE[0]}")/lib/monitor_utils.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/monitor_performance_advisor.sh"

# Error handling function
handle_monitor_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Monitor error occurred at line $line_number , exit code: $exit_code"
    log_warn "Stopping monitoring processes..."
    cleanup_monitor_processes
    exit $exit_code
}

# Set error trap - only enable when script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    trap 'handle_monitor_error $LINENO' ERR
fi

source "$(dirname "${BASH_SOURCE[0]}")/../core/common_functions.sh"
source "$(dirname "${BASH_SOURCE[0]}")/iostat_collector.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/metrics_json_writer.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/block_height_csv_reader.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/qps_runtime_reader.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/performance_data_line_builder.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/cloud_provider_resolver.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/ena_data_normalizer.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/sample_count_tracker.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/cgroup_collector_wrapper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/node_execution_collector_wrapper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/system_collectors.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/process_collectors.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/monitoring_overhead.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/monitor_error_recovery.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/monitoring_overhead_csv.sh"

# Avoid redefining readonly variables - use definitions from config_loader.sh
if [[ -z "${UNIFIED_LOG:-}" ]]; then
    UNIFIED_LOG="${LOGS_DIR}/performance_${SESSION_TIMESTAMP}.csv"
fi

# MONITORING_OVERHEAD_LOG is set in detect_deployment_paths() function in config_loader.sh

# Monitoring overhead CSV header definition - loaded from config_loader.sh
# OVERHEAD_CSV_HEADER is defined in config_loader.sh

MONITOR_PIDS=()
START_TIME=""
END_TIME=""

# Initialize monitoring environment
init_monitoring() {
    echo "🔧 Initializing unified monitoring environment..."

    # Basic configuration validation
    if ! basic_config_check; then
        echo "❌ Monitoring system startup failed: configuration validation failed" >&2
        return 1
    fi

    # Validate devices (degraded mode allowed unless STRICT_DEVICE_VALIDATION=true)
    if ! validate_devices; then
        return 1
    fi

    # Surface degraded-mode notice if validate_devices set the flag
    if [[ "${DEVICE_VALIDATION_DEGRADED:-0}" == "1" ]]; then
        echo "⚠️  Running in DEGRADED MODE: disk I/O monitoring disabled (devices unavailable); CPU/mem/net still active" >&2
        log_warn "DEVICE_VALIDATION_DEGRADED=1 — iostat columns will be NaN placeholders"
    fi

    # Check required commands - gracefully handle missing commands
    local missing_commands=()
    local critical_missing=()

    # Check availability of each command
    for cmd in mpstat iostat sar free; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_commands+=("$cmd")
            # All monitoring commands are critical, missing any will affect functionality
            critical_missing+=("$cmd")
        fi
    done

    if [[ ${#missing_commands[@]} -gt 0 ]]; then
        log_warn "Missing some monitoring commands: ${missing_commands[*]}"
        echo "⚠️  Missing monitoring commands: ${missing_commands[*]}"
        echo "💡 Recommended installation: sudo apt-get install sysstat procps"

        # Fail if critical commands are missing
        if [[ ${#critical_missing[@]} -gt 0 ]]; then
            log_error "Missing critical commands: ${critical_missing[*]}, cannot continue"
            echo "❌ Missing critical commands: ${critical_missing[*]}, monitoring functionality cannot start"
            return 1
        fi
    fi

    log_info "Unified monitoring environment initialization completed"
    return 0
}

# Generate complete CSV header - support conditional ENA fields + cgroup fields
generate_csv_header() {
    # The basic header is generated through csv_schema_registry when available.
    # Fallback literal is byte-identical to registry _BASIC_FIELDS when registry is unavailable.
    local basic_header
    if declare -F csv_registry_basic_header >/dev/null 2>&1; then
        basic_header="$(csv_registry_basic_header)"
    else
        log_warn "csv_registry_basic_header unavailable; using fallback basic header literal"
        basic_header="timestamp,cpu_usage,cpu_usr,cpu_sys,cpu_iowait,cpu_soft,cpu_idle,mem_used,mem_total,mem_usage"
    fi
    local device_header=$(generate_all_devices_header)
    local network_header="net_interface,net_rx_mbps,net_tx_mbps,net_total_mbps,net_rx_gbps,net_tx_gbps,net_total_gbps,net_rx_pps,net_tx_pps,net_total_pps"
    local overhead_header="monitoring_iops_per_sec,monitoring_throughput_mibs_per_sec"
    local block_height_header
    if declare -F csv_registry_block_header >/dev/null 2>&1; then
        block_height_header="$(csv_registry_block_header)"
    else
        block_height_header="local_block_height,mainnet_block_height,block_height_diff,local_health,mainnet_health,data_loss,sync_mode,sync_status,lag_value,lag_unit,freshness_gap_seconds,probe_error"
    fi
    local qps_header="current_qps,rpc_latency_ms,qps_data_available"
    local execution_header=$(get_execution_header)
    local node_cpu_header=$(get_node_cpu_header)
    local cgroup_header=$(get_cgroup_header)

    # Configuration-driven ENA header generation
    # cloud_provider is appended as the final column, preserving existing column order;
    # readers access columns by name, so appending this column is safe.
    if [[ "$ENA_MONITOR_ENABLED" == "true" ]]; then
        local ena_header=$(build_ena_header)
        echo "$basic_header,$device_header,$network_header,$ena_header,$overhead_header,$block_height_header,$qps_header,$execution_header,$node_cpu_header,$cgroup_header,cloud_provider"
    else
        echo "$basic_header,$device_header,$network_header,$overhead_header,$block_height_header,$qps_header,$execution_header,$node_cpu_header,$cgroup_header,cloud_provider"
    fi
}

# Log performance data - support conditional ENA data and JSON generation
log_performance_data() {
    local timestamp=$(get_unified_timestamp)
    local cpu_data=$(get_cpu_data)
    local memory_data=$(get_memory_data)
    local device_data=$(get_all_devices_data)
    local network_data=$(get_network_data)
    local overhead_data=$(get_monitoring_overhead)
    # cgroup_collector integration (fail-soft 19 fields)
    local cgroup_data=$(get_cgroup_data)
    # Mark which provider produced this row (aws|gcp|other).
    local cloud_provider_val
    cloud_provider_val=$(resolve_cloud_provider_value)

    # Collect current QPS test data
    local current_qps rpc_latency_ms qps_data_available
    IFS=',' read -r current_qps rpc_latency_ms qps_data_available < <(
        get_qps_runtime_fields "${TMP_DIR}/qps_test_status" "${VEGETA_RESULTS_DIR}"
    )

    local execution_data
    execution_data=$(get_execution_data)

    local node_cpu_data
    node_cpu_data=$(get_node_cpu_data)

    # Get block height / sync-health data (if block_height monitoring is enabled)
    local block_height_data
    block_height_data=$(get_block_height_csv_fields "${BLOCK_HEIGHT_DATA_FILE:-}")

    # Conditionally add ENA data
    local ena_data=""
    if [[ "$ENA_MONITOR_ENABLED" == "true" ]]; then
        ena_data=$(get_ena_allowance_data)
        
        # Add ENA data validation and debugging
        log_debug "ENA data debug: '$ena_data'"
        log_debug "ENA data length: ${#ena_data}"
        
        # Validate ENA data format (only contains numbers and commas)
        if [[ ! "$ena_data" =~ ^[0-9,]+$ ]]; then
            log_error "ENA data format anomaly: '$ena_data'"
            log_error "First 100 chars: '$(echo "$ena_data" | cut -c1-100)'"
            
            # Replace abnormal data with default values
            ena_data=$(normalize_ena_data "$ena_data" "$ENA_ALLOWANCE_FIELDS_STR")
            log_error "Using default ENA data: '$ena_data'"
        fi
    fi

    local data_line
    data_line=$(build_performance_data_line \
        "$ENA_MONITOR_ENABLED" \
        "$timestamp" \
        "$cpu_data" \
        "$memory_data" \
        "$device_data" \
        "$network_data" \
        "$ena_data" \
        "$overhead_data" \
        "$block_height_data" \
        "$current_qps" \
        "$rpc_latency_ms" \
        "$qps_data_available" \
        "$execution_data" \
        "$node_cpu_data" \
        "$cgroup_data" \
        "$cloud_provider_val")
    
    # Final data line validation
    log_debug "Final data line length: ${#data_line}"
    if [[ ${#data_line} -gt 10000 ]]; then
        log_error "Data line abnormally long: ${#data_line} characters"
        log_error "First 200 chars of data line: '$(echo "$data_line" | cut -c1-200)'"
    fi

    # If CSV file doesn't exist or is empty, write header first
    if [[ ! -f "$UNIFIED_LOG" ]] || [[ ! -s "$UNIFIED_LOG" ]]; then
        local csv_header=$(generate_csv_header)
        echo "$csv_header" > "$UNIFIED_LOG"
    fi

    # Use concurrent-safe CSV writing
    if safe_write_csv "$UNIFIED_LOG" "$data_line"; then
        log_debug "CSV data safely written: $UNIFIED_LOG"
    else
        echo "ERROR: CSV data write failed: $UNIFIED_LOG" >&2
        return 1
    fi

    # Write separate monitoring overhead log
    write_monitoring_overhead_log

    # Periodic performance analysis (analyze once every 100 records)
    local sample_count_file="${MEMORY_SHARE_DIR}/sample_count"
    local current_count
    current_count=$(next_sample_count "$sample_count_file")

    # Perform performance analysis every 100 samples
    if (( current_count % 100 == 0 )); then
        log_info "🔍 Executing periodic performance analysis - sample $current_count"
        auto_performance_optimization_advisor
    fi

    # Generate complete report every 1000 samples
    if (( current_count % 1000 == 0 )); then
        log_info "📊 Generating performance impact report - sample $current_count"
        generate_performance_impact_report
    fi

    # Generate JSON file
    generate_json_metrics "$timestamp" "$cpu_data" "$memory_data" "$device_data" "$network_data" "$ena_data" "$overhead_data"
}

# Start unified monitoring - support follow QPS test mode
start_unified_monitoring() {
    local duration="$1"
    local interval=${2:-$MONITOR_INTERVAL}

    # =====================================================================
    # Monitoring system initialization phase
    # =====================================================================
    
    log_info "🚀 Starting unified performance monitoring system..."
    
    # Step 1: Initialize command cache - key performance optimization step
    log_info "📋 Step 1: Initialize system command cache"
    init_command_cache

    # Step 2: Initialize error handling system
    log_info "🛡️ Step 2: Initialize error handling system"
    initialize_error_handling_system

    START_TIME=$(get_unified_timestamp)

    # =====================================================================
    # Monitoring configuration information display
    # =====================================================================
    
    echo ""
    echo "🎯 ===== Unified Performance Monitoring System ====="
    echo "📅 Start time: $START_TIME"
    echo "⏱️  Monitoring interval: ${interval} seconds"

    if [[ "$duration" -eq 0 ]]; then
        echo "🔄 Run mode: Follow framework lifecycle (no time limit)"
        echo "🎛️  Control file: $TMP_DIR/qps_test_status"
    else
        echo "⏰ Run mode: Timed monitoring (${duration} seconds)"
    fi

    echo "📊 Data file: $UNIFIED_LOG"
    
    # Display system capability detection results
    echo ""
    echo "🔧 ===== System Capability Detection ====="

    # Display configuration status
    log_info "DATA device: $LEDGER_DEVICE"

    if is_accounts_configured; then
        log_info "ACCOUNTS device: $ACCOUNTS_DEVICE Volume type: $ACCOUNTS_VOL_TYPE"
    else
        echo "ℹ️  ACCOUNTS device not configured"
    fi

    if [[ -n "$NETWORK_INTERFACE" ]]; then
        log_info "Network interface: $NETWORK_INTERFACE"
    fi

    # Display provider-specific NIC monitoring status
    if [[ "$ENA_MONITOR_ENABLED" == "true" ]]; then
        log_info "Provider NIC monitoring: AWS ENA enabled"
    else
        echo "ℹ️  Provider NIC monitoring: GCP/generic collector path"
    fi

    # Create CSV header
    local csv_header=$(generate_csv_header)
    echo "$csv_header" > "$UNIFIED_LOG"

    # Create latest file symlink for bottleneck detection use
    local latest_csv="${PERFORMANCE_LATEST_CSV:-${LOGS_DIR}/performance_latest.csv}"
    ln -sf "$(basename "$UNIFIED_LOG")" "$latest_csv"

    log_info "CSV header created - $(echo "$csv_header" | tr ',' '\n' | wc -l) fields"
    log_info "Latest file link created: $latest_csv"
    echo ""

    # Record monitoring process PID
    MONITOR_PIDS+=($BASHPID)

    # =====================================================================
    # Main monitoring loop
    # =====================================================================
    
    echo ""
    echo "🔄 ===== Starting Monitoring Loop ====="
    
    local start_time=$(date +%s)
    local sample_count=0
    local last_status_time=0
    local status_interval=30  # Display status every 30 seconds

    echo "⏰ Starting data collection..."

    # Unified monitoring loop logic - choose control method based on duration parameter
    if [[ "$duration" -eq 0 ]]; then
        # duration=0 means follow framework lifecycle - check status file
        while [[ -f "$TMP_DIR/qps_test_status" ]]; do
            # Collect unified monitoring data
            log_debug "📊 Data collection #${sample_count} starting..."
            local current_system_load=$(assess_system_load)

            log_performance_data
            sample_count=$((sample_count + 1))
            
            # Periodically display monitoring status
            local current_time=$(date +%s)
            if [[ $((current_time - last_status_time)) -ge $status_interval ]]; then
                local elapsed=$((current_time - start_time))
                echo "📈 Monitoring status: Collected $sample_count data points, runtime ${elapsed}s (following framework lifecycle)"
                last_status_time=$current_time
            fi

            # Progress report
            if (( sample_count % 12 == 0 )); then
                local current_time=$(date +%s)
                local elapsed=$((current_time - start_time))
                local avg_interval=$(awk -v e="$elapsed" -v s="$sample_count" 'BEGIN {printf "%.2f", (s > 0) ? e / s : 0}' 2>/dev/null || echo "N/A")
                echo "📈 Monitoring status: Collected $sample_count samples, runtime ${elapsed}s, average interval ${avg_interval}s (following framework lifecycle)"
            fi

            # Wait until next scheduled time
            local now=$(date +%s)
            local next_run=$((start_time + sample_count * CURRENT_MONITOR_INTERVAL))
            if (( now < next_run )); then
                sleep $((next_run - now))
            fi
        done
    else
            # Fixed duration logic
            local end_time=$((start_time + duration))

            while [[ $(date +%s) -lt $end_time ]]; do
            # Collect unified monitoring data
            log_debug "📊 Data collection #${sample_count} starting..."
            local current_system_load=$(assess_system_load)

            log_performance_data
            sample_count=$((sample_count + 1))
            
            # Periodically display monitoring status
            local current_time=$(date +%s)
            if [[ $((current_time - last_status_time)) -ge $status_interval ]]; then
                local elapsed=$((current_time - start_time))
                local remaining=$((end_time - current_time))
                local progress_percent=$(awk "BEGIN {printf "%.1f", $elapsed * 100 / $duration}" 2>/dev/null || echo "N/A")
                echo "📈 Monitoring status: Collected $sample_count data points, progress ${progress_percent}%, runtime ${elapsed}s, remaining ${remaining}s"
                last_status_time=$current_time
            fi

            # Progress report
            if (( sample_count % 12 == 0 )); then
                local current_time=$(date +%s)
                local elapsed=$((current_time - start_time))
                local remaining=$((end_time - current_time))
                local avg_interval=$(awk -v e="$elapsed" -v s="$sample_count" 'BEGIN {printf "%.2f", (s > 0) ? e / s : 0}' 2>/dev/null || echo "N/A")
                local progress_percent=$(awk -v e="$elapsed" -v d="$duration" 'BEGIN {printf "%.1f", (d > 0) ? e * 100 / d : 0}' 2>/dev/null || echo "N/A")
                echo "📈 Monitoring status: Collected $sample_count samples, progress ${progress_percent}%, runtime ${elapsed}s, remaining ${remaining}s, average interval ${avg_interval}s"
            fi

            # Wait until next scheduled time
            local now=$(date +%s)
            local next_run=$((start_time + sample_count * CURRENT_MONITOR_INTERVAL))
            if (( now < next_run )); then
                sleep $((next_run - now))
            fi
        done
    fi

    END_TIME=$(get_unified_timestamp)

    # =====================================================================
    # Monitoring completion statistics report
    # =====================================================================
    
    local final_time=$(date +%s)
    local total_elapsed=$((final_time - start_time))
    local avg_sample_interval=$(awk -v t="$total_elapsed" -v s="$sample_count" 'BEGIN {printf "%.2f", (s > 0) ? t / s : 0}' 2>/dev/null || echo "N/A")
    local file_size=$(du -h "$UNIFIED_LOG" 2>/dev/null | cut -f1 || echo "unknown")
    local line_count=$(wc -l < "$UNIFIED_LOG" 2>/dev/null || echo "unknown")
    
    echo ""
    echo "✅ ===== Unified Performance Monitoring Completed ====="
    echo "📅 Start time: $START_TIME"
    echo "📅 End time: $END_TIME"
    echo "⏱️  Total runtime: ${total_elapsed} seconds"
    echo "📊 Total samples: $sample_count times"
    echo "📈 Average sampling interval: ${avg_sample_interval} seconds"
    echo "📄 Data file: $UNIFIED_LOG"
    echo "📋 Data statistics: $line_count lines, file size $file_size"
    
    # Performance efficiency assessment
    if [[ "$sample_count" -gt 0 ]] && [[ "$total_elapsed" -gt 0 ]]; then
        local efficiency=$(awk -v s="$sample_count" -v t="$total_elapsed" 'BEGIN {printf "%.1f", (t > 0) ? s * 100 / t : 0}' 2>/dev/null || echo "N/A")
        echo "⚡ Monitoring efficiency: ${efficiency} samples/second"
    fi
    
    # Data quality assessment
    if [[ "$line_count" != "unknown" ]] && [[ "$sample_count" -gt 0 ]]; then
        local data_integrity=$(awk -v l="$line_count" -v s="$sample_count" 'BEGIN {printf "%.1f", (s > 0) ? (l - 1) * 100 / s : 0}' 2>/dev/null || echo "N/A")
        echo "📊 Data integrity: ${data_integrity}% (${line_count} data lines/${sample_count} samples)"
    fi
    
    echo ""
    echo "🧹 ===== Cleaning System Resources ====="
}

# Stop monitoring - prevent duplicate calls
STOP_MONITORING_CALLED=false
stop_unified_monitoring() {
    # Prevent duplicate calls
    if [[ "$STOP_MONITORING_CALLED" == "true" ]]; then
        return 0
    fi
    STOP_MONITORING_CALLED=true

    echo "🛑 Stopping unified monitoring..."
    
    local cleanup_count=0
    local cleanup_errors=0

    # Terminate all related processes
    echo "🔄 Cleaning up monitoring processes..."
    for pid in "${MONITOR_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            if kill "$pid" 2>/dev/null; then
                cleanup_count=$((cleanup_count + 1))
                log_debug "✅ Terminated process PID: $pid"
            else
                cleanup_errors=$((cleanup_errors + 1))
                log_debug "❌ Unable to terminate process PID: $pid"
            fi
        fi
    done

    # iostat collectors are owned by iostat_collector.sh and tracked by its
    # PID files rather than by the wrapper process array above.
    if declare -F stop_iostat_collectors >/dev/null 2>&1; then
        stop_iostat_collectors
    fi

    # Generate error recovery report
    if [[ "$ERROR_RECOVERY_ENABLED" == "true" ]]; then
        echo "📋 Generating error recovery report..."
        generate_error_recovery_report
    fi

    # Cleanup completion summary
    echo "✅ Resource cleanup completed: Terminated $cleanup_count processes"
    if [[ "$cleanup_errors" -gt 0 ]]; then
        echo "⚠️  Cleanup warning: $cleanup_errors processes could not be terminated normally"
    fi
    
    log_info "Unified monitoring stopped"
}

# Get monitoring time range (for use by other scripts)
get_monitoring_time_range() {
    echo "start_time=$START_TIME"
    echo "end_time=$END_TIME"
}

# Main function
main() {
    echo "🔧 Unified Performance Monitor"
    echo "=================="
    echo ""

    # Initialize
    if ! init_monitoring; then
        exit 1
    fi

    # Parse parameters - add follow QPS test mode
    local duration=0  # 0 means run indefinitely, stopped by external control
    local interval=$MONITOR_INTERVAL
    local background=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--duration)
                duration="$2"
                shift 2
                ;;
            -i|--interval)
                interval="$2"
                shift 2
                ;;
            -b|--background)
                background=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  -d, --duration SECONDS    Monitor duration, 0=follow framework lifecycle, default: 0"
                echo "  -i, --interval SECONDS    Monitor interval, default: $MONITOR_INTERVAL"
                echo "  -b, --background          Run in background"
                echo "  -h, --help               Show help"
                echo ""
                echo "Features:"
                echo "  ✅ Unified monitoring entry, eliminate duplicate monitoring"
                echo "  ✅ Standard time format: $TIMESTAMP_FORMAT"
                echo "  ✅ Complete metric coverage: CPU, Memory, Disk, Network"
                echo "  ✅ Real monitoring overhead statistics"
                echo "  ✅ Unified field naming convention"
                echo "  ✅ Follow QPS test lifecycle"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    if [[ "$background" == "true" ]]; then
        echo "🚀 Starting in background mode..."
        # Background call logic, uniformly use duration=0 follow framework lifecycle mode
        nohup "$0" -i "$interval" > "${LOGS_DIR}/unified_monitor.log" 2>&1 &
        echo "Background process PID: $!"
        echo "Log file: ${LOGS_DIR}/unified_monitor.log"
        echo "Data file: $UNIFIED_LOG"
    else
        # Set signal handling
        trap stop_unified_monitoring EXIT INT TERM

        start_unified_monitoring "$duration" "$interval"
    fi
}

# Monitoring system integrity check
monitoring_system_integrity_check() {
    log_info "🔍 Executing monitoring system integrity check..."

    local integrity_issues=()

    # Check critical files
    local critical_files=("$UNIFIED_LOG" "$MONITORING_OVERHEAD_LOG")
    for file in "${critical_files[@]}"; do
        if [[ -n "$file" ]] && [[ -f "$file" ]]; then
            if [[ ! -r "$file" ]]; then
                integrity_issues+=("File not readable: $file")
            fi
            if [[ ! -w "$file" ]]; then
                integrity_issues+=("File not writable: $file")
            fi
        fi
    done

    # Check configuration integrity
    local required_vars=("LOGS_DIR" "MONITOR_INTERVAL" "LEDGER_DEVICE")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            integrity_issues+=("Required configuration variable not set: $var")
        fi
    done

    # Check process configuration - use standardized array access
    if [[ -z "$MONITORING_PROCESS_NAMES_STR" ]]; then
        integrity_issues+=("Monitoring process name configuration is empty")
    fi

    # Check permissions
    if [[ ! -w "$LOGS_DIR" ]]; then
        integrity_issues+=("Insufficient log directory permissions: $LOGS_DIR")
    fi

    # Report integrity status
    if [[ ${#integrity_issues[@]} -eq 0 ]]; then
        log_info "✅ Monitoring system integrity check passed"
        return 0
    else
        log_warn "⚠️  Found ${#integrity_issues[@]} integrity issues:"
        for issue in "${integrity_issues[@]}"; do
            log_warn "  - $issue"
        done
        return 1
    fi
}

# Auto-fix functionality
auto_fix_common_issues() {
    log_info "🔧 Attempting to auto-fix common issues..."

    local fixes_applied=0

    # Fix log directory permissions
    if [[ ! -w "$LOGS_DIR" ]]; then
        log_info "Fixing log directory permissions..."
        if mkdir -p "$LOGS_DIR" 2>/dev/null && chmod 755 "$LOGS_DIR" 2>/dev/null; then
            log_info "✅ Log directory permissions fixed"
            fixes_applied=$((fixes_applied + 1))
        else
            log_warn "❌ Unable to fix log directory permissions"
        fi
    fi

    # Fix log file permissions
    for log_file in "$UNIFIED_LOG" "$MONITORING_OVERHEAD_LOG" "$PERFORMANCE_LOG" "$ERROR_LOG"; do
        if [[ -n "$log_file" ]] && [[ -f "$log_file" ]] && [[ ! -w "$log_file" ]]; then
            log_info "Fixing log file permissions: $log_file"
            if chmod 644 "$log_file" 2>/dev/null; then
                log_info "✅ Log file permissions fixed: $log_file"
                fixes_applied=$((fixes_applied + 1))
            else
                log_warn "❌ Unable to fix log file permissions: $log_file"
            fi
        fi
    done

    # Clean disk space
    local disk_usage=$(df "$LOGS_DIR" 2>/dev/null | awk 'NR==2 {print $5}' | sed 's/%//' || echo "0")
    if [[ $disk_usage -gt 90 ]]; then
        log_info "Cleaning disk space..."
        local cleaned_files=0

        # Clean log files older than 7 days
        if find "$LOGS_DIR" -name "*.log" -mtime +7 -delete 2>/dev/null; then
            cleaned_files=$((cleaned_files + 1))
        fi

        # Clean CSV files older than 3 days
        if find "$LOGS_DIR" -name "*.csv" -mtime +3 -delete 2>/dev/null; then
            cleaned_files=$((cleaned_files + 1))
        fi

        if [[ $cleaned_files -gt 0 ]]; then
            log_info "✅ Old log files cleaned"
            fixes_applied=$((fixes_applied + 1))
        fi
    fi

    log_info "Auto-fix completed, applied $fixes_applied fixes"
    return $fixes_applied
}

# Error handling system initialization
initialize_error_handling_system() {
    if [[ "$ERROR_RECOVERY_ENABLED" != "true" ]]; then
        log_info "Error recovery system disabled"
        return 0
    fi

    log_info "🚀 Initializing error handling system..."

    # Create error log file
    if [[ ! -f "$ERROR_LOG" ]]; then
        echo "timestamp,function_name,error_code,error_message,consecutive_count" > "$ERROR_LOG"
        log_info "Error log file created: $ERROR_LOG"
    fi

    # System health check removed - conflicts with extreme testing philosophy

    # Execute integrity check
    monitoring_system_integrity_check

    # Attempt auto-fix
    auto_fix_common_issues

    log_info "✅ Error handling system initialization completed"
}

# Script entry point - only call main function when executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
