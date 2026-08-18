#!/bin/bash
# Disk Real-time Bottleneck Detection, Console Output
# High-frequency monitoring of Disk performance, real-time detection of IOPS and Throughput bottlenecks

# Import dependencies
# Safely load configuration file, avoid readonly variable conflicts
if ! source "$(dirname "${BASH_SOURCE[0]}")/../config/config_loader.sh" 2>/dev/null; then
    echo "Warning: Configuration file loading failed, using default configuration"
    LOGS_DIR=${LOGS_DIR:-"/tmp/blockchain-node-benchmark/logs"}
fi
source "$(dirname "${BASH_SOURCE[0]}")/../utils/unified_logger.sh"

# Initialize unified logger manager
init_logger "disk_bottleneck_detector" $LOG_LEVEL "${LOGS_DIR}/disk_bottleneck_detector.log"

source "$(dirname "${BASH_SOURCE[0]}")/../utils/disk_converter.sh"
# CSV Schema Registry: resolve disk physical column names through registry
source "$(dirname "${BASH_SOURCE[0]}")/../config/csv_schema_registry.sh"

# Bottleneck detection configuration
# Use unified monitoring interval, loaded from config.sh
# Threshold configuration (can be overridden by environment variables)
BOTTLENECK_DISK_IOPS_THRESHOLD=${BOTTLENECK_DISK_IOPS_THRESHOLD:-90}      # IOPS utilization threshold (%)
BOTTLENECK_DISK_THROUGHPUT_THRESHOLD=${BOTTLENECK_DISK_THROUGHPUT_THRESHOLD:-90}  # Throughput utilization threshold (%)

# Threshold configuration
readonly BOTTLENECK_IOPS_THRESHOLD=$(awk "BEGIN {printf \"%.4f\", ${BOTTLENECK_DISK_IOPS_THRESHOLD:-90} / 100}")
readonly BOTTLENECK_THROUGHPUT_THRESHOLD=$(awk "BEGIN {printf \"%.4f\", ${BOTTLENECK_DISK_THROUGHPUT_THRESHOLD:-90} / 100}")

# Global variables
declare -A DEVICE_LIMITS
declare -gA CSV_FIELD_MAP  # CSV field mapping: field name -> column index

# Initialize Disk limits configuration
init_disk_limits() {
    echo "🔧 Initializing Disk limits configuration..."
    
    # DATA volume limits (must exist)
    if [[ -n "$DATA_VOL_MAX_IOPS" ]]; then
        case "$DATA_VOL_TYPE" in
            "gp3")
                DEVICE_LIMITS["${LEDGER_DEVICE}_max_iops"]="$DATA_VOL_MAX_IOPS"
                DEVICE_LIMITS["${LEDGER_DEVICE}_max_throughput"]="${DATA_VOL_MAX_THROUGHPUT:-1000}"
                ;;
            "io2")
                DEVICE_LIMITS["${LEDGER_DEVICE}_max_iops"]="$DATA_VOL_MAX_IOPS"
                DEVICE_LIMITS["${LEDGER_DEVICE}_max_throughput"]="$DATA_VOL_MAX_THROUGHPUT"
                ;;
            "instance-store")
                DEVICE_LIMITS["${LEDGER_DEVICE}_max_iops"]="$DATA_VOL_MAX_IOPS"
                DEVICE_LIMITS["${LEDGER_DEVICE}_max_throughput"]="$DATA_VOL_MAX_THROUGHPUT"
                ;;
            *)
                DEVICE_LIMITS["${LEDGER_DEVICE}_max_iops"]="$DATA_VOL_MAX_IOPS"
                DEVICE_LIMITS["${LEDGER_DEVICE}_max_throughput"]="$DATA_VOL_MAX_THROUGHPUT"
                ;;
        esac
        echo "  DATA Volume (${LEDGER_DEVICE}): ${DEVICE_LIMITS["${LEDGER_DEVICE}_max_iops"]} IOPS, ${DEVICE_LIMITS["${LEDGER_DEVICE}_max_throughput"]} MiB/s"
    fi
    
    # ACCOUNTS volume limits
    if is_accounts_configured; then
        case "$ACCOUNTS_VOL_TYPE" in
            "gp3")
                DEVICE_LIMITS["${ACCOUNTS_DEVICE}_max_iops"]="$ACCOUNTS_VOL_MAX_IOPS"
                DEVICE_LIMITS["${ACCOUNTS_DEVICE}_max_throughput"]="${ACCOUNTS_VOL_MAX_THROUGHPUT:-1000}"
                ;;
            "io2")
                DEVICE_LIMITS["${ACCOUNTS_DEVICE}_max_iops"]="$ACCOUNTS_VOL_MAX_IOPS"
                DEVICE_LIMITS["${ACCOUNTS_DEVICE}_max_throughput"]="$ACCOUNTS_VOL_MAX_THROUGHPUT"
                ;;
            "instance-store")
                DEVICE_LIMITS["${ACCOUNTS_DEVICE}_max_iops"]="$ACCOUNTS_VOL_MAX_IOPS"
                DEVICE_LIMITS["${ACCOUNTS_DEVICE}_max_throughput"]="$ACCOUNTS_VOL_MAX_THROUGHPUT"
                ;;
            *)
                DEVICE_LIMITS["${ACCOUNTS_DEVICE}_max_iops"]="$ACCOUNTS_VOL_MAX_IOPS"
                DEVICE_LIMITS["${ACCOUNTS_DEVICE}_max_throughput"]="$ACCOUNTS_VOL_MAX_THROUGHPUT"
                ;;
        esac
        echo "  ACCOUNTS Volume (${ACCOUNTS_DEVICE}): ${DEVICE_LIMITS["${ACCOUNTS_DEVICE}_max_iops"]} IOPS, ${DEVICE_LIMITS["${ACCOUNTS_DEVICE}_max_throughput"]} MiB/s"
    fi
}

# CSV field mapping initialization
init_csv_field_mapping() {
    local csv_file="$1"
    
    if [[ ! -f "$csv_file" ]]; then
        log_error "CSV file does not exist: $csv_file"
        return 1
    fi
    
    local header_line=$(head -n 1 "$csv_file" 2>/dev/null)
    if [[ -z "$header_line" ]]; then
        log_error "Unable to read CSV file header: $csv_file"
        return 1
    fi
    
    # Clear existing mapping
    declare -gA CSV_FIELD_MAP
    
    # Build field name to index mapping
    local index=0
    IFS=',' read -ra header_fields <<< "$header_line"
    for field in "${header_fields[@]}"; do
        # Remove whitespace from field name
        field=$(echo "$field" | tr -d ' \t\r\n')
        CSV_FIELD_MAP["$field"]=$index
        ((index++))
    done
    
    log_info "✅ CSV field mapping initialization completed, mapped $index fields"
    return 0
}

# Read provider from the CSV cloud_provider column instead of probing the runtime
# environment. This keeps offline analysis aligned with the writer environment.
# Input: fields array name (nameref). Fallback is provider-neutral "other".
_disk_provider_from_row() {
    local -n _fields_ref="$1"
    local idx="${CSV_FIELD_MAP[cloud_provider]:-}"
    local p=""
    [[ -n "$idx" ]] && p="${_fields_ref[$idx]:-}"
    p=$(echo "$p" | tr -d ' \t\r\n')
    case "$p" in
        aws|gcp|other) echo "$p" ;;
        *) echo "other" ;;   # Unknown or empty provider falls back to other.
    esac
}

# Resolve logical field names to physical CSV column names through the registry.
# Input: logical_name prefix provider.
# If the registry is unavailable, return empty so callers use their existing
# zero-value fallback path.
_disk_resolve() {
    local logical="$1" prefix="$2" provider="$3"
    if declare -F csv_registry_resolve >/dev/null 2>&1; then
        csv_registry_resolve "$logical" "$provider" "$prefix"
    else
        echo ""   # Missing registry returns empty; callers use their existing zero fallback.
    fi
}

# Extract Disk data from CSV row
get_disk_data_from_csv() {
    local device="$1"
    local csv_line="$2"
    
    if [[ -z "$device" || -z "$csv_line" ]]; then
        echo "0,0,0,0,0,0,0"
        return
    fi
    
    # Parse CSV row
    IFS="," read -ra fields <<< "$csv_line"
    
    # Determine field prefix based on device type
    local prefix=""
    if [[ "$device" == "$LEDGER_DEVICE" ]]; then
        prefix="data_${device}"
    elif [[ -n "$ACCOUNTS_DEVICE" && "$device" == "$ACCOUNTS_DEVICE" ]]; then
        prefix="accounts_${device}"
    else
        log_warn "Unknown device: $device, returning default values"
        echo "0,0,0,0,0,0,0"
        return
    fi
    
    # provider comes from CSV cloud_provider and registry-resolved physical column names,
    # keeping offline analysis aligned with the writer environment.
    local provider
    provider="$(_disk_provider_from_row fields)"
    local std_iops_field std_throughput_field
    std_iops_field="$(_disk_resolve disk_iops_provider_adjusted "$prefix" "$provider")"
    std_throughput_field="$(_disk_resolve disk_throughput_provider_adjusted "$prefix" "$provider")"

    # Extract field values using CSV_FIELD_MAP
    local util_index="${CSV_FIELD_MAP["${prefix}_util"]:-}"
    local total_iops_index="${CSV_FIELD_MAP["${prefix}_total_iops"]:-}"
    local std_iops_index="${CSV_FIELD_MAP["$std_iops_field"]:-}"
    local std_throughput_index="${CSV_FIELD_MAP["$std_throughput_field"]:-}"
    local r_await_index="${CSV_FIELD_MAP["${prefix}_r_await"]:-}"
    local w_await_index="${CSV_FIELD_MAP["${prefix}_w_await"]:-}"
    
    # Safely extract field values, use default value 0
    local util="${fields[$util_index]:-0}"
    local total_iops="${fields[$total_iops_index]:-0}"
    local std_iops="${fields[$std_iops_index]:-0}"
    local std_throughput="${fields[$std_throughput_index]:-0}"
    local r_await="${fields[$r_await_index]:-0}"
    local w_await="${fields[$w_await_index]:-0}"
    
    # Numeric validation: ensure all values are valid numbers
    if ! [[ "$util" =~ ^[0-9]*\.?[0-9]+$ ]]; then util="0"; fi
    if ! [[ "$total_iops" =~ ^[0-9]*\.?[0-9]+$ ]]; then total_iops="0"; fi
    if ! [[ "$std_iops" =~ ^[0-9]*\.?[0-9]+$ ]]; then std_iops="0"; fi
    if ! [[ "$std_throughput" =~ ^[0-9]*\.?[0-9]+$ ]]; then std_throughput="0"; fi
    if ! [[ "$r_await" =~ ^[0-9]*\.?[0-9]+$ ]]; then r_await="0"; fi
    if ! [[ "$w_await" =~ ^[0-9]*\.?[0-9]+$ ]]; then w_await="0"; fi
    
    # Return standardized format: util,total_iops,std_iops,std_throughput,r_await,w_await,avg_io_kib
    echo "$util,$total_iops,$std_iops,$std_throughput,$r_await,$w_await,0"
}

# Validate required CSV fields exist
validate_required_csv_fields() {
    local required_fields=()           # Provider-independent fields matched exactly.
    local provider_aware_prefixes=()   # Prefixes for provider-aware registry validation.
    
    # Add required fields for LEDGER_DEVICE
    if [[ -n "$LEDGER_DEVICE" ]]; then
        required_fields+=("data_${LEDGER_DEVICE}_util")
        required_fields+=("data_${LEDGER_DEVICE}_total_iops")
        required_fields+=("data_${LEDGER_DEVICE}_r_await")
        required_fields+=("data_${LEDGER_DEVICE}_w_await")
        provider_aware_prefixes+=("data_${LEDGER_DEVICE}")
    fi
    
    # Add required fields for ACCOUNTS_DEVICE (if configured)
    if [[ -n "$ACCOUNTS_DEVICE" ]]; then
        required_fields+=("accounts_${ACCOUNTS_DEVICE}_util")
        required_fields+=("accounts_${ACCOUNTS_DEVICE}_total_iops")
        required_fields+=("accounts_${ACCOUNTS_DEVICE}_r_await")
        required_fields+=("accounts_${ACCOUNTS_DEVICE}_w_await")
        provider_aware_prefixes+=("accounts_${ACCOUNTS_DEVICE}")
    fi
    
    # Validate each provider-independent field exists in CSV_FIELD_MAP.
    for field in "${required_fields[@]}"; do
        if [[ -z "${CSV_FIELD_MAP[$field]:-}" ]]; then
            log_error "❌ Critical field missing: $field"
            log_error "❌ CSV format may be incompatible or device configuration incorrect"
            log_error "❌ Current configuration: LEDGER_DEVICE=$LEDGER_DEVICE, ACCOUNTS_DEVICE=$ACCOUNTS_DEVICE"
            return 1
        fi
    done

    # Validate provider-aware fields (IOPS/throughput) through the registry.
    # Any supported provider variant is acceptable because CSVs carry their own
    # cloud_provider value and current provider naming is normalized.
    local pfx logical phys variant_found
    for pfx in "${provider_aware_prefixes[@]}"; do
        for logical in disk_iops_provider_adjusted disk_throughput_provider_adjusted; do
            variant_found=0
            for variant in aws gcp other; do
                phys="$(_disk_resolve "$logical" "$pfx" "$variant")"
                if [[ -n "${CSV_FIELD_MAP[$phys]:-}" ]]; then
                    variant_found=1
                    break
                fi
            done
            if [[ $variant_found -eq 0 ]]; then
                log_error "❌ Critical provider-aware field missing: ${pfx} / ${logical} (no provider variant exists in the CSV header)"
                log_error "❌ CSV format may be incompatible or device configuration incorrect"
                log_error "❌ Current configuration: LEDGER_DEVICE=$LEDGER_DEVICE, ACCOUNTS_DEVICE=$ACCOUNTS_DEVICE"
                return 1
            fi
        done
    done
    
    log_info "✅ All critical fields validated, verified ${#required_fields[@]} fields"
    return 0
}

# CSV event-driven monitoring
start_csv_monitoring() {
    local duration="$1"
    local csv_file="${PERFORMANCE_LATEST_CSV:-${LOGS_DIR}/performance_latest.csv}"
    
    # Set cleanup function
    cleanup_csv_monitoring() {
        log_info "Cleaning up CSV monitoring process..."
        # Clean up possible tail processes
        pkill -P $$ -f "tail.*performance_latest.csv" 2>/dev/null || true
        exit 0
    }
    
    # Set signal handling
    trap cleanup_csv_monitoring EXIT INT TERM
    
    log_info "🚀 Starting CSV event-driven monitoring mode"
    log_info "📊 Data source: $csv_file"
    if [[ "$duration" -eq 0 ]]; then
        log_info "⏱️  Monitoring mode: Follow framework lifecycle"
    else
        log_info "⏱️  Monitoring duration: ${duration}s"
    fi
    
    # Initialize CSV field mapping
    if ! init_csv_field_mapping "$csv_file"; then
        log_error "❌ CSV field mapping initialization failed"
        return 1
    fi
    
    # Validate required fields
    if ! validate_required_csv_fields; then
        log_error "❌ Required field validation failed"
        return 1
    fi
    
    log_info "📊 Event-driven mode: Listening for CSV file changes"
    
    # Select monitoring mode based on duration
    if [[ "$duration" -eq 0 ]]; then
        # Poll complete CSV ranges so removing the lifecycle marker wakes the
        # loop even when no new sample arrives. A persistent tail pipeline can
        # otherwise outlive the benchmark and retain the runner's descriptors.
        log_info "📊 Using framework lifecycle control mode"
        local next_line=2
        while [[ -f "$TMP_DIR/qps_test_status" ]]; do
            local total_lines=0
            total_lines=$(wc -l < "$csv_file" 2>/dev/null || echo 0)
            if (( total_lines < next_line - 1 )); then
                log_warn "⚠️  Detected CSV rotation, reinitializing field mapping"
                init_csv_field_mapping "$csv_file"
                next_line=2
            fi
            if (( total_lines >= next_line )); then
                while IFS= read -r line; do
                    [[ "$line" =~ ^timestamp ]] && continue
                    [[ -z "$line" ]] && continue

                    local timestamp
                    timestamp=$(echo "$line" | cut -d',' -f1)
                    if [[ ! "$timestamp" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2} ]]; then
                        log_warn "⚠️  Detected CSV format change, reinitializing field mapping"
                        init_csv_field_mapping "$csv_file"
                        continue
                    fi

                    for device in "$LEDGER_DEVICE" "$ACCOUNTS_DEVICE"; do
                        [[ -z "$device" ]] && continue
                        local metrics
                        metrics=$(get_disk_data_from_csv "$device" "$line")
                        if [[ -n "$metrics" && "$metrics" != "0,0,0,0,0,0,0" ]]; then
                            IFS=',' read -r util total_iops std_iops std_throughput r_await w_await _ <<< "$metrics"
                            local avg_latency
                            avg_latency=$(awk "BEGIN {printf \"%.2f\", ($r_await + $w_await) / 2}" 2>/dev/null || echo "0")
                            detect_disk_bottleneck "$device" "$total_iops" "$std_iops" "$std_throughput" "$avg_latency" "$timestamp"
                            local bottleneck_detected=$?
                            log_info "$timestamp,$device,$total_iops,$std_throughput,$avg_latency,$bottleneck_detected"
                        fi
                    done
                done < <(sed -n "${next_line},${total_lines}p" "$csv_file")
                next_line=$((total_lines + 1))
            fi
            sleep 1
        done
    else
        # Fixed duration mode: keep original timeout logic
        log_info "📊 Using fixed duration mode: ${duration} seconds"
        timeout "$duration" tail -F "$csv_file" 2>/dev/null | while IFS= read -r line; do
            # Skip header and empty lines
            [[ "$line" =~ ^timestamp ]] && continue
            [[ -z "$line" ]] && continue
            
            # Detect file rotation: if timestamp format is abnormal, reinitialize field mapping
            local timestamp=$(echo "$line" | cut -d',' -f1)
            if [[ ! "$timestamp" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2} ]]; then
                log_warn "⚠️  Detected CSV format change, reinitializing field mapping"
                init_csv_field_mapping "$csv_file"
                continue
            fi
            
            # Monitor each configured device
            for device in "$LEDGER_DEVICE" "$ACCOUNTS_DEVICE"; do
                [[ -z "$device" ]] && continue
                
                # Extract Disk data from CSV
                local metrics=$(get_disk_data_from_csv "$device" "$line")
                
                if [[ -n "$metrics" && "$metrics" != "0,0,0,0,0,0,0" ]]; then
                    IFS=',' read -r util total_iops std_iops std_throughput r_await w_await _ <<< "$metrics"
                    
                    # Calculate average latency
                    local avg_latency=$(awk "BEGIN {printf "%.2f", ($r_await + $w_await) / 2}" 2>/dev/null || echo "0")
                    
                    # Perform bottleneck detection (using provider-standardized parameters)
                    detect_disk_bottleneck "$device" "$total_iops" "$std_iops" "$std_throughput" "$avg_latency" "$timestamp"
                    
                    local bottleneck_detected=$?
                    log_info "$timestamp,$device,$total_iops,$std_throughput,$avg_latency,$bottleneck_detected"
                fi
            done
        done
    fi
    
    # Handle tail -F abnormal exit
    local exit_code=$?
    if [[ $exit_code -ne 0 && $exit_code -ne 124 ]]; then  # 124 is timeout normal exit
        log_error "⚠️  CSV listening exited abnormally (exit code: $exit_code)"
        return $exit_code
    fi
    
    log_info "✅ CSV event-driven monitoring completed"
    return 0
}

# Wait for CSV file to be ready
wait_for_csv_ready() {
    local csv_file="${PERFORMANCE_LATEST_CSV:-${LOGS_DIR}/performance_latest.csv}"
    local max_wait=60  # 60 second timeout
    local wait_count=0
    
    log_info "⏳ Waiting for CSV file to be ready: $csv_file"
    
    while [[ $wait_count -lt $max_wait ]]; do
        # Check if symlink exists
        if [[ -L "$csv_file" ]]; then
            local target=$(readlink "$csv_file")
            local target_file="${LOGS_DIR}/$target"
            
            # Check if target file exists and has content
            if [[ -f "$target_file" && -s "$target_file" ]]; then
                # Check if file has data rows (not just header)
                local line_count=$(wc -l < "$target_file" 2>/dev/null || echo 0)
                if [[ $line_count -gt 1 ]]; then
                    # Validate header format
                    local header_line=$(head -n 1 "$target_file" 2>/dev/null)
                    if [[ -n "$header_line" && "$header_line" =~ ^timestamp ]]; then
                        log_info "✅ CSV file ready: $csv_file -> $target_file"
                        return 0
                    fi
                fi
            fi
        fi
        
        echo "   Waiting for CSV data generation... ($((wait_count + 1))/$max_wait)"
        sleep 1
        ((wait_count++))
    done
    
    log_error "❌ Timeout: CSV file not ready within ${max_wait} seconds"
    log_error "❌ Please ensure unified_monitor.sh is running and generating CSV data"
    return 1
}




# Detect Disk bottleneck
detect_disk_bottleneck() {
    local device=$1
    local current_iops=$2
    local current_aws_iops=$3
    local current_throughput=$4
    local current_latency=$5
    local timestamp=$6
    
    local bottleneck_detected=false
    local bottleneck_type=""
    local severity=""
    
    # Get device limits
    local max_iops=${DEVICE_LIMITS["${device}_max_iops"]}
    local max_throughput=${DEVICE_LIMITS["${device}_max_throughput"]}
    
    if [[ -z "$max_iops" || -z "$max_throughput" ]]; then
        return 0
    fi
    
    # IOPS bottleneck detection
    local iops_utilization=$(awk "BEGIN {printf \"%.4f\", $current_aws_iops / $max_iops}")
    if (( $(awk "BEGIN {print ($iops_utilization > $BOTTLENECK_IOPS_THRESHOLD) ? 1 : 0}") )); then
        bottleneck_detected=true
        bottleneck_type="${bottleneck_type}IOPS,"
        
        # Use configurable threshold instead of hardcoded value
        local critical_threshold=$(awk "BEGIN {printf \"%.2f\", (${BOTTLENECK_DISK_IOPS_THRESHOLD:-90} + 5) / 100}")
        local high_threshold=$(awk "BEGIN {printf \"%.2f\", ${BOTTLENECK_DISK_IOPS_THRESHOLD:-90} / 100}")
        
        if (( $(awk "BEGIN {print ($iops_utilization > $critical_threshold) ? 1 : 0}") )); then
            severity="CRITICAL"
        elif (( $(awk "BEGIN {print ($iops_utilization > $high_threshold) ? 1 : 0}") )); then
            severity="HIGH"
        else
            severity="MEDIUM"
        fi
    fi
    
    # Throughput bottleneck detection
    local throughput_utilization=$(awk "BEGIN {printf \"%.4f\", $current_throughput / $max_throughput}")
    if (( $(awk "BEGIN {print ($throughput_utilization > $BOTTLENECK_THROUGHPUT_THRESHOLD) ? 1 : 0}") )); then
        bottleneck_detected=true
        bottleneck_type="${bottleneck_type}THROUGHPUT,"
        
        # Use configurable threshold instead of hardcoded value
        local critical_threshold=$(awk "BEGIN {printf \"%.2f\", (${BOTTLENECK_DISK_THROUGHPUT_THRESHOLD:-90} + 5) / 100}")
        local high_threshold=$(awk "BEGIN {printf \"%.2f\", ${BOTTLENECK_DISK_THROUGHPUT_THRESHOLD:-90} / 100}")
        
        if (( $(awk "BEGIN {print ($throughput_utilization > $critical_threshold) ? 1 : 0}") )); then
            severity="CRITICAL"
        elif (( $(awk "BEGIN {print ($throughput_utilization > $high_threshold) ? 1 : 0}") )); then
            severity="HIGH"
        else
            severity="MEDIUM"
        fi
    fi
    
    # Latency bottleneck detection
    local latency_threshold=${BOTTLENECK_DISK_LATENCY_THRESHOLD:-50}
    if (( $(awk "BEGIN {print ($current_latency > $latency_threshold) ? 1 : 0}") )); then
        bottleneck_detected=true
        bottleneck_type="${bottleneck_type}LATENCY,"
        
        # Latency severity classification
        local critical_latency_threshold=$(awk "BEGIN {printf \"%.2f\", $latency_threshold * 2}")
        if (( $(awk "BEGIN {print ($current_latency > $critical_latency_threshold) ? 1 : 0}") )); then
            severity="CRITICAL"
        else
            severity="HIGH"
        fi
    fi
    
    # Log bottleneck event
    if [[ "$bottleneck_detected" == "true" ]]; then
        local bottleneck_record="$timestamp,$device,$bottleneck_type,$severity,$current_aws_iops,$max_iops,$iops_utilization,$current_throughput,$max_throughput,$throughput_utilization,$current_latency"
        log_info "$bottleneck_record"
        
        # Real-time warning
        echo "⚠️  [$(date '+%H:%M:%S')] Disk BOTTLENECK DETECTED: $device - $bottleneck_type (Severity: $severity)"
        echo "   IOPS: $current_aws_iops/$max_iops (${iops_utilization%.*}%), Throughput: $current_throughput/$max_throughput MiB/s (${throughput_utilization%.*}%)"
        
        return 1
    fi
    
    return 0
}

# Start high-frequency monitoring
start_high_freq_monitoring() {
    local duration="$1"
    local qps_test_mode="${2:-false}"  # Whether in QPS test mode
    
    # Add continuous running mode support
    if [[ "$duration" -eq 0 ]]; then
        log_info "🔄 Continuous running mode (follow framework lifecycle)"
    fi

    log_info "🚀 Starting Disk bottleneck detection (producer-consumer mode)"
    log_info "   Duration: ${duration}s"
    log_info "   Data Source: iostat_collector.sh → unified_monitor.sh → performance_latest.csv"
    log_info "   Consumer Mode: Event-driven with dynamic field mapping"
    
    # Initialize Disk limits configuration
    init_disk_limits
    

    # Try CSV event-driven mode
    if wait_for_csv_ready; then
        log_info "✅ Using CSV event-driven mode"
        start_csv_monitoring "$duration"
        local csv_result=$?
        
        if [[ $csv_result -eq 0 ]]; then
            log_info "✅ CSV event-driven monitoring completed successfully"
            return 0
        else
            log_error "❌ CSV event-driven monitoring failed (exit code: $csv_result)"
            return $csv_result
        fi
    else
        log_error "❌ CSV data source unavailable, exiting and reporting dependency failure"
        log_error "❌ Please ensure unified_monitor.sh is running and generating CSV data"
        log_error "❌ Check if monitoring coordinator correctly started dependent services"
        exit 1
    fi
}

# Generate monitoring summary
generate_monitoring_summary() {
    local data_file=$1
    local summary_file="${data_file%.*}_summary.txt"
    
    echo "📊 Generating monitoring summary..."
    
    {
        echo "==============================================="
        echo "Disk High-Frequency Monitoring Summary"
        echo "==============================================="
        echo "Generated: $(date)"
        echo "Data File: $data_file"
        echo "Bottleneck Log: $BOTTLENECK_LOG_FILE"
        echo ""
        
        echo "=== Disk Configuration ==="
        for device in "$LEDGER_DEVICE" "$ACCOUNTS_DEVICE"; do
            if [[ -n "$device" && -n "${DEVICE_LIMITS["${device}_max_iops"]}" ]]; then
                echo "$device:"
                echo "  Max IOPS: ${DEVICE_LIMITS["${device}_max_iops"]}"
                echo "  Max Throughput: ${DEVICE_LIMITS["${device}_max_throughput"]} MiB/s"
                echo "  Peak IOPS Observed: ${PEAK_VALUES["${device}_max_iops"]}"
                echo "  Peak Throughput Observed: ${PEAK_VALUES["${device}_max_throughput"]} MiB/s"
                echo "  IOPS Bottleneck Events: ${BOTTLENECK_COUNTERS["${device}_iops_exceeded"]}"
                echo "  Throughput Bottleneck Events: ${BOTTLENECK_COUNTERS["${device}_throughput_exceeded"]}"
                echo ""
            fi
        done
        
        echo "=== Bottleneck Analysis ==="
        if [[ -f "$BOTTLENECK_LOG_FILE" ]]; then
            local total_bottlenecks=$(tail -n +2 "$BOTTLENECK_LOG_FILE" | wc -l)
            echo "Total Bottleneck Events: $total_bottlenecks"
            
            if [[ $total_bottlenecks -gt 0 ]]; then
                echo ""
                echo "Bottleneck Event Details:"
                echo "------------------------"
                tail -n +2 "$BOTTLENECK_LOG_FILE" | while IFS=',' read -r timestamp device bottleneck_type severity current_aws_iops max_iops iops_util current_throughput max_throughput throughput_util latency; do
                    echo "[$timestamp] $device: $bottleneck_type (Severity: $severity)"
                    echo "  IOPS: $current_aws_iops/$max_iops ($(awk "BEGIN {printf \"%.0f\", $iops_util * 100}")%)"
                    echo "  Throughput: $current_throughput/$max_throughput MiB/s ($(awk "BEGIN {printf \"%.0f\", $throughput_util * 100}")%)"
                    echo ""
                done
            fi
        else
            echo "No bottleneck events detected ✅"
        fi
        
        echo "==============================================="
        
    } > "$summary_file"
    
    echo "📄 Summary saved to: $summary_file"
    
    # Display key information
    echo ""
    echo "🎯 Key Findings:"
    for device in "$LEDGER_DEVICE" "$ACCOUNTS_DEVICE"; do
        if [[ -n "$device" && -n "${DEVICE_LIMITS["${device}_max_iops"]}" ]]; then
            local peak_iops=${PEAK_VALUES["${device}_max_iops"]}
            local max_iops=${DEVICE_LIMITS["${device}_max_iops"]}
            local peak_utilization=$(awk "BEGIN {printf \"%.1f\", $peak_iops / $max_iops * 100}")
            
            echo "  $device: Peak utilization ${peak_utilization}% (${peak_iops}/${max_iops} IOPS)"
            
            if (( $(awk "BEGIN {print ($peak_utilization > 85) ? 1 : 0}") )); then
                echo "    ⚠️  HIGH UTILIZATION - Consider upgrading Disk configuration"
            elif (( $(awk "BEGIN {print ($peak_utilization > 70) ? 1 : 0}") )); then
                echo "    ⚠️  MODERATE UTILIZATION - Monitor closely"
            else
                echo "    ✅ NORMAL UTILIZATION"
            fi
        fi
    done
}

# Start Disk monitoring during QPS test
start_disk_monitoring_for_qps_test() {
    local qps_duration="$1"
    local qps_start_time="$2"
    
    if [[ -z "$qps_duration" ]]; then
        echo "❌ QPS test duration not specified"
        return 1
    fi
    
    echo "🔗 Starting Disk bottleneck monitoring (QPS test mode)"
    echo "   QPS test duration: ${qps_duration}s"
    echo "   QPS start time: ${qps_start_time:-$(date +'%Y-%m-%d %H:%M:%S')}"
    echo "   Disk monitoring will run synchronously with QPS test"
    echo ""
    
    # Record QPS test time range
    export QPS_TEST_START_TIME="${qps_start_time:-$(date +'%Y-%m-%d %H:%M:%S')}"
    export QPS_TEST_DURATION="$qps_duration"
    
    # Start Disk monitoring synchronized with QPS test
    start_high_freq_monitoring "$qps_duration" "true"
}

# Stop Disk monitoring - new function
stop_disk_monitoring() {
    echo "🛑 Stopping Disk bottleneck monitoring..."
    
    # Terminate all related monitoring processes
    pkill -f "disk_bottleneck_detector" 2>/dev/null || true
    pkill -f "iostat.*${MONITOR_INTERVAL}" 2>/dev/null || true
    
    echo "✅ Disk monitoring stopped"
    
    # Generate monitoring summary
    if [[ -f "$BOTTLENECK_LOG_FILE" ]]; then
        local bottleneck_count=$(wc -l < "$BOTTLENECK_LOG_FILE" 2>/dev/null || echo "0")
        echo "📊 Detected $bottleneck_count Disk bottleneck events during monitoring"
        
        if [[ $bottleneck_count -gt 0 ]]; then
            echo "⚠️  Disk bottleneck details see: $BOTTLENECK_LOG_FILE"
        fi
    fi
}

# Main function
main() {
    echo "🔧 Disk Bottleneck Detector"
    echo "=========================="
    echo ""
    
    # Initialize
    init_disk_limits
    echo ""
    
    # Parse arguments
    local duration=300
    local background=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--duration)
                duration="$2"
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
                echo "  -d, --duration SECONDS    Monitoring duration (default: 300)"
                echo "  -b, --background          Run in background"
                echo "  -h, --help               Show this help"
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
        echo "🔄 Framework lifecycle integration mode"
        echo "📝 Logging to: ${LOGS_DIR}/disk_bottleneck_detector.log"
        # Directly call monitoring function, don't restart process
        # duration=0 means continuous running, follow framework lifecycle
        # Redirect output to log file
        exec > "${LOGS_DIR}/disk_bottleneck_detector.log" 2>&1
        start_high_freq_monitoring 0
    else
        start_high_freq_monitoring "$duration"
    fi
}

# If executing this script directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
