#!/bin/bash
# =====================================================================
# iostat Data Collector
# =====================================================================
# Unified iostat data collection and processing logic
# Eliminate empirical values, calculate precisely based on real-time data
# =====================================================================

# Safely load configuration file to avoid readonly variable conflicts.
# Unit tests can skip the full config loader when they only need parser helpers.
if [[ "${IOSTAT_COLLECTOR_SKIP_CONFIG_LOAD:-0}" != "1" ]]; then
    if ! source "$(dirname "${BASH_SOURCE[0]}")/../config/config_loader.sh" 2>/dev/null; then
        echo "Warning: Configuration file loading failed, using default configuration"
        LOGS_DIR=${LOGS_DIR:-"/tmp/blockchain-node-benchmark/logs"}
    fi
else
    LOGS_DIR=${LOGS_DIR:-"/tmp/blockchain-node-benchmark/logs"}
fi
source "$(dirname "${BASH_SOURCE[0]}")/../utils/disk_converter.sh"
# CSV Schema Registry for disk headers
source "$(dirname "${BASH_SOURCE[0]}")/../config/csv_schema_registry.sh"

# Load logging functions. In parser-only unit tests, avoid unified_logger
# because it intentionally loads the full runtime configuration.
if [[ "${IOSTAT_COLLECTOR_SKIP_CONFIG_LOAD:-0}" == "1" ]]; then
    log_warn() { echo "WARN: $*" >&2; }
    log_debug() { :; }
else
    source "$(dirname "${BASH_SOURCE[0]}")/../utils/unified_logger.sh" 2>/dev/null || {
    # Provide simple alternatives if logging functions are unavailable
        log_warn() { echo "WARN: $*" >&2; }
        log_debug() { :; }
    }
fi

_iostat_field_by_header() {
    local header="$1"
    local stats="$2"
    local field_name="$3"
    local default_value="${4:-0}"

    awk -v header="$header" -v stats="$stats" -v field="$field_name" -v default_value="$default_value" '
        BEGIN {
            split(header, header_fields)
            split(stats, stats_fields)
            for (i = 1; i in header_fields; i++) {
                if (header_fields[i] == field) {
                    if ((i in stats_fields) && stats_fields[i] != "") {
                        print stats_fields[i]
                    } else {
                        print default_value
                    }
                    exit
                }
            }
            print default_value
        }
    '
}

_device_visible_to_iostat() {
    local device="$1"

    [[ -z "$device" ]] && return 1
    command -v iostat >/dev/null 2>&1 || return 1

    iostat -dx 1 1 2>/dev/null | awk -v dev="$device" '
        $1 == dev { found = 1 }
        END { exit(found ? 0 : 1) }
    '
}

_device_is_monitorable() {
    local device="$1"

    [[ -z "$device" ]] && return 1
    [[ -b "/dev/$device" ]] && return 0
    _device_visible_to_iostat "$device"
}

# Get complete iostat data
get_iostat_data() {
    local device="$1"
    local logical_name="$2"  # data or accounts
    
    if [[ -z "$device" || -z "$logical_name" ]]; then
        echo "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
        return
    fi
    
    # Implement true iostat continuous sampling
    local monitor_rate=${DISK_MONITOR_RATE:-1}
    local iostat_runtime_dir="${TMP_DIR:-/tmp}"
    mkdir -p "$iostat_runtime_dir" 2>/dev/null || true
    local iostat_pid_file="${iostat_runtime_dir}/iostat_${device}_${logical_name}.pid"
    local iostat_data_file="${iostat_runtime_dir}/iostat_${device}_${logical_name}.data"
    
    # Check if continuous sampling process already exists
    if [[ ! -f "$iostat_pid_file" ]] || ! kill -0 "$(cat "$iostat_pid_file" 2>/dev/null)" 2>/dev/null; then
        # Start continuous sampling process
        if [[ "$(uname -s)" == "Linux" ]]; then
            iostat -dx "$monitor_rate" > "$iostat_data_file" &
            local iostat_pid=$!
            echo "$iostat_pid" > "$iostat_pid_file"
            log_debug "Started iostat continuous sampling: $device, PID: $iostat_pid, Rate: ${monitor_rate}s, Data file: $iostat_data_file"
        else
            log_warn "iostat functionality only available in Linux environment, current system: $(uname -s)"
            return 1
        fi
    fi
    
    # Get latest iostat header and matching device data line.
    # sysstat field order differs across Linux distributions and versions
    # (for example discard/flush fields may appear before aqu-sz/%util), so
    # parsing by fixed column index silently corrupts disk metrics.
    local iostat_record
    iostat_record=$(tail -n 120 "$iostat_data_file" 2>/dev/null | awk -v dev="$device" '
        $1 == "Device" { header = $0 }
        $1 == dev { latest_header = header; latest = $0 }
        END {
            if (latest != "") {
                print latest_header
                print latest
            }
        }
    ')
    local iostat_header
    iostat_header=$(printf "%s\n" "$iostat_record" | sed -n '1p')
    local device_stats
    device_stats=$(printf "%s\n" "$iostat_record" | sed -n '2p')
    
    if [[ -z "$device_stats" || -z "$iostat_header" ]]; then
        echo "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
        return
    fi

    # Extract iostat fields by header name instead of fixed position.
    local r_s=$(_iostat_field_by_header "$iostat_header" "$device_stats" "r/s")
    local rkb_s=$(_iostat_field_by_header "$iostat_header" "$device_stats" "rkB/s")
    local rrqm_s=$(_iostat_field_by_header "$iostat_header" "$device_stats" "rrqm/s")
    local rrqm_pct=$(_iostat_field_by_header "$iostat_header" "$device_stats" "%rrqm")
    local r_await=$(_iostat_field_by_header "$iostat_header" "$device_stats" "r_await")
    local rareq_sz=$(_iostat_field_by_header "$iostat_header" "$device_stats" "rareq-sz")
    local w_s=$(_iostat_field_by_header "$iostat_header" "$device_stats" "w/s")
    local wkb_s=$(_iostat_field_by_header "$iostat_header" "$device_stats" "wkB/s")
    local wrqm_s=$(_iostat_field_by_header "$iostat_header" "$device_stats" "wrqm/s")
    local wrqm_pct=$(_iostat_field_by_header "$iostat_header" "$device_stats" "%wrqm")
    local w_await=$(_iostat_field_by_header "$iostat_header" "$device_stats" "w_await")
    local wareq_sz=$(_iostat_field_by_header "$iostat_header" "$device_stats" "wareq-sz")
    local aqu_sz=$(_iostat_field_by_header "$iostat_header" "$device_stats" "aqu-sz")
    local util=$(_iostat_field_by_header "$iostat_header" "$device_stats" "%util")
    
    # Calculate derived metrics (based on real-time data, no empirical values)
    local total_iops=$(awk "BEGIN {printf \"%.2f\", $r_s + $w_s}" 2>/dev/null || echo "0")
    local total_throughput_kbs=$(awk "BEGIN {printf \"%.2f\", $rkb_s + $wkb_s}" 2>/dev/null || echo "0")
    local total_throughput_mibs=$(awk "BEGIN {printf \"%.2f\", $total_throughput_kbs / 1024}" 2>/dev/null || echo "0")
    
    # Calculate separate read/write throughput (KB/s → MiB/s)
    local read_throughput_mibs=$(awk "BEGIN {printf \"%.2f\", $rkb_s / 1024}" 2>/dev/null || echo "0")
    local write_throughput_mibs=$(awk "BEGIN {printf \"%.2f\", $wkb_s / 1024}" 2>/dev/null || echo "0")
    
    # Calculate provider-standard throughput
    local standard_throughput_mibs="0"
    if command -v convert_to_standard_throughput >/dev/null 2>&1; then
        # Calculate weighted average IO size
        local weighted_avg_io_kib
        if [[ $(awk "BEGIN {print ($total_iops > 0) ? 1 : 0}") -eq 1 ]]; then
            weighted_avg_io_kib=$(awk "BEGIN {printf \"%.2f\", $total_throughput_kbs / $total_iops}" 2>/dev/null || echo "0")
        else
            weighted_avg_io_kib="0"
        fi
        
        if [[ "$weighted_avg_io_kib" != "0" ]]; then
            standard_throughput_mibs=$(convert_to_standard_throughput "$total_throughput_mibs" "$weighted_avg_io_kib")
        else
            standard_throughput_mibs="$total_throughput_mibs"  # Use raw value if average IO size cannot be calculated
        fi
    else
        log_debug "convert_to_standard_throughput function unavailable, using raw throughput value"
        standard_throughput_mibs="$total_throughput_mibs"
    fi
    
    local avg_await=$(awk "BEGIN {printf \"%.2f\", ($r_await + $w_await) / 2}" 2>/dev/null || echo "0")
    
    # Calculate average I/O size (based on real-time data)
    local avg_io_kib
    if [[ $(awk "BEGIN {print ($total_iops > 0) ? 1 : 0}") -eq 1 ]]; then
        avg_io_kib=$(awk "BEGIN {printf \"%.2f\", $total_throughput_kbs / $total_iops}" 2>/dev/null || echo "0")
    else
        avg_io_kib="0"
    fi
    
    # Calculate provider-standard IOPS (based on real-time data)
    # NOTE: HDD-specific conversion is not implemented. convert_to_standard_iops
    # defaults to a 256 KiB I/O cap, which matches SSD-oriented cloud volumes.
    # If HDD support is added later, provider conversion should branch by volume
    # type and pass the correct I/O cap.
    local standard_iops
    if [[ $(awk "BEGIN {print ($avg_io_kib > 0) ? 1 : 0}") -eq 1 ]]; then
        standard_iops=$(convert_to_standard_iops "$total_iops" "$avg_io_kib")
    else
        standard_iops="$total_iops"
    fi
    
    # Return complete data (21 fields)
    echo "$r_s,$w_s,$rkb_s,$wkb_s,$r_await,$w_await,$avg_await,$aqu_sz,$util,$rrqm_s,$wrqm_s,$rrqm_pct,$wrqm_pct,$rareq_sz,$wareq_sz,$total_iops,$standard_iops,$read_throughput_mibs,$write_throughput_mibs,$total_throughput_mibs,$standard_throughput_mibs"
}

# Generate CSV header for device
# Column names are generated through csv_schema_registry.
# Provider-normalized disk fields use normalized_iops/normalized_throughput_mibs.
#   Provider identity is carried by the CSV cloud_provider column.
# The optional provider argument is passed through to the registry.
# When omitted, get_provider_name is used.
generate_device_header() {
    local device="$1"
    local logical_name="$2"
    local provider="${3:-}"
    if [[ -z "$provider" ]]; then
        if declare -F get_provider_name >/dev/null 2>&1; then
            provider="$(get_provider_name 2>/dev/null)"
        fi
        provider="${provider:-other}"
    fi

    # Use unified naming convention {logical_name}_{device_name}_{metric}
    # DATA device uses data prefix, ACCOUNTS device uses accounts prefix
    local prefix
    case "$logical_name" in
        "data") prefix="data_${device}" ;;
        "accounts") prefix="accounts_${device}" ;;
        *) prefix="${logical_name}_${device}" ;;
    esac

    # Generate the 21-column disk header through the registry.
    if declare -F csv_registry_disk_header >/dev/null 2>&1; then
        csv_registry_disk_header "$prefix" "$provider"
    else
        # Defensive fallback when the registry was not loaded.
        local dfp="normalized"
        declare -F get_disk_field_prefix >/dev/null 2>&1 && dfp="$(get_disk_field_prefix 2>/dev/null || echo normalized)"
        log_warn "csv_registry_disk_header unavailable — fallback header (dfp=$dfp)"
        echo "${prefix}_r_s,${prefix}_w_s,${prefix}_rkb_s,${prefix}_wkb_s,${prefix}_r_await,${prefix}_w_await,${prefix}_avg_await,${prefix}_aqu_sz,${prefix}_util,${prefix}_rrqm_s,${prefix}_wrqm_s,${prefix}_rrqm_pct,${prefix}_wrqm_pct,${prefix}_rareq_sz,${prefix}_wareq_sz,${prefix}_total_iops,${prefix}_${dfp}_iops,${prefix}_read_throughput_mibs,${prefix}_write_throughput_mibs,${prefix}_total_throughput_mibs,${prefix}_${dfp}_throughput_mibs"
    fi
}

# Get data for all configured devices
get_all_devices_data() {
    local device_data=""

    # Degraded mode: device(s) unavailable — emit NaN placeholders matching header shape
    if [[ "${DEVICE_VALIDATION_DEGRADED:-0}" == "1" ]]; then
        # 21 NaN fields per device (matches get_iostat_data output)
        local nan_row="NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN"
        device_data="$nan_row"
        if is_accounts_configured; then
            device_data="${device_data},$nan_row"
        fi
        echo "$device_data"
        return 0
    fi

    # DATA device - use data as logical name prefix (required)
    if [[ -n "$DATA_VOL_TYPE" ]]; then
        local data_stats=$(get_iostat_data "$LEDGER_DEVICE" "data")
        device_data="$data_stats"
    else
        log_error "DATA_VOL_TYPE not configured - this is required"
        return 1
    fi

    # ACCOUNTS device - use accounts as logical name prefix
    if is_accounts_configured; then
        local accounts_stats=$(get_iostat_data "$ACCOUNTS_DEVICE" "accounts")
        if [[ -n "$device_data" ]]; then
            device_data="${device_data},$accounts_stats"
        else
            device_data="$accounts_stats"
        fi
    fi

    echo "$device_data"
}

# Generate CSV header for all devices
generate_all_devices_header() {
    local device_header=""

    # Writer-side provider comes from get_provider_name at runtime;
    # reader-side provider must come from the CSV cloud_provider column. Do not mix the two.
    # Resolve once and pass it to each generate_device_header call.
    local provider="aws"
    if declare -F get_provider_name >/dev/null 2>&1; then
        local _p; _p=$(get_provider_name 2>/dev/null)
        [[ -n "$_p" ]] && provider="$_p"
    fi

    # Degraded mode: use "NA" as device-name placeholder so header column count is stable
    if [[ "${DEVICE_VALIDATION_DEGRADED:-0}" == "1" ]]; then
        local data_dev="${LEDGER_DEVICE:-NA}"
        device_header=$(generate_device_header "$data_dev" "data" "$provider")
        if is_accounts_configured; then
            local acc_dev="${ACCOUNTS_DEVICE:-NA}"
            local accounts_header=$(generate_device_header "$acc_dev" "accounts" "$provider")
            device_header="${device_header},$accounts_header"
        fi
        echo "$device_header"
        return 0
    fi

    # DATA device header - use data as logical name prefix (required)
    if [[ -n "$DATA_VOL_TYPE" ]]; then
        device_header=$(generate_device_header "$LEDGER_DEVICE" "data" "$provider")
    else
        log_error "DATA_VOL_TYPE not configured - this is required"
        return 1
    fi

    # ACCOUNTS device header - use accounts as logical name prefix
    if is_accounts_configured; then
        local accounts_header=$(generate_device_header "$ACCOUNTS_DEVICE" "accounts" "$provider")
        if [[ -n "$device_header" ]]; then
            device_header="${device_header},$accounts_header"
        else
            device_header="$accounts_header"
        fi
    fi

    echo "$device_header"
}

# Validate device availability
# Supports STRICT_DEVICE_VALIDATION env var:
#   - true  : hard fail on missing devices (original behavior, for AWS EC2)
#   - false : degraded mode (default) — WARN, set DEVICE_VALIDATION_DEGRADED=1, return 0
validate_devices() {
    local errors=()
    local strict="${STRICT_DEVICE_VALIDATION:-false}"

    # DATA device validation (required)
    if [[ -z "$LEDGER_DEVICE" ]]; then
        errors+=("LEDGER_DEVICE is required but not configured")
    elif ! _device_is_monitorable "$LEDGER_DEVICE"; then
        errors+=("LEDGER_DEVICE '$LEDGER_DEVICE' is not visible as /dev/$LEDGER_DEVICE or in iostat output")
    fi

    if [[ -n "$ACCOUNTS_DEVICE" ]] && ! _device_is_monitorable "$ACCOUNTS_DEVICE"; then
        errors+=("ACCOUNTS_DEVICE '$ACCOUNTS_DEVICE' is not visible as /dev/$ACCOUNTS_DEVICE or in iostat output")
    fi

    if [[ ${#errors[@]} -gt 0 ]]; then
        if [[ "$strict" == "true" ]]; then
            printf "❌ Device validation failed:\n"
            printf "  - %s\n" "${errors[@]}"
            return 1
        else
            printf "⚠️  Device validation WARN (degraded mode, STRICT_DEVICE_VALIDATION=false):\n" >&2
            printf "  - %s\n" "${errors[@]}" >&2
            printf "⚠️  Disk I/O columns will be filled with N/A; CPU/mem/net monitoring still active.\n" >&2
            printf "💡 Set STRICT_DEVICE_VALIDATION=true to enforce hard failure when disk devices are not monitorable.\n" >&2
            export DEVICE_VALIDATION_DEGRADED=1
            return 0
        fi
    fi

    return 0
}

# If this script is executed directly, run test
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "🔧 iostat Data Collector Test"
    echo "========================="
    
    if validate_devices; then
        echo "✅ Device validation passed"
        echo ""
        echo "📊 CSV Header:"
        echo "timestamp,$(generate_all_devices_header)"
        echo ""
        echo "📊 Current Data:"
        echo "$(date +"$TIMESTAMP_FORMAT"),$(get_all_devices_data)"
    else
        echo "❌ Device validation failed"
        exit 1
    fi
fi
