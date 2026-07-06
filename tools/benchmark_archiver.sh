#!/bin/bash

# =====================================================================
# QPS Test Archiver - Archive test data by execution count
# =====================================================================

# Safely load configuration file, avoiding readonly variable conflicts
if ! source "$(dirname "${BASH_SOURCE[0]}")/../config/config_loader.sh" 2>/dev/null; then
    echo "Warning: Configuration file loading failed, using default configuration"
    DATA_DIR=${DATA_DIR:-"/tmp/blockchain-node-benchmark"}
fi

# Global variables
ARCHIVES_DIR="${DATA_DIR}/archives"
CURRENT_TEST_DIR="${DATA_DIR}/current"
TEST_HISTORY_FILE="${DATA_DIR}/test_history.json"

# Get next run number
get_next_run_number() {
    if [[ -f "$TEST_HISTORY_FILE" ]]; then
        local total_tests=$(jq -r '.total_tests // 0' "$TEST_HISTORY_FILE")
        echo $(printf "%03d" $((total_tests + 1)))
    else
        echo "001"
    fi
}

# Copy shared memory statistics files to archive
copy_shared_memory_stats() {
    local archive_path="$1"
    local stats_dir="$archive_path/stats"
    mkdir -p "$stats_dir"
    
    # Copy data_loss_stats.json to archive
    if [[ -f "$MEMORY_SHARE_DIR/data_loss_stats.json" ]]; then
        cp "$MEMORY_SHARE_DIR/data_loss_stats.json" "$stats_dir/"
        echo "✅ data_loss_stats.json archived to: $stats_dir/"
    else
        echo "⚠️ data_loss_stats.json file does not exist, skipping archive"
    fi
    
    # Copy other important statistics files
    if [[ -f "${BOTTLENECK_STATUS_FILE:-${MEMORY_SHARE_DIR}/bottleneck_status.json}" ]]; then
        cp "${BOTTLENECK_STATUS_FILE:-${MEMORY_SHARE_DIR}/bottleneck_status.json}" "$stats_dir/"
        echo "✅ bottleneck_status.json archived to: $stats_dir/"
    fi
    
    # Copy qps_status.json to archive (system-level bottleneck detection data)
    if [[ -f "${QPS_STATUS_FILE:-${MEMORY_SHARE_DIR}/qps_status.json}" ]]; then
        cp "${QPS_STATUS_FILE:-${MEMORY_SHARE_DIR}/qps_status.json}" "$stats_dir/"
        echo "✅ qps_status.json archived to: $stats_dir/"
    fi
}

# Auto-detect bottleneck information (development environment optimized version)
auto_detect_bottlenecks() {
    local bottleneck_file="${BOTTLENECK_STATUS_FILE:-${MEMORY_SHARE_DIR}/bottleneck_status.json}"
    
    if [[ -f "$bottleneck_file" ]]; then
        # Validate JSON format
        if ! jq empty "$bottleneck_file" 2>/dev/null; then
            echo "none|none|false"
            return
        fi
        
        local detected=$(jq -r '.bottleneck_detected' "$bottleneck_file" 2>/dev/null || echo "false")
        if [[ "$detected" == "true" ]]; then
            # Directly use new format (no backward compatibility needed)
            local types_array=$(jq -r '.bottleneck_types[]?' "$bottleneck_file" 2>/dev/null)
            local values_array=$(jq -r '.bottleneck_values[]?' "$bottleneck_file" 2>/dev/null)
            
            if [[ -n "$types_array" ]]; then
                local types_csv=$(echo "$types_array" | tr '\n' ',' | sed 's/,$//')
                local values_csv=$(echo "$values_array" | tr '\n' ',' | sed 's/,$//')
                echo "$types_csv|$values_csv|true"
            else
                echo "none|none|false"
            fi
        else
            echo "none|none|false"
        fi
    else
        echo "none|none|false"
    fi
}

# Generate test summary
generate_test_summary() {
    local run_id="$1"
    local benchmark_mode="$2"
    local max_qps="$3"
    local start_time="$4"
    local end_time="$5"
    local initial_qps qps_step duration_per_level
    case "$benchmark_mode" in
        sync_observe)
            initial_qps="0"
            qps_step="0"
            duration_per_level="0"
            ;;
        quick)
            initial_qps="${QUICK_INITIAL_QPS:-1000}"
            qps_step="${QUICK_QPS_STEP:-500}"
            duration_per_level="${QUICK_DURATION:-60}"
            ;;
        intensive)
            initial_qps="${INTENSIVE_INITIAL_QPS:-50000}"
            qps_step="${INTENSIVE_QPS_STEP:-250}"
            duration_per_level="${INTENSIVE_DURATION:-600}"
            ;;
        standard|*)
            initial_qps="${STANDARD_INITIAL_QPS:-2000}"
            qps_step="${STANDARD_QPS_STEP:-500}"
            duration_per_level="${STANDARD_DURATION:-600}"
            ;;
    esac
    
    # Auto-detect bottleneck information
    local bottleneck_info=$(auto_detect_bottlenecks)
    local bottleneck_types=$(echo "$bottleneck_info" | cut -d'|' -f1)
    local bottleneck_values=$(echo "$bottleneck_info" | cut -d'|' -f2)
    local bottleneck_detected=$(echo "$bottleneck_info" | cut -d'|' -f3)
    
    local archive_path="${ARCHIVES_DIR}/${run_id}"
    local summary_file="${archive_path}/test_summary.json"

    if [[ "$benchmark_mode" == "sync_observe" && ( -z "$start_time" || -z "$end_time" ) ]]; then
        local performance_csv="${archive_path}/logs/performance_latest.csv"
        if [[ -f "$performance_csv" ]]; then
            local inferred_times
            inferred_times=$(awk -F',' '
                NR == 1 {
                    for (i = 1; i <= NF; i++) {
                        if ($i == "timestamp") {
                            ts_col = i
                        }
                    }
                    next
                }
                ts_col && $ts_col != "" {
                    if (start == "") {
                        start = $ts_col
                    }
                    end = $ts_col
                }
                END {
                    if (start != "" || end != "") {
                        printf "%s|%s", start, end
                    }
                }
            ' "$performance_csv")
            if [[ -n "$inferred_times" ]]; then
                [[ -z "$start_time" ]] && start_time="${inferred_times%%|*}"
                [[ -z "$end_time" ]] && end_time="${inferred_times##*|}"
            fi
        fi
    fi
    
    # Calculate test duration
    local duration_minutes=0
    if [[ -n "$start_time" && -n "$end_time" ]]; then
        local start_epoch=$(date -d "$start_time" +%s 2>/dev/null || echo 0)
        local end_epoch=$(date -d "$end_time" +%s 2>/dev/null || echo 0)
        if [[ $start_epoch -gt 0 && $end_epoch -gt 0 ]]; then
            duration_minutes=$(( (end_epoch - start_epoch) / 60 ))
        fi
    fi
    
    # Calculate data size
    local logs_mb=$(du -sm "${archive_path}/logs" 2>/dev/null | cut -f1 || echo 0)
    local reports_mb=$(du -sm "${archive_path}/reports" 2>/dev/null | cut -f1 || echo 0)
    local vegeta_mb=$(du -sm "${archive_path}/vegeta_results" 2>/dev/null | cut -f1 || echo 0)
    local total_mb=$((logs_mb + reports_mb + vegeta_mb))
    
    # Generate optimized JSON summary (development environment version)
    local bottleneck_types_json=""
    local bottleneck_values_json=""
    
    if [[ "$bottleneck_detected" == "true" && "$bottleneck_types" != "none" ]]; then
        # Convert to JSON array format
        bottleneck_types_json=$(echo "[$bottleneck_types]" | sed 's/,/","/g' | sed 's/\[/["/' | sed 's/\]/"]/')
        bottleneck_values_json=$(echo "[$bottleneck_values]" | sed 's/,/","/g' | sed 's/\[/["/' | sed 's/\]/"]/')
    else
        bottleneck_types_json="[]"
        bottleneck_values_json="[]"
    fi
    
    cat > "$summary_file" << EOF
{
  "run_id": "$run_id",
  "benchmark_mode": "$benchmark_mode",
  "start_time": "$start_time",
  "end_time": "$end_time",
  "duration_minutes": $duration_minutes,
  "max_successful_qps": $max_qps,
  "bottleneck_detected": $bottleneck_detected,
  "bottleneck_types": $bottleneck_types_json,
  "bottleneck_values": $bottleneck_values_json,
  "bottleneck_summary": "$bottleneck_types",
  "test_parameters": {
    "initial_qps": ${initial_qps},
    "max_qps": ${max_qps},
    "qps_step": ${qps_step},
    "duration_per_level": ${duration_per_level}
  },
  "data_size": {
    "logs_mb": $logs_mb,
    "reports_mb": $reports_mb,
    "vegeta_results_mb": $vegeta_mb,
    "total_mb": $total_mb
  },
  "archived_at": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
    
    echo "✅ Test summary generated: $summary_file"
}

# Update test history index
update_test_history() {
    local run_id="$1"
    local benchmark_mode="$2"
    local max_qps="$3"
    local status="$4"
    
    # If history file does not exist, create initial structure
    if [[ ! -f "$TEST_HISTORY_FILE" ]]; then
        cat > "$TEST_HISTORY_FILE" << EOF
{
  "total_tests": 0,
  "latest_run": "",
  "tests": []
}
EOF
    fi
    
    # Add new test record
    local temp_file=$(mktemp)
    jq --arg run_id "$run_id" \
       --arg benchmark_mode "$benchmark_mode" \
       --argjson max_qps "$max_qps" \
       --arg status "$status" \
       '.total_tests += 1 | 
        .latest_run = $run_id | 
        .tests += [{
          "run_id": $run_id,
          "benchmark_mode": $benchmark_mode,
          "max_qps": $max_qps,
          "status": $status,
          "archived_at": now | strftime("%Y-%m-%d %H:%M:%S")
        }]' "$TEST_HISTORY_FILE" > "$temp_file" && mv "$temp_file" "$TEST_HISTORY_FILE"
    
    echo "✅ Test history updated: $TEST_HISTORY_FILE"
}

# Auto-archive current test
archive_current_test() {
    local benchmark_mode="$1"
    local max_qps="$2"
    local start_time="$3"
    local end_time="$4"
    
    echo "🗂️  Starting to archive current test data..."
    
    # Check if current test directory has data
    if [[ ! -d "$CURRENT_TEST_DIR" ]] || [[ -z "$(ls -A "$CURRENT_TEST_DIR" 2>/dev/null)" ]]; then
        echo "⚠️  Current test directory is empty, no need to archive"
        return 1
    fi
    
    # Generate run ID
    local timestamp=${SESSION_TIMESTAMP}
    local run_number=$(get_next_run_number)
    local run_id="run_${run_number}_${timestamp}"
    
    # Auto-detect bottleneck information for display
    local bottleneck_info=$(auto_detect_bottlenecks)
    local bottleneck_types=$(echo "$bottleneck_info" | cut -d'|' -f1)
    local bottleneck_detected=$(echo "$bottleneck_info" | cut -d'|' -f3)
    
    echo "📋 Archive information:"
    echo "   Run ID: $run_id"
    echo "   Benchmark mode: $benchmark_mode"
    echo "   Max QPS: $max_qps"
    echo "   Bottleneck detected: $bottleneck_detected"
    if [[ "$bottleneck_detected" == "true" ]]; then
        echo "   Bottleneck types: $bottleneck_types"
    fi
    
    # Create archive directory
    local archive_path="${ARCHIVES_DIR}/${run_id}"
    mkdir -p "$archive_path"
    
    # Copy important statistics files from shared memory
    copy_shared_memory_stats "$archive_path"
    
    # Move current test data to archive
    if mv "$CURRENT_TEST_DIR"/* "$archive_path/" 2>/dev/null; then
        echo "✅ Test data moved to archive directory"
    else
        echo "❌ Failed to move test data"
        return 1
    fi
    
    # Generate test summary
    generate_test_summary "$run_id" "$benchmark_mode" "$max_qps" "$start_time" "$end_time"
    
    # Determine test status
    local status="completed_successfully"
    if [[ "$bottleneck_detected" == "true" ]]; then
        status="completed_with_bottleneck"
    fi
    
    # Update test history index
    update_test_history "$run_id" "$benchmark_mode" "$max_qps" "$status"
    
    # Clean up archived shared memory files
    if [[ -n "${MEMORY_SHARE_DIR:-}" ]] && [[ -d "$MEMORY_SHARE_DIR" ]]; then
        echo "🧹 Cleaning up archived shared memory files..."
        rm -f "${BOTTLENECK_STATUS_FILE:-${MEMORY_SHARE_DIR}/bottleneck_status.json}" 2>/dev/null || true
        rm -f "${QPS_STATUS_FILE:-${MEMORY_SHARE_DIR}/qps_status.json}" 2>/dev/null || true
        rm -f "$MEMORY_SHARE_DIR"/data_loss_stats.json 2>/dev/null || true
        rm -f "${EVENT_NOTIFICATION_FILE:-${MEMORY_SHARE_DIR}/event_notification.json}" 2>/dev/null || true
        rm -f "$MEMORY_SHARE_DIR"/*.flag 2>/dev/null || true
        echo "✅ Archived shared memory files cleaned up"
    fi
    
    echo "🎉 Test archiving completed: $run_id"
    echo "📊 Data size: $(du -sh "$archive_path" | cut -f1)"
    
    return 0
}

# List test history
list_test_history() {
    echo "📊 QPS Test History"
    echo "=================="
    
    if [[ -f "$TEST_HISTORY_FILE" ]]; then
        local total_tests=$(jq -r '.total_tests' "$TEST_HISTORY_FILE")
        local latest_run=$(jq -r '.latest_run' "$TEST_HISTORY_FILE")
        
        echo "Total tests: $total_tests"
        echo "Latest test: $latest_run"
        echo ""
        echo "Historical test list:"
        
        jq -r '.tests[] | "🔹 \(.run_id) | Mode: \(.benchmark_mode) | Max QPS: \(.max_qps) | Status: \(.status) | Time: \(.archived_at)"' "$TEST_HISTORY_FILE"
    else
        echo "No test history available"
    fi
}

# Compare test results
compare_tests() {
    local run1="$1"
    local run2="$2"
    
    if [[ -z "$run1" || -z "$run2" ]]; then
        echo "❌ Error: Please provide two test IDs for comparison"
        echo "💡 Usage: $0 --compare <run_id_1> <run_id_2>"
        echo "🔍 Use --list to view available test IDs"
        return 1
    fi
    
    echo "📈 Test comparison: $run1 vs $run2"
    echo "=========================="
    
    local summary1="${ARCHIVES_DIR}/${run1}/test_summary.json"
    local summary2="${ARCHIVES_DIR}/${run2}/test_summary.json"
    
    if [[ ! -f "$summary1" ]]; then
        echo "❌ Error: Summary file for test '$run1' does not exist"
        echo "💡 File path: $summary1"
        echo "🔍 Use --list to view available test IDs"
        return 1
    fi
    
    if [[ ! -f "$summary2" ]]; then
        echo "❌ Error: Summary file for test '$run2' does not exist"
        echo "💡 File path: $summary2"
        echo "🔍 Use --list to view available test IDs"
        return 1
    fi
    
    # Validate JSON file format
    if ! jq empty "$summary1" 2>/dev/null; then
        echo "❌ Error: Summary file format for test '$run1' is invalid"
        echo "💡 File may be corrupted, please check: $summary1"
        return 1
    fi
    
    if ! jq empty "$summary2" 2>/dev/null; then
        echo "❌ Error: Summary file format for test '$run2' is invalid"
        echo "💡 File may be corrupted, please check: $summary2"
        return 1
    fi
    
    echo "📊 Performance comparison:"
    printf "%-30s %-15s %-15s\n" "Metric" "$run1" "$run2"
    echo "------------------------------------------------------------"
    printf "%-30s %-15s %-15s\n" "Max QPS" \
        "$(jq -r '.max_successful_qps' "$summary1")" \
        "$(jq -r '.max_successful_qps' "$summary2")"
    printf "%-30s %-15s %-15s\n" "Duration (minutes)" \
        "$(jq -r '.duration_minutes' "$summary1")" \
        "$(jq -r '.duration_minutes' "$summary2")"
    printf "%-30s %-15s %-15s\n" "Bottleneck type" \
        "$(jq -r '.bottleneck_summary // "none"' "$summary1")" \
        "$(jq -r '.bottleneck_summary // "none"' "$summary2")"
    printf "%-30s %-15s %-15s\n" "Data size (MB)" \
        "$(jq -r '.data_size.total_mb' "$summary1")" \
        "$(jq -r '.data_size.total_mb' "$summary2")"
    
    echo ""
    echo "📅 Time comparison:"
    echo "  $run1: $(jq -r '.start_time' "$summary1") - $(jq -r '.end_time' "$summary1")"
    echo "  $run2: $(jq -r '.start_time' "$summary2") - $(jq -r '.end_time' "$summary2")"
}

# Clean up old test data
cleanup_old_tests() {
    local keep_count=${1:-10}
    
    # Validate keep count parameter
    if ! [[ "$keep_count" =~ ^[0-9]+$ ]] || [[ "$keep_count" -eq 0 ]]; then
        echo "❌ Error: Keep count must be a positive integer, current value: '$keep_count'"
        echo "💡 Example: cleanup_old_tests 5"
        return 1
    fi
    
    echo "🗑️  Cleaning up old test data, keeping the most recent $keep_count tests"
    
    if [[ ! -d "$ARCHIVES_DIR" ]]; then
        echo "ℹ️  Archive directory does not exist, no cleanup needed"
        echo "💡 Directory path: $ARCHIVES_DIR"
        return 0
    fi
    
    # Check directory permissions
    if [[ ! -w "$ARCHIVES_DIR" ]]; then
        echo "❌ Error: No write permission for archive directory"
        echo "💡 Directory path: $ARCHIVES_DIR"
        echo "🔧 Please check directory permissions or run as appropriate user"
        return 1
    fi
    
    # Get all test directories, sorted by time
    local test_dirs=($(ls -1t "$ARCHIVES_DIR" | grep "^run_"))
    local total_tests=${#test_dirs[@]}
    
    if [[ $total_tests -le $keep_count ]]; then
        echo "Current test count ($total_tests) does not exceed keep count ($keep_count), no cleanup needed"
        return 0
    fi
    
    echo "Found $total_tests tests, will delete the oldest $((total_tests - keep_count))"
    
    # Delete old tests exceeding keep count
    for ((i=$keep_count; i<$total_tests; i++)); do
        local old_test="${test_dirs[$i]}"
        local old_path="${ARCHIVES_DIR}/${old_test}"
        local size=$(du -sh "$old_path" | cut -f1)
        
        echo "Deleting: $old_test (size: $size)"
        rm -rf "$old_path"
    done
    
    # Rebuild test history index
    rebuild_test_history
    
    echo "✅ Cleanup completed"
}

# Rebuild test history index
rebuild_test_history() {
    echo "🔄 Rebuilding test history index..."
    
    # Create new history file
    cat > "$TEST_HISTORY_FILE" << EOF
{
  "total_tests": 0,
  "latest_run": "",
  "tests": []
}
EOF
    
    # Scan all tests in archive directory
    if [[ -d "$ARCHIVES_DIR" ]]; then
        local test_dirs=($(ls -1t "$ARCHIVES_DIR" | grep "^run_"))
        
        for test_dir in "${test_dirs[@]}"; do
            local summary_file="${ARCHIVES_DIR}/${test_dir}/test_summary.json"
            
            if [[ -f "$summary_file" ]]; then
                local benchmark_mode=$(jq -r '.benchmark_mode' "$summary_file")
                local max_qps=$(jq -r '.max_successful_qps' "$summary_file")
                local bottleneck=$(jq -r '.bottleneck_detected' "$summary_file")
                local status="completed_successfully"
                
                if [[ "$bottleneck" == "true" ]]; then
                    status="completed_with_bottleneck"
                fi
                
                update_test_history "$test_dir" "$benchmark_mode" "$max_qps" "$status"
            fi
        done
    fi
    
    echo "✅ Test history index rebuild completed"
}

# Display help information
show_help() {
    cat << 'EOF'
📦 Benchmark Test Archiver - Development Environment Optimized Version

Usage:
  $0 <operation> [options]

Operations:
  --archive                    Archive current test data
    --benchmark-mode <mode>    Benchmark mode (required)
                              Supported: quick, standard, intensive, sync_observe
    --max-qps <qps>           Maximum successful QPS (required, positive integer)
    --start-time <time>       Test start time (optional)
                              Format: 'YYYY-MM-DD HH:MM:SS'
    --end-time <time>         Test end time (optional)
                              Format: 'YYYY-MM-DD HH:MM:SS'
    Note: Bottleneck information will be automatically extracted from system detection results

  --list                       List test history

  --compare <run1> <run2>      Compare two test results
                              run1, run2: Test run IDs
                              Use --list to view available test IDs

  --cleanup [--keep <count>]   Clean up old test data
                              count: Number of tests to keep (default: 10)
                              Must be a positive integer

  --rebuild-history           Rebuild test history index

  --help                      Display this help information

Examples:
  # Archive test (basic usage)
  $0 --archive --benchmark-mode standard --max-qps 2500

  # Archive test (complete information)
  $0 --archive --benchmark-mode intensive --max-qps 3500 \
     --start-time "2025-01-01 10:00:00" --end-time "2025-01-01 12:00:00"

  # List historical tests
  $0 --list
  
  # Compare two tests
  $0 --compare run_001_20250101_100000 run_002_20250101_110000
  
  # Clean up old tests, keep the most recent 5
  $0 --cleanup --keep 5

Notes:
  • All time formats use: 'YYYY-MM-DD HH:MM:SS'
  • QPS value must be a positive integer
  • Bottleneck information is automatically detected, no manual specification needed
  • In development environment, error handling is more strict and friendly

Error handling:
  • Parameter validation: Strictly validate format and validity of all parameters
  • Friendly prompts: Provide specific error messages and usage suggestions
  • Quick help: Display relevant usage tips on errors
EOF
}

# Main function
main() {
    case "$1" in
        --archive)
            shift
            local mode="full"
            local max_qps="0"
            local start_time=""
            local end_time=""
            
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --benchmark-mode) 
                        if [[ -z "$2" ]]; then
                            echo "❌ Error: --benchmark-mode parameter value cannot be empty"
                            echo "💡 Supported modes: quick, standard, intensive, sync_observe"
                            exit 1
                        fi
                        if [[ "$2" != "quick" && "$2" != "standard" && "$2" != "intensive" && "$2" != "sync_observe" ]]; then
                            echo "❌ Error: Invalid benchmark mode '$2'"
                            echo "💡 Supported modes: quick, standard, intensive, sync_observe"
                            exit 1
                        fi
                        mode="$2"; shift 2 ;;
                    --max-qps) 
                        if [[ -z "$2" ]]; then
                            echo "❌ Error: --max-qps parameter value cannot be empty"
                            echo "💡 Example: --max-qps 2500"
                            exit 1
                        fi
                        if ! [[ "$2" =~ ^[0-9]+$ ]] || { [[ "$2" -eq 0 ]] && [[ "$mode" != "sync_observe" ]]; }; then
                            echo "❌ Error: --max-qps must be a positive integer (or 0 for sync_observe), current value: '$2'"
                            echo "💡 Example: --max-qps 2500"
                            exit 1
                        fi
                        max_qps="$2"; shift 2 ;;
                    --start-time) 
                        if [[ -z "$2" ]]; then
                            echo "❌ Error: --start-time parameter value cannot be empty"
                            echo "💡 Format: 'YYYY-MM-DD HH:MM:SS'"
                            exit 1
                        fi
                        start_time="$2"; shift 2 ;;
                    --end-time) 
                        if [[ -z "$2" ]]; then
                            echo "❌ Error: --end-time parameter value cannot be empty"
                            echo "💡 Format: 'YYYY-MM-DD HH:MM:SS'"
                            exit 1
                        fi
                        end_time="$2"; shift 2 ;;
                    --help)
                        show_help
                        exit 0 ;;
                    -*) 
                        echo "❌ Error: Unknown parameter '$1'"
                        echo ""
                        echo "💡 Supported parameters:"
                        echo "   --benchmark-mode <mode>  Benchmark mode (quick/standard/intensive/sync_observe)"
                        echo "   --max-qps <qps>         Maximum successful QPS (positive integer)"
                        echo "   --start-time <time>     Test start time"
                        echo "   --end-time <time>       Test end time"
                        echo "   --help                  Display complete help information"
                        echo ""
                        echo "🔍 Use --help to view complete usage instructions"
                        exit 1 ;;
                    *) 
                        echo "❌ Error: Invalid parameter '$1'"
                        echo "💡 Tip: Parameters must start with --"
                        echo "🔍 Use --help to view supported parameters"
                        exit 1 ;;
                esac
            done
            
            # Validate required parameters
            if [[ -z "$mode" ]]; then
                echo "❌ Error: Missing required parameter --benchmark-mode"
                echo "💡 Example: --benchmark-mode standard"
                exit 1
            fi
            
            if [[ -z "$max_qps" ]]; then
                echo "❌ Error: Missing required parameter --max-qps"
                echo "💡 Example: --max-qps 2500"
                exit 1
            fi
            
            archive_current_test "$mode" "$max_qps" "$start_time" "$end_time"
            ;;
        --list)
            list_test_history
            ;;
        --compare)
            if [[ $# -lt 3 ]]; then
                echo "❌ Error: --compare requires two test ID parameters"
                echo "💡 Usage: --compare <run_id1> <run_id2>"
                echo "🔍 Use --list to view available test IDs"
                exit 1
            fi
            local run1="$2"
            local run2="$3"
            if [[ -z "$run1" || -z "$run2" ]]; then
                echo "❌ Error: Test ID cannot be empty"
                echo "💡 Usage: --compare run_001_20250101_120000 run_002_20250101_130000"
                exit 1
            fi
            compare_tests "$run1" "$run2"
            ;;
        --cleanup)
            local keep_count=10  # Default keep 10 tests
            if [[ -n "$2" && "$2" == "--keep" ]]; then
                if [[ -z "$3" ]]; then
                    echo "❌ Error: --keep parameter requires specifying keep count"
                    echo "💡 Usage: --cleanup --keep 5"
                    exit 1
                fi
                if ! [[ "$3" =~ ^[0-9]+$ ]] || [[ "$3" -eq 0 ]]; then
                    echo "❌ Error: Keep count must be a positive integer, current value: '$3'"
                    echo "💡 Usage: --cleanup --keep 5"
                    exit 1
                fi
                keep_count="$3"
            elif [[ -n "$2" ]]; then
                echo "❌ Error: Invalid parameter for --cleanup '$2'"
                echo "💡 Usage: --cleanup [--keep <count>]"
                exit 1
            fi
            cleanup_old_tests "$keep_count"
            ;;
        --rebuild-history)
            rebuild_test_history
            ;;
        --help)
            show_help
            ;;
        "")
            echo "❌ Error: Missing operation parameter"
            echo ""
            echo "💡 Available operations:"
            echo "   --archive                    Archive current test"
            echo "   --list                       List test history"
            echo "   --compare <run1> <run2>      Compare two tests"
            echo "   --cleanup [--keep <count>]   Clean up old tests"
            echo "   --rebuild-history            Rebuild test history"
            echo "   --help                       Display help"
            echo ""
            echo "🔍 Use --help to view detailed instructions"
            exit 1
            ;;
        *)
            echo "❌ Error: Unknown operation '$1'"
            echo ""
            echo "💡 Available operations:"
            echo "   --archive                    Archive current test"
            echo "   --list                       List test history"
            echo "   --compare <run1> <run2>      Compare two tests"
            echo "   --cleanup [--keep <count>]   Clean up old tests"
            echo "   --rebuild-history            Rebuild test history"
            echo "   --help                       Display help"
            echo ""
            echo "🔍 Use --help to view detailed instructions"
            exit 1
            ;;
    esac
}

# If directly executing this script
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
