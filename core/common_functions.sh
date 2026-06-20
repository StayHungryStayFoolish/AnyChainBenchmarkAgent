#!/bin/bash

# =====================================================================
# Blockchain Node Benchmark QPS Shared Function Library
# Contains functions shared across multiple scripts
# =====================================================================

# Load system configuration to get timestamp function
LOCAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${LOCAL_SCRIPT_DIR}/../config/system_config.sh" 2>/dev/null || {
    # If system config cannot be loaded, provide fallback timestamp function
    get_unified_timestamp() {
        date +"%Y-%m-%d %H:%M:%S"
    }
}

# Check Block Height difference
# Buffered data write
buffered_write() {
    local file="$1"
    local data="$2"
    local buffer_file="${file}.buffer"
    local buffer_size="${3:-10}"
    local buffer_count=0
    
    # Check if directory exists
    local dir=$(dirname "$file")
    if [[ ! -d "$dir" ]]; then
        echo "Warning: Directory $dir does not exist, skipping write" >&2
        return 1
    fi
    
    # Safely handle buffer file
    if [[ ! -f "$buffer_file" ]]; then
        if ! echo "0" > "$buffer_file" 2>/dev/null; then
            echo "Warning: Cannot create buffer file $buffer_file, using direct write" >&2
            # Direct write without buffering
            echo "$data" >> "$file" 2>/dev/null || return 1
            return 0
        fi
    fi
    
    # Safely read buffer count
    if ! buffer_count=$(cat "$buffer_file" 2>/dev/null); then
        echo "Warning: Cannot read buffer file $buffer_file, resetting to 0" >&2
        buffer_count=0
    fi
    
    # Verify buffer count is numeric
    if ! [[ "$buffer_count" =~ ^[0-9]+$ ]]; then
        buffer_count=0
    fi
    
    # Safely write data
    if ! echo "$data" >> "$file" 2>/dev/null; then
        echo "Warning: Cannot write to file $file" >&2
        return 1
    fi
    
    # Update buffer count
    buffer_count=$((buffer_count + 1))
    
    # Flush buffer
    if [[ $buffer_count -ge $buffer_size ]]; then
        sync "$file" 2>/dev/null || true
        buffer_count=0
    fi
    
    # Safely update buffer count
    if ! echo "$buffer_count" > "$buffer_file" 2>/dev/null; then
        echo "Warning: Cannot update buffer count for $buffer_file" >&2
    fi
    
    return 0
}

# Get cached Block Height data
current_epoch_ms() {
    local timestamp_ns
    timestamp_ns=$(date +%s%N 2>/dev/null || true)
    if [[ "$timestamp_ns" =~ ^[0-9]{19,}$ ]]; then
        echo "${timestamp_ns:0:${#timestamp_ns}-6}"
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
        return 0
    fi

    echo "$(date +%s)000"
}

get_fresh_block_height_cache() {
    local cache_file="$1"
    local max_age_seconds="${2:-1}"

    [[ -f "$cache_file" ]] || return 1

    local cache_data
    if ! cache_data=$(cat "$cache_file" 2>/dev/null); then
        return 1
    fi

    local timestamp_ms
    timestamp_ms=$(echo "$cache_data" | jq -r '.timestamp_ms // empty' 2>/dev/null) || return 1

    if [[ -z "$timestamp_ms" || "$timestamp_ms" == "null" || ! "$timestamp_ms" =~ ^[0-9]+$ ]]; then
        return 1
    fi

    local current_time_ms
    current_time_ms=$(current_epoch_ms)

    if awk "BEGIN {exit !(($current_time_ms - $timestamp_ms) <= ($max_age_seconds * 1000))}"; then
        echo "$cache_data"
        return 0
    fi

    return 1
}

get_chain_sync_health_field() {
    local field="$1"
    local default_value="$2"
    local blockchain_type
    blockchain_type=$(echo "${BLOCKCHAIN_NODE:-solana}" | tr '[:upper:]' '[:lower:]')
    local chain_file="${LOCAL_SCRIPT_DIR}/../config/chains/${blockchain_type}.json"

    if [[ -f "$chain_file" ]] && command -v jq >/dev/null 2>&1; then
        local value
        value=$(jq -r --arg field "$field" '._meta.sync_health[$field] // empty' "$chain_file" 2>/dev/null || true)
        if [[ -n "$value" && "$value" != "null" ]]; then
            echo "$value"
            return 0
        fi
    fi

    echo "$default_value"
}

get_chain_meta_probe_value() {
    local probe_name="$1"
    local rpc_url="$2"
    local blockchain_type
    blockchain_type=$(echo "${BLOCKCHAIN_NODE:-solana}" | tr '[:upper:]' '[:lower:]')
    local chain_file="${LOCAL_SCRIPT_DIR}/../config/chains/${blockchain_type}.json"

    if [[ -z "$probe_name" || -z "$rpc_url" || ! -f "$chain_file" ]]; then
        return 1
    fi
    command -v jq >/dev/null 2>&1 || return 1
    command -v curl >/dev/null 2>&1 || return 1

    local probe_json
    probe_json=$(jq -c --arg probe "$probe_name" '._meta[$probe] // empty' "$chain_file" 2>/dev/null || true)
    if [[ -z "$probe_json" || "$probe_json" == "null" ]]; then
        return 1
    fi

    local rpc_method params parse_jq request_body response value
    rpc_method=$(echo "$probe_json" | jq -r '.method // .rpc_method // empty' 2>/dev/null)
    params=$(echo "$probe_json" | jq -c '.params // []' 2>/dev/null)
    parse_jq=$(echo "$probe_json" | jq -r '.parse_jq // ".result"' 2>/dev/null)
    if [[ -z "$rpc_method" || "$rpc_method" == "null" || -z "$params" || "$params" == "null" ]]; then
        return 1
    fi

    request_body=$(jq -cn --arg method "$rpc_method" --argjson params "$params" \
        '{jsonrpc:"2.0",id:1,method:$method,params:$params}' 2>/dev/null) || return 1
    response=$(curl -s --max-time 5 -H "Content-Type: application/json" -d "$request_body" "$rpc_url" 2>/dev/null) || return 1
    value=$(echo "$response" | jq -r "$parse_jq" 2>/dev/null) || return 1

    if [[ -z "$value" || "$value" == "null" ]]; then
        return 1
    fi

    echo "$value"
}

read_previous_sync_health_field() {
    local cache_file="$1"
    local field="$2"
    local default_value="$3"

    if [[ -f "$cache_file" ]] && command -v jq >/dev/null 2>&1; then
        local value
        value=$(jq -r --arg field "$field" '.[$field] // empty' "$cache_file" 2>/dev/null || true)
        if [[ -n "$value" && "$value" != "null" ]]; then
            echo "$value"
            return 0
        fi
    fi

    echo "$default_value"
}

get_node_sync_health() {
    local local_rpc_url="$1"
    local mainnet_rpc_url="$2"
    local previous_cache_file="${3:-}"

    local sync_mode
    sync_mode=$(get_chain_sync_health_field "mode" "absolute_gap")
    local lag_unit
    lag_unit=$(get_chain_sync_health_field "threshold_unit" "block")

    local timestamp_ms
    timestamp_ms=$(current_epoch_ms)
    local timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    local local_block_height="N/A"
    local mainnet_block_height="N/A"
    local local_health="0"
    local mainnet_health="1"
    local block_height_diff="N/A"
    local data_loss="0"
    local sync_status="unknown"
    local lag_value="N/A"
    local freshness_gap_seconds="N/A"
    local probe_error="none"

    local_block_height=$(get_block_height "$local_rpc_url")
    local_health=$(check_node_health "$local_rpc_url")

    case "$sync_mode" in
        absolute_gap)
            mainnet_block_height=$(get_block_height "$mainnet_rpc_url")
            mainnet_health=$(check_node_health "$mainnet_rpc_url")

            if [[ "$local_block_height" != "N/A" && "$mainnet_block_height" != "N/A" ]]; then
                block_height_diff=$((mainnet_block_height - local_block_height))
                lag_value="$block_height_diff"
                sync_status="healthy"
                if [[ "$block_height_diff" != "N/A" && $block_height_diff -gt ${BLOCK_HEIGHT_DIFF_THRESHOLD:-50} ]]; then
                    sync_status="behind"
                elif [[ "$block_height_diff" != "N/A" && $block_height_diff -lt -${BLOCK_HEIGHT_DIFF_THRESHOLD:-50} ]]; then
                    sync_status="ahead"
                fi
            elif [[ "$local_block_height" == "N/A" ]]; then
                data_loss="1"
                sync_status="unhealthy"
                probe_error="local_height_unavailable"
            elif [[ "$mainnet_block_height" == "N/A" ]]; then
                sync_status="unknown"
                probe_error="mainnet_height_unavailable"
            fi
            ;;
        conditional_gap)
            if [[ "$local_block_height" != "N/A" ]]; then
                lag_value="$local_block_height"
                block_height_diff="$lag_value"
                sync_status="healthy"
                if [[ "$lag_value" =~ ^-?[0-9]+$ && $lag_value -gt ${BLOCK_HEIGHT_DIFF_THRESHOLD:-50} ]]; then
                    sync_status="behind"
                fi
            else
                data_loss="1"
                sync_status="unhealthy"
                probe_error="conditional_gap_unavailable"
            fi
            ;;
        reported_lag)
            if [[ "$local_block_height" != "N/A" ]]; then
                lag_value="$local_block_height"
                local height_cursor
                height_cursor=$(get_chain_meta_probe_value "local_height_probe" "$local_rpc_url" 2>/dev/null || true)
                if [[ "$height_cursor" =~ ^-?[0-9]+$ ]]; then
                    local_block_height="$height_cursor"
                fi
                sync_status="healthy"
                if [[ "$lag_value" =~ ^-?[0-9]+$ && $lag_value -gt ${BLOCK_HEIGHT_DIFF_THRESHOLD:-50} ]]; then
                    sync_status="behind"
                fi
            else
                data_loss="1"
                sync_status="unhealthy"
                probe_error="reported_lag_unavailable"
            fi
            ;;
        freshness_only|health_only)
            lag_unit="N/A"
            if [[ "$local_block_height" != "N/A" ]]; then
                sync_status="healthy"
                local prev_height prev_timestamp
                prev_height=$(read_previous_sync_health_field "$previous_cache_file" "local_block_height" "")
                prev_timestamp=$(read_previous_sync_health_field "$previous_cache_file" "timestamp_ms" "")
                if [[ -n "$prev_height" && "$prev_height" == "$local_block_height" && "$prev_timestamp" =~ ^[0-9]+$ ]]; then
                    freshness_gap_seconds=$(((timestamp_ms - prev_timestamp) / 1000))
                    if [[ $freshness_gap_seconds -gt ${BLOCK_HEIGHT_TIME_THRESHOLD:-300} ]]; then
                        sync_status="stale"
                    fi
                else
                    freshness_gap_seconds="0"
                fi
            else
                data_loss="1"
                sync_status="unhealthy"
                probe_error="local_freshness_unavailable"
            fi
            ;;
        *)
            sync_mode="health_only"
            lag_unit="N/A"
            if [[ "$local_health" == "1" ]]; then
                sync_status="healthy"
            else
                data_loss="1"
                sync_status="unhealthy"
                probe_error="local_health_unavailable"
            fi
            ;;
    esac

    if [[ "$local_health" == "0" ]]; then
        sync_status="unhealthy"
        data_loss="1"
        if [[ "$probe_error" == "none" ]]; then
            probe_error="local_health_unavailable"
        fi
    fi

    local new_data
    new_data=$(jq -n \
        --arg timestamp_ms "$timestamp_ms" \
        --arg timestamp "$timestamp" \
        --arg local_block_height "$local_block_height" \
        --arg mainnet_block_height "$mainnet_block_height" \
        --arg block_height_diff "$block_height_diff" \
        --arg local_health "$local_health" \
        --arg mainnet_health "$mainnet_health" \
        --arg data_loss "$data_loss" \
        --arg sync_mode "$sync_mode" \
        --arg sync_status "$sync_status" \
        --arg lag_value "$lag_value" \
        --arg lag_unit "$lag_unit" \
        --arg freshness_gap_seconds "$freshness_gap_seconds" \
        --arg probe_error "$probe_error" \
        '{
            timestamp_ms: ($timestamp_ms | tonumber),
            timestamp: $timestamp,
            local_block_height: (if $local_block_height == "N/A" then null else ($local_block_height | tonumber) end),
            mainnet_block_height: (if $mainnet_block_height == "N/A" then null else ($mainnet_block_height | tonumber) end),
            block_height_diff: (if $block_height_diff == "N/A" then null else ($block_height_diff | tonumber) end),
            local_health: $local_health,
            mainnet_health: $mainnet_health,
            data_loss: $data_loss,
            sync_mode: $sync_mode,
            sync_status: $sync_status,
            lag_value: (if $lag_value == "N/A" then null else ($lag_value | tonumber) end),
            lag_unit: (if $lag_unit == "N/A" then null else $lag_unit end),
            freshness_gap_seconds: (if $freshness_gap_seconds == "N/A" then null else ($freshness_gap_seconds | tonumber) end),
            probe_error: (if $probe_error == "none" then null else $probe_error end)
        }')

    echo "$new_data"
    return 0
}

get_cached_block_height_data() {
    local cache_file="$1"
    local max_age_seconds="${2:-1}"  # Default 1 second (adjusted from 3 seconds)
    local local_rpc_url="$3"
    local mainnet_rpc_url="$4"
    
    # Reuse a fresh cache when the monitor tick already produced one.
    local cache_data
    if cache_data=$(get_fresh_block_height_cache "$cache_file" "$max_age_seconds"); then
        echo "$cache_data"
        return 0
    fi

    local new_data
    new_data=$(get_node_sync_health "$local_rpc_url" "$mainnet_rpc_url" "$cache_file")
    echo "$new_data" > "$cache_file"
    
    # Return new data
    echo "$new_data"
    return 0
}

# Clean up old cache data (keep last 5 minutes)
cleanup_block_height_cache() {
    local cache_dir="$1"
    local max_age_minutes="${2:-5}"  # Default 5 minutes
    
    # Find and delete old cache files
    find "$cache_dir" -name "block_height_*.json" -type f -mmin +$max_age_minutes -delete
}

hash_url_for_cache() {
    local value="$1"
    if command -v md5sum >/dev/null 2>&1; then
        echo "$value" | md5sum | cut -d' ' -f1
    elif command -v md5 >/dev/null 2>&1; then
        echo "$value" | md5 -q
    else
        echo "$value" | sed 's/[^A-Za-z0-9_.-]/_/g'
    fi
}

get_node_health_cache_file() {
    local rpc_url="$1"
    local url_hash
    url_hash=$(hash_url_for_cache "$rpc_url")
    local cache_dir="${NODE_HEALTH_CACHE_DIR:-${MEMORY_SHARE_DIR:-/tmp}/node_health_cache}"
    mkdir -p "$cache_dir" 2>/dev/null || true
    echo "${cache_dir}/node_health_${url_hash}.cache"
}

get_block_height_via_adapter() {
    local rpc_url="$1"
    local blockchain_type="$2"
    local health_cache_file="$3"
    local repo_root="${LOCAL_SCRIPT_DIR}/.."
    local cli="${repo_root}/tools/chain_adapters/cli.py"

    if [[ ! -f "$cli" ]] || ! command -v python3 >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
        return 1
    fi

    local probe
    if ! probe=$(BLOCKCHAIN_NODE="$blockchain_type" python3 "$cli" health-probe --chain "$blockchain_type" --rpc-url "$rpc_url" 2>/dev/null); then
        return 1
    fi

    local method url body
    method=$(echo "$probe" | jq -r '.method // "POST"' 2>/dev/null) || return 1
    url=$(echo "$probe" | jq -r '.url // empty' 2>/dev/null) || return 1
    body=$(echo "$probe" | jq -r '.body // ""' 2>/dev/null) || body=""
    [[ -z "$url" ]] && return 1

    local curl_args=(-sS -m "${BLOCK_HEIGHT_CURL_TIMEOUT:-5}" -X "$method")
    local header
    while IFS= read -r header; do
        [[ -z "$header" ]] && continue
        curl_args+=(-H "$header")
    done < <(echo "$probe" | jq -r '(.headers // {}) | to_entries[] | "\(.key): \(.value)"' 2>/dev/null)
    if [[ -n "$body" ]]; then
        curl_args+=(--data "$body")
    fi
    curl_args+=("$url")

    local response height
    if ! response=$(curl "${curl_args[@]}" 2>/dev/null); then
        return 1
    fi
    [[ -z "$response" ]] && return 1

    if ! height=$(BLOCKCHAIN_NODE="$blockchain_type" python3 "$cli" parse-height --chain "$blockchain_type" <<< "$response" 2>/dev/null); then
        return 1
    fi
    if [[ "$height" =~ ^[0-9]+$ ]]; then
        echo "1" > "$health_cache_file"
        echo "$height"
        return 0
    fi
    return 1
}

# Multi-chain block height retrieval function (adapter/template driven)
get_block_height() {
    local rpc_url="$1"
    local blockchain_type
    blockchain_type=$(echo "${BLOCKCHAIN_NODE:-solana}" | tr '[:upper:]' '[:lower:]')
    local health_cache_file
    health_cache_file=$(get_node_health_cache_file "$rpc_url")
    
    # Validate input parameters
    if [[ -z "$rpc_url" ]]; then
        echo "0" > "$health_cache_file"  # Cache numeric: 0=unhealthy
        echo "N/A"
        return 1
    fi

    local adapter_height
    if adapter_height=$(get_block_height_via_adapter "$rpc_url" "$blockchain_type" "$health_cache_file"); then
        if [[ "$adapter_height" =~ ^[0-9]+$ ]]; then
            echo "$adapter_height"
            return 0
        fi
    fi

    echo "❌ Failed to get block height via adapter for blockchain type: $blockchain_type" >&2
    echo "0" > "$health_cache_file"  # Cache numeric: 0=unhealthy
    echo "N/A"
    return 1
}

# Check node health status - cache-based health check
check_node_health() {
    local rpc_url="$1"
    local health_cache_file
    health_cache_file=$(get_node_health_cache_file "$rpc_url")
    
    # Validate input parameters
    if [[ -z "$rpc_url" ]]; then
        echo "0"  # Return numeric: 0=unhealthy
        return 1
    fi
    
    # Check if cache file exists and is recent (within 60 seconds)
    if [[ -f "$health_cache_file" ]]; then
        local cache_age=$(($(date +%s) - $(stat -c %Y "$health_cache_file" 2>/dev/null || echo 0)))
        if [[ $cache_age -lt 60 ]]; then
            # Use cached health status
            cat "$health_cache_file" 2>/dev/null || echo "0"
            return 0
        fi
    fi
    
    # Cache expired or doesn't exist, test connectivity via get_block_height
    local block_height=$(get_block_height "$rpc_url")
    
    if [[ "$block_height" != "N/A" && "$block_height" =~ ^[0-9]+$ ]]; then
        echo "1" > "$health_cache_file"  # Cache numeric: 1=healthy
        echo "1"  # Return numeric: 1=healthy
        return 0
    else
        echo "0" > "$health_cache_file"  # Cache numeric: 0=unhealthy
        echo "0"  # Return numeric: 0=unhealthy
        return 1
    fi
}
