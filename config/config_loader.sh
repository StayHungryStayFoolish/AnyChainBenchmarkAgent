#!/bin/bash
# =====================================================================
# Blockchain Node Benchmark Framework - Unified Configuration Loader
# =====================================================================
# Function: Load all configuration layers in order and perform dynamic configuration detection
# =====================================================================

# =====================================================================
# High-Performance Configuration Caching Mechanism - Prevent repeated loading and JSON parsing
# =====================================================================
# Load configuration directly
echo "🔧 Starting configuration loading..." >&2

# Check if configuration is already loaded, avoid duplicate output
if [[ "${CONFIG_ALREADY_LOADED:-}" == "true" && "${FORCE_CONFIG_RELOAD:-}" != "true" ]]; then
    return 0
fi

# Get configuration directory
CONFIG_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Load configuration layers in order
echo "🔧 Loading layered configuration..." >&2

# 1. Load user configuration layer
if [[ -f "${CONFIG_DIR}/user_config.sh" ]]; then
    source "${CONFIG_DIR}/user_config.sh"
    echo "✅ User configuration layer loaded" >&2
else
    echo "❌ User configuration layer file does not exist: ${CONFIG_DIR}/user_config.sh" >&2
    exit 1
fi

# User config should remain declarative. Provider-specific normalization that
# mutates derived disk baselines lives in a separate post-processing layer.
if [[ -f "${CONFIG_DIR}/provider_disk_config.sh" ]]; then
    source "${CONFIG_DIR}/provider_disk_config.sh"
    echo "✅ Provider disk configuration post-processing loaded" >&2
fi

# Canonical chain names match config/chains/<name>.json file names.
LOCAL_RPC_URL="${LOCAL_RPC_URL:-http://localhost:8899}"
MAINNET_RPC_URL="${MAINNET_RPC_URL:-}"
BLOCKCHAIN_NODE="${BLOCKCHAIN_NODE:-solana}"
BLOCKCHAIN_NODE=$(printf '%s' "$BLOCKCHAIN_NODE" | tr '[:upper:]' '[:lower:]')
RPC_MODE="${RPC_MODE:-single}"
ACCOUNT_COUNT="${ACCOUNT_COUNT:-1000}"
ACCOUNT_MAX_SIGNATURES="${ACCOUNT_MAX_SIGNATURES:-50000}"
ACCOUNT_TX_BATCH_SIZE="${ACCOUNT_TX_BATCH_SIZE:-100}"
ACCOUNT_SEMAPHORE_LIMIT="${ACCOUNT_SEMAPHORE_LIMIT:-10}"

# 2. Load system configuration layer
if [[ -f "${CONFIG_DIR}/system_config.sh" ]]; then
    source "${CONFIG_DIR}/system_config.sh"
    echo "✅ System configuration layer loaded" >&2
else
    echo "❌ System configuration layer file does not exist: ${CONFIG_DIR}/system_config.sh" >&2
    exit 1
fi

# 3. Load internal configuration layer
if [[ -f "${CONFIG_DIR}/internal_config.sh" ]]; then
    source "${CONFIG_DIR}/internal_config.sh"
    echo "✅ Internal configuration layer loaded" >&2
else
    echo "❌ Internal configuration layer file does not exist: ${CONFIG_DIR}/internal_config.sh" >&2
    exit 1
fi

# 4. Load deployment-mode detector
# Detects runtime environment (VM / Docker / K8s), orthogonal to
# DEPLOYMENT_PLATFORM (cloud provider). Together they form (platform, mode)
# matrix used by cloud variants and runtime_paths.sh.
if [[ -f "${CONFIG_DIR}/deployment_mode_detector.sh" ]]; then
    source "${CONFIG_DIR}/deployment_mode_detector.sh"
    echo "✅ Deployment mode detector loaded" >&2
else
    echo "⚠️  Deployment mode detector not found: ${CONFIG_DIR}/deployment_mode_detector.sh — falling back to vm_bare assumption" >&2
    DEPLOYMENT_MODE="vm_bare"
    DEPLOYMENT_MODE_DETECTED=true
    DEPLOYMENT_MODE_SOURCE="missing_detector_fallback"
    export DEPLOYMENT_MODE DEPLOYMENT_MODE_DETECTED DEPLOYMENT_MODE_SOURCE
fi

# 5. Load runtime host path resolver (HOST_PROC / HOST_SYS / cgroup paths)
# Depends on DEPLOYMENT_MODE being set (above).
if [[ -f "${CONFIG_DIR}/runtime_paths.sh" ]]; then
    source "${CONFIG_DIR}/runtime_paths.sh"
    echo "✅ Runtime path resolver loaded" >&2
else
    echo "⚠️  Runtime path resolver not found: ${CONFIG_DIR}/runtime_paths.sh — using local /proc /sys" >&2
    HOST_PROC="${HOST_PROC:-/proc}"
    HOST_SYS="${HOST_SYS:-/sys}"
    HOST_ROOT="${HOST_ROOT:-/}"
    export HOST_PROC HOST_SYS HOST_ROOT
fi

# =====================================================================
# Dynamic Configuration Detection and Calculation
# =====================================================================

# ----- Automatically Calculated Network Configuration -----
# Automatically convert to Mbps (for internal calculation, users do not need to modify)
NETWORK_MAX_BANDWIDTH_MBPS=$((NETWORK_MAX_BANDWIDTH_GBPS * 1000))

# ----- Deployment Platform Detection Function -----
# Automatically detect deployment platform and adjust ENA monitoring configuration
detect_deployment_platform() {
    if [[ "$DEPLOYMENT_PLATFORM" == "auto" ]]; then
        echo "🔍 Auto-detecting deployment platform..." >&2

        # Delegate to config/cloud_provider.sh for aws/gcp/other detection.
        # Detection validates metadata response bodies, checks GCP first, and uses
        # AWS IMDSv2 only. This avoids classifying HTML proxy responses as AWS.
        local _cp_sh="${CONFIG_DIR}/cloud_provider.sh"
        if [[ -f "$_cp_sh" ]]; then
            # Force re-detection by clearing variables that cloud_provider.sh may reuse.
            unset CLOUD_PROVIDER NIC_DRIVER CLOUD_PROVIDER_VARIANT
            # shellcheck source=/dev/null
            source "$_cp_sh"
            DEPLOYMENT_PLATFORM="${CLOUD_PROVIDER:-other}"
        else
            echo "⚠️  cloud_provider.sh not found at $_cp_sh, falling back to deployment platform 'other'" >&2
            DEPLOYMENT_PLATFORM="other"
        fi

        case "$DEPLOYMENT_PLATFORM" in
            aws)
                ENA_MONITOR_ENABLED=true
                echo "✅ AWS environment detected, ENA monitoring enabled" >&2
                ;;
            gcp)
                ENA_MONITOR_ENABLED=false
                echo "✅ GCP environment detected, ENA monitoring disabled (use gvnic monitor instead)" >&2
                ;;
            *)
                DEPLOYMENT_PLATFORM="other"
                ENA_MONITOR_ENABLED=false
                echo "ℹ️  Non-AWS/GCP environment detected (IDC/other cloud), ENA monitoring disabled" >&2
                ;;
        esac
    else
        echo "🔧 Using manually configured deployment platform: $DEPLOYMENT_PLATFORM" >&2
        case "$DEPLOYMENT_PLATFORM" in
            "aws")
                ENA_MONITOR_ENABLED=true
                echo "✅ AWS environment, ENA monitoring enabled" >&2
                ;;
            "gcp")
                ENA_MONITOR_ENABLED=false
                echo "✅ GCP environment, ENA monitoring disabled (use gvnic monitor instead)" >&2
                ;;
            "other"|"idc")
                ENA_MONITOR_ENABLED=false
                echo "ℹ️  Non-AWS environment, ENA monitoring disabled" >&2
                ;;
            *)
                echo "⚠️  Unknown deployment platform: $DEPLOYMENT_PLATFORM, ENA monitoring disabled" >&2
                ENA_MONITOR_ENABLED=false
                ;;
        esac
    fi
    
    # Output final configuration
    echo "📊 Deployment platform configuration:" >&2
    echo "   Platform type: $DEPLOYMENT_PLATFORM" >&2
    echo "   ENA monitoring: $ENA_MONITOR_ENABLED" >&2
    
    # Mark platform detection as completed and export to subprocesses
    DEPLOYMENT_PLATFORM_DETECTED=true
}

load_provider_contract_for_platform() {
    local _cp_sh="${CONFIG_DIR}/cloud_provider.sh"
    if [[ ! -f "$_cp_sh" ]]; then
        return 0
    fi

    case "${DEPLOYMENT_PLATFORM:-other}" in
        aws|gcp) CLOUD_PROVIDER="$DEPLOYMENT_PLATFORM" ;;
        *) CLOUD_PROVIDER="other" ;;
    esac
    export CLOUD_PROVIDER

    # shellcheck source=/dev/null
    source "$_cp_sh"
}

# ----- Network Interface Detection Function -----
# Detect the primary network interface in a provider-neutral way.
detect_network_interface() {
    if [[ -n "${NETWORK_INTERFACE:-}" ]]; then
        return 0
    fi

    local interface=""
    if command -v ip >/dev/null 2>&1; then
        interface=$(ip route 2>/dev/null | awk '/^default/ {print $5; exit}')
        if [[ -z "$interface" ]]; then
            interface=$(ip -o link show up 2>/dev/null \
                | awk -F': ' '$2 != "lo" {print $2; exit}' \
                | cut -d@ -f1)
        fi
    elif command -v route >/dev/null 2>&1; then
        interface=$(route get default 2>/dev/null | awk '/interface:/ {print $2; exit}')
    elif command -v netstat >/dev/null 2>&1; then
        interface=$(netstat -rn 2>/dev/null | awk '/^default/ {print $6; exit}')
    fi

    if [[ -z "$interface" ]]; then
        interface="eth0"
    fi

    NETWORK_INTERFACE="$interface"
}

# ----- Path Detection and Configuration Function -----
ACCOUNT_OUTPUT_FILE="active_accounts.txt"                             # Output filename

# Detect deployment environment and set paths
detect_deployment_paths() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local framework_dir="$(dirname "$script_dir")"
    local deployment_dir="$(dirname "$framework_dir")"
    
    echo "   Framework directory: $framework_dir" >&2
    echo "   Deployment directory: $deployment_dir" >&2
    
    # Set memory sharing directory (independent of data directory by default).
    # Linux production default uses system tmpfs. Optional observability
    # sidecars may override MEMORY_SHARE_DIR to a shared mounted directory.
    MEMORY_SHARE_DIR="${MEMORY_SHARE_DIR:-/dev/shm/blockchain-node-benchmark}"
    echo "🐧 Linux production environment" >&2
    
    # Standardized path configuration
    BASE_FRAMEWORK_DIR="$framework_dir"
    BASE_DATA_DIR="${BLOCKCHAIN_BENCHMARK_DATA_DIR:-${deployment_dir}/blockchain-node-benchmark-result}"
    
    # Validate and fix path settings
    if [[ -z "$BASE_DATA_DIR" || "$BASE_DATA_DIR" == "/blockchain-node-benchmark-result" ]]; then
        echo "⚠️ Data directory path abnormal, using default path" >&2
        BASE_DATA_DIR="${HOME}/blockchain-node-benchmark-result"
    fi

    echo "   Data directory: $BASE_DATA_DIR" >&2
    
    # Support environment variable override
    if [[ -n "${BLOCKCHAIN_BENCHMARK_DATA_DIR:-}" ]]; then
        echo "   (Using environment variable: BLOCKCHAIN_BENCHMARK_DATA_DIR)" >&2
    fi
    
    # Set directory structure - based on new standardized paths
    # Main data directory (QPS test exclusive)
    DATA_DIR="${BASE_DATA_DIR}"
    # Current test data directory
    CURRENT_TEST_DIR="${DATA_DIR}/current"
    # Log directory (performance monitoring data)
    LOGS_DIR="${CURRENT_TEST_DIR}/logs"
    # Report directory (analysis reports and charts)
    REPORTS_DIR="${CURRENT_TEST_DIR}/reports"
    # Vegeta results directory (stress test raw data)
    VEGETA_RESULTS_DIR="${CURRENT_TEST_DIR}/vegeta_results"
    # Temporary file directory (runtime temporary data)
    TMP_DIR="${CURRENT_TEST_DIR}/tmp"
    # Archive directory (historical test data)
    ARCHIVES_DIR="${DATA_DIR}/archives"
    # Error handling and log directories
    ERROR_LOG_DIR="${CURRENT_TEST_DIR}/${ERROR_LOG_SUBDIR}"
    PYTHON_ERROR_LOG_DIR="${CURRENT_TEST_DIR}/${PYTHON_ERROR_LOG_SUBDIR}"
    
    # Generate unified session timestamp (ensure all processes use the same timestamp)
    if [[ -z "${SESSION_TIMESTAMP:-}" ]]; then
        SESSION_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        export SESSION_TIMESTAMP
    fi
    
    # Set dynamic path variables (using unified session timestamp)
    BLOCK_HEIGHT_CACHE_FILE="${BLOCK_HEIGHT_CACHE_FILE:-${MEMORY_SHARE_DIR}/block_height_monitor_cache.json}"
    BLOCK_HEIGHT_DATA_FILE="${BLOCK_HEIGHT_DATA_FILE:-${LOGS_DIR}/block_height_monitor_${SESSION_TIMESTAMP}.csv}"
    ACCOUNTS_OUTPUT_FILE="${ACCOUNTS_OUTPUT_FILE:-${TMP_DIR}/${ACCOUNT_OUTPUT_FILE}}"
    SINGLE_METHOD_TARGETS_FILE="${SINGLE_METHOD_TARGETS_FILE:-${TMP_DIR}/targets_single.json}"
    MIXED_METHOD_TARGETS_FILE="${MIXED_METHOD_TARGETS_FILE:-${TMP_DIR}/targets_mixed.json}"
    QPS_STATUS_FILE="${QPS_STATUS_FILE:-${MEMORY_SHARE_DIR}/qps_status.json}"
    BOTTLENECK_STATUS_FILE="${BOTTLENECK_STATUS_FILE:-${MEMORY_SHARE_DIR}/bottleneck_status.json}"
    BOTTLENECK_COUNTERS_FILE="${BOTTLENECK_COUNTERS_FILE:-${MEMORY_SHARE_DIR}/bottleneck_counters.json}"
    NODE_HEALTH_CACHE_DIR="${NODE_HEALTH_CACHE_DIR:-${MEMORY_SHARE_DIR}/node_health_cache}"
    LATEST_METRICS_FILE="${LATEST_METRICS_FILE:-${MEMORY_SHARE_DIR}/latest_metrics.json}"
    UNIFIED_METRICS_FILE="${UNIFIED_METRICS_FILE:-${MEMORY_SHARE_DIR}/unified_metrics.json}"
    UNIFIED_EVENTS_FILE="${UNIFIED_EVENTS_FILE:-${MEMORY_SHARE_DIR}/unified_events.json}"
    EVENT_MANAGER_LOCK_FILE="${EVENT_MANAGER_LOCK_FILE:-${MEMORY_SHARE_DIR}/event_manager.lock}"
    EVENT_NOTIFICATION_FILE="${EVENT_NOTIFICATION_FILE:-${MEMORY_SHARE_DIR}/event_notification.json}"
    TEST_SESSION_DIR="${TEST_SESSION_DIR:-${TMP_DIR}/session_${SESSION_TIMESTAMP}}"

    UNIFIED_LOG="${UNIFIED_LOG:-${LOGS_DIR}/performance_${SESSION_TIMESTAMP}.csv}"
    PERFORMANCE_LATEST_CSV="${PERFORMANCE_LATEST_CSV:-${LOGS_DIR}/performance_latest.csv}"
    PROXY_METHOD_CSV="${PROXY_METHOD_CSV:-${LOGS_DIR}/proxy_method.csv}"
    PROXY_SELF_CSV="${PROXY_SELF_CSV:-${LOGS_DIR}/proxy_self.csv}"
    RPC_PROXY_LOG="${RPC_PROXY_LOG:-${LOGS_DIR}/rpc_proxy.log}"
    NETWORK_CSV="${NETWORK_CSV:-${LOGS_DIR}/network_${SESSION_TIMESTAMP}.csv}"
    NETWORK_PID_FILE="${NETWORK_PID_FILE:-${TMP_DIR}/network_monitor.pid}"
    
    # Set monitoring overhead optimization related log file paths (using unified timestamp)
    MONITORING_OVERHEAD_LOG="${LOGS_DIR}/monitoring_overhead_${SESSION_TIMESTAMP}.csv"
    PERFORMANCE_LOG="${LOGS_DIR}/monitoring_performance_${SESSION_TIMESTAMP}.log"
    ERROR_LOG="${LOGS_DIR}/monitoring_errors_${SESSION_TIMESTAMP}.log"
    
    # Temporary file pattern (for cleanup)
    TEMP_FILE_PATTERN="${TMP_DIR}/${TEMP_FILE_PREFIX}-*"
    
    # Output final configuration
    echo "📋 Path configuration completed:" >&2
    echo "   Framework directory: $BASE_FRAMEWORK_DIR" >&2
    echo "   Data directory: $BASE_DATA_DIR" >&2
    echo "   Memory sharing: $MEMORY_SHARE_DIR" >&2
    
    # Mark path detection as completed and export to subprocesses
    DEPLOYMENT_PATHS_DETECTED=true
    export DEPLOYMENT_PATHS_DETECTED
}

# ----- Directory Creation Function -----
# Safely create directories function
create_directories_safely() {
    local dirs=("$@")
    local created_dirs=()
    local failed_dirs=()
    
    echo "🔧 Creating necessary directories..." >&2
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            if mkdir -p "$dir" 2>/dev/null; then
                echo "✅ Created directory: $dir" >&2
                created_dirs+=("$dir")
                chmod 755 "$dir" 2>/dev/null || true
            else
                echo "❌ Unable to create directory: $dir" >&2
                failed_dirs+=("$dir")
            fi
        else
            echo "✅ Directory already exists: $dir" >&2
        fi
    done
    
    # Mark directory creation as completed and export to subprocesses
    DIRECTORIES_CREATED=true
    export DIRECTORIES_CREATED
    
    # Return result summary
    if [[ ${#failed_dirs[@]} -gt 0 ]]; then
        echo "⚠️  Some directories failed to create: ${failed_dirs[*]}" >&2
        return 1
    else
        echo "✅ All directories created successfully" >&2
        return 0
    fi
}

# =====================================================================
# Execute Dynamic Configuration Detection
# =====================================================================

# Execute deployment platform detection (cloud provider: gcp / aws / other)
detect_deployment_platform

load_provider_contract_for_platform

# Provider-owned NIC allowance fields. AWS returns ENA counters; GCP/other
# return empty strings because their network collectors use different signals.
if declare -F get_nic_allowance_fields >/dev/null 2>&1; then
    ENA_ALLOWANCE_FIELDS_STR="$(get_nic_allowance_fields)"
else
    ENA_ALLOWANCE_FIELDS_STR=""
fi
export ENA_ALLOWANCE_FIELDS_STR

# Execute deployment mode detection (runtime: VM / Docker / K8s)
# Orthogonal axis to deployment_platform; together → (platform, mode) matrix.
if declare -F detect_deployment_mode >/dev/null 2>&1; then
    detect_deployment_mode
fi

# Resolve runtime host paths (HOST_PROC / HOST_SYS / cgroup paths)
# Must run after detect_deployment_mode (depends on DEPLOYMENT_MODE).
if declare -F resolve_runtime_paths >/dev/null 2>&1; then
    resolve_runtime_paths
fi

# Execute network interface detection
detect_network_interface

# Execute path detection and configuration
detect_deployment_paths

# Create necessary directories
create_directories_safely "$DATA_DIR" "$CURRENT_TEST_DIR" "$LOGS_DIR" "$REPORTS_DIR" "$VEGETA_RESULTS_DIR" "$TMP_DIR" "$ARCHIVES_DIR" "$ERROR_LOG_DIR" "$PYTHON_ERROR_LOG_DIR" "$MEMORY_SHARE_DIR" "$NODE_HEALTH_CACHE_DIR"

# =====================================================================
# Configure Blockchain Node & On-chain Active Addresses
# =====================================================================
# User configuration variables have been moved to the beginning of the file

# =====================================================================
# Unified Blockchain Configuration - Integrate complete configuration for all 36 blockchains
# =====================================================================
# ----- Multi-chain Mainnet Endpoint Dynamic Configuration -----
resolve_mainnet_rpc_url_from_template() {
    local chain_name="$1"
    local chain_file="${CONFIG_DIR}/chains/${chain_name}.json"
    local family mixed url=""

    if [[ ! -f "$chain_file" ]] || ! command -v jq >/dev/null 2>&1; then
        return 1
    fi

    family=$(jq -r '._meta.adapter_family // ""' "$chain_file" 2>/dev/null)
    mixed=$(jq -r '.rpc_methods.mixed // ""' "$chain_file" 2>/dev/null)

    case "$family" in
        tendermint)
            if [[ "$mixed" == *"eth_blockNumber"* ]]; then
                url=$(jq -r '._meta.evm_rpc_url // empty' "$chain_file" 2>/dev/null)
            fi
            if [[ -z "$url" && "$mixed" == *"/cosmos/base/tendermint/v1beta1/blocks/latest"* ]]; then
                url=$(jq -r '._meta.rest_url // empty' "$chain_file" 2>/dev/null)
            fi
            ;;
        hedera_dual)
            url=$(jq -r '._meta.mirror_url // empty' "$chain_file" 2>/dev/null)
            ;;
        rest)
            if [[ "$chain_name" == "ton" ]]; then
                url=$(jq -r '._meta.rest_paths.lookupBlock.base_url // empty' "$chain_file" 2>/dev/null)
            fi
            ;;
    esac

    if [[ -z "$url" ]]; then
        url=$(jq -r '._meta.original_public_endpoints[0].url // empty' "$chain_file" 2>/dev/null)
    fi
    if [[ -z "$url" ]]; then
        url=$(jq -r '._meta.rest_url // ._meta.json_rpc_url // ._meta.evm_rpc_url // empty' "$chain_file" 2>/dev/null)
    fi

    [[ -n "$url" ]] && echo "$url"
}

job_local_chain_override_matches() {
    local blockchain_node="${1:-${BLOCKCHAIN_NODE:-}}"
    local override_file="${CHAIN_CONFIG_OVERRIDE_FILE:-}"
    local blockchain_node_lower
    blockchain_node_lower=$(echo "$blockchain_node" | tr '[:upper:]' '[:lower:]')
    [[ -n "$blockchain_node_lower" && -n "$override_file" && -f "$override_file" ]] || return 1
    jq -e --arg chain "$blockchain_node_lower" '
        type == "object"
        and (._meta.job_local_override == true)
        and (.chain_type | type == "string")
        and ((.chain_type | ascii_downcase) == $chain)
        and (.rpc_methods | type == "object")
    ' "$override_file" >/dev/null 2>&1
}

resolve_active_chain_template_file() {
    local blockchain_node="${1:-${BLOCKCHAIN_NODE:-}}"
    local blockchain_node_lower
    blockchain_node_lower=$(echo "$blockchain_node" | tr '[:upper:]' '[:lower:]')

    if job_local_chain_override_matches "$blockchain_node_lower"; then
        printf '%s\n' "$CHAIN_CONFIG_OVERRIDE_FILE"
        return 0
    fi

    local chains_dir="${CONFIG_LOADER_DIR:-$(dirname "${BASH_SOURCE[0]}")}/chains"
    local canonical_file="$chains_dir/${blockchain_node_lower}.json"
    if [[ -f "$canonical_file" ]]; then
        printf '%s\n' "$canonical_file"
        return 0
    fi
    return 1
}

case "${MAINNET_RPC_URL_DISABLED:-false}" in
    true|TRUE|True|1|yes|YES|Yes)
        MAINNET_RPC_URL_DISABLED="true"
        MAINNET_RPC_URL=""
        ;;
    *)
        MAINNET_RPC_URL="${MAINNET_RPC_URL:-$(resolve_mainnet_rpc_url_from_template "$BLOCKCHAIN_NODE" || true)}"
        ;;
esac
if [[ -z "$MAINNET_RPC_URL" && "${MAINNET_RPC_URL_DISABLED:-false}" != "true" ]]; then
    if job_local_chain_override_matches "$BLOCKCHAIN_NODE"; then
        echo "ℹ️ Job-local chain override has no default MAINNET_RPC_URL; comparison remains disabled unless explicitly configured" >&2
    else
        echo "⚠️ Warning: Unknown blockchain type '${BLOCKCHAIN_NODE}', using default Solana endpoint" >&2
        MAINNET_RPC_URL="https://api.mainnet-beta.solana.com"
    fi
fi


# =====================================================================
# Automatic Configuration Generation Functions
# =====================================================================

# Validate BLOCKCHAIN_NODE value validity
validate_blockchain_node() {
    local blockchain_node="$1"
    local blockchain_node_lower
    blockchain_node_lower=$(echo "$blockchain_node" | tr '[:upper:]' '[:lower:]')

    # Supported chains are discovered from config/chains/*.json instead of a
    # hardcoded list. Each chain template JSON is an authoritative registration.
    local chains_dir="${CONFIG_LOADER_DIR:-$(dirname "${BASH_SOURCE[0]}")}/chains"
    if [[ ! -d "$chains_dir" ]]; then
        echo "❌ Error: chain template directory not found: $chains_dir" >&2
        return 1
    fi

    local target_file="$chains_dir/${blockchain_node_lower}.json"
    if [[ -f "$target_file" ]]; then
        return 0  # Valid
    fi
    if job_local_chain_override_matches "$blockchain_node_lower"; then
        return 0
    fi

    if [[ -n "${CHAIN_CONFIG_OVERRIDE_FILE:-}" ]]; then
        echo "❌ Error: Job-local chain override is malformed or does not match BLOCKCHAIN_NODE '$blockchain_node'" >&2
        echo "   Override: ${CHAIN_CONFIG_OVERRIDE_FILE}" >&2
    fi

    # Invalid blockchain type — discover known chains for diagnostic output
    local known_chains=()
    while IFS= read -r f; do
        known_chains+=("$(basename "$f" .json)")
    done < <(find "$chains_dir" -maxdepth 1 -name '*.json' -type f | sort)

    echo "❌ Error: Unsupported blockchain type '$blockchain_node'" >&2
    echo "   No template at $target_file" >&2
    echo "📋 Supported blockchain types (${#known_chains[@]} discovered):" >&2
    printf "   - %s\n" "${known_chains[@]}" >&2
    echo "💡 Tip: Please check the value of BLOCKCHAIN_NODE environment variable" >&2
    echo "💡 Tip: To add a new chain, create config/chains/<name>.json" >&2
    return 1  # Invalid
}

# Configuration consistency validation function
validate_config_consistency() {
    local blockchain_node_lower
    blockchain_node_lower=$(echo "${BLOCKCHAIN_NODE:-solana}" | tr '[:upper:]' '[:lower:]')
    local rpc_mode_lower
    rpc_mode_lower=$(echo "${RPC_MODE:-single}" | tr '[:upper:]' '[:lower:]')

    # Validate consistency between CHAIN_CONFIG and CURRENT_RPC_METHODS_STRING
    if [[ -n "$CHAIN_CONFIG" && "$CHAIN_CONFIG" != "null" ]]; then
        local expected_method
        expected_method=$(echo "$CHAIN_CONFIG" | jq -r ".rpc_methods.\"$rpc_mode_lower\"")

        if [[ -n "$expected_method" && "$expected_method" != "null" ]]; then
            if [[ "$CURRENT_RPC_METHODS_STRING" != "$expected_method" ]]; then
                echo "⚠️ Configuration inconsistency detected: Expected '$expected_method', Actual '$CURRENT_RPC_METHODS_STRING'" >&2
                echo "🔧 Auto-fixing configuration inconsistency..." >&2
                CURRENT_RPC_METHODS_STRING="$expected_method"

                # Update cache
                local rpc_cache_var_name="CACHED_RPC_METHODS_${blockchain_node_lower//-/_}_${rpc_mode_lower}"
                export "$rpc_cache_var_name"="$CURRENT_RPC_METHODS_STRING"

                echo "✅ Configuration consistency fixed" >&2
            fi
        fi
    fi
}

# Automatically generate configuration based on BLOCKCHAIN_NODE
generate_auto_config() {
    # Clear all configuration cache, ensure clean environment, avoid conflicts between different blockchain configurations
    clear_config_cache
    
    local blockchain_node="${BLOCKCHAIN_NODE:-solana}"
    local blockchain_node_lower
    # Validate BLOCKCHAIN_NODE value
    if ! validate_blockchain_node "$blockchain_node"; then
        # Configuration error, exit directly
        exit 1
    fi
    blockchain_node_lower=$(echo "$blockchain_node" | tr '[:upper:]' '[:lower:]')
    if ! ACTIVE_CHAIN_TEMPLATE_FILE="$(resolve_active_chain_template_file "$blockchain_node_lower")"; then
        echo "❌ Error: No validated active chain template for '$blockchain_node_lower'" >&2
        exit 1
    fi
    export ACTIVE_CHAIN_TEMPLATE_FILE
    echo "🎯 Starting automatic configuration generation..." >&2
    echo "   BLOCKCHAIN_NODE original value: ${BLOCKCHAIN_NODE}" >&2
    echo "   Target blockchain: $blockchain_node_lower" >&2
    
    # Performance optimization: Use cached JSON parsing results
    # NOTE: bash variable names disallow '-', so chain names like
    # avalanche-c / cosmos-hub / zksync-era must have '-' normalized to '_'
    # for the cache var name only. The on-disk file name stays as-is.
    local blockchain_node_var_safe="${blockchain_node_lower//-/_}"
    local cache_var_name="CACHED_CHAIN_CONFIG_${blockchain_node_var_safe}"
    local cached_config="${!cache_var_name:-}"
    
    if [[ -n "$cached_config" ]]; then
        # Use cached configuration
        CHAIN_CONFIG="$cached_config"

    else
        # Read the job-local Agent override first when present. This lets an
        # Agent-confirmed workload change affect target generation without
        # mutating the canonical config/chains/<name>.json template.
        local chains_dir="${CONFIG_LOADER_DIR:-$(dirname "${BASH_SOURCE[0]}")}/chains"
        local chain_file="$chains_dir/${blockchain_node_lower}.json"
        local override_file="${CHAIN_CONFIG_OVERRIDE_FILE:-}"
        if [[ -n "$override_file" && -f "$override_file" ]]; then
            CHAIN_CONFIG=$(jq -c 'del(._meta)' "$override_file")
            echo "   Using Agent chain config override: $override_file" >&2
        elif [[ ! -f "$chain_file" ]]; then
            CHAIN_CONFIG=""
        else
            # The .json file is the authoritative chain template; _meta field
            # is stripped by jq so downstream consumers see the same shape as
            # before.
            CHAIN_CONFIG=$(jq -c 'del(._meta)' "$chain_file")
        fi
        # Cache parsing result
        if [[ "$CHAIN_CONFIG" != "null" && -n "$CHAIN_CONFIG" ]]; then
            export "$cache_var_name"="$CHAIN_CONFIG"
        fi
    fi
    
    # Validate if configuration loaded correctly
    if [[ "$CHAIN_CONFIG" == "null" || -z "$CHAIN_CONFIG" ]]; then
        echo "❌ Error: Unable to load configuration for $blockchain_node_lower" >&2
        echo "   Expected file: config/chains/${blockchain_node_lower}.json" >&2
        echo "   This indicates missing or malformed chain template" >&2
        exit 1
    fi
    
    # Get RPC methods from CHAIN_CONFIG - Fix caching logic
    local rpc_mode_lower
    rpc_mode_lower=$(echo "${RPC_MODE:-single}" | tr '[:upper:]' '[:lower:]')
    
    # Performance optimization: Use cached RPC method parsing results
    local rpc_cache_var_name="CACHED_RPC_METHODS_${blockchain_node_lower//-/_}_${rpc_mode_lower}"
    local cached_rpc_methods="${!rpc_cache_var_name:-}"
    
    if [[ -n "$cached_rpc_methods" ]]; then
        # Use cached RPC methods
        CURRENT_RPC_METHODS_STRING="$cached_rpc_methods"
    else
        # Recalculate and cache
        CURRENT_RPC_METHODS_STRING=$(echo "$CHAIN_CONFIG" | jq -r ".rpc_methods.\"$rpc_mode_lower\"")
        # Cache RPC method parsing result
        if [[ "$CURRENT_RPC_METHODS_STRING" != "null" && -n "$CURRENT_RPC_METHODS_STRING" ]]; then
            export "$rpc_cache_var_name"="$CURRENT_RPC_METHODS_STRING"
        fi
    fi
    # Use configuration directly, framework configuration is complete
    # No need for validation and fallback mechanism
    # Framework configuration is complete, use directly
    
    # Convert to array
    IFS=',' read -ra CURRENT_RPC_METHODS_ARRAY <<< "$CURRENT_RPC_METHODS_STRING"
    
    # Configuration consistency validation (safety check for hybrid solution)
    validate_config_consistency
    
    echo "🎯 Automatic configuration completed:" >&2
    echo "   Blockchain: $blockchain_node_lower" >&2
    echo "   RPC methods: $CURRENT_RPC_METHODS_STRING" >&2
    echo "   Method count: ${#CURRENT_RPC_METHODS_ARRAY[@]}" >&2
}

# Clear expired cache function
clear_config_cache() {
    local cache_pattern="${1:-CACHED_}"
    
    # Clear cache variables
    for var in $(compgen -v | grep "^${cache_pattern}" 2>/dev/null || true); do
        unset "$var" 2>/dev/null || true
    done
    
    # Clear deployment path detection variables
    unset DEPLOYMENT_PATHS_DETECTED 2>/dev/null || true
    
    echo "🧹 Configuration cache cleared" >&2
}

# =====================================================================
# Automatic Configuration Generation Functions
# =====================================================================

# Redesigned RPC method retrieval function
get_current_rpc_methods() {
    local rpc_mode_lower
    rpc_mode_lower=$(echo "${RPC_MODE}" | tr '[:upper:]' '[:lower:]')
    
    # Get corresponding mode methods from CHAIN_CONFIG's rpc_methods field
    local methods_string
    methods_string=$(echo "$CHAIN_CONFIG" | jq -r ".rpc_methods.\"$rpc_mode_lower\"")
    
    # Use configuration directly, framework configuration is complete
    
    # Framework configuration is complete, use directly
    
    echo "$methods_string"
}

get_param_format_from_json() {
    local method="$1"
    local format
    
    # Performance optimization: Use cached parameter format
    local param_cache_var_name="CACHED_PARAM_FORMAT_${method}"
    local cached_format="${!param_cache_var_name:-}"
    
    if [[ -n "$cached_format" ]]; then
        echo "$cached_format"
        return 0
    fi
    
    format=$(echo "$CHAIN_CONFIG" | jq -r ".param_formats.\"$method\"")
    
    if [[ "$format" == "null" || -z "$format" ]]; then
        format="single_address"  # Default format
    fi
    
    export "$param_cache_var_name"="$format"
    echo "$format"
}

# Validate if key variables are correctly set
if [[ -z "$ACCOUNTS_OUTPUT_FILE" ]]; then
    echo "⚠️ Warning: ACCOUNTS_OUTPUT_FILE not correctly set" >&2
fi
if [[ -z "$LOCAL_RPC_URL" ]]; then
    echo "⚠️ Warning: LOCAL_RPC_URL not correctly set" >&2
fi

# Execute automatic configuration generation
echo "Before calling generate_auto_config: BLOCKCHAIN_NODE=$BLOCKCHAIN_NODE" >&2
generate_auto_config

export -f get_current_rpc_methods get_param_format_from_json clear_config_cache generate_auto_config validate_config_consistency resolve_active_chain_template_file job_local_chain_override_matches
export LAST_BLOCKCHAIN_NODE="${BLOCKCHAIN_NODE:-solana}"
export ACCOUNTS_OUTPUT_FILE SINGLE_METHOD_TARGETS_FILE MIXED_METHOD_TARGETS_FILE
export LOCAL_RPC_URL MAINNET_RPC_URL BLOCKCHAIN_NODE BLOCKCHAIN_PROCESS_NAMES RPC_MODE
export ACCOUNT_COUNT ACCOUNT_OUTPUT_FILE ACCOUNT_MAX_SIGNATURES ACCOUNT_TX_BATCH_SIZE ACCOUNT_SEMAPHORE_LIMIT
export CHAIN_CONFIG ACTIVE_CHAIN_TEMPLATE_FILE DEPLOYMENT_PLATFORM_DETECTED
# Deployment mode + K8s/container paths (export for child monitor processes)
export DEPLOYMENT_MODE DEPLOYMENT_MODE_DETECTED DEPLOYMENT_MODE_SOURCE
export HOST_PROC HOST_SYS HOST_ROOT CGROUP_VERSION CGROUP_ROOT
# CGROUP_V1_* only set when cgroup v1, but export unconditionally — harmless if empty
export CGROUP_V1_BLKIO_PATH CGROUP_V1_MEMORY_PATH CGROUP_V1_CPU_PATH
export CURRENT_RPC_METHODS_STRING

export DATA_DIR CURRENT_TEST_DIR LOGS_DIR REPORTS_DIR VEGETA_RESULTS_DIR TMP_DIR ARCHIVES_DIR
export ERROR_LOG_DIR PYTHON_ERROR_LOG_DIR MEMORY_SHARE_DIR
export BLOCK_HEIGHT_CACHE_FILE BLOCK_HEIGHT_DATA_FILE QPS_STATUS_FILE BOTTLENECK_STATUS_FILE BOTTLENECK_COUNTERS_FILE NODE_HEALTH_CACHE_DIR
export LATEST_METRICS_FILE UNIFIED_METRICS_FILE UNIFIED_EVENTS_FILE EVENT_MANAGER_LOCK_FILE EVENT_NOTIFICATION_FILE TEST_SESSION_DIR
export UNIFIED_LOG PERFORMANCE_LATEST_CSV PROXY_METHOD_CSV PROXY_SELF_CSV RPC_PROXY_LOG NETWORK_CSV NETWORK_PID_FILE
export MONITORING_OVERHEAD_LOG PERFORMANCE_LOG ERROR_LOG TEMP_FILE_PATTERN SESSION_TIMESTAMP

export NETWORK_MAX_BANDWIDTH_MBPS DEPLOYMENT_PLATFORM ENA_MONITOR_ENABLED
export NETWORK_INTERFACE ENA_ALLOWANCE_FIELDS_STR
export BASE_FRAMEWORK_DIR BASE_DATA_DIR
export BLOCKCHAIN_PROCESS_NAMES_STR="${BLOCKCHAIN_PROCESS_NAMES[*]}"

export CONFIG_ALREADY_LOADED="true"

echo "🔧 RPC method configuration completed:" >&2
echo "   Blockchain type: $BLOCKCHAIN_NODE" >&2
echo "   RPC mode: $RPC_MODE" >&2
echo "   Current methods: $CURRENT_RPC_METHODS_STRING" >&2
echo "🎉 Layered configuration loading completed!" >&2
