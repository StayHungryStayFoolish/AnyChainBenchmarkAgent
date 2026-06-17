#!/bin/bash
# =====================================================================
# Blockchain Node Benchmark Framework - User Configuration Layer
# =====================================================================
# Target users: All users of the framework
# Configuration content: RPC connection, test parameters, disk devices, basic monitoring configuration
# Modification frequency: Frequently modified
# =====================================================================

# ----- Blockchain Node Configuration -----
# These values are the minimum required inputs for a normal benchmark run.
LOCAL_RPC_URL="${LOCAL_RPC_URL:-http://localhost:8899}"             # Local blockchain node RPC endpoint
MAINNET_RPC_URL="${MAINNET_RPC_URL:-}"                             # Optional: override mainnet/reference RPC endpoint; empty uses config/chains/<chain>.json defaults
BLOCKCHAIN_NODE="${BLOCKCHAIN_NODE:-solana}"                       # Must match a file in config/chains/<chain>.json
RPC_MODE="${RPC_MODE:-single}"                                     # Options: single | mixed

# Blockchain node process-name or command-line keywords used for monitoring
# attribution. The deployment-mode detector also reuses these values as
# systemd unit-name prefixes when auto-detecting vm_systemd.
# Adjust these to match the actual process names on your host.
BLOCKCHAIN_PROCESS_NAMES=(
    "blockchain"
    "validator"
    "agave-validator"
    "node.service"
)

# ----- Cloud / Machine Configuration -----
# Prefer Google Cloud style names so the same config can describe GCP/AWS/other hosts.
CLOUD_PROVIDER="${CLOUD_PROVIDER:-gcp}"                           # Options: gcp | aws | azure | other
CLOUD_REGION="${CLOUD_REGION:-us-central1}"                       # Example: us-central1, ap-east-1
CLOUD_ZONE="${CLOUD_ZONE:-us-central1-a}"                         # Example: us-central1-a; optional outside GCP
MACHINE_TYPE="${MACHINE_TYPE:-c3-standard-22}"                    # Example: c3-standard-22, m7i.4xlarge
# Keep the user-entered provider for reports. Runtime platform detection still uses DEPLOYMENT_PLATFORM.
REPORT_CLOUD_PROVIDER="${REPORT_CLOUD_PROVIDER:-$CLOUD_PROVIDER}"

# ----- Disk Device Configuration -----
LEDGER_DEVICE="${LEDGER_DEVICE:-sdb}"                             # DATA device from lsblk; replace with the real ledger/data disk
ACCOUNTS_DEVICE="${ACCOUNTS_DEVICE:-}"                            # Optional ACCOUNTS device from lsblk; leave empty for single-disk nodes

# Use unified naming convention {logical_name}_{device_name}_{metric}.
# DATA device uses data prefix, ACCOUNTS device uses accounts prefix.
# Data volume configuration
DATA_VOL_TYPE="${DATA_VOL_TYPE:-hyperdisk-extreme}"               # Examples: hyperdisk-extreme, hyperdisk-balanced, pd-ssd, gp3, io2, instance-store
DATA_VOL_SIZE="${DATA_VOL_SIZE:-2000}"                            # Required data size in GiB
DATA_VOL_MAX_IOPS="${DATA_VOL_MAX_IOPS:-30000}"                   # Provisioned disk IOPS or equivalent baseline
DATA_VOL_MAX_THROUGHPUT="${DATA_VOL_MAX_THROUGHPUT:-700}"         # Provisioned disk throughput in MiB/s

# Accounts volume configuration (optional)
ACCOUNTS_VOL_TYPE="${ACCOUNTS_VOL_TYPE:-hyperdisk-extreme}"       # Examples: hyperdisk-extreme, hyperdisk-balanced, pd-ssd, gp3, io2, instance-store
ACCOUNTS_VOL_SIZE="${ACCOUNTS_VOL_SIZE:-500}"                     # Required account data size in GiB
ACCOUNTS_VOL_MAX_IOPS="${ACCOUNTS_VOL_MAX_IOPS:-30000}"           # Provisioned disk IOPS or equivalent baseline
ACCOUNTS_VOL_MAX_THROUGHPUT="${ACCOUNTS_VOL_MAX_THROUGHPUT:-700}" # Provisioned disk throughput in MiB/s

# ----- Network Monitoring Configuration -----
# Machine network bandwidth configuration (unit: Gbps) - User must set according to machine type
NETWORK_INTERFACE="${NETWORK_INTERFACE:-}"                        # Optional: eth0, ens5, ens4, etc.; empty means auto-detect where supported
NETWORK_MAX_BANDWIDTH_GBPS="${NETWORK_MAX_BANDWIDTH_GBPS:-25}"    # Maximum network bandwidth (unit: Gbps) - User must set according to machine type

# Provider-specific NIC limitation monitoring. GCP uses gVNIC/virtio collectors;
# AWS enables ENA dynamically when DEPLOYMENT_PLATFORM=aws.
ENA_MONITOR_ENABLED=${ENA_MONITOR_ENABLED:-false}

# ----- Optional Chain Endpoint Overrides -----
# Leave these empty for fake-node/local closed-loop tests and for chains whose
# selected methods all use LOCAL_RPC_URL. Set them only when the selected chain
# template uses an indexer, REST sidecar, mirror node, or companion EVM RPC.
CHAIN_REST_URL="${CHAIN_REST_URL:-}"                               # REST/LCD/HTTP API override
CHAIN_INDEXER_URL="${CHAIN_INDEXER_URL:-}"                         # Indexer/Explorer API override
CHAIN_SIDECAR_URL="${CHAIN_SIDECAR_URL:-}"                         # Substrate Sidecar override
CHAIN_EVM_RPC_URL="${CHAIN_EVM_RPC_URL:-}"                         # EVM companion RPC override
CHAIN_JSON_RPC_URL="${CHAIN_JSON_RPC_URL:-}"                       # JSON-RPC companion endpoint override
CHAIN_MIRROR_URL="${CHAIN_MIRROR_URL:-}"                           # Mirror-node endpoint override
RPC_API_KEY="${RPC_API_KEY:-}"                                     # Optional API key for public gateway/indexer endpoints

# ----- Optional Chain Sample Overrides -----
# Used by account discovery, target generation, RPC fixture recording, and
# fake-node fixture validation. Leave empty to use the measured defaults
# embedded in config/chains/<chain>.json.
TARGET_ADDRESS="${TARGET_ADDRESS:-}"                               # Chain-specific account/address/contract sample
TARGET_TX_HASH="${TARGET_TX_HASH:-}"                               # Transaction hash sample for chains using tx_hash
TARGET_TXID="${TARGET_TXID:-}"                                     # Transaction id sample for UTXO/REST chains
TARGET_BLOCK_HASH="${TARGET_BLOCK_HASH:-}"                         # Block hash sample
TARGET_BLOCK="${TARGET_BLOCK:-}"                                   # Block identifier sample
TARGET_HEIGHT="${TARGET_HEIGHT:-}"                                 # Block height sample
TARGET_ROUND="${TARGET_ROUND:-}"                                   # Round sample for Algorand-like chains
TARGET_ASSET_ID="${TARGET_ASSET_ID:-}"                             # Asset id sample
TARGET_ASSET="${TARGET_ASSET:-}"                                   # JSON array/string asset sample when required
TARGET_EPOCH="${TARGET_EPOCH:-}"                                   # Epoch sample
TARGET_VP="${TARGET_VP:-}"                                         # Voting-power or validator-position sample
TARGET_POOL_ID="${TARGET_POOL_ID:-}"                               # Pool id sample
TARGET_TOKEN_ACCOUNT="${TARGET_TOKEN_ACCOUNT:-}"                   # Token account sample
TARGET_TOKEN_MINT="${TARGET_TOKEN_MINT:-}"                         # Token mint sample
TARGET_CONTRACT_ADDRESS="${TARGET_CONTRACT_ADDRESS:-}"             # Contract address sample
TARGET_EVM_ADDRESS="${TARGET_EVM_ADDRESS:-}"                       # EVM-form address sample for dual-protocol chains
TARGET_SIGNER_ID="${TARGET_SIGNER_ID:-}"                           # Signer id sample for NEAR-like tx lookup

# ----- Account and Target Generation Configuration -----
ACCOUNT_COUNT="${ACCOUNT_COUNT:-1000}"                             # Active account count used to generate Vegeta targets
ACCOUNT_MAX_SIGNATURES="${ACCOUNT_MAX_SIGNATURES:-50000}"          # Maximum signatures scanned by account discovery
ACCOUNT_TX_BATCH_SIZE="${ACCOUNT_TX_BATCH_SIZE:-100}"              # Transaction batch size for account discovery
ACCOUNT_SEMAPHORE_LIMIT="${ACCOUNT_SEMAPHORE_LIMIT:-10}"           # Account discovery concurrency limit

# ----- Monitoring Configuration -----
# Unified monitoring interval (seconds) - All monitoring tasks use the same interval
MONITOR_INTERVAL="${MONITOR_INTERVAL:-5}"                         # Unified monitoring interval, applicable to system resources, blockchain node, and monitoring overhead statistics
DISK_MONITOR_RATE="${DISK_MONITOR_RATE:-1}"                       # Disk separate monitoring frequency

# ----- Optional Observability Stack -----
# Disabled by default. When set to true, the benchmark entry script starts the
# read-only exporter, Prometheus, and Grafana stack before benchmark traffic.
# The stack is stopped during framework cleanup when auto-stop is true.
OBSERVABILITY_STACK_ENABLED="${OBSERVABILITY_STACK_ENABLED:-false}" # Options: true | false
OBSERVABILITY_STACK_AUTO_STOP="${OBSERVABILITY_STACK_AUTO_STOP:-true}" # Stop stack when benchmark exits
OBSERVABILITY_STACK_MODE="${OBSERVABILITY_STACK_MODE:-local}"       # Options: local | exporter
EXPORTER_PORT="${EXPORTER_PORT:-9108}"                             # Local Prometheus exporter port
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9091}"                         # Local Prometheus UI port
GRAFANA_PORT="${GRAFANA_PORT:-3001}"                                # Local Grafana UI port
PROMETHEUS_EXPORTER_MAX_PROXY_ROWS="${PROMETHEUS_EXPORTER_MAX_PROXY_ROWS:-20000}"

# ----- Optional Agent LLM Configuration -----
# The Agent remains usable without an LLM. When enabled, prefer Vertex AI with
# Google ADC or service-account impersonation for enterprise environments.
LLM_PROVIDER="${LLM_PROVIDER:-vertex_gemini_openai}"               # Options: vertex_gemini_openai | vertex_claude | openai
LLM_MODEL="${LLM_MODEL:-gemini-2.5-pro}"                           # Examples: gemini-2.5-pro, claude-3-7-sonnet@20250219, gpt-4.1
GOOGLE_AUTH_MODE="${GOOGLE_AUTH_MODE:-adc}"                        # Options: adc | attached_service_account | service_account_impersonation | service_account_file
GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-}"                   # Required for Vertex providers
GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-us-central1}"       # Vertex region
GOOGLE_SERVICE_ACCOUNT_EMAIL="${GOOGLE_SERVICE_ACCOUNT_EMAIL:-}"    # Required for service_account_impersonation; no JSON key required
GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-}" # Optional local JSON key fallback
OPENAI_API_KEY="${OPENAI_API_KEY:-}"                                # Required only when LLM_PROVIDER=openai

# ----- Optional Agent Chat Memory Configuration -----
# Defaults assume modern long-context models. The deterministic local estimator
# uses an approximate 4 characters per token when no provider tokenizer exists.
AGENT_CONTEXT_WINDOW_TOKENS="${AGENT_CONTEXT_WINDOW_TOKENS:-1000000}" # Default context window used for auto-compaction
AGENT_COMPACT_TRIGGER_RATIO="${AGENT_COMPACT_TRIGGER_RATIO:-0.7}"      # Auto-compact at context_window * ratio
AGENT_COMPACT_TURN_THRESHOLD="${AGENT_COMPACT_TURN_THRESHOLD:-40}"     # Also compact after this many chat turns
AGENT_COMPACT_KEEP_RECENT_TURNS="${AGENT_COMPACT_KEEP_RECENT_TURNS:-8}" # Recent raw turns kept after compaction

# ----- QPS Benchmark Configuration -----
# Quick benchmark mode (verify basic QPS capability)
QUICK_INITIAL_QPS=${QUICK_INITIAL_QPS:-1000}
QUICK_MAX_QPS=${QUICK_MAX_QPS:-1500}
QUICK_QPS_STEP=${QUICK_QPS_STEP:-500}
QUICK_DURATION=${QUICK_DURATION:-60}   # Test 1 minute per QPS level (avoid resource issues from long-running tests)

# Standard benchmark mode (standard performance testing)
STANDARD_INITIAL_QPS=${STANDARD_INITIAL_QPS:-2000}
STANDARD_MAX_QPS=${STANDARD_MAX_QPS:-50000}
STANDARD_QPS_STEP=${STANDARD_QPS_STEP:-500}
STANDARD_DURATION=${STANDARD_DURATION:-600}

# Intensive benchmark mode (automatically find system bottlenecks)
INTENSIVE_INITIAL_QPS=${INTENSIVE_INITIAL_QPS:-50000}
INTENSIVE_MAX_QPS=${INTENSIVE_MAX_QPS:-9999999}      # No practical upper limit, until bottleneck detected
INTENSIVE_QPS_STEP=${INTENSIVE_QPS_STEP:-250}
INTENSIVE_DURATION=${INTENSIVE_DURATION:-600}
INTENSIVE_AUTO_STOP=${INTENSIVE_AUTO_STOP:-true}      # Enable automatic bottleneck detection stop

# Benchmark interval configuration
QPS_COOLDOWN=${QPS_COOLDOWN:-30}      # Cooldown time between QPS levels (seconds)
QPS_WARMUP_DURATION=${QPS_WARMUP_DURATION:-60}  # Warmup time (seconds)

# Export user configuration variables
export LOCAL_RPC_URL MAINNET_RPC_URL BLOCKCHAIN_NODE RPC_MODE
export CHAIN_REST_URL CHAIN_INDEXER_URL CHAIN_SIDECAR_URL CHAIN_EVM_RPC_URL CHAIN_JSON_RPC_URL CHAIN_MIRROR_URL RPC_API_KEY
export ACCOUNT_COUNT ACCOUNT_MAX_SIGNATURES ACCOUNT_TX_BATCH_SIZE ACCOUNT_SEMAPHORE_LIMIT
export TARGET_ADDRESS TARGET_TX_HASH TARGET_TXID TARGET_BLOCK_HASH TARGET_BLOCK TARGET_HEIGHT TARGET_ROUND
export TARGET_ASSET_ID TARGET_ASSET TARGET_EPOCH TARGET_VP TARGET_POOL_ID TARGET_TOKEN_ACCOUNT TARGET_TOKEN_MINT
export TARGET_CONTRACT_ADDRESS TARGET_EVM_ADDRESS TARGET_SIGNER_ID
export CLOUD_PROVIDER REPORT_CLOUD_PROVIDER CLOUD_REGION CLOUD_ZONE MACHINE_TYPE
export LEDGER_DEVICE ACCOUNTS_DEVICE
export DATA_VOL_TYPE DATA_VOL_SIZE DATA_VOL_MAX_IOPS DATA_VOL_MAX_THROUGHPUT
export ACCOUNTS_VOL_TYPE ACCOUNTS_VOL_SIZE ACCOUNTS_VOL_MAX_IOPS ACCOUNTS_VOL_MAX_THROUGHPUT
export NETWORK_INTERFACE NETWORK_MAX_BANDWIDTH_GBPS ENA_MONITOR_ENABLED MONITOR_INTERVAL DISK_MONITOR_RATE
export OBSERVABILITY_STACK_ENABLED OBSERVABILITY_STACK_AUTO_STOP OBSERVABILITY_STACK_MODE EXPORTER_PORT PROMETHEUS_PORT GRAFANA_PORT PROMETHEUS_EXPORTER_MAX_PROXY_ROWS
export LLM_PROVIDER LLM_MODEL GOOGLE_AUTH_MODE GOOGLE_CLOUD_PROJECT GOOGLE_CLOUD_LOCATION GOOGLE_SERVICE_ACCOUNT_EMAIL GOOGLE_APPLICATION_CREDENTIALS OPENAI_API_KEY
export AGENT_CONTEXT_WINDOW_TOKENS AGENT_COMPACT_TRIGGER_RATIO AGENT_COMPACT_TURN_THRESHOLD AGENT_COMPACT_KEEP_RECENT_TURNS
export QUICK_INITIAL_QPS QUICK_MAX_QPS QUICK_QPS_STEP QUICK_DURATION
export STANDARD_INITIAL_QPS STANDARD_MAX_QPS STANDARD_QPS_STEP STANDARD_DURATION
export INTENSIVE_INITIAL_QPS INTENSIVE_MAX_QPS INTENSIVE_QPS_STEP INTENSIVE_DURATION INTENSIVE_AUTO_STOP
export QPS_COOLDOWN QPS_WARMUP_DURATION
export BLOCKCHAIN_PROCESS_NAMES_STR="${BLOCKCHAIN_PROCESS_NAMES[*]}"
