#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/monitoring/block_height_monitor.sh"

RPC_MODE="sync_observe"
SYNC_OBSERVE_MODE="true"
SYNC_OBSERVE_RPC_URL="http://sync-node:8545"
LOCAL_RPC_URL="http://benchmark-node:8545"

observed="$(get_observed_rpc_url)"
[[ "$observed" == "$SYNC_OBSERVE_RPC_URL" ]] || {
    echo "sync-observe selected the wrong endpoint: $observed" >&2
    exit 1
}

RPC_MODE="single"
SYNC_OBSERVE_MODE="false"
observed="$(get_observed_rpc_url)"
[[ "$observed" == "$LOCAL_RPC_URL" ]] || {
    echo "RPC benchmark selected the wrong endpoint: $observed" >&2
    exit 1
}

RPC_MODE="sync_observe"
SYNC_OBSERVE_MODE="true"
SYNC_OBSERVE_RPC_URL=""
if get_observed_rpc_url >/dev/null 2>&1; then
    echo "sync-observe accepted a missing observation endpoint" >&2
    exit 1
fi

echo "sync-observe monitor endpoint contract: PASS"
