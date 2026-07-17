#!/bin/bash
# =====================================================================
# tests/test_proxy_phase.sh
# Integration test for proxy lifecycle wiring.
#
# Coverage:
#   1. Start fake-node v2 on :19101 to mock the real chain.
#   2. Start proxy on :18545 upstream=http://localhost:19101 via the
#      lib/proxy_lifecycle.sh start_rpc_proxy() helper.
#   3. Send a JSON-RPC getBalance via curl to :18545.
#   4. Stop proxy via stop_rpc_proxy() and assert proxy_method.csv exists,
#      has >1 lines, and contains a getBalance row.
#
# Run:
#   ./tests/test_proxy_phase.sh
# =====================================================================
set -u
trap 'cleanup' EXIT

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SCRIPT_DIR="$REPO_ROOT"
export BLOCKCHAIN_NODE="solana"
export ACTIVE_CHAIN_TEMPLATE_FILE="$REPO_ROOT/config/chains/solana.json"
export PROXY_LISTEN_PORT="18545"

WORK_DIR="$(mktemp -d -t bnb-proxy-itest-XXXX)"
export LOGS_DIR="$WORK_DIR/logs"
mkdir -p "$LOGS_DIR"

FAKE_PORT="19101"
FAKE_BIN="/tmp/fake-node-v2"
FAKE_LOG="$WORK_DIR/fake-node.log"
FAKE_PID=""
RPC_METHOD="getBalance"
RPC_BODY='{"jsonrpc":"2.0","id":1,"method":"getBalance","params":["11111111111111111111111111111111"]}'

cleanup() {
    set +e
    if declare -F stop_rpc_proxy >/dev/null 2>&1 && [[ "${PROXY_ENABLED:-0}" == "1" ]]; then
        stop_rpc_proxy >/dev/null 2>&1
    fi
    if [[ -n "$FAKE_PID" ]] && kill -0 "$FAKE_PID" 2>/dev/null; then
        kill -TERM "$FAKE_PID" 2>/dev/null
        sleep 1
        kill -9 "$FAKE_PID" 2>/dev/null
    fi
    # Keep work dir if FAIL set, for debugging
    if [[ "${KEEP_WORK_DIR:-0}" != "1" ]]; then
        rm -rf "$WORK_DIR"
    else
        echo "ℹ️  Kept work dir: $WORK_DIR"
    fi
}

fail() {
    echo "❌ FAIL: $*" >&2
    export KEEP_WORK_DIR=1
    exit 1
}

echo "=== Step 1: start fake-node v2 on :$FAKE_PORT ==="
if [[ ! -x "$FAKE_BIN" ]]; then
    fail "fake-node v2 binary not found at $FAKE_BIN"
fi
"$FAKE_BIN" -chain=solana -port="$FAKE_PORT" >"$FAKE_LOG" 2>&1 &
FAKE_PID=$!
sleep 1
if ! kill -0 "$FAKE_PID" 2>/dev/null; then
    cat "$FAKE_LOG"
    fail "fake-node failed to start"
fi

# Verify fake-node responds
if ! fake_resp=$(curl -fsS -m 3 -X POST -H 'Content-Type: application/json' \
    -d "$RPC_BODY" \
    "http://localhost:${FAKE_PORT}/" 2>&1); then
    cat "$FAKE_LOG"
    fail "fake-node returned error for ${RPC_METHOD}: $fake_resp"
fi
if [[ -z "$fake_resp" ]]; then
    fail "fake-node not responding on :$FAKE_PORT"
fi
echo "   fake-node OK, sample response: $(echo "$fake_resp" | head -c 120)"

echo "=== Step 2: source lib/proxy_lifecycle.sh and start_rpc_proxy ==="
export LOCAL_RPC_URL="http://localhost:${FAKE_PORT}"
source "${SCRIPT_DIR}/lib/proxy_lifecycle.sh" || fail "cannot source proxy_lifecycle.sh"

start_rpc_proxy
if [[ "${PROXY_ENABLED:-0}" != "1" ]]; then
    cat "$LOGS_DIR/rpc_proxy.log" 2>/dev/null
    fail "start_rpc_proxy did not set PROXY_ENABLED=1"
fi
if [[ "$LOCAL_RPC_URL" != "http://localhost:${PROXY_LISTEN_PORT}" ]]; then
    fail "LOCAL_RPC_URL not redirected; got=$LOCAL_RPC_URL"
fi
if [[ -z "${PROXY_METHOD_CSV:-}" ]]; then
    fail "PROXY_METHOD_CSV not exported"
fi
echo "   PROXY_PID=$PROXY_PID PROXY_METHOD_CSV=$PROXY_METHOD_CSV"

echo "=== Step 3: send ${RPC_METHOD} via proxy at :$PROXY_LISTEN_PORT ==="
for i in 1 2 3; do
    body="${RPC_BODY/\"id\":1/\"id\":$i}"
    if ! proxy_resp=$(curl -fsS -m 3 -X POST -H 'Content-Type: application/json' \
        -d "$body" \
        "http://localhost:${PROXY_LISTEN_PORT}/" 2>&1); then
        fail "proxy returned error for ${RPC_METHOD} on call $i: $proxy_resp"
    fi
    if [[ -z "$proxy_resp" ]]; then
        fail "proxy did not return response on call $i"
    fi
    echo "   call#$i resp: $(echo "$proxy_resp" | head -c 120)"
done

# Give sink a moment to flush
sleep 1

echo "=== Step 4: stop proxy and verify proxy_method.csv ==="
stop_rpc_proxy

if [[ "$LOCAL_RPC_URL" != "http://localhost:${FAKE_PORT}" ]]; then
    fail "LOCAL_RPC_URL not restored after stop; got=$LOCAL_RPC_URL"
fi

CSV="$LOGS_DIR/proxy_method.csv"
if [[ ! -f "$CSV" ]]; then
    fail "proxy_method.csv not produced at $CSV"
fi

lines=$(wc -l < "$CSV")
if [[ "$lines" -le 1 ]]; then
    cat "$CSV"
    fail "proxy_method.csv has only $lines line(s); expected >1"
fi
echo "   proxy_method.csv has $lines lines"
echo "   --- head -3 ---"
head -3 "$CSV"
echo "   --- end ---"

if ! grep -q "$RPC_METHOD" "$CSV"; then
    cat "$CSV"
    fail "proxy_method.csv does not contain '$RPC_METHOD'"
fi

echo ""
echo "✅ PASS: test_proxy_phase.sh — proxy_method.csv produced, contains ${RPC_METHOD}, lifecycle clean"
exit 0
