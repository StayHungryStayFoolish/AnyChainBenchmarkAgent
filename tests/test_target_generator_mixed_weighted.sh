#!/usr/bin/env bash
# Verify target_generator.sh honors rpc_methods.mixed_weighted in mixed mode.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d -t bnb-mixed-weighted-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

ACCOUNTS_FILE="$TMP_DIR/accounts.txt"
TARGETS_FILE="$TMP_DIR/targets.jsonl"
ERR_FILE="$TMP_DIR/target_generator.err"

for i in $(seq 1 100); do
    printf '0x%040x\n' "$i"
done > "$ACCOUNTS_FILE"

(
    cd "$REPO_ROOT"
    BLOCKCHAIN_NODE=ethereum \
    RPC_MODE=mixed \
    LOCAL_RPC_URL=http://127.0.0.1:19000 \
        ./tools/target_generator.sh \
            --rpc-mode mixed \
            --rpc-url http://127.0.0.1:19000 \
            -a "$ACCOUNTS_FILE" \
            -o "$TARGETS_FILE" \
            >/dev/null 2>"$ERR_FILE"
)

python3 - "$TARGETS_FILE" <<'PY'
import base64
import collections
import json
import sys

expected = {
    "eth_getBalance": 25,
    "eth_getTransactionCount": 25,
    "eth_blockNumber": 25,
    "eth_gasPrice": 25,
}

counts = collections.Counter()
with open(sys.argv[1]) as fh:
    for line in fh:
        target = json.loads(line)
        body = json.loads(base64.b64decode(target["body"]))
        counts[body["method"]] += 1

if counts != expected:
    print(f"unexpected mixed_weighted distribution: {dict(counts)}", file=sys.stderr)
    print(f"expected: {expected}", file=sys.stderr)
    raise SystemExit(1)
PY

echo "PASS: target_generator mixed_weighted distribution"

OVERRIDE_DIR="$(mktemp -d -t bnb-mixed-override-XXXXXX)"
trap 'rm -rf "$TMP_DIR" "$OVERRIDE_DIR"' EXIT
OVERRIDE_FILE="$OVERRIDE_DIR/chain_template.override.json"
OVERRIDE_TARGETS_FILE="$OVERRIDE_DIR/targets.jsonl"

cat > "$OVERRIDE_FILE" <<'JSON'
{
  "chain_type": "ethereum",
  "rpc_methods": {
    "single": "eth_customTriple",
    "mixed": "eth_customTriple",
    "mixed_weighted": [
      {"method": "eth_customTriple", "weight": 100}
    ]
  },
  "param_spec": {
    "eth_customTriple": {
      "transport": "jsonrpc_list",
      "params": [
        {"source": "address"},
        {"literal": "latest"},
        {"literal": 7, "type": "int"}
      ]
    }
  },
  "params": {
    "target_address": "0x0000000000000000000000000000000000000000"
  }
}
JSON

(
    cd "$REPO_ROOT"
    BLOCKCHAIN_NODE=ethereum \
    RPC_MODE=mixed \
    LOCAL_RPC_URL=http://127.0.0.1:19000 \
    CHAIN_CONFIG_OVERRIDE_FILE="$OVERRIDE_FILE" \
        ./tools/target_generator.sh \
            --rpc-mode mixed \
            --rpc-url http://127.0.0.1:19000 \
            -a "$ACCOUNTS_FILE" \
            -o "$OVERRIDE_TARGETS_FILE" \
            >/dev/null 2>"$ERR_FILE"
)

python3 - "$OVERRIDE_TARGETS_FILE" <<'PY'
import base64
import collections
import json
import sys

counts = collections.Counter()
sample = None
with open(sys.argv[1]) as fh:
    for line in fh:
        target = json.loads(line)
        body = json.loads(base64.b64decode(target["body"]))
        counts[body["method"]] += 1
        sample = sample or body

if set(counts) != {"eth_customTriple"}:
    print(f"runtime override leaked default mixed methods: {dict(counts)}", file=sys.stderr)
    raise SystemExit(1)
if not sample or sample.get("params", [None, None, None])[1:] != ["latest", 7]:
    print(f"runtime override param_spec was not applied: {sample}", file=sys.stderr)
    raise SystemExit(1)
PY

echo "PASS: target_generator runtime override removes default mixed methods and applies custom param_spec"
