#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

rpc_url="http://geth-dev:8545"
metrics_url="http://geth-dev:6060/debug/metrics/prometheus"

require_bench() {
    if [[ "$(docker compose ps --status running --services bench 2>/dev/null)" != "bench" ]]; then
        echo "bench service is not running; start it with: docker compose up -d bench" >&2
        exit 1
    fi
}

probe() {
    require_bench
    docker compose exec -T bench bash -lc "
        set -euo pipefail
        rpc_response=\$(curl -fsS -H 'Content-Type: application/json' \\
          --data '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"eth_chainId\",\"params\":[]}' \\
          '$rpc_url')
        [[ \"\$rpc_response\" == *'\"result\":\"0x539\"'* ]]
        metrics_response=\$(curl -fsS '$metrics_url')
        [[ \"\$metrics_response\" == *'# TYPE'* ]]
        printf '%s\\n' \"\$rpc_response\"
        printf 'metrics=ready\\n'
    "
}

start() {
    require_bench
    docker compose --profile local-evm up -d geth-dev
    for _ in $(seq 1 30); do
        if probe >/dev/null 2>&1; then
            probe
            return
        fi
        sleep 1
    done
    docker compose logs --no-color geth-dev >&2
    echo "local EVM node did not become ready" >&2
    exit 1
}

stop() {
    docker compose --profile local-evm rm -sf geth-dev >/dev/null
}

case "${1:-}" in
    start) start ;;
    probe) probe ;;
    stop) stop ;;
    *)
        echo "usage: $0 {start|probe|stop}" >&2
        exit 2
        ;;
esac
