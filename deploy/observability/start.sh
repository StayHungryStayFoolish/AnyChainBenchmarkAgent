#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_ROOT="$(dirname "$REPO_ROOT")"

cd "$REPO_ROOT"

if [[ -f "${REPO_ROOT}/config/user_config.sh" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/config/user_config.sh"
fi

if [[ "${OBSERVABILITY_STACK_ENABLED:-false}" != "true" && "${1:-}" != "--force" ]]; then
  cat <<EOF
Observability stack is disabled.

Set OBSERVABILITY_STACK_ENABLED=true in config/user_config.sh or run:
  OBSERVABILITY_STACK_ENABLED=true deploy/observability/start.sh

For one-off local testing, you can also run:
  deploy/observability/start.sh --force
EOF
  exit 0
fi

export BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-${DEPLOYMENT_ROOT}/blockchain-node-benchmark-result}"
export BENCHMARK_MEMORY_DIR="${BENCHMARK_MEMORY_DIR:-${BENCHMARK_DATA_DIR}/current/memory}"
mkdir -p "$BENCHMARK_DATA_DIR/current/logs" "$BENCHMARK_MEMORY_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to start the optional observability stack." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is required to start the optional observability stack." >&2
  exit 1
fi

docker compose -f deploy/observability/docker-compose.yml up -d

cat <<EOF
Observability stack started.

Exporter:   http://localhost:${EXPORTER_PORT:-9108}/metrics
Prometheus: http://localhost:${PROMETHEUS_PORT:-9091}
Grafana:    http://localhost:${GRAFANA_PORT:-3001}

Grafana default login: ${GRAFANA_ADMIN_USER:-admin} / ${GRAFANA_ADMIN_PASSWORD:-admin}

Benchmark data:
  BENCHMARK_DATA_DIR=$BENCHMARK_DATA_DIR
  BENCHMARK_MEMORY_DIR=$BENCHMARK_MEMORY_DIR

For live memory-state metrics, run benchmark with:
  MEMORY_SHARE_DIR="$BENCHMARK_MEMORY_DIR" BLOCKCHAIN_BENCHMARK_DATA_DIR="$BENCHMARK_DATA_DIR" ./blockchain_node_benchmark.sh
EOF
