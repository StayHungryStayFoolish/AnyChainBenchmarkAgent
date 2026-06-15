# Optional Prometheus / Grafana Stack

This directory contains an optional local observability stack for
blockchain-node-benchmark.

It is disabled by default. When `OBSERVABILITY_STACK_ENABLED=true`, the
benchmark entry script starts it before benchmark traffic begins and stops it
during framework cleanup by default. The stack reads the framework's existing
output files through the read-only exporter:

```text
benchmark runtime files -> monitoring/prometheus_exporter.py -> Prometheus -> Grafana
```

## Services

- `exporter`: runs `monitoring/prometheus_exporter.py` and exposes `/metrics`
- `prometheus`: scrapes the exporter
- `grafana`: loads a pre-provisioned datasource and dashboard

## Two Integration Modes

### Mode 1: Local Stack

Use this when you want the framework to start a self-contained local exporter,
Prometheus, and Grafana stack:

```bash
OBSERVABILITY_STACK_ENABLED=true
OBSERVABILITY_STACK_MODE=local
OBSERVABILITY_STACK_AUTO_STOP=true
```

This mode is useful for single-host testing, demos, and users who do not
already operate Prometheus/Grafana. The entrypoint starts the stack before
benchmark traffic and stops it during cleanup when auto-stop is enabled.

### Mode 2: Existing Prometheus/Grafana

Use this when your environment already has Prometheus and Grafana. In this
mode, the framework starts only the read-only exporter:

```bash
OBSERVABILITY_STACK_ENABLED=true
OBSERVABILITY_STACK_MODE=exporter
OBSERVABILITY_STACK_AUTO_STOP=true
EXPORTER_PORT=9108
```

Then add a scrape job to your existing Prometheus. Replace
`<benchmark-host>` with the host or service name where this framework runs:

```yaml
scrape_configs:
  - job_name: blockchain-node-benchmark
    metrics_path: /metrics
    static_configs:
      - targets:
          - <benchmark-host>:9108
```

In an existing Grafana deployment, add or reuse the Prometheus datasource, then
import `deploy/observability/grafana/dashboards/blockchain-node-benchmark.json`
or copy the panel queries from that dashboard.

When the benchmark exits, `OBSERVABILITY_STACK_AUTO_STOP=true` stops the
exporter. Your existing Prometheus keeps already-scraped time series according
to its own retention policy, but it will not receive new samples after the
exporter stops. The durable benchmark outputs remain in the framework archive
and HTML reports.

## Start

From the repository root:

```bash
OBSERVABILITY_STACK_ENABLED=true deploy/observability/start.sh
```

Or set the switch once in `config/user_config.sh`:

```bash
OBSERVABILITY_STACK_ENABLED=true
OBSERVABILITY_STACK_AUTO_STOP=true
```

Then either run the benchmark entry script, which starts and stops the stack
automatically, or start the stack manually:

```bash
deploy/observability/start.sh
```

For one-off local testing without changing config:

```bash
deploy/observability/start.sh --force
```

Open:

- Exporter: `http://localhost:9108/metrics`
- Prometheus: `http://localhost:9091`
- Grafana: `http://localhost:3001`

Default Grafana login:

```text
admin / admin
```

Override it with:

```bash
GRAFANA_ADMIN_USER=admin \
GRAFANA_ADMIN_PASSWORD='change-me' \
docker compose -f deploy/observability/docker-compose.yml up -d
```

## Stop

```bash
deploy/observability/stop.sh
```

To also remove Prometheus/Grafana stored data:

```bash
deploy/observability/stop.sh -v
```

## Runtime Data Paths

By default, the stack reads:

```text
BENCHMARK_DATA_DIR=<deployment-root>/blockchain-node-benchmark-result
BENCHMARK_MEMORY_DIR=<deployment-root>/blockchain-node-benchmark-result/current/memory
```

The framework's Linux production default for live memory state is:

```text
MEMORY_SHARE_DIR=/dev/shm/blockchain-node-benchmark
```

For containerized local testing, run the benchmark with a shared memory-state
directory so the exporter container can read it:

```bash
export BENCHMARK_DATA_DIR="$(dirname "$PWD")/blockchain-node-benchmark-result"
export BENCHMARK_MEMORY_DIR="$BENCHMARK_DATA_DIR/current/memory"

MEMORY_SHARE_DIR="$BENCHMARK_MEMORY_DIR" \
BLOCKCHAIN_BENCHMARK_DATA_DIR="$BENCHMARK_DATA_DIR" \
./blockchain_node_benchmark.sh
```

Then start the observability stack with the same paths:

```bash
BENCHMARK_DATA_DIR="$BENCHMARK_DATA_DIR" \
BENCHMARK_MEMORY_DIR="$BENCHMARK_MEMORY_DIR" \
docker compose -f deploy/observability/docker-compose.yml up -d
```

## Configuration

Common overrides:

```bash
OBSERVABILITY_STACK_ENABLED=true
OBSERVABILITY_STACK_AUTO_STOP=true
OBSERVABILITY_STACK_MODE=local
BLOCKCHAIN_NODE=ethereum
RPC_MODE=mixed
EXPORTER_PORT=9108
PROMETHEUS_PORT=9091
GRAFANA_PORT=3001
PROMETHEUS_EXPORTER_MAX_PROXY_ROWS=20000
```

## Design Boundaries

The observability stack must remain optional:

- it does not start benchmark tests;
- it does not query blockchain RPC endpoints;
- it does not write benchmark runtime files;
- it does not replace CSV/HTML reports;
- exporter failure must not fail a benchmark run.

Retention belongs to the Prometheus that scrapes the exporter:

- Local-stack mode keeps Prometheus/Grafana volumes unless you run
  `deploy/observability/stop.sh -v`.
- Existing-Prometheus mode stores samples in the user's Prometheus. Stopping
  the framework exporter only stops future scrapes; historical samples remain
  until that Prometheus retention expires.

The stack is controlled by `OBSERVABILITY_STACK_ENABLED=false` in
`config/user_config.sh`. When enabled through the benchmark entry command, it is
stopped automatically if `OBSERVABILITY_STACK_AUTO_STOP=true`.
