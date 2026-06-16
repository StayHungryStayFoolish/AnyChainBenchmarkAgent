# AnyChain Benchmark Agent

[English](README.md) | [中文](README_ZH.md)

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License](https://img.shields.io/badge/License-Commercial-green.svg)](COMMERCIAL.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Shell Script](https://img.shields.io/badge/shell-bash-green.svg)](https://www.gnu.org/software/bash/)

A production-oriented Agent and benchmark framework for any blockchain node QPS,
latency, bottleneck, sync-health, and per-RPC-method analysis. The Agent turns a
benchmark goal into a validated plan, runs preflight checks, submits a job,
tracks artifacts, and explains the result with evidence.

The benchmark execution plane remains deterministic: Vegeta, RPC proxy,
monitoring collectors, fake-node, report generation, and archiving are still the
source of truth. LLMs are optional and only draft structured requests; they do
not execute commands directly.

## What It Does

- Supports 36 chain templates across 6 adapter families.
- Generates single or weighted mixed RPC workloads from `config/chains/*.json`.
- Supports custom RPC methods through `rpc_methods`, `param_formats`, optional
  `param_spec`, REST path bindings, and fake-node fixtures.
- Records per-method status, success/failure counts, and P50/P90/P99 latency.
- Monitors CPU, memory, disk, network, cgroup, sync health, and monitor overhead.
- Produces HTML reports and archives every run.
- Provides optional Prometheus/Grafana telemetry through a read-only exporter.
- Provides an Agent control plane for prompt-first planning, risk scoring,
  capability gap analysis, artifact-aware Q&A, and long-running job tracking.

## How The Agent Works

```text
prompt or request
  -> intent routing
  -> request draft
  -> read-only discovery
  -> benchmark plan
  -> risk score and preflight
  -> approved job
  -> artifact index
  -> evidence-based analysis
```

The Agent is intentionally bounded:

- It answers framework questions from local docs and live repository state.
- It reads chain/RPC/fake-node capabilities from current files, not model memory.
- It produces onboarding plans for unsupported chains or RPC methods.
- It runs only allowlisted benchmark commands after approval.
- It writes Agent-generated runtime config to job-local `runtime.env`, not to
  `config/user_config.sh`.

## Quick Start With The Agent

Check local dependencies without modifying the host:

```bash
bash scripts/install_deps.sh --check
```

Inspect what the framework currently supports:

```bash
python3 agent/cli.py capabilities
python3 agent/cli.py ask --prompt "How many chains and RPC methods does the framework support?"
```

Draft a benchmark request from a prompt:

```bash
python3 agent/cli.py draft-request \
  --prompt "Test Solana maximum stable QPS on GKE with fake-node smoke first" \
  --output /tmp/request.json
```

Generate and inspect a plan:

```bash
python3 agent/cli.py plan \
  --request /tmp/request.json \
  --output /tmp/plan.json \
  --discover \
  --dry-run

python3 agent/cli.py preflight --plan /tmp/plan.json
python3 agent/cli.py risk-score --plan /tmp/plan.json
python3 agent/cli.py runbook --plan /tmp/plan.json --output /tmp/runbook.md
```

Submit a lifecycle-only mock job:

```bash
python3 agent/cli.py submit --plan /tmp/plan.json --mock
```

Submit a real benchmark only after reviewing the plan and runbook:

```bash
python3 agent/cli.py submit --plan /tmp/plan.json --approved
```

Check or analyze a job:

```bash
python3 agent/cli.py status --job-id <job_id>
python3 agent/cli.py analyze --job-id <job_id>
python3 agent/cli.py artifact-qa --job-id <job_id> --question "Why are charts empty?"
```

## Optional LLM Providers

The Agent works without an LLM. If enabled, LLMs are used only for request
drafting and intent classification, followed by deterministic validation.

Supported provider contracts:

- `vertex_gemini_openai`: Gemini on Vertex AI through the OpenAI-compatible API.
- `vertex_claude`: Claude partner models on Vertex AI.
- `openai`: OpenAI API.
- `fake`: offline protocol smoke provider for tests.

Recommended enterprise Vertex configuration uses ADC or service-account
impersonation instead of static API keys:

```bash
export LLM_PROVIDER=vertex_gemini_openai
export LLM_MODEL=gemini-2.5-pro
export GOOGLE_AUTH_MODE=service_account_impersonation
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_SERVICE_ACCOUNT_EMAIL=benchmark-agent@your-project.iam.gserviceaccount.com
```

Validate config without calling a model:

```bash
python3 agent/cli.py llm-config
```

Run an offline LLM protocol smoke without credentials:

```bash
python3 agent/cli.py llm-smoke --mock
```

Run a real provider smoke only after credentials are configured:

```bash
python3 agent/cli.py llm-smoke --prompt 'Return JSON only: {"ok": true}'
```

## Traditional Benchmark Entry

You can still run the benchmark engine directly.

Configure the minimum runtime values in `config/user_config.sh`:

```bash
BLOCKCHAIN_NODE="solana"
RPC_MODE="single"
LOCAL_RPC_URL="http://localhost:8899"
MAINNET_RPC_URL=""

BLOCKCHAIN_PROCESS_NAMES=("agave-validator" "solana-validator" "validator")

CLOUD_PROVIDER="gcp"
CLOUD_REGION="us-central1"
MACHINE_TYPE="c3-standard-22"
LEDGER_DEVICE="sdb"
DATA_VOL_TYPE="hyperdisk-extreme"
DATA_VOL_MAX_IOPS="30000"
DATA_VOL_MAX_THROUGHPUT="700"
NETWORK_MAX_BANDWIDTH_GBPS=25
```

Run a quick VM or bare-metal benchmark:

```bash
./blockchain_node_benchmark.sh --quick
```

Run a local fake-node closed loop:

```bash
BLOCKCHAIN_NODE=solana \
RPC_MODE=single \
QUICK_INITIAL_QPS=1 \
QUICK_MAX_QPS=1 \
QUICK_QPS_STEP=1 \
QUICK_DURATION=3 \
QPS_WARMUP_DURATION=0 \
QPS_COOLDOWN=0 \
./blockchain_node_benchmark.sh --quick --single --fake-node
```

For Kubernetes-hosted nodes, deploy the collector first:

```bash
deploy/k8s/validate.sh --preflight
kubectl apply -f deploy/k8s/
kubectl rollout status -n blockchain-bench ds/blockchain-bench-collector
deploy/k8s/validate.sh --post-deploy
```

Then run the benchmark from your selected runner with the same
`config/user_config.sh` settings.

## Reports And Artifacts

Current-run files are written under the runtime `current/` directory and durable
outputs are archived after the run.

Key artifacts:

- `current/reports/performance_report_*.html`
- `current/logs/proxy_method.csv`
- `current/logs/performance_latest.csv`
- `archives/<run-id>/test_summary.json`
- `.agent/jobs/<job_id>/artifact_index.json` for Agent jobs
- `.agent/jobs/<job_id>/runtime.env` for Agent materialized runtime config

## Optional Prometheus/Grafana

Prometheus/Grafana is disabled by default. Enable it in `config/user_config.sh`:

```bash
OBSERVABILITY_STACK_ENABLED=true
OBSERVABILITY_STACK_AUTO_STOP=true
OBSERVABILITY_STACK_MODE=local   # local | exporter
EXPORTER_PORT=9108
PROMETHEUS_PORT=9091
GRAFANA_PORT=3001
```

Use `OBSERVABILITY_STACK_MODE=exporter` when you already have Prometheus and
Grafana and only need this framework to expose a scrape endpoint.

## Extending Chains Or RPC Methods

Use the Agent to inspect gaps before editing templates:

```bash
python3 agent/cli.py gap-analysis \
  --chain solana \
  --method getBalance \
  --method customMethod
```

For unsupported chains, the Agent returns an onboarding plan instead of editing
code automatically. The usual path is:

1. Add `config/chains/<chain>.json` from the template.
2. Select `_meta.adapter_family`.
3. Configure `rpc_methods.single` and `rpc_methods.mixed_weighted`.
4. Add `param_formats` or `param_spec`.
5. Add `proxy_extraction` rules.
6. Record fake-node fixtures.
7. Run preflight and fake-node closed-loop tests.

## Reference Documentation

- [Configuration Guide](config/README.md)
- [Agent Control Plane](agent/README.md)
- [Full Framework Reference](docs/en/framework-reference.md)
- [Framework Flow and Data Lifecycle](docs/en/framework-flow.md)
- [Module Guide](docs/en/module-guide.md)
- [How to Add a Chain or RPC Method](docs/en/how-to-add-chain.md)
- [Local Closed-Loop Testing with fake-node](docs/en/local-closed-loop-testing.md)
- [GitHub PR Gates and Branch Protection](docs/en/github-pr-gates.md)
- [Prometheus / Grafana Observability](deploy/observability/README.md)
- [Kubernetes Collector](deploy/k8s/README.md)

## License

This project is dual licensed:

- AGPL-3.0-or-later for open-source use. See [LICENSE](LICENSE).
- Commercial licensing for proprietary/internal use cases. See [COMMERCIAL.md](COMMERCIAL.md).
