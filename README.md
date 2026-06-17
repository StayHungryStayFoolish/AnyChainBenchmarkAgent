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

## 5-Minute Quick Start

This is the fastest way to use AnyChain Benchmark Agent from a terminal. It does
not require a real blockchain node or an LLM key. The Agent will turn your goal
into a request, discover the local environment, generate a plan, run preflight,
submit a mock job, answer questions, and analyze evidence.

Clone the repository:

```bash
git clone git@github.com:StayHungryStayFoolish/AnyChainBenchmarkAgent.git
cd AnyChainBenchmarkAgent
```

Check dependencies without modifying the host:

```bash
bash scripts/install_deps.sh --check
```

Start the Agent terminal session:

```bash
./bin/anychain-agent
```

Then talk to it:

```text
> What chains and RPC methods do you support?
> Create a Solana fake-node smoke benchmark at 1 QPS
> plan
> preflight
> run mock
> status
> analyze
> compact
> memory
> qa What evidence was generated?
```

You can also run a one-shot prompt:

```bash
./bin/anychain-agent \
  --prompt "Create a Solana fake-node smoke benchmark at 1 QPS"
```

Inside the session, `run mock` submits a lifecycle-only Agent job. Real benchmark
execution requires explicit confirmation with `yes run` after reviewing the
plan and runbook. Long sessions can be summarized with `compact`; the Agent
writes structured memory to `.agent/chat/memory.json` while preserving the
current request, plan, job, evidence paths, open questions, and recent turns.
Auto-compaction defaults to a 1,000,000-token context window with a 70% trigger
ratio, configurable through `AGENT_CONTEXT_WINDOW_TOKENS` and
`AGENT_COMPACT_TRIGGER_RATIO`.

Ask the Agent about framework capabilities at any time:

```bash
./bin/anychain-agent --prompt "How many chains and RPC methods are supported?"
./bin/anychain-agent --prompt "How do I add a custom RPC method with three params?"
```

Use LLM-assisted request drafting only after configuring a provider:

```bash
./bin/anychain-agent \
  --prompt "Test my Ethereum node with a weighted mixed workload" \
  --use-llm
```

Advanced subcommands remain available for CI and automation:

```bash
python3 agent/cli.py --help
```

Validate the offline Agent contract when you modify the project:

```bash
python3 -m unittest tests.test_agent_runtime_contract -v
```

## Run A Local Fake-Node Benchmark

Use this when you want to exercise the real benchmark engine locally without a
production node:

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

After the run, open the generated HTML report:

```bash
ls -lt blockchain-node-benchmark-result/current/reports/*.html
```

## Run Against A Real Node

Edit `config/user_config.sh` first:

```bash
BLOCKCHAIN_NODE="solana"
RPC_MODE="single"
LOCAL_RPC_URL="http://your-node-rpc:8899"
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

Then run:

```bash
./blockchain_node_benchmark.sh --quick
```

The most important output files are:

```text
blockchain-node-benchmark-result/current/reports/performance_report_*.html
blockchain-node-benchmark-result/current/logs/proxy_method.csv
blockchain-node-benchmark-result/current/logs/performance_latest.csv
blockchain-node-benchmark-result/archives/<run-id>/test_summary.json
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
