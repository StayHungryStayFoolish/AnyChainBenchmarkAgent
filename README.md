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
- Guides missing configuration through Agent checklists instead of expecting
  users to understand every benchmark variable up front.
- Explains report charts and empty-chart causes from artifacts.
- Generates plugin-style onboarding plans for new chains, RPC methods, and
  weighted workloads.

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

## Configuration Model

Most users only need one file before starting the Agent:

```text
config/agent_config.sh
```

Use it to configure the Agent itself: LLM provider, model, Vertex/OpenAI
credentials, context compaction, and optional enterprise Knowledge Base
integration. Every variable has an inline comment.

The benchmark engine still has default settings in:

```text
config/user_config.sh
```

You normally do not need to understand or edit all benchmark variables up
front. Start the Agent, run `doctor`, describe what you want to test, and let
the Agent tell you which required values are missing.

When the Agent submits a job, it writes:

```text
.agent/jobs/<job_id>/runtime.env
```

`runtime.env` is the final configuration snapshot for that one job. It has
higher priority than `config/user_config.sh` during Agent-launched benchmark
runs. Do not edit it by hand; it exists so reports and analysis can prove which
values were used.

## 5-Minute Quick Start

This is the fastest way to use AnyChain Benchmark Agent from a terminal. You can
run the deterministic Agent without an LLM key, but model-assisted planning
requires a provider configuration in `config/agent_config.sh`.

Clone the repository:

```bash
git clone git@github.com:StayHungryStayFoolish/AnyChainBenchmarkAgent.git
cd AnyChainBenchmarkAgent
```

Check dependencies without modifying the host:

```bash
bash scripts/install_deps.sh --check
```

Configure the persistent Agent settings in `config/agent_config.sh`.
For deterministic/offline mode, keep the defaults:

```bash
LLM_PROVIDER="fake"                       # fake | vertex_gemini_openai | vertex_claude | openai
LLM_MODEL="fake"
```

For model-assisted planning, configure one real provider:

```bash
LLM_PROVIDER="vertex_gemini_openai"
LLM_MODEL="gemini-2.5-pro"
GOOGLE_AUTH_MODE="adc"                    # adc | attached_service_account | service_account_impersonation | service_account_file
GOOGLE_CLOUD_PROJECT=""                   # required only when using Vertex with --use-llm
GOOGLE_CLOUD_LOCATION="us-central1"
GOOGLE_SERVICE_ACCOUNT_EMAIL=""           # required for service_account_impersonation
GOOGLE_APPLICATION_CREDENTIALS=""         # optional JSON key fallback
OPENAI_API_KEY=""                         # required only when LLM_PROVIDER=openai
```

You can leave benchmark details such as chain, RPC URL, disk, and machine
metadata unset at first. The Agent will detect what it can and ask for missing
required values before a real run.

Start the Agent terminal session:

```bash
./bin/anychain-agent
```

Then talk to it. Lines prefixed with `anychain>` are messages you type inside
the Agent session; they are not shell commands.

```text
anychain> doctor
# Read-only environment check: dependencies, cloud/deployment hints, LLM config,
# Knowledge Base switch, supported chains/RPC methods, and obvious missing items.

anychain> Create a Solana fake-node smoke benchmark at 1 QPS
# Natural-language goal. The Agent turns it into a request, discovers the
# environment, creates a plan, and records any missing required values.

anychain> plan
# Show the current plan: chain, RPC mode, fake-node/real-node mode, QPS profile,
# command, required inputs, generated files, and next actions.

anychain> preflight
# Validate the plan before execution. This catches missing chain templates,
# missing required values, missing fake-node support, unwritable output dirs,
# and incomplete configuration checklist items.

anychain> run mock
# Validate the Agent job lifecycle only. This creates job metadata, artifact
# index, and runtime.env without running Vegeta traffic.

anychain> status
# Show the latest job state.

anychain> analyze
# Analyze generated artifacts and return PASS/WARNING/FAIL/INCONCLUSIVE with
# evidence paths.

anychain> qa What evidence was generated?
# Ask follow-up questions about report files, CSVs, runtime.env, or missing data.

anychain> qa Why are some charts empty?
# Explain each report chart group, which CSV/fields it needs, and why a chart is
# available, empty, or missing.
```

Use `doctor` first on a new host. It performs read-only readiness diagnostics
for cloud/deployment detection, required dependencies, LLM/Vertex configuration,
and current framework capability coverage.

Use `--use-llm` only after configuring an LLM provider:

```bash
./bin/anychain-agent --use-llm
```

Without `--use-llm`, the Agent still works through deterministic parsing and
repository-aware answers.

You can also run a one-shot prompt:

```bash
./bin/anychain-agent \
  --prompt "Create a Solana fake-node smoke benchmark at 1 QPS"
```

Inside the session, `run mock` submits a lifecycle-only Agent job for local
validation. Real benchmark execution requires explicit confirmation with
`yes run` after reviewing the plan and runbook. Long sessions are summarized
automatically; you can also type `compact` to write `.agent/chat/memory.json`.

Ask the Agent about framework capabilities at any time:

```bash
./bin/anychain-agent --prompt "How many chains and RPC methods are supported?"
./bin/anychain-agent --prompt "How do I add a custom RPC method with three params?"
```

Advanced subcommands remain available for CI and automation:

```bash
python3 agent/cli.py --help
```

Validate the offline Agent contract when you modify the project:

```bash
python3 -m unittest tests.test_agent_runtime_contract -v
```

## Agent Entry Points

Use one recommended user entry:

```bash
./bin/anychain-agent
```

Other entry points exist for automation and advanced users:

- `python3 agent/cli.py ...`: developer, CI, and enterprise platform
  integration subcommands. Use this when another Agent platform wants JSON
  input/output instead of a terminal session.
- `./blockchain_node_benchmark.sh`: low-level execution engine. The Agent calls
  this after plan approval. Direct use is for advanced automation only.

All three paths eventually need runtime values such as chain, RPC URL, QPS
mode, process names, disk baseline, and network bandwidth. The difference is
how missing values are handled:

- `./bin/anychain-agent` detects and asks before execution.
- `agent/cli.py` expects request/plan JSON or scripted inputs.
- `blockchain_node_benchmark.sh` expects configuration to already be present.

## Run A Local Fake-Node Benchmark

Use this when you want the Agent to exercise the benchmark flow without a real
production node. Start `./bin/anychain-agent`, then type:

```text
anychain> doctor
anychain> Create a Solana fake-node smoke benchmark at 1 QPS
anychain> preflight
anychain> run mock
anychain> analyze
```

`run mock` validates the Agent lifecycle without sending benchmark traffic. To
ask the Agent for a real fake-node benchmark plan, type:

```text
anychain> Create a Solana fake-node quick benchmark and run the real benchmark engine
anychain> plan
anychain> preflight
anychain> yes run
```

Review the generated runbook before `yes run`; the Agent will execute only the
allowlisted benchmark command.

## Run Against A Real Node

Edit `config/user_config.sh` first. At minimum, set `BLOCKCHAIN_NODE`,
`RPC_MODE`, `LOCAL_RPC_URL`, `BLOCKCHAIN_PROCESS_NAMES`, cloud/machine metadata,
ledger disk settings, and network bandwidth. Then start the Agent:

```bash
./bin/anychain-agent
```

Example real-node conversation:

```text
anychain> doctor
anychain> Test my Solana node at http://your-node-rpc:8899 with a quick single-method benchmark
anychain> plan
anychain> preflight
anychain> yes run
anychain> analyze
```

The most important output files are:

```text
blockchain-node-benchmark-result/current/reports/performance_report_*.html
blockchain-node-benchmark-result/current/logs/proxy_method.csv
blockchain-node-benchmark-result/current/logs/performance_latest.csv
blockchain-node-benchmark-result/archives/<run-id>/test_summary.json
```

## Required Values And Checklists

The Agent has three levels of configuration checks:

- **Agent checklist**: `config/agent_config.sh` validates LLM provider, model,
  Vertex/OpenAI auth, context compaction, and optional Knowledge Base settings.
- **Benchmark checklist**: `plan` and `preflight` validate required runtime
  values such as chain, RPC mode, local RPC URL for real nodes, process names,
  ledger disk, disk baseline, and network bandwidth.
- **Advanced checklist**: system/internal defaults for monitoring intervals,
  bottleneck thresholds, sync-health thresholds, Prometheus/Grafana, Kubernetes,
  and runtime paths. Most users leave these defaults alone.

The Agent should surface missing required values before `yes run`. Advanced
settings remain available for operators who intentionally tune the framework.

## Report Preview And Chart Explanation

Users can preview the report shape without running a benchmark:

- [English HTML preview](docs/report-previews/report-preview.en.html)
- [English PDF preview](docs/report-previews/report-preview.en.pdf)
- [中文 HTML 预览](docs/report-previews/report-preview.zh.html)
- [中文 PDF 预览](docs/report-previews/report-preview.zh.pdf)

After a real run, ask the Agent chart questions:

```text
anychain> qa Why are charts empty?
anychain> qa Explain the CPU-disk correlation chart
anychain> qa Which files prove per-method latency?
```

The Agent checks registered artifacts and explains performance overview,
CPU-disk correlation, disk thresholds, per-method attribution, sync health, and
monitoring overhead.

## Optional LLM Providers

The Agent works without an LLM. If enabled, LLMs are used only for request
drafting and intent classification, followed by deterministic validation.

Supported provider contracts:

- `vertex_gemini_openai`: Gemini on Vertex AI through the OpenAI-compatible API.
- `vertex_claude`: Claude partner models on Vertex AI.
- `openai`: OpenAI API.
- `fake`: offline protocol smoke provider for tests.

Recommended enterprise Vertex configuration uses ADC or service-account
impersonation instead of static API keys. Configure these values persistently in
`config/agent_config.sh`; `./bin/anychain-agent` loads that file at startup, and
environment variables can still override it for temporary tests.

```bash
LLM_PROVIDER="vertex_gemini_openai"
LLM_MODEL="gemini-2.5-pro"
GOOGLE_AUTH_MODE="service_account_impersonation"
GOOGLE_CLOUD_PROJECT="your-project"
GOOGLE_CLOUD_LOCATION="us-central1"
GOOGLE_SERVICE_ACCOUNT_EMAIL="benchmark-agent@your-project.iam.gserviceaccount.com"
```

For OpenAI:

```bash
LLM_PROVIDER="openai"
LLM_MODEL="gpt-4.1"
OPENAI_API_KEY="sk-..."
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

You can still run the benchmark engine directly. This is mainly for automation,
CI, or advanced users who do not want the Agent chat flow.

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

For fake-node closed-loop testing, prefer the Agent conversation in
[Run A Local Fake-Node Benchmark](#run-a-local-fake-node-benchmark). Direct
fake-node engine commands are documented in
[Local Closed-Loop Testing with fake-node](docs/en/local-closed-loop-testing.md).

For Kubernetes-hosted nodes, deploy the collector first:

```bash
deploy/k8s/validate.sh --preflight
kubectl apply -f deploy/k8s/
kubectl rollout status -n blockchain-bench ds/blockchain-bench-collector
deploy/k8s/validate.sh --post-deploy
```

Then run the benchmark from your selected runner with the same
`config/user_config.sh` settings.

## Enterprise Agent Platform Integration

The project can be embedded into enterprise Agent platforms in two ways:

- **Terminal mode**: run `./bin/anychain-agent` in a controlled shell session.
- **Programmatic mode**: call `python3 agent/cli.py` subcommands and exchange
  JSON for `doctor`, `draft-request`, `plan`, `preflight`, `submit`, `status`,
  `analyze`, `artifact-qa`, and `capabilities`.

For enterprise use, configure `config/agent_config.sh` once in the runtime
image or deployment profile. Keep secrets in the enterprise secret manager and
inject them as environment variables at runtime.

Optional Knowledge Base integration is disabled by default:

```bash
AGENT_KNOWLEDGE_PROVIDER="disabled"       # disabled | noop | custom
AGENT_KNOWLEDGE_PROVIDER_MODULE=""        # example: my_company.anychain_kb:Provider
AGENT_KNOWLEDGE_BASE_URL=""
AGENT_KNOWLEDGE_AUTH_REF=""
```

The built-in Agent already answers from repository state: chain templates,
fake-node fixtures, docs, artifacts, and run history. Enable a custom Knowledge
Base only when an enterprise wants private node samples, internal RPC evidence,
incident history, or company-specific workload guidance.

## Reports And Artifacts

Current-run files are written under the runtime `current/` directory and durable
outputs are archived after the run.

Key artifacts:

- `current/reports/performance_report_*.html`
- `current/logs/proxy_method.csv`
- `current/logs/performance_latest.csv`
- `archives/<run-id>/test_summary.json`
- `.agent/jobs/<job_id>/artifact_index.json` for Agent jobs
- `.agent/jobs/<job_id>/runtime.env` for the Agent-generated final config
  snapshot for that job. Users should not edit this file manually.

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

Generate a plugin-style onboarding package:

```bash
python3 agent/cli.py onboarding-plan \
  --chain foochain \
  --adapter-family jsonrpc \
  --method foo_getBalance \
  --method foo_getBlock
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
