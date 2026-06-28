# AnyChain Benchmark Agent

[English](README.md) | [中文](README_ZH.md)

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License](https://img.shields.io/badge/License-Commercial-green.svg)](COMMERCIAL.md)
[![Benchmark Python 3.8+](https://img.shields.io/badge/benchmark_python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ADK Python 3.10+](https://img.shields.io/badge/adk_python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Shell Script](https://img.shields.io/badge/shell-bash-green.svg)](https://www.gnu.org/software/bash/)

A production-oriented benchmark framework for blockchain node QPS, latency,
bottleneck, sync-health, and per-RPC-method analysis, with an ADK-based Agent
control plane.

The benchmark execution plane remains deterministic: Vegeta, RPC proxy,
monitoring collectors, fake-node, report generation, and archiving are still the
source of truth. The intended human-facing entrypoint is `./bin/anychain-agent`.
The Agent must use Google ADK for natural-language understanding, typed intent,
specialized sub-agent delegation, and tool orchestration. Model output may draft
structured requests, plans, and explanations, but deterministic tools and
validators own execution gates.

The benchmark engine is the stable execution layer. The Agent routes
natural-language requests through Google ADK and uses deterministic tools,
validators, preflight checks, smoke tests, and approval gates before execution.

## Contents

1. [Overview](#overview)
2. [Get Started](#get-started)
3. [Use The Agent](#use-the-agent)
4. [Running Benchmarks](#running-benchmarks)
5. [Configuration Reference](#configuration-reference)
6. [Integrations And Operations](#integrations-and-operations)
7. [Extending The Framework](#extending-the-framework)
8. [Reference Documentation](#reference-documentation)
9. [License](#license)

## Overview

### Report Preview

Preview the generated benchmark report before running the framework:

- [Sample English PDF report](docs/en/performance_report_en.pdf)

### What It Does

#### Agent Intelligence

- Turn natural-language benchmark goals into structured, validated plans through
  Google ADK, not terminal keyword matching.
- Detect the local environment and ask only for missing values it cannot safely
  infer.
- Guide missing configuration through Agent checklists instead of expecting
  users to understand every benchmark variable up front.
- Use ADK multi-agent orchestration with deterministic tool and validator gates.
- Score plan risk, run preflight, and require explicit approval before real
  benchmark execution.
- Run long benchmarks in detached/background mode after user approval, so tests
  can continue after the terminal disconnects.
- Restore the latest job context when the Agent is reopened with the same output
  directory.
- Answer framework and result questions from repository state, job artifacts,
  reports, and optional enterprise Knowledge Base integrations.
- Generate onboarding plans and conservative chain-template drafts for new
  chains, RPC methods, and weighted workloads.

#### Benchmark Tools

- Supports 36 chain templates across 6 adapter families.
- Generates single or weighted mixed RPC workloads from `config/chains/*.json`.
- Supports custom RPC methods through `rpc_methods`, `param_formats`, optional
  `param_spec`, REST path bindings, and fake-node fixtures.
- Records per-method status, success/failure counts, and P50/P90/P99 latency.
- Monitors CPU, memory, disk, network, cgroup, sync health, and monitor overhead.
- Produces HTML reports and archives every run.
- Provides optional Prometheus/Grafana telemetry through a read-only exporter.
- Exposes JSON CLI tools, an OpenAI-compatible tool schema, and a stable
  `tool-call` entrypoint for enterprise Agent platforms.

### How The Agent Works

```text
prompt or request
  -> AnyChain terminal shell for stable I/O only
  -> ADK root coordinator
  -> typed intent path
  -> specialized sub-agent delegation
  -> deterministic tool and validator gates
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

## Get Started

Follow these steps in order. Configure the Agent first; benchmark variables can
be inferred and confirmed during the Agent conversation.

### 1. Download The Repository

```bash
git clone git@github.com:StayHungryStayFoolish/AnyChainBenchmarkAgent.git
cd AnyChainBenchmarkAgent
```

### 2. Install The Agent Runtime

Install the isolated ADK terminal runtime first. This does not install ADK into
the Python environment used by production blockchain nodes:

```bash
bash scripts/install_agent_deps.sh --yes
```

If you skip this step and start the interactive Agent anyway, the launcher
checks the required terminal dependency before entering the REPL. When
`prompt-toolkit` is missing, it asks for confirmation and then runs
`scripts/install_agent_deps.sh --yes` for you. It does not silently fall back to
Python `input()`, because reliable Ctrl+C and Chinese/wide-character editing
are baseline Agent terminal requirements.

If your host does not provide `python3.11`, use any Python 3.10+ interpreter.
The benchmark engine still supports older Python for non-Agent automation, but
the Agent requires a Python 3.10+ ADK runtime environment for model-backed
sessions. The product launcher uses its own terminal workflow and does not ask
users to run `adk run` directly.

Do not start by installing benchmark-engine dependencies manually. In the
normal Agent flow, users install the Agent runtime once, configure the LLM, then
let the Agent inspect benchmark dependencies and ask for approval before it
calls `scripts/install_deps.sh --yes` through the confirmation-gated
`install_dependencies` tool. Direct `scripts/install_deps.sh` usage is kept for
CI, Docker images, and non-Agent automation.

### 3. Configure The Agent

Choose one of two setup paths.

#### Option A: AI-Assisted Setup

If you want another AI assistant to configure or operate this project for you,
give it these files first:

```text
AGENTS.md
README.md
config/agent_config.sh
agent/README.md
docs/en/anychain-agent-ai-work-gate.md
```

`AGENTS.md` is the short handoff guide for AI assistants. It explains how to
configure provider credentials safely, where local secrets belong, which
commands validate the Agent, and what an assistant must not bypass. Real API
keys, ADC settings, Vertex settings, and other local provider choices should go
into `config/agent_config.local.sh`, which is gitignored.

For code changes, the assistant must also read `AI_CODING_GUIDE.md` before
editing files. For user-only setup, `AGENTS.md` plus this README is usually
enough to get started.

#### Option B: Manual Setup

If you prefer to configure the project yourself, start with:

```text
config/agent_config.sh
```

Use it to configure the Agent itself: LLM provider, model, Vertex/OpenAI
credentials, context compaction, and optional enterprise Knowledge Base
integration. Every variable has an inline comment.

Write real secrets and local provider choices to:

```text
config/agent_config.local.sh
```

That file is gitignored. Another AI assistant helping a user should put API
keys, ADC/Vertex local settings, and local provider choices there instead of
committing them to the repository defaults.

Configure one real provider before starting a natural-language session. Direct
API-key mode works for Gemini, `claude`, OpenAI, and DeepSeek. Google
service-account modes work for Gemini or `claude` through Vertex AI.

```bash
LLM_PROVIDER="gemini"
LLM_MODEL="gemini-3.1-pro"
LLM_AUTH_MODE="api_key"                   # api_key | google_adc | attached_service_account | service_account_impersonation | service_account_file
GEMINI_API_KEY=""                         # or GOOGLE_API_KEY, required for Gemini API-key mode
ANTHROPIC_API_KEY=""                      # required for `claude` API-key mode
OPENAI_API_KEY=""                         # required for OpenAI
DEEPSEEK_API_KEY=""                       # required for DeepSeek
GOOGLE_CLOUD_PROJECT=""                   # required only for Google service-account modes
GOOGLE_CLOUD_LOCATION="global"           # Vertex AI location/region
GOOGLE_SERVICE_ACCOUNT_EMAIL=""           # required for service_account_impersonation
GOOGLE_APPLICATION_CREDENTIALS=""         # required only for service_account_file
```

Choose one authentication path:

- Gemini API key: set `LLM_PROVIDER=gemini`, `LLM_AUTH_MODE=api_key`, and
  `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- `claude` API key: set `LLM_PROVIDER=claude`, `LLM_AUTH_MODE=api_key`, and
  `ANTHROPIC_API_KEY`.
- OpenAI API key: set `LLM_PROVIDER=openai`, `LLM_AUTH_MODE=api_key`, and
  `OPENAI_API_KEY`.
- DeepSeek API key: set `LLM_PROVIDER=deepseek`, `LLM_AUTH_MODE=api_key`, and
  `DEEPSEEK_API_KEY`.
- Google Vertex AI: set `LLM_PROVIDER=gemini` or `LLM_PROVIDER=claude`, set
  `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`, then choose
  `google_adc`, `attached_service_account`, `service_account_impersonation`, or
  `service_account_file`.

Web research is provider-limited. ADK `google_search` is enabled only when the
Agent is running with a Gemini-family model and valid Gemini/Google
authentication. `claude` on Vertex, DeepSeek, OpenAI, and `claude` API-key modes
do not enable ADK `google_search`; in those modes the Agent uses repository
facts and optional enterprise KB evidence, or asks you to provide official docs
and request/response samples.

Google Cloud CLI is needed only for local ADC workflows such as
`LLM_AUTH_MODE=google_adc`, or when a host must create ADC before
service-account impersonation. The Agent can detect whether `gcloud` and the
local ADC file are present through `doctor`; after explicit approval it can
install Google Cloud CLI with:

```bash
bash scripts/install_agent_deps.sh --yes --with-gcloud
```

After `gcloud` is available, create local ADC credentials when that auth mode
is used:

```bash
gcloud auth application-default login
```

On GCE/GKE/Cloud Run with an attached service account, `gcloud` is not required
for runtime auth if the workload identity already has Vertex AI access.

#### Benchmark Configuration Model

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

Agent-launched jobs default to `.agent/jobs`. Low-level
`python3 agent/cli.py submit` uses the same location unless `--jobs-dir` is
provided.

### 4. Validate The Agent Configuration

Validate the Agent and LLM configuration before starting an interactive
session:

```bash
python3 agent/cli.py adk-status
python3 agent/cli.py llm-config
```

You can leave benchmark details such as chain, RPC URL, disk, and machine
metadata unset at first. The Agent will detect what it can and ask for missing
required values before a real run.

## Use The Agent

### 5. Start The Agent

Start the Agent. This command opens the AnyChain product terminal. Google ADK
owns natural-language understanding and orchestration, while the terminal owns
stable input/output and startup checks:

```bash
./bin/anychain-agent
```

Then talk to it from the `User>` prompt. The Agent replies as `Agent>`, keeps
the response language aligned with the user's input, stores job/session
artifacts under `.agent`, and asks for explicit approval before installing
dependencies, running smoke, or launching a benchmark job.

```text
User> doctor
Agent> ...summarizes dependencies and asks before installing anything...

User> I want to benchmark a Solana node.
Agent> ...asks whether this is a fake-node closed-loop test or a real node...

User> 1
Agent> ...confirms host, disk, network, RPC mode, workload, and optional
       observability settings one item at a time...

User> Use mixed workload with getSlot 70% and getBlockHeight 30%.
Agent> ...generates a benchmark plan, runs preflight, then asks before smoke
       and again before launching the benchmark...
```

Use a readiness check first on a new host. The Agent has a read-only doctor tool
for cloud/deployment detection, dependencies, LLM/Vertex configuration,
Knowledge Base configuration, and current framework capability coverage.
If benchmark dependencies are missing, the Agent should explain the planned
changes and ask for explicit approval before installing them.

Real benchmark execution still uses confirmation-gated tools and must pass
preflight and smoke before launch. The model output is never executed directly.

Default real benchmark execution is detached/background in the lower-level job
runner. The benchmark worker continues after the Agent terminal disconnects,
writes `.agent/jobs/<job_id>/job.json`, and streams benchmark output to
`.agent/jobs/<job_id>/benchmark.log` unless a custom jobs directory is passed to
the lower-level CLI.

To resume after disconnecting, start the Agent again from the same repository.
The Agent tools can inspect `.agent/jobs` and recover the latest job state:

```bash
./bin/anychain-agent
```

After restart, the Agent reports the latest job it finds. You can type `status`
or `jobs` to recover job state, runtime.env, and artifact paths.

Ask the Agent about framework capabilities at any time:

```text
User> How many chains and RPC methods are supported?
User> How do I add a custom RPC method with three params?
```

## Running Benchmarks

### Run A Local Fake-Node Benchmark

Use this when you want to validate the full benchmark flow without a production
node. Start `./bin/anychain-agent` and describe the goal in normal language,
for example:

```text
I want to run a Solana fake-node benchmark.
```

The Agent should then walk through the same production-grade checks used for a
real node: chain, RPC mode, workload, custom RPC methods, weighted mixed
methods, local host resources, disks, optional Prometheus/Grafana, preflight,
smoke, and final approval. The fake-node path only avoids requiring a real
`LOCAL_RPC_URL`; it does not skip benchmark configuration validation.

### Run Against A Real Node

Do not start by editing every benchmark variable by hand. Start the Agent, let
it inspect the host, then answer only the missing values it cannot infer. The
Agent writes the confirmed values to the job-local `runtime.env`; ordinary users
do not edit that file.

```bash
./bin/anychain-agent
```

Then describe the goal:

```text
I want to run a quick Solana benchmark against my real node.
```

The Agent should infer what it can, ask one item at a time for values it cannot
prove, and show choices when discovery is ambiguous. Real-node runs require a
confirmed `LOCAL_RPC_URL`; `MAINNET_RPC_URL` is used when the selected chain
sync-health strategy needs a separate reference endpoint. If discovery cannot
safely identify a value, the Agent must ask instead of guessing.

The most important output files are:

```text
blockchain-node-benchmark-result/current/reports/performance_report_*.html
blockchain-node-benchmark-result/current/logs/proxy_method.csv
blockchain-node-benchmark-result/current/logs/performance_latest.csv
blockchain-node-benchmark-result/archives/<run-id>/test_summary.json
```

## Configuration Reference

### Required Values

The Agent checks these configuration layers before submitting a benchmark:

- **Agent configuration**: `config/agent_config.sh` and optional
  gitignored `config/agent_config.local.sh` define the LLM provider, model,
  authentication, context settings, and optional Knowledge Base integration.
- **Benchmark runtime configuration**: the Agent confirms chain, node type,
  RPC mode, RPC methods and weights, real-node RPC URLs, node process names,
  ledger/data disks, disk baseline, network interface, network bandwidth, and
  output paths.
- **Advanced defaults**: `config/internal_config.sh` and related config files
  hold monitoring intervals, bottleneck thresholds, sync-health thresholds,
  Prometheus/Grafana defaults, Kubernetes paths, and runtime paths. Most users
  leave these defaults alone unless the Agent or an operator has a reason to
  change them.

Planning and preflight surface missing required values before real submission.
Agent-confirmed values are written to the job-local `runtime.env`, which takes
priority for that one job.

### LLM Providers

Configure the Agent model in step 3 under
[Configure The Agent](#3-configure-the-agent). The supported provider families
are:

- `gemini`: Gemini API key, or Gemini on Vertex AI with Google auth.
- `claude`: Anthropic API key, or `claude` on Vertex AI with Google auth.
- `openai`: OpenAI API key.
- `deepseek`: DeepSeek API key through the OpenAI-compatible endpoint.

Validate the config without calling a model:

```bash
python3 agent/cli.py llm-config
```

Run a real provider smoke only after credentials are configured:

```bash
python3 agent/cli.py llm-smoke --prompt 'Return JSON only: {"ok": true}'
```

## Integrations And Operations

### Enterprise Agent Platform Integration

The project can be embedded into enterprise Agent platforms in several ways:

- **Terminal mode**: run `./bin/anychain-agent` in a controlled shell session.
- **Programmatic mode**: call `python3 agent/cli.py` subcommands and exchange
  JSON for `doctor`, `plan`, `preflight`, `submit`, `status`, `analyze`,
  `artifact-qa`, and `capabilities`.
- **Tool-schema mode**: call `python3 agent/cli.py tool-schema` to export an
  OpenAI-compatible function-tool schema for enterprise Agent orchestrators.
- **Tool-call mode**: call `python3 agent/cli.py tool-call --name <tool> --arguments '<json>'`
  when a platform wants one stable command to execute a named Agent tool.

For enterprise use, configure `config/agent_config.sh` once in the runtime
image or deployment profile. Keep secrets in the enterprise secret manager and
inject them as environment variables at runtime.

Optional Knowledge Base integration is disabled by default:

```bash
AGENT_KNOWLEDGE_PROVIDER="disabled"       # disabled | noop | http | custom
AGENT_KNOWLEDGE_PROVIDER_MODULE=""        # example: my_company.anychain_kb:Provider
AGENT_KNOWLEDGE_BASE_URL=""               # required when provider=http
AGENT_KNOWLEDGE_AUTH_REF=""
```

The built-in Agent already answers from repository state: chain templates,
fake-node fixtures, docs, artifacts, and run history. Enable a custom Knowledge
Base only when an enterprise wants private node samples, internal RPC evidence,
incident history, or company-specific workload guidance.

For a generic HTTP KB/RAG service, configure `AGENT_KNOWLEDGE_PROVIDER=http`.
Validate the adapter with:

```bash
python3 agent/cli.py knowledge-smoke --query "solana rpc methods" --chain solana
```

Enterprise Agent platforms can inspect and call the tool catalog directly:

```bash
python3 agent/cli.py --help
python3 agent/cli.py tool-schema
python3 agent/cli.py tool-call --name load_capabilities
python3 agent/cli.py tool-call --name draft_request \
  --arguments '{"chain":"solana","goal":"smoke","rpc_mode":"single","use_fake_node":true,"qps_max":1,"source_prompt":"Create a Solana fake-node smoke benchmark at 1 QPS"}'
```

For deterministic CI that does not require a conversational terminal, use the
JSON control-plane commands:

```bash
python3 agent/cli.py plan --request /tmp/request.json --output /tmp/plan.json --dry-run
python3 agent/cli.py preflight --plan /tmp/plan.json
python3 agent/cli.py submit --plan /tmp/plan.json --mock
```

### Reports And Artifacts

Current-run files are written under the runtime `current/` directory and durable
outputs are archived after the run.

Sample report previews:
[English PDF](docs/en/performance_report_en.pdf) |
[Chinese PDF](docs/zh/performance_report_zh.pdf)

Key artifacts:

- `current/reports/performance_report_*.html`
- `current/logs/proxy_method.csv`
- `current/logs/performance_latest.csv`
- `archives/<run-id>/test_summary.json`
- `<agent-output-dir>/jobs/<job_id>/artifact_index.json` for Agent jobs
- `<agent-output-dir>/jobs/<job_id>/runtime.env` for the Agent-generated final config
  snapshot for that job. Users should not edit this file manually.

For low-level CLI job commands, job state defaults to `.agent/jobs` unless
`--jobs-dir` points elsewhere.

Agent job helpers:

```bash
python3 agent/cli.py jobs
python3 agent/cli.py resume --job-id <job_id>
python3 agent/cli.py logs --job-id <job_id>
python3 agent/cli.py diagnose-artifacts --artifact-index <agent-output-dir>/jobs/<job_id>/artifact_index.json
```

`diagnose-artifacts` applies deterministic bottleneck rules to available CSVs:
CPU saturation, disk latency/queueing, disk IOPS or throughput pressure, RPC
method errors/latency, and sync-health warnings.

If a terminal session disconnects, start `./bin/anychain-agent` again from the
same repository. Detached jobs continue in their worker process; use the Agent
or the low-level `jobs`, `status`, `logs`, `resume`, and `analyze` commands to
inspect progress or completed evidence.

Optional job notifications are disabled by default. Set
`AGENT_NOTIFY_WEBHOOK_URL` and `AGENT_NOTIFY_ON` in `config/agent_config.sh`
when an enterprise Agent platform wants webhook events for long-running jobs.

### Optional Prometheus/Grafana

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

## Extending The Framework

### Extending Chains Or RPC Methods

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

Generate a conservative chain template draft for human review:

```bash
python3 agent/cli.py draft-chain-template \
  --chain foochain \
  --adapter-family jsonrpc \
  --method foo_getBalance \
  --method foo_getTransaction \
  --output /tmp/foochain.json
```

The draft is marked `needs_review`; it is not automatically installed into
`config/chains`.

For unsupported chains, the Agent returns an onboarding plan instead of editing
code automatically. The usual path is:

1. Add `config/chains/<chain>.json` from the template.
2. Select `_meta.adapter_family`.
3. Configure `rpc_methods.single` and `rpc_methods.mixed_weighted`.
4. Add `param_formats` or `param_spec`.
5. Add `proxy_extraction` rules.
6. Record fake-node fixtures.
7. Run preflight and fake-node closed-loop tests.

When Gemini web research is enabled, the Chain/RPC onboarding flow may use ADK
`google_search` to find official RPC documentation, node operator docs, and
official API examples for unsupported chains or custom RPC methods. Search
results are evidence only: they do not skip endpoint confirmation, real
request/response samples, fixture recording, template validation, or fake-node
smoke.

## Reference Documentation

- [AI Assistant Operator Guide](AGENTS.md)
- [Configuration Guide](config/README.md)
- [Agent Control Plane](agent/README.md)
- [ADK Agent Architecture](docs/en/adk-agent-architecture.md)
- [AnyChain Agent AI Work Gate](docs/en/anychain-agent-ai-work-gate.md)
- [Full Framework Reference](docs/en/framework-reference.md)
- [Framework Flow and Data Lifecycle](docs/en/framework-flow.md)
- [Module Guide](docs/en/module-guide.md)
- [How to Add a Chain or RPC Method](docs/en/how-to-add-chain.md)
- [Local Closed-Loop Testing with fake-node](docs/en/local-closed-loop-testing.md)
- [Secondary Development Guide](docs/en/secondary-development-guide.md)
- [GitHub PR Gates and Branch Protection](docs/en/github-pr-gates.md)
- [GitHub PR Workflow](docs/en/github-pr-workflow.md)
- [Prometheus / Grafana Observability](deploy/observability/README.md)
- [Kubernetes Collector](deploy/k8s/README.md)

## License

This project is dual licensed:

- AGPL-3.0-or-later for open-source use. See [LICENSE](LICENSE).
- Commercial licensing for proprietary/internal use cases. See [COMMERCIAL.md](COMMERCIAL.md).
