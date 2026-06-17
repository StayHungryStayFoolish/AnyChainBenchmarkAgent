# AnyChain Benchmark Agent

The Agent is a control plane for the existing benchmark engine. It turns user
intent into structured benchmark plans, runs dry-run/preflight checks, submits
background jobs, and analyzes artifacts after execution.

The first implementation is deterministic and does not require an LLM. This is
intentional: CI and local support workflows need the same runtime contract that
the future prompt-first Agent will use.

## UX Principles

- Users describe goals; the Agent translates them into plans.
- The Agent discovers safely before asking.
- Advanced values default conservatively and are shown before execution.
- Host dependency changes require explicit approval.
- Final analysis must cite concrete artifacts.

## Workload Requirements

The Agent must support the same RPC workload model as the benchmark engine:

- single mode with one selected RPC method.
- mixed mode with weighted RPC methods.
- custom RPC methods that use the chain template contracts for `rpc_methods`,
  `param_formats`, REST path bindings, and optional `param_spec`.

Plans must preserve method names, weights, and params so target generation,
fake-node coverage checks, proxy telemetry, per-method success/error counts,
and P50/P90/P99 latency attribution stay aligned.

## Runtime Contract

```text
request -> plan -> preflight -> job -> status -> analysis
```

Phase 2 expands this contract with safe discovery:

```text
prompt/request -> discovery -> plan -> preflight -> job -> status -> analysis
```

Discovery and dependency checks must be read-only by default. The Agent should
prefer Docker or a project-local virtual environment and must not install,
upgrade, or replace system dependencies without explicit user confirmation.

LLM output is never executed directly. When enabled, the model can only produce
an intent classification or a benchmark request draft. The Agent then applies
deterministic normalization, schema checks, preflight checks, command
allowlists, and approval checkpoints before any benchmark command runs.

## Phase 2 Reliability

The Agent must implement these controls before it is considered Phase 2 ready:

- plan confidence and `requires_confirmation`.
- command safety guardrails.
- runbook mode for auditable execution.
- terminal chat entrypoint and interactive wizard.
- plan diff with config snapshots.
- artifact index generation.
- secret redaction.
- machine-readable JSON output.
- result grading: `PASS`, `WARNING`, `FAIL`, `INCONCLUSIVE`.
- experiment history comparison.
- human approval checkpoints for execution, dependency changes, and
  stress/intensive tests.

## Analysis Evidence

Final Agent analysis must include the files and endpoints used as evidence:

- HTML report.
- archive `test_summary.json`.
- proxy/per-method CSV.
- performance CSV.
- sync-health data when relevant.
- job-local `runtime.env`, which records the exact environment values used for
  this run.
- optional Prometheus/Grafana URLs or exporter endpoint.

## Runtime Config Materialization

The Agent does not edit `config/user_config.sh` when it turns QA answers or
discovered values into executable settings. At job submission time it writes a
job-local env file:

```text
.agent/jobs/<job_id>/runtime.env
```

Real benchmark execution loads this file, and the artifact index records it as
evidence. This keeps Agent-generated runtime values reproducible without
polluting the user's default configuration.

## Terminal Entry

The user-facing entrypoint is the terminal Agent:

```bash
./bin/anychain-agent
```

Users can ask framework questions, create plans, run preflight, submit mock
jobs, inspect status, and analyze artifacts in one session:

```text
> doctor
> What chains and RPC methods do you support?
> Create a Solana fake-node smoke benchmark at 1 QPS
> set max qps to 5000
> change mixed weights to getSlot 70%, getBlockHeight 30%
> plan
> preflight
> run mock
> analyze
> compact
> memory
```

`doctor` is a read-only readiness check. It reports dependency gaps,
cloud/deployment hints, LLM/Vertex configuration errors, and live capability
coverage before the user starts a real benchmark.

Long chat sessions use deterministic context compaction. The `compact` command
writes `.agent/chat/memory.json` with the current request, plan, job, evidence
paths, open questions, and recent turns. This keeps the local Agent usable
without an LLM and gives future LLM providers a bounded context contract. The
default window is `1,000,000` estimated tokens with a `0.7` trigger ratio.

The lower-level `agent/cli.py` subcommands are kept for tests, CI, automation,
and advanced scripting.

## CLI Preview

Create a request from a prompt:

```bash
python3 agent/cli.py draft-request \
  --prompt "Test Solana maximum stable QPS on GKE with fake-node smoke first" \
  --output /tmp/agent_request.json
```

Use the configured LLM provider for request drafting, with deterministic
fallback if the model or credentials are unavailable:

```bash
python3 agent/cli.py draft-request \
  --prompt "Test Solana maximum stable QPS on GKE with fake-node smoke first" \
  --use-llm
```

Run the same flow with the offline fake provider for local contract tests:

```bash
python3 agent/cli.py draft-request \
  --prompt "Test Solana maximum stable QPS on GKE with fake-node smoke first" \
  --mock-llm
```

Classify or answer a prompt before planning:

```bash
python3 agent/cli.py route-intent --prompt "How do I configure mixed RPC weights?"
python3 agent/cli.py ask --prompt "How do I use fake-node for local closed-loop testing?"
```

Inspect the Agent's dynamic view of framework capabilities:

```bash
python3 agent/cli.py capabilities
python3 agent/cli.py ask --prompt "How many chains and RPC methods does the framework support?"
```

Capability questions are answered from the current repository state, not from a
static README paragraph. The Agent reads `config/chains/*.json` and
`tools/fake-node/fixtures/` to report supported chains, adapter families,
configured RPC methods, and fake-node fixture coverage.

Analyze support gaps before adding a new chain or RPC method:

```bash
python3 agent/cli.py gap-analysis \
  --chain solana \
  --method getBalance \
  --method customMethod
```

For secondary development, the Agent should produce an onboarding plan,
checklist, template gaps, and validation commands. It should not autonomously
edit chain templates or execute arbitrary generated code just because an LLM
suggested it. Users can review the plan, apply changes, then validate with
fake-node and preflight.

Generate a plan:

```bash
python3 agent/cli.py plan \
  --request /tmp/agent_request.json \
  --output /tmp/agent_plan.json \
  --dry-run
```

Generate a plan with read-only discovery:

```bash
python3 agent/cli.py plan \
  --request /tmp/agent_request.json \
  --output /tmp/agent_plan.json \
  --discover \
  --dry-run
```

Run preflight:

```bash
python3 agent/cli.py preflight --plan /tmp/agent_plan.json
```

Score plan risk:

```bash
python3 agent/cli.py risk-score --plan /tmp/agent_plan.json
```

Run read-only discovery:

```bash
python3 agent/cli.py discover --output /tmp/agent_discovery.json
```

Start the prompt-first wizard:

```bash
python3 agent/cli.py wizard
```

Provide answers non-interactively:

```bash
python3 agent/cli.py wizard \
  --prompt "Run a baseline benchmark on my node" \
  --answers-file /tmp/agent_answers.json \
  --quiet
```

Submit a lifecycle-only mock job:

```bash
python3 agent/cli.py submit --plan /tmp/agent_plan.json --mock
```

Submit a real benchmark job only after approval checkpoints are satisfied:

```bash
python3 agent/cli.py submit --plan /tmp/agent_plan.json --approved
```

Query or analyze:

```bash
python3 agent/cli.py status --job-id <job_id>
python3 agent/cli.py analyze --job-id <job_id>
python3 agent/cli.py artifact-qa --job-id <job_id> --question "Why are charts empty?"
```

Compare plans or archived runs:

```bash
python3 agent/cli.py diff-plan --old /tmp/old_plan.json --new /tmp/new_plan.json
python3 agent/cli.py history --limit 5
python3 agent/cli.py history --compare-latest
```

Run a non-interactive wizard smoke for automation tests:

```bash
python3 agent/cli.py wizard \
  --prompt "Test Solana maximum stable QPS with fake-node smoke first" \
  --output-dir /tmp/agent_wizard \
  --yes \
  --mock
```

## Phase 3 Extension Points

The `knowledge/` package contains the provider contract for future enterprise
knowledge base integrations. Phase 2 keeps this interface stable while the
initial provider remains local and deterministic.

Knowledge providers are capability-based. They can supply chain identification,
RPC method catalogs, parameter samples, response fixtures, workload weights,
chain template fragments, sync-health hints, known bottlenecks, and runtime
notes. Missing high-impact data must still be confirmed with the user.

The built-in framework capability provider is local and deterministic. It is
enough for questions about this repository's supported chains, RPC methods,
adapter families, configuration extension points, and fake-node fixture
coverage. External RAG is only needed when the user wants to integrate
enterprise-private node knowledge, internal RPC samples, incident history, or
unsupported-chain research.

## LLM Providers

The Agent LLM layer uses an internal message contract and provider adapters.
The benchmark engine does not depend on a model provider.

Supported provider contracts:

- `vertex_gemini_openai`: Gemini on Vertex AI through the OpenAI-compatible
  endpoint.
- `vertex_claude`: Claude partner models on Vertex AI.
- `openai`: OpenAI API.

Recommended enterprise configuration uses Google ADC or service-account
impersonation, not static API keys:

```bash
export LLM_PROVIDER=vertex_gemini_openai
export LLM_MODEL=gemini-2.5-pro
export GOOGLE_AUTH_MODE=service_account_impersonation
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_SERVICE_ACCOUNT_EMAIL=benchmark-agent@your-project.iam.gserviceaccount.com
```

`GOOGLE_AUTH_MODE=adc` or `attached_service_account` is preferred on GCE/GKE
when the runtime identity is already bound to the correct service account. Use
`GOOGLE_AUTH_MODE=service_account_file` with `GOOGLE_APPLICATION_CREDENTIALS`
only as a local fallback.

Validate the local configuration without calling a model:

```bash
python3 agent/cli.py llm-config
```

Run an offline LLM smoke test without credentials:

```bash
python3 agent/cli.py llm-smoke --mock
```

Run a real provider smoke only after credentials are configured:

```bash
python3 agent/cli.py llm-smoke \
  --prompt 'Return JSON only: {"ok": true}'
```

The mock smoke validates Agent parsing, routing, fallback, and schema behavior.
It does not prove Vertex or OpenAI credentials are valid; real provider smoke is
the explicit cloud connectivity check.
