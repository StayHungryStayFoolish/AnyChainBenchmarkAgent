# AnyChain ADK Agent Architecture

AnyChain Agent is a Google ADK-based domain agent that controls the
blockchain-node-benchmark engine. ADK owns reasoning and delegation. The
benchmark framework owns deterministic validation, execution, artifacts, and
evidence.

## Architecture Overview

```mermaid
flowchart TD
  U["User terminal"] --> T["AnyChain product terminal<br/>bin/anychain-agent"]
  T --> D["Startup diagnostics<br/>framework context, environment, dependencies, jobs"]
  T --> R["ADK root coordinator"]

  R --> DS["Discovery Agent"]
  R --> DEP["Dependency Agent"]
  R --> CFG["Configuration Agent"]
  R --> RPC["RPC Workload Agent"]
  R --> ONB["Chain/RPC Onboarding Agent"]
  R --> EXE["Execution Agent"]
  R --> ANA["Resume & Analysis Agent"]
  R --> KB["Knowledge Agent"]

  DS --> TOOLS["ADK function tools"]
  DEP --> TOOLS
  CFG --> TOOLS
  RPC --> TOOLS
  ONB --> TOOLS
  EXE --> TOOLS
  ANA --> TOOLS
  KB --> TOOLS

  TOOLS --> VAL["Deterministic validators<br/>config, workload, onboarding, execution gate"]
  TOOLS --> PLAN["Plan and runtime.env builder"]
  TOOLS --> STATE["Workflow state<br/>.agent/sessions/session/conversation_state.json"]
  TOOLS --> JOB["Detached job manager<br/>.agent/jobs/job_id"]
  TOOLS --> SEARCH["Gemini-only google_search<br/>onboarding/custom RPC evidence"]

  VAL --> PRE["Preflight"]
  PRE --> SMOKE["Fake-node smoke or lifecycle smoke"]
  SMOKE --> APPROVE["User approval callback"]
  APPROVE --> BENCH["Benchmark engine<br/>blockchain_node_benchmark.sh"]
  BENCH --> PROXY["Proxy and per-method attribution"]
  BENCH --> MON["Monitoring system"]
  BENCH --> FN["fake-node fixtures"]
  BENCH --> ART["Reports, charts, archives"]
  ART --> ANA
  ANA --> U
```

## Agent Loop

```mermaid
flowchart LR
  A["Understand<br/>ADK + model"] --> B["Plan<br/>ADK coordinator"]
  B --> C["Ask<br/>one blocking topic"]
  C --> D["Configure<br/>typed tools"]
  D --> E["Validate<br/>deterministic gates"]
  E --> F{"Ready?"}
  F -- "No" --> C
  F -- "Yes" --> G["Execute<br/>smoke or detached job"]
  G --> H["Observe<br/>logs and artifacts"]
  H --> I["Analyze<br/>evidence-backed"]
  I --> J["Iterate<br/>update state or next plan"]
  J --> A
```

The loop prevents the Agent from acting like a keyword bot:

- user intent is interpreted by ADK and the configured model;
- confirmed facts are stored as structured workflow state;
- every execution path passes through deterministic validators;
- the Agent asks for missing information instead of inventing values;
- smoke tests are isolated from final benchmark job artifacts;
- real benchmark jobs require preflight, smoke, and user approval;
- analysis must cite generated evidence paths.

## Accuracy Boundaries

The Agent may infer and suggest values, but it must not silently decide:

- `LEDGER_DEVICE` when multiple disks are plausible;
- whether a separate `ACCOUNTS_DEVICE` exists;
- custom RPC parameter contracts;
- mixed workload weights;
- unsupported chain adapter family;
- real-node endpoint validity before preflight;
- external Prometheus/Grafana scraping behavior.

When uncertain, the Agent must show the available evidence and ask the user to
confirm or provide a value.

## Runtime State And Artifacts

```mermaid
flowchart TD
  C["Conversation state"] --> S[".agent/sessions/session/conversation_state.json"]
  P["Plan"] --> E[".agent/jobs/job_id/runtime.env"]
  J["Job metadata"] --> M[".agent/jobs/job_id/job.json"]
  L["Benchmark logs"] --> LOG[".agent/jobs/job_id/benchmark.log"]
  A["Artifact index"] --> IDX[".agent/jobs/job_id/artifact_index.json"]
  R["Report archive"] --> REP["benchmark-data/archives/run_timestamp"]
```

`runtime.env` is the final per-job confirmed configuration. Users should not
edit it manually. If a user changes an earlier answer, ADK must update or revert
workflow state and regenerate downstream runtime artifacts through tools.

## Google Search Boundary

ADK `google_search` is intentionally narrow:

- enabled only for Gemini with Google authentication and an ADK runtime that
  exposes the tool;
- mounted only on the Chain/RPC Onboarding Agent;
- used for unsupported chain and custom RPC research;
- official documentation is preferred;
- search evidence does not replace endpoint tests, fixture recording, template
  validation, or fake-node smoke.

Other model providers must report web research as unavailable unless the
repository explicitly adds and verifies a provider-specific search integration.

## Development Gates

Before changing Agent code, read:

1. `CLAUDE.md`
2. `docs/zh/anychain-agent-ai-work-gate.md`
3. `agent/README.md`

Then run relevant checks:

```bash
python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract
python3 tools/check_agent_boundaries.py --root .
python3 agent/cli.py adk-eval
git diff --check
```

For model-facing behavior, run the live matrices in `tests/agent_live/`.
