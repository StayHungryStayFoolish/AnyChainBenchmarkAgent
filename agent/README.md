# AnyChain Agent

`agent/` contains the AnyChain Agent control plane for the benchmark engine.
The human-facing product entrypoint is:

```bash
./bin/anychain-agent
```

The product terminal owns stable input/output, language detection, startup
diagnostics, dependency-installation consent, and job recovery commands. The
benchmark workflow itself is owned by the LangGraph Harness in
`agent/harness/`. The Harness's own model calls are plain OpenAI-compatible
HTTP requests for every provider (OpenAI, DeepSeek, and Gemini on Vertex
alike) — Google ADK is used for exactly one optional capability, Gemini
`google_search` grounding (`agent/llm/search_grounding.py`), and must not own
product workflow state or a second conversation loop.

## Development Gate

Before changing Agent code, read the repository-level developer behavior
contract:

```text
../AI_CODING_GUIDE.md
```

Then read the project-specific Agent gate:

```text
../docs/en/anychain-agent-ai-work-gate.md
```

Define assumptions, smallest change scope, success criteria, and verification
commands before editing. Agent fixes must be architectural and test-backed; do
not add phrase filters, keyword routers, fuzzy matching lists, or local patch
logic to hide workflow defects.

Mechanical regressions are checked by:

```bash
python3 tools/check_agent_boundaries.py --root .
```

## Runtime Contract

```text
User message
-> terminal shell for I/O, startup checks, dependency consent, and job commands
-> LangGraph Harness checkpointed workflow
-> LLM intent resolver only for ambiguous natural language
-> deterministic group workflow and validators
-> preflight, smoke, benchmark execution, or job/artifact analysis
```

The Harness is group based, not a fixed wizard. Users may jump between groups,
go back, paste logs, switch chains, switch target mode, change QPS, adjust RPC
methods, or revise previous answers. After any interruption, the Harness
validates current state and resumes from the next blocking group in the default
configuration path.

Core groups:

- `opening`
- `target_mode`
- `chain_identity`
- `provider_deployment`
- `ledger_disk`
- `accounts_disk`
- `network`
- `endpoint_process`
- `chain_auxiliary_endpoints`
- `workload_rpc`
- `target_samples_fixtures`
- `qps_profile`
- `sync_observe`
- `observability`
- `advanced_tuning`
- `preflight_smoke_execution`
- `job_monitoring`
- `error_evidence_analysis`
- `report_artifact_analysis`

LLM output is never executed directly. Repository tools own validation,
configuration materialization, execution, monitoring, and evidence-backed
analysis.

## Main Modules

- `harness/graph.py`: LangGraph runtime and checkpoint wiring.
- `harness/groups.py`: deterministic group workflow, transitions, validation
  gates, and next-blocking-question selection.
- `harness/intent.py`: typed LLM intent and free-form answer resolver.
- `harness/state.py`: product workflow state schema and default group order.
- `harness/nodes/`: LangGraph node boundaries for routing and execution.
- `terminal/repl.py`: product terminal shell and job command integration.
- `terminal/io.py`, `terminal/language.py`, `terminal/job_commands.py`: stable
  terminal support code.
- `llm/search_grounding.py`: the sole `google-adk` consumer — Gemini
  `google_search` grounding, called as a plain function from `harness/groups.py`.
- `tools/executor.py`, `tools/schema.py`: the one programmatic tool-dispatch
  surface for `agent/cli.py tool-call`/`tool-schema` (CI and enterprise
  platform integration), calling the same deterministic modules directly.
- `workflows/group_registry.py`, `workflows/requirements.py`: shared
  requirement metadata still consumed by validators.
- `validators/`, `planners/`, `runners/`, `analyzers/`, `knowledge/`,
  `onboarding/`: deterministic benchmark control-plane modules reused by the
  Harness and tools.

Retired files must not return:

- `agent/workflows/conversation_state.py`
- `agent/workflows/transition_executor.py`
- `agent/terminal/input_classifier.py`
- `agent/terminal/pending_answers.py`
- `agent/adk_app/` (the entire package — an ADK-native `Agent`/`Runner`
  tool-calling surface that duplicated the Harness's conversation loop and
  was never the shipped product's entrypoint; its two genuinely-needed pieces
  moved to `agent/diagnostics/adk_status.py` and
  `agent/terminal/startup_state.py`, and its `google_search` capability moved
  to `agent/llm/search_grounding.py`)

## Boundary Notes

Some files remain for compatibility or shared metadata. They are allowed only
within these boundaries:

- `llm/search_grounding.py`: the only file that may import `google.adk`; it
  must stay a plain function callable from Harness code (no persistent
  Agent/Runner, no second conversation loop, no workflow state mutation).
- `workflows/group_registry.py` and `workflows/requirements.py`: pure metadata
  for validators and Harness checks; no user-text parsing, no state mutation,
  and no terminal rendering. `group_registry.py`'s `GROUP_ORDER` must be
  imported from `harness/state.py::DEFAULT_GROUP_ORDER`, never a separately
  hand-typed copy.

If a future change needs product workflow control, add it to `harness/` or the
deterministic domain modules it invokes. Do not build a second Agent/Runner
tool-calling loop to do it.

## Runtime Use

Install the Agent runtime in the isolated Python environment:

```bash
bash scripts/install_agent_deps.sh --yes
```

For interactive sessions, `prompt-toolkit` is required for reliable Ctrl+C and
Chinese/wide-character editing. If it is missing, the launcher asks for
confirmation and runs `scripts/install_agent_deps.sh --yes`; the Agent does not
fall back to Python `input()`.

Start the product terminal:

```bash
./bin/anychain-agent
```

Stable terminal commands for job recovery and logs:

```text
jobs
status
logs <job_id>
follow <job_id>
```

`follow <job_id>` streams `.agent/jobs/<job_id>/benchmark.log`. Pressing Ctrl+C
exits log-follow mode only; it does not stop the benchmark and does not exit
the Agent. The user can paste copied log snippets back at `User>` for
LLM-assisted analysis through the Harness.

Detached benchmark jobs write:

```text
.agent/jobs/<job_id>/job.json
.agent/jobs/<job_id>/artifact_index.json
.agent/jobs/<job_id>/runtime.env
.agent/jobs/<job_id>/benchmark.log
```

`runtime.env` is the per-job final configuration artifact. Users should not edit
it by hand.

## Verification

Run the core no-credential contract checks:

```bash
python3 tools/check_agent_boundaries.py --root .
python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract tests.test_agent_langgraph_harness
```

Run live CLI matrix tests with configured LLM credentials:

```bash
python3 tests/agent_live/run_langgraph_cli_matrix.py
```

The live matrix drives `./bin/anychain-agent` through the same CLI path users
run and inspects LangGraph checkpoint state. It must not read or create legacy
`.agent/sessions/*/conversation_state.json` workflow files.

## CLI Tools For Automation

`python3 agent/cli.py` exposes JSON commands for CI and enterprise Agent
platforms:

```bash
python3 agent/cli.py adk-status
python3 agent/cli.py capabilities
python3 agent/cli.py doctor --format json
python3 agent/cli.py plan --request request.json --out plan.json
python3 agent/cli.py preflight --plan plan.json
python3 agent/cli.py submit --plan plan.json
python3 agent/cli.py job-status --job-id <job_id>
python3 agent/cli.py analyze --artifacts-dir benchmark-data
```
