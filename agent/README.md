# AnyChain Agent

`agent/` contains the AnyChain Agent control plane for the benchmark engine.
The human-facing product entrypoint is:

```bash
./bin/anychain-agent
```

The product terminal owns stable input/output, language detection, startup
diagnostics, dependency-installation consent, and job recovery commands. The
benchmark workflow itself is owned by the LangGraph Harness in
`agent/harness/`. Google ADK remains available as the model/tool surface for
Gemini and Google capabilities, but it must not own product workflow state.

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
- `adk_app/root_agent.py`: ADK model/tool bridge only.
- `adk_app/tools/`: ADK function-tool wrappers around deterministic modules.
- `workflows/group_registry.py`, `workflows/requirements.py`: shared
  requirement metadata still consumed by validators.
- `validators/`, `planners/`, `runners/`, `analyzers/`, `knowledge/`,
  `onboarding/`: deterministic benchmark control-plane modules reused by the
  Harness and tools.

Retired files must not return:

- `agent/workflows/conversation_state.py`
- `agent/workflows/transition_executor.py`
- `agent/adk_app/callbacks.py`
- `agent/adk_app/agents/domain.py`
- `agent/adk_app/tools/workflow_state.py`
- `agent/adk_app/workflow/product_context.py`
- `agent/terminal/input_classifier.py`
- `agent/terminal/pending_answers.py`

## Boundary Notes

Some files remain for compatibility, diagnostics, or shared metadata. They are
allowed only within these boundaries:

- `adk_app/agent.py`: ADK discovery wrapper; it must only expose `root_agent`.
- `adk_app/runtime.py`: developer diagnostic bridge for official `adk run`;
  it is not the product terminal and must not process user benchmark turns.
- `adk_app/runner_bridge.py`: ADK runner availability checks only; no model
  calls, terminal text rewriting, or workflow state mutation.
- `adk_app/workflow/native_smoke.py`: credential-free ADK runtime smoke only;
  it is not product acceptance for multi-turn terminal behavior.
- `adk_app/workflow/schemas.py`: eval/schema diagnostics only; it must not
  define or execute the product workflow state machine.
- `workflows/group_registry.py` and `workflows/requirements.py`: pure metadata
  for validators and Harness checks; no user-text parsing, no state mutation,
  and no terminal rendering.

If a future change needs product workflow control, add it to `harness/` or the
deterministic domain modules it invokes. Do not route product behavior through
ADK compatibility files or terminal helpers.

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
python3 agent/cli.py adk-eval
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
python3 agent/cli.py adk-eval
python3 agent/cli.py capabilities
python3 agent/cli.py doctor --format json
python3 agent/cli.py plan --request request.json --out plan.json
python3 agent/cli.py preflight --plan plan.json
python3 agent/cli.py submit --plan plan.json
python3 agent/cli.py job-status --job-id <job_id>
python3 agent/cli.py analyze --artifacts-dir benchmark-data
```
