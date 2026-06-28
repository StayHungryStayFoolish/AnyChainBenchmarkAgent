# AnyChain ADK Agent

`agent/` contains the AnyChain Agent control plane for the benchmark engine.
The human-facing product entrypoint is:

```bash
./bin/anychain-agent
```

The product terminal owns prompt input, language consistency, startup
diagnostics, dependency-installation consent, and job recovery notices. Google
ADK must own Agent orchestration, session reasoning, natural-language intent
recognition, typed intent routing, and multi-agent delegation. Reusable
benchmark logic remains available as deterministic modules and ADK function
tools.

The accepted architecture is defined by the ADK app code and runtime contract
tests:

- `adk_app/instructions.py`
- `adk_app/agents/domain.py`
- `adk_app/tools/`
- `../tests/test_agent_product_terminal.py`
- `../tests/test_agent_runtime_contract.py`

## Development Entry Contract

Before changing Agent code, read the repository-level developer behavior
contract:

```text
../AI_CODING_GUIDE.md
```

Then read the project-specific Agent gate:

```text
../docs/en/anychain-agent-ai-work-gate.md
```

After that, define the assumption, smallest change scope, success criteria, and
verification commands before editing.

Mechanical Agent regressions are checked by:

```bash
python3 tools/check_agent_boundaries.py --root .
```

Agent changes must preserve the AnyChain Agent Loop:

```text
Understand -> Plan -> Ask -> Configure -> Validate -> Execute -> Observe -> Analyze -> Iterate
```

ADK and the configured model own Understand, Plan, Ask, and Iterate. Deterministic
repository tools own Configure, Validate, Execute, Observe, and evidence-backed
Analyze. Validators and callbacks are mandatory checkpoints between phases.
Terminal code must not replace the loop with keyword routing, fuzzy matching, or
field-by-field wizard logic.

Workflow progress is persisted separately from terminal prompt state:

```text
.agent/sessions/<session_id>/conversation_state.json
```

ADK updates this file through structured workflow-state tools after it has
understood the user's intent and extracted explicit fields. The state file is a
checkpoint artifact for chain, target mode, RPC mode, confirmed environment
values, missing fields, pending question, plan file, and latest job. It is not a
natural-language parser and must not become one.
The state keeps a bounded revision history so ADK can handle "go back" or
"that value was wrong" by reverting or patching structured fields, then running
validators again.

## Runtime Contract

```text
user message
-> AnyChain terminal shell for stable I/O only
-> ADK root coordinator
-> typed intent path
-> specialized sub-agent
-> deterministic tool and validator gates
-> preflight, runbook, and user confirmation
-> lifecycle smoke or isolated fake-node benchmark smoke
-> confirmation-gated benchmark job
-> artifact index
-> evidence-based analysis
```

Benchmark execution still uses the repository's deterministic tools, validator
gates, and approval callbacks. LLM output is never executed directly.

Before smoke or benchmark execution, ADK must confirm:

- target mode: fake-node or real-node;
- chain template and chain-template endpoint/sample variables;
- RPC mode: single or mixed;
- benchmark mode: quick, standard, or intensive;
- QPS profile for the selected mode: initial QPS, max QPS, step, and duration;
- observability mode: disabled, local Prometheus/Grafana, or exporter-only for
  an existing Prometheus/Grafana environment;
- runtime resource metadata from `config/user_config.sh`, including
  `BLOCKCHAIN_PROCESS_NAMES`, `LEDGER_DEVICE`, optional `ACCOUNTS_DEVICE`,
  disk baselines, `NETWORK_INTERFACE`, and network bandwidth.

Every inferred value must allow manual override. When multiple disks are
detected, ADK must show the disk inventory and ask which device is
`LEDGER_DEVICE` and whether a separate accounts/state disk exists. Fake-node
mode skips real RPC URLs, but it does not skip resource metadata.

For QPS profile confirmation, ADK must show the framework default values for
the selected mode and explain each parameter briefly. It should first ask
whether to keep the defaults. Only when the user wants changes should it ask
which single item to adjust, then show the revised profile and ask for
confirmation again.

The terminal must not become a natural-language business intent router. It may
handle stable shell commands, dependency-installation consent, and job recovery
notices, but benchmark, onboarding, custom-RPC, analysis, and observability
requests must flow through ADK and repository tools.

## Main Modules

- `adk_app/instructions.py`: root and specialized sub-agent instructions.
- `adk_app/root_agent.py`: ADK root coordinator construction.
- `adk_app/agents/domain.py`: specialized ADK sub-agents for
  discovery, dependency, configuration, RPC workload, onboarding, execution,
  resume/analyze, and knowledge.
- `adk_app/agent.py`: official ADK discovery module exposing `root_agent`.
- `adk_app/runtime.py`: development bridge for explicit `adk run` diagnostics.
- `adk_app/tools/`: ADK function-tool wrappers around deterministic modules.
- `adk_app/evals/`: no-key ADK package and tool-contract checks.
- `terminal/`: product terminal, prompt input, language policy, startup
  diagnostics, and ADK Runner bridge.
- `validators/`: deterministic configuration, workload, onboarding, and
  execution gates used by ADK tools.
- `workflows/conversation_state.py`: file-backed workflow checkpoint state used
  by ADK tools and injected into each terminal turn.
- `workflows/requirements.py`: shared requirement matrices consumed by
  validators.
- `planners/`, `runners/`, `analyzers/`, `knowledge/`, `onboarding/`: benchmark
  engine control-plane tools reused by ADK.

## ADK Runtime Use

Install the ADK runtime in an isolated Python 3.10+ environment:

```bash
bash scripts/install_agent_deps.sh --yes
```

`./bin/anychain-agent` starts the AnyChain product terminal. Users do not need
to run `adk run` directly.

For interactive sessions, `prompt-toolkit` is a required terminal dependency.
If it is missing, the launcher asks for confirmation and runs
`scripts/install_agent_deps.sh --yes` before entering the REPL. The Agent does
not fall back to Python `input()`, because reliable Ctrl+C and wide-character
editing are part of the product terminal contract.

Then start the human-facing Agent:

```bash
./bin/anychain-agent
```

The terminal must pass natural-language conversation to the ADK root
coordinator. ADK delegates to specialized sub-agents and calls deterministic
tools for discovery, configuration validation, plan generation, preflight,
smoke, execution, and analysis.

```text
User> I want to benchmark Solana
Agent> ...asks fake-node or real-node...
User> fake-node
Agent> ...asks RPC mode, workload, and parameter-sample confirmation...
```

Development-only ADK CLI checks remain available through
`agent/adk_app/runtime.py` and `python3 agent/cli.py adk-status`, but they are
not the product UX.

Detached benchmark jobs write:

```text
.agent/jobs/<job_id>/job.json
.agent/jobs/<job_id>/artifact_index.json
.agent/jobs/<job_id>/runtime.env
.agent/jobs/<job_id>/benchmark.log
```

`runtime.env` is the per-job final configuration artifact. Users should not edit
it by hand.

Stable terminal commands for job recovery and logs:

```text
jobs
status
logs <job_id>
follow <job_id>
```

`follow <job_id>` streams `.agent/jobs/<job_id>/benchmark.log`. Pressing
Ctrl+C exits log-follow mode only; it does not stop the benchmark and does not
exit the Agent. The user can then paste a copied log snippet back at `User>` for
ADK-based analysis.

## CLI Tools For Automation

`python3 agent/cli.py` exposes JSON commands for CI and enterprise Agent
platforms:

```bash
python3 agent/cli.py adk-status
python3 agent/cli.py adk-eval
python3 agent/cli.py capabilities
python3 agent/cli.py tool-call --name load_framework_context --arguments '{"language":"en"}'
python3 agent/cli.py tool-call --name load_execution_contract --arguments '{"use_fake_node":true}'
python3 agent/cli.py tool-call --name prepare_benchmark_run --arguments '{"chain":"solana","goal":"smoke","rpc_mode":"single","use_fake_node":true}'
python3 agent/cli.py plan --request /tmp/request.json --output /tmp/plan.json --dry-run
python3 agent/cli.py preflight --plan /tmp/plan.json
python3 agent/cli.py submit --plan /tmp/plan.json --mock
python3 agent/cli.py status --job-id <job_id>
python3 agent/cli.py analyze --job-id <job_id>
python3 agent/cli.py tool-schema
python3 agent/cli.py tool-call --name load_capabilities
```

Real benchmark submission is confirmation-gated:

```bash
python3 agent/cli.py submit --plan /tmp/plan.json --approved
```

## LLM And Google Auth

Persistent Agent defaults live in:

```text
config/agent_config.sh
```

The human-facing Agent model is executed by ADK. These provider settings are
used to resolve the configured model name and to provide safe setup diagnostics;
direct provider adapters are dev/test diagnostics, not the ADK runtime.

Supported provider modes:

- `gemini`: Gemini API key, Google ADC, attached service account,
  service-account impersonation, or JSON key file.
- `claude`: Anthropic API key or `claude` on Vertex AI with Google auth.
- `openai`: OpenAI API key.
- `deepseek`: DeepSeek API key through the OpenAI-compatible endpoint.

At minimum, configure `LLM_PROVIDER`, `LLM_MODEL`, `LLM_AUTH_MODE`, and the
matching provider credential. For Vertex AI, also configure
`GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.

ADK `google_search` is intentionally Gemini-only. It is exposed only when
`LLM_PROVIDER=gemini`, the model name is Gemini-family, Gemini/Google auth
validates, and the installed ADK runtime can import `google_search`. `claude` on
Vertex, DeepSeek, OpenAI, and `claude` API-key modes must report web research as
unavailable. The tool is mounted only on the Chain/RPC onboarding agent and is
used as evidence gathering for unsupported chains or custom RPC methods, never
as support approval.

Google Cloud CLI is needed only for local ADC workflows such as
`LLM_AUTH_MODE=google_adc`, or when the host must create ADC before
service-account impersonation. The ADK Agent can inspect this through
`doctor` / `inspect_llm_auth`, and can install it after explicit approval:

```bash
bash scripts/install_agent_deps.sh --yes --with-gcloud
```

Validate configuration without calling a model:

```bash
python3 agent/cli.py adk-status
python3 agent/cli.py llm-config
```

Run `python3 agent/cli.py adk-eval` for no-key ADK contract checks. It does
not simulate prompt understanding; real natural-language behavior requires ADK
with a configured model provider.

## Knowledge Base And Enterprise Integration

The local repository capability provider is enough for questions about current
chains, RPC methods, fake-node fixtures, and generated artifacts. Optional
enterprise Knowledge Base integration is configured through
`config/agent_config.sh`.

Validate a configured provider:

```bash
python3 agent/cli.py knowledge-smoke --query "solana rpc methods" --chain solana
```

Enterprise platforms should prefer:

```bash
python3 agent/cli.py tool-schema
python3 agent/cli.py tool-call --name <tool> --arguments '<json>'
```

## Development Checks

Run the no-key ADK contract tests:

```bash
python3 -m unittest tests.test_agent_runtime_contract -v
python3 agent/cli.py adk-eval
```

Before adding a new chain or RPC method, generate an onboarding package and a
draft template, then validate chain templates, fake-node fixtures, target
generation, and smoke execution.
