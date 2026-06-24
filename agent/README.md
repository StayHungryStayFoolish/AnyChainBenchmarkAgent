# AnyChain ADK Agent

`agent/` contains the AnyChain Agent control plane for the benchmark engine.
The human-facing product entrypoint is:

```bash
./bin/anychain-agent
```

The product terminal owns prompt input, language consistency, one-question
confirmation flow, workflow state, and job recovery. Google ADK remains the
Agent runtime layer underneath; reusable benchmark logic remains available as
deterministic modules and ADK function tools.

## Runtime Contract

```text
user message
-> AnyChain terminal workflow state
-> ADK instruction and model reasoning when configured
-> prepare_benchmark_run
-> preflight, runbook, and user confirmation
-> lifecycle smoke or isolated fake-node benchmark smoke
-> confirmation-gated benchmark job
-> artifact index
-> evidence-based analysis
```

ADK may help interpret the user's intent, but benchmark execution still uses
the repository's deterministic tools and terminal workflow gates. LLM output is
never executed directly.

## Main Modules

- `adk_app/instructions.py`: root ADK instruction and migration boundary.
- `adk_app/root_agent.py`: optional real ADK `Agent` construction.
- `adk_app/agent.py`: official ADK discovery module exposing `root_agent`.
- `adk_app/runtime.py`: development bridge for explicit `adk run` diagnostics.
- `adk_app/tools/`: ADK function-tool wrappers around deterministic modules.
- `adk_app/evals/`: no-key ADK package and tool-contract checks.
- `terminal/`: product terminal, prompt input, language policy, and recovery.
- `workflows/`: deterministic benchmark setup state machine and planner bridge.
- `planners/`, `runners/`, `analyzers/`, `knowledge/`, `onboarding/`: benchmark
  engine control-plane tools reused by ADK.

## ADK Runtime Use

Install the ADK runtime in an isolated Python 3.10+ environment:

```bash
bash scripts/install_agent_deps.sh --yes
```

`./bin/anychain-agent` starts the AnyChain product terminal. Users do not need
to run `adk run` directly.

Then start the human-facing Agent:

```bash
./bin/anychain-agent
```

The terminal asks one question at a time and then calls deterministic tools:

```text
User> I want to benchmark Solana
Agent> ...asks fake-node or real-node...
User> fake-node
Agent> ...asks RPC mode, workload, and parameter-sample confirmation...
```

Development-only ADK CLI checks remain available through `agent/adk_app/runtime.py`
and `python3 agent/cli.py adk-status`, but they are not the product UX.

Detached benchmark jobs write:

```text
.agent/jobs/<job_id>/job.json
.agent/jobs/<job_id>/artifact_index.json
.agent/jobs/<job_id>/runtime.env
```

`runtime.env` is the per-job final configuration artifact. Users should not edit
it by hand.

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
- `claude`: Anthropic API key or Claude on Vertex AI with Google auth.
- `openai`: OpenAI API key.

At minimum, configure `LLM_PROVIDER`, `LLM_MODEL`, `LLM_AUTH_MODE`, and the
matching provider credential. For Vertex AI, also configure
`GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.

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
