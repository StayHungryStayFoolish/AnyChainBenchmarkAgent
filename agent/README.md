# AnyChain ADK Agent

`agent/` contains the Google ADK control plane for the benchmark engine. The
human-facing Agent entrypoint is ADK-only:

```bash
./bin/anychain-agent
```

The old custom chat loop, wizard, intent router, workflow router, and prompt
orchestrator have been removed. Reusable benchmark tools remain available as
deterministic modules and ADK function tools.

## Runtime Contract

```text
user message
-> ADK instruction and model reasoning
-> prepare_benchmark_run
-> preflight, runbook, and user confirmation
-> lifecycle smoke or isolated fake-node benchmark smoke
-> confirmation-gated benchmark job
-> artifact index
-> evidence-based analysis
```

ADK may help interpret the user's intent, but benchmark execution still uses
the repository's deterministic tools. LLM output is never executed directly.

## Main Modules

- `adk_app/instructions.py`: root ADK instruction and migration boundary.
- `adk_app/root_agent.py`: optional real ADK `Agent` construction.
- `adk_app/agent.py`: official ADK discovery module exposing `root_agent`.
- `adk_app/runtime.py`: thin bridge to the official `adk run` CLI.
- `adk_app/tools/`: ADK function-tool wrappers around deterministic modules.
- `adk_app/evals/`: no-key ADK package and tool-contract checks.
- `planners/`, `runners/`, `analyzers/`, `knowledge/`, `onboarding/`: benchmark
  engine control-plane tools reused by ADK.

## ADK Runtime Use

Install the ADK runtime in an isolated Python 3.10+ environment:

```bash
bash scripts/install_agent_deps.sh --yes
```

`./bin/anychain-agent` automatically prefers `.venv-adk/bin/adk`, so users do
not need to activate the venv before starting the Agent.

Then start the human-facing Agent:

```bash
./bin/anychain-agent
```

This delegates to:

```bash
adk run agent/adk_app
```

Natural-language benchmark requests should be handled by the ADK model through
the registered function tools:

```text
Prepare a Solana fake-node smoke benchmark at 1 QPS
Run a real fake-node benchmark smoke only after I approve it
```

The ADK runtime should call planning, preflight, action, and read-only tools as
needed. The old local terminal facade has been removed.

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
