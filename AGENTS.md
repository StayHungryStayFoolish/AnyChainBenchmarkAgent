# AGENTS.md

This file is for AI assistants helping a user configure, run, validate, or
extend AnyChain Benchmark Agent.

If you are an AI coding agent modifying this repository, read `AI_CODING_GUIDE.md` and
`docs/en/anychain-agent-ai-work-gate.md` first. If you are helping a user use
the project without changing code, start with this file and the linked runtime
docs.

## What This Project Is

AnyChain Benchmark Agent is a Google ADK-based terminal Agent for blockchain
node benchmark workflows. The Agent helps users:

- configure an LLM provider;
- inspect local environment and dependencies;
- run fake-node closed-loop benchmark tests;
- prepare real-node benchmark plans;
- validate chain/RPC workloads;
- run preflight, smoke, detached jobs, log follow, and report analysis;
- generate secondary-development plans for new chains or RPC methods.

The deterministic benchmark engine remains the source of truth. The Agent
must use ADK and deterministic tools instead of terminal keyword routing.

## Fast Path For Helping A User Configure The Agent

Do not ask the user to understand every benchmark variable first. Help them
configure the Agent provider, then let the Agent inspect the benchmark
environment.

1. Ask which model provider they can use:
   - Gemini API key
   - Gemini on Vertex AI with Google ADC, attached service account,
     service-account impersonation, or service-account JSON file
   - `claude` API key
   - `claude` on Vertex AI
   - OpenAI API key
   - DeepSeek API key
2. Write secrets and local provider choices to:

   ```text
   config/agent_config.local.sh
   ```

   That file is gitignored. Do not commit it.
3. Keep repository defaults in `config/agent_config.sh` generic.
4. Install the isolated Agent runtime:

   ```bash
   bash scripts/install_agent_deps.sh --yes
   ```

5. Validate the Agent configuration:

   ```bash
   source config/agent_config.sh
   python3 agent/cli.py adk-status
   python3 agent/cli.py adk-eval
   ```

6. Start the product terminal:

   ```bash
   ./bin/anychain-agent
   ```

7. Tell the user to describe the benchmark goal in natural language. The Agent
   should inspect the environment, ask for missing values, run preflight/smoke,
   and request approval before execution.

## Fast Path For Helping A User Operate The Agent

After provider setup is complete, help the user use the Agent rather than
teaching them every benchmark variable.

1. Start with `./bin/anychain-agent`.
2. Ask the user to run `doctor` or describe the benchmark goal directly.
3. Let the Agent inspect the host, dependencies, cloud/deployment shape, disks,
   network, existing jobs, and framework capabilities.
4. For a local closed-loop test, guide the user toward fake-node first:

   ```text
   Prepare a Solana fake-node smoke benchmark at 1 QPS.
   ```

   The Agent should still confirm benchmark metadata such as RPC mode, workload,
   disk/network baseline, and report context. Fake-node only removes the need
   for a real `LOCAL_RPC_URL`.
5. For a real node, ask for only the values discovery cannot safely infer:
   chain, local RPC URL, mainnet/reference RPC URL when needed, node process
   names, ledger/accounts disk mapping, volume baselines, network bandwidth,
   RPC mode, method selection, and mixed weights.
6. Tell the user that real benchmark execution must pass preflight, smoke, and
   explicit approval. Long jobs run detached by default and can be resumed with
   `status`, `jobs`, `logs`, or `resume`.
7. After a run, help the user ask for evidence-backed analysis. The Agent should
   cite report, CSV, log, runtime.env, and artifact-index paths.
8. For unsupported chains or custom RPC methods, the Agent should generate a
   conservative onboarding plan or draft template. It must not claim support
   until templates, parameter samples, fixtures, validation, and fake-node smoke
   pass.

Useful user prompts:

```text
doctor
What chains and RPC methods are supported?
Prepare a Solana fake-node smoke benchmark at 1 QPS.
Prepare a real-node benchmark for my Solana RPC endpoint.
Show job status and logs.
Analyze the latest report and cite the evidence files.
How do I add a custom RPC method with three params?
Generate an onboarding plan for a new jsonrpc-family chain.
```

## Provider Configuration Examples

Use `config/agent_config.local.sh` for real values.

Gemini API key:

```bash
LLM_PROVIDER="gemini"
LLM_MODEL="gemini-3.1-pro-preview"
LLM_AUTH_MODE="api_key"
GEMINI_API_KEY="<secret>"
```

Gemini on Vertex AI with ADC:

```bash
LLM_PROVIDER="gemini"
LLM_MODEL="gemini-3.1-pro-preview"
LLM_AUTH_MODE="google_adc"
GOOGLE_CLOUD_PROJECT="<project-id>"
GOOGLE_CLOUD_LOCATION="global"
```

`claude` API key:

```bash
LLM_PROVIDER="claude"
LLM_MODEL="<available-claude-model>"
LLM_AUTH_MODE="api_key"
ANTHROPIC_API_KEY="<secret>"
```

OpenAI API key:

```bash
LLM_PROVIDER="openai"
LLM_MODEL="<available-openai-model>"
LLM_AUTH_MODE="api_key"
OPENAI_API_KEY="<secret>"
```

DeepSeek API key:

```bash
LLM_PROVIDER="deepseek"
LLM_MODEL="deepseek-chat"
LLM_AUTH_MODE="api_key"
DEEPSEEK_API_KEY="<secret>"
```

## Google Search Boundary

ADK `google_search` is supported only when the Agent runs with Gemini-family
models and valid Gemini/Google authentication, and only in chain/RPC onboarding
or custom-RPC research flows. It is evidence gathering, not permission to skip
fixture recording, endpoint validation, template validation, or fake-node smoke.

For `claude`, OpenAI, DeepSeek, or `claude`-on-Vertex modes, tell the user that web
research is unavailable unless a provider-specific integration is added later.

## What Another AI Should Not Do

- Do not commit secrets or local runtime files.
- Do not edit `config/user_config.sh` for ordinary users unless they explicitly
  want persistent benchmark defaults.
- Do not ask users to manually fill every benchmark variable before starting
  the Agent.
- Do not bypass preflight, smoke, or approval gates.
- Do not claim a new chain or RPC method is supported until templates, parameter
  samples, fixtures, validation, and smoke tests pass.
- Do not implement natural-language intent handling with keyword lists, fuzzy
  matching, or regex routing in terminal code.

## Documents To Read For Deeper Work

- `README.md`: user-facing quick start and full overview.
- `agent/README.md`: Agent runtime and development contract.
- `docs/en/adk-agent-architecture.md`: ADK architecture and Agent Loop.
- `docs/en/anychain-agent-ai-work-gate.md`: project-specific AI coding gate.
- `docs/zh/adk-agent-architecture.md` and
  `docs/zh/anychain-agent-ai-work-gate.md`: Chinese-directory mirrors for users
  browsing localized docs.
- `docs/zh/how-to-add-chain.md`: adding chain and RPC support.
- `docs/zh/local-closed-loop-testing.md`: fake-node closed-loop testing.
- `docs/zh/secondary-development-guide.md`: secondary-development handoff.
- `docs/en/github-pr-workflow.md` and `docs/zh/github-pr-workflow.md`:
  maintainer workflow for creating PRs, waiting for checks, handling review
  gates, and merging safely. This is for coding/maintenance work, not ordinary
  user setup.

## Documentation Maintenance Rules

- Keep long-lived user, operator, architecture, gate, and extension docs.
- Do not commit temporary task plans, debugging transcripts, one-off execution
  plans, or model-specific work logs.
- When behavior changes, update the user-facing README and the relevant
  reference document in the same PR.
- Keep `docs/en` and `docs/zh` aligned for long-lived documentation. If a doc is
  intentionally available in only one language, explain why in the relevant
  docs index.
- Keep AI-facing gate and architecture docs current because other AI assistants
  use them as repository context.

## Minimum Validation For AI-Generated Changes

When code changes are made, run:

```bash
python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract
python3 tools/check_agent_boundaries.py --root .
python3 agent/cli.py adk-eval
git diff --check
```

When model-facing behavior changes and credentials are available, also run the
live matrices in `tests/agent_live/`.
