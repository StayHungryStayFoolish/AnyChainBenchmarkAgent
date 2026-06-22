# Secondary Development Guide

[中文](../zh/secondary-development-guide.md) | [English](secondary-development-guide.md)

This guide is for users who want to extend AnyChain Benchmark Agent after the
Agent is configured. It covers Knowledge Base integration, adding a chain in an
existing protocol family, adding a new protocol family, adding RPC methods,
closed-loop validation, and pull request requirements.

The Agent can help draft plans, inspect repository capabilities, and generate
checklists. It should not silently merge a new chain, protocol, or workload
method without deterministic validation and human review.

## Extension Boundaries

Keep these boundaries stable:

- User intent and QA live in `agent/`.
- Run-specific confirmed values live in `.agent/jobs/<job_id>/runtime.env`.
- Default user configuration lives in `config/user_config.sh`.
- Agent provider configuration lives in `config/agent_config.sh`.
- Chain support lives in `config/chains/*.json` and `tools/chain_adapters/`.
- fake-node replay data lives in `tools/fake-node/fixtures/`.
- fake-node family behavior lives in `tools/fake-node/configs/` and
  `tools/fake-node/handlers/`.
- Benchmark execution starts from `blockchain_node_benchmark.sh`.
- Reports and archives are generated through the runtime path registry and
  archived under the benchmark result directory.

Do not add chain-specific behavior directly to shared shell monitor code. Add
it to the chain template, adapter, fake-node mapping, or sync-health registry.

## Main Call Chains

Agent-launched benchmark:

```text
user prompt
-> Agent intent router / workflow
-> benchmark plan
-> preflight and risk checks
-> .agent/jobs/<job_id>/runtime.env
-> blockchain_node_benchmark.sh
-> target generator
-> proxy
-> vegeta
-> monitoring collectors
-> analysis
-> HTML reports and archive
```

Chain/RPC runtime path:

```text
config/chains/<chain>.json
-> tools/chain_adapters/
-> tools/target_generator.sh
-> tools/proxy/
-> tools/fake-node/fixtures/<chain>/
-> analysis/per_method_attribution.py
-> visualization/report_generator.py
```

Knowledge Base path:

```text
agent_config.sh
-> agent/knowledge/loader.py
-> local repo provider or HTTP provider
-> Agent grounding prompt
-> deterministic schema and safety checks
-> user-facing answer or benchmark plan
```

## 1. Integrate an Enterprise Knowledge Base

Use this when a company has an internal KB with supported chains, real RPC
samples, benchmark policies, or workload recommendations.

Development locations:

- `config/agent_config.sh`: provider selection and endpoint settings.
- `agent/knowledge/base.py`: provider contract.
- `agent/knowledge/http_provider.py`: generic HTTP adapter.
- `agent/knowledge/loader.py`: provider selection.
- `agent/adk_app/instructions.py`: how ADK should ground KB evidence and avoid
  unsupported claims.
- `agent/adk_app/tools/read_only.py`: ADK read-only tools that expose KB search
  and local capability evidence.
- `agent/cli.py`: smoke commands and integration entrypoints.

Expected contract:

- KB answers must be treated as evidence, not executable code.
- The Agent must still validate chain templates, RPC params, fixtures, and
  workload generation with repository tools.
- If KB evidence is missing or conflicts with repository state, the Agent must
  ask for confirmation or produce an onboarding plan.

Minimum HTTP adapter behaviors:

```text
POST /search
GET /chains/{chain}/rpc-methods
GET /chains/{chain}/rpc-samples
POST /workload/suggest
```

Validation:

```bash
python3 agent/cli.py knowledge-smoke
python3 -m unittest tests.test_agent_runtime_contract -v
```

PR expectations:

- Document the provider contract.
- Add redaction for secrets or private endpoints.
- Add smoke tests for unavailable KB, empty KB, and conflicting KB responses.
- Do not commit private KB responses.

## 2. Integrate an Enterprise Agent Platform

Use this when a company wants AnyChain Benchmark Agent to run as a tool inside
an internal Agent platform instead of only as a terminal chat.

Development locations:

- `agent/cli.py`: JSON CLI entrypoint.
- `agent/tools/schema.py`: OpenAI-compatible tool catalog.
- `agent/tools/executor.py`: stable named tool execution.
- `config/agent_config.sh`: LLM, Google auth, and optional KB defaults.
- `agent/runners/job_manager.py`: job status, artifact index, and detached run
  lifecycle.
- `agent/adk_app/instructions.py`: root ADK instruction.
- `agent/adk_app/tools/`: ADK function-tool wrappers.
- `agent/adk_app/evals/`: no-key ADK package and tool-contract checks.

Supported integration modes:

- Human terminal: `./bin/anychain-agent`
- JSON CLI: `python3 agent/cli.py <command>`
- Tool schema export: `python3 agent/cli.py tool-schema`
- Named tool call:
  `python3 agent/cli.py tool-call --name <tool> --arguments '<json>'`

Typical platform tools:

- `discover_environment`
- `load_capabilities`
- `draft_request`
- `generate_plan`
- `run_preflight`
- `submit_job`
- `get_job_status`
- `tail_job_log`
- `analyze_artifacts`
- `answer_artifact_question`
- `diagnose_artifacts`
- `draft_chain_template`
- `gap_analysis`
- `knowledge_search`

Boundaries:

- The enterprise platform may orchestrate tools, but benchmark execution still
  requires Agent preflight and approval rules.
- Real long-running benchmarks should use detached/background job mode.
- Platform sessions should persist `job_id`, `artifact_index`, and the archive
  path so users can resume after terminal or platform-session disconnects.
- Secrets must come from the enterprise secret manager or runtime environment,
  not from committed config files.
- KB evidence does not replace local validation.

Validation:

```bash
python3 agent/cli.py tool-schema
python3 agent/cli.py tool-call --name load_capabilities
python3 agent/cli.py tool-call --name discover_environment
python3 -m unittest tests.test_agent_runtime_contract -v
```

PR expectations:

- Keep tool names and schemas backward compatible where possible.
- Add tests for new tools.
- Document any new required arguments.
- Do not make enterprise platforms depend on private local paths.

## 3. Add a Chain in an Existing Protocol Family

Use this when the new chain fits one of the current families:

- `jsonrpc`
- `bitcoin_jsonrpc`
- `rest`
- `substrate`
- `tendermint`
- `hedera_dual`

Development locations:

- `config/chains/<chain>.json`
- `config/chain_template.json.bak`
- `tools/chain_adapters/`
- `tools/fake-node/configs/`
- `tools/fake-node/fixtures/<chain>/`
- `docs/en/how-to-add-chain.md`
- `docs/zh/how-to-add-chain.md`

Required template fields:

- `chain_type`
- `rpc_url`
- `rpc_methods.single` or `rpc_methods.mixed_weighted`
- `param_formats` or `param_spec`
- `proxy_extraction`
- `_meta.adapter_family`
- sync-health metadata when the chain should support node health checks

Validation:

```bash
python3 tools/chain_adapters/cli.py validate-template --chain <chain>
python3 tools/fake-node/check_fixture_coverage.py --json
python3 tools/fake-node/runtime_probe.py --chain <chain>
python3 tools/fake-node/runtime_probe_block_height.py --chain <chain>
```

Closed-loop check:

```bash
./bin/anychain-agent
```

Then ask the Agent to create a fake-node smoke benchmark for the new chain,
run preflight, run the mock job, and analyze the generated archive.

PR expectations:

- Add the chain template.
- Add real fake-node fixtures for every workload method.
- Add or update docs only when the extension changes user behavior.
- Show validation commands and archive paths in the PR body.

## 4. Add a New Protocol Family

Use this only when the chain cannot be expressed by the current six families.

Development locations:

- `tools/chain_adapters/<family>.py`
- `tools/chain_adapters/base.py`
- `tools/fake-node/handlers/<family>.go`
- `tools/fake-node/configs/<family>.yaml`
- `tools/fake-node/main.go` if the handler registry needs an entry
- `config/chains/<chain>.json`
- `docs/en/how-to-add-chain.md`
- `docs/zh/how-to-add-chain.md`

Required design decisions:

- Request envelope and transport.
- Method extraction path for proxy attribution.
- Parameter schema support.
- Response fixture matching.
- Block height or sync-health parsing.
- Whether the family can share existing report and per-method attribution.

Validation:

```bash
python3 tests/test_chain_adapters.py
python3 tests/test_param_spec.py
python3 tools/chain_adapters/cli.py validate-template --chain <chain>
(cd tools/fake-node && go test ./...)
python3 tools/fake-node/runtime_probe.py --chain <chain>
bash tests/test_full_entrypoint_fake_node_lifecycle_smoke.sh
```

PR expectations:

- Include one minimal chain template for the new family.
- Include fake-node handler tests.
- Include runtime probe evidence.
- Explain why the existing six families were not enough.

## 5. Add an RPC Method

RPC method support is not only a method name. The framework needs request
construction, parameter samples, fake-node response fixtures, proxy attribution,
and report attribution to line up by the same method identity.

Development locations:

- `config/chains/<chain>.json`
- `tools/chain_adapters/param_spec.py`
- `tools/fake-node/configs/`
- `tools/fake-node/fixtures/<chain>/`
- `docs/audit/rpc-fixtures/`

For simple methods, use `param_formats`. For positional params, object params,
REST path params, query params, or request bodies, use `param_spec`.

Example three-argument method:

```json
{
  "rpc_methods": {
    "mixed_weighted": [
      {"method": "eth_getBalance", "weight": 40},
      {"method": "eth_blockNumber", "weight": 30},
      {"method": "eth_getStorageAt", "weight": 30}
    ]
  },
  "param_spec": {
    "eth_getStorageAt": {
      "transport": "jsonrpc_list",
      "params": [
        {"source": "address"},
        {"source": "target_storage_slot"},
        {"literal": "latest"}
      ]
    }
  }
}
```

Validation:

```bash
python3 tools/chain_adapters/cli.py validate-template --chain <chain>
bash tests/test_target_generator_mixed_weighted.sh
tools/fake-node/record_rpc_fixtures.sh <chain>
python3 tools/fake-node/check_fixture_coverage.py --json
python3 tools/fake-node/runtime_probe.py --chain <chain>
```

Report check:

- `logs/proxy_method.csv` contains the new method.
- per-method CSV includes success, error, latency, P50, P90, and P99 values.
- HTML report shows the method in per-method attribution charts.

## 6. Add or Change Workload Logic

Development locations:

- `config/chains/*.json`
- `tools/target_generator.sh`
- `analysis/per_method_attribution.py`
- `visualization/per_method_visualizer.py`
- `visualization/report_generator.py`

Rules:

- `mixed_weighted` is the source of weighted mixed-mode generation.
- Weights should sum to 100 for readability.
- Sync-health RPC methods should not be counted as workload methods.
- Per-method report charts should only describe benchmark workload traffic.

Validation:

```bash
bash tests/test_target_generator_mixed_weighted.sh
python3 tests/test_per_method_attribution.py
python3 tests/test_per_method_charts.py
python3 tests/test_per_method_report.py
```

## 7. Closed-Loop Test Requirements

Every functional extension should prove these layers:

1. Template validation.
2. Request generation.
3. fake-node fixture coverage.
4. fake-node runtime probe.
5. block height / sync-health probe when applicable.
6. proxy method attribution.
7. report generation.
8. archive generation.

Recommended smoke sequence:

```bash
python3 tools/chain_adapters/cli.py validate-template --chain all
python3 tools/fake-node/check_fixture_coverage.py --json
python3 tools/fake-node/runtime_probe.py
python3 tools/fake-node/runtime_probe_block_height.py
python3 -m unittest tests.test_agent_runtime_contract -v
python3 tools/check_public_repo_markers.py --root .
git diff --check
```

For monitoring or lifecycle changes, use Linux or Docker:

```bash
bash tests/test_monitoring_lifecycle_smoke.sh
bash tests/test_monitoring_runtime_contract.sh
bash tests/test_full_entrypoint_fake_node_lifecycle_smoke.sh
```

## 8. PR Requirements

Before opening a PR:

- Run the test set that matches the touched area.
- Update English and Chinese docs for user-visible behavior.
- Do not commit runtime archives, local `.agent/` jobs, secrets, API keys,
  private endpoints, local machine paths, or generated binaries.
- Use a Conventional Commit PR title.
- Fill in `.github/pull_request_template.md`.

High-risk paths require extra care:

- `blockchain_node_benchmark.sh`
- `monitoring/`
- `tools/proxy/`
- `tools/fake-node/`
- `tools/target_generator.sh`
- `tools/benchmark_archiver.sh`
- `config/chains/`
- `agent/`

See [GitHub PR Gates and Branch Protection](github-pr-gates.md) for the full
repository policy.

## What the Agent Should Do for Developers

For extension requests, the Agent should:

- inspect current repository capabilities;
- classify whether the request is KB, existing-family chain, new-family chain,
  RPC method, workload, monitoring, or report related;
- generate a plan with touched files and validation commands;
- identify missing params, fixtures, or sync-health evidence;
- ask for confirmation before writing config or launching long benchmarks;
- run deterministic validation after changes;
- cite exact files and archive paths in the final answer.

For unsupported or risky requests, the Agent should produce a development plan
instead of pretending the framework already supports the behavior.
