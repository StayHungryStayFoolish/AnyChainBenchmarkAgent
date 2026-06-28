# Agent Live Acceptance Scenarios

This directory contains live-model acceptance scenarios for the AnyChain product terminal.
They are not part of the default unit-test suite because they require a real LLM
credential and may consume provider quota.

These JSON files are test fixtures only. They do not participate in runtime
intent recognition, routing, planning, or workflow execution. Runtime
natural-language understanding must be handled by Google ADK and the configured
model. The scenarios here only verify that real model output respects product
contracts and common domain boundaries.

## Current Status

`agent_intent_smoke_scenarios.json` is the current minimal live smoke scenario
set for the ADK-native product terminal. `agent_product_acceptance_scenarios.json`
is the broader product acceptance set. `agent_chaos_conversation_scenarios.json`
adds multi-turn chaos cases such as vague requests, chain-only replies,
fake-node/real-node switching, bad weights, arbitrary placeholders, mid-flow
onboarding questions, and gate-skipping attempts. They check representative
natural-language turns with a real model provider and reject common regressions
such as internal tool name leakage, keyword-router leakage, scratchpad leakage,
and fake-node configuration shortcuts.

`agent_edge_acceptance_scenarios.json` covers additional product edges that are
easy to miss in happy-path tests:

- reverting a previously confirmed variable;
- multi-disk confirmation with numbered choices and manual override;
- fake-node still requiring resource metadata confirmation;
- quick/standard/intensive QPS profile explanation and item-level adjustment;
- external Prometheus/Grafana and port-conflict handling;
- detached job log-follow behavior and safe Ctrl+C expectations;
- out-of-order user input such as URL before chain;
- chain switching that invalidates RPC workload assumptions;
- custom RPC requests without parameter/response samples.

The authoritative contracts are now code and tests:

- `agent/adk_app/instructions.py`
- `agent/adk_app/agents/domain.py`
- `agent/adk_app/tools/`
- `tests/test_agent_product_terminal.py`
- `tests/test_agent_runtime_contract.py`

## Run With DeepSeek

Use the gitignored local config file or export the key in the shell. Do not
commit credentials.

```bash
source config/agent_config.sh
docker exec blockchain-node-benchmark-bench-1 bash -lc '
  cd /workspace &&
  source config/agent_config.sh &&
  ANYCHAIN_AGENT_PYTHON=/tmp/anychain-adk/bin/python \
  /tmp/anychain-adk/bin/python tests/agent_live/run_live_matrix.py \
    --matrix tests/agent_live/agent_intent_smoke_scenarios.json \
    --require-live \
    --provider deepseek \
    --model deepseek-chat \
    --agent-python /tmp/anychain-adk/bin/python \
    --timeout 180
'
```

Logs are written to `/tmp/anychain-agent-live-matrix` inside the container.
The runner redacts API-key patterns before writing logs.

Run the chaos matrix the same way:

```bash
source config/agent_config.sh
docker exec blockchain-node-benchmark-bench-1 bash -lc '
  cd /workspace &&
  source config/agent_config.sh &&
  ANYCHAIN_AGENT_PYTHON=/tmp/anychain-adk/bin/python \
  /tmp/anychain-adk/bin/python tests/agent_live/run_live_matrix.py \
    --matrix tests/agent_live/agent_chaos_conversation_scenarios.json \
    --require-live \
    --provider deepseek \
    --model deepseek-chat \
    --agent-python /tmp/anychain-adk/bin/python \
    --timeout 180
'
```

Run the edge matrix the same way:

```bash
source config/agent_config.sh
docker exec blockchain-node-benchmark-bench-1 bash -lc '
  cd /workspace &&
  source config/agent_config.sh &&
  ANYCHAIN_AGENT_PYTHON=/tmp/anychain-adk/bin/python \
  /tmp/anychain-adk/bin/python tests/agent_live/run_live_matrix.py \
    --matrix tests/agent_live/agent_edge_acceptance_scenarios.json \
    --require-live \
    --provider deepseek \
    --model deepseek-chat \
    --agent-python /tmp/anychain-adk/bin/python \
    --timeout 180
'
```

Each scenario uses an isolated `--session-id` so workflow-state artifacts do not
cross-contaminate scenarios.

## Coverage Intent

The scenario set should cover:

- language matching and terminal hygiene
- framework capability answers grounded in repo facts
- fake-node and real-node benchmark planning
- dependency and preflight gate refusal
- custom RPC and unsupported-chain onboarding handoff
- mixed workload weight validation
- job resume and artifact-analysis questions
- Prometheus/Grafana local and exporter modes
- enterprise KB and Agent platform integration answers
- dangerous-operation refusal

Add a scenario whenever a bug is found in a real user session.
