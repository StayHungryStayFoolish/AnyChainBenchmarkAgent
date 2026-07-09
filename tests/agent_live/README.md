# Agent Live CLI Matrix

This directory contains live-model acceptance tests for the AnyChain product
terminal. These tests require a configured LLM provider and may consume provider
quota.

The current live runner is:

```bash
python3 tests/agent_live/run_langgraph_cli_matrix.py
```

It drives the same product entrypoint a user runs:

```bash
./bin/anychain-agent
```

Each scenario uses an isolated terminal state file and an isolated LangGraph
checkpoint database through `ANYCHAIN_AGENT_CHECKPOINT_PATH`. The runner reads
LangGraph checkpoint state after the conversation. It must not read or write
legacy `.agent/sessions/*/conversation_state.json` files.

## Required Dual-AI Chaos Test

The scripted matrix is not enough to claim product readiness. For every broad
Agent workflow or Harness change, run a real terminal conversation where:

- Agent side: start the real `./bin/anychain-agent` CLI in Docker/Linux with the
  configured live provider, normally DeepSeek for non-search validation.
- User side: Codex acts as the user simulator and decides each next user turn
  dynamically from the Agent's previous response.
- The user simulator must not replay only a fixed script. It should behave like
  a real technical user: jump between groups, answer partially, paste noisy
  config/log/request/response blocks, switch language, change chain/mode/QPS/RPC
  choices, go back, contradict earlier answers, and ask explanatory questions.
- The full transcript is the primary evidence. Passing state assertions without
  a stable user-facing transcript does not pass this gate.

Minimum dual-AI chaos coverage:

1. Normal fake-node path from startup through resource confirmation, workload,
   QPS, observability, review, preflight, and smoke.
2. The same path with mid-group jumps: disk to QPS, workload to chain switch,
   observability to RPC, and back/undo requests.
3. Language switching and technical scalar inputs with punctuation or copied
   formatting; the Agent must keep the user's language except for technical
   identifiers.
4. Case 1: supported chain with custom RPC methods, including request/response
   evidence, endpoint probe, fixture evidence, default replacement, and weights.
5. Case 2: unknown chain in an existing adapter family, including unknown chain
   identity confirmation, endpoint validation, custom RPC method validation,
   weights, and continuation into normal benchmark setup.
6. Case 3: unsupported or uncertain protocol, including official-doc handoff
   behavior and return to a supported chain or another case.
7. Multi-intent natural language and pasted YAML/JSON/env content that maps to
   multiple groups; inferred values must be reviewed with the user before use.
8. Resume behavior after a previous partial session: continue, modify, and clear
   must all be tested.

Dual-AI chaos failures must be fixed in the owning Harness group, validator,
or terminal boundary. Do not patch failures with keyword lists, fuzzy matching,
or terminal-side business routing.

## Current Coverage

The matrix covers representative product paths that previously regressed:

- unknown chain identity gate before any partial alias can be accepted;
- mid-flow target-mode jump with explicit confirmation;
- capability detour that returns to benchmark setup;
- existing-chain custom RPC method validation using a real endpoint;
- custom RPC scope selection such as replacing template defaults;
- mixed workload weights that must sum to 100 and remain job-local.

Add a scenario whenever a real user session exposes a new class of bug. Prefer
multi-turn user-style prompts over single-step API assertions.

## Run In Docker

Use the repository Docker/Linux environment. macOS is not a supported product
runtime target for the Agent.

```bash
docker compose exec bench bash -lc '
  cd /workspace &&
  source config/agent_config.sh &&
  .venv-adk/bin/python tests/agent_live/run_langgraph_cli_matrix.py
'
```

Run one scenario:

```bash
docker compose exec bench bash -lc '
  cd /workspace &&
  source config/agent_config.sh &&
  .venv-adk/bin/python tests/agent_live/run_langgraph_cli_matrix.py \
    --scenario existing_chain_custom_rpc_replace_defaults
'
```

Failed scenario transcripts are written to:

```text
.agent/live-matrix/<scenario>.transcript.txt
```

Those transcripts are debugging artifacts and should not be committed.
