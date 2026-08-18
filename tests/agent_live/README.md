# Agent Live CLI Matrix

This directory contains live-model evidence providers for the AnyChain product
terminal. These tests require a configured LLM provider and may consume provider
quota. The sole product-acceptance authority is
`tests/agent_live/run_product_acceptance.py`; no provider in this directory may
independently advance a phase or close G0-G6.

The fixed CLI regression runner is:

```bash
python3 tests/agent_live/run_langgraph_cli_matrix.py
```

It drives the same product entrypoint a user runs:

```bash
./bin/anychain-agent
```

Each scenario uses an isolated terminal state file, isolated session id, isolated
LangGraph checkpoint database, and `session_purpose=live-matrix`. The runner
passes `--state-file`, `--session-id`, `--checkpoint-path`, and
`--session-purpose live-matrix` to the same CLI users run. It also sets the
matching `ANYCHAIN_AGENT_*` environment variables so endpoint probe evidence can
be traced back to the test session. The runner reads LangGraph checkpoint state
after the conversation. It must not read or write legacy
`.agent/sessions/*/conversation_state.json` files.

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
- The user simulator must run named personas, not a neutral checklist. At
  minimum include: first-time confused evaluator, impatient operations engineer,
  copy/paste-heavy technical user, requirement-changing user, mixed-language
  user, report/debug analyst, custom-RPC integrator, new-chain evaluator, and
  resume-session user. Each persona must choose turns from the live Agent
  response and may ask basic product questions before giving benchmark inputs.
- The full transcript is the primary evidence. Passing state assertions without
  a stable user-facing transcript does not pass this gate.

Phase 6 closes only deterministic gate G2. Retained real-user regression
replay, real CLI, response-driven dual-AI Chaos, and real execution belong to
Phase 8 and close G3-G6 only after `run_product_acceptance.py` validates and
admits their revision-bound artifacts. Matrix, ledger, PTY, simulator, and
execution runners are subordinate evidence providers. Their direct exit codes
are raw evidence, not product acceptance.

Phase 8 uses finite, frozen denominators. G3 contains 60 retained-regression
obligations: exact, isomorphic, negative, and neighboring variants for each of
15 sanitized real-user cases. The current G4 catalog contains 732
response-driven obligations: 177 constrained pairwise rows plus 555 high-risk
state-control triples. This number is descriptive, not a second source of
truth: every run must validate the denominator and per-model counts from
`product_chaos_obligation_report()` in the revision-bound catalog before
freezing or admitting evidence. G5 contains four independent executions:
fake-node smoke, Geth real-node smoke, Geth real-node final, and bounded
sync-observe. Catalog generation is never execution evidence.

The Phase 8 execution adapters are:

- `retained_regression_runner.py` for G3 exact real-CLI contracts and open
  response-driven Journey contracts;
- `product_chaos_journey_provider.py` for converting each frozen G4 obligation
  into a runtime-compatible Journey definition and converting completed
  Journey artifacts into atomic obligation evidence;
- `product_obligation_evidence.py` for revision-, contract-, verifier-, and
  artifact-bound G3/G4 admission.
- `export_approved_plan.py` for exporting one immutable plan from a real
  LangGraph approval checkpoint. The G5 runner independently reopens the
  exported checkpoint and rejects caller-authored plans without matching
  current-turn approval provenance. These artifacts are confidential,
  owner-only local evidence (`0700` directory, `0400` files), must never be
  published, and are deleted by the Phase 8 authority only after G3-G6 reach
  one terminal passing decision. A content-addressed cleanup receipt remains;
  the checkpoint, plan, and approval sources do not. Failed or incomplete
  acceptance retains them owner-only for diagnosis; they require explicit
  cleanup before any evidence directory is copied or published.
- `execute_real_execution_ledger.py` consumes four approved-plan artifacts
  and produces the four G5 jobs: fake-node smoke, real-node smoke,
  real-node final benchmark, and sync-observe. The two real-node approvals
  bind the same immutable plan to two distinct user decisions. RPC benchmark plans own
  `LOCAL_RPC_URL`; only sync-observe plans own
  `NODE_PROMETHEUS_METRICS_URL`. The fixed Geth metrics probe remains an
  external G5 runtime attestation for both real Geth lanes.
- `completed_journey_batch.py` validates a complete frozen Journey batch,
  converts every declared G3/G4 obligation through the existing strict
  per-obligation adapter, and publishes one read-only evidence index.
  `retained_regression_evidence_set.py` combines the exact execution index and
  the declared open-batch index into the immutable 60-row G3 collection.
  Product acceptance consumes these indexes and never discovers G3/G4
  evidence by directory globbing.

An adapter is not a pass. G3 remains blocked while any retained semantic
postcondition lacks a reviewed machine evaluator. Its current executable
registry fails those postconditions closed; caller-authored verifier results
cannot be supplied to the evidence adapter. G4 evidence must contain complete
response-bound decision provenance for every turn, including hashes for the
preceding Agent response and submitted user message plus monotonic selection
and submission timestamps.

### Baseline Persona Transcript That Must Pass

Every broad Agent workflow or Harness change must include a dual-AI chaos
transcript equivalent to this first-time confused evaluator path. It must start
from a non-empty previous checkpoint, normally a partial `sync-observe / bsc`
session, not from a clean state:

1. Ask identity and orientation questions, for example `who are u?`,
   `你从哪里来`, `你要去哪里`, `你可以做什么？`, and `那我们现在可以从哪里开始？`.
   The Agent must explain itself and give a usable next step instead of dumping
   raw framework facts or saying only that no pending question exists.
2. Ask mode concepts, for example `fake node，real node，sync observe 都是什么？`.
   The Agent must explain the three modes and the boundaries between
   fake-node, real-node, and sync-observe.
3. Choose `fake node` from the old `sync-observe / bsc` session. The Agent must
   explicitly handle the old chain/mode state: either ask whether to keep the
   old chain or ask for the chain. It must not silently reuse a stale chain.
4. Complete the normal fake-node resource, workload, QPS, observability, and
   preflight/smoke confirmation path using short answers such as `1`, `y`,
   `n`, and natural-language confirmations.
5. When the Agent asks whether to run preflight and smoke, a valid positive
   answer such as `1`, `Y`, or `是的` must execute the preflight/smoke path or
   produce a clear execution blocker. It must not route the answer to chain
   selection, capability text, or generic help.
6. After confirmation, ask `你执行过测试了么？`, `当前是什么状态？`,
   `那你接下来要做什么？`, and `那你该做什么了？`. The Agent must answer from
   checkpoint/job/preflight/smoke state and provide the concrete next action.
   It must not answer only with a generic workflow description.

This baseline is intentionally basic. If it fails, the Agent is not ready for
more advanced custom-RPC, unknown-chain, or report-analysis claims.

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
9. Execution-side effects: after the user approves preflight/smoke or a run, the
   test must verify the expected job/preflight/smoke state, artifact paths, or a
   clear blocker. A reply that merely says something without changing state is a
   failure.
10. Current-state and next-action questions: after every major group transition
   and especially after execution approval, ask the Agent what happened, what
   the current state is, and what it will do next. The answer must be specific
   to the checkpoint and job state.

Dual-AI chaos failures must be fixed in the owning Harness group, validator,
or terminal boundary. Do not patch failures with keyword lists, fuzzy matching,
or terminal-side business routing.

### Simulator And Evidence Boundary

The repository does not define or call a Codex API. The dynamic runner accepts
an external `Simulator` implementation that receives the complete latest Agent
response plus one revision-bound scheduled coverage target. The external Codex
session returns one decision, rationale, persona, goal, and exact user turn.
`tests/agent_live/chaos_scheduler.py` writes the immutable schedule; it is
generation intent, not execution evidence.

A dynamic turn qualifies only when all of the following agree:

- the schedule, authoritative ledger, and worktree revision;
- explicit provider/model identity printed by the real CLI;
- a baseline product runtime event and exactly one newly committed event;
- thread, session purpose, turn indexes, and chained state fingerprints;
- a Linux/Docker postcondition verifier that inspects committed state or job
  artifacts independently of the PTY transport;
- the target edge and observed coverage ids recorded by that verifier.

The PTY returning a response never marks a row passed. Missing provider text,
missing or stale events, an unknown edge, a failed assertion, or a required
execution edge without hashed job artifacts fails the scheduled row closed.
The schedule result remains `incomplete` or `failed`, and no passing evidence
artifact is written.

The runner resolves provider/model with the production Agent configuration
loader and freezes that identity into the child PTY environment. A private
configuration file cannot change the model between schedule creation and CLI
startup. Extra environment values are copied into an immutable snapshot;
provider/model/local-config keys are reserved, and the frozen identity is
applied last. The canonical private config may supply credentials, while the
config loader restores any caller-frozen provider/model after sourcing it.
Evidence conversion uses the same resolved identity rather than a hard-coded
default model name.

Covering-array reports expose generated rows/tuples separately from observed
rows/tuples. Critical-sequence reports derive coverage ids from validated turn
artifacts; simulator-supplied or invented evidence ids do not count.

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

All product acceptance, live-model, observed-coverage, and real-execution
claims from this directory are Docker-only. Host execution may diagnose a test
or run fast unit checks, but it cannot produce qualifying coverage evidence.
The fixed matrix and fake-node path are regression subsets: fake-node proves
only its closed loop and cannot qualify local real-node, custom-RPC live-probe,
sync-observe, or final execution edges.

Before reporting coverage, bind every observation to the tested worktree
revision and report these categories separately: cataloged/generated,
observed-pass, observed-fail, not-run, externally-blocked, uncovered pairs,
uncovered required triples, completed critical sequences, and execution edges
with hashed job artifacts. A generated schedule/row, simulator assertion, or
successful PTY return is never sufficient by itself.

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
Each retained transcript starts with scenario/session/checkpoint metadata so
state bleed can be audited before any product claim.
