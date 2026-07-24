# AnyChain Agent External AI Handoff And Product Verification Spec

Status: long-lived handoff and verification contract, tracked in Git

Runtime target: Linux in Docker only
Live external gap: Gemini with real `google_search` grounding

## Start Here

This document lets another coding AI continue Agent work without the original
conversation. Before editing code, read these files in order:

1. `AI_CODING_GUIDE.md`
2. `AGENTS.md`
3. `docs/en/anychain-agent-ai-work-gate.md`
4. `docs/en/adk-agent-architecture.md`
5. `agent/README.md`
6. this document
7. `tests/agent_live/README.md`
8. `tests/agent_live/legacy_issue_map.py`

Do not follow deleted dated task plans or reconstruct behavior from old
transcripts. The code registries and the documents above are authoritative.
The legacy map is the only supported index for findings migrated from the
deleted 2026-07-10 repair plan and 2026-07-11 known-issues register. An item is
not closed merely because either deleted document once called it fixed.

## Product Architecture

The shipped entrypoint is `./bin/anychain-agent`. Terminal code owns input,
output, startup diagnostics, dependency consent, Ctrl+C, and exact job
commands. It does not route benchmark business flow.

The LangGraph Harness is the single conversation and workflow authority:

```text
complete terminal turn
-> exact active option or typed literal fast path when deterministic
-> otherwise hierarchical semantic partition and bounded owner compilation
-> admission and dependency-safe ordering
-> select exactly one durable action
-> route exactly one owning domain
-> commit one typed result
-> persist/invoke/receipt when the action has an external side effect
-> repeat selection through graph transitions while admitted work remains
-> canonical fallback and one response with at most one blocking question
-> validate and checkpoint schema version 14
```

The single metadata authority is
`agent/workflows/group_registry.py::GROUPS`. It currently defines 20 groups,
their fields, questions, dependencies, invalidations, and exactly one owner per
group. `agent/harness/state.py::DEFAULT_GROUP_ORDER` and
`agent/harness/domains/registry.py` are derived runtime views and must never
become independent registries.

| Order | Group | Owner |
|---:|---|---|
| 1 | `opening` | `orientation` |
| 2 | `target_mode` | `chain_rpc` |
| 3 | `chain_identity` | `chain_rpc` |
| 4 | `provider_deployment` | `environment` |
| 5 | `ledger_disk` | `environment` |
| 6 | `accounts_disk` | `environment` |
| 7 | `network` | `environment` |
| 8 | `endpoint_process` | `chain_rpc` |
| 9 | `chain_auxiliary_endpoints` | `chain_rpc` |
| 10 | `workload_rpc` | `chain_rpc` |
| 11 | `target_samples_fixtures` | `chain_rpc` |
| 12 | `qps_profile` | `performance` |
| 13 | `sync_observe` | `sync_observe` |
| 14 | `observability` | `performance` |
| 15 | `advanced_tuning` | `performance` |
| 16 | `preflight_smoke_execution` | `execution` |
| 17 | `job_monitoring` | `execution` |
| 18 | `failure_recovery` | `recovery` |
| 19 | `error_evidence_analysis` | `analysis` |
| 20 | `report_artifact_analysis` | `analysis` |

The eight owners are:

- `orientation`: opening and session decisions
- `environment`: provider/deployment, ledger, accounts, and network
- `chain_rpc`: target mode, chain identity, endpoints, workload, and fixtures
- `performance`: QPS, observability, and advanced tuning
- `sync_observe`: synchronization observation configuration
- `execution`: preflight/smoke, final execution, and job monitoring
- `recovery`: actionable failure recovery
- `analysis`: pasted evidence and report/artifact analysis

Important implementation files:

- `agent/harness/graph.py`: LangGraph/checkpoint runtime
- `agent/harness/coordinator.py`: graph-node transitions and sole typed commit
  boundary
- `agent/harness/hierarchical_planner.py`: sole semantic-planning entry; Stage
  A partitions the complete turn and Stage B compiles owner-scoped actions
  before whole-plan admission
- `agent/harness/action_registry.py`: typed action contracts and prerequisites
- `agent/harness/admission.py`: admission, conflict, prerequisite, and semantic
  coverage validation
- `agent/harness/queue.py`: dependency ordering and pending-barrier eligibility
- `agent/harness/routing.py`: navigation and canonical fallback authority
- `agent/harness/response.py`: single response-composition authority
- `agent/harness/contracts.py`: durable action/result/effect/turn contracts
- `agent/harness/checkpoint_migrations.py`: isolated historical checkpoint
  migration boundary only
- `agent/harness/questions.py`: typed question/option contracts
- `agent/harness/transitions.py`: invalidation and reconfiguration state
- `agent/harness/invariants.py`: state and expected-patch enforcement
- `agent/harness/intent.py`: focused adjudication and schema/admission helpers,
  not a second product planner
- `agent/harness/domains/`: the eight product domain owners
- `agent/terminal/repl.py`: product terminal shell only
- `agent/runners/benchmark_pipeline.py`: plan materialization and submission

The retired monolithic group owner, retired graph-node layer, and terminal or
workflow business routers must not return. Architecture tests enforce these
boundaries without making retired implementation files part of the active
design contract.

The runtime compiles one graph with its SQLite checkpointer. It must not create
a second uncheckpointed turn graph or drain the complete durable queue inside
one Python node. Current-version turns must not invoke checkpoint compatibility
code. Version 12 crosses the isolated adapter once and is persisted through the
current schema; version 13 additionally migrates deferred-queue retention into
the typed `pending_question.resume_action_queue` contract before persistence as
version 14. Older state is quarantined and only allowlisted environment facts
may be offered for reconfirmation.

## Migrated Historical Findings

Run the migration audit from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/agent_live/legacy_issue_map.py \
  > /tmp/anychain-legacy-issue-map.json
```

The map covers every A.1-A.71 defect row, every still-relevant B/C/D item, and
all RP.1-RP.30 phases from the two retired documents. Each entry names its
current group(s), owner, disposition, and current test evidence. Its rules are
deliberately conservative:

- `guarded`: every named test exists in the current tree; this records a
  regression guard, not a claim that it passed on the current worktree;
- `superseded`: a current architecture contract and test replace the old
  implementation requirement;
- `open`: evidence is missing, manual-only, stale, indirect, external, or no
  longer asserts the same behavior.

Do not restore either retired dated document. Update the migration map when an
open item gains current evidence, and keep the current task plan responsible
for execution status.

## Workflow Contract

The Agent is not a linear wizard. A user may interrupt any partially completed
group, jump to another group, jump again, ask a question, paste structured
content, revise a confirmed value, switch language, switch chain/mode, or go
back. Partial state and interruption return points remain checkpointed. After
each action, the shared validator chooses the earliest relevant incomplete
group; completed groups are not repeated unless dependency invalidation makes
them incomplete.

Consultation and analysis are read-only detours. They answer the user and then
render the unchanged active question exactly once. A rendered option is valid
only when it maps to a typed action/answer with an owner, prerequisite,
postcondition, return policy, and localized copy.

## Chain And RPC Cases

### Case 1: configured chain, custom RPC

Preserve the canonical `config/chains` template. Collect or infer the method,
ordered params and types/meaning, request, response structure or official docs,
and a reachable validation endpoint. The endpoint embedded in an example is
not automatically the final `LOCAL_RPC_URL`. Validate the final selected
endpoint and method, record fixture evidence when fake-node is requested, then
choose single replacement, mixed replacement, or mixed addition. Enabled
weights are positive and total exactly 100. Only a job-local override is
materialized, and traffic evidence must prove the selected method executed.

### Case 2: unconfigured chain, existing adapter family

First ask the configured LLM whether the chain exists and which protocol it
uses. When Gemini search is available, ground that result with official search
evidence. The user confirms chain identity and adapter family. Then use the
same endpoint/schema/request/response validation contract as Case 1. A runtime
template and fixtures are provisional until smoke succeeds; never claim
permanent support or mutate canonical templates.

### Case 3: unsupported or uncertain adapter family

Collect official protocol, RPC, endpoint, request, and response evidence. Use
Gemini search when available; otherwise ask the user for official material.
Generate a real secondary-development handoff and stop benchmark execution.
The user may later return to Cases 1/2, a configured chain, or another mode.

## Execution Contracts

- fake-node validates fixtures, execution, logs, and reports; it does not
  measure real node performance.
- real-node validates the final endpoint and workload, performs preflight,
  submits an isolated safe smoke, reconciles its persisted result, and asks for
  a separate final-benchmark approval. Repeated approval is idempotent.
- sync-observe requires a real node/source. It observes sync height,
  process/core CPU, memory, disk, network, and client metrics without Vegeta,
  QPS, proxy targets, or fixture recording. Missing MGas/s is unavailable, not
  a meaningful zero.
- every submitted plan owns deterministic output and memory directories; job
  status is derived from process exit plus business artifacts.
- a failed/partial job exposes evidence and executable recovery choices, lets
  the user correct only the affected group, and returns through canonical
  fallback without duplicating a job.

## Dual-AI Chaos Acceptance

The fixed CLI matrix is regression coverage, not product acceptance. Product
acceptance requires a live Docker CLI in which DeepSeek is the AnyChain Agent
model and Codex acts as the user. Codex must read each actual Agent response
before choosing the next turn. A prewritten prompt sequence is not dual-AI
Chaos.

`tests/agent_live/run_product_acceptance.py` is the sole authority that may
advance phases or close G0-G6. Phase 6 closes only deterministic gate G2.
Retained real-user replay, real CLI, response-driven dual-AI Chaos, and real
execution belong to Phase 8 and close G3-G6 only after the controller admits
their revision-bound evidence. Ledger, matrix, PTY, simulator, and execution
scripts are subordinate evidence providers; their direct exit codes never
declare product readiness. Until Phase 8 is implemented and passes, the
product status remains not ready.

Generate the registry ledger first:

```bash
python3 tests/agent_live/generate_harness_coverage_ledger.py \
  --output .agent/evidence/harness-coverage-ledger.json
```

Every group, question option, and typed action edge must have deterministic
evidence. High-risk multi-edge sequences additionally require dynamic CLI
evidence. Mark an edge passed only after checking checkpoint state and, where
relevant, runtime env, generated targets, job record, logs, CSV, HTML, and
artifact index.

Dynamic personas must include first-time/confused, impatient/short-answer,
copy-paste-heavy technical, contradictory integrator, expert reconfiguration,
resume/recovery, mixed-language, and report/debug users. Vary greetings,
punctuation, whitespace, typos, natural-language alternatives to options,
multiline JSON/YAML/env/curl/logs, incomplete data, corrections, repeated
jumps, backtracking, refusals, and questions while another question is active.

Mandatory sequence families:

1. fresh/resume/modify/clear and startup-discovery consultation
2. all target-mode changes, including downstream and post-job changes
3. every environment subgroup, multiple disks/interfaces, partial jumps, and
   dependency invalidation
4. Case 1 single/mixed/default/custom/replace/add/weights
5. Case 2 identity, family, endpoint, schema, fixture/smoke, and continuation
6. Case 3 evidence/handoff and return to Cases 1/2 or a configured chain
7. QPS, observability, advanced tuning, and out-of-order prerequisites
8. fake-node, real-node two-stage execution, sync-observe, jobs, failure,
   correction, retry, logs, and reports
9. compound natural-language and multiline structured input spanning several
   groups, with independent actions preserved in order
10. consultation during every major pending contract, with exact restoration

Continue with new phrasing/path combinations until the ledger has no
`not_run`/`failed` entries and two consecutive new dynamic rounds find no S1/S2
defect. Real Gemini `google_search` may be marked externally blocked; its typed
routing boundary still requires local tests.

## Local Docker Evidence

Use `tests/agent_live/local_evm_node.sh` to start/probe/stop the digest-pinned
Geth development service. It is test infrastructure, not a product dependency.
It can prove endpoint validation, custom EVM RPC traffic, real-node smoke/final
jobs, idempotency, and sync-observe metrics boundaries. It cannot prove mainnet
catch-up performance or meaningful MGas/s.

Never commit provider keys, endpoint credentials, checkpoints, job output,
transcripts, local coverage ledgers, caches, or known development keys.

## Required Verification

Run in Docker/Linux:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_agent_product_terminal \
  tests.test_agent_runtime_contract \
  tests.test_agent_langgraph_harness \
  tests.test_agent_harness_architecture \
  tests.test_agent_legacy_issue_map \
  tests.test_agent_failure_recovery \
  tests.test_agent_benchmark_pipeline
python3 tools/check_agent_boundaries.py --root .
PYTHONDONTWRITEBYTECODE=1 python3 -m compileall -q agent tests
git diff --check
```

Also run dynamic DeepSeek-backed conversations and required real executions.
Do not claim completion from unit tests or scripted transcripts alone.

## Repair Rules

When a failure appears, record its root cause and owner before editing. Repair
the shared contract, rerun neighboring paths, then continue the ledger. Never
add transcript phrases, fuzzy chain coercion, terminal-side business routing,
duplicated state, a second Agent loop, or a weaker assertion to make a failure
disappear. Remove obsolete implementations rather than isolating them as
permanent compatibility debt.
