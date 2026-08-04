# AnyChain Agent

`agent/` contains the AnyChain Agent control plane for the benchmark engine.
The human-facing product entrypoint is:

```bash
./bin/anychain-agent
```

The product terminal owns stable input/output, language detection, startup
diagnostics, dependency-installation consent, and job recovery commands. The
benchmark workflow itself is owned by the LangGraph Harness in
`agent/harness/`. Every provider implements the same `LLMProvider` contract:
OpenAI, DeepSeek, and Vertex Gemini use OpenAI-compatible transports; Gemini
API-key mode uses native `generateContent`, Claude API-key mode uses the
Anthropic Messages API, and Vertex Claude uses `rawPredict`. Google ADK is used
for exactly one optional capability, Gemini
`google_search` grounding for unknown-chain/protocol, custom-RPC schema, and
sync-observe client research (`agent/llm/search_grounding.py`), and must not
own product workflow state or a second conversation loop.

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
-> prepare and deterministic adjudication
-> semantic partition when exact local handling is insufficient
-> one-owner-at-a-time compilation through checkpointed transitions
-> independent whole-plan semantic review
-> deterministic action admission
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
- `chain_auxiliary_endpoints`
- `workload_rpc`
- `target_samples_fixtures`
- `qps_profile`
- `sync_observe`
- `observability`
- `advanced_tuning`
- `preflight_smoke_execution`
- `job_monitoring`
- `failure_recovery`
- `error_evidence_analysis`
- `report_artifact_analysis`

`agent/workflows/group_registry.py::GROUPS` is the sole workflow metadata
authority. It defines 20 `GroupSpec` entries and exactly one domain owner for
each group. The graph also has a `coordinator` control owner for typed
pending-answer dispatch and other graph control actions; it is not a ninth
`GroupSpec` owner.

Only an exact option id/value/label/number declared by the active question,
including declared Y/N choices, exact terminal commands, evidence-transport
framing, and empty input may use the deterministic local path. Manually entered
typed values, natural-language alternatives, multi-intent prose, and structured
content that requires semantic ownership enter the hierarchical planner before
the pending contract and owning domain validate them.

LLM output is never executed directly. The semantic path checkpoints
`partition`, repeats `compile_owner` once per scheduled owner, checkpoints
`review_plan` for independent whole-plan semantic admission, and then enters
`admit` for deterministic schema, provenance, conflict, prerequisite, and
pending-contract validation. Repository tools own configuration
materialization, execution, monitoring, and evidence-backed analysis.
If a plan contains a registered semantic value that changes product state,
`review_plan` requires two independent admissions of the same immutable plan.
Either semantic rejection or either malformed review rejects the complete
transaction. The resulting consensus receipt binds both review hashes and
request evidence to every admitted envelope; read-only plans retain one
review.

If an atomized turn contains both valid candidate actions and exact unresolved
atoms, the coordinator persists one non-executable `SemanticPlanDraft`.
Candidates remain outside business state and `action_queue`; one question is
bound to the exact draft revision and atom. After the last answer, the Harness
restores the original source contract and recompiles the complete turn with
the resolution evidence. Only a fresh whole-plan review and deterministic
admission can enqueue the complete action set. Back, cancel, reset, migration,
session/schema/registry changes, and workflow-precondition changes are handled
by the draft lifecycle rather than conversational branches. A draft can never
authorize an external execution action.

For a single source-anchored scalar pending value,
`harness/bounded_semantic_lane.py` may map once into a finite typed catalog.
Its immutable receipt binds source, candidate, question/registry/catalog,
provider/model, prompt/response hashes, and deterministic checks. A verified
bounded receipt enters deterministic admission directly; ambiguous or
out-of-catalog input uses the general hierarchical path.

The user-visible conversation is committed through a separate Product Head
authority. `harness/turn_transactions.py` creates one isolated physical
checkpoint attempt per turn, advances the logical Product Head only on commit,
and writes one immutable terminal outbox result. Cancellation, timeout, and
provider/runtime failure do not advance the head. Uncertain external effects
block new turns behind explicit reconciliation. Invariant recovery commits on
the original attempt so one user input cannot produce competing replayable
results.

All providers use one completion executor with the turn's absolute deadline,
one bounded retry budget, and fail-closed completion parsing. Read-only
requests may retry a transport failure or an otherwise normal text completion
that contains no text; abnormal finish reasons, refusals, safety outcomes,
tool calls, and alternate structured outputs are never accepted as text.
The turn-scoped collector is the only provider-attempt evidence authority.
Successful turns bind its bounded, secret-free sequence to the committed
checkpoint. Failed turns bind the same sequence to the immutable physical
attempt checkpoint and terminal diagnostic while leaving Product Head
unchanged.

Within that one deadline, only semantically independent work is executed with
bounded concurrency: frozen owner compilations and fixed open-identity,
Stage-B, and whole-plan jury members. Each worker receives its own copied
runtime context while sharing the same deadline, cancellation signal, and
thread-safe provider-attempt collector. Results and owner audit receipts merge
in canonical submission order. The first worker failure cancels and joins its
siblings and fails the complete turn; it never creates a second scheduler,
deadline, retry budget, or state authority. Dependent Stage-A convergence and
contract repair remain serial. Direct component calls outside a runtime turn
also remain serial.

Stage A receives two normalized authoritative registries: `groups` contains
group metadata, while `routing_purposes` contains each registered action's
owner, purpose, semantic operations, and route groups exactly once. Proposal,
convergence, and admission review receive both complete registries. Action
purposes must not be copied into every applicable group because that changes
wire cost without adding semantic authority.

`harness/terminal_protocol.py` defines the shared non-secret delivery
projection. The terminal renders and flushes the complete frame, durably
appends the projection, and only then acknowledges the outbox row as
delivered. A typed startup session event separately binds the process instance,
session purpose, provider/model, complete startup presentation, replayed
projection set, and originating revision. Runtime event schema v6 binds the
same transaction, terminal event, Product Head, render hash, and originating
revision; its canonical event type is separate from the specific observation.
Missing post-commit runtime observations are reconstructed from exact
committed checkpoints before terminal presentation. Live runners join these
typed identities and never infer outcomes from localized text, question IDs,
or numeric option positions, and no test callback may rewrite an event after
seeing the expected terminal outcome. Cross-process JSONL publication is
serialized with a Linux file lock. Acceptance runners require independent
runtime-event and terminal-projection producers and pass both through the
production validators before joining them.

The current live runtime contract is schema v6. It carries the explicit
Product Head authority plus the physical attempt thread and attempt checkpoint
ID, so a terminal outcome and runtime event cannot agree with each other while
pointing at the wrong attempt. Terminal outcome projection schema v3 preserves
the complete base/attempt/product lineage. Terminal detour projection schema
v3 binds the same authority and a durable runtime-event publication fence;
detours may stream or terminate a session but may never advance Product Head.
The SQLite turn-transaction ledger is schema v15. Its explicit v12-to-v13
migration assigns stable runtime-event identities to older pending committed
outcomes. Schema v14 adds the complete canonical legacy-detour record to its
audit quarantine instead of retaining only a partial hash. Schema v15 permits
an aborted provider failure to bind its immutable physical-attempt diagnostic
checkpoint while the logical Product Head remains unchanged. Current reads and
legacy migration use one shared semantic validator for the historical v10/v12
field set, row identity, typed termination, response and rolling-stream
hashes, Product Head lineage, runtime fence, lifecycle status, and
reconciliation provenance. Terminal projections independently reject unknown
effect classes and unprojectable effect statuses. Incomplete, mismatched, or
correctly hashed but semantically invalid audit records fail closed. A
v10 ledger with
committed history is rejected because that schema
cannot prove revision order; legacy detours without a publication fence are
audited and removed from live delivery. If a runtime JSONL contains a
pre-v6 event, the authority's committed publication receipts are requeued
first, the complete file is atomically retained under a content-hashed
quarantine name, and v6 events are rebuilt in contiguous Product Head revision
order from exact checkpoints. Each runtime JSONL path has a durable
single-authority marker; default paths also include an authority hash, so one
session or purpose cannot quarantine another authority's events. Startup
session schema v3 requires a ready session to have an already committed Product
Head and a complete publication fence. Interactive and one-shot terminal
entrypoints close the runtime deterministically on success, failure, EOF, and
interruption.

The product compiles one checkpointer-backed graph. Each execution transition
selects, routes, and commits at most one durable action. Side effects are
persisted as an intent before invocation and as a receipt afterward. Checkpoint
schema version 23 is the current contract. Version 12 crosses the isolated
migration boundary; version 13 migrates deferred-queue retention into the typed
pending-question contract; version 14 initializes typed response fragments
before persistence as version 15; version 16 materializes the explicit
pending-question owner and typed Chain/RPC case context; version 17 introduces
checkpointed semantic-planning state; version 18 retires persisted
turn-local response text/manifests in favor of the current response authority.
Version 19 introduces the durable, non-executable semantic-draft boundary.
Migration to version 19 clears incompatible in-flight semantic planning,
draft state, and any draft-bound question while retaining compatible durable
workflow configuration. Version 20 binds every new draft and finalization
receipt to the real Product Head checkpoint lineage and to the complete
group/action/question contract authority. Migration from version 19 clears
in-flight drafts that cannot prove those bindings. A current-version
checkpoint whose contract authority has changed invalidates the draft
fail-closed instead of attempting to execute it under the new registry.
Version 21 adds atom-level semantic evidence, opaque secret references, signed
question scheduling fields, and atomic finalization receipts; incompatible
version 20 in-flight finalizations are quarantined as complete transactions.
Version 22 adds the durable-state secret-binding registry. Version 23 makes
input sensitivity part of the signed question/group contract, separates
deterministic semantic hashes from salted memory-hard secret verifiers, and
commits process-local registry mutations with the Product Head transaction.
An exact sensitive scalar is projected as one opaque reference before the LLM
or checkpoint boundary. For compound input, deterministic typed candidates
such as endpoint URLs and explicitly structured credentials are projected
individually so sibling demands remain available to semantic planning.
Declared numbered/Y-N options remain ordinary contract values.
Checkpoints and durable preparation/execution plans persist only references
and verifiers. Raw values are materialized only at the job-local execution
boundary; job directories are owner-only (`0700`) and secret-capable files are
owner read/write (`0600`). Runtime close uses a cached ownership index and is
observable, idempotent, and retryable. A version 21 checkpoint containing raw
credentials or secret references, and a version 22 checkpoint containing old
secret bindings or references, is quarantined rather than guessed or resumed.
Missing current-version material installs a signed question-contract-v6
re-entry request before any dependent action can run.
Older state is quarantined for explicit reconfirmation.

## Main Modules

- `harness/graph.py`: the single LangGraph runtime and checkpoint wiring,
  including distinct `partition`, `compile_owner`, `review_plan`, `admit`,
  owner, commit, side-effect, fallback, composition, and validation nodes.
- `harness/coordinator.py`: graph-node transitions, the sole semantic-routing
  authority, and the sole typed commit boundary. It chooses exact
  deterministic handling, finite-catalog bounded mapping, or general planning
  without owning terminal/domain business rules.
- `harness/hierarchical_planner.py`: the sole general semantic planner; Stage A
  partitions the complete turn and Stage B compiles actions through
  owner-scoped schemas before whole-plan admission.
- `harness/bounded_semantic_lane.py`: the finite-catalog, source-anchored
  semantic mapper. It has no general routing, state-mutation, or commit
  authority.
- `harness/semantic_drafts.py`: the sole pure authority for non-admitted draft
  construction, content identities, lifecycle transitions, stale detection,
  validation, and finalization receipts.
- `harness/admission.py`: action validation, conflicts, prerequisites, and
  semantic-coverage reconciliation.
- `harness/queue.py`: dependency-safe durable action ordering and pending
  barrier eligibility.
- `harness/routing.py`: navigation prerequisites, return policy, and canonical
  fallback selection.
- `harness/response.py`: the single terminal-response assembler and
  `visible_response` writer. Domains and the coordinator emit only registered
  semantic fragments; `harness/response_catalog.py` and
  `harness/response_messages/` are the sole localized product-prose authority.
- `harness/contracts.py`: typed actions, domain results, side-effect
  intent/receipt, navigation commands, and turn receipts.
- `harness/state.py`: the sole checkpoint-schema boundary. Durable compatible
  configuration may survive migration, but every legacy in-flight question,
  queue, semantic draft, or side-effect state is quarantined and never
  recompiled into current actions.
- `harness/semantic_admission.py`: immutable semantic-document preparation and
  admission after owner-scoped compilation; it exposes no planner entry.
- `harness/bounded_semantic_lane.py`: finite-catalog, source-anchored semantic
  mapping with an immutable evidence receipt.
- `harness/turn_transactions.py`: logical Product Head, physical attempts,
  reconciliation, and terminal outbox authority.
- `harness/terminal_protocol.py`: versioned non-secret terminal projection and
  durable append contract shared with live acceptance.
- `harness/advisory.py`: model-backed chain/RPC/evidence analysis with no state
  mutation or graph-transition authority.
- `workflows/group_registry.py`: the single metadata authority for group order,
  fields, questions, dependencies, invalidations, and ownership.
- `harness/state.py`: product workflow state schema; its default group order is
  derived from `workflows/group_registry.py`.
- `harness/domains/`: eight domain owners plus the deterministic execution
  runtime used after explicit approval. `harness/domains/registry.py` derives
  its runtime owner view from `workflows/group_registry.py`.
- `terminal/repl.py`: product terminal shell and job command integration.
- `terminal/io.py`, `terminal/language.py`, `terminal/job_commands.py`: stable
  terminal support code.
- `llm/search_grounding.py`: the sole `google-adk` consumer — Gemini
  `google_search` grounding, called as a plain function from the owning
  chain-identity, RPC-endpoint, and sync-observe domain modules.
- `tools/executor.py`, `tools/schema.py`: the one programmatic tool-dispatch
  surface for `python3 -m agent.cli tool-call`/`tool-schema` (CI and enterprise
  platform integration), calling the same deterministic modules directly.
- `validators/`, `planners/`, `runners/`, `analyzers/`, `knowledge/`,
  `onboarding/`: deterministic benchmark control-plane modules reused by the
  Harness and tools.

Retired files must not return:

- `agent/workflows/conversation_state.py`
- `agent/workflows/transition_executor.py`
- `agent/terminal/input_classifier.py`
- `agent/terminal/pending_answers.py`
- `agent/harness/intent.py`
- `agent/adk_app/` (the entire package — an ADK-native `Agent`/`Runner`
  tool-calling surface that duplicated the Harness's conversation loop and
  was never the shipped product's entrypoint; its two genuinely-needed pieces
  moved to `agent/diagnostics/adk_status.py` and
  `agent/terminal/startup_state.py`, and its `google_search` capability moved
  to `agent/llm/search_grounding.py`)

## Boundary Notes

Some files remain for compatibility or shared metadata. They are allowed only
within these boundaries:

- `llm/search_grounding.py`: the only file that may import `google.adk`; it
  must stay a plain function callable from Harness code (no persistent
  Agent/Runner, no second conversation loop, no workflow state mutation).
- `workflows/group_registry.py`: pure metadata for Harness checks; no user-text
  parsing, no state mutation, and no terminal rendering.

If a future change needs product workflow control, add it to `harness/` or the
deterministic domain modules it invokes. Do not build a second Agent/Runner
tool-calling loop to do it.

## Runtime Use

Install the Agent runtime in the isolated Python environment:

```bash
bash scripts/install_agent_deps.sh --yes
```

This default installs the core LangGraph terminal runtime only. The historical
`.venv-adk`, `requirements-adk.txt`, and `--adk-venv` names remain compatibility
aliases; they do not mean Google ADK is a core dependency. Install the optional
Gemini search bridge explicitly:

```bash
bash scripts/install_agent_deps.sh --yes --with-google-search
```

DeepSeek/OpenAI/Claude and non-search Gemini operation must remain available
when `google-adk` is absent.

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
it by hand. The job directory is mode `0700`; `runtime.env`, `plan.json`,
`job.json`, and job-local logs are mode `0600`.

## Verification

Run the core no-credential contract checks:

```bash
python3 tools/check_agent_boundaries.py --root .
python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract tests.test_agent_langgraph_harness
```

Run live CLI matrix tests with configured LLM credentials:

```bash
python3 tests/agent_live/run_langgraph_cli_matrix.py
```

The live matrix drives `./bin/anychain-agent` through the same CLI path users
run and inspects LangGraph checkpoint state. It must not read or create legacy
`.agent/sessions/*/conversation_state.json` workflow files.

The fixed matrix is not product acceptance. The sole authority that can admit
evidence and report G0-G6 status is
`tests/agent_live/run_product_acceptance.py`. Phase 6 closes only the
deterministic G2 gate. Phase 8 is implemented as an evidence-admission
controller: it creates revision-bound obligation catalogs and validates
evidence produced by subordinate real-CLI, dynamic Chaos, and real-execution
providers; it does not itself conduct those conversations or jobs. G3-G6 remain
open until all required external provider evidence is produced and admitted.
Follow
`tests/agent_live/README.md` and
`docs/en/agent-handoff-product-verification.md` for dynamic
dual-AI Chaos: DeepSeek runs the real CLI while Codex chooses each next user
turn from the actual previous response. Ledger, matrix, PTY, simulator, and
execution scripts are subordinate evidence providers; their direct exit codes
cannot declare product readiness.

For local real-node and sync-observe orchestration checks, use the digest-pinned
Geth development service through `tests/agent_live/local_evm_node.sh` inside
Docker/Linux. It proves runtime wiring and artifacts, not mainnet catch-up
performance or meaningful MGas/s. Host-only runs and fake-node-only runs do not
qualify as full workflow coverage.

The execution application boundary resolves every side-effecting request
through `runners/execution_scenarios.py`. RPC smoke, RPC final benchmark, and
bounded sync-observe are distinct scenarios with strict workflow, command,
artifact, and evidence contracts; they are not interchangeable aliases.
Custom RPC execution additionally requires a current owner-issued method-probe
receipt and durable semantic probe contract. One validator recomputes the
chain, materialized endpoint identity, selected method, exact parameters,
adapter family, request/schema hash, successful HTTP observation, response
shape, evidence bytes, and catalog lineage at catalog admission and Case 2
promotion. Before real-node execution, the selected effective workload is
compared with the canonical chain template. Only selected methods absent from
that template require custom-method proof and replay against the final
materialized `LOCAL_RPC_URL`; historical catalog methods outside the workload
are not execution requirements. The endpoint is committed only after the owner
receipt binds every required custom-method evidence one-to-one. Persisted
endpoint values remain opaque secret references; raw endpoint material crosses
only the probe or job-local tool boundary.
Product gate G5 requires four fresh jobs from three distinct approved plans:
fake-node smoke, Geth real-node smoke, Geth real-node final, and bounded
sync-observe. Reused job identities and missing hashed runtime/report artifacts
fail closed.

## CLI Tools For Automation

`python3 -m agent.cli` exposes JSON commands for CI and enterprise Agent
platforms:

```bash
python3 -m agent.cli adk-status
python3 -m agent.cli capabilities
python3 -m agent.cli doctor --output doctor.json
python3 -m agent.cli plan --request request.json --output plan.json
python3 -m agent.cli preflight --plan plan.json
python3 -m agent.cli submit --plan plan.json --approved
python3 -m agent.cli status --job-id <job_id>
python3 -m agent.cli analyze --job-id <job_id>
```
