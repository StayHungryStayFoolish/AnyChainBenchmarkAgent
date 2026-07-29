# Agent CLI Verification Guide

This guide is for an independent AI assistant that needs to verify the
AnyChain Benchmark Agent from a real terminal. It is a long-lived verification
checklist, not a temporary implementation plan.

Use fake-node closed-loop tests only unless the repository owner explicitly
provides a real endpoint for that run. Do not use private production endpoints,
customer data, or personal credentials in shared evidence.

## Goal

Verify that `./bin/anychain-agent` behaves like a product Agent:

- starts cleanly in a real terminal;
- uses the configured LLM through the LangGraph Harness, with Google ADK used
  only by the optional Gemini `google_search` grounding function;
- correctly reports the configured auth mode, including Google ADC or attached
  service-account modes when available;
- keeps input/output stable for English and Chinese;
- detects the local environment and missing dependencies;
- asks for missing benchmark variables one item at a time;
- supports fake-node benchmark preparation without skipping resource
  validation;
- handles ambiguous answers, corrections, backtracking, and mode changes;
- keeps long-running jobs detached and resumable;
- uses Gemini-only ADK `google_search` only for unknown-chain/protocol,
  custom-RPC schema, and sync-observe client-setup evidence.

The workflow metadata authority is
`agent/workflows/group_registry.py::GROUPS`: 20 `GroupSpec` entries, each with
exactly one of eight domain owners. The graph has an additional `coordinator`
control owner for typed pending-answer and graph-control actions; it owns no
group.

The current semantic graph is split into observable checkpoint transitions:

```text
prepare -> adjudicate
  -> partition
  -> compile_owner (one scheduled owner per transition; repeat as required)
  -> review_plan (independent whole-plan semantic admission)
  -> admit (deterministic action validation)
  -> select_action -> owner -> commit_action
  -> side-effect intent/invoke/receipt when required
  -> fallback -> compose -> validate
```

Only exact declared options, including declared numbered and Y/N choices, exact
terminal commands, evidence-transport framing, and empty input use the local
deterministic path. A manual typed value, natural-language alternative,
multi-intent turn, or structured block requiring semantic ownership must pass
through the split semantic path before deterministic pending/domain
validation.

## Required Reading

Read these files before testing or changing code:

1. `AGENTS.md`
2. `AI_CODING_GUIDE.md`
3. `README.md`
4. `agent/README.md`
5. `docs/en/anychain-agent-ai-work-gate.md`
6. `docs/en/adk-agent-architecture.md`
7. `tests/agent_live/README.md`

## Environment Rules

- Start from a clean checkout of the target branch and record the commit hash.
- Use an isolated Python 3.10+ Agent environment. `.venv-adk` is a retained
  compatibility path, not a statement that Google ADK is required.
- Run `bash scripts/install_agent_deps.sh --yes` before terminal testing, or
  let the Agent request approval to install missing Agent dependencies.
- Do not commit API keys, ADC files, service account JSON, `.agent/`, live logs,
  generated benchmark archives, or terminal recordings containing secrets.
- Redact credentials, local usernames, hostnames, internal project IDs, and
  private endpoint URLs from shared evidence.
- Install Google ADK only when search coverage is requested:
  `bash scripts/install_agent_deps.sh --yes --with-google-search`.
- Record whether the configured model and optional extra support ADK
  `google_search`.
- If Google/Gemini search is unavailable, still run all non-search terminal
  and fake-node boundaries.
- Put local model credentials only in `config/agent_config.local.sh` or the
  execution environment. Never edit repository defaults with real secrets.
- Do not claim that real-node execution was tested unless an approved endpoint
  was explicitly provided for that run.
- Fake-node validates only the fake-node closed loop. It is not sufficient for
  full Agent CLI workflow coverage and cannot qualify real-node,
  sync-observe, custom-RPC live-probe, or real execution edges. The Agent must
  still collect the machine/resource metadata required by the selected path.
- Acceptance runs are Docker/Linux-only. A host run may be a developer check,
  but it must not be recorded as product workflow or coverage evidence.

## Baseline Commands

```bash
git status --short --branch
git rev-parse HEAD
bash scripts/install_agent_deps.sh --yes
python3 -m agent.cli adk-status
python3 -m agent.cli llm-config
./bin/anychain-agent
```

Expected startup behavior:

- startup diagnostics run automatically;
- model provider, model name, auth mode, and web-research status are shown;
- when `LLM_AUTH_MODE=google_adc`, ADC presence and quota/project errors are
  surfaced clearly without printing credential contents;
- previous job state is shown when `.agent/jobs` has existing jobs;
- missing dependency messages do not block normal conversation unless the
  missing dependency is required for the requested action;
- the terminal prompt remains stable.

If startup reports missing benchmark-engine dependencies, ask the Agent to
explain what it will install and verify it requests approval before invoking
`scripts/install_deps.sh --yes`.

## Automated Checks

Run before any fix and again after any fix:

```bash
python3 -m unittest \
  tests.test_agent_product_terminal \
  tests.test_agent_runtime_contract \
  tests.test_agent_langgraph_harness \
  tests.test_agent_harness_architecture \
  tests.test_agent_response_authority
python3 tools/check_agent_boundaries.py --root .
git diff --check
```

Run these commands inside the Linux `bench` service. Checkpoint schema version
23 is the current contract. Migration coverage must prove the version-16
pending-owner/Chain-RPC context boundary, version-17 semantic planning,
version-18 response authority, version-19 drafts, version-20 Product Head
binding, version-21 atom evidence and secret references, version-22 durable
bindings, and version-23 signed sensitivity, memory-hard verifiers, registry
transactions, and reference-only durable plans. Version-21 raw
credentials/references and version-22 legacy bindings/references must be
quarantined.

Passing domain behavior through a test adapter that directly constructs
`review_plan` state does not prove the graph split. Split-stage evidence must
execute real `partition`, every scheduled `compile_owner`, `review_plan`, and
`admit` transitions. It must verify checkpoint recovery before and after each
stage, owner cursor/document consistency, multi-owner semantic order, planner
failure behavior, and that a completed owner is not invoked again after
resume.

Then run the LangGraph CLI matrix with the configured model. It drives the same
`./bin/anychain-agent` entrypoint users run, isolates terminal/checkpoint state
per scenario, and validates LangGraph checkpoint state:

```bash
python3 tests/agent_live/run_langgraph_cli_matrix.py
```

Use `provider=gemini` only when Gemini credentials are configured. Use another
repository-supported provider for non-search live validation, but do not claim
Google Search coverage unless Gemini ADK `google_search` is actually available.

`tests/agent_live/run_product_acceptance.py` is the sole Phase 8
evidence-admission controller. It generates revision-bound catalogs and admits
evidence created by subordinate retained-regression, real-CLI, dynamic Chaos,
and real-execution providers; it does not run those conversations or jobs
itself. Phase 8 is implemented, but G3-G6 remain open until all required
provider evidence and product-review evidence are admitted.

## Dual-AI Chaos Verification

For broad Agent workflow, routing, group-state, or Harness changes, the scripted
matrix is only a regression guard. It does not prove the product conversation is
stable.

Run an additional dual-AI chaos session:

- Start the real Docker/Linux CLI: `./bin/anychain-agent`.
- Configure the Agent to use a real provider such as DeepSeek for non-search
  validation.
- Let Codex act as the user simulator. Codex must choose each next user message
  from the live Agent response, not from a fixed list of scripted prompts.
- Save the complete transcript and review the user-visible flow, not only final
  checkpoint state.

Codex must use explicit user personas. Do not run a neutral happy-path checklist.
At minimum exercise these personas:

- first-time confused evaluator: asks who the Agent is, where to start, what
  fake-node/real-node/sync-observe mean, and what the Agent will do next;
- impatient operations engineer: gives short answers, asks whether execution
  happened, and expects direct state and next-action summaries;
- copy/paste-heavy technical user: pastes env/YAML/JSON/log/request/response
  blocks with whitespace, punctuation, and partial data;
- requirement-changing user: changes chain, target mode, QPS, RPC mode,
  custom RPC methods, endpoint, region/zone, and observability mid-flow;
- mixed-language user: alternates Chinese and English while using technical
  identifiers;
- report/debug analyst: asks for latest job, logs, reports, errors, and
  evidence interpretation;
- custom-RPC integrator: exercises supported-chain custom RPC and new-chain
  existing-family onboarding with incomplete or contradictory evidence;
- unsupported-chain evaluator: triggers secondary-development handoff and then
  returns to a supported path;
- resume-session user: starts from a previous partial checkpoint, then tests
  continue, modify, clear, and natural-language detours.

The first-time confused evaluator is a mandatory baseline. Start from a
non-empty previous checkpoint, normally a partial `sync-observe / bsc` session,
then ask questions equivalent to:

```text
who are u?
你从哪里来
你要去哪里
你可以做什么？
那我们现在可以从哪里开始？
fake node，real node，sync observe 都是什么？
fake node
```

The Agent must explain itself, explain modes, handle the old session explicitly,
and must not silently reuse a stale chain. Continue the path through resource
confirmation, workload, QPS, observability, and preflight/smoke approval. A
positive answer to the preflight/smoke prompt, such as `1`, `Y`, or `是的`, must
execute preflight/smoke or return a concrete blocker. It must not be consumed by
chain selection, generic help, or framework capability text.

After execution approval, the persona must ask:

```text
你执行过测试了么？
当前是什么状态？
那你接下来要做什么？
那你该做什么了？
```

The Agent must answer from checkpoint/job/preflight/smoke state and provide a
specific next action. Generic workflow descriptions or "no pending question"
answers fail this gate.

The Codex user simulator must cover at least these behaviors:

1. Start with arbitrary text, not only `Hi` or `hello`.
2. Answer a pending question with a valid short option, an invalid value, and a
   natural-language detour.
3. Jump between groups: disk, network, chain, target mode, workload/RPC, QPS,
   observability, sync-observe, evidence analysis, and report analysis.
4. Return/backtrack from one group to another, including a half-completed group.
5. Change chain, target mode, RPC mode, QPS profile, custom RPC methods, and
   observability after some values have already been confirmed.
6. Paste copied config in env/YAML/JSON form and require the Agent to infer,
   summarize, and ask for confirmation before applying it.
7. Paste request/response samples, endpoint URLs, official-doc excerpts, and
   contradictory evidence for custom RPC flows.
8. Exercise Case 1, Case 2, and Case 3 chain/RPC onboarding paths, then jump
   back to a supported chain or another case.
9. Switch languages and include technical scalar values with whitespace or
   punctuation.
10. Resume from a partial previous session and test continue, modify, and clear.
11. Verify execution side effects after approval: preflight/smoke/job state,
    artifact paths, or a clear blocker must exist.
12. Ask current-state and next-action questions after major transitions. The
    response must be grounded in state, not generic documentation.

If a failure appears in this session, classify it before changing code:

- terminal shell problem: input, Ctrl+C, language persistence, transcript
  rendering, dependency prompt;
- Harness state problem: group completion, fallback order, interruption stack,
  invalidation, resume;
- semantic-planning problem: LLM partition/compilation or unsupported ambiguity
  handling;
- validator/tool problem: endpoint probe, custom RPC schema, fixture, QPS,
  observability, preflight/smoke;
- documentation drift.

Fix the owning layer. Do not add terminal keyword routing, fuzzy matching, or
special-case patches that bypass the LangGraph Harness.

## Manual Terminal Tests

Run `./bin/anychain-agent` in a real terminal, not only through scripted
prompts.

### 1. Startup, Auth, And Local Discovery

Prompt:

```text
doctor
```

Expected:

- Agent reports provider, model, auth mode, and web-research status;
- Agent reports cloud/deployment discovery such as GCE, GKE, EC2, EKS,
  generic Kubernetes, VM, container, or unknown;
- Agent reports CPU, memory, network interface candidates, and disk candidates
  when the host exposes them;
- when metadata services are unavailable, Agent says what is unknown instead
  of inventing cloud, region, zone, or machine type;
- if benchmark dependencies are missing, Agent asks for installation approval
  before using `scripts/install_deps.sh --yes`;
- if Agent runtime dependencies are missing, the launcher asks before using
  `scripts/install_agent_deps.sh --yes`.

For ADC/Gemini environments, also verify:

```bash
python3 -m agent.cli llm-config
python3 -m agent.cli llm-smoke --prompt 'Return JSON only: {"ok": true}'
```

Expected:

- `llm-config` reports the configured provider/auth mode without exposing
  secrets;
- `llm-smoke` succeeds or returns a clear auth/quota/model error;
- ADK `google_search` is reported as enabled only for Gemini-family configs
  that can actually import and use the ADK search tool.

### 2. Terminal Editing And Interrupts

Test:

1. Type Chinese text with a typo, delete part of it, and retype.
2. Type English text with a typo, delete part of it, and retype.
3. Press `Ctrl+C` at an empty prompt.
4. Press `Ctrl+C` while entering text.

Expected:

- no ghost spaces;
- no duplicate `User>` prompts;
- `Ctrl+C` clears the current input or exits the current follow/log mode;
- `Ctrl+C` does not stop a detached benchmark job;
- response language follows the latest user intent except for code, variable
  names, paths, and commands.

Also test mixed-language turns:

```text
User> 我要测试 solana
User> use fake-node first
User> 改成 mixed，getSlot 70%，getBlockHeight 30%
```

Expected:

- Agent preserves technical tokens such as `solana`, `fake-node`, `mixed`,
  `getSlot`, and `getBlockHeight`;
- natural-language explanations follow the latest user language.

### 3. Fake-Node Benchmark Flow

Prompt:

```text
I want to benchmark Solana using fake-node first.
```

Expected:

- Agent asks whether this is fake-node or real-node if unclear;
- Agent confirms chain, RPC mode, benchmark mode, QPS profile, observability,
  process names, disk metadata, and network metadata;
- fake-node may default RPC endpoint values, but it must still validate
  resource metadata;
- Agent does not submit benchmark execution without preflight, smoke, and user
  approval.
- any smoke artifacts are written into isolated job/runtime directories and do
  not overwrite a later real benchmark run.

### 4. Benchmark Mode And QPS Profile

Prompt:

```text
Use quick mode. What does quick mean, and what can I change?
```

Repeat for `standard` and `intensive`.

Expected:

- Agent explains the selected profile in user-facing language;
- Agent names the relevant tunable values, such as initial QPS, max QPS,
  QPS step, duration, and RPC mode, without forcing the user to edit config
  files directly;
- Agent asks whether to accept defaults or adjust a specific item;
- if the user changes one item, only that item changes and validators re-run.

### 5. RPC Mode, Weights, And Custom Methods

Prompt:

```text
Use mixed workload with getSlot 70% and getBlockHeight 30%.
```

Expected:

- weights are normalized or rejected with a clear reason if they do not sum to
  100%;
- per-method attribution is not lost for weighted methods;
- Agent asks whether to use default chain-template methods or add custom RPC
  methods.

### 5A. Sync-Observe Without RPC Load

Prompt:

```text
Observe my BSC node while it is syncing. Do not send RPC benchmark load.
```

Expected:

- Agent routes to sync-observe, not quick/standard/intensive RPC benchmark;
- Agent does not ask for RPC mode, mixed weights, custom RPC methods, Vegeta,
  or QPS profile;
- Agent confirms chain/sync-health reference behavior, node process identity,
  optional node Prometheus metrics endpoint, disk/network metadata, and stop
  condition: until stopped, fixed duration, or until synced;
- generated report paths include sync/resource charts. MGas/s may be zero or
  unavailable if the node metrics endpoint exposes no usable gas metric, but
  the report must preserve metric source/status evidence.

### 6. Multi-Disk And Optional Accounts Disk

If the host has multiple disks, verify that the Agent shows numbered disk
candidates. If it does not, ask:

```text
If lsblk shows three disks, how will you ask me to choose LEDGER_DEVICE and ACCOUNTS_DEVICE?
```

Expected:

- numbered choices are shown;
- manual override is offered;
- `ACCOUNTS_DEVICE` is explicitly optional;
- disk baseline fields are requested when a device is selected.

### 7. Backtracking And Corrections

Prompt:

```text
I selected the wrong LEDGER_DEVICE. Go back and change it to <device>.
```

Expected:

- Agent acknowledges the correction;
- workflow state is updated or reverted;
- validators are re-run;
- next blocking question is asked.

### 8. Fake-Node To Real-Node Delta

Prompt:

```text
I tested fake-node. Now switch the same plan to a real node.
```

Expected:

- Agent reuses already confirmed resource metadata;
- Agent asks only for the delta: real `LOCAL_RPC_URL`, `MAINNET_RPC_URL` or
  sync-health decision, chain changes, RPC mode, method weights, and final
  approval;
- Agent does not force the user through the entire setup again.

If no real endpoint is provided, do not run the real-node benchmark. Verify the
planning conversation only.

### 9. Custom RPC Method

Prompt:

```text
Add a custom RPC method with three parameters to a mixed workload.
```

Expected:

- Agent asks for method name, endpoint provenance, request/response evidence,
  workload scope, and fixture plan;
- zero-parameter methods preserve explicit empty params; positional and object
  params preserve list order or object keys and individually confirm index/name,
  JSON wire type, blockchain semantic type or encoding, meaning,
  required/optional status, and example;
- the endpoint is probed using the confirmed wire request before the method is
  admitted to the versioned catalog;
- `single_replace` selects one validated method; `mixed_replace` removes
  defaults; `mixed_add` retains defaults; every active mixed weight is a
  positive integer and the exact total is 100;
- the canonical chain template remains unchanged and only a job-local runtime
  override is materialized;
- Agent does not claim production support before fixture recording and smoke
  validation.

### 10. Unsupported Chain

Prompt:

```text
I want to test a chain that is not in the current 36 templates.
```

Expected:

- Agent checks whether the chain belongs to an existing adapter family;
- if Gemini ADK `google_search` is available, the onboarding path may use it to
  find official RPC docs and examples;
- Agent asks for missing official docs, endpoints, request/response samples,
  and fixture evidence;
- Agent produces a coding handoff plan instead of pretending support exists.

Test both variants:

```text
The new chain is JSON-RPC and looks EVM-like.
The new chain does not fit any current adapter family.
```

Expected:

- same-family onboarding asks for chain-template evidence and fake-node
  fixtures;
- new-family onboarding asks for protocol docs, extractor design, adapter
  design, fixture recording, and smoke-test requirements;
- Agent does not generate unsupported production claims.

### 11. Knowledge Base Status

Prompt:

```text
Do you have an enterprise Knowledge Base connected?
```

Expected:

- Agent reports whether KB is disabled, noop, HTTP, or custom;
- if disabled, Agent explains it will answer from repository state and
  generated artifacts;
- if HTTP/custom is configured, Agent runs or suggests `knowledge-smoke`;
- KB errors do not block local fake-node benchmark validation.

### 12. Observability Choice

Prompt:

```text
Should I use the built-in Prometheus/Grafana or connect to an existing environment?
```

Expected:

- Agent explains disabled, local Prometheus/Grafana, and exporter-only modes;
- local ports are checked before local stack startup;
- for existing Prometheus/Grafana, Agent explains the exporter endpoint and
  that the external Prometheus must scrape it.
- local Prometheus/Grafana ports are not assumed available; conflicts must be
  surfaced before startup.

To force a port-conflict explanation without starting services, ask:

```text
What will you check before opening Prometheus on 9091 and Grafana on 3001?
```

Expected:

- Agent mentions port availability, exporter mode, auto-stop behavior, and
  user confirmation before local stack startup.

### 13. Detached Jobs, Logs, And Resume

After all gates pass, run only a short fake-node benchmark.

Expected:

- job runs detached by default;
- terminal shows where logs and job state are written;
- `jobs`, `status`, `logs <job_id>`, and `follow <job_id>` work through the
  Agent;
- `Ctrl+C` exits log-follow mode but does not kill the detached job;
- restarting `./bin/anychain-agent` from the same repository recovers the
  latest job state.

### 14. User Disorder And Contradictions

Test users who speak out of order:

```text
I want a benchmark.
Actually use real node.
No, switch back to fake-node.
Use mixed.
Wait, make it single.
The disk I gave before was wrong.
```

Expected:

- Agent updates intent and workflow state instead of appending contradictory
  values;
- previous answers are reused only when still valid;
- changed answers trigger re-validation;
- Agent asks the next blocking question instead of dumping a large checklist.

### 15. Out-Of-Scope Requests

Prompt:

```text
Write a trading bot for this chain.
```

Expected:

- Agent declines or redirects to benchmark-relevant capabilities;
- Agent does not execute unrelated shell commands;
- Agent can still answer benchmark-related follow-up questions afterward.

## Fixing Failures

If a scenario fails twice, inspect the redacted logs and fix the smallest
responsible code path:

- semantic planning issue: `agent/harness/hierarchical_planner.py`;
- semantic document/admission issue:
  `agent/harness/semantic_admission.py`;
- deterministic guard or tool issue: `agent/validators/` or
  `agent/tools/executor.py`;
- terminal UX issue: `agent/terminal/`;
- workflow state issue: `agent/harness/`;
- live matrix gap: `tests/agent_live/run_langgraph_cli_matrix.py`;
- documentation drift: update the relevant README or docs page.

Do not fix business behavior by adding keyword lists, fuzzy matching, or regex
intent routing in terminal code. Ambiguous semantic understanding must remain
model-driven through the LangGraph hierarchical planner and whole-plan
admission boundary, with deterministic group workflows used as validation and
execution gates.

## Evidence To Return

Return a concise report with:

- commands run;
- commit hash and branch tested;
- provider/model/auth mode used, with secrets redacted;
- whether Gemini ADK `google_search` was available;
- live matrix output paths;
- manual terminal cases tested;
- failures found and files changed;
- tests run after fixes;
- boundaries still untested, if any.

Coverage reports must distinguish cataloged/generated from observed execution.
Report observed pass, observed fail, not-run, externally blocked, uncovered
pairs/triples, completed critical sequences, and hashed real job artifacts.
Neither a PTY response nor a generated schedule/row is a passing observation.

Use this report shape:

```text
Summary:
- ...

Environment:
- branch:
- commit:
- provider/model/auth:
- google_search:

Automated checks:
- command: result, evidence path

Manual terminal boundaries:
- boundary: pass/fail, notes

Fixes:
- file: reason

Remaining gaps:
- ...
```
