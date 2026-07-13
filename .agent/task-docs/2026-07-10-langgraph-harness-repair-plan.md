# LangGraph Harness Repair Plan (Phased)

Status: draft task/design document, not yet reviewed/approved.
Git policy: do not commit this file unless the user explicitly asks.
Supersedes a broken reference: `.agent/task-docs/2026-07-07-langgraph-agent-harness-refactor.md`
is named as a mandatory read in `.agent/task-docs/2026-07-10-agent-handoff-for-external-ai.md`
but does not exist anywhere in git history on this branch or `main`. Treat that
reference as stale; this document is the current authoritative task/design
document.

File-by-file reading receipt: `.agent/task-docs/2026-07-10-agent-code-reading-coverage.md`
confirms all 105 `.py` files under `agent/` have been read in full (not
sampled) and points each one at its relevant finding, if any. Read that
first if you need to confirm a specific file was actually audited before
trusting a claim about it below.

Evidence base: `.agent/task-docs/2026-07-10-architecture-audit-findings.md`
is a full first-hand read of 100% of `agent/harness/`, `agent/terminal/`,
`agent/adk_app/`, `agent/onboarding/`, `agent/validators/`,
`agent/planners/`, and `agent/workflows/` (not a sample). It found that the
repeated instability is not primarily keyword/fuzzy-matching pollution (that
class of bug was already fixed in this branch — see its "What Was Checked
And Found Clean" section) but a structural pattern repeated at least six
times: two or more independent implementations of the same concept (group
order, next-action computation, reset semantics, intent classification,
question wording, adapter-family enumeration) that are expected to stay in
sync by hand and have already drifted. Read that document before touching
any file listed below; it names exact line numbers for every claim here.

The phases below are ordered by dependency and risk, not by the order
failures were reported. Phase 0 and Phase 1 are prerequisites for Phase 2-4
being safe to do at all: you cannot reliably fix Case1/Case2 duplication
(Phase 3) while two different group-order lists and two different
next-action engines can disagree about what group you are even in.

## Phase 1: Resume/Reset/Discovery Semantics

Status: implemented and verified. `agent/harness/state.py` (added
`RESET_PRESERVED_KEYS`), `agent/harness/graph.py` (`reset()` now preserves
the same keys `groups._reset_workflow_state` does, both derived from the
shared constant), `agent/harness/intent.py` (added `startup_discovery`
topic to both live prompts), `agent/harness/oracle.py` (added
`format_startup_discovery`), `agent/harness/groups.py` (wired the topic,
plus one additional fix below). Verified via 160 unit tests (2 new:
`test_graph_runtime_reset_preserves_startup_discovery`,
`test_startup_discovery_topic_explains_inference_separately_from_confirmed_config`),
`tools/check_agent_boundaries.py`, `agent/cli.py adk-eval`, `git diff
--check`, and two live runs against the configured DeepSeek model
reproducing Matrix 1 from the handoff document end to end (not just
against a mocked LLM resolver).

### Additional bug found and fixed during live verification (not in the original audit)

A live DeepSeek run of Matrix 1 surfaced a real bug the mocked-LLM unit
tests could not: `agent/harness/groups.py:_answer_fits_pending`'s
`manual_value` and `device` branches, and the `url` kind's generic
fallback branch, accepted `_is_plain_scalar_answer(raw)` /
`_is_single_turn_value(raw)` as sufficient to treat free text as an answer
to the current pending question — unlike the sibling `confirm_or_value`
branch, which already also required `not _looks_like_user_question(raw)`.
Concretely: while `CLOUD_ZONE` was pending, the turn "你有一个环境依赖的检测脚本，
这个脚本会帮我推断一些变量，这些变量推断了么" was silently accepted as the literal
value for `CLOUD_ZONE` instead of being routed to intent classification,
because `_looks_like_user_question` did not recognize colloquial Chinese
yes/no questions ending in a sentence-final particle (了么/呢) without a
"？" character. Fixed by (1) adding the same `and not
_looks_like_user_question(raw)` guard to the `manual_value`, `device`, and
generic `url` branches, and (2) teaching `_looks_like_user_question` to
recognize "吗" and sentence-final 么/呢. Covered by a new regression test,
`test_colloquial_question_without_question_mark_does_not_answer_manual_value_pending`.
This is the same failure class as the handoff document's "Failure D"
(consultation misrouted to workflow), just for a different question kind
(`manual_value`/`device` rather than `yes_no`) — evidence that live
dual-AI chaos testing (Phase 7) will keep surfacing this pattern in other
kinds/branches that were not exercised by this specific scenario.

### Root Cause (confirmed by code reading, not by transcript patching)

### Root cause 1: `resume_harness_session` is not a Harness pending question at all

`agent/terminal/repl.py:344` creates `self.state.current_question_id =
"resume_harness_session"` as a **terminal-local** field
(`TerminalSession.current_question_id`), never written into the LangGraph
state's `pending_question`. `repl.py:_handle_pending_confirmation` (lines
272-321) then hand-parses the user's raw reply (`"1"/"continue"`,
`"2"/"modify"/"change"`, `"3"/"clear"/"reset"/"fresh"`) and, for option 3,
calls `self._ensure_harness().reset(language=...)` directly
(`repl.py:293`).

This is a second, terminal-owned workflow state machine, exactly the pattern
`AGENTS.md` / `docs/en/anychain-agent-ai-work-gate.md` forbid ("Do not add a
second workflow brain in terminal ... code").

### Root cause 2: two incompatible `reset` implementations exist, and the terminal calls the destructive one

- `agent/harness/graph.py:45-55` `AnyChainGraphRuntime.reset()` builds a
  completely fresh `AgentGraphState` via `state.new_state()` and pushes it as
  a full-schema patch. `state.new_state()` sets `"discovery": {}`
  (`agent/harness/state.py:132`), so this wipes startup environment inference
  along with confirmed config.
- `agent/harness/groups.py:266-274` `_reset_workflow_state()` is the
  Harness-owned implementation backing the already-existing `reset_session`
  typed intent action (`groups.py:636`, documented in `intent.py:137,443`).
  It explicitly preserves `discovery`, `framework_summary`, `web_research`,
  `latest_job_id`, `job`, and `report_context` while clearing user
  configuration (`groups.py:270-272`).

`repl.py:293` calls the first (destructive) implementation instead of routing
through the existing `reset_session` action, which would use the second
(correct) implementation. This is the direct cause of Failure A in the
2026-07-10 handoff document: "已清空之前的 Agent 配置会话" followed by the
Agent being unable to answer whether startup inference still exists — because
`reset()` actually deleted it.

### Root cause 3: no topic formats a "what was inferred at startup" answer

Even where `discovery` survives, nothing reads it back out for the user.
Searched `agent/harness/intent.py`, `agent/harness/groups.py`, and
`agent/harness/oracle.py`: no topic, action, or formatter exists for
"startup discovery summary" (confirmed by full-text search for `discovery`,
`environment inference`, `discovery_summary`). `oracle.py`'s
`format_current_state`/`format_current_context` only ever surface
`chain_identity`, `target_mode`, `qps_profile`, `observability`, and
`confirmed_config`.

### Root cause 4 (verify before fixing): Failure B may already be wired

`agent_capabilities` is a real, existing topic: `intent.py:141,285,439,441`
document `answer_opening_question` with `topic=agent_capabilities` for "what
can the Agent do" questions, and `groups.py:5387-5388` routes it to
`_agent_capabilities_response` (`groups.py:5566`). Per `AI_CODING_GUIDE.md`
("Debug by investigating... Reproduce the problem"), do not assume Failure B
needs new plumbing. First reproduce it live; if the plumbing already answers
correctly, the only remaining gap is intent-classification recall for that
exact phrasing, which is an `intent.py` prompt-quality fix, not an
architecture fix.

## Scope

### In scope for Phase 1

1. `agent/terminal/repl.py`: remove the bespoke `resume_harness_session`
   terminal-side state machine (`current_question_id` branch in
   `_handle_pending_confirmation`, lines 272-304, and its creation at line
   344). Replace it with a Harness-owned mechanism: on startup, when a prior
   session snapshot exists, the Harness sets a real `pending_question` (e.g.
   `pending_question.id == "resume_session"`) as part of graph state. The
   user's reply is resolved through the existing typed intent action queue
   (`resume_session_continue` / `resume_session_modify` / `reset_session`),
   the same way every other pending question in the product is resolved.
   `reset_session` already exists and already preserves `discovery`
   correctly (`groups.py:636`, `_reset_workflow_state`) — reuse it verbatim,
   do not write a new reset path.
2. `agent/harness/graph.py`: `AnyChainGraphRuntime.reset()` must not be
   reachable from a "clear previous config" user action. Either remove it, or
   rename/restrict it to true process-level fresh-session bootstrapping
   (e.g. `--fresh-session` CLI flag at process start, before any state
   exists), and assert (via a boundary test) that no in-conversation user
   action can call it.
3. `agent/harness/intent.py` + `agent/harness/groups.py` +
   `agent/harness/oracle.py`: add one new topic, e.g.
   `startup_discovery_summary`, under `answer_opening_question` (same family
   as `agent_capabilities`, `current_config`). It must:
   - list inferred values (CLOUD_PROVIDER, deployment, CPU, memory, disk
     candidates, network candidates, dependency status), reading from
     `state["discovery"]`;
   - list not-yet-inferred/needs-confirmation values (CLOUD_REGION,
     CLOUD_ZONE, MACHINE_TYPE, LEDGER_DEVICE, volume type/iops/throughput,
     accounts disk existence, target mode, chain, RPC workload, QPS,
     observability);
   - explicitly state whether confirmed config is currently empty, using
     `confirmed_config`, never conflating it with `discovery`.
4. Add regression tests in `tests/test_agent_langgraph_harness.py` and/or
   `tests/test_agent_product_terminal.py` for:
   - clearing config preserves `discovery`;
   - a follow-up "什么被推断了" / "还有配置么" style question after clearing
     config returns the discovery summary, not fallback text;
   - `resume_session` is a real `pending_question` in graph state, not a
     terminal-local field (guard test similar to the existing
     `test_terminal_uses_langgraph_harness_not_retired_workflow_runtime`).
5. Reproduce Matrix 1 and the transcript in Failure B from the handoff
   document against a live model (DeepSeek, per handoff doc) before claiming
   either is fixed.

### Explicitly out of scope for Phase 1 (do not touch)

- `agent/harness/groups.py` file-size/duplication cleanup (the
  `_new_chain_schema_conflict`/`_custom_rpc_schema_conflict` and
  `_validate_custom_rpc_schema`/`_validate_new_chain_rpc_schema` near-duplicate
  pairs). Real, but separate, larger effort — track as Phase 2.
- Consolidating business-policy duplication between `intent.py`'s
  natural-language prompt rules and `groups.py`'s deterministic checks (e.g.
  QPS-mode-explicit logic in both places). Track as Phase 2.
- Pinning `google-adk`/`langgraph`/`langgraph-checkpoint-sqlite` versions in
  `requirements-adk.txt` and building `.venv-adk`. Track as Phase 0
  (environment hygiene) — recommended to do in parallel since it is
  independent and low-risk, but it is not a code-behavior fix and must not be
  mixed into the same commit as the Phase 1 changes above.
- Failure C (unknown chain `sola` Case2 prompt) and Failure D (workload
  consultation misrouting). Not yet root-caused at the code level; require
  their own investigation pass before a fix is written, per the "Design Gate
  Before Code" rule. Track as Phase 3/4.

## Files In Scope

- `agent/terminal/repl.py`
- `agent/harness/graph.py`
- `agent/harness/intent.py`
- `agent/harness/groups.py`
- `agent/harness/oracle.py`
- `agent/harness/state.py` (only if a new `pending_question` shape requires a
  schema note; no field semantics should change)
- `tests/test_agent_langgraph_harness.py`
- `tests/test_agent_product_terminal.py`

## Files That Must Not Be Touched In Phase 1

- `agent/adk_app/**` (no product workflow logic lives there; unrelated to
  this root cause)
- `agent/validators/**`, `agent/planners/**`, `agent/runners/**` (unrelated
  deterministic domain modules)
- `config/chains/*.json` (chain templates; unrelated to session reset/resume)

## Success Criteria

1. Matrix 1 from the handoff document passes against a live model in Docker:
   clearing config, then asking about startup inference, produces a
   discovery summary, not fallback-only text.
2. `resume_session` exists as a graph-state `pending_question`, resolved only
   through typed intent actions; `current_question_id` in
   `TerminalSession` no longer special-cases `resume_harness_session`.
3. `AnyChainGraphRuntime.reset()` is not reachable from any in-conversation
   user action; a boundary/regression test enforces this.
4. `tests.test_agent_product_terminal`, `tests.test_agent_langgraph_harness`,
   and `tools/check_agent_boundaries.py --root .` all pass.
5. Failure B transcript is re-run live; result recorded as pass, or if it was
   already passing, recorded as "already correct, no change made" with
   evidence.
6. No new keyword/fuzzy/regex business routing introduced anywhere,
   especially not in `repl.py`.

## Verification Commands

```bash
python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract tests.test_agent_langgraph_harness
python3 tools/check_agent_boundaries.py --root .
python3 agent/cli.py adk-eval
git diff --check
```

Then, in Docker, with DeepSeek configured as the Agent model:

```bash
docker compose exec -T bench bash -lc 'cd /workspace && source config/agent_config.sh >/dev/null 2>&1 || true && .venv-adk/bin/python -m unittest tests.test_agent_product_terminal tests.test_agent_langgraph_harness -v'
docker compose exec bench bash -lc 'cd /workspace && source config/agent_config.sh >/dev/null 2>&1 || true && ./bin/anychain-agent --fresh-session'
```

Run Matrix 1 and the Failure B transcript manually or via a Codex/other-AI
chaos user, and capture the transcript as evidence per the handoff document's
"Definition Of Done" section.

## Rollback / Cleanup Expectations

- If removing `resume_harness_session` from `repl.py` breaks other terminal
  flows (e.g. `install_agent_runtime`/`install_dependencies` pending
  questions, which are legitimately terminal-only), keep those two intact —
  they are not business/workflow routing, they are dependency-install
  consent, which `AGENTS.md` explicitly allows in terminal code.
- If `AnyChainGraphRuntime.reset()` is still needed for `--fresh-session`
  process bootstrap, keep it but rename to make misuse obvious (e.g.
  `bootstrap_fresh_state()`), and add a boundary check so no
  conversation-turn code path can call it.
- Remove `.agent/checkpoints/*.sqlite` test artifacts created during manual
  verification; do not commit them.

## Phase 2: Single Source Of Truth For Group Order And Next-Action

Status: **done and verified.** Implemented:

1. Extracted `agent/harness/routing.py` with the single
   `next_group_and_reason(state)` implementation; both
   `groups._next_group` and `oracle._next_group_and_reason` now delegate to
   it (Finding B1 closed).
2. `hardware_discovery` removed from `state.DEFAULT_GROUP_ORDER` (no spec
   anywhere, per project-owner decision).
3. `advanced_tuning` fully implemented (was a declared-but-dead group):
   asks about `MONITOR_INTERVAL`, `DISK_MONITOR_RATE`,
   `SUCCESS_RATE_THRESHOLD`, `MAX_LATENCY_THRESHOLD`, and 8
   `BOTTLENECK_*_THRESHOLD` fields grounded in the real
   `config/user_config.sh`/`config/internal_config.sh` variable names —
   explain defaults, ask Y/N, and support a per-field adjustment loop on N.
4. `chain_auxiliary_endpoints` fully implemented (was declared only in
   `agent/workflows/group_registry.py`, never in `state.py` or `groups.py`):
   uses the pre-existing `agent.planners.chain_template_requirements.inspect_chain_template()`'s
   `runtime_endpoint_variables` to ask **only** about the specific
   `CHAIN_REST_URL`/`CHAIN_INDEXER_URL`/`CHAIN_SIDECAR_URL`/`CHAIN_EVM_RPC_URL`/`CHAIN_JSON_RPC_URL`/`CHAIN_MIRROR_URL`/`RPC_API_KEY`
   fields a chain's template actually substitutes at runtime (currently only
   `RPC_API_KEY`, for `starknet`/`bch`/`dogecoin`/`litecoin`) — most chains
   need none of these seven fields and skip the group silently.
5. `agent/workflows/group_registry.py`'s `GROUP_ORDER` now exactly matches
   `state.DEFAULT_GROUP_ORDER` (added `job_monitoring`, removed
   `hardware_discovery`, kept `chain_auxiliary_endpoints` and gave it real
   `questions=`); its docstring's false "source of truth... used by the
   LangGraph Harness" claim was corrected.
6. `agent/harness/intent.py`'s `ALLOWED_GROUPS` now imports
   `state.DEFAULT_GROUP_ORDER` directly instead of retyping it.

**Critical bug found and fixed during implementation, not part of the
original plan**: see architecture audit Finding G. `advanced_tuning` was
never added to the `AgentGraphState` TypedDict, so LangGraph silently
dropped it on every real turn despite all mocked-resolver unit tests
passing. Caught only by live DeepSeek CLI verification. Fixed, and guarded
going forward by two new tests:
`test_agent_graph_state_typed_fields_match_new_state_keys` (static
invariant — must hold for every future field) and
`test_advanced_tuning_state_survives_langgraph_checkpoint` (real
`AnyChainGraphRuntime` round-trip, not a direct `process_turn` call).

Verified via: 169 unit tests (10 new), `tools/check_agent_boundaries.py`,
`agent/cli.py adk-eval`, `git diff --check`, full `agent/` `py_compile`, and
live DeepSeek CLI runs driving the complete `chain_auxiliary_endpoints` and
`advanced_tuning` flows end to end (RPC_API_KEY prompt for `starknet`,
skipped entirely for `bsc`; advanced tuning Y-path and full N-then-adjust-
then-finish loop with a real field override recorded).

**Known follow-up, not yet done**: natural-language phrasing like "我想改一下
CPU 瓶颈阈值" (I want to adjust the CPU bottleneck threshold) while
`advanced_tuning_confirm` is pending gets classified as
`answer_opening_question topic=config_explanation` (an explanation) rather
than being treated as an implicit "N" / redirect into the adjustment flow.
`agent/harness/intent.py` has an explicit prompt rule for this exact pattern
for QPS ("if the user wants to configure QPS but doesn't name a mode, use
change_group group=qps_profile"); no equivalent rule exists yet for
advanced tuning. Direct Y/N and numbered-choice answers work correctly; only
this specific ambiguous-natural-language redirect case is unhandled. Low
priority since it degrades to "explain the field" rather than corrupting
state. Tracked as Phase 6 item 7, with the exact fix and test to add.

<details>
<summary>Original planning notes (superseded by the summary above)</summary>

Status: blocked on one product decision (see below); not started.

**Decision needed before coding**: tracing the live routing path confirmed
exactly which group names `groups.py` can ever produce (16 — see audit
document's "Finding A Addendum"). Two declared groups are dead:
`advanced_tuning` and `hardware_discovery`. `advanced_tuning` corresponds to
a real, documented feature in `docs/en/anychain-agent-ai-work-gate.md`
("Advanced tuning group": monitoring intervals, disk monitor rate,
thresholds — the Agent must explain these before asking whether to change
them) that was apparently never implemented. Two options:
(a) remove `advanced_tuning`/`hardware_discovery` from the canonical group
list, accepting that this documented feature is out of scope for now, or
(b) implement them as real groups with `_question_for_group` branches before
unifying the group-order lists. This is a product-scope decision, not a
structural cleanup, so it should not be made silently.

Root cause: see architecture audit Finding A and B1. Three group-order lists
disagree (`agent/harness/state.py:75-95`, `agent/harness/intent.py:26-45`,
`agent/workflows/group_registry.py:23-242`), a fourth hardcoded routing
chain (`groups.py:_next_group`) reads none of them, and `oracle.py`
independently reimplements the same routing chain
(`oracle.py:_next_group_and_reason`) for display purposes only.

Plan:

1. Choose `agent/harness/state.py`'s `DEFAULT_GROUP_ORDER` as canonical (it
   is the state schema owner). Add `hardware_discovery` handling to
   `_next_group`/`_question_for_group` or remove it from
   `DEFAULT_GROUP_ORDER` if it is intentionally not yet implemented — do not
   leave a group name that nothing can ever select.
2. Delete `agent/workflows/group_registry.py`'s `GROUP_ORDER`/`GROUPS` if
   nothing in the live conversational path needs it, or make
   `agent/validators/config_contract.py` import the canonical order instead
   of maintaining its own. Update `agent/README.md`'s description of
   `group_registry.py`'s role to match reality.
3. Make `intent.py`'s `ALLOWED_GROUPS` derive from the canonical list
   (import it) instead of retyping it.
4. Extract `oracle.py`'s `_next_group_and_reason` to call
   `groups._next_group` (or move the shared logic to a function both files
   import) so status text and live routing cannot diverge.

Files in scope: `agent/harness/state.py`, `agent/harness/intent.py`,
`agent/harness/groups.py`, `agent/harness/oracle.py`,
`agent/workflows/group_registry.py`, `agent/validators/config_contract.py`,
`agent/README.md`.

Success criteria: exactly one place defines group membership/order; a test
asserts `intent.ALLOWED_GROUPS`, `state.DEFAULT_GROUP_ORDER`, and any group
list `config_contract.py` uses are the same object or derived from the same
source; `oracle.format_current_state`'s reported next group always matches
what `groups.process_turn` would actually ask next for the same state.

</details>

## Phase 1/2 Code Review — Findings And Fixes (done and verified)

A high-effort `/code-review` pass (8 finder angles + 1-vote verify) ran over
the full Phase 1 + Phase 2 diff before Phase 3 started, per explicit user
request. 5 of 8 finder angles (all 3 correctness angles, plus simplification
and efficiency) returned usable results; the other 3 (reuse, altitude,
conventions) stalled mid-execution after a context compaction and were
killed rather than re-polled indefinitely. 3 candidate bugs were verified
CONFIRMED and fixed immediately (not deferred):

1. **`_looks_like_user_question`'s "current" substring false-positive**
   (`agent/harness/groups.py`). Phase 1 added `and not
   _looks_like_user_question(raw)` guards to the `manual_value`, `device`,
   and generic `url` branches of `_answer_fits_pending` (previously only
   `confirm_or_value` had this guard). The pre-existing `"current"` marker
   was a plain substring check, so any legitimate single-token answer
   containing "current" (e.g. an API key `concurrent-tier-01`, a device path
   `/dev/current-disk`) was wrongly rejected as looking like a question.
   Fixed by requiring a whitespace-bounded match for English question words
   (`what`/`why`/`how`/`which`/`current`) — a bare single-token identifier
   cannot be an English sentence, so the check now only fires on multi-word
   text. CJK markers are unaffected (Chinese has no space-delimited words).
   Regression test:
   `test_scalar_value_containing_current_as_substring_is_not_treated_as_a_question`.

2. **`CONFIRMABLE_CONFIG_FIELDS` missing the 7 new `chain_auxiliary_endpoints`
   field names** (`agent/harness/groups.py`). The new group's fields
   (`CHAIN_REST_URL`, `CHAIN_INDEXER_URL`, `CHAIN_SIDECAR_URL`,
   `CHAIN_EVM_RPC_URL`, `CHAIN_JSON_RPC_URL`, `CHAIN_MIRROR_URL`,
   `RPC_API_KEY`) were never added to the allowlist `_build_config_proposal`
   checks, so a proactively-pasted value (e.g. "RPC_API_KEY: abc123") fell
   into `unmapped_values` instead of `confirmed_config`, and the harness
   would re-ask for a value the user already supplied. Fixed by adding all 7
   field names to `CONFIRMABLE_CONFIG_FIELDS`. Regression test:
   `test_proactively_pasted_chain_auxiliary_field_is_applied_not_discarded`.

3. **`AnyChainGraphRuntime.reset()` vs `groups._reset_workflow_state()`
   `audit_events` divergence** (`agent/harness/graph.py`). Phase 1's stated
   goal was to unify the two reset paths via `RESET_PRESERVED_KEYS`, but
   `graph.reset()` never appended the `{"event": "workflow_reset"}` marker
   that `groups._reset_workflow_state()` does, despite `graph.reset()`'s own
   docstring claiming the two "must agree on what reset preserves." Fixed by
   appending the same marker. Regression test: extended
   `test_graph_runtime_reset_preserves_startup_discovery` to assert the
   marker is present.

Lower-severity cleanup items surfaced by the Simplification/Efficiency
angles but not fixed (tracked here so they are not lost, per the user's
"will this survive to Phase N" concern): `oracle._next_group_and_reason` is
now a redundant one-line pass-through wrapper around
`routing.next_group_and_reason` with only one call site — candidate for
deletion in Phase 3/4 when call sites are touched anyway;
`chain_template_requirements.inspect_chain_template()` re-reads and
re-parses the chain JSON from disk on every call with no caching, which
`routing.chain_auxiliary_fields_needed()` now calls on every turn —
low-cost today (small JSON files) but worth a session-scoped cache if this
path gets busier; `_ADVANCED_TUNING_DEFAULTS`/`_ADVANCED_TUNING_FIELD_LABELS`
are two hand-synced dicts with no test enforcing their keys match.

All 158 tests pass (156 pre-existing + 2 new regression tests) after these
fixes: `.venv-adk/bin/python -m pytest tests/test_agent_langgraph_harness.py -q`.

## Phase 3: Consolidate The Two Question-Generation Engines

Root cause: see architecture audit Finding C1. `agent/validators/config_contract.py`
(`build_missing_config_questions`) and `agent/planners/config_questions.py`
(`required_questions`) are two independently-worded full question engines
that both run inside `prepare_benchmark_run` and return unreconciled results.

Plan:

1. Decide which engine is canonical. Recommendation: neither — both should
   stop generating prompt text. `agent/harness/groups.py`/`oracle.py`
   already own question wording for the live conversation; `config_contract.py`
   and `config_questions.py` should shrink to pure blocker/field validators
   (return which fields are missing, not what to say about them) consumed by
   `agent/planners/strategy_planner.py`'s preflight path.
2. If full removal is too large for one pass, at minimum: stop returning
   both question sets from `prepare_benchmark_run`; pick one, and have the
   other's call sites (`agent/tools/executor.py`, `agent/adk_app/tools/validators.py`)
   consume the same one.
3. Update the four-to-five independent "required fields" catalogs (Finding
   C2: `agent/workflows/requirements.py`, `agent/planners/config_checklist.py`,
   `agent/validators/config_contract.py`'s `WORKLOAD_CONFIRMATION_KEYS`/
   `SYNC_OBSERVE_BLOCKERS`, `agent/planners/strategy_planner.py`'s
   `_ordered_required_inputs`, `agent/harness/groups.py`'s
   `CONFIRMABLE_CONFIG_FIELDS`) to derive from one list.

Files in scope: `agent/validators/config_contract.py`,
`agent/planners/config_questions.py`, `agent/planners/config_checklist.py`,
`agent/planners/strategy_planner.py`, `agent/workflows/requirements.py`,
`agent/adk_app/tools/planning.py`, `agent/adk_app/tools/validators.py`,
`agent/tools/executor.py`.

Success criteria: a given missing field produces identical prompt wording
regardless of which call path (live conversation vs. preflight) surfaces it;
`prepare_benchmark_run`'s response contains one question list, not two.

**Status: done and verified (full architectural fix, per user's explicit choice
of "full architectural fix" over the minimal-catalogs-only option).**

What shipped:

- `agent/knowledge/entry_contract.py`'s `RuntimeField` dataclass extended with
  `description`/`applies_to`; two new tuples (`WORKFLOW_CONFIRMATION_FIELDS`,
  `SYNC_OBSERVE_FIELDS`); `field_specs_for(mode)` as the single ordered/mode-scoped
  field catalog; `ENV_TO_KEY` moved here from `config_contract.py`. This is now
  the canonical source Finding C2's five mismatched catalogs derive from.
- New `agent/planners/question_prompts.py` — the single module that authors
  configuration-question prompt text. `agent/harness/groups.py`'s wording
  (the live conversation's "real" author) was relocated here verbatim for the
  ~15 fields both `config_contract.py`/`config_questions.py` also asked about
  (CLOUD_REGION/CLOUD_ZONE/MACHINE_TYPE, network fields, chain question, device
  prompts, benchmark mode, QPS profile, observability mode, sync-observe stop
  condition, workload customization, advanced tuning, chain-auxiliary fields,
  has_accounts_device). `agent/harness/groups.py`, `config_contract.py`, and
  `config_questions.py` all now call into this module instead of hand-typing
  strings — this is a new (permitted) `harness → planners` dependency, verified
  clean by `tools/check_agent_boundaries.py`.
- `agent/workflows/requirements.py`, `agent/planners/config_checklist.py`,
  `agent/validators/config_contract.py` (`SYNC_OBSERVE_BLOCKERS`),
  `agent/planners/strategy_planner.py` (`_ordered_required_inputs`), and
  `agent/harness/groups.py` (`CONFIRMABLE_CONFIG_FIELDS`'s static portion) all
  now derive their field lists from `entry_contract.py` instead of hand-typed
  literals. Two catalogs whose live *order* is itself product behavior
  (`SYNC_OBSERVE_BLOCKERS`, `_ordered_required_inputs`) keep an explicit
  ordered literal with a runtime `assert` that it stays in sync with the
  canonical catalog's membership, rather than silently reordering live
  question sequencing.
- `build_missing_config_questions` and `required_questions` keep their exact
  original signatures/return shapes (both are registered ADK tool names in
  `agent/tools/schema.py`) — only internal prompt-string sourcing changed.
  `agent/tools/executor.py`, `agent/adk_app/tools/validators.py`,
  `agent/runners/runbook.py`, `agent/adk_app/tools/actions.py` needed no
  changes; verified via new shape-contract tests instead.
- While reconciling wording, the new tests caught two more real Finding-C1
  instances beyond what the original audit named: `config_questions.py`'s
  `checklist["environment"]` loop built its own generic
  `f"Confirm {description}"` text for `cloud_region`/`cloud_zone`/`machine_type`
  instead of matching `groups.py`/`config_contract.py`'s wording; and
  `has_accounts_device` had three different phrasings across
  `config_questions.py`, `config_contract.py`, and `groups.py`. Both fixed to
  use `question_prompts.text_for("has_accounts_device")`/canonical text.
- Scoping decisions made deliberately, not from time pressure: (1) `groups.py`'s
  deeply state-dependent, non-duplicated branches (endpoint validation,
  new-chain workload-scope flows, sync-observe source selection menus) were
  left untouched — they have no counterpart in the other two engines, so
  touching them would add risk to a 5800-line dispatcher with no Finding-C1
  benefit. (2) `benchmark_mode_confirmed`/`observability_choice_confirmed`/
  `sync_observe_stop_condition`'s *candidate option list descriptions* (not
  the prompt text) were left as-is in `config_questions.py` where they were
  richer than `config_contract.py`'s — only the `"prompt"` field itself was
  reconciled, which is what Finding C1 was actually about. (3) Two checklist
  *descriptions* (not conversational prompts) for `chain`/`ledger_device`
  lost sync-observe-specific phrasing when derived from the single-description
  `RuntimeField` model (e.g. "used for sync-resource charts" became the
  generic "used for disk charts") — a disclosed, low-impact simplification,
  not a behavior bug, since these are internal checklist item descriptions,
  not the actual question prompts (already reconciled separately).
- New test file `tests/test_agent_question_prompts.py` (7 tests): wording
  agreement between the two engines, full field-catalog prompt coverage,
  `prepare_benchmark_run`-equivalent reconciliation, the 5-catalogs-collapsed-to-1
  invariant, `runbook.py`/`actions.py` shape-contract guards, and one wording
  pin for `groups.py`'s live conversational text (`CLOUD_REGION`).
- All 406 tests pass (399 pre-existing + 7 new):
  `.venv-adk/bin/python -m pytest tests/ -q` (excluding 4 pre-existing,
  unrelated pandas-import failures in visualization test files — pandas is
  not installed in this isolated `.venv-adk`, not something this phase
  touched). `tools/check_agent_boundaries.py` and `python3 agent/cli.py
  adk-eval` both pass.

### Phase 3 Code Review — Findings And Fixes (done and verified)

A high-effort `/code-review` pass (8 finder angles + verification) ran over
the Phase 3 diff specifically (scoped to exclude the already-reviewed Phase 1/2
hunks, since nothing was committed between phases). 3 real bugs were found,
verified live, and fixed immediately:

1. **`_prompt_for_key`/`_required_prompt` never forwarded the real target
   mode** (`agent/validators/config_contract.py`, `agent/planners/config_questions.py`).
   The "chain" question always rendered "Current target mode: not selected,"
   even when real-node/fake-node/sync-observe was already confirmed — a
   regression introduced by routing the chain prompt through
   `question_prompts.chain_prompt`, which is target-mode-aware, without
   threading the mode through. Fixed by deriving `target_mode` from each
   engine's own already-available context (the `target_mode` parameter in
   `config_contract.py`; `plan.get("use_fake_node")`/sync-observe detection in
   `config_questions.py`) and passing it through. Regression test:
   `test_chain_prompt_reflects_already_known_target_mode`.

2. **`question_prompts.qps_profile_prompt()` dropped a sanitization
   `config_contract.py` used to have**: an unrecognized `benchmark_mode_confirmed`
   value used to render as "selected" instead of being echoed back next to
   numbers that were never chosen for it (e.g. "Default QPS profile for
   `bogus_mode`: INITIAL_QPS=1000..."). Fixed by restoring the "selected"
   fallback for any non-empty, unrecognized mode while keeping "quick" as the
   fallback for the empty-mode case (matching `groups.py`'s pre-existing
   behavior, since `groups.py` remains the wording's real author). Regression
   test: `test_qps_profile_prompt_does_not_echo_an_unrecognized_mode`.

3. **The two new anti-drift checks (`SYNC_OBSERVE_BLOCKERS` /
   `_REQUIRED_INPUT_PRIORITY` vs. `entry_contract.field_specs_for`) used
   `assert`**, which `python -O`/`PYTHONOPTIMIZE=1` strips entirely, silently
   disabling the single-source-of-truth guarantee in optimized runs. Fixed by
   converting both to explicit `if not (...): raise RuntimeError(...)` checks.

Also applied as low-risk cleanups (no behavior change, multiple review angles
converged on these independently): removed the `_qps_profile_confirmed_prompt`
wrapper in `question_prompts.py` (gave `qps_profile_prompt` a `**_ignored` and
referenced it directly in the dispatch dict); replaced
`config_checklist.py`'s triple `field.key != "..."` chain with a named
`_ENVIRONMENT_REVIEW_KEYS` set; trimmed `entry_contract.ENV_TO_KEY`'s manual
override dict from 6 entries to the 2 that are actually not derivable from a
`RuntimeField.env` (the other 4 were exact duplicates of what the
comprehension above already produced).

Findings not fixed, deliberately, with rationale recorded so they aren't
re-litigated: (a) `_options_from_candidates` and `_field_for` remain
independently duplicated in both `config_contract.py` and `config_questions.py`
— pre-existing duplication Phase 3 didn't fully close given the scope
decision to focus on prompt *wording* (Finding C1's actual subject), not
every shared helper; (b) `entry_contract.field_specs_for`'s inability to
express a custom per-consumer order (forcing `SYNC_OBSERVE_BLOCKERS`/
`_REQUIRED_INPUT_PRIORITY` to keep explicit ordered literals) is a real
architectural limitation, but redesigning it risks a live behavior change
(question ordering) that wasn't asked for; (c) `cloud_region`/`cloud_zone`/
`machine_type` are absent from `_REQUIRED_INPUT_PRIORITY` despite being in
the canonical real-node catalog — confirmed to be a **pre-existing** gap
(present before Phase 3 touched this file), not a regression, and out of
scope to silently "fix" by reordering; (d) `agent/adk_app/tools/read_only.py`'s
`_field_payload` doesn't surface `RuntimeField`'s new `description`/`applies_to`
fields to its ADK tool output — an optional enhancement, not a bug, since
nothing currently depends on that tool exposing them.

All 408 tests pass (406 + 2 new) after these fixes:
`.venv-adk/bin/python -m pytest tests/ -q`. `tools/check_agent_boundaries.py`
and `python3 agent/cli.py adk-eval` both still pass.

## Phase 4: Collapse Case1/Case2 Structural Duplication In `groups.py`

Root cause: see architecture audit Finding D. Roughly 400-500 lines across
`_validate_custom_rpc_schema`/`_validate_new_chain_rpc_schema`,
`_custom_rpc_schema_conflict`/`_new_chain_schema_conflict`,
`_validated_custom_methods`/`_validated_new_chain_methods`, and the
`custom_rpc_*`/`new_chain_*` branches inside `_apply_endpoint_answer` and
`_endpoint_active_validation_question` implement the same
probe-endpoint-then-validate-method-then-confirm-schema-then-scope-then-weights
state machine twice, parameterized only by which state dict (`custom_rpc` vs
`chain_identity`) and question-id prefix is used.

Plan: extract one parameterized implementation (e.g. a small
`_EndpointValidationContext` describing which state key, question-id prefix,
and result-application callback to use) and have both Case 1 and Case 2 call
it. Do this only after Phase 2 and Phase 3 land, since this function touches
`_question_for_group`/`_next_group` call sites that Phase 2 changes.

Files in scope: `agent/harness/groups.py` only.

Success criteria: line count for the endpoint-validation logic drops by
roughly the duplicated amount; existing Case 1 and Case 2 tests in
`tests/test_agent_langgraph_harness.py` pass unmodified (behavior must not
change, only structure); a bug fix applied to one case's flow (e.g. schema
conflict detection) is now structurally impossible to "forget" for the other
case.

**Status: done and verified.** A deep investigation found the duplication was
**not** pure "swap the dict key" parameterization — several real behavioral
differences existed, two of which looked like bugs. The user was asked how to
handle them and explicitly chose: **unify Case 1 to Case 2's more complete
behavior for both** (an authorized, deliberate behavior change; everything
else preserves exact current behavior per the success criteria above).

What shipped:

1. **`_validated_custom_methods`/`_validated_new_chain_methods`** collapsed
   into one `_validated_rpc_methods(case_dict, *, fallback_key)`; both
   originals are now 1-line wrappers (pure parameterization, no behavior
   change).
2. **`_validate_custom_rpc_schema`/`_validate_new_chain_rpc_schema`** collapsed
   into `_validate_rpc_schema(state, params, *, case)`. Along the way, fixed a
   genuine pre-existing bug found during investigation: Case 1's adapter-family
   resolution read `chain_identity.adapter_family` directly, which is *always*
   empty in practice for Case 1 (the confirmation path that enters Case 1 never
   sets it) — it only "worked" because `validate_rpc_endpoint` has its own
   independent fallback lookup. Both cases now use the robust `_chain_adapter_family(state)`
   helper Case 2 already had. Case-specific wording, the `visible_response`
   append-vs-overwrite difference, the `active_group`-setting difference, and
   the Case-1-only inline-workload-hint shortcut were all preserved exactly.
3. **`_custom_rpc_schema_conflict`/`_new_chain_schema_conflict`** collapsed
   into `_rpc_schema_conflict(state, draft, *, case)`, with the authorized
   Case-1 behavior change: Case 1's schema/protocol conflict recovery now
   actively re-asks the adapter family (new `custom_rpc_adapter_family_confirm`
   question + answer branch, extracted `_adapter_family_options`/
   `_route_unsupported_adapter_family` helpers) instead of only printing an
   explanatory message. Verified that reusing Case 2's `_protocol_family_question`
   verbatim would have **misrouted** the answer (its handler unconditionally
   resets `chain_identity` into Case-2-entry mode) — Case 1 needed its own
   question id and answer branch, routed through the `endpoint_process`
   group/dispatcher rather than Case 2's `chain_identity` one. Deliberately
   chose to clear `custom["endpoint_ready"]=False` unconditionally in both
   conflict directions for Case 1's brand-new code, rather than copying Case
   2's own internal asymmetry (which only clears it one way).
4. **Single-method disambiguation** (the other authorized change): Case 1 now
   asks which validated method to use as `single` when 2+ methods are
   validated (new `custom_rpc_single_method` question + answer branch,
   modeled directly on Case 2's `new_chain_single_method`), instead of
   silently taking the first one. New shared `_single_method_disambiguation_question`
   helper used by both cases (Case 2's existing call site in `_question_for_group`
   was refactored to use it too — pure dedup there, no behavior change).
5. **Deliberately did NOT relocate** Case 2's `new_chain_workload_scope`/
   `single_method`/`custom_weights` question-building (in `_question_for_group`)
   to live alongside Case 1's (in `_endpoint_active_validation_question`),
   despite the original plan sketch suggesting this — verified during
   implementation that `_endpoint_active_validation_question` is called
   *before* several shared real-node/sync-observe questions in `_question_for_group`,
   so relocating Case 2's questions there would have silently reversed their
   priority relative to those shared questions, with zero test coverage to
   catch it. Both question-building call sites were left exactly where they
   are; only the shared pure helper functions were extracted.
6. **Backfilled a real, pre-existing test-coverage gap**: Case 2's own
   `existing_family_needs_single_method`/`new_chain_single_method` path had
   zero prior test coverage, despite Case 1 being unified to match its
   behavior. Added `test_new_chain_workload_scope_single_replace_with_multiple_methods_asks_new_chain_single_method`
   against the pre-refactor code first, to confirm the investigation's
   description of Case 2's actual behavior before relying on it as the model
   for Case 1's new behavior.

New tests (5): the Case-2 backfill above, plus
`test_custom_rpc_scope_single_replace_with_multiple_methods_asks_disambiguation`,
`test_custom_rpc_scope_single_replace_with_one_method_skips_disambiguation`
(positive/negative pair for the single-method change), and
`test_custom_rpc_schema_conflict_rest_evidence_reasks_adapter_family`,
`test_custom_rpc_schema_conflict_jsonrpc_evidence_from_rest_reasks_adapter_family`
(both conflict directions, modeled on Case 2's existing
`test_new_chain_rest_evidence_does_not_probe_as_jsonrpc_method`).

**Explicitly not touched, flagged as follow-ups** (found during investigation,
not authorized to fix — same discipline as prior phases' "found but not
fixed" notes):
- `_apply_custom_rpc_inline_workload_hint` has its own separate
  `methods[0]`-no-question shortcut for the same single-method-disambiguation
  scenario, reached via the Case-1-only inline-hint path rather than the
  `custom_rpc_scope` answer branch that was fixed. Same bug class, different
  code path, not requested.
- `custom_rpc_weights`'s answer branch lacks the missing/unknown-method
  validation that `new_chain_custom_weights` has (checking for validated
  methods missing from the weight spec, or unknown methods present) — only
  checks that the total equals 100.
- The `visible_response` append-vs-overwrite divergence between Case 1
  (append) and Case 2 (overwrite) in the collapsed `_validate_rpc_schema`.
- `active_group` being set at nearly every transition in Case 2's `_apply_endpoint_answer`
  branches but only once (at `custom_rpc_continue`) in Case 1's — appears
  benign since Case 1 has no equivalent "force a different group" step, but
  not unified.

All 413 tests pass (408 + 5 new):
`.venv-adk/bin/python -m pytest tests/ -q`. `tools/check_agent_boundaries.py`
and `python3 agent/cli.py adk-eval` both still pass.

### Phase 4 Code Review — Findings And Fixes (done and verified)

A high-effort `/code-review` pass (8 finder angles + verification) ran over
the Phase 4 diff specifically (scoped to exclude the already-reviewed Phase
1/2/3 hunks in the same files). One angle (line-by-line) independently
live-reproduced a real, high-severity bug; the rest converged on a mix of
confirmed cleanup items and disclosed-but-not-fixed follow-ups.

Fixed:

1. **`agent/harness/routing.py`'s custom_rpc status allowlist never extended
   for the two new Phase 4 statuses** (`needs_single_method`,
   `needs_adapter_family_confirmation`). This file is the single shared
   "what's next" source of truth for both `groups._next_group` (live
   routing) and `oracle`'s status text — its own docstring's whole point is
   that the two can never disagree. Live-reproduced: calling
   `next_group_and_reason(state)` with `custom_rpc.status="needs_single_method"`
   returned `("workload_rpc", "choose RPC mode")` instead of
   `("endpoint_process", ...)`, meaning the harness would silently skip past
   the pending disambiguation/re-ask question while `custom_rpc.status`
   stayed stuck forever. Fixed by adding both statuses to the allowlist.
   Regression test: `test_routing_recognizes_phase4_custom_rpc_statuses`.
2. **`agent/workflows/group_registry.py` not extended for the two new
   question ids** — the `endpoint_process` questions tuple,
   `SETUP_QUESTION_IDS`, and the id→group alias table had the parallel Case
   2 ids (`new_chain_single_method`, etc.) but not the new Case 1 ones
   (`custom_rpc_adapter_family_confirm`, `custom_rpc_single_method`). No
   live caller hit this gap today, but `config_contract.py` does import from
   this registry — left unfixed, this was exactly the kind of drift Phase 2
   was built to prevent. Fixed by adding both ids to all three structures.
3. **Unreachable `return None` in `_rpc_schema_conflict`** — a merge
   artifact from collapsing the two old conflict functions' trailing
   fallbacks, flagged independently by 4 of 8 review angles. Deleted.
4. **Duplicated adapter-family validity guard** between
   `_enter_case_for_adapter_family` (Case 2 entry) and the new
   `custom_rpc_adapter_family_confirm` answer branch (Case 1 re-ask) — both
   independently did the identical strip/assign/validate-or-route-to-handoff
   sequence. Extracted into a shared `_confirm_adapter_family(state, family) -> bool`
   helper used by both.

Found but deliberately not fixed, recorded so they aren't re-litigated:

- **`_route_unsupported_adapter_family` leaves stale `custom_rpc` state when
  reached via Case 1's conflict-recovery re-ask** (it only ever touches
  `chain_identity`/`secondary_handoff`, never resets
  `custom_rpc.status`/`endpoint_ready`). Not a live bug today (the turn stops
  via `_stop_after_response`), but a real latent inconsistency if a future
  code path ever resumes back into `endpoint_process` after an abandoned
  Case-1-triggered handoff. Needs a small design decision about what
  "cleanup" means for Case 1's state before fixing, not a one-line patch.
- **`load_framework_capabilities()` has no caching**, and Phase 4's fix to
  Case 1's adapter-family resolution (unifying it to the more robust
  `_chain_adapter_family(state)` helper, see Phase 4's main section above)
  means this uncached disk scan is now reachable from the `custom_rpc` path
  too when `chain_identity.adapter_family` is unset. Considered adding
  `functools.lru_cache`, but at least three call sites
  (`framework_index.py`, `gap_analyzer.py`, `framework_context.py`) pass an
  explicit `root` param that tests may point at temp directories with
  changing content across a single test run — caching risks subtle test
  flakiness that would need its own verification pass, so left as a
  follow-up rather than fixed opportunistically here.
- **`custom_rpc_weights` still lacks the missing/unknown-method validation
  `new_chain_custom_weights` has** — confirmed still true (not accidentally
  partially unified), same pre-existing, disclosed asymmetry from Phase 4's
  main implementation notes above.

All 414 tests pass (413 + 1 new):
`.venv-adk/bin/python -m pytest tests/ -q`. `tools/check_agent_boundaries.py`
and `python3 agent/cli.py adk-eval` both still pass.

## Phase 5: Fix Harness → ADK Dependency Inversion

Root cause: see architecture audit Finding E. `agent/harness/nodes/execution.py`
imports execution functions from `agent/adk_app/tools/{actions,planning}.py`,
inverting the documented dependency direction (ADK should depend on nothing
in `harness/`; `harness/` should not depend on `adk_app/`). `planning.py`'s
`_apply_assumed_smoke_defaults` (lines 569-699) and `actions.py`'s
`_fake_node_smoke_plan`/`run_quick_assumed_fake_node_smoke` contain real
assumed-config/QPS business decisions that do not belong in a "bridge" layer.

Plan: extract the assumed-defaults and plan-preparation logic into a new or
existing module under `agent/planners/` or `agent/runners/` (peer to
`strategy_planner.py`/`job_manager.py`). Have both
`agent/harness/nodes/execution.py` and `agent/adk_app/tools/{actions,planning}.py`
import that shared module as thin wrappers.

Files in scope: `agent/adk_app/tools/actions.py`,
`agent/adk_app/tools/planning.py`, `agent/harness/nodes/execution.py`, plus
a new/target module under `agent/planners/` or `agent/runners/`.

Success criteria: `agent/adk_app/` has no inbound import from
`agent/harness/`; a dependency-direction check (can be as simple as a grep in
`tools/check_agent_boundaries.py`) enforces this going forward.

**Status: done and verified.**

An investigation (Explore agent + direct reads of the three files in scope)
refined the audit's claim before implementation: the three imported functions
(`prepare_benchmark_run`, `run_fake_node_smoke_benchmark`,
`submit_benchmark_job`) are dual-purpose — registered, `approved`-gated ADK
tool callables *and* the only place the actual discover → plan → preflight →
runbook and materialize-smoke-plan → submit-job pipelines live. Extracting
only `_apply_assumed_smoke_defaults`/`_fake_node_smoke_plan` (the two
functions the audit named) would not have been enough — `execution.py` would
still import the outer three functions themselves. `_apply_assumed_smoke_defaults`
also turned out to not be reachable via the Harness's actual call path today
(`execution.py`'s `_prepare_kwargs` never sets `assumed_for_smoke=True`; it's
only reached via the ADK-only `run_quick_assumed_fake_node_smoke`) — moving it
was still required for the dependency-direction rule to hold, just not a
live-behavior fix for the Harness specifically.

Implementation:
- New `agent/runners/tool_result.py`: the ADK-agnostic `{status, data,
  evidence_paths, warnings, next_actions, requires_user_confirmation}`
  envelope builder, moved out of `agent/adk_app/tools/read_only.py`'s private
  `_tool_result` and renamed to a public `tool_result()` since it's now
  imported across package boundaries. `read_only.py`, `actions.py`,
  `planning.py`, `auth.py`, `validators.py` all import it back under the same
  `_tool_result` local alias, so no call site elsewhere changed.
- New `agent/runners/benchmark_pipeline.py`: owns `prepare_benchmark_run`,
  `run_fake_node_smoke_benchmark`, `submit_benchmark_job` (the real pipeline
  logic, moved verbatim from `planning.py`/`actions.py` minus the
  `approved`/`_confirmation_required` gate, which is an ADK/LLM-safety
  concern, not pipeline logic — the Harness already gates via its own
  `preflight.approved` state before ever calling in), plus their private
  helpers (`_structured_request`, `_apply_assumed_smoke_defaults`,
  `_inferred_values`, `_prepare_next_actions`, `_target_mode_for_plan`,
  `_confirmed_values_from_plan`, `_fake_node_smoke_plan`,
  `_FAKE_NODE_IGNORED_REQUIREMENTS`, `_job_terminal_commands`,
  `_job_user_next_actions`, `_nested_job`).
- `agent/adk_app/tools/planning.py`'s `prepare_benchmark_run` and
  `agent/adk_app/tools/actions.py`'s `run_fake_node_smoke_benchmark`/
  `submit_benchmark_job` are now thin wrappers: same signature/docstring (the
  LLM-facing tool contract), body is the `approved` gate (for the two
  action tools) followed by a single call into the pipeline core.
  `draft_benchmark_request` (which also needed `_structured_request`) and
  `run_quick_assumed_fake_node_smoke` (which needed `_job_terminal_commands`/
  `_nested_job`) now import those private helpers from
  `agent.runners.benchmark_pipeline` instead of defining them locally.
- `agent/harness/nodes/execution.py` now imports `prepare_benchmark_run`,
  `run_fake_node_smoke_benchmark`, `submit_benchmark_job` from
  `agent.runners.benchmark_pipeline` directly; its two call sites that used
  to pass `approved=True` now call with no `approved` kwarg (the core
  functions don't take one — the Harness's own state-machine approval already
  gated whether the call happens at all).
- `tools/check_agent_boundaries.py`: added a new check (`HARNESS_FORBIDDEN_MARKERS
  = ["adk_app"]`) that scans every `.py` file under `agent/harness/` and fails
  if any contains `adk_app` — there was previously no check for this
  direction at all, only the reverse (ADK files must not reach into
  `agent.harness`). Verified passing after the fix.
- New `tests/test_agent_benchmark_pipeline.py` (5 tests): direct coverage of
  `prepare_benchmark_run`/`run_fake_node_smoke_benchmark`/`submit_benchmark_job`
  on `agent.runners.benchmark_pipeline` (previously zero direct coverage
  existed for any of these three functions), including a test asserting the
  ADK wrapper and the core function produce identical `inferred_values` for
  the same input — a direct regression guard for "thin wrapper, no behavior
  change." `tests/test_agent_question_prompts.py`'s
  `test_fake_node_smoke_plan_filters_required_questions_by_id` updated to
  import `_fake_node_smoke_plan` from its new home.
- Verified end-to-end: called `run_approved_preflight_and_smoke` directly
  with a fake-node state through the new import path — plan generation,
  preflight blocking, and the blocked-message's evidence-path threading (which
  reads both the inner `data` dict and the outer envelope's `evidence_paths`)
  all worked identically to before.

Not touched, out of scope: `agent/tools/executor.py` (a fourth caller of the
same three ADK-side names, for a separate schema-driven dispatcher — keeps
importing from `agent.adk_app.tools.*` unchanged, unaffected since those
names/signatures/behavior didn't change from its point of view);
`load_framework_capabilities()`'s lack of caching (unrelated pre-existing note
from Phase 4's review).

All 419 tests pass (414 + 5 new). `tools/check_agent_boundaries.py` and
`python3 agent/cli.py adk-eval` both pass.

### Phase 5 Code Review — Findings And Fixes (done and verified)

A high-effort `/code-review` pass ran over the Phase 5 diff (manually scoped
file list, since nothing is committed). Two of eight finder angles (cross-file
tracer, reuse) converged independently on the same duplication finding; a
third stalled mid-execution after a context-window compaction and was killed
rather than re-polled indefinitely (same pattern as Phase 1/2's review). Two
real bugs were verified live and fixed immediately:

1. **`agent/harness/nodes/execution.py`'s `_smoke_message` iterated the
   `terminal_commands` dict directly instead of its values** — `_job_terminal_commands`
   always returns a dict (`{"status": ..., "logs": ..., ...}`), never the list
   `_smoke_message` defaulted to, so iterating it yielded the dict's *keys*.
   Live-verified before/after: the fake-node smoke confirmation message read
   "Use: status; logs; follow; analyze" (the literal field names) instead of
   "Use: status job-123; logs job-123; follow job-123; analyze latest job"
   (the actual runnable commands). This bug predates Phase 5 (the function
   itself wasn't touched by the move) but was only surfaced by the review's
   cross-file trace of the `terminal_commands` shape now that it's produced
   by the new shared `benchmark_pipeline.py`. Fixed by switching to
   `commands.values()`, matching the sibling `_job_message` helper in the
   same file. Regression test:
   `test_smoke_message_renders_actual_terminal_commands_not_dict_keys`.
2. **`agent/adk_app/tools/read_only.py` kept its own byte-for-byte duplicate
   of `_job_terminal_commands`/`_job_user_next_actions`** instead of
   importing the versions now canonically owned by
   `agent.runners.benchmark_pipeline` — found independently by both the
   cross-file-tracer and reuse angles. This directly defeated this phase's
   own stated goal ("business logic lives in exactly one place"), since
   `actions.py` already imports these same names from the new module. Fixed
   by having `read_only.py` import both helpers from `benchmark_pipeline`
   and deleting its local copies. Regression test:
   `test_read_only_job_terminal_commands_is_the_shared_pipeline_helper`
   (asserts object identity, not just equal output, so the duplicate cannot
   silently reappear).

Found but deliberately not fixed, recorded so they aren't re-litigated:

- **`run_quick_assumed_fake_node_smoke`'s hardcoded 13-item `confirmations`
  list is dead** — it's a strict subset of the 24-item set
  `_apply_assumed_smoke_defaults` unconditionally unions in whenever
  `assumed_for_smoke=True`, which this call always passes. Pre-existing (this
  function's body wasn't touched by Phase 5, only its helper imports moved);
  best addressed alongside Phase 6's dead-code cleanup rather than
  opportunistically here.
- **`prepare_benchmark_run` calls `_run_doctor()` with no arguments**,
  causing `diagnostics/doctor.py` to recompute discovery from scratch via a
  second full `discover_environment()` subprocess scan instead of reusing the
  dict already computed one line above. Confirmed pre-existing — the original
  `planning.py` had the identical pattern before this phase's move, so this
  is not a regression introduced here. Worth a follow-up passing
  `_run_doctor(discovery)` to halve this cost on every approved-preflight
  turn and every ADK `prepare_benchmark_run` call.
- **The ADK wrapper functions (`prepare_benchmark_run`, `draft_benchmark_request`)
  re-spell all ~45 parameter names in their forwarding calls into the
  pipeline core** — a future parameter addition could be added to one
  signature but not the forwarding call, silently dropping the field. This
  matches the established convention for every other ADK tool wrapper in the
  codebase (explicit named parameters, needed for ADK tool-schema
  introspection); collapsing to `**locals()` would reduce schema clarity and
  is a larger behavioral change than this phase's pure-structural-move scope.

All 421 tests pass (419 + 2 new): `.venv-adk/bin/python -m pytest tests/ -q`.
`tools/check_agent_boundaries.py` and `python3 agent/cli.py adk-eval` both
still pass.

## Phase 6: Cleanup — Dead Code And Prompt Contract Drift

**Status: done and verified.** All items (1-7) complete. Verified via 427
tests + 10 subtests (excluding 4 pre-existing pandas-import failures in
visualization test files — pandas is not installed in `.venv-adk`), `tools/check_agent_boundaries.py
--root .`, `python3 agent/cli.py adk-eval`, `git diff --check`, and full
`agent/` `py_compile`, all green.

Item 1 (earlier pass): deleted `agent/harness/events.py` (confirmed zero
references); removed `agent/diagnostics/doctor.py:format_doctor_report` (dead;
`cli.py`'s `doctor` command uses the generic JSON `_emit` path instead);
removed `agent/knowledge/gap_analyzer.py:answer_gap_question` and its four
support-only private helpers (`_find_chain`, `_find_unknown_chain`,
`_find_methods`, `_format_gap_answer`); removed
`agent/knowledge/entry_contract.py:field_by_key` (dead).

What shipped for items 2-7 (per the user's explicit "no technical debt"
directive, item 3 was resolved by deletion, not the keep-both option):

- **Item 5 (dead `answer` field)**: removed the never-read `answer` field from
  both intent action schemas (the `_action_queue_prompt` schema string, the
  `_action_queue_payload` output schema, and the now-deleted `_payload`). A
  full-text search confirmed nothing ever read `action["answer"]`, so raw model
  text could never reach `visible_response` through it.
- **Item 7 (advanced-tuning redirect)**: added the "adjust advanced tuning /
  monitoring / bottleneck threshold without an active `advanced_tuning` pending
  question → change_group group=advanced_tuning" rule to `_action_queue_prompt`
  (the `_system_prompt` copy was made moot by item 3's deletion). Regression
  test `test_free_text_advanced_tuning_adjust_redirects_into_advanced_tuning_group`
  drives it from the free-text phrasing "我想改一下 CPU 瓶颈阈值" with
  `resolve_action_queue` mocked to return the redirect action, proving the
  change_group→advanced_tuning plumbing fires.
- **Item 4 (capabilities topic normalization)**: `groups._opening_consultation_response`
  mapped the `capabilities` topic to `supported_chains` (the raw chain list),
  contradicting the sibling `capability`/`what_can_you_do` normalization
  (agent capabilities). Fixed `capabilities` → `agent_capabilities`, and removed
  the now-redundant bare `capabilities` value from `intent.py`'s topic enum
  (kept in the normalization map as a tolerated synonym). Regression test
  `test_bare_capabilities_topic_maps_to_agent_capabilities_not_chain_list`.
- **Item 6 (adapter-family single source)**: `agent.onboarding.families.SUPPORTED_FAMILIES`
  is now the sole definition; `groups.SUPPORTED_ADAPTER_FAMILIES`,
  `intent.ADAPTER_FAMILIES`/`ADAPTER_FAMILY_HINT_ENUM` (5 former retypes across
  prompt strings and payloads), `gap_analyzer.py:onboarding_plan`'s prompt
  string, and `template_drafter.py`'s two proxy-transport subsets
  (`REST_TRANSPORT_FAMILIES`/`JSONRPC_TRANSPORT_FAMILIES`) all derive from it.
  `groups._adapter_family_options` derives its option values from the canonical
  list with a minimal `_ADAPTER_FAMILY_OPTION_LABELS` override for the one
  family whose label differs ("jsonrpc / EVM"). Fixed
  `endpoint_probe.py:_should_use_generic_jsonrpc_probe`'s value-drift set
  `{"jsonrpc","evm","ethereum","ethereum_jsonrpc"}` → the canonical
  `GENERIC_JSONRPC_PROBE_FAMILIES = {"jsonrpc"}`; verified the EVM aliases were
  provably unreachable (`_adapter_family_hint_from_text` normalizes "EVM"→
  "jsonrpc" upstream, and chain templates store the canonical `jsonrpc`).
  Regression tests `test_adapter_family_lists_derive_from_single_source` and
  `test_generic_jsonrpc_probe_uses_canonical_family_not_evm_alias`.
- **Item 3 (single free-text resolver — resolved by deletion)**: investigation
  confirmed `resolve_intent_action` (backed by `_route_single_action`, prompt
  `_system_prompt`, payload `_payload`) handled a strict *subset* of the action
  types `resolve_action_queue`/`_apply_queue_action` handle, using a strictly
  weaker prompt — a vestigial second LLM call reached only when the queue
  resolver already returned nothing meaningful. Deleted all four (resolver,
  `_route_single_action`, `_system_prompt`, `_payload`) so free-text turns route
  through exactly one resolver. Found and fixed a real latent gap: a lone
  `greeting` action was excluded by `_has_meaningful_queue`, so greetings had
  been handled *only* by the now-deleted fallback — added `greeting` to the
  meaningful-action set so the queue path handles it. Converted the 15 tests
  that mocked only `resolve_intent_action` to mock `resolve_action_queue`
  (`{"intent": X}` → `{"actions":[{"type": X}]}`); dropped the redundant
  `resolve_intent_action` patch from the 4 tests that mocked both; converted the
  1 terminal test likewise. Guard test
  `test_single_free_text_resolver_is_the_action_queue` asserts the deleted
  symbols cannot reappear.
- **Item 2 (handoff nav → resolver)**: removed the `_looks_like_handoff_navigation`
  keyword blocklist. `_route_secondary_handoff_text` keeps the deterministic
  evidence/generation shape gates, then routes the evidence-vs-navigation
  decision through the shared `resolve_action_queue` (`_handoff_text_is_navigation`
  + the `_HANDOFF_NAVIGATION_ACTIONS` set), so an evidence-shaped turn that is
  actually a navigation/command (e.g. "analyze my latest http report") is routed
  away instead of captured. Regression test
  `test_case3_handoff_evidence_shaped_navigation_routes_through_resolver`; the
  two evidence/generation handoff tests gained explicit `resolve_action_queue`
  mocks so they deterministically exercise "evidence-shaped + not-navigation →
  capture."
- **Review-pass cleanups (self-review before code-review)**: removed the bare
  `capabilities` enum value (item 4); refactored `_adapter_family_options` to
  derive from the canonical list (item 6); and stripped three pre-existing
  trailing-blank-line-at-EOF whitespace errors in `agent/adk_app/tools/{actions,
  planning,read_only}.py` (Phase 5 artifacts) that `git diff --check` flagged.

Root cause: see architecture audit Findings B4, B5, B6, B7, C3, F.

Plan:

1. ~~Delete `agent/harness/events.py`~~ — done.
2. Remove `_looks_like_handoff_navigation`'s keyword-list routing
   (`groups.py:5214-5236`); route Case-3 evidence-vs-navigation
   classification through `resolve_action_queue`/`resolve_intent_action`
   like every other ambiguous-text decision.
3. Resolve the two intent-resolver generations (`resolve_action_queue` vs
   `resolve_intent_action` in `intent.py`): either delete the older one and
   fold its still-needed fallback trigger conditions into the action-queue
   resolver, or explicitly keep both but make them share one policy text
   source so they cannot drift again.
4. Fix the `capabilities`/`agent_capabilities` topic normalization mismatch
   (`groups.py:5356-5365`) so every enum value documented in `intent.py`'s
   schema (`intent.py:285`) maps to the response the schema text implies.
5. Remove the unused `answer` field from both action schemas in `intent.py`
   (lines 299, 484), or wire it up deliberately with an explicit boundary
   test proving raw model text still cannot reach `visible_response`
   unformatted.
6. Unify the six-copy adapter-family list (Finding C3) into one constant
   imported everywhere, and fix `agent/validators/endpoint_probe.py:108,111`'s
   divergent `{"jsonrpc","evm","ethereum","ethereum_jsonrpc"}` set, which is
   not just reordered but contains different values than the canonical list.
7. **Add an `intent.py` redirect rule for ambiguous "adjust advanced tuning"
   phrasing** (found live during Phase 2, recorded here so it is not lost to
   context compaction — see Phase 2's "Known follow-up" note for the exact
   reproduction). Concretely: a turn like "我想改一下 CPU 瓶颈阈值" ("I want
   to adjust the CPU bottleneck threshold") sent while
   `advanced_tuning_confirm` is the pending question currently gets
   classified as `answer_opening_question topic=config_explanation` (an
   explanation of the field) instead of being redirected into the
   advanced-tuning adjustment flow. Direct `Y`/`N`/numbered-choice answers
   already work correctly (verified live); only this specific
   natural-language-redirect phrasing is unhandled. Fix by adding a rule to
   both `_action_queue_prompt()` and `_system_prompt()` in
   `agent/harness/intent.py` mirroring the existing QPS pattern at
   `intent.py:172` ("If the user wants to configure or adjust QPS but does
   not name quick/standard/intensive, use change_group group=qps_profile
   instead.") — i.e. add: "If the user wants to adjust advanced tuning /
   monitoring / bottleneck threshold settings without an active
   `advanced_tuning` pending question, use change_group
   group=advanced_tuning." Add a regression test mirroring
   `test_advanced_tuning_adjustment_loop_records_override_then_finishes` in
   `tests/test_agent_langgraph_harness.py`, but driving it from this
   free-text phrasing (with `resolve_action_queue` mocked) instead of a
   pre-set `pending_question`, to prove the redirect actually fires.

Files in scope: `agent/harness/events.py` (delete),
`agent/harness/groups.py`, `agent/harness/intent.py`,
`agent/onboarding/families.py`, `agent/onboarding/template_drafter.py`,
`agent/validators/endpoint_probe.py`.

Success criteria: `tools/check_agent_boundaries.py --root .` (extended if
needed) flags any future reintroduction of a keyword-routing function
outside `terminal/language.py`'s rendering-only scope; a single grep for the
six adapter-family names returns exactly one list definition plus its
importers.

### Phase 6 Code Review — Findings And Fixes (done and verified)

A high-effort code-review pass (8 finder angles) ran over the Phase 6 diff
specifically (intent.py resolver deletion, groups.py queue/handoff changes,
adapter-family unification, endpoint_probe). One real correctness bug was
found and fixed; three lower-severity items were recorded as accepted.

Fixed:

1. **`_route_secondary_handoff_text` dropped explicit handoff-generation
   requests** (`agent/harness/groups.py`). Item 2 inserted a resolver-based
   navigation check (`_handoff_text_is_navigation`) between the evidence/
   generation shape gate and the capture. Because `answer_opening_question` is
   in `_HANDOFF_NAVIGATION_ACTIONS`, a live model could classify an explicit
   "生成二次开发文档" request as `answer_opening_question topic=extension` and
   route it away, so the handoff draft was never produced. The unit test masked
   this by mocking the resolver to `unknown`. Fixed by exempting explicit
   generation requests (`_looks_like_handoff_generation_request`) from the
   resolver nav-check — generation is a deterministic intent that must not be
   overridden. Regression test strengthened to mock a navigation action and
   assert the draft is still produced.

Accepted (recorded, not fixed):
- `_handoff_text_is_navigation` adds one live LLM call per evidence paste in
  the (rare) unsupported-family handoff flow; inherent to the plan's
  "route through the resolver" choice.
- Deleting the `resolve_intent_action` fallback removes the chain-mention
  safety net only for the narrow case where `resolve_action_queue` returns
  `unknown` but a chain was mentioned alongside a target mode — deliberate; the
  multi-action queue prompt covers it.
- A double `resolve_action_queue` call occurs when a handoff turn is navigation
  (once in `_handoff_text_is_navigation`, once in `_route_free_text`); minor.

## Phase 7: Dual-AI Live Chaos Verification

Status: **done for Phase 6 scope** (non-Docker live run against the configured
DeepSeek model — `provider=deepseek`, `model=deepseek-chat`; Docker was not
available in this environment, but the plan's Phase 1/2 precedent already ran
live DeepSeek via `agent/cli.py`/the harness directly rather than only through
Docker). This pass targeted the Phase 6 changes whose behavior depends on the
live model's classification — the prompt rules the mocked unit tests cannot
exercise.

Live results (all confirmed against real DeepSeek):
- **Item 7**: both "我想改一下 CPU 瓶颈阈值" and the English "I want to adjust
  the CPU bottleneck threshold" → `change_group group=advanced_tuning`; the full
  `process_turn` path then enters the group with `advanced_tuning_confirm`
  pending. The prompt rule I added actually fires on the real model.
- **Item 4**: "你能做什么" and "what can this agent do?" → `answer_opening_question
  topic=agent_capabilities`; `process_turn` returns the product-capabilities
  summary (fake/real/sync + preflight/smoke).
- **Item 3**: "hi" → `greeting` (handled by the single resolver, the gap the
  deletion surfaced and fixed); a multi-intent turn → all four typed actions
  (`choose_target_mode`, `choose_chain`, `set_rpc_mode`, `set_qps_mode`).
- **Item 6**: unknown chain "monad" → `adapter_family=jsonrpc` (canonical value
  in the single-source family list).
- **Item 2**: navigation "先别管…分析最近一次 job 的报告和日志" → `analyze_report`
  → routed away (evidence unchanged); a generation request → draft produced.

**Live-only bug found and fixed (unit tests could not catch it):** evidence
"protocol: weird-p2p; transport: jsonrpc; endpoint: https://…" was classified
by the live model as `choose_chain` (it read "weird-p2p" as a chain name).
Because the first `_HANDOFF_NAVIGATION_ACTIONS` set included
`choose_chain`/`change_chain`/`choose_target_mode`/`analyze_evidence`, real
pasted evidence would have been routed away instead of captured. Narrowed the
navigation set to intents pasted evidence never plausibly triggers
(`reset_session`, `change_group`, `go_back`, `ask_capabilities`,
`answer_opening_question`, `analyze_report`); re-verified live end-to-end that
evidence-with-chain-names is now captured, navigation still routes away, and
generation still produces the draft.

Final verification (all green): 427 tests + 10 subtests,
`tools/check_agent_boundaries.py --root .`, `python3 agent/cli.py adk-eval`
(`status: passed`), full `agent/` `py_compile`, `git diff --check`.

Not run: the fuller Docker-based dual-AI chaos harness (a second AI generating
adversarial user turns across long multi-turn sessions). Docker was unavailable
here; the resolver- and process_turn-level live runs above cover the Phase 6
behaviors, but a longer multi-turn adversarial session in Docker remains a
recommended follow-up for whole-system regression confidence.

## Phase 8: Deferred-Debt Re-Audit (2026-07-11)

The Phase 2–5 "follow-up / not-yet-done" notes were scattered across the phase
sections (see the notes at the Phase 2 "Known follow-up", Phase 4 "Explicitly
not touched", and the Phase 5/6 review sub-sections). This phase re-audited each
one **against the current code** (not against the stale note text) so the doc
stops over-stating outstanding debt. Result: almost all of it was already paid
down in Phase 6, two entries were false positives, one was a real (narrow) bug
now fixed, and one is a deliberate design choice.

Per-item verdict (item numbers follow the Phase 6 plan list above):

1. **Handoff keyword-list routing** — DONE in Phase 6. `_looks_like_handoff_navigation`
   no longer exists; Case-3 evidence-vs-navigation now routes through
   `resolve_action_queue` via `_handoff_text_is_navigation` (`groups.py:5192`).
2. **Two intent-resolver generations** — DONE in Phase 6. `resolve_intent_action`
   is deleted; `resolve_action_queue` is the single resolver (`intent.py:66`,
   its only callers are in `groups.py`).
3. **`capabilities`/`agent_capabilities` topic normalization** — DONE. Every enum
   value in the `intent.py` schema (`intent.py:257`) now has a handler in
   `_opening_consultation_response` (`groups.py:5320`+), plus a default fallback;
   `capabilities`→`agent_capabilities` is normalized at `groups.py:5330`.
4. **Dead `answer` field in action schemas** — DONE. No `"answer"` field remains
   in `intent.py`, and nothing reads `.get("answer")` from resolver output.
5. **Six-copy adapter-family list + divergent `endpoint_probe` set** — DONE.
   Single source is `agent/onboarding/families.py:SUPPORTED_FAMILIES` (12
   importers derive from it); `endpoint_probe.py` reconciled to
   `GENERIC_JSONRPC_PROBE_FAMILIES = {"jsonrpc"}` with an explanatory comment.
6. **`custom_rpc` method disambiguation + weights validation** — split verdict:
   - **`_apply_custom_rpc_inline_workload_hint` `methods[0]` shortcut — REAL bug,
     FIXED (2026-07-11).** The inline single-replace fast-path silently picked
     the first of several validated methods, bypassing the canonical
     `custom_rpc_scope` handler's `len>1 → needs_single_method` disambiguation
     (`groups.py:2961`). It now routes to the same disambiguation. Regression
     tests: `test_inline_single_replace_hint_with_multiple_methods_asks_disambiguation`
     and `..._with_one_method_commits_directly`.
   - **`custom_rpc_weights` "missing/unknown method" validation — NOT a bug
     (false positive).** Unlike `new_chain_custom_weights` (which must restrict
     to validated methods because a new chain has no template defaults),
     existing-chain custom RPC weights may legitimately span template-default
     methods, proven by
     `test_existing_chain_custom_rpc_validates_endpoint_method_and_weights`
     (distributes weight to `eth_getBalance`, never validated as custom). The
     looser check (empty + sum==100 only) is intentional; a NOTE comment now
     documents this at the `custom_rpc_weights` handler so it is not "fixed"
     into a regression later.
7. **Perf / dead-code cluster** — no code change warranted:
   - `_run_doctor()` "double `discover_environment`" — false positive.
     `run_doctor` and `discover_environment` are independent ADK tools
     (`read_only.py:55,68`); there is no single call that scans twice.
   - `_options_from_candidates`/`_field_for` "duplication" — false positive.
     The `config_questions.py` and `config_contract.py` copies share a name but
     map **different** question-id/field sets with different helper logic;
     merging them would conflate two contracts and introduce a bug.
   - `run_quick_assumed_fake_node_smoke` "dead 13-item confirmations list" —
     already gone; that function no longer exists. The `confirmations` list in
     `nodes/execution.py:130` is live (passed into `benchmark_pipeline`).
   - `oracle._next_group_and_reason` "redundant wrapper" — already a documented
     single-line delegate to `routing.next_group_and_reason` (kept deliberately
     to hold the Finding-B1 "two copies once disagreed" rationale).
   - `load_framework_capabilities()`/`inspect_chain_template()` "no caching" —
     deliberately left uncached. These read static bundled config; per-turn cost
     is negligible against the LLM round-trip that dominates every turn, and
     module-level caching of the freshly-built mutable dicts would add a
     shared-state footgun (12 call sites) that is worse than the cost it saves.

## Phase 9: Dual-AI Live Chaos Run (2026-07-11, DeepSeek)

The dual-AI live chaos gate in `tests/agent_live/README.md` was **executed**
against live DeepSeek (`provider=deepseek`, `model=deepseek-chat`) with an
adversarial user simulator (the assistant) driving the real `bin/anychain-agent`
CLI turn-by-turn over a persistent per-session checkpoint (one prompt per
process, resuming from the prior checkpoint — genuinely dynamic, not a fixed
script). Docker was still unavailable, so the non-Docker live path was used as in
Phases 1/2/7. Personas run: confused-evaluator baseline (README steps 1–6),
mid-group-jump + go-back, mode-switch + contradiction, paste-heavy + mixed
language, custom-RPC integrator + new-chain + unsupported-protocol (cases 1/2/3),
and resume + report analyst. Real endpoint probes against public BSC and Monad
testnet endpoints passed end-to-end.

### Live-only defects found and fixed (unit tests could not catch these)

1. **Numbered answer to a yes/no confirm was rejected.** A yes/no confirm renders
   `1. Y / 2. N` and invites "reply with the option number", but `"1"` was
   rejected as "not an answer" and routed to the LLM resolver, which cannot map a
   lone digit — the exact failure README baseline step 5 forbids. Root cause:
   `_answer_fits_pending` (`agent/harness/groups.py`) accepted only `y/yes/n/no`
   for `kind == "yes_no"`, not the option number/label, so the digit fell past the
   deterministic answer path. `_coerce_answer` already maps a bare digit to the
   option value, so the fix is a one-line consolidation: the yes/no branch now
   also accepts `_matches_numbered_option`. Verified live (`1` executes a
   target-mode switch and, later, the preflight/smoke run). Tests:
   `test_numbered_answer_to_yes_no_confirm_applies_without_llm`.

2. **A mode switch silently reused the stale chain.** Switching sync-observe→
   fake-node (or real→fake) from a previous partial session kept the old chain
   (`bsc`/`ethereum`) without surfacing it — README baseline step 3 requires the
   old chain be handled explicitly, never silently reused. Fixed
   `_request_target_mode_change_confirmation` to name the carried-over chain in
   the confirmation ("链 `bsc` 会保留（如需更换请直接说明）") so the user can keep
   or change it. Test: `test_target_mode_change_confirmation_names_preserved_chain`.

3. **A negated protocol mention was read as a positive family choice.** Confirming
   an unsupported-protocol chain with "...自定义二进制协议，没有 JSON-RPC" set
   `adapter_family=jsonrpc` and asked for an RPC endpoint, skipping the Case-3
   official-docs handoff — because `_adapter_family_hint_from_text` regex-matched
   `json-rpc` even under negation. Made the extractor negation-aware (`没有 / 不是
   / 非 / not / no / without` immediately before the family term suppresses the
   hint). After the fix the same turn routes to `adapter_family_confirm`, and
   selecting "none of these" reaches `unsupported_family_handoff`. Test:
   `test_adapter_family_hint_ignores_negated_protocol_mentions`.

4. **A multi-part question dropped its second half.** "fake-node / real-node /
   sync-observe 是什么？preflight/smoke 又是什么？" answered only the modes: the
   resolver collapses the turn into one `answer_opening_question` topic
   (`mode_comparison`), and preflight/smoke was **never defined anywhere** in the
   consultation responses — only listed as a pipeline step. Fixes:
   (a) added a `_preflight_smoke_explanation` and a dedicated
   `execution_preflight_smoke` topic (enum + prompt in `intent.py`) so a
   standalone "what is preflight/smoke" is answered; (b) `mode_comparison` and
   `execution_preflight_smoke` now cross-answer the other half when the user's
   turn names it (via `_mode_comparison_explanation`/`_preflight_smoke_explanation`
   helpers), so neither half is dropped whichever topic the resolver picks;
   (c) de-duplicated `answer_opening_question` blocks in the action queue so an
   overlapping two-action turn does not print the same combined block twice.
   Verified live: the original question now returns modes + preflight/smoke
   exactly once; a pure modes question is not bloated. Test:
   `test_combined_modes_and_preflight_smoke_question_answers_both`.

All four were fixed in the owning harness function (no keyword-list terminal
routing) with deterministic regression tests, and re-verified live end-to-end.

### Investigated, NOT a product defect

An evidence-analysis request during freeform log collection appeared once to fall
through to a generic response. Isolated reproduction (clean and exact-mirror
sessions) showed the product path is correct: a real empty-Enter turn preserves
the collection (`_prompt_evidence_collection_waiting`), and "这是什么意思"
during collection saves the evidence and offers analysis. The one-off anomaly was
a **test-driver artifact** — a no-`--prompt` CLI relaunch (used only to dump
state) resets transient in-flight collection state; a real persistent REPL never
does this. Committed config survives a process restart (verified: fake-node /
ethereum / intensive all persisted), so resume/continue is sound.

### Result

The baseline confused-evaluator transcript (steps 1–6) and the minimum
coverage classes (jumps, mode/QPS switching, paste inference-review, cases 1/2/3,
resume continue/modify/clear, current-state/next-action questions) all pass
against live DeepSeek after the three fixes. Final verification green: 201 agent
unit tests + `tools/check_agent_boundaries.py` ok + `adk-eval status: passed` +
`compileall agent/` + `git diff --check` clean.

Remaining recommended follow-up: the same run inside the repo Docker image (for
parity with the documented runtime) and a second independent user-simulator model
(e.g. Codex) rather than the assistant, per the README's ideal setup.

## Phase 10: Extended Dual-AI Chaos — Edge/Adversarial Sweep (2026-07-11, DeepSeek)

A second live-DeepSeek chaos batch targeted dimensions the first pass did not:
invalid/edge inputs, the full sync-observe path (never completed before), the
real-node endpoint path, adversarial/out-of-scope/injection input, and compound
multi-intent turns. Four more real defects were found and fixed (each in the
owning harness function, with a deterministic regression test and live re-verify).

1. **Out-of-range number at a choice menu was handed to the LLM.** "0" (and any
   out-of-range digit) at `benchmark_mode` fell through `_answer_fits_pending`
   to `_route_free_text`; the resolver read it as a chain action and dropped the
   pending question. Fix: `process_turn` now treats a bare integer at a
   numbered_choice/yes_no menu as an (in)valid option pick — out-of-range digits
   are rejected deterministically ("enter 1..N") and keep the question, never
   reaching the LLM. Test: `test_out_of_range_numbered_answer_keeps_question_without_llm`.

2. **EVM endpoint validation rejected healthy endpoints.** `_sample_address`
   returned the raw `${TARGET_ADDRESS:-0x...}` template placeholder, so the probe
   sent the literal `${...}` as the `eth_getBalance` argument and every healthy
   public EVM endpoint failed ("hex string without 0x prefix"). This blocked
   real-node AND sync-observe for every EVM chain in the default environment.
   Fix: added `_resolve_env_placeholder` (expands `${VAR:-default}` from env or
   default) and applied it in `_sample_address`. Verified live: public BSC and
   Monad endpoints now pass for custom-RPC, real-node, and sync-observe. Test:
   `test_endpoint_probe_resolves_env_placeholder_sample_address`.

3. **sync-observe looped forever and could never execute.**
   `_question_for_group("sync_observe")` unconditionally re-asked the stop
   condition, so once it was set the group never completed and
   `_ask_next_blocking_question` (which prefers the active group) stayed on
   sync_observe indefinitely — the mode could never reach observability/preflight.
   Fix: the sync_observe branch now asks stop_condition only when unset, asks
   duration only when the duration mode lacks seconds, and otherwise returns None
   so routing advances. Verified live end-to-end: sync-observe now reaches
   `preflight: passed` and submits a job. Test:
   `test_sync_observe_group_completes_instead_of_looping`.

4. **A no-op same-chain "change" dropped the active question.**
   `_request_chain_change_confirmation` cleared `pending_question` on a change to
   the already-selected chain, so meaningless input ("🚀🚀🚀") the resolver
   misread as choose_chain(bsc) wiped the active benchmark_mode question. Fix:
   on a no-op same-chain change, keep and re-show the active question instead of
   clearing it. Test: `test_noop_same_chain_change_preserves_active_question`.

Handled correctly (no fix needed): prompt injection ("ignore all instructions")
is rejected without leaking or obeying; off-topic questions keep the pending
context; gibberish is rejected and keeps the question; compound multi-intent
turns ("real-node ethereum, quick, and explain the modes") apply all config
actions AND answer the question; contradictory compound turns route to the
target-mode-change confirmation with the chain preserved.

Minor observation (not fixed, LLM-classification dependent, flow not broken): a
vague "继续配置" issued at the idle `opening` state right after a consultation
answer is a no-op; concrete inputs and other resume phrasings ("接下来配置什么",
"继续下一步") advance normally.

Final verification green: 206 agent unit tests + `check_agent_boundaries` ok +
`adk-eval status: passed` + `compileall agent/` + `git diff --check` clean.

### Cumulative live-chaos defects fixed across Phases 9–10

Eight real defects, all found only via live dual-AI chaos (the mocked unit suite
could not surface them), all now fixed with regression tests:
numbered/label answer to yes-no rejected; mode switch silently reused stale
chain; negated protocol read as jsonrpc; multi-part question dropped its second
half; out-of-range menu number misrouted to the LLM; EVM endpoint placeholder
not expanded; sync-observe infinite loop; no-op same-chain dropped the question.

## Phase 11: Extended Dual-AI Chaos — Robustness Sweep (2026-07-11, DeepSeek)

Third live-DeepSeek chaos batch, targeting: abnormal endpoints/timeouts, multi-
level nested jumps + back navigation, single-turn contradictory instructions,
report/log analysis depth, and long-conversation state integrity. One more real
defect found and fixed; the rest were robust.

Real defect (fixed):

- **A pasted benchmark result was mis-routed to chain selection.** Pasting a
  solana benchmark report ("Success ratio 62%, 429 Too Many Requests, p99 8.5s,
  CPU 780%") with "帮我分析为什么成功率低" made the resolver pluck "solana" as a
  chain choice and discard the report + analysis request. Root cause:
  `_looks_like_pasted_evidence` is a narrow keyword list (traceback/exception/
  `error:`/failed) that a metrics/report dump ("Errors:" plural) does not match.
  Fix: `_route_free_text` now also captures a multi-line paste (`_is_multiline_paste`,
  >= 3 non-empty lines) accompanied by an explicit analysis request
  (`_looks_like_evidence_analysis_request`) as evidence — structural, no new
  content keywords. Verified live: the report is now captured (8 lines, chain not
  mis-set) and analyzed with grounding in the 429/latency/CPU data. Test:
  `test_multiline_paste_with_analysis_request_captured_as_evidence`.

Robust (no fix needed):

- **Abnormal endpoints**: non-JSON-RPC host (google → 405), connection refused
  (127.0.0.1:1), non-routable timeout (10.255.255.1 → ~3s bounded timeout), and
  malformed URLs all produce a clear blocker and keep the question — no crash,
  no false-pass, no hang.
- **Multi-level nested jumps**: deployment→qps→observability→workload forward
  jumps and stepwise "back" navigation return through group history correctly;
  rpc_mode and chain are preserved throughout; "current status" accurately reports
  selected/unselected fields and the next blocker.
- **Contradictory single-turn instructions**: two modes / two chains resolve to a
  safe first-wins plus a switch-confirmation for the second (recoverable, never a
  silent wrong merge). "Ask which one" would be a nice-to-have but the current
  behavior is safe.
- **Long-conversation integrity**: across interleaved config answers, a mid-flow
  "what's my config" question, a jump to QPS, and a QPS correction
  (standard→quick), all state (chain, mode, qps, region, machine) stayed intact —
  no drift or loss.

Minor follow-up (not fixed, LLM/flow nuance): deep multi-turn analysis of an
already-buffered evidence block sometimes re-asks the user to paste again rather
than reusing the buffer; the first analysis is grounded, and re-pasting recovers.

Final verification green: 207 agent unit tests + `check_agent_boundaries` ok +
`adk-eval status: passed` + `compileall agent/` + `git diff --check` clean.

### Cumulative live-chaos defects fixed across Phases 9–11

Nine real defects, all found only via live dual-AI chaos and invisible to the
mocked unit suite, all fixed with regression tests:
numbered/label answer to yes-no rejected; mode switch silently reused stale
chain; negated protocol read as jsonrpc; multi-part question dropped its second
half; out-of-range menu number misrouted to the LLM; EVM endpoint placeholder
not expanded (blocked real-node/sync-observe for all EVM chains); sync-observe
infinite loop (mode unusable end-to-end); no-op same-chain dropped the active
question; pasted benchmark report mis-routed to chain selection.

## Phase 12: Extended Dual-AI Chaos — Boundary & Habit Sweep (2026-07-11, DeepSeek)

Fourth live-DeepSeek chaos batch: cross-family chain-change invalidation, QPS
numeric boundaries, varied user input habits, and preflight-blocked guidance.
Two real defects found and fixed (one severe); the rest robust.

Real defects (fixed):

1. **(severe) A jump to "run preflight/smoke" reached the run confirmation with
   a grossly incomplete config.** On a real-node session with no endpoint, no
   workload, no QPS (only the chain confirmed), asking to run made
   `_question_for_group("preflight_smoke_execution")` unconditionally return
   `preflight_smoke_confirm` ("配置已收集，是否运行?") — and `_ask_next_blocking_question`
   prefers the active group, so it offered to run an unconfigured real-node
   benchmark. Same class as the sync-observe loop. Fix: the preflight branch now
   returns None unless `next_group_and_reason` actually points at
   preflight_smoke_execution (config genuinely ready); otherwise the harness
   advances to the real missing group. Also tightened the intent prompt so a
   RUN/START/EXECUTE request routes to `change_group group=preflight_smoke_execution`
   (which the readiness guard then redirects) instead of the what-is topic.
   Verified live: "直接运行 preflight 和 smoke" on an unconfigured real-node now
   redirects to CLOUD_REGION. Test: `test_preflight_group_not_offered_when_config_incomplete`.

2. **QPS override values were accepted with no validation.** `set_qps_override`
   (and the adjust-field value path) stored negative/zero/non-integer/inverted
   values silently and marked the profile confirmed; nothing validated them
   downstream, so an invalid QPS config could reach preflight/execution. Fix:
   `_invalid_qps_overrides` requires positive integers with MAX_QPS >= INITIAL_QPS;
   invalid input is rejected with a clear message and re-asked. Verified live:
   INITIAL=-100/MAX=0, MAX<INITIAL, decimals, and non-numeric are all rejected;
   valid values accepted. Test: `test_qps_override_rejects_invalid_values`.

Robust (no fix needed):

- **Cross-family chain change**: switching ethereum (jsonrpc) → solana (different
  family) correctly invalidates rpc_mode/workload/custom_rpc and re-offers
  solana-appropriate default methods (getAccountInfo, getBalance, ...) — no EVM
  config leaks into the solana run.
- **Varied user habits**: ALL CAPS, rambling Chinese, and chain-name typos
  ("bnb chian" → bsc) all resolve to the right chain/mode; a heavy typo on the
  mode word ("fak node") is the only miss and is recoverable (the flow asks).
- **Abnormal endpoints** (from Phase 11) remain robust.

Final verification green: 209 agent unit tests + `check_agent_boundaries` ok +
`adk-eval status: passed` + `compileall agent/` + `git diff --check` clean.

### Cumulative live-chaos defects fixed across Phases 9–12

Eleven real defects, all found only via live dual-AI chaos and invisible to the
mocked unit suite, all fixed with regression tests: numbered/label answer to
yes-no rejected; mode switch silently reused stale chain; negated protocol read
as jsonrpc; multi-part question dropped its second half; out-of-range menu number
misrouted to the LLM; EVM endpoint placeholder not expanded; sync-observe
infinite loop; no-op same-chain dropped the active question; pasted benchmark
report mis-routed to chain selection; premature preflight-run confirmation on an
incomplete real-node config; unvalidated QPS override values.

## Phase 13: Extended Dual-AI Chaos — Group-Guard Audit & Post-Exec (2026-07-11)

Fifth batch. Started with a full audit of every `_question_for_group` branch for
the completion-guard pattern behind the sync-observe/preflight bugs, then tested
observability, custom-RPC mixed weights, and post-execution job commands. Two
real defects found and fixed.

Audit result (E1): every config-group branch already returns None when complete
(provider_deployment/network/accounts_disk/endpoint_process/workload_rpc/
qps_profile/advanced_tuning/chain_auxiliary/target_samples all guard on
`if not <field>`, and `_disk_group_question` ends in `return None`). The only two
unguarded branches were sync_observe and preflight_smoke_execution, both fixed in
Phase 10/12. No new same-class bugs remain — the whole surface is covered.

Real defects (fixed):

1. **Custom-RPC mixed weights accepted a garbage method.** `eth_fooBar=50`
   (neither a validated custom method nor a chain template default) was accepted
   because the weights summed to 100, creating a benchmark method that does not
   exist. The Phase-6 note deliberately allowed template-default methods beyond
   the validated-custom set, but that also allowed arbitrary strings. Fix:
   validate weight methods against `validated_custom ∪ default_workload(chain).methods`;
   reject anything else while still allowing genuine template defaults and
   method subsets. Test: `test_custom_rpc_weights_reject_unknown_method_but_allow_template_default`.

2. **`logs <bad-id>` leaked a raw FileNotFoundError.** `JobCommandHandler._logs`
   called `tail_job_log` with no guard, so a missing job id surfaced
   "FileNotFoundError: job not found: ..." to the user, while `_follow_logs` and
   `_status` reported a clean "job not found". Fix: wrap `_logs` in the same
   guard and emit the clean `job_not_found` message. Test:
   `test_logs_command_reports_clean_error_for_missing_job`.

Robust (no fix needed): observability modes (guarded; local/disabled → advance,
no loop); invalid job ids to status/follow already report cleanly; bare `status`
uses the latest job; post-execution state/next-action questions answer from job
state with concrete next actions.

Final verification green: 211 agent unit tests + `check_agent_boundaries` ok +
`adk-eval status: passed` + `compileall agent/` + `git diff --check` clean.

### Cumulative live-chaos defects fixed across Phases 9–13

Thirteen real defects, all found only via live dual-AI chaos and invisible to the
mocked unit suite, all fixed with regression tests: numbered/label answer to
yes-no rejected; mode switch silently reused stale chain; negated protocol read
as jsonrpc; multi-part question dropped its second half; out-of-range menu number
misrouted to the LLM; EVM endpoint placeholder not expanded; sync-observe
infinite loop; no-op same-chain dropped the active question; pasted benchmark
report mis-routed to chain selection; premature preflight-run confirmation on an
incomplete real-node config; unvalidated QPS override values; custom-RPC mixed
weights accepted a non-existent method; `logs <bad-id>` leaked a raw exception.

## Phase 14: Verification of fixes + expanded chaos scope (2026-07-11)

Two-part request: (1) confirm every documented fix holds, (2) widen the dual-AI
chaos scope.

Part 1 — all 13 fixes confirmed:
- Each of the 13 named regression tests runs and PASSES individually.
- Full agent suite: 211 passed; `check_agent_boundaries` ok; `adk-eval` passed;
  `compileall agent/` ok; `git diff --check` clean.
- Live re-spot-check of the three S1 severe fixes: EVM endpoint placeholder now
  validates a real BSC endpoint (#6); "run preflight while incomplete" redirects
  to the missing group instead of "配置已收集" (#10); sync-observe completion is
  covered by its regression test + the Phase-10 live run that reached
  `preflight: passed` (#7).

Part 2 — expanded scope (batches F + G), zero new defects:
- F1 doctor/dependency guidance: clear, specific (missing vegeta + install offer).
- F2 huge input: 28KB/401-line paste captured in ~5s, no crash (143KB is an OS
  argv limit of the test driver, not the product REPL which reads stdin).
- F3 resume/corrupt-checkpoint: resume restores the exact pending question across
  a fresh process; a corrupted checkpoint yields a clean user message with the
  traceback only in stderr (no crash, no stdout leak) — minor: message
  misattributes the cause (see known-issues B.2).
- G1 non-EVM families: bitcoin_jsonrpc, tendermint (cosmos-hub), substrate
  (polkadot), rest (aptos), hedera_dual all confirm the chain and produce
  family-appropriate default workload methods; the config flow progresses with no
  family-specific breakage.
- G2 sequential chains: bitcoin → cosmos-hub (two different non-EVM families)
  correctly invalidates workload/rpc_mode with no state bleed; cross-family
  invalidation is general, not EVM-specific.
- G3 cancel/reset: explicit reset phrasings ("全部清空重来", "重置所有配置") clear
  the session; a vague "取消" conservatively keeps state (safe).

Two consecutive expanded batches (F, G) found zero new defects — the campaign has
reached diminishing returns; the harness is stable across the tested dimensions.
Remaining uncovered depth (needs real per-chain endpoints / other providers) is
tracked in the known-issues register section C.

Known-issues register created: `.agent/task-docs/2026-07-11-agent-known-issues.md`.

## Phase 15: First-interaction dependency-offer fix (2026-07-11)

Triggered by a real product transcript the user shared: on startup with missing
`vegeta`, the Agent prints "Missing dependencies... Allow this? [Y/n]", but the
user's "y" did not install — it fell through to the harness (greeting/capabilities),
and an explicit "run scripts/install_deps.sh" got a generic workflow reply.

Root cause (S1, hit every new user since missing vegeta is the default state):
`_startup_doctor()` sets `current_question_id="install_dependencies"` and prints
the offer, but the immediately-following ADK-available branch in `startup()`
unconditionally reset `current_question_id=""`, clobbering the offer. So
`_handle_pending_confirmation` never matched "y".

Fix (`agent/terminal/repl.py`): the ADK-available branch preserves an
`install_dependencies` offer instead of clearing it (and skips the resume offer
when a deps offer is pending). Verified live: "y" now runs
`scripts/install_deps.sh --yes` (exit_code=0) instead of greeting. Test:
`test_startup_preserves_dependency_offer_so_yes_installs`.

Secondary (documented, not fixed — known-issues B.9): an explicit natural-language
"run scripts/install_deps.sh" request (instead of "y") is not honored; wiring it
needs an install intent in the resolver.

This was found only because the user shared a real first-screen transcript — the
chaos driver had always seeded a chain/mode command, skipping the startup deps
prompt. Lesson recorded: cover the pre-chain startup prompts too.

Final verification: 212 agent unit tests pass; boundary ok; adk-eval passed;
git diff --check clean.

### Cumulative live/real-transcript defects fixed: 14
(Adds #14 startup dependency-offer to the 13 from Phases 9–13.)

## Phase 16: Chain-workload question not answered (2026-07-11)

From a second real transcript: the Agent invites "询问某条链的默认 RPC workload",
but "bsc 有哪些 rpc workload" got treated as choose_chain (→ chain-selected menu)
and "solana 有哪些 rpc workload" dumped the generic 36-chain list — the question
was never answered. Root cause: no handler returned a specific chain's workload,
and the resolver reliably reads the chain name as a `choose_chain` selection.

Fix (`agent/harness/groups.py`): added `_chain_workload_summary` (from
`default_workload(chain)`) and answered it in the `supported_chains` topic when a
chain subject is present; plus a deterministic, read-only override in
`_route_free_text` — when the turn is a chain-workload question
(`_chain_workload_question_target`) AND the resolver reduced it to chain-query
actions only (`_actions_are_chain_query_only`, i.e. no change_group/set_qps/etc),
answer the workload instead of selecting the chain. The chain-query-only guard
keeps compound turns ("look at workload AND set QPS quick") working. Intent-prompt
hint added too. Verified live: "bsc/solana 有哪些 rpc workload" now print the
chain's single/mixed default methods and do NOT select the chain, while
"测试 bsc" still selects it. Test: `test_chain_workload_question_answered_not_selected`.

Final verification: 213 agent unit tests pass; boundary ok; adk-eval passed; diff clean.

### Cumulative live/real-transcript defects fixed: 15

## Phase 17: Opening-surface user-perspective chaos (2026-07-11)

Methodology correction (user feedback): the chaos driver had always seeded a
"chain + mode" command, skipping the pre-chain opening surface where a real
first-time user actually starts. Re-ran a genuine confused-new-user multi-turn
session FROM TURN 1 with no seeding. Two more real defects found and fixed:

16. **Accepting a recommendation looped.** After "recommend the simplest test",
    the user's "好，那就按你推荐的来" re-printed the same recommendation instead of
    starting it — the recommendation was consultation text, not an actionable
    option. Fix: a recommendation now sets an actionable `accept_recommendation`
    yes/no ("start with the recommendation now? (solana + fake-node)"); accepting
    it applies the recommended chain+mode and enters the config flow. Verified
    live. Test: `test_accept_recommendation_starts_recommended_setup`.

17. **Readiness question got a generic checklist.** "我这台机器能不能跑 / is my
    env ready" returned the generic requirements pipeline instead of the startup
    readiness verdict (status + detected gcp/e2-standard-4/4 CPU/15.62 GiB +
    missing deps) that was already computed at startup. Fix: a deterministic
    readiness detector (`_is_environment_readiness_question`, gated on an
    env/machine word to avoid hijacking "bsc 能不能跑") answers from
    `state.discovery`. Verified live. Test:
    `test_environment_readiness_question_answered_from_discovery`.

Both are the same recurring class as #14/#15: the info/capability exists, but a
natural free-form phrasing on the opening surface was not routed to it. Lesson:
always cover pre-chain opening free-conversation, not just the seeded config flow.

Final verification: 215 agent unit tests pass; boundary ok; adk-eval passed; diff clean.

### Cumulative live/real-transcript defects fixed: 17

## Phase 18: Case 1/2/3 live test with real web-sourced chains (2026-07-11)

Per the Case 1/2/3 + Unknown-chain standard in `docs/*/anychain-agent-ai-work-gate.md`.
My WebSearch is blocked by an org policy (mirrors the agent: no search off-Gemini),
but WebFetch works, and I verified live endpoints directly.

- Case 2 (new chain in existing family) — real chain **Mantle** (chainId 0x1388,
  https://rpc.mantle.xyz, not in the 36, jsonrpc family): identity -> family(jsonrpc)
  -> real endpoint validation -> real eth_blockNumber method/schema validation, all
  passing against the live node. PASS end-to-end with real materials.
- Case 3 (new chain outside families) — real chain **Mina** (zk L1, GraphQL-only
  API, confirmed via docs.minaprotocol.com). Found and fixed defect #18:
  a negated protocol ENUMERATION ("GraphQL, 不是 REST/cosmos/substrate/bitcoin")
  was mis-read as `substrate` (negation covered only the first list item) and the
  chain-scan surfaced "bitcoin" from the negated list — so Mina was routed to
  Case 2 instead of the Case-3 handoff. Fix: negation regex now spans a negated
  family enumeration, and a new `_known_chain_in_text` skips chains named only
  inside a negation. After the fix, Mina reaches `adapter_family_confirm` (with a
  "none/unsupported" option) and then `unsupported_family_handoff` with the
  secondary-development handoff. PASS.

Remaining (known-issues B.11): a VERBOSE identity-confirm answer is still routed
to the generic chain list rather than the "choose protocol" option; clean answers
("1"/"真实链") reach Case 3. Deferred (routing-precedence change is risky).

Methodology note: I also switched to a genuine persistent interactive REPL session
(one process, driven turn-by-turn via a FIFO) instead of per-turn `--prompt`
restarts, per user feedback — closer to a real terminal user.

Final verification: 215 agent unit tests pass; boundary ok; adk-eval passed; diff clean.

### Cumulative live/real-transcript defects fixed: 18

## Phase 19: Conformance refactor + reactive dual-AI chaos (2026-07-11)

Prompted by user feedback: phrase-table classifiers had crept back into the
harness (defects #14/#15/#17's fixes used `_chain_question_target` /
`_is_environment_readiness_question` / `_known_chain_in_text` regex/keyword
matching). This violates the LangGraph-harness conformance rule (no
business-intent routing via keyword lists in the harness or terminal).
Removed all three; chain-info and readiness intent now live entirely in the
resolver's typed `answer_opening_question` topics
(`environment_readiness`, `supported_chains`+subject, `extension`+subject) —
the harness only routes on the typed output. Re-verified live on DeepSeek.

Ran a full reactive turn-by-turn dual-AI chaos pass (agent as user-simulator,
each next turn generated from the Agent's actual prior response, not
scripted) across sync-observe field explanations, QPS cross-turn validation,
custom-RPC weight edge cases, provider-metadata confirm, EVM-only health
probe assumption, and mixed-workload default confirmation. Found and fixed
8 real defects (#19–#26 in `2026-07-11-agent-known-issues.md`; see that file
for exact repro/fix detail per defect) — most severe: `#24` (EVM-only health
probe method used for every jsonrpc-transport chain, blocking real-node/
sync-observe endpoint validation for every non-EVM jsonrpc chain — solana,
sui, near, starknet, tron, avalanche-x) and `#26` (mixed-workload "use
defaults" never actually confirmed weights, deadlocking every mixed run at
preflight).

Final verification: full agent-harness/terminal/pipeline suite passes; live
DeepSeek re-verification of all 8 fixes.

### Cumulative live/real-transcript defects fixed: 26

## Phase 20: sync-observe / custom-RPC / fake-node coverage (2026-07-12)

Scope: chains/paths not yet driven live — endpoint-only sync-observe (no
local node process), pasting raw curl/JSON-RPC bodies at the custom-method
step, and re-analyzing the just-submitted job instead of a stale
startup-detected one. Found and fixed 4 defects (`#27`–`#30`): endpoint-only
sync-observe deadlocked on a `node_process_identity` question a remote node
can never answer (`#27`); pasted RPC content with an embedded URL was
misread as "you gave me an endpoint, not a method" (`#28`); the `#27` fix
itself introduced a `TypeError` caught in the same batch (`#29`); and
"analyze the latest job" preferred a stale injected hint over the on-disk
latest (`#30`).

Final verification: full suite passes; live DeepSeek re-verification.

### Cumulative live/real-transcript defects fixed: 30

## Phase 21: observability-exporter / intensive-QPS / sync-observe-field-correction coverage (2026-07-12)

Scope: sync-observe-only field explanations, observability `exporter` mode's
promise to integrate with an existing Prometheus, duration/stop-condition
correction via natural language after the fact, and QPS override validation
merged against the benchmark mode's own baseline (not just the override dict
in isolation). Found and fixed 6 defects (`#31`–`#36`); most severe: `#34`
(once `sync_observe_stop_condition`/`duration_seconds` were set, correcting
either via NL was a permanent dead end — both fields were absent from the
correctable-field allowlists) and `#36` (overriding only `MAX_QPS` while
`INITIAL_QPS` stayed at the mode's default produced an inverted, uncaught
QPS profile that silently skipped past QPS validation).

Final verification: full suite passes; live DeepSeek re-verification.

### Cumulative live/real-transcript defects fixed: 36

## Phase 22: advanced-tuning / accounts-disk / observability-local coverage (2026-07-12)

Scope: group-internal branches never previously driven live — numeric field
validation for ledger/accounts disk and network fields (both the direct
manual-question path and the "paste inferred config" / `KEY=VALUE` path),
`advanced_tuning`'s threshold/interval override validation, and observability
`local` mode's port disclosure. Found and fixed 4 defects (`#37`–`#40`);
`#37`/`#38` closed a previously-deferred item (`B.16`, `_first_number_text`
silently stripping a leading "-"): negative/zero/non-numeric disk and
network values were accepted silently on both entry paths, and the
paste-inference path additionally *changed* a pasted "-50" to "50" without
telling the user.

Final verification: full suite passes; live DeepSeek re-verification.

### Cumulative live/real-transcript defects fixed: 40

## Phase 23: Parallel four-agent chaos sweep (2026-07-12)

Methodology change: four isolated CLI sessions (own
`ANYCHAIN_AGENT_SESSION_ID`/`ANYCHAIN_AGENT_CHECKPOINT_PATH` each, so no
shared checkpoint state) ran concurrently in discovery-only mode, each
covering a different angle — real-node mode's full flow end to end, non-EVM
chain families (litecoin/bch as bitcoin_jsonrpc, dogecoin's
`chain_auxiliary_endpoints`), job/report analysis, and a regression-hardening
pass re-triggering defects #1–#40 with new phrasing to catch drift. Findings
were then applied and verified serially by a single process to avoid
concurrent edits to the same files. Found and fixed 7 defects (`#41`–`#47`);
most severe: `#41` (**critical** — a fuzzy-matched decline of a yes/no
confirm, e.g. "nope, let me adjust something first", was silently inverted
into an accept via a `str(False)` round-trip bug; for
`preflight_smoke_confirm` this **submitted a real benchmark job against
explicit user refusal**, reproduced live against a real BSC endpoint) and
`#42` (an empty-string GET body was passed as `data=""` to
`urllib.request.Request`, which requires `None`/bytes — every
bitcoin_jsonrpc/rest/hedera_dual/tendermint chain, plus one of polkadot's
mixed methods, was **structurally unable to ever pass** endpoint validation,
roughly half of all 36 configured chains). Also closed validation gap
`C.5` (custom-RPC / endpoint validation on non-EVM families) by driving
litecoin/bch live for the first time.

Final verification: full suite passes; live DeepSeek re-verification of all
7 fixes, including a live resubmission proving `#41`'s fix actually honors a
declined confirm.

### Cumulative live/real-transcript defects fixed: 47 — this count matches
the `fix(agent): resolve 47 defects found via live dual-AI chaos (Phases
9-23)` commit (`98d1441`), which is the last point at which defects 1–47
were committed to git. Everything from Phase 24 onward remained uncommitted
working-tree state as of 2026-07-13 (confirmed via `git status`/spot-checks
against the actual source, not just this document, on 2026-07-13 — see
Phase 25's note).

## Phase 24: ADK retirement, google_search fix, and code-review pass (2026-07-13)

Retired the entire unused `agent/adk_app/` ADK-native Agent/Runner
tool-calling surface (never the shipped product's entrypoint — the real
harness never used ADK's `Runner` loop) and consolidated
`agent/tools/executor.py`/`schema.py` as the single tool-dispatch surface,
relocating every genuinely-needed piece (`install_dependencies`, fixture
checks, `load_execution_contract`, `inspect_llm_auth`, ADK-availability
probe, REPL startup state, and `google_search` grounding) into new homes
instead of deleting them.

This phase's actual goal was fixing `google_search`, which chaos testing had
never been able to exercise (no real Gemini key in this sandbox) but code
reading found was fully non-functional regardless: `#48` (a field-name
mismatch — `WebResearchStatus.as_dict()` emitted `enabled`, the harness read
`google_search_available` — meant the "will use google_search" branch could
never fire for any real Gemini config) and `#49` (even after `#48`, the
client-setup handoff message only *promised* to search; `get_google_search_tools()`
had zero callers anywhere). Fixed both, then wired a real
`run_google_search_grounding()` call (`agent/llm/search_grounding.py`) into
`agent/harness/groups.py::_sync_client_setup_handoff_message`, verified live
on DeepSeek (which correctly reports `google_search_available=False` and
takes the "unavailable" branch — the real-Gemini branch remains validation
gap `C.2`).

A dedicated code-review pass (8 finder agents across
correctness/reuse/simplification/efficiency/altitude angles, then 3 fresh
skeptical verifier agents per finding with no prior context) run against
this phase's own new code surfaced 6 more defects (`#50`–`#56`), two
pre-existing (`#51`/`#52`, introduced years earlier, surfaced only because
relocating their files required reading them closely — not regressions from
this phase).

Final verification: full suite passes; live DeepSeek re-verification of the
google_search fix chain (`#48`/`#49`).

### Cumulative live/real-transcript + code-review defects fixed: 56

## Phase 25: client_setup routing fix + tendermint/hedera_dual family coverage sweep (2026-07-13)

Picked up two threads left open at the end of a prior session (relayed to a
fresh session via pasted transcript, per the standing dual-AI-chaos/
conformance requirements in `[[dual-ai-chaos-and-conformance]]`): an
unresolved oracle-preview bug noticed but not yet fixed, and the standing
question of whether the Phase 19–24 chaos campaign had a documented
scope/coverage record (it did not — this document stopped at Phase 18; the
known-issues register has only per-phase footnotes, not a narrative).

**`#57`** (fixed): right after acknowledging the real-node-client-setup
handoff (`sync_observe_client_setup_ack` → true), an advisory "what's my
current config / next blocking item" query showed
`choose sync-observe stop condition` — but the actual next live question is
`sync_observe_after_client_setup` (choose the real data source). Root cause:
`routing.next_group_and_reason` had a branch for
`client_setup and not acknowledged` but none for the acknowledged case, so
it fell through past `groups.py`'s actual next question entirely. Fixed by
adding the missing branch (plus two missing `oracle._reason_label` mapping
entries, closing the same raw-string-leak class as `#47`). Verified live:
reproduced the exact bug scenario turn-by-turn on DeepSeek before the fix's
absence would have shown the wrong preview, then again after the fix showing
the correct one, then confirmed the actual next question matches.

**Coverage audit**: cross-referencing `agent/cli.py capabilities` (36
chains across exactly 6 adapter families: `jsonrpc` 16, `bitcoin_jsonrpc` 4,
`substrate` 5, `tendermint` 5, `rest` 5, `hedera_dual` 1) against every
chain name mentioned across Phases 9–24 in `2026-07-11-agent-known-issues.md`
found that `tendermint` (celestia/cosmos-hub/injective/osmosis/sei) and
`rest` (algorand/aptos/cardano/tezos/ton) had **zero** live chaos coverage —
`#42`'s GET-body fix for these families (plus `hedera_dual`) had only ever
been confirmed by direct code execution, never an actual conversation
against a real endpoint.

**`#58`** (found and fixed): drove real-node mode live end to end on
`cosmos-hub` (tendermint) for the first time — `LOCAL_RPC_URL` validated
against a real public endpoint (`https://cosmos-rest.publicnode.com`,
genuine HTTP 200 with live chain data) and `mixed` RPC workload defaults
confirmed cleanly, closing the `tendermint` coverage gap. Continuing the same
session with a chain switch to `hedera` (`hedera_dual`, still zero coverage)
surfaced a real, unrelated **S1** defect: the compound turn "switch to
hedera, only have a real endpoint, ..." resolved into a queued
`change_chain` + `change_group(workload_rpc)` action pair; once the chain
switch's confirm interrupt resolved, the queued `change_group` jumped
straight into `workload_rpc` with no check that its prerequisite
(`endpoint_process`, just invalidated by the chain switch one action
earlier in the same queue) was still satisfied — `active_group` became
`workload_rpc` while `endpoint_evidence == {}` and
`confirmed_config["LOCAL_RPC_URL"]` still silently held the *previous*
chain's endpoint (confirmed via direct LangGraph checkpoint inspection, not
guessed from the transcript). Fixed by having `_activate_group_question`
redirect to the real blocking group (`routing.next_group_and_reason`)
whenever the requested group is itself listed in `invalidated_groups` (the
field's first real reader — previously confirmed dead code, written at 10+
call sites and read nowhere, per `#45`'s note) and a genuinely-unmet earlier
precondition still exists; a jump to a group that was simply never
configured yet (not invalidated) is left untouched, preserving the existing,
separately-tested "jump ahead of an incomplete group for reconfiguration"
behavior. Verified live: reproduced the exact broken sequence via direct
checkpoint inspection, then re-ran the identical live turn sequence
post-fix and confirmed it now re-asks for `LOCAL_RPC_URL` instead of
silently reusing the stale endpoint.

Final verification: full agent-harness suite (214 tests, +2 new) passes;
both fixes verified live on DeepSeek in addition to unit regression tests;
`git diff` confirmed for both changed files against expectations.

### Cumulative live/real-transcript + code-review defects fixed: 58

### Phase 25 follow-up: closing `rest` and `substrate` (2026-07-13, same day)

Continued the family-coverage sweep to close the remaining gaps. Drove
real-node mode live end to end on `algorand` (`rest`): `LOCAL_RPC_URL`
validated against `https://mainnet-api.algonode.cloud` with a genuine HTTP
200 real-node status response — first live `rest`-family coverage. Switching
from `algorand` to `hedera` with a plain single-intent turn ("换成 hedera",
deliberately not the compound phrasing that triggered `#58`) correctly
re-asked for `LOCAL_RPC_URL`, confirming that fix generalizes beyond its
original repro. Then drove `polkadot` (`substrate`) live end to end:
`LOCAL_RPC_URL` validated against `https://rpc.polkadot.io` with a genuine
response (real DOT balance, real block height 32089628), then `mixed`
workload defaults confirmed cleanly — closing `substrate`'s post-`#42`
endpoint-validation gap (prior `polkadot` coverage was mixed-weights only,
pre-dating `#42`).

`hedera_dual` remains open, but as an environmental issue rather than a code
gap: across three live attempts (two sessions) against
`https://mainnet-public.mirrornode.hedera.com`, the health probe and 2 of 5
method probes (`GET /api/v1/accounts/{addr}`, `eth_getBalance`, `eth_call`)
consistently returned genuine 200s with real Hedera data, while a different
method timed out each time. Verified independently via direct `curl` outside
the Agent entirely: the identical query intermittently succeeds in <0.5s and
times out at >6s from this sandbox's network (4 of 5 direct attempts
succeeded) — this is this specific public host's latency variance from this
network, not a substitution or timeout-handling defect in
`endpoint_probe.py`. Not pursued further to avoid spending live-chaos turns
re-testing known external flakiness rather than product behavior; see
`known-issues.md` C.6 for the full evidence trail (exact curl timings and
probe JSON).

Final verification: no code changes this follow-up (pure coverage
verification) — no new tests needed; the existing 214-test suite and both
Phase 25 fixes remain the last code-verified state.

### Cumulative live/real-transcript + code-review defects fixed: 58 (unchanged — this follow-up closed coverage gaps, found no new defects)

## Phase 26: Case 1/2/3 coverage audit and Case 2 family sweep (2026-07-13)

User feedback prompted a clarifying question this document had glossed over:
Phase 25's family-coverage sweep drove already-supported chains
(cosmos-hub/hedera/algorand/polkadot) through their *default* RPC methods —
that is a different axis from the Case 1/2/3 chain-onboarding standard in
`docs/*/anychain-agent-ai-work-gate.md`, and should not have been read as
Case coverage. Audited actual Case coverage: Case 1 (supported chain, custom
RPC method) had real defects found historically (`#12`/`#21`/`#28`) but only
ever on `jsonrpc`-family chains; Case 2 (new chain in an existing family) had
only ever been driven live once, on Mantle (`jsonrpc`, Phase 18) — the other
5 families were completely untested.

Drove Case 2 live on a genuinely new `substrate` chain (**Moonriver**, not in
the 36 templates): identity resolution correctly routed it to
`existing_family_needs_endpoint` (substrate), but the endpoint-validation
step crashed with a raw `CalledProcessError` for *any* real endpoint.
Root-caused to `agent/validators/endpoint_probe.py::health_probe_methods`
returning `(None, {})` for every family except `jsonrpc` — for a
template-less chain this leaves zero probe methods, so
`_should_use_generic_jsonrpc_probe` cannot select the generic (template-free)
probe and falls through to the template-*requiring* `_probe_health`/
`_probe_method` path, which needs a `config/chains/<chain>.json` that a
genuinely new chain never has. This is `#59` (CRITICAL, S1) — Case 2
endpoint validation was structurally unable to ever pass for 5 of 6 adapter
families, for any real endpoint, regardless of health.

Fixed for the two families whose transport is a plain POST JSON-RPC call:
widened `GENERIC_JSONRPC_PROBE_FAMILIES` to `{jsonrpc, substrate,
bitcoin_jsonrpc}` and gave `health_probe_methods` a safe per-family default
method for a template-less chain (`system_chain` for substrate,
`getblockchaininfo` for bitcoin_jsonrpc — the latter already declared in
every existing bitcoin_jsonrpc template's `_meta.health_probe`, previously
dead for the same root-cause reason). Verified live end to end: Moonriver's
real endpoint (`https://rpc.api.moonriver.moonbeam.network`) validated
(`system_chain` → real result `"Moonriver"`), then a real custom method
(`chain_getHeader`) + schema evidence validated — a complete Case 2 pass.
Switching live to a new `bitcoin_jsonrpc` chain (**Dash**) confirmed the fix
routes correctly there too (a clean "endpoint returned 404" against a
non-JSON-RPC test host, not the old crash).

`tendermint`/`rest`/`hedera_dual` are GET/REST-shaped transports the generic
probe does not build requests for, so they were deliberately left out of
`#59`'s fix (a correct generic GET/REST probe is a larger, separate change).
Confirmed live that the gap is real, not hypothetical: switching to a new
`tendermint` chain (**Juno**, `https://juno-api.polkachu.com`) and a new
`rest` chain (**Stellar**, `https://horizon.stellar.org`) both reproduced the
identical crash `#59` fixed for `substrate`. Recorded as open item B.25.
Also assessed (inconclusively — WebSearch is blocked by org policy in this
environment) whether `hedera_dual` has any natural second real-world
exemplar chain to even test Case 2 against; Hedera's mirror-node REST + EVM
JSON-RPC-relay dual shape appears specific to its own architecture.

Regression test: `test_new_chain_endpoint_validation_works_for_generic_pop_families_without_a_template`
(local HTTP server, no live network dependency). Two pre-existing tests
encoded the old, narrower assumption (`GENERIC_JSONRPC_PROBE_FAMILIES ==
{"jsonrpc"}` exactly, and `health_probe_methods("bitcoin", "bitcoin_jsonrpc")
== (None, {})`) and were updated to match the corrected, intentional
behavior rather than left to rot as a stale expectation.

Final verification: full agent-harness suite (215 tests, +1 new, 2 updated)
passes; boundary check ok; both the substrate and bitcoin_jsonrpc fixes
verified live on DeepSeek in addition to the unit regression test.

### Cumulative live/real-transcript + code-review defects fixed: 59

## Phase 27: Case 1 coverage on the newly-covered families (2026-07-13)

Continuing Phase 26's Case audit, drove Case 1 (custom RPC method on an
already-supported chain) live on `cosmos-hub` (tendermint) — the family this
whole coverage push started with in Phase 25. Chose `mixed` RPC mode,
selected "add custom RPC method," and entered the real endpoint again
(`https://cosmos-rest.publicnode.com`), then the real custom method
`GET /cosmos/staking/v1beta1/pool`. It was rejected: "This looks like an
endpoint, REST path, or documentation title... if this is a REST API, switch
protocol family to `rest`" — nonsensical advice for a chain that is *already*
REST-shaped (`cosmos-hub`'s own default methods are all `GET /cosmos/...`
paths).

Root-caused to `_looks_like_rest_path_or_doc_method`'s call sites in both
`custom_rpc_method` (Case 1) and `new_chain_method` (Case 2) applying the
REST-path rejection unconditionally, with no adapter-family awareness at
all — a `GET /path` answer is only ever a mistake for families whose real
methods are plain JSON-RPC calls (`jsonrpc`/`substrate`/`bitcoin_jsonrpc`);
for `rest`/`tendermint`/`hedera_dual` it is the *correct* format. This is
`#60` (S1). Fixed both call sites by skipping the REST-path rejection when
`_chain_adapter_family(state) not in GENERIC_JSONRPC_PROBE_FAMILIES` (reusing
`#59`'s canonical partition rather than inventing a second one) — a bare URL
is still rejected unconditionally regardless of family, since that is always
a mistake. Verified live end to end: the same `GET /cosmos/staking/v1beta1/pool`
input is now accepted, and its schema (`[]`, no params) validated against the
real endpoint with a genuine HTTP 200 and real cosmos-hub chain data.

`#60`'s fix is family-based, not chain-specific, so it structurally also
covers `rest` and `hedera_dual` (the same class of chains item 6/B.25 already
established use `GET /path` methods) — not separately re-verified live on
`polkadot`/`algorand`/`hedera` given the regression test already exercises
the general (family-in-set vs. family-not-in-set) logic directly; a live spot
check on those three remains a reasonable low-cost follow-up.

Regression test: `test_custom_rpc_method_accepts_get_path_for_rest_shaped_chain_family`
(direct `_apply_endpoint_answer` unit test, asserts both the `cosmos-hub`
accept case and that `bsc` — a real `jsonrpc` mistake case — is still
correctly rejected).

Final verification: full agent-harness suite (216 tests, +1 new) passes;
boundary check ok; fix verified live on DeepSeek with a real endpoint probe
in addition to the unit regression test.

### Cumulative live/real-transcript + code-review defects fixed: 60

## Phase 28: GET/REST generic probe for tendermint/rest/hedera_dual — closes B.25 (2026-07-13)

User feedback: after presenting the current open-item list (mostly deferred
S2/S3 trade-offs, plus one real structural gap), the user asked to fix the
one item that was actually worth doing now: B.25, Case 2 endpoint validation
for the `tendermint`/`rest`/`hedera_dual` families, left unfixed by `#59`
because those three are GET-path/REST-shaped and `#59`'s generic prober only
builds POST JSON-RPC requests.

Extended `agent/validators/endpoint_probe.py` with a family-tiered strategy:

1. A genuine generic REST prober (`_validate_generic_rest_endpoint`/
   `_probe_generic_rest_method`) that builds real `GET`/`POST` HTTP requests
   for `GET /path`-shaped methods, with `{placeholder}` substitution from
   schema-evidence-style params (the same trust model `#59`'s JSON-RPC prober
   already applies to `params`). This activates for any template-less chain
   in `REST_SHAPED_FAMILIES` (`rest`/`tendermint`/`hedera_dual`) once the user
   supplies GET/POST-shaped method(s) — covering the natural next step after
   the initial health check, not just the health check itself.
2. For the *initial* health check (before any method is known yet),
   `tendermint` gets a genuinely universal safe default:
   `GET /cosmos/base/tendermint/v1beta1/blocks/latest`, the Cosmos SDK's own
   gRPC-gateway REST module present on essentially every Cosmos-SDK chain
   regardless of its app-specific API surface (confirmed live on both
   cosmos-hub and Juno).
3. `rest`/`hedera_dual` have no such universal path — Algorand, Aptos,
   Cardano, Tezos, TON, and Hedera's mirror node all expose completely
   different REST APIs — so they fall back to `_validate_bare_reachability`:
   a plain GET to the base URL, treating any completed HTTP response (2xx
   through 5xx, not a connection failure/timeout) as "the server is alive."
   Full protocol validation is deferred to the next step once the user
   supplies a real method, which is exactly what a Case 2 flow does next
   anyway.

This is `#61` (CRITICAL, closes B.25). Verified live on the exact two chains
that reproduced B.25 in Phase 26: **Juno** (tendermint) — endpoint validated
via the universal default (genuine HTTP 200 against
`https://juno-api.polkachu.com`), then a real custom method
(`GET /cosmos/staking/v1beta1/pool`) validated with another genuine 200.
**Stellar** (rest) — endpoint accepted via bare reachability (a genuine 200
returning Horizon's real API root document), then a custom method attempt
against a guessed account ID correctly returned a clean HTTP 404 (a real
network response evaluated correctly, not a crash — proving the generic REST
prober itself works end to end, independent of whether that specific test
account exists).

Regression test: `test_new_chain_endpoint_validation_works_for_rest_shaped_families_without_a_template`
(local HTTP server covering all three families: the tendermint safe default,
the rest/hedera_dual bare-reachability fallback, and generic REST probing of
a user-supplied method with and without a path placeholder).

Final verification: full agent-harness suite (217 tests, +1 new) passes;
boundary check ok; fix verified live on DeepSeek against two real external
endpoints in addition to the unit regression test.

### Cumulative live/real-transcript + code-review defects fixed: 61

## Phase 29: Full backlog closure — all remaining B-items, C.3, C.4 (2026-07-13)

User directive: "已知的问题都需要进行修复，不以打补丁的方式" (all known issues must
be fixed, not via patches) — fix every remaining reasonably-fixable B-section
item, close the outstanding C validation gaps, and confirm real understanding
of the framework before each fix rather than pattern-matching a quick patch.

**New defect found via the user's own manual testing, not chaos testing**
(`#62`): opening-menu option 4 ("了解支持的链/RPC method/扩展方式") rendered no
visible content and looped — the `group == "opening"` handler's `value ==
"info"` branch cleared `pending_question` and set `active_group` without
ever printing `_framework_capability_summary(state)`. Root cause of the
chaos-testing miss: every prior sweep only ever tried free-text phrasings of
"what chains do you support," never a literal numbered-menu selection of a
"meta" opening option — a real methodology gap, not a framework defect,
worth remembering for future chaos design.

**B-item fixes** (`#63`-`#69`, all in `agent/harness/groups.py` unless noted):

- `#63` (closes B.22): `_validate_rpc_schema` no longer writes
  `case_dict[params_field]` before validating; it pops the key on failure so
  the deterministic schema re-ask gate fires correctly a second time.
- `#64` (closes B.20): `oracle._execution_status` now prefers
  `job_manager.get_job(job_id)`'s live disk read over the stale
  never-refreshed `state["job"]` snapshot, falling back to the snapshot only
  on error.
- `#65` (closes B.18): new `_text_mentions_field_skip` widens the
  `chain_auxiliary_endpoints` optional-field decline matcher to common
  English/Chinese skip phrasing, mirroring the existing
  `has_accounts_device` NL matchers.
- `#66` (closes B.23): `workload_choice`'s `weights` branch no longer shares
  code with `custom_rpc` — it goes straight to `needs_weights`, and
  `_question_for_group`'s prompt falls back to the chain template's own
  default methods when nothing custom has been validated yet.
- `#67` (closes B.1): required two rounds of user correction before landing
  on the right design. First attempt wrongly proposed removing the dead
  `needs_google_search` field outright; the user clarified this is a real,
  intentional framework capability (Gemini+google_search grounding as a
  second-pass verification step whenever adding a new chain or custom RPC
  method) that must be wired, not deleted. Second attempt wired it gated on
  the resolver's own per-turn `needs_google_search` judgment; the user
  clarified (confirmed via `AskUserQuestion`) that the intended trigger is
  **unconditional** whenever `google_search_available` is true, not gated by
  the underlying LLM's self-assessed confidence. Final fix: new
  `_augment_chain_resolution_with_search`/`_augment_schema_draft_with_search`
  call `run_google_search_grounding()` (the same function `#48`/`#49` use)
  whenever `state["web_research"]["google_search_available"]` is true, for
  both the chain-identity and custom-RPC/new-chain schema-evidence
  resolution paths; the search evidence summary is appended to the
  user-facing confirmation prompt; the now-genuinely-dead
  `needs_google_search` field is removed from both resolver prompts/schemas
  in `agent/harness/intent.py`.
- `#68` (closes B.17): root cause was not missing go_back/change_group
  navigation (which already worked) but that leaving a stuck
  `custom_rpc_endpoint`/`new_chain_endpoint` question never abandoned the
  half-finished attempt, so routing bounced straight back to the same unmet
  precondition. New `_abandon_stuck_endpoint_question_if_needed` resets the
  in-progress `custom_rpc`/`chain_identity` state at the top of both action
  handlers when leaving one of these two questions; `intent.py` also gained
  resolver guidance to route an explicit give-up/cancel utterance to
  go_back/change_group instead of misreading it as an attempted endpoint.
- `#69` (closes B.21): new `_report_log_highlights` tails `benchmark.log`
  via `job_manager.tail_job_log`, filters to error/failure markers (or
  success markers when the job did not fail), and appends a real log-excerpt
  block to `_report_artifact_entry_response`'s answer instead of only
  listing artifact paths.

Each fix followed the established two-layer verification standard: a
scripted regression test in `tests/test_agent_langgraph_harness.py` that
fails without the fix and passes with it, **and** a live re-verification
against the real DeepSeek-driven `bin/anychain-agent` (the user explicitly
re-confirmed mid-session that script-only verification would not be
acceptable — live re-verification is required to actually prove a fix
works).

**C.3 closed**: installed pandas/matplotlib/scipy/seaborn/numpy/
scikit-learn/statsmodels into `.venv-adk` (`requirements.txt`'s framework
dependency set). 512/512 tests pass, zero skips.

**C.4 closed**: installed a Go toolchain (needed to compile
`tools/fake-node`'s simulator) and ran a real fake-node job past the point
every previous attempt had stalled. Root cause: `blockchain_node_benchmark.sh`
calls bare `python3` for its own analysis/report scripts — not any Agent
venv — and the system `python3` lacks pandas/matplotlib/etc. `.venv`
(documented by `scripts/install_deps.sh` as `PROJECT_VENV_DIR`, a separate
venv from the Agent's own `.venv-adk`) already had every needed package
installed, but the Agent process spawning the benchmark subprocess does not
necessarily have `.venv/bin` on `PATH`. Fixed with a new
`agent/runners/materialize.py::benchmark_subprocess_env` that prepends
`.venv/bin` to `PATH` for the benchmark subprocess when it exists, wired
into both `job_manager.py`'s sync-execution path and `job_worker.py`'s
detached-worker path. Diagnosed with a manual PATH override first to confirm
the hypothesis, then implemented the permanent fix and re-verified with a
genuinely clean run (`job_20260713155411_fb2c0867`, ordinary environment, no
manual PATH override) completing fully end to end: real Vegeta load, real
analysis scripts, real bilingual HTML report, real archiving,
`status: completed`. Regression tests in
`tests/test_agent_benchmark_pipeline.py::BenchmarkSubprocessEnvTest`.

**C.1 closed (not a real gap)**: the user clarified that (a) Docker was only
ever needed because their own dev machine is a Mac without Linux — moot
inside this Linux sandbox, where the real CLI already runs natively, and (b)
the "two AI" chaos-testing bar does not require a genuinely independent
second model; one assistant reactively playing both the Agent-tester and
user-simulator roles from the real prior turn's output already matches the
intended standard.

**New open item found, not fixed this phase** (B.26): an off-topic status
question asked immediately after an informational "confirm and stop"
message (`sync_observe_demo_ack`/`sync_observe_client_setup_ack`, both of
which intentionally clear `pending_question` after showing their message)
lands on `oracle`'s generic "no pending question, next action is X" branch,
which is technically accurate but does not directly answer "is anything
running right now" — confusing in practice (observed live: the user could
not tell from the response whether sync-observe was actually executing).
Surfaced while live-verifying sync-observe via a parallel subagent chaos run
during this phase; deferred as a distinct, smaller UX fix outside this
batch's scope.

**Second defect found via the user's own manual testing this same phase**
(`#70`, CRITICAL): the user pasted a live transcript of their own manual
terminal session testing sync-observe, showing a self-contradictory status
dump — `配置状态：complete` (config status: complete) alongside
`下一个阻塞项：choose sync-observe stop condition` (next blocking item: choose
sync-observe stop condition) in the same response. Root cause:
`oracle.compute_next_action` had a line forcing `config_status` to
`"complete"` whenever `execution_status` showed any job status at all
(`job_running`/`job_completed`/`job_failed`), without checking whether that
job belonged to the in-progress workflow. Because `job`/`latest_job_id`
deliberately survive a full reset (`RESET_PRESERVED_KEYS` — so
`analyze_report`/`status` keep working for the last completed job even
after the user starts a fresh configuration), a brand-new sync-observe setup
with a stale, unrelated `job_failed` left over from before the reset got
falsely marked "complete." Fixed by deleting the override entirely —
`group in {"job_monitoring", ""}` (from `routing.next_group_and_reason`, the
harness's single source of truth for workflow completeness) was already the
correct and sufficient test; the override was redundant at best and actively
wrong whenever a stale job existed. Regression test:
`test_config_status_not_forced_complete_by_a_stale_unrelated_job`
(reproduces the exact live state: a fresh, confirmed-chain sync-observe
setup with a leftover `job_failed`, asserting `config_status == "incomplete"`
and a non-empty `blockers` tuple). Two live user-found defects in one phase
(`#62` and `#70`) is a real signal about where dual-AI chaos testing
systematically under-covers: literal numbered-menu selection of "meta"
options, and stale cross-session/cross-reset state — worth deliberately
adding both to future chaos test design rather than treating as one-offs.

**`#71`, found by a parallel dual-AI chaos subagent (S2)**: dispatched a
subagent to close a real coverage gap — `sync_observe` had never had a
dedicated live dual-AI chaos pass of its own, only incidental coverage from
other sweeps. It confirmed `demo_only`, the fake-node/sync-observe mode
round trip, `endpoint_only`'s failure and single-probe success paths (one
probe against a real public endpoint only, no sustained polling, per the
company's no-real-node-install/no-load-testing constraint), and plain-English
declines for `existing_local_node`/`client_setup` all work correctly — and
found a real defect: no cancel/back path out of a stuck `SYNC_OBSERVE_RPC_URL`
question, the same bug class as `#68`/B.17 (`_abandon_stuck_endpoint_question_if_needed`
had no branch for this pending id). Fixed with a new branch resetting
`sync_observe.source` and related flags; added
`test_go_back_out_of_stuck_sync_observe_rpc_url_abandons_it`; verified live
by restarting the driver and reproducing then escaping the stuck state.

Final verification for this phase: full test suite (498 tests across
`tests/`, including both `#70`'s and `#71`'s regression tests together)
passes with no failures; `tools/check_agent_boundaries.py` passes. `#70`
was additionally re-verified live end to end after the merge (a real
fake-node job submitted to populate `state["job"]`, a session reset, a fresh
incomplete sync-observe setup, confirming the status dump now reads
`配置状态：in_progress` instead of the previous contradictory `complete`).

### Cumulative live/real-transcript + code-review defects fixed: 71

## Phase 30 (2026-07-13, same day) — UNFINISHED, handed off mid-work

After Phase 29 closed, two more changes were made to `agent/harness/groups.py` based on
continued live user testing, and both introduced regressions that were caught by the
user's own manual testing before verification was complete. Work was halted by explicit
user instruction before the fix could be tested or live-verified. Full details,
including exact root causes and the drafted (but unverified) fix, are in
`.agent/task-docs/2026-07-11-agent-known-issues.md` section **D. Active regressions from
this session's own fixes — NOT VERIFIED**. Summary:

1. Made `answer_opening_question` fall through to the real next question instead of a
   static FAQ paragraph when the user is mid-workflow and asks a generic "what do I do
   now" question. Verified live — this part works.
2. Made `sync_observe_demo_ack` auto-execute immediately (per explicit user direction,
   since the old flow forced 4 more questions whose semantics only apply to a real sync
   observation). Only verified with a mocked test, not live — and the user's live test
   immediately found it always fails preflight (`missing: mainnet_rpc_url_reviewed,
   node_process_identity`), because `agent/planners/config_checklist.py`'s waiver logic
   only ever knew about `endpoint_only`, never `demo_only`.
3. Change 1 also increased how often three independently hand-written "nothing to
   confirm" text templates (in `oracle.py` and `groups.py`) fire back-to-back in the
   same turn, producing 2-3 near-duplicate sentences per reply — also found live.

A fix for item 2 was drafted and the code written (thread `sync_observe_source` through
`execution.py` → `benchmark_pipeline.py` → `config_checklist.py`, replacing the old
`sync_observe_local_attribution` boolean with a per-source waiver table in
`config_checklist.py`; two tests updated/added), but **the test suite has not been
re-run since**, and there has been **no live re-verification**. A fix for item 3 was
designed (consolidate the three templates into one `oracle.format_no_pending_question_message`
helper) but never implemented.

**Before continuing this work:** run
`.venv-adk/bin/python3 -m unittest discover -s tests` and
`.venv-adk/bin/python3 tools/check_agent_boundaries.py` first, then live-verify both the
checklist fix (demo_only should reach a submitted job, not a blocked preflight) and the
duplicated-text bug (still unfixed) before trusting or building on any of this phase's
code.
