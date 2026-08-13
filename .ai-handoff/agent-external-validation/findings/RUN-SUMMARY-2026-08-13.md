# AnyChain Agent External Validation — Run Summary 2026-08-13

Consolidated summary for the 12-scenario `WW-*` external-validation
run under the Vertex Sonnet 4.6 dual-AI Chaos driver. All 12
scenarios executed; each produced a fresh-state transcript, a
response-driven end-user turn stream from Vertex Sonnet 4.6
`us-east5` rawPredict, and per-file SHA-256 hashes.

## Framing

- Manifest: `.ai-handoff/agent-external-validation/VALIDATION_MANIFEST.yaml`
- Source revision under test: `a22108e53401cf7d9314917b870b71ccf338619c`
- Source branch: `docs/agent-handoff-chaos`
- Acceptance class: `discovery` (Leland-authorized Claude
  substitution for manifest role `gemini_via_vertex_adc`)
- Agent under test: `claude / claude-sonnet-4-6 / google_adc`
  (Vertex, region-agnostic ADC)
- End-user simulator: `claude-sonnet-4-6` via Vertex `us-east5`
  rawPredict (ADC project `claude-ttft-test`)
- Runtime: Docker container `bench-preflight` on Debian 13 gLinux VM
- Bundle root:
  `.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/`
  (NOT committed to git; per-file SHA-256 hashes are recorded in
  each finding and in `SHA256SUMS` at the bundle root)

## Scenario results

| Scenario | Outcome | Turns | Wall (s) | Findings |
|---|---|---|---|---|
| WW-STARTUP-001 | pass (rerun) | 20 | prev | `PASS-WW-STARTUP-001-rerun.md` + `EXT-20260813-001` (language switch) |
| WW-LANGUAGE-001 | pass (rerun) | 20 | 388 | `PASS-WW-LANGUAGE-001-rerun.md` + `EXT-20260813-002` (Chinese Y/N terse) |
| WW-RESUME-001 | pass (rerun, partial) | 20 | 332 | `PASS-WW-RESUME-001-rerun.md` — 3-way choice not exercised (harness caveat) |
| WW-NAV-001 | pass (partial) | 8 | 191 | `PASS-WW-NAV-001-rerun.md` + `EXT-20260813-003` (chatty NL rejected in strict field) |
| WW-MULTI-001 | pass (partial) | 8 | 184 | `PASS-WW-MULTI-001-rerun.md` + `EXT-20260813-004` (multi-key env rejected) |
| WW-MODE-001 | pass (partial) | 10 | 134 | `PASS-WW-MODE-001-rerun.md` — mode-switch not exercised (harness caveat) |
| WW-CASE1-001 | pass (partial) | 20 | 492 | `PASS-WW-CASE1-001-rerun.md` + `EXT-20260813-005` (default method rejected as custom) + `EXT-20260813-006` (probe-N infinite loop) |
| WW-CASE2-001 | substantive pass | 8 | 302 | `PASS-WW-CASE2-001-rerun.md` — unknown-chain path fully exercised |
| WW-CASE3-001 | pass (partial) | 6 | 172 | `PASS-WW-CASE3-001-rerun.md` + `EXT-20260813-007` (evidence parser rejects prose+JSON) |
| WW-EXEC-001 | substantive pass | 20 | 241 | `PASS-WW-EXEC-001-rerun.md` — fake-node smoke submitted `job_20260813103335_267777e4` |
| WW-ANALYSIS-001 | pass (partial) | 20 | 227 | `PASS-WW-ANALYSIS-001-rerun.md` — analysis commands not exercised (harness caveat); turn-20 anomaly noted |
| WW-RESTART-001 | pass (partial) | 6 | 262 | `PASS-WW-RESTART-001-rerun.md` + `EXT-20260813-008` (restart verbs not recognized) |

Total wall: ~2925 s (48 min 45 s) for the 11-scenario `run_remaining`
run; WW-STARTUP-001 was rerun in an earlier session.

## Product failures raised (8 total in this run)

Reused prior-session IDs (EXT-20260808-001..008 remain intact) and
added 8 new discovery findings dated 2026-08-13:

- `EXT-20260813-001`: no language-switch command recognized;
  `session-state.language` never updates
- `EXT-20260813-002`: Chinese terse Y/N answers (`是的`, `不用，换个`)
  rejected on Y/N confirmations
- `EXT-20260813-003`: chatty NL sentence with recognizable chain
  token rejected at strict free-value field; `help` inside pending
  question drops pending context (ancillary)
- `EXT-20260813-004`: multi-field key=value env-style input rejected
  without key extraction
- `EXT-20260813-005`: well-known default method `eth_getBalance`
  rejected at "custom RPC method name" prompt; only offered "Cancel
  plan"
- `EXT-20260813-006`: probe-N branch after Y-Y-Y contract confirm
  loops back to "provide evidence" with no escape hatch; `done` /
  `skip` common terminators also rejected
- `EXT-20260813-007`: unsupported-adapter handoff evidence prompt
  rejects prose containing a valid JSON RPC example
- `EXT-20260813-008`: neither `restart` nor `/restart` recognized;
  starter hint doesn't list a restart verb

**Common root-cause hypothesis** (see individual findings for detail):
`EXT-20260813-001/003/004/005/006/007/008` all bucket into the same
generic clarify template with the literal input echoed. This suggests
a single strict-match dispatcher at the pending-question / orientation
handler with no NL / mixed-content / synonym extraction. Fixing this
one dispatcher would likely close 7 of 8 findings.

## Substantive passing behaviors

- **Startup diagnostics** (WW-STARTUP-001): fresh cold-start banner,
  environment inference draft with structured disk / iface candidates,
  orientation menu, hint line
- **Endpoint validation** (WW-CASE1-001 turns 2-4): real HTTP probe
  against user-provided RPC URLs, evidence bundle persistence,
  clear pass/fail semantics
- **Custom-RPC schema flow** (WW-CASE1-001 turns 8-11): param-by-param
  JSON-RPC schema extraction with types / semantic types / encoding
  / meaning; full contract summary with confidence / conflicts
- **Unknown-chain / adapter-family menu** (WW-CASE2-001 turns 2-4):
  3-way accept/override/restart + 7-choice adapter family + "unsure"
  fallback; structured handoff evidence collection
- **Structured failure recovery** (WW-CASE2-001 turn 6-7):
  `HARNESS_INVARIANT_FAILED` with stable failure ID + LLM-driven
  root-cause hypothesis (3 causes + 5 validation steps)
- **Full job submission** (WW-EXEC-001 turn 20): 20-turn config flow
  reaches `Fake-node smoke submitted: status=ok, job_id=...`
  with actionable follow-up commands
- **Response locale mirrors input** (WW-LANGUAGE-001 turns 2-5):
  agent replies in the language of the most-recent input; short-form
  chain aliases resolve (`eth` → `ethereum`)
- **Interruption context preservation** (WW-NAV-001 turn 2):
  `what chains are supported?` at pending "Which chain" inline-answers
  AND re-emits pending question

## Scenario-integrity caveats (harness limitations, NOT product bugs)

- **WW-RESUME-001**: driver's default `.agent/` cleanup wiped
  `terminal/` and `langgraph/` state, so the "continue / modify /
  reset" 3-way choice at REPL launch never appeared. Codex must
  seed a prior-partial checkpoint to exercise this branch.
- **WW-MODE-001**: driver's 6-10 turn efficiency heuristic caused
  Sonnet to `exit` at turn 10 before reaching the RPC-mode /
  observability stages where mode switching would occur. Codex must
  increase turn budget or preseed a later-stage checkpoint.
- **WW-ANALYSIS-001**: driver cleanup wiped the fake-node job from
  `WW-EXEC-001`, so Sonnet had no job to analyze and re-walked the
  config flow. Codex must preserve prior-scenario job state or seed
  a synthetic prior-job fixture.
- **WW-RESTART-001**: driver launches one REPL per scenario, so the
  manifest's "exit and re-enter" pathway was not exercised. Codex
  should test it separately.

## Anomalies (not elevated to EXT — pending Codex reproduction)

- **WW-ANALYSIS-001 turn 20**: final `Y` to "Configuration is
  collected. Run preflight and smoke?" returned "The final state of
  this turn's external operation is uncertain. The Agent stopped
  without advancing the session configuration." — differs from
  identical turn in `WW-EXEC-001` which returned submit-ok. Could
  indicate race with prior-scenario subprocess or genuine
  retry-recovery behavior. Recorded in `PASS-WW-ANALYSIS-001-rerun.md`.

## Coverage gaps (not exercised in this run)

- `google_search` typed receipts: `google_search_available: false`
  under Claude provider — the entire real-web-search pathway is
  invisible. `WW-CASE2-001` and `WW-CASE3-001` manifest intents
  explicitly reference search handoff.
- `analyze`, `logs <id>`, `follow <id>`, `jobs` verbs (WW-ANALYSIS-001)
- Mode-switch invalidation (WW-MODE-001)
- 3-way continue/modify/reset resume prompt (WW-RESUME-001)
- Real-node benchmark (`use real-node` option not exercised in any
  scenario — all runs used fake-node)
- Sync-observe (`Start sync-observe` option not exercised)

## Preflight evidence hashes (shared across all scenarios)

Bundle root
`.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/`:

- `revision.txt` sha256=`68a7feed455ede33fbfa4f2f08fa83a57ceb526e35169fb8c1ebf4f36d3d5de5`
- `environment.txt` sha256=`fa64360cbcfd8bd94e5adb3513b27132dae6c34ecc6b821894c17a6767a22b33`
- `provider-status.json` sha256=`8616d45872f17429cb4c3db4804efe2646b581e0dd947136cd4159b732125e9f`
- `doctor.json` sha256=`780b0628b50c946793169131a41af5ab30b61619ea5fe126fd0593b69dede366`
- `llm-smoke.json` sha256=`c6adcc50a9f1178d36f6085773215366a9ee8542ffdc32b54ba87780f2fd8592`
- `adk-status.json` sha256=`6e4b77c8eabc5a07c562367056cfca3320c6a28852109d9561fb4255ae5bf2a6`
- `SHA256SUMS` sha256=`f8a2a53adaf58169ecb6a4fe51d4f1b9f78ee7d0c7d499e69fcd4892d18c1710`

## Model-substitution disclosure

Per the master briefing: this run uses `claude-sonnet-4-6` as the
end-user simulator, substituting for the manifest's Gemini-via-Vertex
role. This substitution is explicitly limited to `Acceptance class:
discovery`. It disables the `formal` closure pathway for all
findings raised here. Codex must reproduce on the exact source
revision with the same or equivalent user simulator to advance any
finding to `formal` closure.

## Follow-up owners

- **Codex (personal workstation)**: reproduce EXT-20260813-001
  through -008; fill in `## 个人电脑 Codex 确定性修复` sections;
  design framework-level repairs (NOT phrase / keyword patches);
  return with `Repair revision` and `Focused deterministic tests`.
- **Codex or workstation validator**: re-run WW-RESUME-001,
  WW-MODE-001, WW-ANALYSIS-001, WW-RESTART-001 with corrected
  harness state seeding to actually exercise the intended manifest
  branches.
- **Provider / infrastructure**: if `google_search` typed-receipt
  validation is required for closure, re-run under Gemini provider
  path (this run used Claude provider only).

## Prior-session commits preserved

Kept intact: `EXT-20260808-001..008`, `PASS-WW-STARTUP-001`,
`PASS-WW-RESUME-001`, `PASS-WW-LANGUAGE-001`, and the two
`*-rerun` files from the earlier session (`PASS-WW-STARTUP-001-rerun`
and `PASS-WW-LANGUAGE-001-rerun`). No prior file was modified in
this run.

---

Recorder: Claude Sonnet 4.6 (2026-08-13, driver
`/tmp/anychain-driver/driver.py` + `run_remaining.py`)
