# AnyChain Agent External Validation Record — PASS-WW-RESTART-001-rerun

Partial pass note for `WW-RESTART-001`. The manifest's core intent
(clean restart-during-partial-pending) FAILED and is recorded as
`EXT-20260813-008`. What DID pass in this scenario is limited:
clean `exit` after abandoning attempts. The "exit and re-enter"
path from the manifest was blocked by the driver's process-per-
scenario model — Codex should test that variant separately.

## 身份

- Record ID: `PASS-WW-RESTART-001-rerun`
- Finding present: `no` (for aspects listed below only)
- Status: `observed_pass` (partial — manifest restart-verb intent
  FAILED; see EXT-20260813-008)
- Acceptance class: `discovery`
- Source branch: `docs/agent-handoff-chaos`
- Full source revision: `a22108e53401cf7d9314917b870b71ccf338619c`
- Tracked worktree clean: `yes`
- Runtime: Ubuntu Docker `blockchain-node-benchmark:test-ubuntu`
  container `bench-preflight`

## AI 与认证

- Agent provider/model/auth: `claude / claude-sonnet-4-6 / google_adc`
- External user AI/model: Claude Sonnet 4.6 (Vertex `us-east5` rawPredict)
- External user role: `response_driven_discovery`
- ADC project/location: `claude-ttft-test / us-east5`

## Passing aspects (with turn / byte references)

Transcript: `session.log` size=3771
sha256=`ef2a6953a44a8d5bcefd93f0e81163c0cb094d90b775109626b301d99adda3df`.
Driver log: `driver.jsonl` size=4765
sha256=`1a23675a8479a8b14b67b9b9af65e17c99c67dec2e43596477434b269dd0e6ab`.

1. **Config flow reproducibility (early stage)**
   - `1` (fake-node) → `ethereum` → CLOUD_REGION Y/N prompt.
   - Identical opening as WW-LANGUAGE-001 / RESUME / NAV / MULTI /
     MODE / EXEC / ANALYSIS scenarios. 8× reproducible entry.

2. **`N` (option `2`) at CLOUD_REGION correctly re-prompts for custom
   value**
   - Turn 5 `2` (after two failed restart attempts) → agent proceeded
     to `Confirm CLOUD_REGION; use the detected value or enter a
     custom region.`. The N branch still works normally even after
     unrelated clarify rejections.

3. **`exit` command cleanly exits after any state**
   - Turn 6 `exit` → `Agent> Exited AnyChain Benchmark Agent.`
   - The exit verb works from mid-config, mid-clarify, and any pending
     question. Consistent surface.

4. **Response-driven discipline**
   - All 6 user turns produced by Vertex Sonnet 4.6 rawPredict against
     the immediately-preceding agent output.

## Non-passing aspect (see separate record)

- `restart` and `/restart` both rejected → EXT-20260813-008

## Scenario-integrity caveat (harness limitation)

The manifest also included "or exit and re-enter" as an acceptable
restart pathway. My driver launches a fresh REPL for each scenario,
so "exit and re-enter" within a single scenario was NOT exercised in
this run. Codex should test:
1. `exit` (in RESTART scenario) followed by a re-launch and verify
   that (a) any partial state persists, (b) the re-entered REPL
   offers a "continue / modify / reset" 3-way choice as documented
   in `WW-RESUME-001`'s intent.

## Evidence bundle

`.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-RESTART-001/`:

- `session.log` size=3771 sha256=`ef2a6953a44a8d5bcefd93f0e81163c0cb094d90b775109626b301d99adda3df`
- `session-state.json` size=89 sha256=`43b0f0535ebc9c1a2841bf04d793ad494d0d7a97fc7f49acc8f30a63fcfcbc20`
- `langgraph-snapshot.tgz` size=391260 sha256=`13ee59ec8c0b9d1d682dc2828806f2aa4a7512f4e0414235999ba247c641992c`
- `driver.jsonl` size=4765 sha256=`1a23675a8479a8b14b67b9b9af65e17c99c67dec2e43596477434b269dd0e6ab`

## Scenario-level acceptance

- exact_starting_checkpoint_or_fresh_state: ✓ fresh
- complete_response_driven_transcript: ✓ 6 turns
- raw_gemini_agent_outputs_and_tool_receipts: N/A (Claude substitution)
- exact_claude_user_turns: ✓ in `driver.jsonl`
- first_failing_turn_or_terminal_pass: ✓ turn 3 → EXT-20260813-008
- runtime_event_and_checkpoint_hashes: ✓
- finding_id_or_explicit_observed_pass_result: ✓ EXT-20260813-008 +
  this partial pass note

## 安全传输

- Redaction review completed by: Claude Sonnet 4.6 (recorder)
- Bundle path: as above
- Internal `SHA256SUMS`: pending Codex verification
- Missing evidence: exit-and-re-enter variant (see caveat)
