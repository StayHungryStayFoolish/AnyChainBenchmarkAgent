# AnyChain Agent External Validation Record — PASS-WW-NAV-001-rerun

Partial pass for `WW-NAV-001` under corrected Vertex Sonnet 4.6 driver
(2026-08-13). The interruption-and-resume pattern DID hold for the
"informational question" flavor of interruption (`what chains are
supported?`): the agent inline-answers and re-emits the pending
question. The `help` command interruption did NOT re-emit the pending
question — minor UX inconsistency documented as an ancillary
observation in EXT-20260813-003. The "conversational sentence"
interruption resulted in a full clarify-template rejection — recorded
as `EXT-20260813-003`.

## 身份

- Record ID: `PASS-WW-NAV-001-rerun`
- Finding present: `no` (for aspects listed below only)
- Status: `observed_pass` (partial — see EXT-20260813-003)
- Acceptance class: `discovery`
- Source branch: `docs/agent-handoff-chaos`
- Full source revision: `a22108e53401cf7d9314917b870b71ccf338619c`
- Tracked worktree clean: `yes`
- Runtime: Ubuntu Docker `blockchain-node-benchmark:test-ubuntu`
  container `bench-preflight`

## AI 与认证

- Agent provider/model/auth: `claude / claude-sonnet-4-6 / google_adc`
- External user AI/model: Claude Sonnet 4.6 (Vertex `us-east5` rawPredict,
  discovery substitution for manifest role `gemini_via_vertex_adc`)
- External user role: `response_driven_discovery`
- ADC project/location: `claude-ttft-test / us-east5`
- Preflight evidence hashes: see PASS-WW-STARTUP-001-rerun

## Passing aspects (with turn / byte references)

Transcript: `session.log` size=4747
sha256=`a6a1895622c1ed51a186d0b5b780a141eb32ee5eca9a31564c14a79df1bd185e`.
Driver log: `driver.jsonl` size=6912
sha256=`894dc098a6e13879aec0782e022b9cccb476fa26d60938d324aa0e165e37cd20`.

1. **Informational-question interruption preserves pending question**
   - Turn 2 at pending "Which chain do you want to benchmark?": input
     `what chains are supported?` → agent inline-lists all 36 chains
     AND re-emits the pending "Which chain" prompt. Excellent context
     preservation.

2. **`help` interruption does NOT re-emit pending question**
   - Turn 3 at pending "Which chain": input `help` → agent emits the
     starter hint (`Try: benchmark solana, use fake-node, doctor, ...`)
     but drops the pending question. Minor UX gap; see
     EXT-20260813-003 ancillary observation.

3. **Canonical resume via bare chain token works**
   - Turn 5 `ethereum` → `Agent> Confirmed chain: ethereum.` and the
     config flow resumes at CLOUD_REGION.

4. **Full 8-turn interruption+resume+config-flow sequence**
   - `1` → chain prompt → interrupted by `what chains are supported?`
     → interrupted by `help` → chatty resume attempt fails → bare
     `ethereum` succeeds → CLOUD_REGION Y → CLOUD_ZONE Y → MACHINE_TYPE
     is the next pending question at driver stop.

5. **Response-driven discipline**
   - All 8 user turns produced by Vertex Sonnet 4.6 rawPredict against
     the immediately-preceding agent output.

## Non-passing aspect (see separate record)

- Chatty natural-language resume attempt with recognizable chain token
  → EXT-20260813-003 (plus ancillary `help`-drops-pending observation)

## Evidence bundle

`.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-NAV-001/`:

- `session.log` size=4747 sha256=`a6a1895622c1ed51a186d0b5b780a141eb32ee5eca9a31564c14a79df1bd185e`
- `session-state.json` size=89 sha256=`43b0f0535ebc9c1a2841bf04d793ad494d0d7a97fc7f49acc8f30a63fcfcbc20`
- `langgraph-snapshot.tgz` size=994070 sha256=`aea3a28d00d025b46753cc55b41135e47d8f49dc69f22ae9123c17721fa1cd4a`
- `driver.jsonl` size=6912 sha256=`894dc098a6e13879aec0782e022b9cccb476fa26d60938d324aa0e165e37cd20`

## Scenario-level acceptance

- exact_starting_checkpoint_or_fresh_state: ✓ fresh
- complete_response_driven_transcript: ✓ 8 turns
- raw_gemini_agent_outputs_and_tool_receipts: N/A (Claude substitution)
- exact_claude_user_turns: ✓ in `driver.jsonl`
- first_failing_turn_or_terminal_pass: ✓ turn 4 (chatty resume) → EXT-20260813-003
- runtime_event_and_checkpoint_hashes: ✓
- finding_id_or_explicit_observed_pass_result: ✓ EXT-20260813-003 +
  this partial pass note

## 安全传输

- Redaction review completed by: Claude Sonnet 4.6 (recorder)
- Bundle path: as above
- Internal `SHA256SUMS`: pending Codex verification
- Missing evidence: none
