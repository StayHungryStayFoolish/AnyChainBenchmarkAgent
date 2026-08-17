# AnyChain Agent External Validation Record — PASS-WW-MULTI-001-rerun

Partial pass note for `WW-MULTI-001`. The scenario's core intent
(multi-field submission with field-by-field confirmation) FAILED on
key=value env-style input (recorded as `EXT-20260813-004`), so this
partial pass covers only the recovery path (bare-token chain input
was accepted after the multi-key attempt was rejected).

## 身份

- Record ID: `PASS-WW-MULTI-001-rerun`
- Finding present: `no` (for aspects listed below only)
- Status: `observed_pass` (partial — see EXT-20260813-004)
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

## Passing aspects (with turn / byte references)

Transcript: `session.log` size=5391
sha256=`73a1d78970a7c569e244b7ffe580ce9a9c1cadd002e27a00468d12a30f862cbf`.
Driver log: `driver.jsonl` size=8357
sha256=`ea90e9b77e0655611584e3ae936d017b671f578cf2c5536a4fb878949eee7bff`.

1. **Recovery path after multi-key rejection works**
   - After turn 3 (multi-key rejected), turn 4 `ethereum` (bare token)
     → accepted, config flow starts. The rejection does not permanently
     poison the session.

2. **Pending question is re-emitted after clarify template**
   - Both turn 1 and turn 3 rejections re-emit the current pending
     question (orientation menu at turn 1, "Which chain" at turn 3)
     immediately after the clarify template. Users can retry without
     re-typing a resume command. Good.

3. **Orientation menu → chain prompt → config flow ordering intact**
   - Standard field flow: CLOUD_REGION Y → CLOUD_ZONE Y → MACHINE_TYPE
     Y → LEDGER_DEVICE 1 = same order as observed in other scenarios.

4. **Response-driven discipline**
   - All 8 user turns produced by Vertex Sonnet 4.6 rawPredict against
     the immediately-preceding agent output.

## Non-passing aspect (see separate record)

- Multi-field key=value env-style rejected → EXT-20260813-004
- YAML / JSON / prose variants not exercised (turn budget exhausted)

## Evidence bundle

`.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-MULTI-001/`:

- `session.log` size=5391 sha256=`73a1d78970a7c569e244b7ffe580ce9a9c1cadd002e27a00468d12a30f862cbf`
- `session-state.json` size=89 sha256=`43b0f0535ebc9c1a2841bf04d793ad494d0d7a97fc7f49acc8f30a63fcfcbc20`
- `langgraph-snapshot.tgz` size=991362 sha256=`6cc5eede6bf108d2d52769f063f1094480b4e4c47dd30bc18cd283ce3a6d089b`
- `driver.jsonl` size=8357 sha256=`ea90e9b77e0655611584e3ae936d017b671f578cf2c5536a4fb878949eee7bff`

## Scenario-level acceptance

- exact_starting_checkpoint_or_fresh_state: ✓ fresh
- complete_response_driven_transcript: ✓ 8 turns
- raw_gemini_agent_outputs_and_tool_receipts: N/A (Claude substitution)
- exact_claude_user_turns: ✓ in `driver.jsonl`
- first_failing_turn_or_terminal_pass: ✓ turn 1 → EXT-20260813-004
- runtime_event_and_checkpoint_hashes: ✓
- finding_id_or_explicit_observed_pass_result: ✓ EXT-20260813-004 +
  this partial pass note

## 安全传输

- Redaction review completed by: Claude Sonnet 4.6 (recorder)
- Bundle path: as above
- Internal `SHA256SUMS`: pending Codex verification
- Missing evidence: YAML / JSON / prose variants (see EXT-20260813-004)
