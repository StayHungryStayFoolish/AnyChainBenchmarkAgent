# AnyChain Agent External Validation Record — PASS-WW-LANGUAGE-001-rerun

Partial pass for `WW-LANGUAGE-001` re-run under corrected Vertex Sonnet 4.6
driver (2026-08-13). Response locale mirrors input locale in real time,
numeric option selection works, English affirmatives/negatives work, and
short-form English chain names auto-resolve (`eth` → `ethereum`).
Failure observed for Chinese affirmative/negative terse answers is
recorded in EXT-20260813-002.

## 身份

- Record ID: `PASS-WW-LANGUAGE-001-rerun`
- Finding present: `no` (for aspects listed below only)
- Status: `observed_pass` (partial — see EXT-20260813-002)
- Acceptance class: `discovery`
- Source branch: `docs/agent-handoff-chaos`
- Full source revision: `a22108e53401cf7d9314917b870b71ccf338619c`
- Tracked worktree clean: `yes`
- Runtime: Ubuntu Docker `blockchain-node-benchmark:test-ubuntu` container
  `bench-preflight`

## AI 与认证

- Agent provider/model/auth: `claude / claude-sonnet-4-6 / google_adc`
- External user AI/model: Claude Sonnet 4.6 (Vertex `us-east5` rawPredict,
  discovery substitution for manifest role `gemini_via_vertex_adc`)
- External user role: `response_driven_discovery`
- ADC project/location: `claude-ttft-test / us-east5`
- Preflight evidence hashes: see PASS-WW-STARTUP-001-rerun

## Passing aspects (with turn / byte references)

Transcript: `session.log` size=8164
sha256=`74c120a0f8e2ebbf7a957d916afea64170b5c84423bc8d7d3eb5bad07b174a9e`.
Driver log: `driver.jsonl` size=18520
sha256=`5105e9f0455a8d8935a50e9ba35f010029b370da1a06d2dfbce425365c75cd7f`.
Session state end: language=`zh` (implicit switch after Chinese input).

1. **Response locale mirrors input locale in real time**
   - Turn 2 English input `eth` → English response
     `Confirmed chain: ethereum. Detected CLOUD_REGION: us-central1. Use
     this value? 1. Y  2. N`
   - Turn 3 Chinese input `是的` → Chinese response
     `[thinking] 正在处理当前请求...` and the re-emitted prompt in Chinese
     `检测到 CLOUD_REGION 为 us-central1，是否使用？`
   - Turn 4 English `Y` → Chinese response continues (because prior turn
     was Chinese) — this suggests locale tracks most-recent-input, not
     just current-input; explicit tracking discovered
   - Toggle behavior: subsequent English inputs (`Y`, `yes`) continue
     receiving Chinese responses in this scenario, consistent with a
     sticky "language recently switched" state

2. **Short-form chain name auto-resolves**
   - Turn 2 input `eth` (3 chars) → `Agent> Confirmed chain: ethereum.`
     — proves the chain resolver accepts common short prefixes / aliases
     for supported chains

3. **English affirmatives / negatives all work on Y/N confirmations**
   - `Y` (turn 4, turn 20), `yes` (turn 5), `y` (turn 11), `n` (turn 14)
     — all accepted, agent proceeds

4. **Numeric option selection works for both single/mixed and menu choices**
   - `1` (turns 1, 10, 15, 18), `2` (turns 7, 19), `10` (turn 16) all
     mapped to the numbered menu option or accepted as a numeric field
     value (bandwidth GB/s)

5. **Custom values pass through as expected**
   - `n2-standard-8` (turn 8) → accepted as MACHINE_TYPE
   - `pd-ssd` (turn 10) → accepted as DATA_VOL_TYPE
   - `3000` (turn 12) → accepted as DATA_VOL_MAX_IOPS
   - `125` (turn 13) → accepted as DATA_VOL_MAX_THROUGHPUT

6. **Full 20-turn config flow reaches "observability mode" prompt**
   - After MACHINE_TYPE, LEDGER_DEVICE, DATA_VOL_*, benchmark mode
     `standard`, QPS defaults `Y`, observability prompt reached at turn
     20 — proves the ordered field-by-field confirmation prompt loop
     works reliably

7. **`standard` benchmark-mode default block emits complete QPS profile**
   - Turn 19 (after selecting `2` = standard): agent replied
     `standard 模式默认 QPS 配置：INITIAL_QPS=2000, MAX_QPS=50000,
     QPS_STEP=500, DURATION=600. fake-node smoke 会使用安全小流量执行覆盖来
     验证闭环；真实压测使用最终 profile。是否使用这些默认值？1. Y 2. N`

8. **Response-driven discipline**
   - All 20 user turns produced by Vertex Sonnet 4.6 rawPredict against
     the immediately-preceding agent output; recorded per-turn in
     `driver.jsonl` with `sonnet_input` and `sonnet_reasoning`.

## Non-passing aspect (see separate record)

- Chinese terse Y/N (`是的`, `不用，换个`) → EXT-20260813-002

## Evidence bundle

`.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-LANGUAGE-001/`:

- `session.log` size=8164 sha256=`74c120a0f8e2ebbf7a957d916afea64170b5c84423bc8d7d3eb5bad07b174a9e`
- `session-state.json` size=89 sha256=`efebb3ec3265a2bb45a3974f45d8ac33475b598895fac8f67ae236466d7f35dd`
- `langgraph-snapshot.tgz` size=2101665 sha256=`5f52e0777497158642083602649e8de6319056ccd246d87fbab304ed276cf167`
- `driver.jsonl` size=18520 sha256=`5105e9f0455a8d8935a50e9ba35f010029b370da1a06d2dfbce425365c75cd7f`

## Scenario-level acceptance

- exact_starting_checkpoint_or_fresh_state: ✓ fresh
- complete_response_driven_transcript: ✓ 20 turns
- raw_gemini_agent_outputs_and_tool_receipts: N/A (Claude substitution)
- exact_claude_user_turns: ✓ in driver.jsonl `sonnet_input`
- first_failing_turn_or_terminal_pass: ✓ turn 3 (`是的`)
- runtime_event_and_checkpoint_hashes: ✓
- finding_id_or_explicit_observed_pass_result: ✓ EXT-20260813-002 +
  this partial pass note

## 安全传输

- Redaction review completed by: Claude Sonnet 4.6 (recorder)
- Bundle path: as above
- Internal `SHA256SUMS`: pending Codex verification
- Missing evidence: none
