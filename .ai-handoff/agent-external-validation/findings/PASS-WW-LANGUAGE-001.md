# AnyChain Agent External Validation Record — PASS-WW-LANGUAGE-001 (partial)

Scenario `WW-LANGUAGE-001` (chinese_english_short_answers_punctuation_
clarification) partial-pass note. Per-turn language detection,
clarification-bucket protocol structure, Chinese meta-command
recognition, and locale-appropriate exit-message rendering all behave
per contract. Related failures recorded as EXT-20260808-007 (locale-lock
intent unrecognized) and EXT-20260808-008 (punctuation not stripped
before intent match).

## 身份

- Record ID: `PASS-WW-LANGUAGE-001`
- Finding present: `no` (for aspects listed below only)
- Status: `observed_pass` (partial)
- Acceptance class: `discovery`
- Source branch: `docs/agent-handoff-chaos`
- Full source revision: `a22108e53401cf7d9314917b870b71ccf338619c`

## AI 与认证

- Agent provider/model/auth: `claude / claude-sonnet-4-6 / google_adc`
- External user AI/model: Claude Sonnet 4.6
- External user role: `response_driven_discovery`
- ADC project/location: `claude-ttft-test / us-east5`

## Passing aspects (with transcript byte ranges)

Transcript source: `.agent/external-validation/a22108e5.../transcripts/
WW-LANGUAGE-001/session.log` sha256=`15e6ba1b8232876ce69823cb25165868
b872fa2cdb13aa67d06f63df8e36d2f9` (6072 bytes, 5 turns).

1. **Fresh startup renders in English by default**
   - No prior `.agent/terminal/session.json` → fresh session with no
     established locale → startup output rendered in English (including
     orientation menu and meta-command hint)

2. **Per-turn language detection works both directions**
   - Turn 1 English input "y" → English clarification response
   - Turn 2 Chinese input "从现在开始，请只用中文和我对话" → Chinese
     clarification response
   - Turn 3 Chinese input "使用 solana" → Chinese success response
   - Turn 4 English-dominant input "??? benchmark solana ???" →
     English clarification response
   - Turn 5 English input "exit" (final session locale en) → English
     exit response "Exited AnyChain Benchmark Agent."
   - Note: this confirms EXT-20260808-003 hypothesis that per-turn
     input-locale detection is the active response-locale selector,
     not session state

3. **Clarification-bucket protocol structure**
   - Consistent format across turns: `[thinking]` marker →
     bucket-header sentence ("To avoid applying only part of this
     turn, clarify the following items that were not mapped safely:"
     / Chinese equivalent) → hyphenated list of unmapped items →
     orientation-menu re-emission
   - Protocol is discoverable and consistent; user knows what to do
     next (rephrase or pick from menu)

4. **Chinese meta-command recognition works for known intents**
   - Turn 3 `使用 solana` → agent recognized as chain-binding intent,
     responded "已确认链为 `solana`。" — proves the intent classifier
     can accept Chinese-form meta-commands for chain binding (this
     is the Chinese equivalent of the English `use solana`
     meta-command listed in the startup hint)

5. **Exit-message locale honors last-turn locale**
   - Turn 5 `exit` (English input, session locale en from turn 4) →
     "Exited AnyChain Benchmark Agent." (English)
   - Contrast: WW-STARTUP-001 turn 7 same `exit` input, session
     locale zh from turn 6 → "已退出 AnyChain Benchmark Agent。"
     (Chinese)
   - Consistent per-turn detection behavior applied to exit path too

## Non-passing aspects (see separate records)

- Explicit locale-lock directive not recognized as system command →
  **EXT-20260808-007**
- Punctuation-decorated commands fragmented instead of normalized →
  **EXT-20260808-008**

## Latency observation (adjacent to EXT-20260808-006)

Turn latencies for this scenario (driver-measured):

- Turn 1 "y" → 53s
- Turn 2 "从现在开始，请只用中文和我对话" → 79s
- Turn 3 "使用 solana" → 57s
- Turn 4 "??? benchmark solana ???" → 45s
- Turn 5 "exit" → <5s

All non-trivial turns (1-4) had 45-79s latency. This pattern extends
EXT-20260808-006 beyond just novel-chain classification — it may
indicate that any input reaching the LLM-reasoning fallback path
(rather than a fast dispatch table) incurs this latency. Only exact
meta-command or numeric-option matches short-circuit to the fast
path. Codex should treat EXT-20260808-006 as a broader latency
finding when analyzing the classification pipeline.

## Evidence bundle

`.agent/external-validation/a22108e5.../transcripts/WW-LANGUAGE-001/`:

- `session.log` sha256=`15e6ba1b8232876ce69823cb25165868b872fa2cdb13aa
  67d06f63df8e36d2f9` (6072 bytes, 5-turn PTY transcript)
- `session-state.json` sha256=`43b0f0535ebc9c1a2841bf04d793ad494d0d7a
  97fc7f49acc8f30a63fcfcbc20` (89 bytes, final session state)
- `langgraph-snapshot.tgz` sha256=`82adaee2d01dd9641b1f1e9ccd8bb2cfba
  f12998dca2d42aa27f24596cff791c` (395328 bytes, langgraph
  checkpointer state)

## Scenario-level acceptance

Per manifest `required_per_scenario`:

- exact_starting_checkpoint_or_fresh_state: ✓ fresh `.agent/` state,
  container not restarted
- complete_response_driven_transcript: ✓ session.log 5 turns
- raw_agent_outputs_and_tool_receipts: ✓ raw output in session.log
- exact_claude_user_turns: ✓ verbatim in session.log
- first_failing_turn_or_terminal_pass: ✓ turn 2 (locale-lock), turn
  4 (punctuation) per findings
- runtime_event_and_checkpoint_hashes: ✓ langgraph-snapshot.tgz
- finding_id_or_explicit_observed_pass_result: ✓ two findings + this
  partial pass note
