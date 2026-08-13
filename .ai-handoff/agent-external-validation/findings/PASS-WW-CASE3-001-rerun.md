# AnyChain Agent External Validation Record — PASS-WW-CASE3-001-rerun

Partial pass for `WW-CASE3-001` (unsupported-adapter chain onboarding).
The chain-not-found + adapter-family selection + "None of listed"
branch worked correctly. However, the follow-on evidence collection
rejected chatty prose containing a JSON RPC example — recorded as
`EXT-20260813-007`. The intended manifest handoff pathway is
therefore blocked in practice for users who don't have pure-format
docs to paste.

## 身份

- Record ID: `PASS-WW-CASE3-001-rerun`
- Finding present: `no` (for aspects listed below only)
- Status: `observed_pass` (partial — see EXT-20260813-007)
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

Transcript: `session.log` size=5704
sha256=`0ed045505a976927af24cc35d7e8950c7035ef7d9c761166be7bbafb7e37a289`.
Driver log: `driver.jsonl` size=7097
sha256=`c51074f922c590594209a1e85343b662abde4ad678887c1b33771a06ed479d98`.

1. **Made-up chain `foochain` routed correctly to unknown-chain path**
   - Turn 1 `I want to benchmark foochain` → agent responded:
     `foochain` is not a configured chain or known alias, and the
     model could not reliably confirm it as a supported-family chain.
     Is this a real chain, or should the name be corrected?
     `1. Real chain; continue to protocol confirmation`
     `2. Re-enter the chain name`
   - Correct distinction from `WW-CASE2-001`'s `flow` which was
     inferred as a "rest" family chain. `foochain` was truthfully
     labeled as unconfirmable. Good honesty.

2. **Adapter-family menu identical to CASE2 (deterministic UI)**
   - Turn 3 `2` (after selecting "Real chain") → same 7-choice
     adapter-family menu shown in CASE2. Consistent surface.

3. **"None of listed families" routes to structured handoff prompt**
   - Turn 4 (after `7`) → same structured "provide docs / evidence"
     prompt shown in CASE2. Consistent handoff pathway.

4. **Clean `exit` after failure loop**
   - After two rejected evidence submissions (see EXT-20260813-007),
     Sonnet cleanly typed `exit` and got `Agent> Exited AnyChain
     Benchmark Agent.`. Graceful abandonment path works.

5. **Response-driven discipline**
   - All 6 user turns produced by Vertex Sonnet 4.6 rawPredict against
     the immediately-preceding agent output.

## Non-passing aspect (see separate record)

- Prose + JSON evidence rejected at handoff evidence prompt →
  EXT-20260813-007
- `google_search` handoff (per manifest "hands off to search") not
  exercised because search is `unavailable for current provider`
  (Claude ADC path); would need Gemini provider to test

## Evidence bundle

`.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-CASE3-001/`:

- `session.log` size=5704 sha256=`0ed045505a976927af24cc35d7e8950c7035ef7d9c761166be7bbafb7e37a289`
- `session-state.json` size=89 sha256=`43b0f0535ebc9c1a2841bf04d793ad494d0d7a97fc7f49acc8f30a63fcfcbc20`
- `langgraph-snapshot.tgz` size=436801 sha256=`d7fc3bf5a471c37ad11755714a461d0eb784bf8a2e70614103b935cba212011d`
- `driver.jsonl` size=7097 sha256=`c51074f922c590594209a1e85343b662abde4ad678887c1b33771a06ed479d98`

## Scenario-level acceptance

- exact_starting_checkpoint_or_fresh_state: ✓ fresh
- complete_response_driven_transcript: ✓ 6 turns
- raw_gemini_agent_outputs_and_tool_receipts: N/A (Claude substitution;
  google_search unavailable)
- exact_claude_user_turns: ✓ in `driver.jsonl`
- first_failing_turn_or_terminal_pass: ✓ turn 4 → EXT-20260813-007
- runtime_event_and_checkpoint_hashes: ✓
- finding_id_or_explicit_observed_pass_result: ✓ EXT-20260813-007 +
  this partial pass note

## 安全传输

- Redaction review completed by: Claude Sonnet 4.6 (recorder)
- Bundle path: as above
- Internal `SHA256SUMS`: pending Codex verification
- Missing evidence: search-handoff pathway (Claude provider limitation)
