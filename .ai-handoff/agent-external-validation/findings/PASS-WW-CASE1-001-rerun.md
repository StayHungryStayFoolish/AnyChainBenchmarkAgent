# AnyChain Agent External Validation Record — PASS-WW-CASE1-001-rerun

Partial pass for `WW-CASE1-001` (supported chain + custom RPC method +
endpoint validation). Many high-value behaviors PASSED (chatty NL
routing at orientation menu, real endpoint probing with evidence
persistence, structured param-by-param JSON-RPC schema confirmation,
detailed contract summary). Two product failures observed and recorded
as EXT-20260813-005 (well-known default method rejected as "custom")
and EXT-20260813-006 (probe-N infinite loop).

## 身份

- Record ID: `PASS-WW-CASE1-001-rerun`
- Finding present: `no` (for aspects listed below only)
- Status: `observed_pass` (partial — see EXT-20260813-005 and -006)
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

Transcript: `session.log` size=12882
sha256=`b16fa4e5f1dfa3d827f45490b3d9684fe12541b807b13977a22a34bdce2cbe31`.
Driver log: `driver.jsonl` size=20700
sha256=`ffb16a94088f83b87cc3cecaa8a7102aa9d6208e763aa9d9d0cb9e012c224cc5`.

1. **Chatty NL routing works at orientation menu**
   - Turn 1 `I want to benchmark ethereum with a custom RPC split` →
     agent extracted BOTH the chain (`ethereum`) AND the custom-RPC
     intent, jumping straight to the endpoint prompt.
   - This is a positive contrast to `WW-NAV-001` turn 4 where the
     same chatty style at a strict free-value field ("Which chain
     do you want to benchmark? Enter a value.") was REJECTED (see
     EXT-20260813-003). Suggests the NL handler is orientation-menu
     specific.

2. **Endpoint validation runs real HTTP probes with evidence
   persistence**
   - Turn 2 `https://mainnet.infura.io/v3/abc123testkey` →
     `endpoint_health_probe: status=401, sample=invalid project id`
     with evidence path `.agent/evidence/endpoint-probes/ethereum-380b4e2faf5e.json`.
   - Turn 3 `https://eth.llamarpc.com` →
     `endpoint_health_probe: status=521, sample=error code: 521`.
   - Turn 4 `https://cloudflare-eth.com` →
     `Endpoint validation passed. Evidence: ...ethereum-c5c08e4a1574.json.`
   - This is high-quality behavior: real probe, real error, real
     evidence bundle, clear pass/fail delineation.

3. **Custom RPC method schema flow is well-structured**
   - Turn 7 `eth_getBlockByNumber` accepted → schema evidence prompt
   - Turn 8 `["0x1b4", true]` → agent parsed params into named JSON-RPC
     schema (`blockNumber` / `fullTransactionObjects`) with types,
     semantic types, encoding, required flag, example — a rich
     structured extraction.
   - Turns 9-10: param-by-param Y/N confirmation with EACH parameter's
     details shown individually. Excellent UX for schema validation.
   - Turn 11 (contract summary): full contract shown with method,
     parameters, response summary (unknown), confidence, evidence
     summary — very informative.

4. **Y branch to probe-live-endpoint offered explicitly**
   - "`Y` authorizes only this probe; `N` returns to evidence
     correction." — permission-explicit design. (The N branch itself
     is buggy per EXT-20260813-006, but the permission gating is
     clear.)

5. **Response-driven discipline**
   - All 20 user turns produced by Vertex Sonnet 4.6 rawPredict against
     the immediately-preceding agent output.

## Non-passing aspects (see separate records)

- Well-known default method `eth_getBalance` rejected at "custom RPC
  method name" prompt → EXT-20260813-005
- Probe-N branch loops back to "provide evidence" with no escape hatch
  → EXT-20260813-006
- `done`/`skip` common terminators rejected (turns 12, 13, 17, 19)
  — same root-cause family as EXT-20260813-003

## Evidence bundle

`.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-CASE1-001/`:

- `session.log` size=12882 sha256=`b16fa4e5f1dfa3d827f45490b3d9684fe12541b807b13977a22a34bdce2cbe31`
- `session-state.json` size=89 sha256=`43b0f0535ebc9c1a2841bf04d793ad494d0d7a97fc7f49acc8f30a63fcfcbc20`
- `langgraph-snapshot.tgz` size=4056468 sha256=`a73a41805352c388f418444737ff37f5c8e60dc81ef9b339276f1080832745de`
- `driver.jsonl` size=20700 sha256=`ffb16a94088f83b87cc3cecaa8a7102aa9d6208e763aa9d9d0cb9e012c224cc5`

## Scenario-level acceptance

- exact_starting_checkpoint_or_fresh_state: ✓ fresh
- complete_response_driven_transcript: ✓ 20 turns
- raw_gemini_agent_outputs_and_tool_receipts: N/A (Claude substitution)
- exact_claude_user_turns: ✓ in `driver.jsonl`
- first_failing_turn_or_terminal_pass: ✓ turn 5 → EXT-20260813-005
- runtime_event_and_checkpoint_hashes: ✓
- finding_id_or_explicit_observed_pass_result: ✓ EXT-20260813-005 +
  EXT-20260813-006 + this partial pass note

## 安全传输

- Redaction review completed by: Claude Sonnet 4.6 (recorder)
- Bundle path: as above
- Internal `SHA256SUMS`: pending Codex verification
- Missing evidence: none
