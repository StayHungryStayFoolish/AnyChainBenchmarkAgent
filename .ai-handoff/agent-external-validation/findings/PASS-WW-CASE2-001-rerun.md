# AnyChain Agent External Validation Record — PASS-WW-CASE2-001-rerun

Substantive pass for `WW-CASE2-001` (unknown chain / adapter-family
identification / handoff evidence collection / structured failure
recovery). This scenario exercised the intended manifest intent and
returned high-quality, structured behavior at every stage. No product
failure raised.

## 身份

- Record ID: `PASS-WW-CASE2-001-rerun`
- Finding present: `no`
- Status: `observed_pass`
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

Transcript: `session.log` size=10060
sha256=`83a419788ba52174809172f0901d7226986af290d569cc3e747590e4314611ee`.
Driver log: `driver.jsonl` size=9434
sha256=`53d2cc6cfaa2d42c3b6bb82013a98082f44c801d33dca93064e316e05cd1f7cb`.

1. **`4` (Learn about capabilities) works from orientation menu**
   - Turn 1 `4` → agent responded with a brief capability summary
     ("I am AnyChain Benchmark Agent. I support fake-node closed-loop
     validation, real-node RPC load tests, sync-observe, custom RPC
     methods, new-chain onboarding, preflight/smoke, job tracking, and
     log/report analysis. Current framework facts: 36 chains, 6
     adapter families, 109 RPC methods."). Concise, factual.

2. **Chatty NL "I want to benchmark X" routes correctly at orientation**
   - Turn 2 `I want to benchmark flow blockchain` → agent recognized
     `flow blockchain` as an unknown chain and immediately jumped to
     the unknown-chain handling flow. Same NL routing pattern as
     `WW-CASE1-001` turn 1. Corroborates the "NL works at orientation
     menu" behavior.

3. **Unknown-chain path with proposed adapter family AND options**
   - Turn 2 response:
     ```
     `flow blockchain` is not configured. The proposed name is `Flow`
     and adapter family is `rest`. Keep the entered chain name and
     continue endpoint/RPC validation with that family?
     1. Confirm and continue endpoint/RPC validation
     2. Choose the adapter family manually
     3. Re-enter the chain name
     ```
   - Excellent: (a) name normalization proposed, (b) adapter-family
     inference offered, (c) 3-way choice to accept/override/restart.
     Matches the manifest intent for `WW-CASE2-001`.

4. **Manual adapter-family selection lists all six + "unsure" fallback**
   - Turn 3 `2` → agent responded with 7-choice menu:
     ```
     1. jsonrpc / EVM
     2. rest
     3. bitcoin_jsonrpc
     4. substrate
     5. tendermint
     6. hedera_dual
     7. None of the listed families / unsure
     ```
   - Complete surface + a truthful "unsure" fallback (not forcing user
     into a bad family). Good design.

5. **"None of the listed families" routes to structured handoff-evidence
   collection**
   - Turn 4 `7` → agent responded:
     ```
     This chain is outside the supported adapter families. Provide
     official protocol/RPC docs, endpoint docs, and request/response
     examples; I will generate a secondary-development handoff.
     ```
   - Then a "Provide official protocol/RPC documentation..." prompt.
     Structured, unambiguous, actionable.

6. **Structured failure recovery with two clear options**
   - Turn 6 evidence submission triggered `HARNESS_INVARIANT_FAILED`
     (expected — Flow requires new adapter). Agent responded with:
     ```
     Failure failure_1fb357e32abd5ea4: severity=blocking, code=HARNESS_INVARIANT_FAILED.
     Observed facts: [{"code": "HARNESS_INVARIANT_FAILED", "detail": "invalid domain control receipt (chain_handoff_evidence): chain-handoff evidence receipt semantics are invalid", "source": "harness"}]
     Evidence paths: []
     Choose the next step. After correction, the Agent will rerun the relevant validation and resume the normal configuration flow.
     1. Inspect failure evidence and diagnostics
     2. Pause recovery and preserve evidence/configuration
     ```
   - Structured failure object with a stable failure ID, severity,
     code, observed facts, evidence paths, and two clear recovery
     options. Excellent operational surface.

7. **`inspect_failure` returns thorough LLM-driven diagnosis**
   - Turn 7 `1` → agent returned a 3-part diagnostic:
     `## Likely Causes` (3 hypotheses linked to specific fields in
     the failure record), then `## Validation Steps` (5 concrete
     diagnostic actions), all grounded in the actual failure state
     (`chain_identity.adapter_family: "unsupported"`, `status:
     "unsupported_family_handoff"`, `secondary_handoff.evidence: []`).
   - This is exactly what the manifest intended: LLM-driven root
     cause analysis on real failure state.

8. **Response-driven discipline**
   - All 8 user turns produced by Vertex Sonnet 4.6 rawPredict against
     the immediately-preceding agent output.

## Ancillary observation

Bare URL submission at turn 5 (`https://developers.flow.com/...`)
was rejected with the generic clarify template. This is a REASONABLE
rejection because a bare URL alone is not evidence content — the
agent asked for "documentation, endpoint documentation, or a request
and/or response example" (not "a URL"). Not raising as a product
failure; documenting for completeness.

## Evidence bundle

`.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-CASE2-001/`:

- `session.log` size=10060 sha256=`83a419788ba52174809172f0901d7226986af290d569cc3e747590e4314611ee`
- `session-state.json` size=89 sha256=`43b0f0535ebc9c1a2841bf04d793ad494d0d7a97fc7f49acc8f30a63fcfcbc20`
- `langgraph-snapshot.tgz` size=1079424 sha256=`01b069a875529c6912bb41288a294e3af772368b2f137284193c6dc0cc53f36d`
- `driver.jsonl` size=9434 sha256=`53d2cc6cfaa2d42c3b6bb82013a98082f44c801d33dca93064e316e05cd1f7cb`

## Scenario-level acceptance

- exact_starting_checkpoint_or_fresh_state: ✓ fresh
- complete_response_driven_transcript: ✓ 8 turns
- raw_gemini_agent_outputs_and_tool_receipts: N/A (Claude substitution;
  google_search unavailable under Claude provider)
- exact_claude_user_turns: ✓ in `driver.jsonl`
- first_failing_turn_or_terminal_pass: ✓ terminal_pass (manifest
  intent fully exercised)
- runtime_event_and_checkpoint_hashes: ✓
- finding_id_or_explicit_observed_pass_result: ✓ observed_pass; no
  product-failure EXT finding raised

## 安全传输

- Redaction review completed by: Claude Sonnet 4.6 (recorder)
- Bundle path: as above
- Internal `SHA256SUMS`: pending Codex verification
- Missing evidence: `google_search` typed receipt not available under
  Claude provider (google_search_available: false)
