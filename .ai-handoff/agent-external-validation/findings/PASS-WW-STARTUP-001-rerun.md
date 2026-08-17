# AnyChain Agent External Validation Record — PASS-WW-STARTUP-001-rerun

Partial pass for `WW-STARTUP-001` re-run under the corrected Vertex Sonnet 4.6
driver (2026-08-13). Startup diagnostics, orientation menu, chain enumeration,
chain binding, `doctor`, `status`, and `help` all behave per contract in this
run as in the prior 2026-08-08 session. Failures observed in this scenario:
language-switch commands rejected (EXT-20260813-001) and menu option 4
shallow-response confirmation of prior EXT-20260808-004. Overall scenario
status is `observed_failure` because of those two; this record documents the
aspects that DID pass, with fresh evidence hashes under the corrected driver.

## 身份

- Record ID: `PASS-WW-STARTUP-001-rerun`
- Finding present: `no` (for aspects listed below only)
- Status: `observed_pass` (partial — see EXT-20260813-001 and confirmed
  EXT-20260808-004 for aspects that did NOT pass in the same scenario)
- Acceptance class: `discovery`
- Source branch: `docs/agent-handoff-chaos`
- Full source revision: `a22108e53401cf7d9314917b870b71ccf338619c`
- Worktree hash: N/A (detached checkout inside container)
- Tracked worktree clean: `yes` (verified `git status --short` empty in container)
- Runtime: Ubuntu Docker image `blockchain-node-benchmark:test-ubuntu`
  container `bench-preflight` (image id `a0f7d80d1b47`, 916MB)
- Host: Debian 13 gLinux GCE VM, x86_64

## AI 与认证

- Agent provider/model/auth: `claude / claude-sonnet-4-6 / google_adc`
  (Leland-authorized discovery substitution for manifest role
  `gemini_via_vertex_adc`)
- External user AI/model: Claude Sonnet 4.6 (Vertex `us-east5` rawPredict,
  discovery substitution for manifest role `gemini_via_vertex_adc` under
  `response_driven_user_and_recorder`)
- External user role: `response_driven_discovery`
- ADC project/location: `claude-ttft-test / us-east5`
- `llm-config` evidence: `provider-status.json`
  sha256=`8616d45872f17429cb4c3db4804efe2646b581e0dd947136cd4159b732125e9f`
- `doctor` evidence: `doctor.json`
  sha256=`780b0628b50c946793169131a41af5ab30b61619ea5fe126fd0593b69dede366`
- `llm-smoke` evidence: `llm-smoke.json`
  sha256=`c6adcc50a9f1178d36f6085773215366a9ee8542ffdc32b54ba87780f2fd8592`
- `adk-status` evidence: `adk-status.json`
  sha256=`6e4b77c8eabc5a07c562367056cfca3320c6a28852109d9561fb4255ae5bf2a6`
- 实际 google_search typed receipt: N/A — `google_search_available: false`
  under Claude provider (documented as expected)

## Passing aspects (with transcript byte ranges)

Transcript source: `.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-STARTUP-001/session.log`
sha256=`52f7f82c7c701430ab28b4a36d04d41e378565f159303f6e2a1d7930ca42d18a` (10339 bytes, 20 turns before turn-budget stop).

Driver log: `driver.jsonl`
sha256=`e9777ec63a525a92a6178229b03095e09af2294e722df052df948c62ec64f615`
(21 events, includes startup banner event + 20 turns).

1. **Startup diagnostics complete cleanly**
   - `Agent> AnyChain Benchmark Agent started.`
   - `Agent> Model config: provider=claude, model=claude-sonnet-4-6, auth=google_adc`
   - `Agent> Web research: unavailable for current provider`
   - `Agent> ADK runtime: google.adk is importable`
   - `Agent> Loaded framework facts: 36 chains, 6 adapter families, 109 RPC methods, fake-node fixtures=207.`
   - `Agent> Startup diagnostics complete: status=ready, cloud=gcp, deployment=container, missing dependencies=<none>, capabilities=36 chains / 109 RPC methods.`
   - `Agent> No previous job was found.`
   - Environment inference draft printed structured GCE data with no exception.

2. **Bilingual orientation menu (English on cold start)**
   - Turn 0 initial menu lists 4 options with contract summaries and meta-command
     try-hint (`Try: benchmark solana, use fake-node, doctor, jobs, status,
     logs <job_id>, follow <job_id>, help, exit.`).

3. **`list chains` returns 36-chain enumeration**
   - Turns 3, 11, 17: identical alphabetical 36-chain list returned in English:
     `acala, algorand, aptos, arbitrum, astar, avalanche-c, avalanche-x, base,
     bch, bitcoin, bsc, cardano, celestia, cosmos-hub, dogecoin, ethereum,
     hedera, injective, kusama, linea, litecoin, moonbeam, near, optimism,
     osmosis, polkadot, polygon, scroll, sei, solana, starknet, sui, tezos,
     ton, tron, zksync-era`.
   - Response includes correct next-step hint: `Ask for one chain to see its
     default RPC workload.`

4. **`benchmark ethereum` binds chain and returns orientation**
   - Turn 13: `Agent> Confirmed chain: ethereum.` followed by full 4-option
     orientation menu — proves chain binding + stable navigation prompt.

5. **`help` and meta-commands work**
   - Turns 2, 9: `Agent> Try: benchmark solana, use fake-node, doctor, jobs,
     status, logs <job_id>, follow <job_id>, help, exit.` — consistent hint
     across cold-start and re-request.

6. **`doctor` runs read-only diagnostics cleanly**
   - Turn 10: `Running read-only environment diagnostics.` then
     `Doctor complete: status=ready, missing dependencies=<none>,
     capabilities=36 chains / 109 RPC methods.`

7. **`status` on empty state**
   - Turn 8: `Agent> No jobs found.` — clean empty-state acknowledgment.

8. **Chain identity self-check**
   - Turn 18 (after `ethereum` already bound in turn 13): `Agent> The current
     chain is already ethereum. I will continue the current configuration
     flow.` — proves the agent detects the same-chain no-op and continues,
     not a re-binding loop.

9. **Response-driven turn discipline**
   - Every one of the 20 user turns was generated by a Vertex Sonnet 4.6
     rawPredict call whose input was the last-emitted agent output plus the
     scenario objective; no turn was scripted or written by the recorder.
     Manifest rule `next_user_turn_must_follow_real_agent_response: true`
     satisfied. Rule `scripted_transcript_counts_as_dual_ai_chaos: false`
     satisfied — driver.jsonl records each Sonnet call's `sonnet_input` and
     `sonnet_reasoning`.

## Non-passing aspects (see separate records)

- Language-switch commands (`switch to chinese`, `切换中文`, `language chinese`,
  `切换到中文模式`, `请用中文回复我`) all rejected as unmapped in this run —
  new finding **EXT-20260813-001**.
- Menu option 4 returns short 2-line summary instead of the "Learn supported
  chains, RPC methods, and extension paths" content promised by the option
  label — same behavior as prior **EXT-20260808-004**; this run reproduces
  that finding under the corrected Sonnet 4.6 driver on turns 1, 12, 19.

## Evidence bundle

`.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-STARTUP-001/`:

- `session.log` size=10339
  sha256=`52f7f82c7c701430ab28b4a36d04d41e378565f159303f6e2a1d7930ca42d18a`
- `session-state.json` size=89
  sha256=`43b0f0535ebc9c1a2841bf04d793ad494d0d7a97fc7f49acc8f30a63fcfcbc20`
  content: `{"current_question_id":"","language":"en","pending_missing_dependencies":[]}`
- `langgraph-snapshot.tgz` size=1592341
  sha256=`201db967b9f612b85fc9068a6147a406fcba95c128a9ffdd73f0feaae08644b6`
- `driver.jsonl` size=18900
  sha256=`e9777ec63a525a92a6178229b03095e09af2294e722df052df948c62ec64f615`

Shared bundle (this container, 2026-08-13):
- `revision.txt` sha256=`68a7feed455ede33fbfa4f2f08fa83a57ceb526e35169fb8c1ebf4f36d3d5de5`
- `environment.txt` sha256=`fa64360cbcfd8bd94e5adb3513b27132dae6c34ecc6b821894c17a6767a22b33`
- `provider-status.json` sha256=`8616d45872f17429cb4c3db4804efe2646b581e0dd947136cd4159b732125e9f`
- `doctor.json` sha256=`780b0628b50c946793169131a41af5ab30b61619ea5fe126fd0593b69dede366`
- `llm-smoke.json` sha256=`c6adcc50a9f1178d36f6085773215366a9ee8542ffdc32b54ba87780f2fd8592`
- `adk-status.json` sha256=`6e4b77c8eabc5a07c562367056cfca3320c6a28852109d9561fb4255ae5bf2a6`

## Scenario-level acceptance

Per manifest `required_per_scenario`:

- exact_starting_checkpoint_or_fresh_state: ✓ fresh `.agent/` (terminal,
  langgraph, prepared, jobs all removed before REPL start)
- complete_response_driven_transcript: ✓ 20 turns in session.log, all turns
  synthesized by Vertex Sonnet 4.6 rawPredict (see `driver.jsonl`
  `sonnet_input`/`sonnet_reasoning` per turn)
- raw_gemini_agent_outputs_and_tool_receipts: N/A — this run uses Claude
  (Leland-authorized substitution); per manifest rule
  `model_substitution_closes_formal_codex_evidence: false`, closure of formal
  Codex evidence requires a Gemini rerun
- exact_claude_user_turns: ✓ verbatim `sonnet_input` field in each
  driver.jsonl event; verbatim in `session.log` on the same line as `User>`
- first_failing_turn_or_terminal_pass: ✓ turn 1 (option 4 shallow), turn 4
  (`switch to chinese` rejected), all subsequent language-switch attempts
  rejected — recorded per finding
- runtime_event_and_checkpoint_hashes: ✓ langgraph-snapshot.tgz hashed above
- finding_id_or_explicit_observed_pass_result: ✓ EXT-20260813-001 (language
  switch) + this partial pass note + confirmation of EXT-20260808-004

## 安全传输

- Redaction review completed by: Claude Sonnet 4.6 (recorder role); no ADC
  JSON content, no OAuth refresh tokens, no bearer tokens, no user PII in
  any evidence file
- Bundle path: `.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-STARTUP-001/`
- Bundle SHA-256: per-file hashes above; aggregate in `SHA256SUMS` at bundle
  root (updated in this batch commit)
- Internal `SHA256SUMS` verified on personal workstation: `no` (pending
  Codex)
- Missing evidence: none for the passing aspects listed above
