# AnyChain Agent External Validation Record — PASS-WW-MODE-001-rerun

Partial pass for `WW-MODE-001`. The scenario's core intent (mode
switching mid-flow to trigger dependent-field invalidation) COULD NOT
BE EXERCISED — the Sonnet driver's 10-turn efficiency heuristic
caused it to type `exit` at turn 10 while still deep in the config
prompt sequence (at DATA_VOL_MAX_THROUGHPUT), never reaching the
RPC-mode or observability stages where mode switching would occur.
Documented as a driver caveat (NOT a product failure). Passing
aspects covered: standard config-flow ordering, exit command.

## 身份

- Record ID: `PASS-WW-MODE-001-rerun`
- Finding present: `no` (for aspects listed below only)
- Status: `observed_pass` (partial — 3+ config stages not reached
  under driver's efficiency stop; see scenario-integrity caveat)
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

Transcript: `session.log` size=4696
sha256=`a3888b82ff588b558456a9db8e0674c7de60508d945f698c241953f1f998d5d2`.
Driver log: `driver.jsonl` size=7284
sha256=`e22bd4590f5f15351839a7e01dde951e7307a0b8d7f50290f9049f735f3a7235`.

1. **Standard config-flow ordering reproduces**
   - fake-node → ethereum → CLOUD_REGION Y → CLOUD_ZONE Y →
     MACHINE_TYPE Y → LEDGER_DEVICE 1 → DATA_VOL_TYPE pd-ssd →
     DATA_VOL_SIZE Y → DATA_VOL_MAX_IOPS 3000 → DATA_VOL_MAX_THROUGHPUT
     — same ordering as `WW-LANGUAGE-001` / `WW-RESUME-001` /
     `WW-MULTI-001` / `WW-MODE-001` = 4× reproducible ordering.

2. **`exit` command cleanly exits the REPL**
   - Turn 10 `exit` → `Agent> Exited AnyChain Benchmark Agent.` and
     process terminates.

3. **Response-driven discipline**
   - All 10 user turns produced by Vertex Sonnet 4.6 rawPredict against
     the immediately-preceding agent output.

## Scenario-integrity caveat (harness limitation)

**The manifest intent** for `WW-MODE-001` is:
> "Configure some fields (chain, mode, qps, observability), then try to
> switch a mode (e.g., from fake-node to real-node, or single to
> mixed). Confirm the agent invalidates dependent fields."

**What actually happened**: the config-flow requires ~14+ ordered
turns to reach the RPC-mode or observability prompt (chain → 8 disk /
network fields → RPC mode → workload defaults → benchmark mode → QPS
defaults → observability). My driver's SYS_TMPL efficiency prompt
encourages Sonnet to stop after 6-10 turns once the agent's behavior
is understood. Sonnet decided at turn 10 that the config-flow
mechanics were sufficiently exercised and typed `exit`, never
reaching the mode-switch trigger.

**Why this is not a product failure**:
- The agent correctly walks the config-flow one field at a time.
- No mode-switch attempt was made in this transcript, so no claim
  can be made about mode invalidation behavior in either direction.

**What Codex needs to do to properly exercise this scenario**:
1. Increase driver TURN_MAX for MODE-001 specifically to 25.
2. Adjust SYS_TMPL to instruct Sonnet: "for WW-MODE-001, do NOT stop
   until you have successfully triggered a mode switch mid-config".
3. Or: preseed a checkpoint with fake-node config already at
   RPC-mode / QPS / observability stage, then attempt a switch to
   real-node.

## Evidence bundle

`.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-MODE-001/`:

- `session.log` size=4696 sha256=`a3888b82ff588b558456a9db8e0674c7de60508d945f698c241953f1f998d5d2`
- `session-state.json` size=89 sha256=`43b0f0535ebc9c1a2841bf04d793ad494d0d7a97fc7f49acc8f30a63fcfcbc20`
- `langgraph-snapshot.tgz` size=627698 sha256=`4ae6cdee635c147219ba7f73464b7d97fb20843011b9ec7fedcd8aee6996bb9c`
- `driver.jsonl` size=7284 sha256=`e22bd4590f5f15351839a7e01dde951e7307a0b8d7f50290f9049f735f3a7235`

## Scenario-level acceptance

- exact_starting_checkpoint_or_fresh_state: ✓ fresh
- complete_response_driven_transcript: ✓ 10 turns
- raw_gemini_agent_outputs_and_tool_receipts: N/A (Claude substitution)
- exact_claude_user_turns: ✓ in `driver.jsonl`
- first_failing_turn_or_terminal_pass: terminal_pass on config-flow
  ordering; mode-switch behavior NOT exercised (harness caveat)
- runtime_event_and_checkpoint_hashes: ✓
- finding_id_or_explicit_observed_pass_result: ✓ partial pass; no
  product-failure EXT finding raised

## 安全传输

- Redaction review completed by: Claude Sonnet 4.6 (recorder)
- Bundle path: as above
- Internal `SHA256SUMS`: pending Codex verification
- Missing evidence: mode-switch invalidation branch not exercised
  (see caveat above)
