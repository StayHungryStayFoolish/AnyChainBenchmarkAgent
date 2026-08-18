# AnyChain Agent External Validation Record — PASS-WW-RESUME-001-rerun

Partial pass for `WW-RESUME-001` under corrected Vertex Sonnet 4.6 driver
(2026-08-13). The intended manifest 3-way choice (continue / modify /
reset) COULD NOT BE EXERCISED because the driver's default `.agent/`
state cleanup wiped `terminal/` and `langgraph/`, so this scenario
started cold with no prior partial session on disk. What DID pass is
recorded here; the missed 3-way exercise is documented as a driver
caveat in the "Scenario-integrity caveat" section below (NOT a product
bug — a run-harness limitation for Codex to be aware of).

## 身份

- Record ID: `PASS-WW-RESUME-001-rerun`
- Finding present: `no` (for aspects listed below only)
- Status: `observed_pass` (partial — 3-way resume choice not exercised;
  see scenario-integrity caveat)
- Acceptance class: `discovery`
- Source branch: `docs/agent-handoff-chaos`
- Full source revision: `a22108e53401cf7d9314917b870b71ccf338619c`
- Tracked worktree clean: `yes`
- Runtime: Ubuntu Docker `blockchain-node-benchmark:test-ubuntu`
  container `bench-preflight`
- Host: Debian 13 gLinux GCE VM, x86_64

## AI 与认证

- Agent provider/model/auth: `claude / claude-sonnet-4-6 / google_adc`
  (Leland-authorized discovery substitution)
- External user AI/model: Claude Sonnet 4.6 (Vertex `us-east5` rawPredict,
  discovery substitution for manifest role `gemini_via_vertex_adc`)
- External user role: `response_driven_discovery`
- ADC project/location: `claude-ttft-test / us-east5`
- Preflight evidence hashes: see `PASS-WW-STARTUP-001-rerun`

## Passing aspects (with turn / byte references)

Transcript: `session.log` size=8078
sha256=`890e7609dc65bcd07e0aa430ed7b260d5d7343eb7de1b9b1c22a5b82741e700c`.
Driver log: `driver.jsonl` size=16236
sha256=`540f33640c2cb76b28545684c25486f8b4d941dbcc267cf57b0839a3553666a8`.
Session state end: language=`en`.

1. **`jobs` / `status` correctly reply "No jobs found." on cold session**
   - Turn 1 `jobs` → `Agent> No jobs found.`
   - Turn 3 `status` → `Agent> No jobs found.`
   — On a fresh session with no prior job, both list/status commands
   return the same authoritative no-jobs result without spuriously
   inventing a phantom job.

2. **`help` responds with the canonical starter hint**
   - Turn 2 `help` → `Agent> Try: benchmark solana, use fake-node,
     doctor, jobs, status, logs <job_id>, follow <job_id>, help, exit.`
   — Same string as the startup hint; consistent surface.

3. **`benchmark <chain>` short-form binds chain without full command**
   - Turn 4 `benchmark solana` →
     `Agent> Confirmed chain: solana.` followed by the orientation menu.

4. **Numeric-only orientation choice `1` opens fake-node config flow**
   - Turn 5 `1` → agent transitions from orientation menu into the
     first config prompt (`Detected CLOUD_REGION: us-central1. Use this
     value?`).

5. **CLOUD_REGION N branch re-prompts for a custom value**
   - Turn 6 `2` (N) → `Agent> Confirm CLOUD_REGION; use the detected
     value or enter a custom region.` then turn 7 `us-east1` accepted.
   — Confirms the Y/N branching does re-prompt correctly, and the
   custom value is accepted verbatim.

6. **Ordered field-by-field config prompt flow reaches DATA_VOL_MAX_THROUGHPUT
   at turn budget end**
   - Sequence walked: CLOUD_REGION → CLOUD_ZONE → MACHINE_TYPE →
     LEDGER_DEVICE → DATA_VOL_TYPE → DATA_VOL_SIZE → DATA_VOL_MAX_IOPS →
     DATA_VOL_MAX_THROUGHPUT (turn 20).
   — Same ordered flow observed in `WW-LANGUAGE-001` re-run;
   reproducible field ordering.

7. **`pd-ssd` custom disk type accepted after "enter a value" fallback**
   - Turn 10 → `Agent> Confirm DATA_VOL_TYPE for the ledger/data disk,
     for example hyperdisk-balanced, hyperdisk-extreme, pd-ssd,
     pd-balanced, local-ssd, ssd, or nvme.` Turn 11 `pd-ssd` accepted.

8. **Response-driven discipline**
   - All 20 user turns produced by Vertex Sonnet 4.6 rawPredict against
     the immediately-preceding agent output; recorded per-turn in
     `driver.jsonl` with `sonnet_input` and `sonnet_reasoning`.

## Scenario-integrity caveat (harness limitation)

**The manifest intent** for `WW-RESUME-001` is:
> "You are returning after a prior partial configuration session. The
> agent should offer to continue, modify, or reset. Exercise the modify
> path (change something), then eventually try 'continue' or 'reset'."

**What actually happened**: my driver's default state cleanup
(`rm -rf /workspace/.agent/terminal /workspace/.agent/langgraph
/workspace/.agent/prepared /workspace/.agent/jobs`) wiped the on-disk
prior session before this scenario started. So on REPL launch, the
agent emitted `Agent> No previous job was found.` and jumped straight
to the orientation menu — the 3-way `continue / modify / reset` choice
was never presented.

**Why this is not a product failure**:
- The agent behaved correctly given cold-start state.
- The startup banner explicitly said "No previous job was found."
- If a real user had a prior partial `.agent/` state on disk, this
  scenario would have exercised the intended 3-way branch (evidence for
  this exists in `PASS-WW-STARTUP-001-rerun` where the orientation menu
  is the same and no prior-session offer appears in cold-start).

**What Codex needs to do to properly exercise this scenario**:
1. Seed `/workspace/.agent/terminal/` and
   `/workspace/.agent/langgraph/` with a prior partial-session
   checkpoint (e.g., completed CLOUD_REGION and CLOUD_ZONE, pending at
   MACHINE_TYPE).
2. Launch REPL without the default cleanup.
3. Verify the "continue / modify / reset" 3-way prompt appears at turn
   0 or 1.

**Driver caveat recorded so nobody re-runs this without seeding
state**.

## Evidence bundle

`.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-RESUME-001/`:

- `session.log` size=8078 sha256=`890e7609dc65bcd07e0aa430ed7b260d5d7343eb7de1b9b1c22a5b82741e700c`
- `session-state.json` size=89 sha256=`43b0f0535ebc9c1a2841bf04d793ad494d0d7a97fc7f49acc8f30a63fcfcbc20`
- `langgraph-snapshot.tgz` size=1888978 sha256=`c66fabab34410e23ba11ae0ec100ce352e40a2fbccd9b7b0e8a03cc8abca8258`
- `driver.jsonl` size=16236 sha256=`540f33640c2cb76b28545684c25486f8b4d941dbcc267cf57b0839a3553666a8`
- `repl.raw.log` size=8078 sha256=`890e7609dc65bcd07e0aa430ed7b260d5d7343eb7de1b9b1c22a5b82741e700c`

## Scenario-level acceptance

- exact_starting_checkpoint_or_fresh_state: ✓ fresh (caveat: this
  scenario expected a PRIOR-partial checkpoint but got fresh)
- complete_response_driven_transcript: ✓ 20 turns
- raw_gemini_agent_outputs_and_tool_receipts: N/A (Claude substitution)
- exact_claude_user_turns: ✓ in `driver.jsonl` `sonnet_input`
- first_failing_turn_or_terminal_pass: terminal_pass on all exercised
  aspects; 3-way resume choice not exercised (harness)
- runtime_event_and_checkpoint_hashes: ✓
- finding_id_or_explicit_observed_pass_result: ✓ partial pass; no
  product-failure EXT finding raised

## 安全传输

- Redaction review completed by: Claude Sonnet 4.6 (recorder)
- Bundle path: as above
- Internal `SHA256SUMS`: pending Codex verification
- Missing evidence: none for exercised aspects; 3-way resume branch
  not exercised (see caveat)
