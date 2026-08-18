# AnyChain Agent External Validation Record — PASS-WW-ANALYSIS-001-rerun

Partial pass for `WW-ANALYSIS-001` (log/report/CSV analysis pathway).
The scenario's core intent (invoke `analyze latest job`, `logs <id>`,
or `jobs` on an existing job) was NOT exercised — Sonnet re-walked
the full config flow instead of routing to analysis commands.
Documented as scenario-integrity caveat (harness / driver-prompt
limitation). One observed anomaly at turn 20 (last preflight/smoke
submission returned "The final state of this turn's external
operation is uncertain") could indicate a race condition or a real
retry-recovery message — documented in "Anomaly" section but not
elevated to a full EXT finding pending Codex reproduction on cleaner
state.

## 身份

- Record ID: `PASS-WW-ANALYSIS-001-rerun`
- Finding present: `no` (for aspects listed below only)
- Status: `observed_pass` (partial — manifest analysis path not
  exercised; anomaly observed on final submission)
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

Transcript: `session.log` size=8453
sha256=`fe378e909aefef9d0ecafca8c9adac5ffc876aafd7e17271163793defa786e45`.
Driver log: `driver.jsonl` size=14212
sha256=`fd1ecfa24d96c1b6ce22151d3d9852c2a2ee50ed27564c8901a6d88be43f84f3`.

1. **Full-config flow reproducibility**
   - Same 19-turn ordered flow as `WW-EXEC-001` (fake-node → ethereum
     → CLOUD_REGION Y → ... → advanced thresholds Y → Configuration
     collected). Reproduces deterministically across scenarios.

2. **Response-driven discipline**
   - All 20 user turns produced by Vertex Sonnet 4.6 rawPredict
     against the immediately-preceding agent output.

## Anomaly (not elevated to EXT — pending Codex reproduction)

At turn 20 (final `Y` to "Configuration is collected. Run preflight
and smoke?"), the agent responded:
```
Agent> The final state of this turn's external operation is uncertain.
The Agent stopped without advancing the session configuration.
Reconcile the latest job/execution record before submitting the same
operation again.
```

This differs from `WW-EXEC-001` which returned:
```
Agent> Fake-node smoke submitted: status=ok,
job_id=job_20260813103335_267777e4. ...
```

The only externally visible difference between the two scenarios is
timing: `WW-ANALYSIS-001` started immediately after `WW-EXEC-001`
completed (in the same container, with fresh `.agent/` cleanup by the
driver). The "uncertain terminal state" message suggests the smoke
executor detected a lingering job / state artifact from the previous
scenario despite the driver's cleanup — OR — a genuine race with the
prior job's subprocess. Not enough evidence to raise a product bug;
recording for Codex to reproduce on a container that has not just
run `WW-EXEC-001`.

## Scenario-integrity caveat (harness limitation)

**The manifest intent** for `WW-ANALYSIS-001` is:
> "If a prior job exists (or start a fake-node one), ask the agent to
> analyze logs/reports/CSV. Try 'analyze latest job' or 'logs <id>'.
> Stop when you have exercised the analysis path."

**What actually happened**: Sonnet did not have visibility into the
`WW-EXEC-001` job that was submitted immediately prior (fresh `.agent/`
cleanup wipes the local job registry between scenarios). Sonnet
therefore started a new config flow, exhausted its turn budget
before submission completed, and never issued any analysis command.

**What Codex needs to do to properly exercise this scenario**:
1. Do NOT clean the `.agent/jobs/` and `.agent/terminal/` between
   `WW-EXEC-001` and `WW-ANALYSIS-001` — preserve the prior job so
   ANALYSIS can invoke `jobs`, `analyze latest job`, `logs <job_id>`,
   `follow <job_id>` on it.
2. Or seed a synthetic prior job fixture in `.agent/jobs/` and skip
   the config flow entirely, starting Sonnet with the goal "run
   analyze latest job".

## Evidence bundle

`.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-ANALYSIS-001/`:

- `session.log` size=8453 sha256=`fe378e909aefef9d0ecafca8c9adac5ffc876aafd7e17271163793defa786e45`
- `session-state.json` size=89 sha256=`43b0f0535ebc9c1a2841bf04d793ad494d0d7a97fc7f49acc8f30a63fcfcbc20`
- `langgraph-snapshot.tgz` size=2086053 sha256=`8b119f88bdf38e6280985e8299b943ca50edbf421538d578576450876d4971f9`
- `driver.jsonl` size=14212 sha256=`fd1ecfa24d96c1b6ce22151d3d9852c2a2ee50ed27564c8901a6d88be43f84f3`

## Scenario-level acceptance

- exact_starting_checkpoint_or_fresh_state: ✓ fresh (caveat: this
  scenario expected a PRIOR job to analyze; got cleanly wiped state)
- complete_response_driven_transcript: ✓ 20 turns
- raw_gemini_agent_outputs_and_tool_receipts: N/A (Claude substitution)
- exact_claude_user_turns: ✓ in `driver.jsonl`
- first_failing_turn_or_terminal_pass: terminal_pass on config-flow
  reproducibility; analysis path NOT exercised (harness caveat);
  anomaly at turn 20 recorded above
- runtime_event_and_checkpoint_hashes: ✓
- finding_id_or_explicit_observed_pass_result: ✓ partial pass; no
  product-failure EXT finding raised

## 安全传输

- Redaction review completed by: Claude Sonnet 4.6 (recorder)
- Bundle path: as above
- Internal `SHA256SUMS`: pending Codex verification
- Missing evidence: analysis commands (`analyze`, `logs`, `follow`,
  `jobs`) not exercised (see caveat above)
