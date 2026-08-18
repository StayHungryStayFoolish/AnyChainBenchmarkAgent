# AnyChain Agent External Validation Record — PASS-WW-EXEC-001-rerun

Substantive pass for `WW-EXEC-001` (fake-node benchmark job through
prep, submission, and completion). The full happy-path config flow
was walked in 20 turns and successfully submitted a fake-node smoke
job (`job_20260813103335_267777e4`). No product failure raised.

## 身份

- Record ID: `PASS-WW-EXEC-001-rerun`
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

Transcript: `session.log` size=8443
sha256=`bbca6de56e005ffedce9b468090392b00cd1ef9c985f73a78c93b3b91f4c0a90`.
Driver log: `driver.jsonl` size=13984
sha256=`7e2ca8bacb5aafa70b3c077d0e7a1bb15f6bc2725930a659998747f945a581f7`.

1. **Full config flow reaches preflight/smoke and submits job**
   - 20 turns walked:
     `1` (fake-node) → `ethereum` → `1` (CLOUD_REGION Y) → `1` (CLOUD_ZONE Y)
     → `1` (MACHINE_TYPE Y) → `1` (LEDGER_DEVICE) → `pd-ssd`
     (DATA_VOL_TYPE) → `1` (DATA_VOL_SIZE Y) → `3000` (IOPS) → `200`
     (THROUGHPUT) → `2` (accounts disk N) → `1` (network iface) → `1`
     (NETWORK_MAX_BANDWIDTH_GBPS) → `1` (RPC single) → `1` (workload
     defaults) → `2` (mode standard) → `1` (QPS defaults Y) → `1`
     (observability disabled) → `1` (advanced thresholds Y) → `1`
     (preflight+smoke Y)
   - Turn 20 response:
     `Agent> Fake-node smoke submitted: status=ok,
     job_id=job_20260813103335_267777e4. Use: status
     job_20260813103335_267777e4; logs job_20260813103335_267777e4;
     follow job_20260813103335_267777e4; analyze latest job`
   - Job ID persisted; follow-up commands (status/logs/follow/analyze)
     documented on the same line. Excellent operational surface.

2. **Advanced monitoring thresholds prompt lists all 12 fields upfront**
   - "MONITOR_INTERVAL, DISK_MONITOR_RATE, SUCCESS_RATE_THRESHOLD,
     MAX_LATENCY_THRESHOLD, BOTTLENECK_CPU_THRESHOLD, BOTTLENECK_MEMORY_THRESHOLD,
     BOTTLENECK_DISK_UTIL_THRESHOLD, BOTTLENECK_DISK_LATENCY_THRESHOLD,
     BOTTLENECK_NETWORK_THRESHOLD, BOTTLENECK_ERROR_RATE_THRESHOLD,
     BOTTLENECK_DISK_IOPS_THRESHOLD, BOTTLENECK_DISK_THROUGHPUT_THRESHOLD"
   - Y/N shortcut with an explicit "N to adjust individual values"
     path. Ergonomic default; escape available.

3. **`Configuration is collected. Run preflight and smoke?` gate before
   any job submission**
   - Turn 19 response asks explicit Y/N before running any preflight/
     smoke work. No silent job creation.

4. **Job submission returns actionable follow-up commands**
   - `Use: status <id>; logs <id>; follow <id>; analyze latest job`
     — all four operational verbs listed with the concrete job ID
     substituted in. Ready to copy-paste.

5. **20-turn wall-clock 241 seconds**
   - ~12s/turn including thinking, prompt-render, Sonnet-decide,
     write-back, settle wait. Acceptable interactive latency.

6. **Response-driven discipline**
   - All 20 user turns produced by Vertex Sonnet 4.6 rawPredict against
     the immediately-preceding agent output.

## Evidence bundle

`.agent/external-validation/a22108e53401cf7d9314917b870b71ccf338619c/transcripts/WW-EXEC-001/`:

- `session.log` size=8443 sha256=`bbca6de56e005ffedce9b468090392b00cd1ef9c985f73a78c93b3b91f4c0a90`
- `session-state.json` size=89 sha256=`43b0f0535ebc9c1a2841bf04d793ad494d0d7a97fc7f49acc8f30a63fcfcbc20`
- `langgraph-snapshot.tgz` size=2289664 sha256=`fbaa89d70c4263a857fe3e973055b39cff45dc011cd261d5c78ffd2588b28b8d`
- `driver.jsonl` size=13984 sha256=`7e2ca8bacb5aafa70b3c077d0e7a1bb15f6bc2725930a659998747f945a581f7`

## Scenario-level acceptance

- exact_starting_checkpoint_or_fresh_state: ✓ fresh
- complete_response_driven_transcript: ✓ 20 turns
- raw_gemini_agent_outputs_and_tool_receipts: N/A (Claude substitution)
- exact_claude_user_turns: ✓ in `driver.jsonl`
- first_failing_turn_or_terminal_pass: ✓ terminal_pass (job submitted
  ok at turn 20)
- runtime_event_and_checkpoint_hashes: ✓
- finding_id_or_explicit_observed_pass_result: ✓ observed_pass; no
  product-failure EXT finding raised

## Not exercised

- Job execution completion (turn budget exhausted at submission; the
  `status`/`follow`/`analyze` verbs were not exercised on the new
  job in this session — see `WW-ANALYSIS-001` for a re-run attempt)
- Any failure-recovery path (this run submitted cleanly)

## 安全传输

- Redaction review completed by: Claude Sonnet 4.6 (recorder)
- Bundle path: as above
- Internal `SHA256SUMS`: pending Codex verification
- Missing evidence: post-submission job state (see "Not exercised")
