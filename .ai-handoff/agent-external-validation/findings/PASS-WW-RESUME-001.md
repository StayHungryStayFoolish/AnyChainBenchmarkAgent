# AnyChain Agent External Validation Record — PASS-WW-RESUME-001 (partial)

Scenario `WW-RESUME-001` (fresh_partial_complete_session_continue_modify_
reset) partial-pass note. Resume detection on top of prior `.agent/`
state, chain-preservation across REPL restart, modify-menu structure,
and cancel dispatch all behave per contract. Two related failures
observed in same session recorded as EXT-20260808-005 (malformed cancel
label "2. N") and EXT-20260808-006 (novel-chain classification 80s
latency).

## 身份

- Record ID: `PASS-WW-RESUME-001`
- Finding present: `no` (for aspects listed below only)
- Status: `observed_pass` (partial)
- Acceptance class: `discovery`
- Source branch: `docs/agent-handoff-chaos`
- Full source revision: `a22108e53401cf7d9314917b870b71ccf338619c`
- Runtime: Ubuntu 24.04 Docker `blockchain-node-benchmark:test-ubuntu`

## AI 与认证

- Agent provider/model/auth: `claude / claude-sonnet-4-6 / google_adc`
  (Leland-authorized discovery substitution)
- External user AI/model: Claude Sonnet 4.6
- External user role: `response_driven_discovery`
- ADC project/location: `claude-ttft-test / us-east5`

## Passing aspects (with transcript byte ranges)

Transcript source: `.agent/external-validation/a22108e5.../transcripts/
WW-RESUME-001/session.log` sha256=`4a7ae93d1a77d3b99c4f5cef8938a4ca5d
13a943b2594c7834fc0be35ee14303` (5691 bytes, 6 turns).

1. **Resume detection at startup on prior `.agent/` state**
   - Agent recognized prior session — line: "检测到之前的 Agent 配置
     会话："
   - Full state dump: `target_mode: <not selected>`, `workflow_mode:
     <not selected>`, `chain: solana`, `rpc_mode: <not selected>`,
     `qps: <not selected>`, `observability: <not selected>`,
     `已确认字段：BLOCKCHAIN_NODE`, `延后请求数量：0`, `已保存
     workflow 目标：<none>`
   - Chain=solana correctly restored from WW-STARTUP-001 turn 5 which
     set it via `use solana`
   - 3-option resume menu emitted: 继续 / 修改 / 重置

2. **Session locale honored at startup**
   - Session-state.json had `language: "zh"` — startup output
     rendered entirely in Chinese including model config lines
     ("当前模型配置：..."), diagnostics ("启动检查完成：..."),
     framework facts ("已加载框架事实：..."). Contrast with
     EXT-20260808-003 where per-turn response locale was NOT honored;
     the startup path is a different code path that DOES read
     session locale.

3. **Modify menu structure (turn 1 "2")**
   - 8-group modify menu emitted with per-group field enumeration:
     `1. target_mode — 可配置字段：target_mode, workflow_mode,
     use_fake_node`, ..., `2. chain_identity — 可配置字段:
     BLOCKCHAIN_NODE, chain_identity, secondary_handoff`, ...,
     through 8 groups. Rich, contract-conformant menu.
   - Numeric-option dispatch worked correctly for this menu — user
     input "2" reached the chain_identity flow. Contrast with
     EXT-20260808-004 (orientation menu option "4" numeric dispatch
     was broken); confirms the bug in EXT-20260808-004 is specific
     to the orientation menu, not a general numeric-dispatch failure.

4. **Chain-switch prompt (turn 2 "2")**
   - Agent emitted correct prompt: "请输入要切换到的链名。当前链
     `solana` 会保留到新链被确认。请直接输入值。"
   - User is informed that current chain is preserved during the
     switch attempt — good state-transition UX

5. **Cancel dispatch (turn 4 "2")**
   - Despite the malformed label ("2. N", per EXT-20260808-005), the
     underlying dispatch is correct: pressing "2" cancels and
     returns to orientation menu, preserving chain=solana

6. **State query (turn 5 "当前绑定的链是什么？")**
   - Agent returned structured state dump including "chain: solana",
     "已确认字段：BLOCKCHAIN_NODE", "待确认问题：opening_next_action",
     "待确认字段：target_mode"
   - This confirms chain state is queryable via natural language when
     asked directly (contrast with EXT-20260808-002 which is
     specific to workload-render placeholder, not state queries)

7. **Clean exit (turn 6 "exit")**
   - Exit response: "已退出 AnyChain Benchmark Agent。"
   - Locale-appropriate; session-state.json persisted with final
     values

## Non-passing aspects (see separate records)

- Novel-chain confirmation dialog cancel label malformed →
  **EXT-20260808-005**
- Novel-chain classification latency (80s silent wait) →
  **EXT-20260808-006**

## Evidence bundle

`.agent/external-validation/a22108e5.../transcripts/WW-RESUME-001/`:

- `session.log` sha256=`4a7ae93d1a77d3b99c4f5cef8938a4ca5d13a943b2594c
  7834fc0be35ee14303` (5691 bytes, 6-turn PTY transcript)
- `session-state.json` sha256=`efebb3ec3265a2bb45a3974f45d8ac33475b598
  895fac8f67ae236466d7f35dd` (89 bytes; unchanged from
  WW-STARTUP-001 — language "zh" persistent)
- `langgraph-snapshot.tgz` sha256=`d597f955bf95493054cc543775a60b8fd54
  defd8f764fc8f17326e23a8fc3ebd` (1025641 bytes; larger than
  WW-STARTUP-001 snapshot due to additional turn checkpoints)

## Scenario-level acceptance

Per manifest `required_per_scenario`:

- exact_starting_checkpoint_or_fresh_state: ✓ prior `.agent/` state
  preserved from WW-STARTUP-001; container not restarted
- complete_response_driven_transcript: ✓ session.log 6 turns
- raw_agent_outputs_and_tool_receipts: ✓ raw output in session.log
- exact_claude_user_turns: ✓ verbatim in session.log
- first_failing_turn_or_terminal_pass: ✓ turn 3 (label + latency)
  per findings
- runtime_event_and_checkpoint_hashes: ✓ langgraph-snapshot.tgz
- finding_id_or_explicit_observed_pass_result: ✓ two findings + this
  partial pass note
