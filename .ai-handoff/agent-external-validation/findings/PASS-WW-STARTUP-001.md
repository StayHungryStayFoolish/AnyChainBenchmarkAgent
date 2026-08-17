# AnyChain Agent External Validation Record — PASS-WW-STARTUP-001 (partial)

Scenario `WW-STARTUP-001` (adc_readiness_cli_startup_language_and_orientation)
partial-pass note. Startup diagnostics, bilingual orientation, chain
enumeration, chain binding via `use <chain>`, and locale-appropriate exit
handling all behave per contract. Three related failures observed on
adjacent aspects are recorded separately as EXT-20260808-002 (chain
auto-bind), EXT-20260808-003 (session-locale persistence bypass), and
EXT-20260808-004 (menu option 4 shallow). Scenario overall status
`observed_failure` due to those three; this record documents the aspects
that DID pass so Codex has both sides of the boundary.

## 身份

- Record ID: `PASS-WW-STARTUP-001`
- Finding present: `no` (for aspects listed below only)
- Status: `observed_pass` (partial — see linked failure records for
  aspects that did NOT pass in the same scenario)
- Acceptance class: `discovery`
- Source branch: `docs/agent-handoff-chaos`
- Full source revision: `a22108e53401cf7d9314917b870b71ccf338619c`
- Runtime: Ubuntu 24.04 in Docker image `blockchain-node-benchmark:test-ubuntu`

## AI 与认证

- Agent provider/model/auth: `claude / claude-sonnet-4-6 / google_adc`
- External user AI/model: Claude Sonnet 4.6
- External user role: `response_driven_discovery`
- ADC project/location: `claude-ttft-test / us-east5`
- Startup evidence sha256s: provider-status `8616d458...`, doctor
  `780b0628...`, llm-smoke `c6adcc50...`, adk-status `6e4b77c8...`

## Passing aspects (with transcript byte ranges)

Transcript source: `.agent/external-validation/a22108e5.../transcripts/
WW-STARTUP-001/session.log` sha256=`2c678e8058617dba798525e2b9e02efa2f62c62
93855b3a405d630f791c56bc2`.

1. **Startup diagnostics complete cleanly (L1-L27)**
   - Model config announced: `provider=claude, model=claude-sonnet-4-6,
     auth=google_adc` (L3)
   - ADK importability confirmed (L5)
   - Framework facts loaded: 36 chains / 6 adapter families / 109 RPC
     methods / 207 fake-node fixtures (L6)
   - Startup diagnostics `status=ready, cloud=gcp, deployment=container,
     missing dependencies=<none>` (L8)
   - Environment inference produced structured GCE draft (L9-L27) — no
     panic, no missing-dep, no exception
   - "No previous job was found." (L27) — clean cold-start acknowledged

2. **Bilingual orientation menu (L28-L34 English, L50-L58 Chinese
   translation after zh input)**
   - Initial menu in English lists 4 options with contract summaries
     (L28-L33) and meta-command hint (L34)
   - After first Chinese input (turn 1), agent re-emits the same menu
     translated to Chinese (L54-L58) with parallel structure — proves the
     menu-rendering path has both locales wired

3. **Free-form chain-list enumeration (L77-L79)**
   - Query "请列出支持的所有链名称" returns full 36-chain enumeration:
     `acala, algorand, aptos, arbitrum, astar, avalanche-c, avalanche-x,
     base, bch, bitcoin, bsc, cardano, celestia, cosmos-hub, dogecoin,
     ethereum, hedera, injective, kusama, linea, litecoin, moonbeam,
     near, optimism, osmosis, polkadot, polygon, scroll, sei, solana,
     starknet, sui, tezos, ton, tron, zksync-era`
   - Response includes correct next-step hint: "你可以指定一条链，查看它
     的默认 RPC workload。"

4. **`use <chain>` binding + confirmation (L104-L112)**
   - Input `use solana` → `Confirmed chain: solana.` (L106)
   - Followed by orientation menu re-emission (L107-L112) — proves chain
     binding is followed by a stable navigation prompt

5. **Post-binding workload template render (L120-L129)**
   - Query "现在告诉我 solana 的默认 RPC workload" (with chain now
     bound) → real workload template: "链 `solana` 的模板 workload：
     single 默认 method 为 `getAccountInfo`；mixed 默认权重为
     getAccountInfo=20, getBalance=20, getTokenAccountBalance=20,
     getLatestBlockhash=20, getBlockHeight=20."
   - Values match a plausible default single/mixed split; no placeholder
     tokens

6. **Locale-appropriate exit (L143-L144)**
   - Input `exit` (English) → response "已退出 AnyChain Benchmark
     Agent。" (Chinese)
   - Notable: this is the ONE case where session-locale persistence
     appears to be honored — see EXT-20260808-003 which flags that
     normal (non-exit) responses do NOT honor it. This inconsistency
     is documented in EXT-20260808-003's Supporting evidence section
     as narrowing the bug scope.

7. **Response-driven turn discipline**
   - Every user turn was sent only after the prior agent turn
     completed with a `User>` prompt appearing at the tail of
     `/tmp/ac-out`. Manifest rule
     `next_user_turn_must_follow_real_agent_response: true` satisfied.

## Non-passing aspects (see separate records)

- Chain auto-bind from natural-language query → **EXT-20260808-002**
- Session-locale persistence read at response time → **EXT-20260808-003**
- Menu option 4 drill-down dispatch → **EXT-20260808-004**

## Evidence bundle

`.agent/external-validation/a22108e5.../transcripts/WW-STARTUP-001/`:

- `session.log` sha256=`2c678e8058617dba798525e2b9e02efa2f62c6293855b3a
  405d630f791c56bc2` (7043 bytes, full 7-turn PTY transcript)
- `session-state.json` sha256=`efebb3ec3265a2bb45a3974f45d8ac33475b59889
  5fac8f67ae236466d7f35dd` (89 bytes, final session state)
- `langgraph-snapshot.tgz` sha256=`2fb3dde546eea778584568a622d8833345f0d
  15ea8987ba5c29f5cdf696d17f5` (535293 bytes, langgraph checkpointer
  state)

Startup bundle (shared across scenarios on this container):
`.agent/external-validation/a22108e5.../` (revision.txt, environment.txt,
provider-status.json, doctor.json, llm-smoke.json, adk-status.json,
anychain-agent-startup.txt, evidence/ — see EXT-20260808-001 for those
hashes).

## Scenario-level acceptance

Per manifest `required_per_scenario`:

- exact_starting_checkpoint_or_fresh_state: ✓ fresh state, container cold
  start
- complete_response_driven_transcript: ✓ session.log 7 turns
- raw_gemini_agent_outputs_and_tool_receipts: ✓ raw agent outputs in
  session.log; note this run used Claude (Leland-authorized substitution),
  so no Gemini outputs — closure of formal Codex evidence requires
  Gemini rerun per manifest rule `model_substitution_closes_formal_codex
  _evidence: false`
- exact_claude_user_turns: ✓ verbatim in session.log
- first_failing_turn_or_terminal_pass: ✓ turn 2 (option 4), turn 4
  (auto-bind), turn 5 (locale) — recorded per finding
- runtime_event_and_checkpoint_hashes: ✓ langgraph-snapshot.tgz hashed
- finding_id_or_explicit_observed_pass_result: ✓ three findings +
  this partial pass note
