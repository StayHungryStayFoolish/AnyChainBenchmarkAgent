# AnyChain Agent External AI Handoff And Repair Spec

Status: handoff document for another AI coding agent.  
Git policy: do not commit this file unless the user explicitly asks.  
Runtime target: Linux/Docker only. macOS is not a supported runtime target.  
Primary goal: repair AnyChain Benchmark Agent at the Harness/workflow level, not by transcript patches.

## Why This Document Exists

Another AI may need to continue the Agent repair without access to the long
conversation history. This document is the handoff entry point. It must be read
before any code changes.

The current Agent has been refactored toward a LangGraph Harness, but real
terminal conversations still expose product-level failures. Unit tests are not
enough. The next AI must reproduce real CLI behavior in Docker with DeepSeek as
the Agent model and Codex/another AI acting as a chaotic user.

## Paste This To The Next AI

Use this exact instruction when handing the task to another AI:

```text
You are taking over repair of AnyChain Benchmark Agent. Do not edit code first.
Read `.agent/task-docs/2026-07-10-agent-handoff-for-external-ai.md`,
`AI_CODING_GUIDE.md`, `AGENTS.md`, and
`.agent/task-docs/2026-07-07-langgraph-agent-harness-refactor.md`.

Your job is to repair the Agent Harness/workflow architecture so real Docker
CLI conversations pass, especially clearing config then asking about startup
environment inference, unknown-chain Case2, custom RPC consultation, group
jumps, pasted config/logs, and previous-session handling.

Before coding:
1. Run `git status --short`.
2. Audit the Agent files named in the handoff document.
3. Write a short implementation plan mapping each known failure to owning
   components and tests.
4. Confirm you will not use terminal keyword patches, fuzzy chain coercion, or
   a second workflow brain.

After coding:
1. Run the Docker unittest command from the handoff document.
2. Run real Docker CLI chaos with DeepSeek as the Agent model.
3. Provide the transcripts, issues found, fixes, and retest results.
```

## Mandatory Gates

Before editing product code, read and apply:

- `AI_CODING_GUIDE.md`
- `AGENTS.md`
- `docs/en/anychain-agent-ai-work-gate.md`
- `.agent/task-docs/2026-07-07-langgraph-agent-harness-refactor.md`
- this file

Hard rules:

1. Do not fix one transcript with local keyword patches.
2. Do not add a second workflow brain in terminal, prompt, callback, sanitizer,
   test harness, or runner code.
3. Do not use terminal code to decide benchmark business flow.
4. Do not use fuzzy matching to silently coerce chain names.
5. Do not let the model mutate state directly outside typed Harness actions.
6. Do not preserve obsolete code because it still imports. Remove or migrate it.
7. Do not claim product quality from unit tests alone. Real Docker CLI chaos is
   required.
8. If a new failure exposes a missing design rule, update this document or the
   current authoritative task document first, then code.

## Current Architecture Map

Key files:

- `agent/harness/graph.py`
  - LangGraph runtime, checkpoint path, session id, session purpose.
  - Must be the product workflow runtime owner.
- `agent/harness/groups.py`
  - Main group workflow implementation.
  - Owns pending questions, group transitions, config application, custom RPC,
    new-chain Case1/2/3 flow, evidence/report routing, and fallback path.
- `agent/harness/intent.py`
  - LLM action queue resolver.
  - Should classify ambiguous user turns into typed actions only.
  - Must not invent state or directly ask product questions.
- `agent/harness/oracle.py`
  - Product-facing next-action and context explanation.
  - Should explain current state without leaking group ids.
- `agent/harness/state.py`
  - State schema defaults and session metadata.
- `agent/harness/turns.py`
  - Turn adjudication: empty input, evidence continuation, normal user turn.
- `agent/terminal/repl.py`
  - Terminal I/O, startup banner, prompt-toolkit, Ctrl+C, simple shell commands.
  - Must not own benchmark workflow decisions.
- `agent/terminal/language.py`
  - Language detection and terminal-facing language behavior.
- `agent/validators/endpoint_probe.py`
  - Endpoint validation evidence. Must preserve runtime session metadata.
- `agent/workflows/group_registry.py`
  - Group metadata, if still used by docs/tests. Must not become a second
    runtime.

Tests:

- `tests/test_agent_langgraph_harness.py`
- `tests/test_agent_product_terminal.py`
- `tests/agent_live/run_langgraph_cli_matrix.py`
- `tests/agent_live/README.md`

## Product State Model That Must Be Enforced

The Agent must distinguish these state domains:

1. `startup_discovery`
   - Read-only environment inference produced at Agent startup.
   - Includes CPU, memory, cloud/deployment guess, disk candidates, network
     candidates, dependency status, latest job facts.
   - Clearing user configuration must not delete this.
2. `confirmed_config`
   - User-confirmed benchmark configuration.
   - Cleared by "clear previous config / start over".
3. `session_config`
   - Saved in-progress Agent workflow state from a previous session.
   - Must be offered on startup with continue / modify / clear.
4. `latest_job`
   - Historical run result and report/log artifacts.
   - Not the same as current config.
5. `pending_question`
   - Exactly one current user-facing question.
   - User can answer it, ask what it means, or jump to another group.
6. `evidence_buffer` / `evidence_collection`
   - Logs, errors, report snippets, request/response samples, docs pasted by
     user.
   - Must not accidentally trigger fallback config questions.

## Workflow Model

The workflow is not a fixed linear wizard. It is:

```text
LLM intent/action queue
  -> deterministic group workflow
  -> typed state update
  -> validator/oracle computes next blocking group
  -> one user-facing question or one user-facing answer
```

Default fallback order when the user gives no explicit jump:

1. opening / target mode
2. chain identity
3. provider/deployment metadata
4. ledger disk
5. accounts disk
6. network
7. endpoint/process
8. workload/RPC
9. target samples / fixtures
10. QPS profile
11. sync-observe specifics, when workflow mode is sync-observe
12. observability
13. preflight/smoke
14. job monitoring / report analysis

Users may jump at any time:

- ask a concept question
- change chain
- change target mode
- change QPS profile
- change RPC mode
- add custom RPC method
- switch Case1/2/3 chain path
- paste logs/errors
- paste YAML/JSON/env/config snippets
- go back
- reset

After the jump is resolved, the Harness must recompute the next missing group
from state. It must not blindly resume the previous prompt if the state has
changed.

## Chain And RPC Method Cases

These cases are global. They may be triggered from any group.

### Case1: Supported Chain With Custom RPC Method

When the chain is one of the configured chains or known aliases, but the user
wants custom RPC methods:

Required flow:

1. Keep the original chain template unchanged.
2. Ask for or extract:
   - method name
   - params sample
   - response sample or official docs
   - reachable validation endpoint
3. Validate endpoint and method behavior.
4. Ask how to apply the custom method:
   - replace single workload
   - replace mixed workload with only custom methods
   - add to existing mixed workload
5. Confirm weights. Total enabled mixed weights must be 100.
6. Generate job-local/runtime workload override only.
7. Do not mutate `config/chains/*.json`.
8. Continue fallback path after validation.

Important:

- The endpoint used for method validation is not automatically the final
  `LOCAL_RPC_URL`.
- User-provided request/response may be incomplete or inconsistent. Ask for
  missing pieces and validate what can be validated.

### Case2: New Chain In Existing Adapter Family

When the chain is not one of the 36 configured chains or aliases, but belongs
to an existing adapter family:

Required flow:

1. First, identify whether the chain exists.
   - Use LLM knowledge.
   - If Gemini + google_search is available, search official docs too.
   - If web research is unavailable, say so and ask the user for official docs
     or endpoint evidence.
2. Confirm canonical chain name with the user.
3. Confirm adapter family:
   - jsonrpc/EVM
   - substrate
   - rest
   - tendermint
   - bitcoin_jsonrpc
   - hedera_dual
4. Explain that fake-node is not automatically available for this new chain.
   It needs runtime template/fixture validation first.
5. Ask for a reachable endpoint and at least one RPC method with request/params
   and response sample or official docs.
6. Probe endpoint and method.
7. Generate runtime template/fixtures only after validation evidence exists.
8. Run smoke if the runtime template/fixture path is ready.
9. Continue normal fallback path after successful validation.

Bad behavior to avoid:

- Asking only for endpoint without explaining method/request/response/docs.
- Pretending a new chain can use fake-node before fixtures/runtime template
  validation.
- Silently treating an unknown typo as a known chain.

### Case3: New Chain Outside Supported Families

When the chain does not belong to supported adapter families:

Required flow:

1. Gather official docs, protocol/RPC examples, endpoint docs, and user notes.
2. If Gemini + google_search is available, search official docs.
3. If web research is unavailable, ask user to provide docs.
4. Generate a secondary-development handoff document.
5. Do not claim benchmark support.
6. Do not continue to smoke/preflight as if support exists.

## Known Failures From Real CLI Transcripts

### Failure A: Clear Config Then Ask About Startup Inference

Transcript:

```text
User> 3
Agent> 已清空之前的 Agent 配置会话...
User> 还有配置么？
Agent> 当前没有待确认问题。下一步需要继续确认入口选择...
User> 我现在关于最初推断的信息还有么？
Agent> 当前状态：chain <未选择>...
User> 你有一个环境依赖的检测脚本，这个脚本会帮我推断一些变量，这些变量推断了么
Agent> 当前没有待确认问题。下一步需要继续确认入口选择...
```

Problems:

1. Agent did not distinguish cleared confirmed config from startup discovery.
2. Agent did not answer whether environment inference exists.
3. Agent repeated fallback path instead of answering the user.

Required behavior:

- "已确认配置已经清空。启动时的只读环境推断仍然存在，可作为后续配置参考。"
- List inferred values:
  - CLOUD_PROVIDER
  - deployment
  - CPU
  - Memory
  - disk candidates
  - network candidates
  - dependency status
- List not inferred / needs confirmation:
  - CLOUD_REGION
  - CLOUD_ZONE
  - MACHINE_TYPE
  - LEDGER_DEVICE
  - volume type/iops/throughput
  - accounts disk existence
  - target mode, chain, RPC workload, QPS, observability

Likely owning components:

- `agent/harness/intent.py`
  - Add or improve action topic for startup discovery / environment inference.
- `agent/harness/groups.py`
  - Route that action to a product answer.
- `agent/harness/oracle.py`
  - Add formatter for startup discovery summary.

Forbidden fix:

- Do not add terminal keyword handling for "推断".

### Failure B: "现在可以做什么" Gets Fallback Text Only

Transcript:

```text
User> 那现在可以做什么？
Agent> 当前没有待确认问题。下一步需要继续确认入口选择：选择测试模式...
```

Required behavior:

- Explain available high-level actions:
  - run fake-node closed-loop test
  - run real-node RPC benchmark
  - run sync-observe
  - analyze latest job/report/logs
  - inspect startup environment inference
  - explain supported chains/RPC methods
  - onboard custom RPC or new chain
- Then ask a concise next-step question.

Likely owning components:

- `agent/harness/intent.py`
  - Topic `agent_capabilities` / `opening_options`.
- `agent/harness/groups.py`
  - Opening consultation response.

### Failure C: Unknown Chain `sola` Case2 Prompt Incomplete

Transcript:

```text
User> fake-node
Agent> 你想测试哪条链？当前目标模式：fake-node。
User> sola
Agent> `sola` 不在当前模板中。模型认为你可能想输入 `solana`...
User> 3
Agent> 请确认该链属于哪个协议族。
User> 1
Agent> 请提供可访问的 RPC endpoint，用于验证该新链和 RPC method。
```

Problems:

1. After user says `sola` is a real chain, Agent did not explain the lack of
   google_search under DeepSeek.
2. Agent did not confirm canonical chain name.
3. Agent asked only endpoint, not method/request/response/docs.
4. Agent did not explain fake-node cannot be used for a new chain until runtime
   template/fixture validation exists.

Required behavior:

- Explain:
  - `sola` is not configured.
  - Current provider has no web research.
  - User must confirm whether `sola` is the canonical chain name or provide
    official docs.
  - If it is jsonrpc/EVM, provide endpoint plus at least one method
    request/params/response or official docs.
  - fake-node smoke comes only after runtime validation.

Likely owning components:

- `agent/harness/groups.py`
  - unknown-chain confirmation
  - protocol confirmation
  - new-chain endpoint/method questions
- `agent/harness/oracle.py`
  - context explanation for Case2 endpoint/method questions

### Failure D: Consultation Misrouted To Workflow

Known examples:

```text
User> 默认 workload 是什么？我可以加自定义 rpc 吗？
```

Required behavior:

- If RPC mode/workload is pending, explain current chain defaults and custom RPC
  requirements.
- Do not start custom RPC endpoint workflow unless the user clearly provides a
  method, endpoint, request/response/docs, or explicitly says to add/validate a
  concrete method.

Likely owning components:

- `agent/harness/intent.py`
- `agent/harness/groups.py`
- `agent/harness/oracle.py`

## Required Repair Plan

### Step 0: Audit Before Code

Read:

- all files listed in "Current Architecture Map"
- current tests listed above
- `config/chains/ethereum.json`
- `config/chains/solana.json`
- `config/chains/bsc.json`
- `agent/validators/rpc_workload.py`
- `agent/validators/endpoint_probe.py`
- `tools/target_generator.sh`

Write notes before editing:

- which file owns each failure
- which file must not be touched
- which behavior is a workflow rule versus a user-facing explanation

### Step 1: Fix State Explanation Semantics

Implement product answers for:

- current confirmed config
- startup discovery summary
- latest job summary
- previous session summary
- "what can I do now"

Do not return raw internal group ids.

### Step 2: Fix Case2 Question Sequence

When user chooses real unknown chain in an existing family:

1. Confirm canonical chain name.
2. Confirm research availability:
   - google_search available or not.
3. Confirm adapter family.
4. Explain endpoint + method evidence requirements.
5. Ask for endpoint and method evidence in a way a user can answer.

Question wording must support:

- endpoint only first
- method only first
- request/response together
- official docs pasted first
- user saying "what do you need from me?"

### Step 3: Fix Consultation Versus Workflow Answering

For any pending question, user may ask:

- "what is this?"
- "why do you need this?"
- "what can I do?"
- "what did you infer?"
- "can I add custom RPC?"
- "what is fake-node for?"

The Agent must answer the question and preserve the pending question unless the
user explicitly changes group or provides an answer.

### Step 4: Re-run Real Dual-AI Chaos

Must run in Docker:

```bash
docker compose exec bench bash
cd /workspace
source config/agent_config.sh
./bin/anychain-agent --fresh-session
```

DeepSeek must be loaded as the Agent model. Codex/another AI must act as the
user.

Do not use only scripted single-turn tests.

## Required Chaos Personas

Run at least these personas:

1. New user / confused user
   - asks who the agent is
   - asks what it can do
   - asks what values were inferred
   - asks what to prepare
2. Impatient benchmark user
   - says "test solana max QPS and bottleneck"
   - then changes between fake-node and real-node
3. Technical paste user
   - pastes YAML/JSON/env snippets
   - mixes natural language with config values
   - asks the Agent to infer first, then confirm
4. Chain typo / unknown chain user
   - enters `sola`
   - insists it is a real chain
   - chooses jsonrpc/EVM
   - asks what endpoint/method evidence is needed
5. Custom RPC user
   - supported chain + custom method
   - gives endpoint from docs but says it is not final LOCAL_RPC_URL
   - gives request first, response later
   - asks about weights
6. Jumping user
   - configures disk halfway
   - jumps to QPS
   - jumps to RPC
   - asks log/report question
   - comes back
7. Reset/resume user
   - starts with previous session
   - asks what history exists
   - clears config
   - asks whether startup inference still exists

## Minimum Transcript Matrix

These exact transcripts or close variants must pass:

### Matrix 1: Clear Then Discovery Questions

```text
User> 3
User> 还有配置么？
User> 我现在关于最初推断的信息还有么？
User> 你有一个环境依赖的检测脚本，这个脚本会帮我推断一些变量，这些变量推断了么
User> 那现在可以做什么？
```

Expected:

- no fallback-only answer
- startup discovery summary appears
- confirmed config is empty
- available actions are explained

### Matrix 2: Unknown Chain Case2

```text
User> fake-node
User> sola
User> 3
User> 1
User> 我需要提供什么？
```

Expected:

- no silent coercion to solana
- clear DeepSeek/no-google-search boundary
- asks for canonical chain name if needed
- explains endpoint + method/request/response/docs
- explains fake-node needs runtime fixture/template validation first

### Matrix 3: Workload Consultation

```text
User> 使用 bsc fake-node
User> single
User> 默认 workload 是什么？我可以加自定义 rpc 吗？
User> 1
```

Expected:

- explains workload/custom RPC
- does not enter custom endpoint workflow
- `1` still applies the pending workload choice

### Matrix 4: Pasted Config

```text
User> 使用 solana fake-node
User> cloud:
User>   region: asia-east1
User>   zone: asia-east1-c
User> storage:
User>   ledger_device: vda
User>   data_vol_type: hyperdisk-balanced,
User>   iops: 20000 IOPS
User> 我没有 accounts 盘，QPS 用 quick，RPC 模式 single。
User> Y
```

Expected:

- values are accumulated
- no stale partial proposal reappears after Y
- follow-up actions continue
- language does not unexpectedly switch

## Verification Commands

Use Docker only:

```bash
docker compose exec -T bench bash -lc 'cd /workspace && source config/agent_config.sh >/dev/null 2>&1 || true && .venv-adk/bin/python -m unittest tests.test_agent_product_terminal tests.test_agent_langgraph_harness -v'
```

Then run real CLI chaos:

```bash
docker compose exec bench bash -lc 'cd /workspace && source config/agent_config.sh >/dev/null 2>&1 || true && ./bin/anychain-agent --fresh-session'
```

Capture transcript evidence for every chaos run:

- scenario name
- full user turns
- full Agent responses
- observed issue
- fix commit/diff summary
- retest result

## Definition Of Done

The repair is done only when:

1. The known failures in this document pass in real Docker CLI.
2. Unit/harness tests pass.
3. At least three dual-AI chaos rounds pass with different personas.
4. No runtime artifacts, local secrets, `__pycache__`, temporary checkpoints, or
   chaos transcripts are left as untracked product files unless deliberately
   stored under an ignored task/evidence path.
5. Agent code has one workflow owner: LangGraph Harness.
6. Terminal code does not contain benchmark business routing.
7. The Agent can explain startup discovery versus confirmed config.
8. Case1/2/3 behavior is globally reachable from any group.
9. Remaining limitations are clearly listed, especially Gemini + google_search
   if not available in the local environment.
