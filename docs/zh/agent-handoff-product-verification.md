# AnyChain Agent 外部 AI 交接与产品验收规范

状态：长期维护、提交到 Git 的交接与验收契约

运行目标：仅支持 Linux/Docker

外部能力缺口：真实 Gemini `google_search` grounding

## 从这里开始

另一个 coding AI 在没有原始会话上下文时，必须依次阅读：

1. `AI_CODING_GUIDE.md`
2. `AGENTS.md`
3. `docs/zh/anychain-agent-ai-work-gate.md`
4. `docs/zh/adk-agent-architecture.md`
5. `agent/README.md`
6. 本文档
7. `tests/agent_live/README.md`
8. `tests/agent_live/legacy_issue_map.py`

不要恢复已删除的日期型 task plan，也不要从旧 transcript 推导产品行为。
代码 registry 与上述长期文档是权威来源。旧问题只有在当前 revision 上存在
可验证证据时才算关闭。

## 产品架构

产品入口是 `./bin/anychain-agent`。terminal 仅负责输入输出、启动诊断、
依赖安装确认、`Ctrl+C` 和精确 job command，不负责 benchmark 业务路由。

LangGraph Harness 是唯一的对话与 workflow 控制平面：

```text
完整 terminal turn
-> prepare / adjudicate
   -> 确定性路径：精确的已声明选项、Y/N 或可信 command/transport
   -> 语义路径：partition
      -> 每个 graph transition 编译一个 owner
      -> review_plan 对完整不可变 plan 做独立语义 admission
      -> admit 做确定性 action、依赖、冲突与顺序校验
-> 选择一个 durable action
-> 路由到唯一 owner
-> commit 一个 typed result
-> 外部副作用先持久化 intent，再 invoke，最后持久化 receipt
-> 通过 graph transition 继续处理已 admission 的工作
-> canonical fallback
-> 每轮最多一个 blocking question
-> 校验并 checkpoint schema version 18
```

任意手动输入值、自然语言、multi-intent prose 或需要语义归属的结构化内容都
不能绕过 semantic planning。只有 active question 明确声明的精确选项、精确
Y/N、可信 terminal command、evidence transport framing 和空输入可以走本地
确定性快路径。

唯一 workflow metadata 权威是
`agent/workflows/group_registry.py::GROUPS`。当前有 20 个 group，每个 group
只属于 8 个 domain owner 之一：

| 顺序 | Group | Owner |
|---:|---|---|
| 1 | `opening` | `orientation` |
| 2 | `target_mode` | `chain_rpc` |
| 3 | `chain_identity` | `chain_rpc` |
| 4 | `provider_deployment` | `environment` |
| 5 | `ledger_disk` | `environment` |
| 6 | `accounts_disk` | `environment` |
| 7 | `network` | `environment` |
| 8 | `endpoint_process` | `chain_rpc` |
| 9 | `chain_auxiliary_endpoints` | `chain_rpc` |
| 10 | `workload_rpc` | `chain_rpc` |
| 11 | `target_samples_fixtures` | `chain_rpc` |
| 12 | `qps_profile` | `performance` |
| 13 | `sync_observe` | `sync_observe` |
| 14 | `observability` | `performance` |
| 15 | `advanced_tuning` | `performance` |
| 16 | `preflight_smoke_execution` | `execution` |
| 17 | `job_monitoring` | `execution` |
| 18 | `failure_recovery` | `recovery` |
| 19 | `error_evidence_analysis` | `analysis` |
| 20 | `report_artifact_analysis` | `analysis` |

8 个 owner 的职责：

- `orientation`：入口、session continue/modify/reset；
- `environment`：provider/deployment、ledger、accounts、network；
- `chain_rpc`：target mode、chain identity、endpoint、workload、fixture；
- `performance`：QPS、observability、advanced tuning；
- `sync_observe`：同步观察配置；
- `execution`：preflight/smoke、final execution、job monitoring；
- `recovery`：可执行的失败恢复；
- `analysis`：粘贴证据、日志、报告和 artifact 分析。

graph 另有 `coordinator` control owner，仅处理 typed pending answer 和 graph
control action。它不拥有 `GroupSpec`，不是第 9 个 domain owner。

关键实现职责：

- `graph.py`：唯一 checkpointer-backed LangGraph runtime；
- `coordinator.py`：graph transition 与唯一 typed commit boundary；
- `hierarchical_planner.py`：唯一语义规划入口；
- `semantic_admission.py`：不可变 semantic document 的准备与校验；
- `admission.py`：action、冲突、前置条件和 coverage 校验；
- `queue.py`：依赖安全排序和 pending barrier；
- `routing.py`：navigation、return policy 和 canonical fallback；
- `response.py`：唯一 `visible_response` writer；
- `response_catalog.py`、`response_messages/`：唯一的本地化产品文案权威；
- `advisory.py`：不能修改状态或控制 graph 的 model-backed 分析；
- `domains/`：8 个业务 owner；
- `terminal/repl.py`：纯 terminal shell；
- `runners/benchmark_pipeline.py`：plan materialization 与 job submission。

runtime 只能编译一个带 SQLite checkpointer 的 graph。禁止第二个
uncheckpointed turn graph，禁止在一个 Python node 中 drain 完整 durable
queue。checkpoint schema version 18 是当前契约：v12 是隔离 migration
边界，v13 迁移 deferred queue retention，v14/v15 初始化 typed response，
v16 增加显式 pending owner 与 Chain/RPC case context，v17 增加
checkpointed semantic planning，v18 移除持久化的 turn-local response
文本与旧 response manifest。旧 state 必须 fail closed、迁移或 quarantine。

## 历史问题迁移

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/agent_live/legacy_issue_map.py \
  > /tmp/anychain-legacy-issue-map.json
```

`guarded` 只表示当前存在对应 regression guard，不代表当前 revision 已运行
通过；`superseded` 表示当前架构契约替代旧实现；`open` 表示证据缺失、
过时、间接或外部阻塞。不要恢复退休的 dated issue 文档。

## Workflow 契约

Agent 不是线性 wizard。用户可以在任意 partially completed group 中：

- 跳转到另一 group，完成或不完成后再跳转；
- 回退、修正已确认值、切换语言、链或模式；
- 提问、粘贴多行 JSON/YAML/env/curl/log；
- 一次表达多个独立需求。

partial state、interruption return point 和 deferred action 必须 checkpoint。
每个 action 后由共享 validator 选择最早且适用的 incomplete group。已完成
group 只有在声明的 dependency invalidation 后才可以重新打开。

consultation 与 analysis 是只读 detour：回答后必须恢复原 active blocking
question，且只能显示一次。每个显示给用户的选项都必须映射到有 owner、
prerequisite、postcondition、return policy 和本地化 message 的 typed action。

## Chain 与 RPC 三类流程

### Case 1：已配置链增加自定义 RPC

不得修改 canonical `config/chains` template。需要收集或识别：

- method 名称；
- ordered params、每个参数的 JSON wire type、区块链语义/编码、含义、
  required/optional 与示例；
- request、response 结构或官方文档；
- 可访问的最终 validation endpoint。

示例中的 endpoint 不自动成为最终 `LOCAL_RPC_URL`。必须用确认后的 wire
request 验证最终 endpoint 与 method。fake-node 需要录制 fixture。workload
scope 为 `single_replace`、`mixed_replace` 或 `mixed_add`；active mixed
weight 必须为正整数且精确合计 100。只生成 job-local runtime override，
canonical chain template 保持不变。

### Case 2：未配置链，但属于已有 adapter family

先让当前 LLM 判断链是否存在及其协议；Gemini search 可用时，再用官方搜索
证据 grounding。用户确认 chain identity 与 adapter family 后，复用 Case 1
的 endpoint/schema/request/response/fixture/weight 契约。runtime template
和 fixture 在 smoke 成功前都是 provisional，不得声称永久支持，也不得修改
canonical template。

### Case 3：不支持或无法确认 adapter family

收集官方 protocol、RPC、endpoint、request 与 response 证据。Gemini search
不可用时要求用户提供官方材料。生成可交给 coding AI 的二次开发 handoff，
并停止 benchmark execution。用户可以随时返回 Case 1、Case 2、已支持链或
其他模式。

## 执行契约

- `fake-node` 验证 fixture、执行、日志与报告闭环，不测真实节点性能；
- `real-node` 验证最终 endpoint/workload，执行 preflight 和隔离安全 smoke，
  reconcile 持久化结果，再单独询问 final benchmark approval；
- `sync-observe` 需要真实节点或数据源，只观察高度、process/core CPU、
  memory、disk、network 和 client metrics，不走 Vegeta、QPS、proxy target
  或 fixture 录制；缺少 MGas/s 必须标记 unavailable，不能伪装成有效的 0；
- 重复 approval 必须 idempotent；
- 每个 plan 拥有隔离的 output 与 memory 目录；
- failed/partial job 必须提供证据和可执行 recovery choice，允许用户只修正
  受影响 group，并通过 canonical fallback 恢复，不能重复提交 job。

## 双 AI Chaos 验收

固定 CLI matrix 只是 regression，不是产品验收。产品验收必须在真实
Linux/Docker CLI 中运行：AnyChain Agent 使用 DeepSeek，Codex 扮演用户，
Codex 必须先读取本轮真实 Agent response，再决定下一轮输入。预写 prompt
序列不属于双 AI Chaos。

`tests/agent_live/run_product_acceptance.py` 是唯一 G0-G6 evidence admission
与状态权威。Phase 8 接收下级 provider 生成的 revision-bound evidence，但
不自己伪造 conversation 或 job。

固定分母：

- G3：60 个 retained-regression obligation；
- G4：625 个 response-driven dynamic Chaos obligation，每轮完整覆盖，连续
  两轮使用不同 round/session/request/execution/evidence identity，并且没有
  新 S1/S2 root class；
- G5：4 个真实执行 lane；
- G6：9 类 revision-bound product-review artifact。

必须覆盖的 persona 包括：首次使用且困惑、急躁短答、重度复制粘贴、频繁改
需求、混合语言、报告/日志分析、自定义 RPC 集成、unsupported chain、
resume/modify/reset。每次都要根据实际 response 继续，不得仅复读已知失败
transcript。

关键 sequence family：

1. fresh/resume/modify/reset 与 startup discovery consultation；
2. 所有 target mode 切换；
3. 所有 environment subgroup、partial jump 与 invalidation；
4. Case 1 single/mixed/default/custom/replace/add/weight；
5. Case 2 identity/family/endpoint/schema/fixture/smoke；
6. Case 3 evidence/handoff 并返回 Case 1/2；
7. QPS、observability、advanced tuning 与乱序 prerequisite；
8. fake-node、real-node 两阶段执行、sync-observe、job、failure/retry；
9. 跨多个 group 的 compound natural language 与多行结构化输入；
10. major pending contract 中的 consultation 与精确恢复。

## 本地 Docker 真实证据

`tests/agent_live/local_evm_node.sh` 可启动、探测和停止 digest-pinned Geth
development service。它是测试基础设施，不是产品依赖。它可以验证 endpoint、
custom EVM RPC traffic、real-node smoke/final job、idempotency 和
sync-observe metrics boundary，但不能证明 mainnet catch-up 性能或有意义的
MGas/s。

不得提交 provider key、endpoint credential、checkpoint、job output、
transcript、local coverage ledger、cache 或开发 key。

## 必须验证

所有命令在 Docker/Linux 中运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_offline_python_suite.py
python3 tools/check_agent_boundaries.py --root .
git diff --check
python3 tests/agent_live/run_product_acceptance.py --through-phase 8
```

单元测试或脚本 transcript 不能单独证明产品完成。必须补充动态
DeepSeek-backed conversation 与 4 个真实 execution lane。

## 修复规则

发现失败后，先记录 root cause 与 owner，再修改代码。修复共享架构契约，
并回归 exact、isomorphic、negative、neighboring 路径。禁止：

- transcript/phrase/language/chain/question-id 专用分支；
- fuzzy matching 代替 LLM semantic interpretation；
- terminal business router；
- 第二状态源、第二 Agent loop 或第二 group registry；
- 恢复退休 current-version compatibility；
- 降低 validator、invariant、postcondition 或测试强度；
- 只让观察到的一个 transcript 变绿。
