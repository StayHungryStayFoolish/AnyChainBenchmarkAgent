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
-> 校验并 checkpoint schema version 24
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

### 术语与生成路径验收

产品 review 必须验证真实对外表面，不能只搜索源码注释：

- startup 可以报告可选 Gemini search/Google ADK 可用性，但核心 runtime 必须是
  LangGraph Harness；
- 模型空输出和 provider failure 必须指向 configured model 或 AnyChain Agent，不能
  声称 ADK 拥有响应；
- structured doctor 的 `warnings` 和 `next_actions` 不得推荐 ADK Agent 或 ADK terminal；
- 提供给模型的 framework context 与 programmatic tool schema 必须遵守同一控制权边界；
- unknown-chain gap/onboarding 输出必须指向真实存在的
  `config/chain_template.json.bak`。

修改实现前先为这些输出增加确定性断言。内部兼容 identifier 可以保留旧名称，但用户或
模型可见的旧描述、以及不存在的生成路径都必须让该验收边界失败。

关键实现职责：

- `graph.py`：唯一 checkpointer-backed LangGraph runtime；
- `coordinator.py`：graph transition、唯一语义路由权威与唯一 typed commit
  boundary；
- `hierarchical_planner.py`：唯一通用语义规划器；
- `bounded_semantic_lane.py`：有限目录语义 mapper，不拥有通用路由、状态修改或
  commit 权限；
- `semantic_admission.py`：不可变 semantic document 的准备与校验；由模型
  grounding 的状态变更必须通过两次独立 whole-plan admission，并携带 durable
  consensus receipt；
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
queue。checkpoint schema version 24 是当前契约：v12 是隔离 migration
边界，v13 迁移 deferred queue retention，v14/v15 初始化 typed response，
v16 增加显式 pending owner 与 Chain/RPC case context，v17 增加
checkpointed semantic planning，v18 移除持久化的 turn-local response
文本与旧 response manifest，v19 增加不可执行的 semantic draft 边界并清除
不兼容的 in-flight draft，v20 绑定 Product Head 与 contract authority，
v21 增加 atom 级 evidence、secret reference 和原子 finalization，v22 增加
durable-state secret-binding registry，v23 增加签名 sensitivity、salted
memory-hard verifier、与 Product Head 同事务的 registry，以及
reference-only durable plan。v24 增加 admission-bound semantic-draft no-op
finalization receipt；它只能清除所有 atom 都是 background、所有 source unit 都
作为 context 通过 admission 的 ready draft，不能放宽普通模型空 plan 的 fail-closed
规则。v21 的原始凭据/reference 与 v22 的旧
binding/reference 必须 quarantine；当前版本缺失 secret 时使用 question
contract v6 重录入。旧 state 必须 fail closed、迁移或 quarantine。

semantic draft 等待澄清时，下一条自由文本不能自动视为答案。hierarchical
Harness 必须先把本轮绑定到精确签名的 draft revision 与 active atom；独立
reviewer 达成 quorum 后，只能授权一个完全一致、逐字节保留的原文片段作为
澄清 evidence。该片段由 coordinator 作为唯一 pending answer 处理，Stage A
仍须无损路由所有未被引用的兄弟片段。未形成 quorum 时 atom 保持 unresolved，
完整输入走普通路由。模型输出本身无权解析 atom，也无权直接修改产品状态。
如果进程在该 barrier 生效期间重启，session entry 必须重放同一个签名 draft
问题；不得使用通用 resume selector 替换它、改变 active group，或创建第二个
pending authority。
finalization 是一次性 replay 边界：它可以生成完整且通过 admission 的 action
plan，但不得针对已经接受澄清的 atom 再创建第二份 semantic draft。如果 owner
compiler 仍无法表示该 atom，原 draft 必须标记为 stale，并进入显式 unresolved /
failure 路径；再次询问同一个澄清问题属于产品失败。

同一个已审核回合可以同时包含上游配置修改，以及针对旧 pending question 的值。
此时必须由已注册的依赖图和失效图先执行上游修改、清除旧上下文派生的证据，并将
旧问题声明的 manual action 保留在新前置条件之后；只有新前置条件确认完成后，该
action 才能在新上下文中执行。新创建的 manual question 不能仅因另一个 action
具有相同的 `accepted_action_type` 就被绕过；该 action 的固定参数还必须与问题
声明的 manual action 一致。

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

每次成功的 method probe 都必须写入一份不含 secret 的 durable probe
contract，绑定 chain、物化后 endpoint 的 hash、所选 method、精确 params
与 adapter family。durable evidence 必须包含该 method 的成功 HTTP
observation、完整 response-shape hash 和 response sample/hash。RPC catalog
与 Case 2 promotion 调用同一 validator，重新计算 contract，并验证 owner
receipt、evidence 文件内容、request/schema hash 与 catalog revision。
real-node 执行前，只对 effective workload 实际选中的 custom method 在最终
物化后的 `LOCAL_RPC_URL` 上 replay；只有不属于 canonical chain template
的 method 才是 custom method，job-local 权重调整不会把模板 method 变成
custom method，空 effective workload 非法。历史 catalog 中未选中的 method
不是本次执行依赖。只有 owner receipt 一对一绑定全部必需的 custom method
proof 后，才能提交 endpoint 并向用户报告 ready。任意现有文件路径、陈旧或
被修改的 contract、
缺失的 secret reference、重复使用的 evidence/input binding 都必须 fail
closed。

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
不自己伪造 conversation 或 job。Phase 6 只关闭确定性 Gate G2。该长期契约不固化
当前 G3-G6 结果；必须针对目标 revision 运行 controller。任一必需 gate 未关闭时，
该 revision 都是 `not ready`，不能把局部 CLI 或双模型通过描述为最终产品验收完成。

PTY worker 只持久化 candidate JSON。worker 启动前，不可变 batch manifest
冻结 Ed25519 公钥信任根；私钥只在 batch controller 内存中生成，不写入共享
文件系统或环境变量。跨 controller 子进程时只通过继承 pipe descriptor 传递，
并在 worker 启动前关闭。controller 将 candidate 与冻结的 shard、schedule、
target、revision 和 runtime boundary 独立校验后，才签发 authority receipt。
已接纳 artifact 快照、receipt 与 commit marker 作为一个不可变 `.admitted`
目录原子发布；缺失、陈旧、部分发布或
由其他密钥签发的组合全部 fail closed。验证还会重新推导投影后的 source hash，
并拒绝 Mapping key 或 value 中的旧 runtime event 与 owning secret capability。
batch result 绑定已提交 bundle 摘要和公钥信任根；evidence 转换还必须由
orchestrator 显式传入该 trust-root ID，manifest 不能自行指定可信身份。
artifact 内部 self-hash 永远不能充当 source authority。
completed-batch source 只捕获一次转换所需 runtime 文件，后续转换只消费该
只读快照。产品证据记录原始 source path 与快照 hash，但后续 admission 不再
重读可变 runtime 文件。G3/G4 只公开 `evidence-batch` 转换命令；standalone
runtime-root 转换不能形成合格证据。

固定分母：

- G3：60 个 retained-regression obligation；
- G4：732 个 response-driven dynamic Chaos obligation，每轮完整覆盖，连续
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
PYTHONDONTWRITEBYTECODE=1 python3 -m compileall -q agent tests
git diff --check
python3 tests/agent_live/run_product_acceptance.py --through-phase 8 \
  --g3-authority-trust-root-id "$G3_AUTHORITY_TRUST_ROOT_ID" \
  --g4-round-1-authority-trust-root-id "$G4_ROUND_1_AUTHORITY_TRUST_ROOT_ID" \
  --g4-round-2-authority-trust-root-id "$G4_ROUND_2_AUTHORITY_TRUST_ROOT_ID"
```

单元测试或脚本 transcript 不能单独证明产品完成。必须补充动态
DeepSeek-backed conversation 与 4 个真实 execution lane。
三个 trust-root ID 必须来自冻结对应 batch 的 controller，禁止从当前待接纳
的 evidence manifest 中反向读取并自证。

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
