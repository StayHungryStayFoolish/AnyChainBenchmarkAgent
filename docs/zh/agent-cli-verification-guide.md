# Agent CLI 验证指南

本文档供独立 AI 在真实 terminal 中验证 AnyChain Benchmark Agent。它是长期
验证清单，不是临时 task plan。

除非 repository owner 明确为本轮提供可用的真实 endpoint，否则只运行
fake-node 闭环。不要把生产 endpoint、客户数据或个人凭据写入共享证据。

## 目标

验证 `./bin/anychain-agent`：

- 在 Linux/Docker 的真实 terminal 中稳定启动；
- 通过 LangGraph Harness 调用配置的 LLM；Google ADK 只用于可选的 Gemini
  `google_search` grounding；
- 正确显示 provider/model/auth 与 web research 状态；
- 中英文输入、宽字符编辑和 `Ctrl+C` 稳定；
- 自动发现环境和依赖，不虚构 metadata；
- 每轮最多询问一个 blocking question；
- 支持乱序、回退、修正、模式切换与 partial group suspension；
- 保持 detached job 可恢复；
- 只在 unknown chain/protocol、custom RPC schema 和 sync-observe client
  setup 中使用 Gemini-only `google_search`。

workflow metadata 权威是 `agent/workflows/group_registry.py::GROUPS`：
20 个 `GroupSpec`，每个只属于 8 个 domain owner 之一。`coordinator` 是
control owner，不拥有 group。

graph transition：

```text
prepare -> adjudicate
  -> partition
  -> compile_owner
  -> review_plan
  -> admit
  -> select_action -> owner -> commit_action
  -> side-effect intent/invoke/receipt
  -> fallback -> compose -> validate
```

只有 active typed question 明确声明的精确编号/Y-N、可信 terminal command、
evidence transport framing 和空输入走确定性快路径。手动值、自然语言、
multi-intent 或结构化内容必须经过 semantic path，再由 pending/domain
contract 做确定性校验。

## 必须阅读

1. `AGENTS.md`
2. `AI_CODING_GUIDE.md`
3. `README.md`
4. `agent/README.md`
5. `docs/zh/anychain-agent-ai-work-gate.md`
6. `docs/zh/adk-agent-architecture.md`
7. `tests/agent_live/README.md`

## 环境规则

- 从 clean checkout 开始，记录 branch 与 commit；
- 产品验收仅在 Docker/Linux 运行；
- 安装 Agent 依赖：`bash scripts/install_agent_deps.sh --yes`；
- 只有需要 Gemini search 时才加 `--with-google-search`；
- credentials 只放 `config/agent_config.local.sh` 或进程环境；
- 不提交 key、ADC、service-account JSON、`.agent/`、live log、archive 或
  transcript；
- fake-node 只证明 fake-node 闭环，不能证明 real-node、sync-observe、
  custom-RPC live probe 或真实性能；
- 未提供已批准的真实 endpoint 时，不得声称 real-node 已测试。

## 基线命令

```bash
git status --short --branch
git rev-parse HEAD
bash scripts/install_agent_deps.sh --yes
python3 -m agent.cli adk-status
python3 -m agent.cli llm-config
./bin/anychain-agent
```

启动必须显示 model/provider/auth、web research、环境推断、历史 job 与 session
状态。无法访问 metadata server 时应显示 unknown，不得虚构 cloud、region、
zone 或 machine type。缺少执行依赖时先解释并请求用户确认。

## 自动化检查

修改前后都运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_offline_python_suite.py
python3 tools/check_agent_boundaries.py --root .
git diff --check
```

checkpoint schema version 23 是当前契约。migration 必须覆盖 v16 pending
owner/Chain-RPC context、v17 semantic planning、v18 response authority，以及
v19 non-executable semantic draft boundary、v20 Product Head/contract
authority、v21 atom evidence/secret reference/finalization、v22
durable-state secret binding，以及 v23 签名 sensitivity、memory-hard
verifier、registry transaction 与 reference-only durable plan；并安全移除或
quarantine 不兼容的 in-flight planning、response scratch、v21
raw/reference 与 v22 legacy binding/reference。

直接构造 `review_plan` state 的 test adapter 不能证明 graph split。证据必须
经过真实 `partition`、所有 scheduled `compile_owner`、`review_plan` 和
`admit` transition，并验证每阶段前后 checkpoint recovery、owner cursor、
semantic order、失败行为和 resume idempotency。

Phase 8 唯一 authority：

```bash
python3 tests/agent_live/run_product_acceptance.py --through-phase 8
```

该 controller 只生成 obligation catalog、admit 并验证下级 evidence，不会
替代真实 conversation 或 job。

## 双 AI Chaos 验证

广泛修改 workflow、routing、state 或 Harness 后，固定 matrix 仅是 regression
guard。还必须运行真实双 AI Chaos：

- 真实启动 Docker/Linux `./bin/anychain-agent`；
- AnyChain Agent 使用 DeepSeek；
- Codex 扮演用户；
- Codex 每轮先读取完整真实 response，再选择下一条消息；
- 保存完整 transcript，并校验 user-visible flow、checkpoint 和副作用。

必须使用多种 persona：

- 首次使用且困惑的 evaluator；
- 急躁、短答的 operations engineer；
- 重度粘贴 env/YAML/JSON/log/request/response 的技术用户；
- 随时更换 chain/mode/QPS/RPC/endpoint/region/observability 的用户；
- 中英文切换用户；
- report/log/error analyst；
- Case 1/2 custom-RPC integrator；
- Case 3 unsupported-chain evaluator；
- continue/modify/reset 的 resume-session 用户。

首次使用 persona 必须从非空 partial checkpoint 开始，询问身份、能力、从哪里
开始、三种模式区别，并验证旧 session 不会被静默复用。执行 approval 后必须
询问“是否执行过、当前状态、下一步是什么”，回答必须来自 checkpoint/job
证据，不能返回泛化帮助或“当前没有待确认问题”。

动态测试至少覆盖：

1. 任意开场，不只 `Hi`；
2. pending question 的有效短选项、无效值和自然语言 detour；
3. disk/network/chain/mode/workload/QPS/observability/analysis 之间跳转；
4. half-completed group 回退与再次恢复；
5. 已确认后修改 chain、mode、RPC、QPS、endpoint 与 observability；
6. 多行 env/YAML/JSON 推断、总结并逐项确认；
7. custom RPC 的 request/response/doc/endpoint 矛盾证据；
8. Case 1、Case 2、Case 3 及相互返回；
9. 中英文和带空格、标点的技术值；
10. partial session 的 continue/modify/reset；
11. approval 后真实 preflight/smoke/job/artifact 或具体 blocker；
12. major transition 后的 state/next-action 咨询。

发现问题先分类到 terminal、Harness state、semantic planning、
validator/tool 或 documentation owner。禁止 terminal keyword router、fuzzy
matching 和 transcript 专用分支。

## 手动终端测试

必须在真实 PTY 中运行 `./bin/anychain-agent`。

### 1. 启动、Auth 与本地发现

输入 `doctor`。验证 provider/model/auth/web research、deployment、CPU、
memory、network interface 与 disk candidate。metadata 不可用时显示 unknown。
依赖安装必须先请求确认。Gemini 环境额外运行：

```bash
python3 -m agent.cli llm-config
python3 -m agent.cli llm-smoke --prompt 'Return JSON only: {"ok": true}'
```

### 2. Terminal 编辑与中断

分别输入并修改中英文错误文本；在空 prompt 和输入过程中按 `Ctrl+C`。不得
出现 ghost space、重复 `User>`，不得终止 detached job。response language
跟随当前用户语言，命令、路径、变量、链和 method 保持原文。

### 3. Fake-Node 流程

输入：`I want to benchmark Solana using fake-node first.`

验证 chain、RPC mode、benchmark mode、QPS、observability、disk、network
逐项确认。fake-node 可以使用默认 endpoint，但不能跳过 resource metadata。
preflight、smoke 和 user approval 之前不得提交执行。

### 4. Benchmark Mode 与 QPS

分别选择 `quick`、`standard`、`intensive`，询问含义和可调整项。Agent 必须
解释 initial QPS、max QPS、QPS step、duration 和 RPC mode，询问保留默认值
还是调整具体一项。只修改用户选择的字段并重新校验。

### 5. RPC Mode、Weight 与自定义 Method

输入：`Use mixed workload with getSlot 70% and getBlockHeight 30%.`

weight 必须精确合计 100，否则给出原因并引导修正。每个 method 的归因不能
丢失。single 也必须展示当前 method，并允许在已配置 method 间切换或进入
custom RPC 流程。

### 5A. Sync-Observe

输入：`Observe my BSC node while it is syncing. Do not send RPC benchmark load.`

必须进入 `sync-observe`，不能询问 RPC mode、weight、Vegeta 或 QPS。确认
chain、sync reference、process、可选 Prometheus metrics endpoint、disk、
network 与 stop condition（直到停止、duration、直到同步完成）。报告保留
height、CPU、memory、disk、network 和 metric source/status；没有 MGas/s
时显示 unavailable。

### 6. 多磁盘与可选 Accounts Disk

显示所有有效 disk candidate、设备名和推断容量；允许编号与手动输入。
`LEDGER_DEVICE` 必须配置；`ACCOUNTS_DEVICE` 明确可选。选择设备后确认 type、
size、IOPS、throughput，推断值使用 Y/N 或自定义值确认。

### 7. 回退与修正

在任意 group 配置一半时跳转、再次跳转、返回并修正。已完成 group 不能重复
询问；dependency invalidation 后必须重新校验受影响字段并继续 canonical
fallback。

### 8. Fake-Node 切换 Real-Node

复用仍有效的资源 metadata，只询问 delta：真实 `LOCAL_RPC_URL`、
`MAINNET_RPC_URL` 或 sync health、chain/workload 变化和 final approval。
没有真实 endpoint 时只验证 planning conversation，不运行 benchmark。

### 9. 自定义 RPC Method

验证以下信息分别收集和确认：

- method 名称；
- endpoint provenance 与最终 validation endpoint；
- request/response 或官方文档；
- 零参数、positional params 或 object params；
- 每个参数的 index/name、JSON wire type、区块链语义/编码、含义、
  required/optional 和 example；
- response 结构；
- workload scope、fixture plan 和 weight。

必须用确认后的 wire request probe endpoint。`single_replace` 只启用一个
validated method；`mixed_replace` 移除 defaults；`mixed_add` 保留 defaults；
mixed weight 全部为正整数且精确合计 100。只生成 job-local override。

### 10. 未支持链

输入一个不在模板和别名中的链。先由 LLM 判断链是否存在与协议；Gemini
search 可用时用官方搜索证据 grounding，然后由用户确认。已有 adapter family
进入 Case 2；不支持或不确定进入 Case 3，生成二次开发 handoff，不得假装已经
支持。

### 11. Knowledge Base

询问是否连接 enterprise KB。Agent 应报告 disabled/noop/HTTP/custom。KB
错误不能阻塞本地 fake-node 闭环。

### 12. Observability

解释 disabled、local Prometheus/Grafana 和 exporter-only。启动本地 stack 前
检查 exporter/Prometheus/Grafana port、自动停止策略并请求确认。外部
Prometheus 必须抓取 exporter endpoint。

### 13. Detached Job、日志与恢复

运行短 fake-node job。验证 `jobs`、`status`、`logs <job_id>`、
`follow <job_id>`；`Ctrl+C` 退出 follow 但不杀 job；重启 CLI 后恢复最新
job 状态。

### 14. 乱序与矛盾

连续切换 real/fake、single/mixed、错误 disk、QPS 与 chain。状态必须替换
矛盾值；仍有效的确认值可以复用；变更触发声明的 invalidation；每轮只问下一个
blocking question。

### 15. 超出范围

例如要求编写 trading bot。Agent 应拒绝或引导回 benchmark 能力，不执行无关
shell command，且后续 benchmark 对话仍可继续。

## 修复失败

定位最小责任层：

- semantic planning：`hierarchical_planner.py`；
- semantic document/admission：`semantic_admission.py`；
- validator/tool：`agent/validators/`、`agent/tools/executor.py`；
- terminal UX：`agent/terminal/`；
- workflow state：`agent/harness/`；
- live provider gap：`tests/agent_live/`；
- documentation drift：对应长期文档。

修复共享契约并回归相邻路径。不要加入 keyword list、fuzzy matching、regex
intent router 或 prompt/terminal 补丁。

## 返回证据

报告必须包含：

- branch、commit、Docker/Linux 环境；
- provider/model/auth，凭据脱敏；
- Gemini `google_search` 是否真实可用；
- 执行命令与 evidence path；
- manual/dynamic boundary 的 pass/fail；
- 发现问题、root owner 与修改文件；
- 修复后测试；
- not-run、externally-blocked 和 remaining gap。

必须区分 generated/cataloged 与 observed execution。只有检查 checkpoint、
runtime env、target、job、log、CSV、HTML 和 artifact 后，相关 edge 才能标记
observed pass。
