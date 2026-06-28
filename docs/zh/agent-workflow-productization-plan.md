# AnyChain Agent 产品化工作流执行计划

本文档用于约束 AnyChain Agent 后续开发。它不替代 `README.md`、
`agent/README.md` 或框架参考文档；它只回答一个问题：如何把 Agent 做成
可交付产品，而不是再次退回关键词匹配、模糊规则或半成品 wizard。

`CLAUDE.md` 是本地开发行为约束文件，不进入公开仓库，也不要在本任务中修改。

## 当前问题复盘

过去几轮失败的核心原因不是 fake-node、benchmark engine 或 ADK 本身，而是
Agent 工程边界没有真正落地：

- 只删除 terminal 关键词路由，但没有建立持久化 workflow state。
- 只把工具暴露给 ADK，但没有把 benchmark 的固定执行门禁变成可恢复流程。
- 只靠 instructions 约束模型，但没有给模型可读写的结构化会话事实。
- 输出清理逻辑一度被用来掩盖模型风格问题，容易滑向业务硬编码。
- 测试覆盖了正常路径，但对乱序、改主意、错误答案、语言切换和重启恢复覆盖不够。

产品级 Agent 必须同时满足两个条件：

- 自然语言理解由 ADK 和配置的 LLM 完成。
- Benchmark 执行流程由 workflow state、validators、preflight、smoke 和 approval
  gates 固定，不允许模型绕过。

## 不可退让边界

- 不允许 terminal、sanitizer、shell wrapper 使用关键词、正则或模糊匹配判断
  benchmark、onboarding、custom RPC、Prometheus/Grafana、结果分析等业务 intent。
- 不允许工具解析任意自然语言并自行决定业务目标。工具只处理结构化字段、
  repo facts、validator input 和 artifact。
- 不允许 fake-node 跳过 host、disk、network、process、volume baseline 等报告
  元数据确认。
- 不允许 real-node 在缺少 `LOCAL_RPC_URL`、`MAINNET_RPC_URL`、进程名、workload、
  sync-health 相关配置时进入 preflight/smoke。
- 不允许 mixed 权重不等于 100 时继续进入执行。
- 不允许 LLM 输出直接作为 shell 命令执行。
- 不允许未授权安装依赖、未通过 preflight/smoke 或未获得用户确认就提交真实压测。
- 不允许把模型常识当作新增链、新协议 family 或 RPC method 已支持的证据。

## 正确架构

AnyChain Agent 使用一个简化但强约束的 Agent Loop：

```text
Understand -> Plan -> Ask -> Configure -> Validate -> Execute -> Observe -> Analyze -> Iterate
```

职责边界：

- ADK/LLM：理解用户自然语言、语言跟随、上下文解释、乱序输入归位、选择 sub-agent。
- Workflow state：保存当前 intent、workflow、step、已确认字段、未决问题、最近变更。
- Deterministic tools：发现环境、读取 chain template、生成计划、校验配置、执行 smoke、
  提交 job、读取 artifact。
- Validators：阻止缺配置、错误权重、跳过 preflight/smoke、未授权安装依赖、未确认真实压测。
- Terminal：只处理稳定 I/O、Ctrl+C、依赖安装授权、job 恢复提示和固定命令。

## Workflow State 合同

Agent 必须把用户会话事实写入文件化状态，避免终端断开或多轮乱序后丢失主线。
最小结构：

```json
{
  "schema_version": 1,
  "session_id": "terminal-session",
  "language": "zh",
  "active_intent": "START_BENCHMARK",
  "active_workflow": "benchmark",
  "workflow_step": "target_mode_selected",
  "target_mode": "fake_node",
  "chain": "solana",
  "rpc_mode": "mixed",
  "rpc_methods": [],
  "mixed_weights": {},
  "custom_rpc": [],
  "confirmed_config": {},
  "inferred_config": {},
  "missing_fields": [],
  "pending_question": {
    "id": "confirm_ledger_device",
    "text": "请选择 LEDGER_DEVICE",
    "choices": []
  },
  "last_user_change": "",
  "latest_plan_file": "",
  "latest_job_id": "",
  "history_summary": ""
}
```

LLM 负责从用户输入中抽取结构化变化并调用 state 工具写入；validators 负责判断
是否允许进入下一步。状态工具不得做自然语言 intent 识别。

## START_BENCHMARK 工作流

用户只说“测一下”时，必须进入以下流程，而不是直接要求用户自己改配置：

1. `intent_detected`：识别为 benchmark 请求。
2. `chain_selected`：确认链，例如 `solana`。如果用户只回复链名，也要理解为
   对当前未决问题的回答。
3. `target_mode_selected`：确认 fake-node 还是 real-node。
4. `environment_discovered`：执行只读环境发现和 doctor。
5. `environment_confirmed`：确认 cloud/provider/deployment、CPU、memory、network、
   disk、ledger/accounts、volume baseline 和 process names。
6. `workload_confirmed`：确认 `single`/`mixed`，默认 RPC method、是否自定义 RPC、
   mixed 权重是否总和 100。
7. `config_validated`：通过 required config、chain template、RPC workload validators。
8. `plan_generated`：生成 plan 和 `runtime.env` 草案。
9. `preflight_passed`：preflight 通过。
10. `smoke_passed`：smoke 或 fake-node smoke 通过。
11. `approval_received`：用户明确确认开始真实/长时间压测。
12. `job_submitted`：后台 detached job 提交。
13. `observed`：读取 job status/logs/artifact index。
14. `analyzed`：基于 artifact 解释 HTML、CSV、per-method、disk/network/sync-health。
15. `iterate`：根据结果建议调整 workload、资源、RPC method 或重跑。

fake-node 到 real-node 切换时，不允许要求用户从头配置。应复用已确认的环境和
workload，只补真实节点需要的 RPC URL、主网/sync-health、进程名和可能变化的链/mode。

## 其他工作流

- `RESUME_JOB`：启动后发现已有 job 时，优先提供 status/logs/analyze/resume。
- `ANALYZE_ARTIFACTS`：必须引用具体 artifact 路径，不能只给自然语言结论。
- `ONBOARD_CHAIN_RPC`：新增链/RPC 时，先确认是否属于六个 family；若不确定，
  要求官方文档、内部 KB 或真实 request/response 样本；输出 coding brief。
- `CONFIG_HELP`：回答配置问题时读取 repo facts 和 docs index。
- `OUT_OF_SCOPE`：框架外问题礼貌说明边界，可给出与 benchmark 相关的替代方向。

## 必测混乱场景

以下场景不允许通过新增 terminal 关键词分支解决，必须依赖 ADK/LLM + workflow state：

- 用户只说“帮我测一下”。
- Agent 问链名后，用户只回复 `solana`。
- 中文对话中用户回复英文链名或 RPC method。
- 用户先选 fake-node，后改为 real-node。
- 用户先选 real-node，后改为 fake-node。
- 用户给出错误 URL、错误磁盘、错误权重后再修正。
- 用户说“随便填”，Agent 必须拒绝任意占位并解释哪些字段可默认、哪些必须确认。
- 用户中途问“怎么新增一个链”，回答后还能回到当前 benchmark workflow。
- 用户要求跳过 preflight/smoke/approval。
- 终端重启后，Agent 能发现上次 job 或 workflow state，并给出下一步。

## 执行任务

### P0. 文档与边界确认

- 保留 `CLAUDE.md` 本地化，不修改、不提交。
- 保留 `docs/zh/agent-product-acceptance.md` 作为产品验收边界。
- 本文档作为工作流产品化执行计划。
- 运行 `tools/check_agent_boundaries.py`，确保旧 wizard/keyword 入口不存在。

### P1. 持久化 Workflow State

- 新增文件化 state store。
- 新增 ADK tools：读取、更新、重置 workflow state。
- 每轮 terminal 调用 ADK 时注入 workflow state。
- State 工具只接收结构化 patch，不解析自然语言。

### P2. Instructions 与 Sub-agent 对齐

- Root instruction 必须要求每个业务回合读取或更新 workflow state。
- Benchmark/config/workload/execution/onboarding sub-agent 必须说明自己推进哪些 step。
- 禁止把 state 当作用户可见内部实现暴露。

### P3. Workflow Gate 测试

- 单元测试：state persist/update/reset。
- 单元测试：terminal prompt 不通过关键词判断业务 intent。
- 单元测试：state 注入到 ADK bridge。
- Live 场景：正常 benchmark、乱序回答、fake/real 切换、自定义 RPC、二次开发问答。

### P4. Fake-node 闭环验证

- 使用 36 链 fake-node 能力验证至少一个完整 smoke。
- 确认 `runtime.env`、job state、artifact index、HTML 报告路径均可追踪。

### P5. 文档收口

- README 只描述用户如何启动和使用 Agent。
- 复杂实现留在 `agent/README.md` 和本文档。
- 二次开发文档必须说明新增链、新 RPC、自定义 workload 的代码路径和验证命令。

## 完成标准

- `python3 tools/check_agent_boundaries.py --root .` 通过。
- `python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract` 通过。
- `python3 agent/cli.py adk-eval` 通过。
- DeepSeek 或其他真实模型 live scenarios 覆盖正常路径和混乱路径。
- fake-node smoke 至少一条完整链路通过，并生成可追踪 artifact。
- 生产运行时不存在业务关键词路由、旧 wizard、旧 responder、旧 fallback brain。
