# AnyChain Agent 产品级验收与剩余任务

本文档是 AnyChain Agent 进入产品级交付前的唯一执行边界。它不替代
用户文档，也不作为框架事实来源。用户文档仍以 `README.md`、
`agent/README.md`、`docs/zh/framework-flow.md`、
`docs/zh/framework-reference.md`、`docs/zh/how-to-add-chain.md` 和
`docs/zh/local-closed-loop-testing.md` 为准。

## 当前结论

当前 Agent 已完成从 terminal 关键词路由到 Google ADK + LLM 编排的主线
切换。terminal 只允许处理稳定 shell 命令、依赖安装确认、job 恢复提示
和固定状态查询。所有 benchmark、onboarding、自定义 RPC、分析和可观测性
请求必须进入 ADK root agent，再由专门 sub-agent 和确定性工具处理。

当前还不能宣称完全产品级。产品级验收必须证明：意图识别、状态推进、
变量确认、preflight、smoke、后台 job、报告分析、错误恢复和二次开发引导
都能在真实模型调用下稳定工作。

工作流产品化的执行细节记录在
`docs/zh/agent-workflow-productization-plan.md`。该文档要求 ADK/LLM 负责
自然语言理解，持久化 workflow state 负责保存已确认事实和当前步骤，
validators/preflight/smoke/approval gates 负责阻止越界执行。

## 不允许回退的边界

- 不允许恢复 terminal 业务关键词路由。
- 不允许恢复旧的 wizard、direct responder 或 fallback chat loop。
- 不允许把 LLM 输出直接作为 shell 命令执行。
- 不允许绕过 deterministic validators、preflight、smoke 和用户确认。
- 不允许 fake-node 模式跳过主机、磁盘、网络、进程名等报告元数据确认。
- 不允许将缺失配置解释为“随便填”或任意占位。
- 不允许输出内部工具名、sub-agent 名、route confidence 或 scratchpad。
- 不允许把模型常识当作新增链或新增 RPC method 的支持证据。

## 执行入口约束

任何 Agent 代码修改开始前，开发者或 LLM coding agent 必须先读取本地
开发行为约束文件（如果存在）。该文件只在本地保存，不进入公开仓库。执行时
必须先写明：

- 当前假设。
- 不确定点。
- 最小修改范围。
- 成功标准。
- 验证命令。

无法自动强制的行为由本地开发行为约束文件约束；可机械化的边界由以下命令检查：

```bash
python3 tools/check_agent_boundaries.py --root .
```

## LLM-first 交互与执行模型

产品级 Agent 的核心不是把用户输入映射到一组硬编码 `if/else`，而是让
底层 LLM 在 ADK 的会话、sub-agent、工具和 validator 约束下理解用户意图，
再把不确定性转化为可验证的下一步。

### AnyChain Agent Loop

AnyChain Agent 的产品形态必须是闭环 Agent，而不是“一次 prompt +
脚本执行”。每一次自然语言交互都必须进入同一个循环：

```text
Understand -> Plan -> Ask -> Configure -> Validate -> Execute -> Observe -> Analyze -> Iterate
```

这个循环吸收了当前 Agent 工程里常见的 Think/Act/Observe 或
Reason/Act/Observe 模式，但按 AnyChain 的 benchmark 场景扩展为可验收的
工程边界：

- Understand：ADK + LLM 理解用户目标、语言、上下文、乱序输入和不确定性。
  terminal 不允许通过关键词或模糊匹配替代这一步。
- Plan：ADK root agent 选择合适 sub-agent，并把用户目标归一化为结构化
  benchmark、配置、分析、onboarding 或知识问答路径。
- Ask：如果配置缺失、冲突或无法安全推断，Agent 一次只问一个小组的阻塞
  问题，优先使用 yes/no 或短编号选项。
- Configure：把用户确认的信息写入计划或 runtime artifact。`runtime.env`
  只能表示本次 job 的最终确认配置，普通用户不手动编辑。
- Validate：调用 deterministic validators、dependency audit、chain template
  checks、workload checks、preflight 和 smoke。模型不能用自然语言承诺替代
  validator 结果。
- Execute：只有在 validate 通过且用户确认后，才能提交 smoke、fake-node
  validation 或 detached benchmark job。真实压测默认后台执行。
- Observe：读取 job state、logs、artifact index、CSV、HTML、Prometheus/
  Grafana 状态和错误信息。Agent 必须基于事实观察进入下一轮。
- Analyze：解释 RPC 成功率、P50/P90/P99、per-method attribution、CPU、
  disk、network、sync-health 和瓶颈证据，并引用具体 artifact 路径。
- Iterate：根据观察和分析建议下一步，例如补配置、改 workload、增加
  RPC method、切换 fake-node/real-node、重跑 smoke 或生成二次开发 handoff。

这个 loop 是开发和验收的主线。任何新增能力都必须说明自己进入 loop 的
哪个阶段、使用哪个确定性工具产生事实、由哪个 validator/callback 阻止
越界，以及如何把结果回写为用户可复查的 artifact。

### Loop 与 ADK 职责关系

ADK 负责 Agent loop 的编排，不负责替代 benchmark engine：

- ADK session/memory 保存对话上下文、已确认字段、当前 job 目标和未决问题。
- 文件化 workflow state 保存可恢复的结构化事实，例如当前 intent、workflow
  step、链、fake-node/real-node、RPC mode、权重、缺失字段、未决问题和最新 job。
- ADK root agent 负责从用户自然语言进入正确 loop 阶段。
- ADK sub-agent 负责领域分工，例如环境发现、依赖、配置、workload、
  onboarding、执行、恢复分析和知识问答。
- ADK tools 只暴露确定性能力：读取事实、生成计划、验证配置、执行 smoke、
  提交 job、读取 artifact。工具不做自然语言意图识别。
- ADK callbacks/validators 是 checkpoint，用来阻止缺配置、跳过 preflight、
  跳过 smoke、未授权安装依赖、未确认提交真实压测等高风险路径。

如果实现方式让 terminal、sanitizer、shell wrapper 或工具函数承担了
Understand/Plan 责任，就是错误实现。它会让 Agent 看起来能跑，但无法处理
用户乱序输入、错误答案、切换目标和二次开发问题。

### Loop 失败判定

出现以下任一情况，即认为 Agent loop 失败，不能宣称产品级：

- 通过关键词数组、正则或模糊匹配决定业务 intent。
- terminal 在未进入 ADK 的情况下推进 benchmark、onboarding、自定义 RPC、
  Prometheus/Grafana 或结果分析流程。
- sanitizer 为了隐藏模型风格或业务错误而重写正常自然语言。
- LLM 没有先观察工具结果，就声称配置正确、链已支持、RPC fixture 存在、
  benchmark 成功或瓶颈已经定位。
- 用户要求跳过 gate 时，Agent 用自然语言答应或寻找绕过路径。
- fake-node 流程没有确认真实报告所需的 host/disk/network/process 元数据。
- 用户从 fake-node 切换 real-node 时，Agent 丢弃已确认配置并要求完全重来。
- 结果分析没有引用实际 artifact、CSV、HTML、job state 或报告文件路径。

### 职责分层

- terminal 只负责稳定输入输出、Ctrl+C、语言初始状态、依赖安装授权、
  job 恢复提示和固定命令。
- LLM/ADK 负责自然语言意图识别、上下文理解、乱序输入解释、用户目标归一化、
  澄清问题生成和 sub-agent 选择。
- sub-agent 负责某一类领域任务，例如环境发现、依赖、配置、workload、
  onboarding、执行、恢复分析和知识问答。
- deterministic tools 负责读取事实、生成计划、校验配置、执行 preflight/smoke、
  创建 runtime artifact 和提交 job。
- validators/callbacks 负责阻止危险路径，不负责猜用户意图。

### 复杂交互必须支持的能力

- 用户不按顺序回答时，LLM 应结合会话状态判断回答属于哪个未决问题。
- 用户一次性提供多项信息时，LLM 应抽取可用字段，剩余缺口交给 validator。
- 用户给错字段时，LLM 应解释错误并引导修正，而不是重新开始整个流程。
- 用户中途从 fake-node 切换 real-node 时，LLM 应复用已确认的环境/资源配置，
  只补 real-node 必需字段。
- 用户从 real-node 切回 fake-node 时，LLM 应保留 workload、链、资源元数据，
  只替换目标 RPC 来源。
- 用户修改链、RPC mode、权重或自定义 method 时，LLM 应识别这是对当前计划
  的变更，而不是新会话。
- 用户问框架能力或二次开发问题时，LLM 应调用知识/框架事实工具，而不是只靠
  模型记忆。
- 用户要求跳过 preflight、smoke 或 approval 时，LLM 应保持友好解释，但
  deterministic gate 必须拒绝执行。

### 禁止的实现方式

- 禁止用中文/英文关键词数组判断 benchmark、onboarding、custom RPC 或
  observability 意图。
- 禁止在 terminal 里写业务流程状态机来替代 ADK session/sub-agent。
- 禁止在 runner bridge 或 sanitizer 中使用短语级风格正则来掩盖 Agent
  行为问题；输出风格应由 ADK instructions、sub-agent 设计和 live tests 约束。
- 禁止用 sanitizer 修业务逻辑；sanitizer 只能移除内部实现泄露或明显
  scratchpad。
- 禁止让工具解析任意自然语言并自行推断业务目标；工具只能处理结构化字段、
  repo facts 和 validator input。

### 正确的执行路径

1. 用户输入自然语言。
2. terminal 将输入和当前 session state 交给 ADK root agent。
3. root agent 通过 LLM 判断意图和不确定性，并选择合适 sub-agent。
4. sub-agent 调用只读工具读取当前 repo、环境、chain template、job 或 artifact。
5. sub-agent 调用 validator 找出缺失、冲突或危险配置。
6. LLM 用用户语言提出一个小组的确认问题，或给出可执行的下一步。
7. 用户确认后，执行类工具才允许生成 runbook、smoke 或 detached job。
8. 每个执行结果都必须回写 artifact，并在回答中引用具体路径。

### 验收方法

复杂交互不能只靠 unit tests。必须用真实模型 live acceptance scenarios 验证：

- 同一个 intent 的多种表达方式。
- 中英文混合表达。
- 用户乱序、漏答、答错、改主意。
- 用户要求绕过 gate。
- 用户提出框架外问题。
- 用户要求新增链或新增 RPC method。

只有当这些场景都不需要新增 terminal 关键词分支即可通过时，才能认为
LLM-first 意图识别和 sub-agent 处理达到可接受水平。

## 必须完成的任务

### T1. 意图识别与路由验收

目标：证明自然语言请求由 ADK + LLM 识别，并进入正确业务流程。

验收场景：

- 问框架支持多少链、哪些 RPC method。
- 发起模糊 benchmark 请求，例如“帮我测一下节点性能”。
- 发起明确 fake-node benchmark 请求。
- 发起真实节点 benchmark 请求。
- 从 fake-node 切换到 real-node。
- 询问已有 Prometheus/Grafana 如何接入。
- 询问新增链、新增协议 family、新增 RPC method。
- 用户乱序输入、给错值、撤销或修改前一个选择。

通过标准：

- terminal 不出现业务关键词路由。
- ADK 输出语言跟随用户输入语言。
- 每个业务请求都进入 ADK，并由工具或 validator 给出下一步。
- 不泄露内部工具名、sub-agent 名、scratchpad 或 confidence。

### T2. 配置推断与交互确认

目标：Agent 启动后先做只读环境检测，再让用户确认无法安全推断的变量。

必须覆盖：

- `CLOUD_PROVIDER`
- GCE/GKE/EC2/EKS/Kubernetes/Other 部署形态
- region、zone、machine type
- CPU、memory
- `NETWORK_INTERFACE`
- `NETWORK_MAX_BANDWIDTH_GBPS`
- `LEDGER_DEVICE`
- `ACCOUNTS_DEVICE`
- `DATA_VOL_TYPE`
- `DATA_VOL_SIZE`
- `DATA_VOL_MAX_IOPS`
- `DATA_VOL_MAX_THROUGHPUT`
- `ACCOUNTS_VOL_TYPE`
- `ACCOUNTS_VOL_SIZE`
- `ACCOUNTS_VOL_MAX_IOPS`
- `ACCOUNTS_VOL_MAX_THROUGHPUT`
- `BLOCKCHAIN_NODE`
- `LOCAL_RPC_URL`
- `MAINNET_RPC_URL`
- `RPC_MODE`
- `BLOCKCHAIN_PROCESS_NAMES`
- benchmark mode：`quick`、`standard`、`intensive`
- 对应 mode 的 QPS profile：`*_INITIAL_QPS`、`*_MAX_QPS`、`*_QPS_STEP`、
  `*_DURATION`
- observability：disabled、local Prometheus/Grafana、existing
  Prometheus/Grafana exporter-only
- chain template 中的 endpoint override、`TARGET_*` sample、single/mixed
  workload 和 custom RPC 扩展点

通过标准：

- 多磁盘时列出候选磁盘，让用户选择 ledger/accounts，而不是静默猜测。
- accounts 盘作为可选项询问。
- 每个推断值都必须允许用户用编号、id 或手动输入覆盖。
- QPS profile 不能让用户从零配置。必须先展示所选 mode 的框架默认值和
  每个参数含义，询问是否采用默认值；只有用户要求调整时，才逐项询问要
  调整哪个 item，并在调整后展示新的 profile 再确认。
- fake-node 只跳过真实 RPC URL，不跳过资源元数据。
- real-node 必须确认 local/mainnet RPC、进程名、workload 和 sync-health 相关配置。
- `runtime.env` 是每个 job 的最终确认配置，普通用户不手动编辑。
- 用户发现前面输入错误时，Agent 必须能回退或修改对应 workflow state，
  然后重新运行 validator。

### T3. workload 与自定义 RPC 闭环

目标：single、mixed、自定义 RPC 都能由 Agent 引导配置并验证。

必须覆盖：

- 默认 single method。
- 默认 mixed workload。
- mixed 权重总和必须等于 100。
- 用户添加自定义 RPC method。
- 三个或更多参数的 RPC method。
- REST path 参数、JSON-RPC params、`param_spec`。
- fake-node fixture 是否存在。
- per-method attribution 是否能识别新增 method。

通过标准：

- Agent 询问是否使用默认 workload，还是自定义 RPC 和权重。
- 新 RPC method 没有 request/response 样本时，只能生成 onboarding handoff，
  不能声称已支持。
- fake-node fixture、chain template、proxy attribution、report 统计链路都要
  在 handoff 中明确。

### T4. 执行 gate 与长任务恢复

目标：任何执行都必须经过 plan、preflight、smoke、用户确认和后台 job。

必须覆盖：

- 依赖缺失时询问是否安装，并由 Agent 执行安装。
- preflight 失败时阻止压测。
- smoke 失败时阻止真实 benchmark。
- 长时间 benchmark 默认后台执行。
- terminal 断开后，重新启动 Agent 能发现最近 job，并提供 status、logs、
  analyze、resume 或新 benchmark。
- Agent 必须告诉用户 `.agent/jobs/<job_id>/benchmark.log` 路径。
- `follow <job_id>` 必须只退出日志跟踪，不停止 benchmark，不退出 Agent。
- 用户复制日志片段回到 `User>` 后，Agent 必须按 evidence 解释并给下一步。

通过标准：

- 用户不能通过“直接开始”“跳过 preflight”等话术绕过 gate。
- 所有 job 都有 `.agent/jobs/<job_id>/job.json`、`runtime.env` 和
  `artifact_index.json`。
- Agent 给出的分析必须引用具体 artifact 路径。

### T5. 错误恢复与边界测试

目标：用户输入错误或乱序时，Agent 能解释原因并继续推进。

必须覆盖：

- 无效 `LOCAL_RPC_URL`。
- 不存在的磁盘设备。
- mixed 权重不等于 100。
- 缺少主网高度或 sync-health 方法。
- RPC fixture 缺失。
- 用户在 fake-node 流程中突然切换 real-node。
- 用户问与框架无关的问题。
- 用户要求执行危险或未授权操作。

通过标准：

- Agent 不崩溃、不死循环、不静默执行。
- Agent 给出下一步可操作问题，且每次只问一个小组的阻塞信息。

### T6. 二次开发与企业集成

目标：Agent 能为开发者生成可执行、可测试、可审查的开发 handoff。

必须覆盖：

- 新增一个属于现有 6 family 的链。
- 新增一个不属于现有 6 family 的链。
- 新增自定义 RPC method。
- 接入企业 Knowledge Base。
- 接入企业 Agent 平台。

通过标准：

- handoff 必须列出要修改的文件、边界、fixture、smoke、质量门禁、
  文档更新要求和 PR 要求。
- 对新增链或新增 RPC，必须要求更新文档，因为文档也是 Agent 知识来源。

## Live Matrix 要求

`tests/agent_live/agent_intent_smoke_scenarios.json` 是当前最小 live smoke。
`tests/agent_live/agent_product_acceptance_scenarios.json` 是更完整的产品验收场景。

这些文件只是测试 fixture，不参与运行时 intent recognition、routing、
planning 或 workflow execution。运行时自然语言理解必须由 Google ADK 和
配置模型完成；测试场景只验证真实模型输出是否满足产品契约。
产品级前必须扩展到至少以下覆盖：

- 10 个 benchmark 配置场景。
- 8 个错误恢复场景。
- 6 个 workload/custom RPC 场景。
- 6 个 onboarding 场景。
- 4 个 observability/Prometheus/Grafana 场景。
- 4 个 job resume/analyze 场景。
- 4 个企业 KB/Agent 平台集成场景。

每个场景都必须检查：

- 是否调用真实模型。
- 是否出现内部工具名或 scratchpad 泄露。
- 是否绕过 preflight/smoke/approval。
- 是否给出错误配置建议。
- 是否保持语言一致。

## 执行顺序

1. 扩展 live acceptance scenarios，先让失败暴露出来。
2. 按失败类型修 ADK instructions、sub-agent tool surface、tool schema 或 validator。
3. 避免用 sanitizer 继续补业务逻辑；sanitizer 只做输出卫生防护。
4. 每一类修复后跑 unit tests、ADK eval、Docker tests 和 live scenarios。
5. 最后再更新 README 和用户文档，用户文档只描述已验证能力。

## 完成定义

只有当以下条件全部满足，才能称为产品级 Agent：

- 本文档 T1-T6 全部通过。
- Docker 内 unit tests 和 ADK eval 通过。
- 真实模型 live scenarios 通过。
- 至少一次 fake-node 完整闭环通过。
- 至少一次 real-node 或 mock real-node preflight/smoke 流程通过。
- 代码中没有 terminal 业务关键词路由。
- 文档中没有临时计划、旧架构或未验证能力冒充事实。
