# Agent Knowledge And Onboarding Execution Plan

本文档定义 AnyChain Benchmark Agent 的本地知识索引、环境变量推断、unsupported chain 分流、自定义 RPC workload 交互和二次开发 handoff 的执行要求。

## 目标

- Agent 启动后必须尽量通过只读环境检查推断配置草案，减少用户手动理解和填写变量的负担。
- 能确定的变量先展示给用户确认；不能确定的变量，例如多个磁盘对应的 `LEDGER_DEVICE` / `ACCOUNTS_DEVICE`，必须通过多轮选择让用户确认。
- Agent 必须从当前仓库事实生成本地 framework knowledge index，避免只依赖模型记忆。
- 新增 chain、RPC method 或 adapter family 时，必须生成可执行、可闭环验证的二次开发 handoff。
- 二次开发 handoff 必须要求同步更新文档，因为文档也是 Agent 的本地知识来源。

## 本地知识索引

Agent 需要提供一个可重新生成的轻量 JSON 索引，而不是先引入数据库服务。

默认索引位置：

```text
.agent/knowledge/framework_index.json
```

索引内容：

- 当前支持的 chain template 列表。
- 每条链的 adapter family。
- 每条链的 single / mixed / mixed_weighted RPC method。
- 参数配置摘要：`param_formats`、`param_spec` 是否存在。
- sync-health 配置摘要。
- fake-node fixture 覆盖摘要。
- 关键文档路径。
- 关键代码路径。
- 必须执行的验证命令。

索引生成命令：

```bash
python3 agent/cli.py framework-index --output .agent/knowledge/framework_index.json
```

Agent 的 `load_framework_context` 和企业 tool API 可以读取这个索引；如果索引不存在，应从仓库事实动态生成，不能返回过期数据。

## 启动环境推断

Agent 启动时执行只读检查：

- cloud provider / platform：GCP、AWS、Other，GCE/GKE/EC2/EKS/K8S/VM/container。
- metadata：region、zone、machine type。
- host：CPU、memory。
- network：default network interface、driver。
- disks：`lsblk` 候选磁盘、mountpoint、size、label、fstype。

交互要求：

- 可以推断的值展示为配置草案，用户确认或覆盖。
- 多个磁盘或无法判断用途时，列出编号，让用户选择 `LEDGER_DEVICE`。
- 单独询问是否存在 `ACCOUNTS_DEVICE`，如果存在再列出候选让用户选择。
- 不能静默 fallback 到不可信配置。

## Unsupported Chain Flow

当用户输入的 chain 不在 `config/chains` 的 36 条链中：

1. Agent 不得只返回“不支持”。
2. Agent 必须询问该 chain 是否属于现有 6 个 adapter family：
   - `jsonrpc`
   - `rest`
   - `bitcoin_jsonrpc`
   - `substrate`
   - `tendermint`
   - `hedera_dual`
   - `unknown/new family`
3. 如果用户选择现有 family：
   - 询问 RPC methods。
   - 要求官方 RPC 文档、内部 KB 页面或真实 request/response samples。
   - 生成 chain template 草案命令和 handoff。
   - 明确 fake-node fixture、proxy attribution、sync-health、per-method report 的验证要求。
4. 如果用户选择 unknown/new family：
   - 停止直接配置 benchmark。
   - 生成 new-family design handoff。
   - 要求说明当前 6 个 family 为什么不足以表达该 chain。

## Custom RPC Method Flow

用户选择 `single` 后，Agent 必须询问：

- 使用默认 method。
- 选择已有 method。
- 输入自定义 method。

用户选择 `mixed` 后，Agent 必须询问：

- 使用默认 mixed weights。
- 修改已有 method 权重。
- 添加自定义 method 并配置权重。

Agent 必须校验：

- mixed 权重总和等于 100。
- 每个 method 都有参数契约或明确不需要参数。
- fake-node fixture 是否存在或需要录制。
- proxy attribution 是否能识别 method name。

## 二次开发 Handoff

二次开发 handoff 必须覆盖：

- 修改哪些文件。
- 哪些文件只在必要时修改。
- 新增 chain 属于现有 family 时的 config-only 路径。
- 新增 family 时的 adapter/fake-node handler/config 路径。
- 新增 RPC method 时的参数、fixture、proxy、report 验证路径。
- 必须更新的文档。
- 必须执行的 smoke/CI 命令。
- 缺少官方文档、KB evidence 或真实 response sample 时，必须停止并向用户索要资料。

## 完成标准

- `python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract`
- `python3 agent/cli.py framework-index --output /tmp/framework_index.json`
- `python3 agent/cli.py tool-call --name load_framework_index`
- unsupported chain 能进入 onboarding flow。
- supported chain 仍能进入 benchmark configuration flow。
- custom RPC method / mixed weights 仍能进入 smoke gate。
