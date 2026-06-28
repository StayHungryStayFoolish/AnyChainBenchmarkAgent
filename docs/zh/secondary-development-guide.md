# 二次开发指南

[English](../en/secondary-development-guide.md) | [中文](secondary-development-guide.md)

这份文档面向已经配置好 AnyChain Benchmark Agent，并希望继续扩展能力的使用者。它说明如何集成企业内部 Knowledge Base，如何在现有协议 family 中新增链，如何新增协议 family，如何新增 RPC method，以及必须执行的闭环测试、smoke 测试和 PR 要求。

Agent 可以帮助生成计划、检查当前仓库能力、输出 checklist，但不应该在没有确定性校验和人工确认的情况下，静默合入新的链、协议或 workload method。

## 扩展边界

请保持这些边界清晰：

- 用户意图、多轮问答和 Agent 编排在 `agent/`。
- 本次 job 最终确认的运行变量在 `.agent/jobs/<job_id>/runtime.env`。
- 用户默认配置在 `config/user_config.sh`。
- Agent provider 配置在 `config/agent_config.sh`。
- 链支持在 `config/chains/*.json` 和 `tools/chain_adapters/`。
- fake-node 回放数据在 `tools/fake-node/fixtures/`。
- fake-node family 行为在 `tools/fake-node/configs/` 和 `tools/fake-node/handlers/`。
- 压测执行入口是 `blockchain_node_benchmark.sh`。
- 报告和归档通过运行时路径注册体系生成，并归档到 benchmark result 目录。

不要把链特定逻辑直接写入共享 shell 监控代码。链相关逻辑应该进入 chain template、adapter、fake-node mapping 或 sync-health registry。

## 主要调用链

Agent 发起压测：

```text
用户 prompt
-> ADK root coordinator
-> typed intent path
-> specialized sub-agent
-> deterministic tool and validator gates
-> benchmark plan
-> preflight 与风险检查
-> .agent/jobs/<job_id>/runtime.env
-> blockchain_node_benchmark.sh
-> target generator
-> proxy
-> vegeta
-> monitoring collectors
-> analysis
-> HTML reports 与 archive
```

链和 RPC 运行路径：

```text
config/chains/<chain>.json
-> tools/chain_adapters/
-> tools/target_generator.sh
-> tools/proxy/
-> tools/fake-node/fixtures/<chain>/
-> analysis/per_method_attribution.py
-> visualization/report_generator.py
```

Knowledge Base 路径：

```text
agent_config.sh
-> agent/knowledge/loader.py
-> local repo provider 或 HTTP provider
-> Agent grounding prompt
-> deterministic schema 与 safety checks
-> 用户回答或 benchmark plan
```

## 1. 集成企业 Knowledge Base

适用于企业内部已有 KB，记录支持的链、真实 RPC samples、压测策略或 workload 推荐。

开发位置：

- `config/agent_config.sh`：provider 选择和 endpoint 配置。
- `agent/knowledge/base.py`：provider contract。
- `agent/knowledge/http_provider.py`：通用 HTTP adapter。
- `agent/knowledge/loader.py`：provider 选择。
- `agent/adk_app/instructions.py`：ADK 如何使用 KB evidence，并避免声明未验证能力。
- `agent/adk_app/tools/read_only.py`：暴露 KB search 和本地 capability evidence 的 ADK read-only tools。
- `agent/cli.py`：smoke 命令和集成入口。

基本原则：

- KB 返回内容只能作为 evidence，不能作为可直接执行的代码。
- Agent 仍然必须使用仓库内工具校验 chain template、RPC params、fixtures 和 workload generation。
- 如果 KB evidence 缺失，或者和当前仓库状态冲突，Agent 必须继续追问，或者生成 onboarding plan。

最小 HTTP adapter 行为：

```text
POST /search
GET /chains/{chain}/rpc-methods
GET /chains/{chain}/rpc-samples
POST /workload/suggest
```

验证：

```bash
python3 agent/cli.py knowledge-smoke
python3 -m unittest tests.test_agent_runtime_contract -v
```

PR 要求：

- 文档说明 provider contract。
- 增加 secret/private endpoint 脱敏逻辑。
- 增加 KB 不可用、KB 为空、KB 与仓库冲突的 smoke 测试。
- 不提交企业内部 KB 返回内容。

## 2. 集成企业 Agent 平台

适用于企业希望把 AnyChain Benchmark Agent 作为内部 Agent 平台里的一个工具，而不是只通过终端 chat 使用。

开发位置：

- `agent/cli.py`：JSON CLI 入口。
- `agent/tools/schema.py`：OpenAI-compatible tool catalog。
- `agent/tools/executor.py`：稳定的 named tool execution。
- `config/agent_config.sh`：LLM、Google auth 和可选 KB 默认配置。
- `agent/runners/job_manager.py`：job status、artifact index 和 detached run 生命周期。
- `agent/adk_app/instructions.py`：ADK root instruction。
- `agent/adk_app/tools/`：ADK function-tool wrappers。
- `agent/adk_app/evals/`：无 key ADK package 和 tool-contract checks。

支持的集成模式：

- 人类终端：`./bin/anychain-agent`
- JSON CLI：`python3 agent/cli.py <command>`
- Tool schema 导出：`python3 agent/cli.py tool-schema`
- Named tool call：
  `python3 agent/cli.py tool-call --name <tool> --arguments '<json>'`

常用平台工具：

- `discover_environment`
- `load_capabilities`
- `draft_request`
- `generate_plan`
- `run_preflight`
- `submit_job`
- `get_job_status`
- `tail_job_log`
- `analyze_artifacts`
- `answer_artifact_question`
- `diagnose_artifacts`
- `draft_chain_template`
- `gap_analysis`
- `knowledge_search`

边界：

- 企业平台可以编排工具，但真正执行压测仍然必须遵循 Agent preflight 和 approval 规则。
- 长时间真实压测应该使用 detached/background job mode。
- 平台会话应该持久化 `job_id`、`artifact_index` 和 archive 路径，方便终端或平台会话断开后恢复。
- 密钥必须来自企业 secret manager 或运行时环境变量，不应该写入提交到 git 的配置文件。
- KB evidence 不能替代本地确定性校验。

验证：

```bash
python3 agent/cli.py tool-schema
python3 agent/cli.py tool-call --name load_capabilities
python3 agent/cli.py tool-call --name discover_environment
python3 -m unittest tests.test_agent_runtime_contract -v
```

PR 要求：

- 尽量保持 tool name 和 schema 向后兼容。
- 新增 tool 必须增加测试。
- 文档说明新增参数。
- 不要让企业平台依赖私有本地路径。

## 3. 在现有协议 Family 中新增链

适用于新链可以归入当前六类 family 的情况：

- `jsonrpc`
- `bitcoin_jsonrpc`
- `rest`
- `substrate`
- `tendermint`
- `hedera_dual`

开发位置：

- `config/chains/<chain>.json`
- `config/chain_template.json.bak`
- `tools/chain_adapters/`
- `tools/fake-node/configs/`
- `tools/fake-node/fixtures/<chain>/`
- `docs/en/how-to-add-chain.md`
- `docs/zh/how-to-add-chain.md`

必需 template 字段：

- `chain_type`
- `rpc_url`
- `rpc_methods.single` 或 `rpc_methods.mixed_weighted`
- `param_formats` 或 `param_spec`
- `proxy_extraction`
- `_meta.adapter_family`
- 如果该链需要节点健康检查，还需要 sync-health metadata

验证：

```bash
python3 tools/chain_adapters/cli.py validate-template --chain <chain>
python3 tools/fake-node/check_fixture_coverage.py --json
python3 tools/fake-node/runtime_probe.py --chain <chain>
python3 tools/fake-node/runtime_probe_block_height.py --chain <chain>
```

闭环检查：

```bash
./bin/anychain-agent
```

然后让 Agent 为该链创建 fake-node smoke benchmark，执行 preflight，运行 mock job，并分析生成的 archive。

PR 要求：

- 增加 chain template。
- 为每个 workload method 增加真实 fake-node fixture。
- 用户行为发生变化时，同步更新文档。
- 在 PR body 中给出验证命令和 archive 路径。

## 4. 新增协议 Family

只有在当前六类 family 无法表达该链行为时，才新增协议 family。

开发位置：

- `tools/chain_adapters/<family>.py`
- `tools/chain_adapters/base.py`
- `tools/fake-node/handlers/<family>.go`
- `tools/fake-node/configs/<family>.yaml`
- 如果 handler registry 需要新增入口，修改 `tools/fake-node/main.go`
- `config/chains/<chain>.json`
- `docs/en/how-to-add-chain.md`
- `docs/zh/how-to-add-chain.md`

必须明确的设计点：

- 请求 envelope 和 transport。
- proxy attribution 的 method extraction path。
- 参数 schema 支持方式。
- 响应 fixture 匹配方式。
- 区块高度或 sync-health 解析。
- 是否可以复用现有报告和 per-method attribution。

验证：

```bash
python3 tests/test_chain_adapters.py
python3 tests/test_param_spec.py
python3 tools/chain_adapters/cli.py validate-template --chain <chain>
(cd tools/fake-node && go test ./...)
python3 tools/fake-node/runtime_probe.py --chain <chain>
bash tests/test_full_entrypoint_fake_node_lifecycle_smoke.sh
```

PR 要求：

- 提供一个最小可用 chain template。
- 提供 fake-node handler tests。
- 提供 runtime probe 证据。
- 说明为什么现有六类 family 不足以支持该链。

## 5. 新增 RPC Method

RPC method 支持不只是增加一个 method name。框架需要 request construction、参数样本、fake-node response fixture、proxy attribution 和 report attribution 都按同一个 method identity 对齐。

开发位置：

- `config/chains/<chain>.json`
- `tools/chain_adapters/param_spec.py`
- `tools/fake-node/configs/`
- `tools/fake-node/fixtures/<chain>/`
- `docs/audit/rpc-fixtures/`

简单 method 使用 `param_formats`。如果需要 positional params、object params、REST path params、query params 或 request body，使用 `param_spec`。

三参数 method 示例：

```json
{
  "rpc_methods": {
    "mixed_weighted": [
      {"method": "eth_getBalance", "weight": 40},
      {"method": "eth_blockNumber", "weight": 30},
      {"method": "eth_getStorageAt", "weight": 30}
    ]
  },
  "param_spec": {
    "eth_getStorageAt": {
      "transport": "jsonrpc_list",
      "params": [
        {"source": "address"},
        {"source": "target_storage_slot"},
        {"literal": "latest"}
      ]
    }
  }
}
```

验证：

```bash
python3 tools/chain_adapters/cli.py validate-template --chain <chain>
bash tests/test_target_generator_mixed_weighted.sh
tools/fake-node/record_rpc_fixtures.sh <chain>
python3 tools/fake-node/check_fixture_coverage.py --json
python3 tools/fake-node/runtime_probe.py --chain <chain>
```

报告检查：

- `logs/proxy_method.csv` 包含新增 method。
- per-method CSV 包含 success、error、latency、P50、P90、P99。
- HTML 报告的 per-method attribution 图表展示新增 method。

## 6. 新增或调整 Workload 逻辑

开发位置：

- `config/chains/*.json`
- `tools/target_generator.sh`
- `analysis/per_method_attribution.py`
- `visualization/per_method_visualizer.py`
- `visualization/report_generator.py`

规则：

- `mixed_weighted` 是 mixed 模式按权重生成请求的来源。
- 权重建议总和为 100，便于审计。
- sync-health RPC method 不应该计入 workload method。
- per-method 报告图表只描述压测 workload traffic。

验证：

```bash
bash tests/test_target_generator_mixed_weighted.sh
python3 tests/test_per_method_attribution.py
python3 tests/test_per_method_charts.py
python3 tests/test_per_method_report.py
```

## 7. 闭环测试要求

每个功能扩展都应该证明以下链路：

1. Template validation。
2. Request generation。
3. fake-node fixture coverage。
4. fake-node runtime probe。
5. 如适用，block height / sync-health probe。
6. proxy method attribution。
7. report generation。
8. archive generation。

推荐 smoke sequence：

```bash
python3 tools/chain_adapters/cli.py validate-template --chain all
python3 tools/fake-node/check_fixture_coverage.py --json
python3 tools/fake-node/runtime_probe.py
python3 tools/fake-node/runtime_probe_block_height.py
python3 -m unittest tests.test_agent_runtime_contract -v
python3 tools/check_public_repo_markers.py --root .
git diff --check
```

监控或生命周期相关变更，建议在 Linux 或 Docker 中执行：

```bash
bash tests/test_monitoring_lifecycle_smoke.sh
bash tests/test_monitoring_runtime_contract.sh
bash tests/test_full_entrypoint_fake_node_lifecycle_smoke.sh
```

## 8. PR 要求

提交 PR 前：

- 执行和修改范围匹配的测试。
- 用户可见行为变化需要同步更新中英文文档。
- 不提交 runtime archives、本地 `.agent/` jobs、secrets、API keys、private endpoints、本地机器路径或生成的二进制文件。
- PR title 使用 Conventional Commits。
- 填写 `.github/pull_request_template.md`。

高风险路径需要额外谨慎：

- `blockchain_node_benchmark.sh`
- `monitoring/`
- `tools/proxy/`
- `tools/fake-node/`
- `tools/target_generator.sh`
- `tools/benchmark_archiver.sh`
- `config/chains/`
- `agent/`

完整仓库策略见 [GitHub PR Gate 与分支保护](github-pr-gates.md)。

## Agent 应该如何帮助开发者

对于扩展类需求，Agent 应该：

- 检查当前仓库能力；
- 判断需求属于 KB、现有 family 链、新 family 链、RPC method、workload、monitoring 或 report；
- 生成包含修改文件和验证命令的计划；
- 识别缺失的 params、fixtures 或 sync-health evidence；
- 在写配置或启动长时间压测前要求确认；
- 修改后执行确定性校验；
- 在最终回复中引用具体文件和 archive 路径。

对于不支持或风险较高的需求，Agent 应该生成开发计划，而不是假装框架已经支持。
