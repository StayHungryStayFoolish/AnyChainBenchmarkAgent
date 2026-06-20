# AnyChain Benchmark Agent

[English](README.md) | [中文](README_ZH.md)

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License](https://img.shields.io/badge/License-Commercial-green.svg)](COMMERCIAL.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Shell Script](https://img.shields.io/badge/shell-bash-green.svg)](https://www.gnu.org/software/bash/)

这是一个面向生产环境的多链节点 benchmark Agent 与压测框架，用于分析节点
QPS、延迟、瓶颈、同步健康和每个 RPC method 的表现。Agent 可以把用户测试目标
转换成可验证的 plan，执行 preflight，提交 job，跟踪 artifact，并基于证据解释结果。

压测执行面仍然是确定性的：Vegeta、RPC proxy、监控 collector、fake-node、报告生成
和归档是事实来源。LLM 是可选能力，只能生成结构化 request 草案，不能直接执行命令。

## 报告预览

运行框架前，可以先查看生成报告的 PDF 预览：

- [中文 PDF 报告预览](docs/zh/performance_report_zh.pdf)
- [英文 PDF 报告预览](docs/en/performance_report_en.pdf)

## 能做什么

### Agent 智能能力

- 将自然语言 benchmark 目标转换成结构化、可验证的 plan。
- 自动检测本地环境，只询问无法安全推断的缺失值。
- 通过 Agent checklist 引导用户补齐缺失配置，而不是要求用户先理解所有变量。
- 进行风险评分、preflight，并在真实 benchmark 前要求用户明确确认。
- 真实 benchmark 默认以 detached/background 方式执行，长时间压测不会因为终端断开而停止。
- 使用同一个 output-dir 重新打开 Agent 时，会自动恢复最近一次 job 状态。
- 基于当前仓库、job artifact、报告和可选企业 Knowledge Base 回答框架与结果问题。
- 为新增链、RPC method 和 weighted workload 生成 onboarding plan 和保守的 chain template 草案。

### Benchmark Tools 能力

- 支持 36 个 chain template，覆盖 6 个 adapter family。
- 从 `config/chains/*.json` 生成 single 或 weighted mixed RPC workload。
- 通过 `rpc_methods`、`param_formats`、可选 `param_spec`、REST path 绑定和
  fake-node fixture 支持自定义 RPC method。
- 记录每个 method 的状态、成功/失败次数和 P50/P90/P99 延迟。
- 监控 CPU、内存、磁盘、网络、cgroup、同步健康和监控系统自身开销。
- 生成 HTML 报告并归档每次运行。
- 通过只读 exporter 可选接入 Prometheus/Grafana。
- 提供 JSON CLI tools、OpenAI-compatible tool schema 和稳定的 `tool-call`
  入口，方便企业 Agent 平台集成。

## Agent 如何工作

```text
prompt 或 request
  -> intent routing
  -> request draft
  -> read-only discovery
  -> benchmark plan
  -> risk score 和 preflight
  -> approved job
  -> artifact index
  -> evidence-based analysis
```

Agent 是有边界的：

- 只基于本地 docs 和当前仓库状态回答框架问题。
- 从当前文件读取 chain/RPC/fake-node 能力，不依赖模型记忆。
- 对未支持链或 RPC method 生成 onboarding plan。
- 只有经过 approval 后才运行 allowlisted benchmark command。
- Agent 生成的运行配置写入 job-local `runtime.env`，不会修改
  `config/user_config.sh`。

## 配置模型

大多数用户启动 Agent 前只需要关注一个文件：

```text
config/agent_config.sh
```

这个文件只配置 Agent 自身：LLM provider、模型、Vertex/OpenAI 认证、上下文压缩和
可选企业 Knowledge Base 集成。每个变量后面都有注释。

底层 benchmark engine 的默认配置仍然在：

```text
config/user_config.sh
```

用户不需要一开始理解所有 benchmark 变量。启动 Agent 后先运行 `doctor`，再描述测试
目标，Agent 会告诉你还缺哪些必需值。

Agent 提交 job 时会生成：

```text
<agent-output-dir>/jobs/<job_id>/runtime.env
```

`runtime.env` 是这一次 job 的最终配置快照。通过 Agent 启动 benchmark 时，它的优先级
高于 `config/user_config.sh`。用户不需要也不应该手动编辑它；它是报告和分析时证明
本次运行使用了哪些变量的证据。

终端 Agent 默认 output-dir 是 `.agent/chat`，因此默认 runtime 文件路径是
`.agent/chat/jobs/<job_id>/runtime.env`。低阶 `python3 agent/cli.py submit`
默认使用 `.agent/jobs`，除非显式传入 `--jobs-dir`。

## 5 分钟快速开始

这是在终端里使用 AnyChain Benchmark Agent 的最快路径。不配置 LLM 时，Agent 仍可
使用确定性解析和本地仓库能力回答问题；如果希望启用模型辅助 planning，需要先在
`config/agent_config.sh` 中配置 provider、模型和认证方式。

克隆仓库：

```bash
git clone git@github.com:StayHungryStayFoolish/AnyChainBenchmarkAgent.git
cd AnyChainBenchmarkAgent
```

只检查依赖，不修改宿主机：

```bash
bash scripts/install_deps.sh --check
```

在 `config/agent_config.sh` 中配置持久化的 Agent 参数。确定性/离线模式保持默认值：

```bash
LLM_PROVIDER="fake"                       # fake | gemini | claude | openai
LLM_MODEL="fake"
```

如果需要模型辅助 planning，配置一个真实 provider。Gemini、Claude、OpenAI 都支持
API Key 模式；Gemini 或 Claude 也可以通过 Vertex AI 使用 Google service account。

```bash
LLM_PROVIDER="gemini"
LLM_MODEL="gemini-3.1-pro"
LLM_AUTH_MODE="api_key"                   # api_key | google_adc | attached_service_account | service_account_impersonation | service_account_file
GEMINI_API_KEY=""                         # Gemini API-key 模式必填，也可用 GOOGLE_API_KEY
ANTHROPIC_API_KEY=""                      # Claude API-key 模式必填
OPENAI_API_KEY=""                         # OpenAI 必填
GOOGLE_CLOUD_PROJECT=""                   # Google service-account 模式必填
GOOGLE_CLOUD_LOCATION="us-central1"       # Vertex AI location/region
GOOGLE_SERVICE_ACCOUNT_EMAIL=""           # service_account_impersonation 时必填
GOOGLE_APPLICATION_CREDENTIALS=""         # 可选 JSON key fallback
```

chain、RPC URL、磁盘、机器类型等 benchmark 信息可以先不配置。Agent 会自动发现能
发现的信息，并在真实运行前提示你补充缺少的必需值。

启动 Agent 终端会话：

```bash
./bin/anychain-agent
```

然后直接和它对话。`anychain>` 后面的内容是你在 Agent 交互窗口中输入的消息。

```text
anychain> doctor
# 只读检查：依赖、cloud/deployment、LLM 配置、Knowledge Base 开关、
# 支持的链/RPC method，以及明显缺失的配置。

anychain> Create a Solana fake-node smoke benchmark at 1 QPS
# 用自然语言描述测试目标。Agent 会生成 request，发现环境，生成 plan，
# 并记录还缺哪些必需值。

anychain> plan
# 查看当前计划：chain、RPC mode、fake-node/real-node、QPS 策略、命令、
# 必需输入、生成文件和下一步。

anychain> checklist
# 查看下一项缺失配置。可以直接回复值，也可以使用 `answer <value>` 明确填写。

anychain> preflight
# 执行运行前校验，发现缺失 chain template、必需变量、fake-node 支持、
# 输出目录权限和配置 checklist 未完成项。

anychain> run mock
# 只验证 Agent job 生命周期，不运行 Vegeta 压测流量。
# 会生成 job metadata、artifact_index 和 runtime.env。

anychain> status
# 查看最近一次 job 状态。

anychain> analyze
# 基于 artifact 分析结果，并给出 PASS/WARNING/FAIL/INCONCLUSIVE 和证据路径。

anychain> qa What evidence was generated?
# 对结果继续提问，例如报告在哪里、哪些 CSV 有数据、runtime.env 是什么。

anychain> trace
# 查看最近的 Agent workflow 决策：intent、workflow、prompt bundle、
# deterministic tools、生成文件和下一步。
```

在新环境中建议先输入 `doctor`。它会以只读方式检查 cloud/deployment 识别结果、
必需依赖、LLM/Vertex 配置和当前框架能力覆盖情况。Agent 启动时会显示当前模式：
LLM 配置有效时自动进入 AI-assisted 模式，否则进入 deterministic/offline 模式。

也可以使用一句 prompt 运行：

```bash
./bin/anychain-agent \
  --prompt "Create a Solana fake-node smoke benchmark at 1 QPS"
```

在会话中，`run mock` 会提交 lifecycle-only Agent job，用于本地验证 Agent 生命周期。
真实 benchmark 执行需要在 review plan 和 runbook 后，通过 `yes run` 明确确认。
长会话会自动压缩上下文；也可以输入 `compact` 写入 `.agent/chat/memory.json`。

真实 benchmark 默认以 detached/background 方式执行。benchmark worker 会在 Agent
终端断开后继续运行，状态写入 `<agent-output-dir>/jobs/<job_id>/job.json`，输出写入
`<agent-output-dir>/jobs/<job_id>/benchmark.log`。如果希望压测绑定当前终端，可以在
`yes run` 前输入 `run in foreground`。

如果会话断开，使用同一个 output-dir 重新启动 Agent：

```bash
./bin/anychain-agent
# 如果之前使用了自定义目录：
./bin/anychain-agent --output-dir /path/to/previous-agent-output
```

Agent 启动时会扫描该目录，恢复最近一次 job 状态，并提示下一步命令。

你也可以随时询问 Agent 当前框架能力：

```bash
./bin/anychain-agent --prompt "How many chains and RPC methods are supported?"
./bin/anychain-agent --prompt "How do I add a custom RPC method with three params?"
```

高级子命令仍然保留给 CI 和自动化：

```bash
python3 agent/cli.py --help
```

修改项目后，可以运行离线 Agent contract 测试：

```bash
python3 -m unittest tests.test_agent_runtime_contract -v
```

## 入口

大多数用户只需要启动一个命令：

```bash
./bin/anychain-agent
```

其他入口用于自动化：

- `python3 agent/cli.py ...`：CI、测试或需要 JSON 输入/输出的企业 Agent 平台。
- `./blockchain_node_benchmark.sh`：底层压测执行引擎。Agent 在计划确认后会调用它。
  直接使用时要求配置已经存在。

## 运行本地 Fake-Node Benchmark

如果你希望在没有生产节点的情况下通过 Agent 验证 benchmark 流程，启动
`./bin/anychain-agent` 后输入：

```text
anychain> doctor
anychain> Create a Solana fake-node smoke benchmark at 1 QPS
anychain> preflight
anychain> run mock
anychain> analyze
```

`run mock` 只验证 Agent 生命周期，不会发送真实 benchmark 流量。如果希望让 Agent
规划并执行真实 fake-node benchmark engine，可以输入：

```text
anychain> Create a Solana fake-node quick benchmark and run the real benchmark engine
anychain> plan
anychain> preflight
anychain> yes run
```

执行 `yes run` 前请先 review 生成的 runbook；Agent 只会执行 allowlisted benchmark
command。

## 连接真实节点运行

不要从手工编辑所有 benchmark 变量开始。先启动 Agent，让它检查当前机器或容器环境，
然后只回答它无法安全推断的缺失值。Agent 会把最终确认的值写入本次 job 的
`runtime.env`；普通用户不要手动编辑这个文件。

```bash
./bin/anychain-agent
```

真实节点对话示例：

```text
anychain> doctor
anychain> Test my Solana node at http://your-node-rpc:8899 with a quick single-method benchmark
anychain> plan
anychain> preflight
anychain> yes run
anychain> analyze
```

在 planning/preflight 阶段，Agent 会检查 chain、RPC mode、本地 RPC URL、节点进程名、
ledger/data 磁盘、磁盘 IOPS/吞吐基线、网络带宽和输出路径。如果 discovery 无法可靠
识别某个值，Agent 会把它标记为缺失，而不是猜测。高级用户仍然可以在
`config/user_config.sh` 中设置默认值，但 Agent 为本次 job 确认并写入 `runtime.env`
的值优先级更高。

最重要的输出文件：

```text
blockchain-node-benchmark-result/current/reports/performance_report_*.html
blockchain-node-benchmark-result/current/logs/proxy_method.csv
blockchain-node-benchmark-result/current/logs/performance_latest.csv
blockchain-node-benchmark-result/archives/<run-id>/test_summary.json
```

## 必需值

Agent 会检查三层配置：

- **Agent checklist**：`config/agent_config.sh`，检查 LLM provider、模型、
  Vertex/OpenAI 认证、上下文压缩和可选 Knowledge Base。
- **Benchmark checklist**：`plan` 和 `preflight` 检查 chain、RPC mode、真实节点
  `LOCAL_RPC_URL`、进程名、ledger 磁盘、磁盘基线和网络带宽等运行值。
- **Advanced checklist**：监控频率、瓶颈阈值、同步健康阈值、Prometheus/Grafana、
  Kubernetes 和 runtime paths。大多数用户不需要修改。

`plan` 和 `preflight` 会在 `yes run` 前暴露缺失的必需值；高级配置保留给需要调优的
operator。

## LLM Provider

Agent 不依赖 LLM 也能工作。开启 LLM 后，模型只用于 request drafting 和 intent
classification，执行前仍会经过确定性校验。

支持的 provider 和认证方式：

- `gemini`：Gemini API key，或通过 Vertex AI 使用 Google 认证。
- `claude`：Anthropic API key，或通过 Vertex AI 使用 Google 认证。
- `openai`：OpenAI API key。
- `fake`：离线 protocol smoke provider，用于测试。

请在 `config/agent_config.sh` 中持久化配置这些变量；`./bin/anychain-agent` 启动时会
自动加载该文件，临时测试时仍可用环境变量覆盖。

Gemini API key：

```bash
LLM_PROVIDER="gemini"
LLM_MODEL="gemini-3.1-pro"
LLM_AUTH_MODE="api_key"
GEMINI_API_KEY="AIza..."
```

Claude API key：

```bash
LLM_PROVIDER="claude"
LLM_MODEL="claude-opus-4-8"
LLM_AUTH_MODE="api_key"
ANTHROPIC_API_KEY="sk-ant-..."
```

通过 Vertex AI 和 service-account impersonation 使用 Gemini 或 Claude：

```bash
LLM_PROVIDER="gemini"
LLM_MODEL="gemini-3.1-pro"
LLM_AUTH_MODE="service_account_impersonation"
GOOGLE_CLOUD_PROJECT="your-project"
GOOGLE_CLOUD_LOCATION="us-central1"
GOOGLE_SERVICE_ACCOUNT_EMAIL="benchmark-agent@your-project.iam.gserviceaccount.com"
```

如果使用 OpenAI：

```bash
LLM_PROVIDER="openai"
LLM_MODEL="gpt-5.5"
OPENAI_API_KEY="sk-..."
```

不调用模型，只检查配置：

```bash
python3 agent/cli.py llm-config
```

不需要凭据的离线 LLM protocol smoke：

```bash
python3 agent/cli.py llm-smoke --mock
```

配置好凭据后再运行真实 provider smoke：

```bash
python3 agent/cli.py llm-smoke --prompt 'Return JSON only: {"ok": true}'
```

## 传统 Benchmark 入口

你仍然可以直接运行压测引擎。

在 `config/user_config.sh` 中配置最少运行参数：

```bash
BLOCKCHAIN_NODE="solana"
RPC_MODE="single"
LOCAL_RPC_URL="http://localhost:8899"
MAINNET_RPC_URL=""

BLOCKCHAIN_PROCESS_NAMES=("agave-validator" "solana-validator" "validator")

CLOUD_PROVIDER="gcp"
CLOUD_REGION="us-central1"
MACHINE_TYPE="c3-standard-22"
LEDGER_DEVICE="sdb"
DATA_VOL_TYPE="hyperdisk-extreme"
DATA_VOL_MAX_IOPS="30000"
DATA_VOL_MAX_THROUGHPUT="700"
NETWORK_MAX_BANDWIDTH_GBPS=25
```

在 VM 或裸机上运行 quick benchmark：

```bash
./blockchain_node_benchmark.sh --quick
```

本地 fake-node 闭环建议优先使用
[运行本地 Fake-Node Benchmark](#运行本地-fake-node-benchmark) 中的 Agent 对话。
直接调用 fake-node engine 的高级命令见
[使用 fake-node 进行本地闭环测试](docs/zh/local-closed-loop-testing.md)。

如果节点部署在 Kubernetes 中，请先部署 collector：

```bash
deploy/k8s/validate.sh --preflight
kubectl apply -f deploy/k8s/
kubectl rollout status -n blockchain-bench ds/blockchain-bench-collector
deploy/k8s/validate.sh --post-deploy
```

然后在选定的 runner 上使用同一份 `config/user_config.sh` 运行 benchmark。

## 企业 Agent 平台集成

本项目可以通过两种方式嵌入企业内部 Agent 平台：

- **终端模式**：在受控 shell 中运行 `./bin/anychain-agent`。
- **程序化模式**：调用 `python3 agent/cli.py` 子命令，通过 JSON 交换数据。
  常用命令包括 `doctor`、`capabilities`、`draft-request`、`plan`、`preflight`、
  `submit`、`status`、`analyze` 和 `artifact-qa`。
- **工具 schema 模式**：调用 `python3 agent/cli.py tool-schema`，导出
  OpenAI-compatible function-tool schema，供企业 Agent 编排平台接入。
- **工具调用模式**：调用 `python3 agent/cli.py tool-call --name <tool> --arguments '<json>'`，
  让企业平台通过一个稳定命令执行指定 Agent tool。

企业环境建议在运行镜像或部署 profile 中配置一次 `config/agent_config.sh`。密钥应由
企业 secret manager 注入环境变量，不要写入 git。

Knowledge Base 集成默认关闭：

```bash
AGENT_KNOWLEDGE_PROVIDER="disabled"       # disabled | noop | http | custom
AGENT_KNOWLEDGE_PROVIDER_MODULE=""        # example: my_company.anychain_kb:Provider
AGENT_KNOWLEDGE_BASE_URL=""               # provider=http 时必填
AGENT_KNOWLEDGE_AUTH_REF=""
```

内置 Agent 已经可以基于仓库状态回答：chain template、fake-node fixture、docs、
artifact 和运行历史。只有企业需要私有节点样本、内部 RPC 证据、事故历史或公司内部
workload 建议时，才需要启用自定义 Knowledge Base。

通用 HTTP KB/RAG 服务可以使用 `AGENT_KNOWLEDGE_PROVIDER=http`。验证命令：

```bash
python3 agent/cli.py knowledge-smoke --query "solana rpc methods" --chain solana
```

## 报告与 Artifact

当前运行文件写入 runtime `current/` 目录，最终结果会在运行结束后归档。

报告预览：
[中文 PDF](docs/zh/performance_report_zh.pdf) |
[英文 PDF](docs/en/performance_report_en.pdf)

关键 artifact：

- `current/reports/performance_report_*.html`
- `current/logs/proxy_method.csv`
- `current/logs/performance_latest.csv`
- `archives/<run-id>/test_summary.json`
- `<agent-output-dir>/jobs/<job_id>/artifact_index.json`
- `<agent-output-dir>/jobs/<job_id>/runtime.env`：Agent 为该 job 生成的最终配置快照，用户不要手动编辑。
- `.agent/chat/workflow_trace.jsonl`：Agent 的意图识别、workflow 分流、prompt bundle、
  工具调用、artifact 路径和 next actions 记录。

终端 Agent 的 `<agent-output-dir>` 默认是 `.agent/chat`。低阶 CLI job 命令默认使用
`.agent`，除非通过 `--jobs-dir` 指向其他目录。

## 可选 Prometheus/Grafana

Prometheus/Grafana 默认关闭。在 `config/user_config.sh` 中开启：

```bash
OBSERVABILITY_STACK_ENABLED=true
OBSERVABILITY_STACK_AUTO_STOP=true
OBSERVABILITY_STACK_MODE=local   # local | exporter
EXPORTER_PORT=9108
PROMETHEUS_PORT=9091
GRAFANA_PORT=3001
```

如果你已有 Prometheus/Grafana，使用 `OBSERVABILITY_STACK_MODE=exporter`，只让框架
暴露 scrape endpoint。

## 扩展链或 RPC Method

修改模板前，先用 Agent 检查缺口：

```bash
python3 agent/cli.py gap-analysis \
  --chain solana \
  --method getBalance \
  --method customMethod
```

生成插件化 onboarding package：

```bash
python3 agent/cli.py onboarding-plan \
  --chain foochain \
  --adapter-family jsonrpc \
  --method foo_getBalance \
  --method foo_getBlock
```

对于未支持链，Agent 会生成 onboarding plan，而不是自动修改代码。通常流程是：

1. 基于模板新增 `config/chains/<chain>.json`。
2. 选择 `_meta.adapter_family`。
3. 配置 `rpc_methods.single` 和 `rpc_methods.mixed_weighted`。
4. 添加 `param_formats` 或 `param_spec`。
5. 添加 `proxy_extraction` 规则。
6. 录制 fake-node fixture。
7. 运行 preflight 和 fake-node 闭环测试。

## Reference 文档

- [配置指南](config/README.md)
- [Agent 控制平面](agent/README.md)
- [完整框架 Reference](docs/zh/framework-reference.md)
- [框架流程与数据生命周期](docs/zh/framework-flow.md)
- [模块说明](docs/zh/module-guide.md)
- [如何新增区块链或 RPC Method](docs/zh/how-to-add-chain.md)
- [使用 fake-node 进行本地闭环测试](docs/zh/local-closed-loop-testing.md)
- [二次开发指南](docs/zh/secondary-development-guide.md)
- [GitHub PR Gate 与分支保护](docs/zh/github-pr-gates.md)
- [Prometheus / Grafana Observability](deploy/observability/README.md)
- [Kubernetes Collector](deploy/k8s/README.md)

## License

本项目采用双许可证：

- 开源使用：AGPL-3.0-or-later，见 [LICENSE](LICENSE)。
- 商业/专有/内部使用：见 [COMMERCIAL.md](COMMERCIAL.md)。
