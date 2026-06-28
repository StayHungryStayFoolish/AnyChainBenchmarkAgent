# AnyChain Benchmark Agent

[English](README.md) | [中文](README_ZH.md)

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License](https://img.shields.io/badge/License-Commercial-green.svg)](COMMERCIAL.md)
[![Benchmark Python 3.8+](https://img.shields.io/badge/benchmark_python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ADK Python 3.10+](https://img.shields.io/badge/adk_python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Shell Script](https://img.shields.io/badge/shell-bash-green.svg)](https://www.gnu.org/software/bash/)

这是一个面向生产环境的多链节点 benchmark Agent 与压测框架，用于分析节点
QPS、延迟、瓶颈、同步健康和每个 RPC method 的表现。Agent 可以把用户测试目标
转换成可验证的 plan，执行 preflight，提交 job，跟踪 artifact，并基于证据解释结果。

压测执行面仍然是确定性的：Vegeta、RPC proxy、监控 collector、fake-node、报告生成
和归档是事实来源。面向用户的 Agent 运行时基于 Google ADK，需要配置真实模型；
无凭据的离线检查只用于 CI 和开发测试。模型只能通过工具生成结构化 request、plan
和解释，不能直接执行命令。
Agent 必须使用 Google ADK 做自然语言理解、typed intent、专用 sub-agent 委派和
tool orchestration；确定性工具和 validator 负责最终执行门禁。

## 目录

1. [概览](#概览)
2. [开始使用](#开始使用)
3. [使用 Agent](#使用-agent)
4. [运行 Benchmark](#运行-benchmark)
5. [配置参考](#配置参考)
6. [集成与运维](#集成与运维)
7. [扩展框架](#扩展框架)
8. [Reference 文档](#reference-文档)
9. [License](#license)

## 概览

### 报告预览

运行框架前，可以先查看生成报告的 PDF 预览：

- [中文 PDF 报告预览](docs/zh/performance_report_zh.pdf)

### 能做什么

#### Agent 智能能力

- 将自然语言 benchmark 目标转换成结构化、可验证的 plan。
- 自动检测本地环境，只询问无法安全推断的缺失值。
- 通过 Agent checklist 引导用户补齐缺失配置，而不是要求用户先理解所有变量。
- 使用 ADK multi-agent orchestration，并通过确定性 tool 和 validator gate 约束执行。
- 进行风险评分、preflight，并在真实 benchmark 前要求用户明确确认。
- 真实 benchmark 默认以 detached/background 方式执行，长时间压测不会因为终端断开而停止。
- 使用同一个 output-dir 重新打开 Agent 时，会自动恢复最近一次 job 状态。
- 基于当前仓库、job artifact、报告和可选企业 Knowledge Base 回答框架与结果问题。
- 为新增链、RPC method 和 weighted workload 生成 onboarding plan 和保守的 chain template 草案。

#### Benchmark Tools 能力

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

### Agent 如何工作

```text
prompt 或 request
  -> AnyChain terminal shell，只负责稳定 I/O
  -> ADK root coordinator
  -> typed intent path
  -> specialized sub-agent delegation
  -> deterministic tool and validator gates
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

## 开始使用

请按下面步骤顺序执行。先配置 Agent；benchmark 变量可以在 Agent 对话中由系统推断，
再由用户确认。

### 1. 下载仓库

```bash
git clone git@github.com:StayHungryStayFoolish/AnyChainBenchmarkAgent.git
cd AnyChainBenchmarkAgent
```

### 2. 安装 Agent Runtime

先安装隔离的 ADK 终端运行环境。这个脚本只为 AnyChain Benchmark Agent 创建环境，
不要把 ADK 安装到生产区块链节点使用的 Python 环境中：

```bash
bash scripts/install_agent_deps.sh --yes
```

如果用户跳过这一步并直接启动交互式 Agent，启动器会先检查终端必需依赖。缺少
`prompt-toolkit` 时，Agent 会先请求用户确认，然后自动运行
`scripts/install_agent_deps.sh --yes`。它不会静默回退到 Python `input()`，因为可靠的
Ctrl+C、中文输入和宽字符删除能力是 Agent 终端的基础要求。

如果宿主机没有 `python3.11`，可以使用任意 Python 3.10+ 解释器。底层 benchmark
engine 的非 Agent 自动化仍可使用较旧 Python，但 ADK Agent 运行时需要 Python 3.10+。
启动脚本会自动优先使用 `.venv-adk/bin/python`，所以用户不需要先手动 activate venv
再运行 `./bin/anychain-agent`。

普通 Agent 使用路径不要求用户先手动安装 benchmark engine 依赖。用户只需要先安装
Agent runtime、配置 LLM，然后进入 Agent。Agent 会检查 benchmark 依赖，并在说明将要
安装的内容后请求用户授权；用户确认后，Agent 会通过受控的 `install_dependencies`
工具调用 `scripts/install_deps.sh --yes`。直接运行 `scripts/install_deps.sh` 主要保留给
CI、Docker 镜像和非 Agent 自动化场景。

### 3. 配置 Agent

选择一种配置方式。

#### 方式 A：AI 自动配置

如果你希望让另一个 AI 帮你配置、运行或排查 AnyChain Benchmark Agent，请先把这些
文件交给它阅读：

```text
AGENTS.md
README_ZH.md
config/agent_config.sh
agent/README.md
docs/zh/anychain-agent-ai-work-gate.md
```

`AGENTS.md` 是给 AI 助手的快速接手文档，里面说明了如何安全配置 provider
凭据、真实密钥应该放在哪里、应该执行哪些验证命令，以及哪些 gate 不能绕过。
真实 API key、ADC 设置、Vertex 设置和本地 provider 选择都应该写入
`config/agent_config.local.sh`，该文件已被 git ignore。

如果另一个 AI 需要修改代码，还必须先阅读 `AI_CODING_GUIDE.md`。如果只是帮助用户
完成配置和启动，通常先读 `AGENTS.md` 和本 README 就足够开始。

#### 方式 B：手动配置

如果你希望自己配置，先从这个文件开始：

```text
config/agent_config.sh
```

这个文件只配置 Agent 自身：LLM provider、模型、Vertex/OpenAI 认证、上下文压缩和
可选企业 Knowledge Base 集成。每个变量后面都有注释。

真实密钥和本地 provider 选择建议写入：

```text
config/agent_config.local.sh
```

该文件已被 git ignore。另一个 AI 帮助用户配置时，也应该把真实 API key、ADC/Vertex
本地设置写入这个 local 文件，而不是修改仓库默认配置并提交。

启动自然语言会话前，至少配置一个真实 provider。API Key 模式支持 Gemini、`claude`、
OpenAI 和 DeepSeek；Google service-account 模式支持通过 Vertex AI 使用 Gemini 或
`claude`。

```bash
LLM_PROVIDER="gemini"
LLM_MODEL="gemini-3.1-pro"
LLM_AUTH_MODE="api_key"                   # api_key | google_adc | attached_service_account | service_account_impersonation | service_account_file
GEMINI_API_KEY=""                         # Gemini API-key 模式必填，也可用 GOOGLE_API_KEY
ANTHROPIC_API_KEY=""                      # `claude` API-key 模式必填
OPENAI_API_KEY=""                         # OpenAI 必填
DEEPSEEK_API_KEY=""                       # DeepSeek 必填
GOOGLE_CLOUD_PROJECT=""                   # Google service-account 模式必填
GOOGLE_CLOUD_LOCATION="global"           # Vertex AI location/region
GOOGLE_SERVICE_ACCOUNT_EMAIL=""           # service_account_impersonation 时必填
GOOGLE_APPLICATION_CREDENTIALS=""         # 可选 JSON key fallback
```

选择一种认证路径：

- Gemini API key：设置 `LLM_PROVIDER=gemini`、`LLM_AUTH_MODE=api_key`，并填写
  `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`。
- `claude` API key：设置 `LLM_PROVIDER=claude`、`LLM_AUTH_MODE=api_key`，并填写
  `ANTHROPIC_API_KEY`。
- OpenAI API key：设置 `LLM_PROVIDER=openai`、`LLM_AUTH_MODE=api_key`，并填写
  `OPENAI_API_KEY`。
- DeepSeek API key：设置 `LLM_PROVIDER=deepseek`、`LLM_AUTH_MODE=api_key`，并填写
  `DEEPSEEK_API_KEY`。
- Google Vertex AI：设置 `LLM_PROVIDER=gemini` 或 `LLM_PROVIDER=claude`，填写
  `GOOGLE_CLOUD_PROJECT` 和 `GOOGLE_CLOUD_LOCATION`，然后选择 `google_adc`、
  `attached_service_account`、`service_account_impersonation` 或
  `service_account_file`。

Web research 受 provider 限制。只有当 Agent 使用 Gemini-family 模型，并且 Gemini /
Google 认证有效时，ADK `google_search` 才会启用。`claude` on Vertex、DeepSeek、
OpenAI 和 `claude` API-key 模式不会启用 ADK `google_search`；这些模式下，Agent 会使用
仓库事实、可选企业 KB 证据，或要求用户提供官方文档和 request/response 样本。

Google Cloud CLI 只在本地 ADC 工作流中需要，例如 `LLM_AUTH_MODE=google_adc`，或者
当前机器需要先创建 ADC 再进行 service-account impersonation。Agent 可以通过
`doctor` 检查 `gcloud` 和本地 ADC 文件是否存在；在用户明确确认后，也可以帮你安装
Google Cloud CLI：

```bash
bash scripts/install_agent_deps.sh --yes --with-gcloud
```

如果使用 `google_adc`，安装完成后还需要创建本地 ADC 凭据：

```bash
gcloud auth application-default login
```

如果 Agent 运行在已经绑定 service account 的 GCE/GKE/Cloud Run 上，并且该身份已经有
Vertex AI 权限，运行时认证不要求安装 `gcloud`。

#### Benchmark 配置模型

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

Agent 启动的 job 默认写入 `.agent/jobs`。低阶 `python3 agent/cli.py submit`
也使用同一位置，除非显式传入 `--jobs-dir`。

### 4. 验证 Agent 配置

启动交互会话前先验证 Agent 和 LLM 配置：

```bash
python3 agent/cli.py adk-status
python3 agent/cli.py llm-config
```

chain、RPC URL、磁盘、机器类型等 benchmark 信息可以先不配置。Agent 会自动发现能
发现的信息，并在真实运行前提示你补充缺少的必需值。

## 使用 Agent

### 5. 启动 Agent

启动 Agent。该命令会打开 AnyChain 产品终端。底层使用 Google ADK runtime
能力，但不会把原始 `adk run` 终端 UI 暴露给用户：

```bash
./bin/anychain-agent
```

然后在 `User>` 提示符里直接输入你的需求。Agent 会以 `Agent>` 回复，响应语言会跟随
用户输入语言，并按一项一项确认的方式检查环境、准备 benchmark run、生成 plan、
执行 preflight、请求确认、运行 smoke，并只提交经过确认的 job。

```text
User> doctor
Agent> ...只读检查环境和依赖，如果缺少依赖，会先询问是否允许安装...

User> 我要压测一个区块链节点
Agent> ...询问要测试哪个链...

User> solana
Agent> ...询问 [1] fake-node 闭环测试 / [2] real-node 真实节点...

User> 1
Agent> ...逐项确认 host、磁盘、网络、RPC mode、workload 和可选 observability...

User> 使用 mixed，getSlot 70%，getBlockHeight 30%。
Agent> ...生成 benchmark plan，执行 preflight，然后在 smoke 和正式 benchmark 前再次确认...
```

在新环境中建议先输入 `doctor`。它会以只读方式检查 cloud/deployment 识别结果、
必需依赖、LLM/Vertex 配置、Knowledge Base 配置和当前框架能力覆盖情况。
如果缺少 benchmark 依赖，Agent 应该先说明计划安装的内容，并在用户明确授权后再执行安装。

真实 benchmark 执行仍然需要通过确认门：模型输出不会被直接执行，Agent 必须先通过
preflight 和 smoke，并在用户确认后调用受控工具。

真实 benchmark 默认以 detached/background 方式执行。benchmark worker 会在 Agent
终端断开后继续运行，状态写入 `<agent-output-dir>/jobs/<job_id>/job.json`，输出写入
`<agent-output-dir>/jobs/<job_id>/benchmark.log`。如果希望压测绑定当前终端，可以在
`yes run` 前输入 `run in foreground`。

如果会话断开，使用同一个 jobs 目录重新启动 Agent：

```bash
./bin/anychain-agent
```

重新启动后，让 Agent 检查 `.agent/jobs` 中的最新 job。它可以通过文件化 job 工具恢复
status、logs、runtime.env 和 artifact 路径。

你也可以随时询问 Agent 当前框架能力：

```text
User> 当前支持多少个链和 RPC method？
User> 如何新增一个带 3 个参数的自定义 RPC method？
```

## 运行 Benchmark

### 运行本地 Fake-Node Benchmark

如果你希望在没有生产节点的情况下验证完整 benchmark 流程，启动
`./bin/anychain-agent` 后用自然语言描述目标，例如：

```text
我想运行一个 Solana fake-node benchmark。
```

Agent 应该按照真实节点同样的标准检查：chain、RPC mode、workload、自定义 RPC method、
mixed 权重、本地机器资源、磁盘、可选 Prometheus/Grafana、preflight、smoke 和最终确认。
fake-node 路径只是不需要真实 `LOCAL_RPC_URL`，不代表可以跳过 benchmark 配置校验。

### 连接真实节点运行

不要从手工编辑所有 benchmark 变量开始。先启动 Agent，让它检查当前机器或容器环境，
然后只回答它无法安全推断的缺失值。Agent 会把最终确认的值写入本次 job 的
`runtime.env`；普通用户不要手动编辑这个文件。

```bash
./bin/anychain-agent
```

然后描述目标：

```text
我想对真实 Solana 节点跑一个 quick benchmark。
```

Agent 应该推断能证明的信息，对无法证明的信息逐项询问，并在 discovery 有歧义时展示
可选项。真实节点运行必须确认 `LOCAL_RPC_URL`；如果所选链的同步健康策略需要单独的
参考端点，则还会使用 `MAINNET_RPC_URL`。如果 Agent 无法安全确认某个值，必须询问用户，
不能猜测。

最重要的输出文件：

```text
blockchain-node-benchmark-result/current/reports/performance_report_*.html
blockchain-node-benchmark-result/current/logs/proxy_method.csv
blockchain-node-benchmark-result/current/logs/performance_latest.csv
blockchain-node-benchmark-result/archives/<run-id>/test_summary.json
```

## 配置参考

### 必需值

Agent 提交 benchmark 前会检查这些配置层：

- **Agent 配置**：`config/agent_config.sh` 和可选的 gitignored
  `config/agent_config.local.sh` 定义 LLM provider、模型、认证、上下文设置和可选
  Knowledge Base。
- **Benchmark 运行配置**：Agent 会确认 chain、节点类型、RPC mode、RPC method 和权重、
  真实节点 RPC URL、节点进程名、ledger/data 磁盘、磁盘基线、网络接口、网络带宽和
  输出路径。
- **高级默认值**：`config/internal_config.sh` 及相关配置文件保存监控频率、瓶颈阈值、
  同步健康阈值、Prometheus/Grafana 默认值、Kubernetes 路径和 runtime paths。除非
  Agent 或 operator 有明确原因，大多数用户不需要修改。

`plan` 和 `preflight` 会在真实提交前暴露缺失的必需值。Agent 确认后的值会写入本次
job 的 `runtime.env`，并在该 job 中拥有更高优先级。

### LLM Provider

Agent 模型配置见上面的 [配置 Agent](#3-配置-agent)。支持的 provider family：

- `gemini`：Gemini API key，或通过 Vertex AI 使用 Google 认证。
- `claude`：Anthropic API key，或通过 Vertex AI 使用 Google 认证。
- `openai`：OpenAI API key。
- `deepseek`：通过 OpenAI-compatible endpoint 使用 DeepSeek API key。

ADK `google_search` 只在 Gemini-family 模型和有效 Gemini/Google 认证下启用，并且只用于
unsupported chain 和 custom RPC onboarding 场景。它提供证据，不替代 endpoint 确认、
fixture 录制、template validation 或 fake-node smoke。

不调用模型，只检查配置：

```bash
python3 agent/cli.py adk-status
python3 agent/cli.py llm-config
```

配置好凭据后再运行真实 provider smoke：

```bash
python3 agent/cli.py llm-smoke --prompt 'Return JSON only: {"ok": true}'
```

## 集成与运维

### 企业 Agent 平台集成

本项目可以通过几种方式嵌入企业内部 Agent 平台：

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

企业 Agent 平台和 CI 可以检查底层 JSON 控制面：

```bash
python3 agent/cli.py --help
python3 agent/cli.py tool-schema
python3 agent/cli.py tool-call --name load_capabilities
python3 agent/cli.py plan --request /tmp/request.json --output /tmp/plan.json --dry-run
python3 agent/cli.py preflight --plan /tmp/plan.json
python3 agent/cli.py submit --plan /tmp/plan.json --mock
```

### 报告与 Artifact

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

低阶 CLI job 命令默认使用 `.agent/jobs`，除非通过 `--jobs-dir` 指向其他目录。

Agent job 辅助命令：

```bash
python3 agent/cli.py jobs
python3 agent/cli.py resume --job-id <job_id>
python3 agent/cli.py logs --job-id <job_id>
python3 agent/cli.py diagnose-artifacts --artifact-index <agent-output-dir>/jobs/<job_id>/artifact_index.json
```

`diagnose-artifacts` 会对已有 CSV 应用确定性瓶颈规则，包括 CPU 饱和、磁盘延迟/队列、
磁盘 IOPS 或吞吐压力、RPC method 错误/延迟和同步健康 warning。

### 可选 Prometheus/Grafana

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

## 扩展框架

### 扩展链或 RPC Method

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

生成需要人工 review 的保守 chain template 草案：

```bash
python3 agent/cli.py draft-chain-template \
  --chain foochain \
  --adapter-family jsonrpc \
  --method foo_getBalance \
  --method foo_getTransaction \
  --output /tmp/foochain.json
```

草案会标记为 `needs_review`，不会自动安装到 `config/chains`。

对于未支持链，Agent 会生成 onboarding plan，而不是自动修改代码。通常流程是：

1. 基于模板新增 `config/chains/<chain>.json`。
2. 选择 `_meta.adapter_family`。
3. 配置 `rpc_methods.single` 和 `rpc_methods.mixed_weighted`。
4. 添加 `param_formats` 或 `param_spec`。
5. 添加 `proxy_extraction` 规则。
6. 录制 fake-node fixture。
7. 运行 preflight 和 fake-node 闭环测试。

当 Gemini web research 启用时，Chain/RPC onboarding 流程可以使用 ADK
`google_search` 查找未支持链或自定义 RPC method 的官方 RPC 文档、节点 operator 文档
和官方 API 示例。搜索结果只是证据：不能跳过 endpoint 确认、真实 request/response
样本、fixture 录制、template validation 或 fake-node smoke。

## Reference 文档

- [AI Assistant Operator Guide](AGENTS.md)
- [配置指南](config/README.md)
- [Agent 控制平面](agent/README.md)
- [ADK Agent 架构](docs/zh/adk-agent-architecture.md)
- [AnyChain Agent AI 工作 Gate](docs/zh/anychain-agent-ai-work-gate.md)
- [完整框架 Reference](docs/zh/framework-reference.md)
- [框架流程与数据生命周期](docs/zh/framework-flow.md)
- [模块说明](docs/zh/module-guide.md)
- [如何新增区块链或 RPC Method](docs/zh/how-to-add-chain.md)
- [使用 fake-node 进行本地闭环测试](docs/zh/local-closed-loop-testing.md)
- [二次开发指南](docs/zh/secondary-development-guide.md)
- [GitHub PR Gate 与分支保护](docs/zh/github-pr-gates.md)
- [GitHub PR 提交流程](docs/zh/github-pr-workflow.md)
- [Prometheus / Grafana Observability](deploy/observability/README.md)
- [Kubernetes Collector](deploy/k8s/README.md)

## License

本项目采用双许可证：

- 开源使用：AGPL-3.0-or-later，见 [LICENSE](LICENSE)。
- 商业/专有/内部使用：见 [COMMERCIAL.md](COMMERCIAL.md)。
