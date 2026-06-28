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

### 配置方式

在开始修改 benchmark 变量前，先选择一种配置方式。

#### AI 自动配置

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

#### 手动配置

如果你希望自己配置，请继续阅读下面的配置模型。先从 `config/agent_config.sh`
开始，把真实本地密钥写入 `config/agent_config.local.sh`，然后启动
`./bin/anychain-agent`。Agent 会在对话中检查 benchmark 环境，并询问缺失的
benchmark 变量。

### 配置模型

大多数用户启动 Agent 前只需要关注一个文件：

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

### 5 分钟快速开始

这是在终端里使用 AnyChain Benchmark Agent 的最快路径。面向用户的 Agent 运行时是
Google ADK，所以需要先在 `config/agent_config.sh` 中配置 provider、模型和认证方式。
无凭据离线检查可以用于 CI 和开发，但不是产品 Agent 运行方式。

克隆仓库：

```bash
git clone git@github.com:StayHungryStayFoolish/AnyChainBenchmarkAgent.git
cd AnyChainBenchmarkAgent
```

先安装 Agent 运行时。这个脚本只为 AnyChain Benchmark Agent 创建隔离的 Google ADK
环境，不要把 ADK 安装到生产区块链节点使用的 Python 环境中：

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

在 `config/agent_config.sh` 中配置持久化的 Agent 参数。建议把真实密钥写入
`config/agent_config.local.sh`。支持的 provider 可以使用 API Key 模式；Gemini 或
Vertex partner model 也可以通过 Vertex AI 使用 Google service account。

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

启动交互会话前先验证 Agent 和 LLM 配置：

```bash
python3 agent/cli.py adk-status
python3 agent/cli.py llm-config
```

chain、RPC URL、磁盘、机器类型等 benchmark 信息可以先不配置。Agent 会自动发现能
发现的信息，并在真实运行前提示你补充缺少的必需值。

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
Agent> ...继续确认 cloud/zone/machine/disk/network，然后询问是否查看高级配置...

User> 2
Agent> ...在 RPC 模式阶段，2 表示 mixed 多方法加权 workload...
```

在新环境中建议先输入 `doctor`。它会以只读方式检查 cloud/deployment 识别结果、
必需依赖、LLM/Vertex 配置、Knowledge Base 配置和当前框架能力覆盖情况。
如果缺少 benchmark 依赖，Agent 应该先说明计划安装的内容，并在用户明确授权后再执行安装。

也可以使用一句 prompt 运行：

```bash
./bin/anychain-agent \
  --prompt "Create a Solana fake-node smoke benchmark at 1 QPS"
```

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

### 入口

大多数用户只需要启动一个命令：

```bash
./bin/anychain-agent
```

其他入口用于自动化：

- `python3 agent/cli.py ...`：CI、测试或需要 JSON 输入/输出的企业 Agent 平台。
- `./blockchain_node_benchmark.sh`：底层压测执行引擎。Agent 在计划确认后会调用它。
  直接使用时要求配置已经存在。

## 运行 Benchmark

### 运行本地 Fake-Node Benchmark

如果你希望在没有生产节点的情况下通过 Agent 验证 benchmark 流程，启动
`./bin/anychain-agent` 后输入：

```text
Check this host and dependencies.
Prepare a Solana fake-node smoke benchmark at 1 QPS.
Run lifecycle smoke after showing me the generated plan.
Show job status.
```

Lifecycle smoke 只验证 Agent job 生命周期，不会发送 benchmark 流量。如果希望让 Agent
调用真实 benchmark engine 并连接 fake-node，可以输入：

```text
Run a real fake-node benchmark smoke in isolated output directories.
```

在要求 Agent 提交任何真实 benchmark 前，请先 review 生成的 runbook 和 smoke 结果。

### 连接真实节点运行

不要从手工编辑所有 benchmark 变量开始。先启动 Agent，让它检查当前机器或容器环境，
然后只回答它无法安全推断的缺失值。Agent 会把最终确认的值写入本次 job 的
`runtime.env`；普通用户不要手动编辑这个文件。

```bash
./bin/anychain-agent
```

真实节点对话示例：

```text
Check this host and dependencies.
Prepare a quick single-method benchmark for my Solana node at http://your-node-rpc:8899.
Show me inferred values and ask me to confirm anything uncertain.
Run lifecycle smoke after preflight passes.
Show job status.
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

## 配置参考

### 必需值

Agent 会检查三层配置：

- **Agent checklist**：`config/agent_config.sh`，检查 LLM provider、模型、
  Vertex/OpenAI 认证、上下文压缩和可选 Knowledge Base。
- **Benchmark checklist**：`plan` 和 `preflight` 检查 chain、RPC mode、真实节点
  `LOCAL_RPC_URL`、进程名、ledger 磁盘、磁盘基线和网络带宽等运行值。
- **Advanced checklist**：监控频率、瓶颈阈值、同步健康阈值、Prometheus/Grafana、
  Kubernetes 和 runtime paths。大多数用户不需要修改。

`plan` 和 `preflight` 会在 `yes run` 前暴露缺失的必需值；高级配置保留给需要调优的
operator。

### LLM Provider

官方 ADK runtime 负责模型执行。AnyChain 会读取 `config/agent_config.sh` 来解析模型
名称并运行安全的认证诊断，但面向用户的 Agent 路径不再使用旧的自研 provider adapter
替代 ADK 模型调用。

无凭据检查只用于 CI/开发：它们验证 ADK package 加载、tool 注册和安全 callback，不会
模拟自然语言意图识别，也不是产品 Agent runtime。

支持的 provider 和认证方式：

- `gemini`：Gemini API key，或通过 Vertex AI 使用 Google 认证。
- `claude`：Anthropic API key，或通过 Vertex AI 使用 Google 认证。
- `openai`：OpenAI API key。
- `deepseek`：通过 OpenAI-compatible endpoint 使用 DeepSeek API key。

请在 `config/agent_config.sh` 中持久化配置默认变量；真实密钥和本地 provider 选择建议
写入 gitignored 的 `config/agent_config.local.sh`。`./bin/anychain-agent` 启动时会自动
加载 `config/agent_config.sh`，后者会再加载 local override；临时测试时仍可用环境变量覆盖。

Gemini API key：

```bash
LLM_PROVIDER="gemini"
LLM_MODEL="gemini-3.1-pro"
LLM_AUTH_MODE="api_key"
GEMINI_API_KEY="AIza..."
```

`claude` API key：

```bash
LLM_PROVIDER="claude"
LLM_MODEL="claude-opus-4-8"
LLM_AUTH_MODE="api_key"
ANTHROPIC_API_KEY="sk-ant-..."
```

通过 Vertex AI 和 service-account impersonation 使用 Gemini 或 `claude`：

```bash
LLM_PROVIDER="gemini"
LLM_MODEL="gemini-3.1-pro"
LLM_AUTH_MODE="service_account_impersonation"
GOOGLE_CLOUD_PROJECT="your-project"
GOOGLE_CLOUD_LOCATION="global"
GOOGLE_SERVICE_ACCOUNT_EMAIL="benchmark-agent@your-project.iam.gserviceaccount.com"
```

如果使用 OpenAI：

```bash
LLM_PROVIDER="openai"
LLM_MODEL="gpt-5.5"
OPENAI_API_KEY="sk-..."
```

如果使用 DeepSeek：

```bash
LLM_PROVIDER="deepseek"
LLM_MODEL="deepseek-chat"
LLM_AUTH_MODE="api_key"
DEEPSEEK_API_KEY="sk-..."
```

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

### 传统 Benchmark 入口

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

## 集成与运维

### 企业 Agent 平台集成

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
