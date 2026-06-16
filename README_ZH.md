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

## 能做什么

- 支持 36 个 chain template，覆盖 6 个 adapter family。
- 从 `config/chains/*.json` 生成 single 或 weighted mixed RPC workload。
- 通过 `rpc_methods`、`param_formats`、可选 `param_spec`、REST path 绑定和
  fake-node fixture 支持自定义 RPC method。
- 记录每个 method 的状态、成功/失败次数和 P50/P90/P99 延迟。
- 监控 CPU、内存、磁盘、网络、cgroup、同步健康和监控系统自身开销。
- 生成 HTML 报告并归档每次运行。
- 通过只读 exporter 可选接入 Prometheus/Grafana。
- 提供 Agent 控制平面：prompt-first planning、risk scoring、capability gap
  analysis、artifact-aware Q&A 和长时间 job 跟踪。

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

## 使用 Agent 快速开始

只检查依赖，不修改宿主机：

```bash
bash scripts/install_deps.sh --check
```

查看当前框架能力：

```bash
python3 agent/cli.py capabilities
python3 agent/cli.py ask --prompt "How many chains and RPC methods does the framework support?"
```

从 prompt 生成 benchmark request：

```bash
python3 agent/cli.py draft-request \
  --prompt "Test Solana maximum stable QPS on GKE with fake-node smoke first" \
  --output /tmp/request.json
```

生成并检查 plan：

```bash
python3 agent/cli.py plan \
  --request /tmp/request.json \
  --output /tmp/plan.json \
  --discover \
  --dry-run

python3 agent/cli.py preflight --plan /tmp/plan.json
python3 agent/cli.py risk-score --plan /tmp/plan.json
python3 agent/cli.py runbook --plan /tmp/plan.json --output /tmp/runbook.md
```

提交 lifecycle-only mock job：

```bash
python3 agent/cli.py submit --plan /tmp/plan.json --mock
```

review plan 和 runbook 后再提交真实 benchmark：

```bash
python3 agent/cli.py submit --plan /tmp/plan.json --approved
```

查询或分析 job：

```bash
python3 agent/cli.py status --job-id <job_id>
python3 agent/cli.py analyze --job-id <job_id>
python3 agent/cli.py artifact-qa --job-id <job_id> --question "Why are charts empty?"
```

## 可选 LLM Provider

Agent 不依赖 LLM 也能工作。开启 LLM 后，模型只用于 request drafting 和 intent
classification，随后仍会经过确定性校验。

支持的 provider contract：

- `vertex_gemini_openai`：Vertex AI 上通过 OpenAI-compatible API 调用 Gemini。
- `vertex_claude`：Vertex AI 上的 Claude partner models。
- `openai`：OpenAI API。
- `fake`：离线 protocol smoke provider，用于测试。

企业环境推荐使用 Vertex AI 的 ADC 或 service-account impersonation，而不是静态 API key：

```bash
export LLM_PROVIDER=vertex_gemini_openai
export LLM_MODEL=gemini-2.5-pro
export GOOGLE_AUTH_MODE=service_account_impersonation
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_SERVICE_ACCOUNT_EMAIL=benchmark-agent@your-project.iam.gserviceaccount.com
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

运行本地 fake-node 闭环：

```bash
BLOCKCHAIN_NODE=solana \
RPC_MODE=single \
QUICK_INITIAL_QPS=1 \
QUICK_MAX_QPS=1 \
QUICK_QPS_STEP=1 \
QUICK_DURATION=3 \
QPS_WARMUP_DURATION=0 \
QPS_COOLDOWN=0 \
./blockchain_node_benchmark.sh --quick --single --fake-node
```

如果节点部署在 Kubernetes 中，请先部署 collector：

```bash
deploy/k8s/validate.sh --preflight
kubectl apply -f deploy/k8s/
kubectl rollout status -n blockchain-bench ds/blockchain-bench-collector
deploy/k8s/validate.sh --post-deploy
```

然后在选定的 runner 上使用同一份 `config/user_config.sh` 运行 benchmark。

## 报告与 Artifact

当前运行文件写入 runtime `current/` 目录，最终结果会在运行结束后归档。

关键 artifact：

- `current/reports/performance_report_*.html`
- `current/logs/proxy_method.csv`
- `current/logs/performance_latest.csv`
- `archives/<run-id>/test_summary.json`
- `.agent/jobs/<job_id>/artifact_index.json`
- `.agent/jobs/<job_id>/runtime.env`

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
- [GitHub PR Gate 与分支保护](docs/zh/github-pr-gates.md)
- [Prometheus / Grafana Observability](deploy/observability/README.md)
- [Kubernetes Collector](deploy/k8s/README.md)

## License

本项目采用双许可证：

- 开源使用：AGPL-3.0-or-later，见 [LICENSE](LICENSE)。
- 商业/专有/内部使用：见 [COMMERCIAL.md](COMMERCIAL.md)。
