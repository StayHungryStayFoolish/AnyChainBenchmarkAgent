# AnyChain Agent 架构

AnyChain Agent 是基于 LangGraph Harness 的产品级 Agent，用于控制
blockchain-node-benchmark 引擎。Harness 负责 workflow 状态、group 路由、
fallback 顺序、校验门禁和执行决策。Harness 自身的模型调用对所有 provider
（OpenAI、DeepSeek、Vertex 上的 Gemini）统一走 OpenAI 兼容的 HTTP 请求；
Google ADK 只用于唯一一项可选能力——Gemini `google_search` 联网检索
（`agent/llm/search_grounding.py`），以普通函数调用的方式从 Harness 代码里触发。
它绝不能拥有第二套 benchmark wizard、第二套对话循环，也不能绕过 Harness 修改
workflow state。

## Architecture Overview

```mermaid
flowchart TD
  U["User terminal"] --> T["AnyChain product terminal<br/>bin/anychain-agent"]
  T --> D["Startup diagnostics<br/>framework context, environment, dependencies, jobs"]
  T --> H["LangGraph Harness<br/>agent/harness"]

  H --> I["Typed intent resolver<br/>configured LLM"]
  H --> G["Group workflows<br/>provider, disk, network, chain, workload, QPS, sync-observe"]
  H --> S["Persistent checkpoint<br/>ANYCHAIN_AGENT_CHECKPOINT_PATH"]
  H --> VAL["Deterministic validators<br/>config, workload, onboarding, execution gate"]
  H --> PLAN["Plan and runtime.env builder"]
  H --> JOB["Detached job manager<br/>.agent/jobs/job_id"]
  H --> SEARCH["Gemini-only google_search<br/>onboarding/custom RPC evidence"]

  I --> G
  G --> VAL

  VAL --> PRE["Preflight"]
  PRE --> SMOKE["隔离 fake-node smoke"]
  SMOKE --> APPROVE["User approval callback"]
  APPROVE --> BENCH["Benchmark engine<br/>blockchain_node_benchmark.sh"]
  BENCH --> PROXY["Proxy and per-method attribution"]
  BENCH --> MON["Monitoring system"]
  BENCH --> FN["fake-node fixtures"]
  BENCH --> ART["Reports, charts, archives"]
  ART --> ANA
  ANA --> U
```

## Agent Loop

```mermaid
flowchart LR
  A["User turn"] --> B["Typed intent<br/>LLM resolver"]
  B --> C["Route to group<br/>LangGraph Harness"]
  C --> D["Ask one blocking question<br/>or activate requested group"]
  D --> E["Validate<br/>deterministic gates"]
  E --> F{"Ready?"}
  F -- "No" --> C
  F -- "Yes" --> G["Execute<br/>smoke or detached job"]
  G --> H["Observe<br/>logs and artifacts"]
  H --> I["Analyze<br/>evidence-backed"]
  I --> J["Iterate<br/>update state or next plan"]
  J --> A
```

The loop prevents the Agent from acting like a keyword bot:

- user intent is interpreted by the configured model and returned as typed graph actions;
- confirmed facts are stored as structured LangGraph state;
- users may jump between groups, go back, or revise prior answers;
- completing an interrupted group falls back to the next missing required group;
- every execution path passes through deterministic validators;
- the Agent asks for missing information instead of inventing values;
- smoke tests are isolated from final benchmark job artifacts;
- real benchmark jobs require preflight, smoke, and user approval;
- analysis must cite generated evidence paths.

## Accuracy Boundaries

The Agent may infer and suggest values, but it must not silently decide:

- `LEDGER_DEVICE` when multiple disks are plausible;
- whether a separate `ACCOUNTS_DEVICE` exists;
- custom RPC parameter contracts;
- mixed workload weights;
- unsupported chain adapter family;
- real-node endpoint validity before preflight;
- external Prometheus/Grafana scraping behavior.

When uncertain, the Agent must show the available evidence and ask the user to
confirm or provide a value.

## Runtime State And Artifacts

```mermaid
flowchart TD
  C["LangGraph checkpoint"] --> S["ANYCHAIN_AGENT_CHECKPOINT_PATH<br/>default .agent/checkpoints/agent.sqlite"]
  T["Terminal session state"] --> TS["--state-file JSON"]
  P["Plan"] --> E[".agent/jobs/job_id/runtime.env"]
  J["Job metadata"] --> M[".agent/jobs/job_id/job.json"]
  L["Benchmark logs"] --> LOG[".agent/jobs/job_id/benchmark.log"]
  A["Artifact index"] --> IDX[".agent/jobs/job_id/artifact_index.json"]
  R["Report archive"] --> REP["benchmark-data/archives/run_timestamp"]
```

`runtime.env` is the final per-job confirmed configuration. Users should not
edit it manually. If a user changes an earlier answer, the Harness must update
or invalidate the affected group state and regenerate downstream runtime
artifacts through deterministic tools.

## Google Search Boundary

ADK `google_search` is intentionally narrow:

- enabled only for Gemini with Google authentication and an ADK runtime that
  exposes the tool (`agent/llm/search_grounding.py::web_research_status`);
- invoked only as a scoped, single-query function call
  (`run_google_search_grounding`) from specific Harness call sites (currently
  real-node client setup) — never a persistent Agent/Runner or a second
  conversation loop;
- used for unsupported chain and custom RPC research;
- official documentation is preferred;
- search evidence does not replace endpoint tests, fixture recording, template
  validation, or fake-node smoke.

Other model providers must report web research as unavailable unless the
repository explicitly adds and verifies a provider-specific search integration.

## Sync-Observe 边界

`sync-observe` 是节点同步/资源观察的一等 workflow，不属于 RPC workload 路径。
当用户希望观察节点追高速度、MGas/s、节点进程 CPU/线程热点、磁盘 latency/iowait
或网络行为，并且不希望发送 benchmark RPC 流量时，Harness 应该路由到
sync-observe workflow。

该 workflow 需要确认：

- chain 和 sync-health/reference 行为；
- 资源元数据，包括磁盘和网络 baseline；
- 用于 CPU/线程归因的节点进程身份；
- 可选的 `NODE_PROMETHEUS_METRICS_URL`，用于 MGas/s 或客户端原生 execution
  metrics；
- 停止条件：用户停止、固定 duration，或 until synced。

除非用户明确切换回 RPC benchmark，否则该 workflow 不应询问 RPC mode、自定义 RPC
workload、mixed weights、Vegeta 或 QPS profile。

## Development Gates

Before changing Agent code, read:

1. `AI_CODING_GUIDE.md`
2. `docs/zh/anychain-agent-ai-work-gate.md`
3. `agent/README.md`

Then run relevant checks:

```bash
python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract tests.test_agent_langgraph_harness
python3 tools/check_agent_boundaries.py --root .
git diff --check
```

For model-facing behavior, run the current product Harness defined by the
reviewed task/design document. Lower-level live/PTY scripts can be provider
drivers or developer helpers, but product readiness requires realistic CLI
scenarios and deterministic assertions.
