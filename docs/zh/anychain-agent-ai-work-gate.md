# AnyChain Agent AI Work Gate

本文档是 AnyChain Agent 的项目级 AI coding Gate，用于补充仓库根目录
`AI_CODING_GUIDE.md` 中的通用行为约束。

修改 Agent 代码前，AI coding agent 必须阅读：

1. `AI_CODING_GUIDE.md`.
2. 本 Gate 文档。
3. `docs/zh/adk-agent-architecture.md`。
4. `agent/README.md`。
5. 当前已经 review 的 Agent task/design 文档。
6. 计划修改的具体文件。

实现捷径与这些规则冲突时，以这些规则为准。

## 修改代码前的文档 Gate

在任务没有被准确记录之前，不得修改 Agent workflow 代码。

任何影响 Agent 行为、Harness prompt、LLM routing、workflow branch、terminal
交互、workflow state、benchmark execution、validator、runner lifecycle、
fake-node smoke、endpoint validation 或 onboarding 的变更，都必须：

1. 阅读 `AI_CODING_GUIDE.md`、本文件、Agent 架构文档和当前 task/design 文档。
2. 确认 task 文档写明根因、范围、需要精读和禁止修改的文件、预期用户行为、
   验证命令、验收证据以及清理要求。
3. 如果 task 文档缺失、过时、模糊或与代码矛盾，先更新并 review 文档，之后
   才能修改代码。
4. 每项代码变更都必须能够追溯到已记录的任务和架构约束。

禁止以下捷径：

- 只修复一段 terminal transcript，而不更新 group workflow 或 task 文档；
- 用局部 if/else、regex、模糊匹配、phrase cleanup 或 fallback 掩盖 workflow 缺陷；
- 未证明现有架构无法承载需求，就创建新的 helper 文件；
- 设计要求删除或迁移旧逻辑时，仅通过改名或隔离继续保留；
- 未运行文档规定的 Gate 就声称某个阶段完成。

live CLI 或测试发现新问题时，必须先在 task 文档中分类并确认根因，再开始编码。
修复应落在 LangGraph Harness workflow、typed state、确定性工具、validator 或
terminal I/O 边界，不能只针对失败 prompt 的原句。若确实采用局部方案，task 文档
必须说明取舍、移除条件并增加 Harness coverage，否则不得提交该方案。

单条 transcript 通过不能证明修复成立。修复必须同时保持相邻分支中的 group
workflow、typed `pending_question`、validator 顺序、用户修正路径和执行 Gate。
配置对话问题至少要区分：pending question 缺失或过时、一个 prompt 暴露多个问题、
确定性 next-question transition 缺失、smoke 值泄漏到正式 benchmark，以及 terminal
层试图补偿 workflow transition 缺失。

## Non-Negotiable Product Boundary

AnyChain Agent 是基于 LangGraph Harness 的区块链节点 benchmark domain agent。
Harness 负责产品 workflow state、group routing、fallback ordering、validator
gates 和 execution decisions。Google ADK 仅由可选的 Gemini `google_search`
grounding 函数使用；它不是 Agent、Runner、workflow owner 或通用 tool bridge。
Agent 必须降低用户配置负担，并安全调用确定性的 benchmark
tools。它不是 shell script wizard，不是 keyword router，也不是 fallback demo 集合。

Google ADK 必须通过 `scripts/install_agent_deps.sh --with-google-search` 显式安装。
core LangGraph terminal runtime 以及 DeepSeek/OpenAI/Claude/不使用 search 的 Gemini
运行都不得依赖它。保留的 `requirements-adk.txt`、`.venv-adk` 和 `--adk-venv`
只是迁移期兼容 alias。

The product loop is:

```text
Understand -> Plan -> Ask -> Configure -> Validate -> Execute -> Observe -> Analyze -> Iterate
```

配置的模型负责把模糊自然语言理解为 typed Harness actions。LangGraph Harness
负责 planning、group selection、question selection、fallback ordering 和 iteration。
仓库工具负责确定性检查、配置物化、benchmark execution、证据采集和基于 artifact 的分析。

对外术语也是控制权边界的一部分。terminal message、doctor action、提供给模型的
framework context、tool schema 和生成的 onboarding plan 都不得把核心 runtime 称为
ADK Agent/terminal/runtime。`ADK` 只能描述可选 Gemini `google_search`、保留的兼容
路径名称或明确的历史代码。所有生成路径都必须根据当前仓库验证；chain template
起始文件是 `config/chain_template.json.bak`。

## Group Workflow 与 Y/N 契约

完整 workflow map 和 Harness 设计必须存在于当前已 review 的 task/design 文档中，
不得只依赖本 Gate 的简表修改 workflow。

每个 yes/no 答案必须绑定唯一 active pending question。只有接受与拒绝路径都已注册，
Agent 才能提出 Y/N 问题：

- 没有 pending question 时，裸 `Y`、`N`、`yes` 或 `no` 不是有效业务决策；
- dependency installation 的 `Y` 只返回需要在当前 shell 执行的准确命令，`N` 拒绝；
  交互终端不得在对话会话内执行安装脚本；
- target mode 问题只接受已注册的 fake-node、real-node、sync-observe 或对应选项；
- disk choice 的编号选择已展示设备，manual value 覆盖推断值；
- quick assumed fake-node smoke 的 `Y` 必须提交对应 detached smoke，不得重复确认；
- 用户反悔、回退或修改旧答案时，必须更新或回滚 state、执行 invalidation，并重新
  运行 validator 后再继续。

detached job 提交后，Agent 应返回 `job_id`、run directory、`benchmark.log` 以及
`status`、`logs`、`follow` 的使用方式。除非 workflow state 中存在对应的已注册
pending action，否则不得临时询问“是否查看日志”等无状态 Y/N 问题。

## Forbidden Patterns

Do not add or reintroduce:

- business intent routing in terminal code through keyword lists, fuzzy matches,
  regex guesses, or language-specific phrase tables;
- workflow shortcuts that bypass LangGraph Harness groups, typed tools, validators, user
  confirmation, preflight, or smoke testing;
- old non-Harness wizard/fallback logic for benchmark planning;
- phrase-patching that rewrites model style instead of fixing instructions or
  Harness workflow behavior;
- claims that an unsupported chain, RPC method, fixture, endpoint, or
  benchmark path works without evidence;
- changes to `config/agent_config.sh` unless the user explicitly asks;
- committed API keys, service account JSON, ADC files, generated runtime state,
  or live benchmark archives.

Stable terminal commands such as `help`, `doctor`, `jobs`, `status`, `logs`,
`follow`, and `exit` are allowed. Business requests must go through the
LangGraph Harness workflow.

## 旧代码污染 Gate

修复 Agent workflow 前，必须审查 `agent/` 中是否仍有旧 custom-Agent 逻辑。
代码不能因为当前仍被 import 就自动获得保留资格。

允许保留的职责只有：

- LangGraph Harness runtime、group workflow、checkpoint 和 typed event；
- `agent/llm/search_grounding.py` 中可选的 Gemini `google_search` grounding；
- 确定性的 planner、validator、runner、analyzer、discovery、onboarding 和 knowledge provider；
- terminal I/O、稳定 shell command、Ctrl+C/log-follow、dependency consent 和进度显示；
- 明确位于产品 runtime 之外的开发工具与测试。

必须删除或迁移：

- 旧的非 Harness benchmark wizard；
- fallback brain、mock agent 或 phrase-repair loop；
- terminal keyword/fuzzy/regex 业务路由；
- 重复 workflow state machine；
- 冒充真实 smoke 的 lifecycle-only mock；
- 只因旧 import 存在而保留的 dead file。

旧逻辑中有价值的确定性能力，应迁移到正确的 planner、validator、runner、analyzer、
onboarding 或 knowledge 组件；旧 conversational wrapper 不得继续存在。只要 legacy
代码仍能绕过 Harness intent routing、typed pending question、validator、preflight、
smoke 或用户授权 Gate，Agent 就不能声明达到产品验收标准。

`agent/adk_app/` 整个 package 是已经退休的第二套 ADK-native conversation loop，
不得恢复。`agent/harness/` 是唯一产品 workflow runtime；terminal 只负责 I/O，
Google ADK 只属于可选 search grounding 边界。

## Required Agent Behavior

At startup, the Agent must load framework context and run local discovery:

- cloud provider and platform: GCP, AWS, other; VM or Kubernetes;
- region, zone, machine type when metadata is available;
- CPU and memory;
- network interface;
- disk inventory from `lsblk` where available;
- dependency status;
- previous job/session status.

When values are inferred, the Agent must show the inferred value and allow
manual override. If multiple disk candidates exist, it must show numbered disk
rows and ask the user to confirm:

- `LEDGER_DEVICE`;
- whether an `ACCOUNTS_DEVICE` exists;
- data/accounts disk baseline values.

Fake-node mode may provide default local RPC endpoint values, but it must not
skip resource metadata confirmation. The transition from fake-node to real-node
must reuse confirmed environment metadata and only ask for the delta, such as
real `LOCAL_RPC_URL`, `MAINNET_RPC_URL`, chain changes, RPC mode, RPC methods,
and weights.

Users must be able to correct prior answers. If the user says a previous value
was wrong, wants to go back, or changes the test target, the Harness must
update or revert workflow state, re-run validators, and ask the next blocking
question.

## Required Configuration Gates

Before smoke or real benchmark execution, validators must confirm:

- target mode: fake-node or real-node;
- workflow type: RPC benchmark or sync-observe;
- chain and chain template requirements;
- RPC mode: single or mixed;
- custom RPC method definitions, parameter samples, fixtures, and weights when
  used;
- benchmark mode: quick, standard, or intensive;
- QPS profile for the selected mode, including initial QPS, max QPS, step, and
  duration;
- observability mode: disabled, local Prometheus/Grafana, or exporter-only for
  an existing environment;
- required runtime metadata from `config/user_config.sh`;
- optional accounts disk metadata when an accounts/state disk exists;
- port availability for fake-node, proxy, Prometheus, Grafana, and exporters
  when those paths are selected.

Smoke tests must use isolated runtime files and must not pollute the final
benchmark job configuration or result archive.

Smoke is a complete closed-loop benchmark execution. It is not a mock and not a
partial check. Quick smoke should use very small QPS settings and short
duration, but it must still exercise traffic generation, fake-node or endpoint
handling, proxy, monitoring, reports, archive creation, and artifact discovery.

`sync-observe` 是独立 workflow type，不是 quick/standard/intensive profile。
它用于观察节点同步和资源行为，不生成 RPC workload、不走 proxy traffic、不使用
Vegeta，也不执行 QPS ramp。它的 validator 必须确认 chain 和 sync-health/reference
行为、资源元数据、节点进程身份、可选节点 Prometheus metrics endpoint，以及停止条件：
用户停止、固定 duration，或 until synced。除非用户切换回 RPC benchmark workflow，
否则不得询问 RPC mode、自定义 RPC method、mixed weights 或 QPS profile。

这也意味着 `sync-observe` 不应进入任何 Vegeta 相关运行路径：不生成 Vegeta
targets，不为 workload traffic 启动 RPC proxy，不运行 QPS executor，也不要求
`vegeta_results` 产物作为报告成功条件。Sync-observe 报告应来自 monitoring、
sync-health、node execution、disk、CPU 和 network 数据。

`sync-observe` 不需要 fake-node fixture 录制。真实节点可能先从 peer 下载快照，
然后从快照高度继续追块；Agent 应该观察这个真实同步行为，而不是把它录制成 RPC
fixtures。当用户提供本地节点 endpoint 或 public reference endpoint 用于同步观察时，
只复用现有 endpoint/sync-health 真实性校验，证明 endpoint 可访问并且能够暴露高度或
同步状态。除非用户明确切回 RPC benchmark 或 custom-RPC onboarding，否则不得进入
custom RPC fixture recording、target sample collection 或 workload schema validation。

用户可以在任意 benchmark setup group 中请求切换到 `sync-observe`，包括
chain/endpoint、workload、自定义 RPC、mixed weights、QPS、observability、
preflight 或 report follow-up。Agent 必须暂停当前 group，将 workflow type 切换为
`sync_observe`，保留可复用的环境、资源和链状态，失效 RPC-only state，并且只询问
sync-observe blockers。如果用户随后切回 RPC benchmark，Agent 必须重新询问 RPC
workload 和 QPS gates，不能复用已经失效的状态。Harness acceptance 必须证明
`--sync-observe` 命令路径不会调用 proxy、Vegeta 或 RPC target generation。

## Configuration Group Workflow Standard

The Agent configuration flow is not a fixed one-way wizard. It is a set of
configuration groups with a default order, global natural-language routing, and
validator-driven recovery. Users may jump between groups, correct earlier
answers, change target chain or mode, paste evidence, or ask questions at any
time. The Agent must route the turn to the right group, preserve valid state,
invalidate affected state, and then return to the next blocking group in the
default order.

`agent/workflows/group_registry.py::GROUPS` 是唯一 group metadata authority。
当前默认顺序为：

| 顺序 | Group | Owner | 适用范围 |
|---:|---|---|---|
| 1 | `opening` | `orientation` | session 入口与恢复 |
| 2 | `target_mode` | `chain_rpc` | 所有 workflow |
| 3 | `chain_identity` | `chain_rpc` | 所有 workflow |
| 4 | `provider_deployment` | `environment` | 所有 workflow |
| 5 | `ledger_disk` | `environment` | 所有 workflow |
| 6 | `accounts_disk` | `environment` | 所有 workflow |
| 7 | `network` | `environment` | 所有 workflow |
| 8 | `endpoint_process` | `chain_rpc` | real-node 与 sync-observe；Case 1/2 需要 live endpoint 验证时也适用于 fake-node |
| 9 | `chain_auxiliary_endpoints` | `chain_rpc` | real-node 与 sync-observe |
| 10 | `workload_rpc` | `chain_rpc` | 仅 RPC benchmark |
| 11 | `target_samples_fixtures` | `chain_rpc` | 仅 RPC benchmark |
| 12 | `qps_profile` | `performance` | 仅 RPC benchmark |
| 13 | `sync_observe` | `sync_observe` | 仅 sync-observe |
| 14 | `observability` | `performance` | 所有 workflow |
| 15 | `advanced_tuning` | `performance` | 所有 workflow |
| 16 | `preflight_smoke_execution` | `execution` | 当前 workflow 的执行 Gate |
| 17 | `job_monitoring` | `execution` | 显式 job 操作；不参与 fallback |
| 18 | `failure_recovery` | `recovery` | 显式失败恢复；不参与 fallback |
| 19 | `error_evidence_analysis` | `analysis` | 显式 evidence 分析；不参与 fallback |
| 20 | `report_artifact_analysis` | `analysis` | 显式报告分析；不参与 fallback |

框架固定为 20 个 group、8 个 domain owner。graph coordinator 负责 typed control
dispatch，但不属于第 9 个 domain owner，也不拥有 `GroupSpec`。fallback 会跳过
workflow/target-mode filter 不匹配的 group，并且不会在没有显式 action 时进入最后
四个非 fallback 的运维与分析 group。

Each group must define:

- required fields and optional fields;
- inferred values and their evidence;
- manual override paths for every inferred value;
- invalidation rules when an upstream value changes;
- validators and evidence that mark the group complete;
- the next blocking group when complete;
- advanced settings, if any, and how to explain them before asking whether to
  adjust them.

The Agent must maintain group-level state, not only a single
`pending_question`. State must track the active group, group progress,
confirmed fields, invalidated fields, evidence, and interruption stack. If the
user jumps from one group to another, the previous group is paused, the target
group runs, validators recompute the next blocking group, and the Agent returns
to the default order unless the user explicitly requests another jump.

Group interruption examples that must be supported:

- user changes chain while answering disk, QPS, endpoint, observability, or
  final approval questions;
- user changes target mode from fake-node to real-node after resource metadata
  is already confirmed;
- user adds custom RPC methods after accepting the default workload;
- user changes QPS profile after selecting a benchmark mode;
- user says a previous disk, network, endpoint, region, or workload answer was
  wrong;
- user returns from unsupported-chain Case 3 to a supported chain, Case 1, or
  Case 2 path.

Invalidation must be precise:

- changing chain invalidates chain endpoints, RPC mode, workload methods,
  weights, target samples, fixtures, endpoint evidence, and runtime chain
  template overrides, but should keep cloud, hardware, disk, network, and
  observability answers unless the user asks to change them;
- changing target mode invalidates endpoint and process-name requirements that
  differ between fake-node and real-node;
- changing `LEDGER_DEVICE` invalidates data disk size, IOPS, and throughput;
- changing `ACCOUNTS_DEVICE` invalidates accounts disk size, IOPS, and
  throughput;
- changing benchmark mode invalidates the selected QPS profile confirmation;
- changing custom RPC methods invalidates method params, weights, fixture
  evidence, and workload validation.

Harness and live CLI tests must cover group jumps, rollback, invalidation, and
return-to-default-order behavior. Passing a linear happy path is not enough.

## Onboarding And Knowledge Boundary

For a chain outside the supported templates, Harness must not jump directly to
adapter-family selection, endpoint collection, or development handoff. It must
first resolve whether the chain identity appears to exist, then confirm the
protocol/adapter family. If framework knowledge is insufficient, ask the user
for official chain documentation, protocol/RPC documentation, endpoint
information, method examples, request/response samples, and fixture evidence.

When an eligible Gemini configuration, valid Gemini/Google authentication, and
the optional Google ADK extra are available, Harness may use
`google_search` only in onboarding and custom-RPC research flows. Search results
are evidence, not authority to skip validation. Official documentation should
be preferred over blogs or forums.

## Chain And RPC Onboarding Cases

Agent workflow, prompts, tools, and Harness tests must distinguish these three
cases. They share validation principles, but they do not have the same product
outcome.

### Case 1: Supported Chain With Custom RPC Methods

This case applies when the chain is one of the 36 committed chain templates, or
an exact known alias for one of them, and the user wants to add, replace, or
reweight RPC methods for the current job.

The Agent must ask the user for:

- whether the custom method is added to the default workload or replaces the
  default workload;
- the RPC mode: `single` or `mixed`;
- method name;
- params sample, including empty params when the method has no params;
- reachable endpoint for live validation;
- request sample when the user has one;
- response sample when the user has one;
- desired workload weights for every method that remains active.

The Agent must do the following before execution:

1. Keep the canonical `config/chains/<chain>.json` unchanged.
2. Create or update a job-local runtime chain template override for the custom
   workload.
3. Classify user-provided method evidence before probing it. A pasted value may
   be a JSON-RPC method name, JSON-RPC request, REST path, REST endpoint URL,
   response sample, official documentation excerpt, or contradictory evidence.
   The Agent must not treat URLs or REST documentation titles as JSON-RPC method
   names.
4. Probe the endpoint with the proposed method and params. User-provided
   request/response samples are evidence to verify, not facts to trust.
   零参数 method 必须保留明确的空 params。positional/object params 必须保留 list
   顺序或 object key，并逐个确认 index/name、JSON wire type、区块链 semantic type
   或 encoding、meaning、required/optional 和 example。
5. Confirm the method shape can be represented by the current chain template
   schema: `param_formats`, `_meta.rest_paths`, or `param_spec`.
6. If schema support is missing, stop and produce a coding handoff. Do not hide
   schema failures behind generic benchmark errors.
7. Record or generate method-specific fixture evidence for every active custom
   method.
8. Validate workload weights. Mixed workload weights must sum to 100%. If the
   user fully replaces the default workload, remove default methods from the
   runtime override so target generation cannot send unwanted default requests.
9. After a method/schema probe passes, immediately ask whether the user wants to
   add another custom method, finish with the current set, or change how the
   custom methods apply to the workload. The Agent must not loop back to the
   same schema-evidence question after a successful probe.
10. Run preflight and fake-node smoke before treating the custom method as usable
   for the job.

If any validation step fails, the Agent must show the failure reason, endpoint
or fixture evidence path when available, and the next corrective action. It must
not continue to benchmark execution with unverified custom RPC methods.

### Case 2: New Chain In An Existing Adapter Family

This case applies when the chain is not one of the 36 committed templates, the
user has confirmed the intended chain identity, and the protocol appears to fit
an existing adapter family.

The Agent must complete two gates before Case 2 begins:

1. Chain existence/identity gate: use the configured LLM with repository facts
   and current workflow state to decide whether the candidate appears to exist,
   is likely a typo for a supported chain, is unknown, or needs user-provided
   evidence. With an eligible Gemini configuration, valid Gemini/Google
   authentication, and the optional extra, use ADK `google_search` on
   official sources before asking the user to confirm the chain identity.
2. Protocol-family gate: only after the chain identity is confirmed, infer the
   adapter family from repository facts, official docs, user-provided samples,
   and search evidence when available. Ask the user to confirm the proposed
   family or type the correct family. If no existing family can be confirmed,
   route to Case 3.

If the user confirms an existing family, the Agent must require a reachable
endpoint, validate safe RPC methods against it, verify method schema support,
record fixtures, and pass fake-node smoke. Only after those gates pass may the
Agent continue into the normal resource, workload, observability, preflight, and
execution flow. The chain remains job-local until separately reviewed and
committed.

### Case 3: New Chain Outside Existing Adapter Families

This case applies when the user has confirmed the intended chain identity and
the Agent cannot confirm that the chain fits an existing adapter family.

The Agent must not continue benchmark setup. With Gemini `google_search`, it
should gather official docs first after chain identity is confirmed. Without
web research, it must ask the user for official protocol docs, RPC docs,
endpoint docs, request examples, and response examples. The output is a
secondary-development handoff for another coding AI or engineer, including
adapter boundaries, files to edit, schema requirements, fixture requirements,
smoke and coverage gates, documentation updates, and PR expectations.

## Unknown Chain Decision Standard

This section is the product standard for any user input that names a chain or
chain-like target that is not an exact supported template name or exact known
alias for one of the 36 committed chain templates. It applies at every point in
the conversation, including while the Agent is waiting for disk, QPS, endpoint,
RPC method, observability, or final approval answers.

The Agent must not silently map an unknown chain-like value to a supported chain
through prefix matching, fuzzy matching, or provider-name assumptions. It must
route the turn into a two-step chain identity workflow and follow this sequence:

1. Identify the proposed chain name from the user's text without destroying the
   user's original wording. Do not shorten multi-token names to a supported
   prefix, and do not treat partial tokens as supported aliases.
2. Resolve chain existence/identity first. Use the configured LLM with
   repository context and current workflow state. If Gemini plus Google
   authentication is available, use ADK `google_search` against official
   sources before asking the user to confirm whether the chain exists, whether
   it is a supported-chain typo, or whether more docs/evidence are needed.
   Do not write the active `chain`, ask for endpoint, or enter Case 2/3 until
   the user confirms the intended chain identity.
3. Resolve protocol family second. Only after chain identity is confirmed,
   determine whether the chain appears to belong to an existing adapter family.
   If an eligible Gemini configuration, valid Gemini/Google authentication,
   and the optional extra are available, use ADK `google_search`
   to look for official RPC documentation, official endpoint examples, and
   official request/response examples before proposing the family. If web
   research is unavailable, the model may use repository context and model
   knowledge, but must tell the user when protocol family evidence is
   uncertain.
4. Ask the user to confirm the proposed protocol family. If the model cannot
   determine the family, ask the user for the protocol family or for official
   documentation.
5. If the family is supported, require a reachable `LOCAL_RPC_URL` or public RPC
   endpoint. Probe the endpoint before trusting it. User-provided endpoints,
   request samples, response samples, and method documentation are evidence to
   verify, not facts to accept blindly.
6. For each user-selected RPC method, classify the evidence first, then validate
   the live request against the endpoint. JSON-RPC evidence must be probed as
   JSON-RPC. REST path or REST documentation evidence must either match a REST
   adapter flow or force the Agent to ask the user to switch protocol family;
   it must not be sent as a JSON-RPC method string. Confirm the parameter schema
   can be expressed by the current chain template schema (`param_formats`,
   `_meta.rest_paths`, or `param_spec`), and record or generate method-specific
   fixture evidence. If request/response samples conflict with live endpoint
   behavior or official docs, stop and ask the user to correct the evidence.
   After each method validates, ask whether to add another method or finish the
   current method set. When the user finishes, ask how to apply the validated
   methods: `single` uses one validated method and does not need weights;
   `mixed` must list every participating validated method and require weights
   whose total is exactly `100`. Do not silently reuse template defaults for a
   new-chain job-local workload.
7. If fixture recording, template validation, or fake-node smoke passes, the
   Agent may continue into the normal benchmark configuration flow and ask the
   remaining resource, workload, observability, preflight, and execution
   questions. The newly supported chain remains job-local until explicitly
   reviewed and committed.
8. If smoke fails, report the failure reason, evidence paths, and likely
   category. If the failure appears to be framework code or schema support, give
   the user a handoff that another coding AI can use to fix the framework. Do
   not hide the failure or continue to a real benchmark.
9. If the family is unsupported, do not run benchmark setup. If Gemini
   `google_search` is available, gather official docs first. Otherwise ask the
   user for official docs, endpoint docs, protocol docs, and RPC samples. Then
   generate a secondary-development handoff with files to edit, adapter-family
   boundaries, schema requirements, fixture requirements, smoke/coverage gates,
   documentation updates, and PR expectations.

For custom RPC methods on either an existing chain or an onboarded
existing-family chain, the same endpoint-validation rule applies. The Agent must
validate the method against a reachable endpoint, verify params and response
shape, record method-specific fixtures, and ensure workload weights sum to
100%. If the user fully replaces the default workload with custom methods, the
runtime chain template override must remove default methods so target generation
does not send unwanted default RPC requests.

Acceptance tests for this standard must include:

- unknown chain entered at the chain-selection step;
- unknown chain entered while another pending question is active;
- ambiguous chain/provider names such as a provider API that is not the same
  protocol as the familiar chain brand;
- an existing-family chain with a valid endpoint and safe method probe;
- an existing-family chain with a bad endpoint or conflicting samples;
- an unsupported-family chain that produces a secondary-development handoff;
- custom RPC methods with request/response evidence and invalid/valid weights.

When generating a secondary-development plan, the Agent must include:

- files to modify;
- chain template fields;
- adapter or new-family boundaries;
- fake-node fixture requirements;
- smoke and coverage checks;
- documentation updates required to keep the Agent knowledge current;
- PR and CI expectations.

## Error, Evidence, And Report Analysis Boundary

The Agent must support diagnostic and analysis conversations as first-class
workflow groups, not as loose chat.

When the user pastes errors, logs, stack traces, validator output, endpoint
probe output, benchmark output, or unfamiliar terminal text:

- treat the pasted content as evidence, not as a configuration-field answer;
- pause the current configuration group;
- read referenced local files when paths are available;
- combine the evidence with framework facts, runtime state, relevant docs, and
  generated artifacts before model analysis;
- return likely root cause, evidence path or summary, category, next action,
  and whether the paused group can continue;
- never stuff pasted logs into fields such as `CLOUD_REGION`, `LEDGER_DEVICE`,
  RPC params, or endpoint URLs unless the user explicitly identifies that line
  as the value.

When the user asks about reports, charts, CSVs, bottlenecks, per-method
attribution, latency, success rate, or N/A data:

- resolve the requested or latest job;
- read structured artifacts such as `test_summary.json`,
  `performance_latest.csv`, per-method CSVs, sync-health CSVs, and report
  paths before model analysis;
- distinguish fake-node smoke evidence from real-node performance evidence;
- cite artifact paths used for the conclusion;
- explain missing, zero, or N/A data as data-collection or scenario evidence
  when that is what the artifacts show;
- preserve the report context for follow-up questions until the user switches
  job or starts a new benchmark.

## Documentation Boundary

When Agent behavior, benchmark behavior, configuration, or extension contracts
change, documentation must change in the same PR.

Keep:

- user-facing README content;
- operator guides such as `AGENTS.md`;
- architecture and gate documents;
- framework references;
- chain/RPC extension guides;
- closed-loop testing guides;
- PR and contribution workflow docs.

Do not commit:

- temporary task plans;
- debugging transcripts;
- one-off implementation plans;
- model work logs;
- generated runtime state, benchmark archives, terminal captures, or local
  credentials.

English and Chinese docs should stay aligned for long-lived public
documentation. If a document intentionally exists in only one language, explain
that in the nearest docs index.

## Verification Matrix

Agent code changes must run the smallest relevant tests first. For broad Agent
workflow changes, run:

```bash
python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract tests.test_agent_langgraph_harness
python3 tools/check_agent_boundaries.py --root .
git diff --check
```

When live model behavior is affected and a safe key is available, run the
LangGraph live CLI matrix in Docker/Linux。产品 acceptance 和 coverage evidence
只接受 Docker 运行；host run 只能作为开发检查。benchmark execution 变更必须在
Docker 中运行适用的 fake-node、local real-node、custom-RPC 或 sync-observe 路径；
fake-node 不能替代其他 workflow path。

对于广泛影响 Agent workflow、group state、routing 或 Harness 的变更，单独运行
LangGraph live CLI matrix 仍然不够。还必须运行真实双 AI Chaos：真实
`./bin/anychain-agent` CLI 使用已配置的 live model，Codex 扮演用户，并且每轮读取
Agent 的最新实际回复后才决定下一条输入。预先写好的 prompt 序列不属于双 AI
Chaos。完整 transcript 必须包含真实用户式的中断、回退、跨 group 跳转、语言切换、
复制粘贴配置或 evidence、自定义 RPC、未知链、session resume，以及最终 review、
preflight 和 smoke。未 review 完整 transcript 时不得声称产品验收完成。

生成的 schedule、cataloged edge、covering row/tuple 和成功返回的 PTY 文本都只是
test intent 或 transport evidence，不是 observed execution。pass 必须绑定当前 revision
的实际 state transition 与独立验证的 postcondition；execution edge 还必须包含 hash
绑定的 job artifact。报告必须分开列出 generated/cataloged、observed-pass、
observed-fail、not-run、externally-blocked 和 uncovered denominator。

Codex 用户模拟器必须使用明确 persona，不能只走中性的 happy path。至少包括：

- 第一次使用且困惑的评估者；
- 急躁的运维工程师；
- 大量复制粘贴内容的技术用户；
- 不断改变需求、回退和跨 group 跳转的用户；
- 中英文混合用户；
- 日志、错误与报告分析用户；
- 自定义 RPC 集成用户；
- Case 2 新链评估用户；
- Case 3 unsupported-family handoff 用户；
- 恢复历史 session 的用户。

第一次使用且困惑的评估者是基线 Gate，并且必须从非空 checkpoint 开始，例如尚未
完成的 `sync-observe / bsc` session。它应询问“你是谁”“你从哪里来”“你能做什么”
“应该从哪里开始”“fake-node、real-node、sync-observe 有什么区别”等基础问题，
随后切换到 fake-node。Harness 必须显式处理旧 session 和 stale chain/mode，不能
静默复用旧链。该 transcript 还要继续覆盖资源确认、workload、QPS、observability、
preflight/smoke；对执行确认的正向回答必须真正执行或返回具体 blocker。之后关于
“执行过测试么”“当前状态是什么”“下一步是什么”的问题必须依据 checkpoint、job、
preflight 和 smoke state 回答，不能退化为通用 workflow 文案。

If a boundary cannot be tested locally, report it as untested. Do not describe
untested behavior as complete.

## Review Checklist

Before finishing an Agent task, answer these internally:

- Did the change preserve LangGraph Harness-owned intent routing and group state?
- Did terminal code remain a stable I/O shell rather than a business router?
- Did every new execution path pass through validators?
- Can users override inferred values?
- Can users correct previous answers?
- Does fake-node testing still follow the same resource-confirmation contract
  as real-node testing?
- Are custom RPC methods handled through chain template, parameter samples,
  fixtures, and workload validation?
- Are docs and framework knowledge updated when behavior changes?
- Are generated files, secrets, live logs, and local credentials excluded from
  commits?
