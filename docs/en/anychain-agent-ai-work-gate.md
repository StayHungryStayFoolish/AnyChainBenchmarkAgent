# AnyChain Agent AI Work Gate

This document is the project-specific gate for AI coding work on AnyChain
Agent. It complements the repository-level `AI_CODING_GUIDE.md` behavior contract.

Before changing Agent code, an AI coding agent must read:

1. `AI_CODING_GUIDE.md`.
2. This gate document.
3. `docs/en/adk-agent-architecture.md`.
4. `agent/README.md`.
5. The current reviewed task/design document for the Agent change.
6. The exact files it plans to edit.

If these rules conflict with an implementation shortcut, the rules win.

## Documentation Gate Before Agent Code

Do not change Agent workflow code until the task is documented.

For any change that affects Agent behavior, Harness prompts, LLM routing,
workflow branches, terminal interaction, workflow state, benchmark execution,
validators, runner lifecycle, fake-node smoke, endpoint validation, or
onboarding:

1. Read `AI_CODING_GUIDE.md`, this file,
   `docs/en/adk-agent-architecture.md`, and the current reviewed task/design
   document for the Agent change.
2. Confirm the task document names the root problem, exact scope, files to
   inspect, files that must not be changed, expected user-facing behavior,
   validation commands, and acceptance evidence.
3. If the document is missing, stale, vague, or inconsistent with code, update
   the document first and review it before touching code.
4. Only then modify code, and keep each code change traceable to the documented
   task.

Forbidden shortcuts:

- patching one terminal transcript without updating the decision tree or task
  document;
- adding local if/else, regex, fuzzy matching, phrase cleanup, or fallback
  behavior to hide a weak Agent workflow;
- creating new helper files before proving the existing architecture needs
  them;
- keeping old code by renaming or isolating it when the design requires
  deletion or migration;
- claiming a phase is complete without running the documented gates.

If a test or live CLI run exposes a new failure, stop and classify it in the
task document before coding the fix. The fix must address the root cause in
LangGraph Harness workflow, typed state, deterministic tools, validators, or
terminal I/O boundaries, not just the exact wording of the failed prompt.

The classification must also state why a local patch is not the right product
fix. If the selected implementation uses a local patch anyway, the task
document must explicitly justify the tradeoff, define its removal condition,
and add Harness coverage so it cannot become hidden technical debt. Otherwise,
do not write the patch.

Passing one transcript is not enough. The proposed fix must preserve the
decision tree, typed `pending_question` contract, validator sequence, user
correction path, and benchmark execution gates across neighboring branches.

For configuration dialogue failures, classification must identify whether the
problem is:

- missing or stale `pending_question`;
- a visible multi-question prompt;
- a missing deterministic next-question transition;
- smoke-only values leaking into full benchmark configuration;
- terminal code attempting to compensate for a missing workflow transition.

## Non-Negotiable Product Boundary

AnyChain Agent is a LangGraph Harness-based domain agent for blockchain node
benchmarking. The Harness owns product workflow state, group routing,
fallback ordering, validation gates, and execution decisions. Google ADK is
used only by the optional Gemini `google_search` grounding function; it is not
an Agent, Runner, workflow owner, or general tool bridge. The Agent must
reduce user configuration burden and call deterministic benchmark tools
safely. It is not a shell script wizard, not a keyword router, and not a
collection of fallback demos.

The product loop is:

```text
Understand -> Plan -> Ask -> Configure -> Validate -> Execute -> Observe -> Analyze -> Iterate
```

The configured model helps interpret ambiguous natural-language turns into
typed Harness actions. LangGraph Harness owns planning, group selection,
question selection, fallback ordering, and iteration. Repository tools own
deterministic checks, configuration materialization, benchmark execution,
evidence collection, and artifact-backed analysis.

## Decision Tree And Y/N Contract

The complete workflow map and product Harness plan must be present in the
current reviewed task/design document for the Agent change. Do not make Agent
workflow changes from this short checklist alone.

Every yes/no answer must be tied to one active pending question. The Agent must
not ask a yes/no question unless the accepted and declined paths are explicit.

Required behavior:

- If there is no pending question, a bare `Y`, `N`, `yes`, or `no` is not a
  valid business decision. Ask what the user wants to confirm.
- If the pending question is dependency installation, `Y` installs dependencies
  through the Agent tool and `N` declines installation.
- If the pending question is target mode, `fake-node`, `real-node`, or a
  numbered/manual choice advances that target-mode path.
- If the pending question is a disk choice, a number selects the listed disk and
  a manual value overrides the inferred value.
- If the pending question is quick assumed fake-node smoke, `Y` must submit the
  detached smoke job without asking another confirmation for the same action.
- If the user changes their mind, says a previous answer was wrong, or asks to
  go back, update or revert workflow state and re-run validators before
  continuing.

After a detached job is submitted, the Agent should provide `job_id`, run
directory, `benchmark.log`, and the commands/options for `status`, `logs`, and
`follow`. It must not ask an unregistered yes/no question such as "view logs
now?" unless that pending action is stored in workflow state.

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
`follow`, and `exit` are allowed. Business requests must go through LangGraph
Harness routing and group workflows.

## Legacy-Code Pollution Gate

Before repairing Agent workflow behavior, audit the existing `agent/` code for
old custom-Agent logic. Do not preserve code merely because it is currently
imported.

Allowed retained code must fit one of these roles:

- LangGraph Harness runtime, group workflow, checkpoint, and typed event code;
- the optional Gemini `google_search` grounding function in
  `agent/llm/search_grounding.py`; it is not an Agent, Runner, workflow owner,
  or general tool-wrapper layer;
- deterministic AnyChain planners, validators, runners, analyzers, discovery,
  onboarding, or knowledge providers;
- terminal I/O, exact shell commands, Ctrl+C/log-follow handling, dependency
  consent, and user-visible progress;
- developer utilities and tests that are clearly outside the product runtime.

Remove or migrate:

- old non-Harness benchmark wizards;
- fallback custom brains or mock agents;
- terminal keyword/fuzzy/regex business routing;
- phrase-repair loops that try to hide bad planning;
- duplicate workflow state machines;
- lifecycle-only mock smoke paths presented as user-facing smoke;
- dead files kept only because old imports reference them.

If useful deterministic behavior exists inside obsolete code, move that
behavior to the correct planner, validator, runner, analyzer, onboarding, or
knowledge component. The old conversational wrapper should not remain.

The Agent is not product-ready while legacy code can bypass LangGraph Harness
intent routing, typed pending questions, deterministic validators, preflight,
smoke, or user approval gates.

The current high-risk areas that must be reviewed before more Agent code
repair are:

- `agent/adk_app/`: must not exist, in whole. The entire package was an
  ADK-native `Agent`/`Runner` tool-calling surface that duplicated the
  Harness's conversation loop and was never the shipped product's entrypoint
  (see `agent/README.md`'s "Retired files must not return"). Broad LLM
  rewrite, regex phrase-repair, or a standalone conversation loop belong
  nowhere in the product path. Product behavior should be fixed in Harness
  prompts/instructions, typed state, tools, and validators. The one exception
  is `agent/llm/search_grounding.py`, the sole permitted `google-adk`
  consumer (optional Gemini `google_search` grounding, called as a plain
  function from Harness code — never a second conversation loop).
- `agent/terminal/repl.py`: may keep exact terminal controls and safe I/O, but
  must not contain benchmark-domain intent routing or field extraction.
- `agent/harness/`: must remain the single product workflow runtime. LangGraph
  checkpoint state, typed pending questions, group transitions, validators,
  rollback/jump behavior, and next-blocking-group selection belong here.
  Retired file-backed workflow state machines must not return.
- `agent/tools/schema.py`, `agent/tools/executor.py`, and
  `agent/runners/job_manager.py`: any `mock` lifecycle support must be removed
  from the product Agent execution path. If it remains for developer or
  enterprise-platform tests, it must be explicitly named as non-product test
  support and unreachable from normal terminal benchmark flows. User-facing
  smoke must be complete traffic, proxy, monitoring, report, archive, and
  artifact discovery.

Preserve deterministic domain tools unless a concrete replacement exists:
`agent/discovery`, `agent/diagnostics`, `agent/planners`, `agent/validators`,
`agent/runners`, `agent/analyzers`, `agent/onboarding`, and `agent/knowledge`
are the benchmark engine's domain surface. The repair task is to wire them into
the Harness correctly, not to replace them with model prose.

Do not use "isolate legacy code" as a cleanup outcome. Isolation is acceptable
for Python environments, smoke output directories, generated job artifacts, or
test fixtures. It is not acceptable as a way to keep old conversational logic,
fallback brains, phrase-repair loops, keyword routing, or duplicate state
machines in the source tree. Legacy product logic must be deleted, or its
useful deterministic behavior must be migrated into the correct domain module
with tests.

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

`sync-observe` is a separate workflow type, not a quick/standard/intensive
profile. It observes node sync and resource behavior without RPC workload,
proxy traffic, Vegeta, or QPS ramp. Its validators must confirm chain and
sync-health/reference behavior, resource metadata, node process identity,
optional node Prometheus metrics endpoint, and stop condition: until stopped,
fixed duration, or until synced. It must not ask for RPC mode, custom RPC
method, mixed weights, or QPS profile unless the user switches back to an RPC
benchmark workflow.

This means no Vegeta-related runtime path is expected in `sync-observe`: do
not generate Vegeta targets, do not start the RPC proxy for workload traffic,
do not run the QPS executor, and do not require `vegeta_results` artifacts for
report success. Sync-observe reports are generated from monitoring, sync-health,
node execution, disk, CPU, and network data.

`sync-observe` does not require fake-node fixture recording. A real node may
start from a downloaded peer snapshot and then catch up from that snapshot
height; the Agent should observe that real sync behavior rather than record it
as RPC fixtures. When the user provides a local node endpoint or public
reference endpoint for sync observation, reuse the existing endpoint/sync-health
sanity checks to prove the endpoint is reachable and can expose height or sync
state. Do not enter custom-RPC fixture recording, target sample collection, or
workload schema validation unless the user explicitly switches back to RPC
benchmark/custom-RPC onboarding.

Users may request `sync-observe` from any benchmark setup group, including
chain/endpoint, workload, custom RPC, mixed weights, QPS, observability,
preflight, or report follow-up. The Agent must pause the current group, switch
the workflow type to `sync_observe`, preserve reusable environment/resource/
chain state, invalidate RPC-only state, and ask only sync-observe blockers. If
the user later switches back to an RPC benchmark, the Agent must re-ask RPC
workload and QPS gates instead of reusing invalidated state. Harness acceptance
must prove the `--sync-observe` command path does not invoke proxy, Vegeta, or
RPC target generation.

## Configuration Group Workflow Standard

The Agent configuration flow is not a fixed one-way wizard. It is a set of
configuration groups with a default order, global natural-language routing, and
validator-driven recovery. Users may jump between groups, correct earlier
answers, change target chain or mode, paste evidence, or ask questions at any
time. The Agent must route the turn to the right group, preserve valid state,
invalidate affected state, and then return to the next blocking group in the
default order.

The default order should match user mental model and manual configuration
order:

1. Provider and deployment group: cloud provider, region, zone, machine type,
   VM vs Kubernetes/container, and platform such as GCE, EC2, GKE, EKS, or
   self-hosted Kubernetes.
2. Hardware group: CPU and memory discovery, network inventory, disk inventory,
   and resource metadata.
3. Ledger disk group: `LEDGER_DEVICE`, `DATA_VOL_TYPE`, `DATA_VOL_SIZE`,
   `DATA_VOL_MAX_IOPS`, and `DATA_VOL_MAX_THROUGHPUT`.
4. Accounts disk group: ask whether a separate accounts/state disk exists; if
   yes, collect `ACCOUNTS_DEVICE`, `ACCOUNTS_VOL_TYPE`, `ACCOUNTS_VOL_SIZE`,
   `ACCOUNTS_VOL_MAX_IOPS`, and `ACCOUNTS_VOL_MAX_THROUGHPUT`.
5. Network group: `NETWORK_INTERFACE`, selected from detected interfaces when
   there is more than one candidate, and `NETWORK_MAX_BANDWIDTH_GBPS`.
6. Chain and endpoint group: `BLOCKCHAIN_NODE`, target mode, `LOCAL_RPC_URL`,
   `MAINNET_RPC_URL` or template sync-health behavior, and
   `BLOCKCHAIN_PROCESS_NAMES` for real-node monitoring.
7. Chain auxiliary endpoint group: chain-template-driven optional overrides
   such as `CHAIN_REST_URL`, `CHAIN_INDEXER_URL`, `CHAIN_SIDECAR_URL`,
   `CHAIN_EVM_RPC_URL`, `CHAIN_JSON_RPC_URL`, `CHAIN_MIRROR_URL`, and
   `RPC_API_KEY`.
8. Workload group: `RPC_MODE`, default workload selection, custom RPC methods,
   mixed weights, runtime chain template override, endpoint validation,
   fixtures, and workload validation.
9. Target sample and fixture group: only the `TARGET_*` values required by the
   selected chain template, methods, adapter family, and custom RPC schema.
10. QPS profile group: quick, standard, or intensive profile; explain defaults
    first; ask whether to keep defaults; if not, collect initial QPS, max QPS,
    QPS step, duration, and relevant cooldown/warmup fields.
11. Sync-observe group: only when the user wants to observe node sync/resource
    behavior without RPC benchmark load. Confirm sync-health/reference
    behavior, node process identity, optional `NODE_PROMETHEUS_METRICS_URL`,
    and stop condition. Skip workload and QPS groups while this workflow is
    active.
12. Observability group: disabled, local Prometheus/Grafana, or exporter-only;
    then ports, auto-stop behavior, scrape endpoint guidance, and port checks.
13. Advanced tuning group: optional account discovery settings, monitoring
    intervals, disk monitor rate, internal bottleneck thresholds, success-rate
    threshold, latency threshold, and other `internal_config.sh` values. The
    Agent must explain these before asking whether the user wants to change
    them.
14. Preflight, smoke, and execution approval group: validate config, run
    preflight, run complete closed-loop smoke, show evidence, and ask for
    approval before the real benchmark job.

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

For a chain outside the supported templates, the Harness must not jump directly to
adapter-family selection, endpoint collection, or development handoff. It must
first resolve whether the chain identity appears to exist, then confirm the
protocol/adapter family. If framework knowledge is insufficient, ask the user
for official chain documentation, protocol/RPC documentation, endpoint
information, method examples, request/response samples, and fixture evidence.

When an eligible Gemini configuration, valid Gemini/Google authentication, and
the optional Google ADK extra are available, the Harness may use
`google_search` only in onboarding and custom-RPC research flows. Search results
are evidence, not authority to skip validation. Official documentation should
be preferred over blogs or forums.

Google ADK is an optional dependency, installed explicitly with
`scripts/install_agent_deps.sh --with-google-search`. The core LangGraph
terminal runtime and DeepSeek/OpenAI/Claude/non-search Gemini operation must
start without it. The retained `requirements-adk.txt`, `.venv-adk`, and
`--adk-venv` names are compatibility aliases only.

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
   Preserve an explicit empty params value for zero-parameter methods. For
   positional or object params, preserve list order or object keys and confirm
   each parameter's index/name, JSON wire type, blockchain semantic type or
   encoding, meaning, required/optional status, and example separately.
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
LangGraph live CLI matrix in Docker/Linux. Product acceptance and coverage
evidence are Docker-only; host runs are developer checks. When benchmark
execution is affected, run the applicable fake-node, local real-node,
custom-RPC, or sync-observe path in Docker. fake-node alone does not qualify
the other workflow paths.

For broad Agent workflow, group-state, routing, or Harness changes, the
LangGraph live CLI matrix is not sufficient by itself. Run a dual-AI chaos
session where the real `./bin/anychain-agent` CLI uses the configured live model
and Codex acts as the user simulator, choosing each next turn dynamically from
the Agent's latest response. The transcript must include real user-style
interruptions, backtracking, group jumps, language switching, pasted
configuration/evidence, custom RPC cases, unknown-chain cases, resume behavior,
and final review/preflight/smoke paths. If the full transcript is not reviewed,
do not claim product readiness.

Generated schedules, cataloged edges, covering rows/tuples, and successful PTY
returns are test intent or transport evidence, not observed execution. Passing
coverage must be revision-bound to an actual state transition and independently
verified postcondition; execution edges additionally require hashed job
artifacts. Reports must keep generated/cataloged, observed-pass, observed-fail,
not-run, externally-blocked, and uncovered denominators separate.

The Codex user simulator must use explicit personas rather than a neutral
happy-path checklist. Required personas include a first-time confused evaluator,
an impatient operations engineer, a copy/paste-heavy technical user, a
requirement-changing user, a mixed-language user, a report/debug analyst, a
custom-RPC integrator, a new-chain evaluator, an unsupported-chain handoff
evaluator, and a resume-session user.

The first-time confused evaluator is the baseline gate and must start from a
non-empty previous checkpoint such as partial `sync-observe / bsc`. It must ask
basic questions like `who are u?`, `你从哪里来`, `你要去哪里`, `你可以做什么？`,
`那我们现在可以从哪里开始？`, and
`fake node，real node，sync observe 都是什么？`, then switch to `fake node`.
The Harness must explicitly handle the old session and stale chain/mode state;
it must not silently reuse the previous chain. The same transcript must continue
through resource confirmation, workload, QPS, observability, and
preflight/smoke approval. A positive answer to the preflight/smoke prompt must
execute that path or return a concrete blocker. Afterward, questions like
`你执行过测试了么？`, `当前是什么状态？`, `那你接下来要做什么？`, and
`那你该做什么了？` must be answered from checkpoint/job/preflight/smoke state,
not with generic workflow prose or "no pending question" text.

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
