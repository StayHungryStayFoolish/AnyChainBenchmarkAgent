# Agent CLI Verification Guide

This guide is for an independent AI assistant that needs to verify the
AnyChain Benchmark Agent from a real terminal. It is a long-lived verification
checklist, not a temporary implementation plan.

Use fake-node closed-loop tests only unless the repository owner explicitly
provides a real endpoint for that run. Do not use private production endpoints,
customer data, or personal credentials in shared evidence.

## Goal

Verify that `./bin/anychain-agent` behaves like a product Agent:

- starts cleanly in a real terminal;
- uses the configured LLM through Google ADK;
- correctly reports the configured auth mode, including Google ADC or attached
  service-account modes when available;
- keeps input/output stable for English and Chinese;
- detects the local environment and missing dependencies;
- asks for missing benchmark variables one item at a time;
- supports fake-node benchmark preparation without skipping resource
  validation;
- handles ambiguous answers, corrections, backtracking, and mode changes;
- keeps long-running jobs detached and resumable;
- uses Gemini-only ADK `google_search` only for unsupported-chain and custom-RPC
  onboarding evidence.

## Required Reading

Read these files before testing or changing code:

1. `AGENTS.md`
2. `AI_CODING_GUIDE.md`
3. `README.md`
4. `agent/README.md`
5. `docs/en/anychain-agent-ai-work-gate.md`
6. `docs/en/adk-agent-architecture.md`
7. `tests/agent_live/README.md`

## Environment Rules

- Start from a clean checkout of the target branch and record the commit hash.
- Use an isolated Python/ADK environment.
- Run `bash scripts/install_agent_deps.sh --yes` before terminal testing, or
  let the Agent request approval to install missing Agent dependencies.
- Do not commit API keys, ADC files, service account JSON, `.agent/`, live logs,
  generated benchmark archives, or terminal recordings containing secrets.
- Redact credentials, local usernames, hostnames, internal project IDs, and
  private endpoint URLs from shared evidence.
- Record whether the configured model supports ADK `google_search`.
- If Google/Gemini search is unavailable, still run all non-search terminal
  and fake-node boundaries.
- Put local model credentials only in `config/agent_config.local.sh` or the
  execution environment. Never edit repository defaults with real secrets.
- Do not claim that real-node execution was tested unless an approved endpoint
  was explicitly provided for that run.
- Fake-node validation is enough for Agent CLI workflow coverage, but the
  Agent must still collect the same machine/resource metadata it would need for
  real-node execution.

## Baseline Commands

```bash
git status --short --branch
git rev-parse HEAD
bash scripts/install_agent_deps.sh --yes
python3 agent/cli.py adk-status
python3 agent/cli.py llm-config
./bin/anychain-agent
```

Expected startup behavior:

- startup diagnostics run automatically;
- model provider, model name, auth mode, and web-research status are shown;
- when `LLM_AUTH_MODE=google_adc`, ADC presence and quota/project errors are
  surfaced clearly without printing credential contents;
- previous job state is shown when `.agent/jobs` has existing jobs;
- missing dependency messages do not block normal conversation unless the
  missing dependency is required for the requested action;
- the terminal prompt remains stable.

If startup reports missing benchmark-engine dependencies, ask the Agent to
explain what it will install and verify it requests approval before invoking
`scripts/install_deps.sh --yes`.

## Automated Checks

Run before any fix and again after any fix:

```bash
python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract
python3 tools/check_agent_boundaries.py --root .
python3 agent/cli.py adk-eval
git diff --check
```

Then run the live matrices with the configured model:

```bash
python3 tests/agent_live/run_live_matrix.py \
  --matrix tests/agent_live/agent_intent_smoke_scenarios.json \
  --require-live \
  --provider <provider> \
  --model <model> \
  --timeout 240

python3 tests/agent_live/run_live_matrix.py \
  --matrix tests/agent_live/agent_product_acceptance_scenarios.json \
  --require-live \
  --provider <provider> \
  --model <model> \
  --timeout 240

python3 tests/agent_live/run_live_matrix.py \
  --matrix tests/agent_live/agent_chaos_conversation_scenarios.json \
  --require-live \
  --provider <provider> \
  --model <model> \
  --timeout 240

python3 tests/agent_live/run_live_matrix.py \
  --matrix tests/agent_live/agent_edge_acceptance_scenarios.json \
  --require-live \
  --provider <provider> \
  --model <model> \
  --timeout 240
```

Use `provider=gemini` only when Gemini credentials are configured. Use another
repository-supported provider for non-search live validation, but do not claim
Google Search coverage unless Gemini ADK `google_search` is actually available.

## Manual Terminal Tests

Run `./bin/anychain-agent` in a real terminal, not only through scripted
prompts.

### 1. Startup, Auth, And Local Discovery

Prompt:

```text
doctor
```

Expected:

- Agent reports provider, model, auth mode, and web-research status;
- Agent reports cloud/deployment discovery such as GCE, GKE, EC2, EKS,
  generic Kubernetes, VM, container, or unknown;
- Agent reports CPU, memory, network interface candidates, and disk candidates
  when the host exposes them;
- when metadata services are unavailable, Agent says what is unknown instead
  of inventing cloud, region, zone, or machine type;
- if benchmark dependencies are missing, Agent asks for installation approval
  before using `scripts/install_deps.sh --yes`;
- if Agent runtime dependencies are missing, the launcher asks before using
  `scripts/install_agent_deps.sh --yes`.

For ADC/Gemini environments, also verify:

```bash
python3 agent/cli.py llm-config
python3 agent/cli.py llm-smoke --prompt 'Return JSON only: {"ok": true}'
```

Expected:

- `llm-config` reports the configured provider/auth mode without exposing
  secrets;
- `llm-smoke` succeeds or returns a clear auth/quota/model error;
- ADK `google_search` is reported as enabled only for Gemini-family configs
  that can actually import and use the ADK search tool.

### 2. Terminal Editing And Interrupts

Test:

1. Type Chinese text with a typo, delete part of it, and retype.
2. Type English text with a typo, delete part of it, and retype.
3. Press `Ctrl+C` at an empty prompt.
4. Press `Ctrl+C` while entering text.

Expected:

- no ghost spaces;
- no duplicate `User>` prompts;
- `Ctrl+C` clears the current input or exits the current follow/log mode;
- `Ctrl+C` does not stop a detached benchmark job;
- response language follows the latest user intent except for code, variable
  names, paths, and commands.

Also test mixed-language turns:

```text
User> 我要测试 solana
User> use fake-node first
User> 改成 mixed，getSlot 70%，getBlockHeight 30%
```

Expected:

- Agent preserves technical tokens such as `solana`, `fake-node`, `mixed`,
  `getSlot`, and `getBlockHeight`;
- natural-language explanations follow the latest user language.

### 3. Fake-Node Benchmark Flow

Prompt:

```text
I want to benchmark Solana using fake-node first.
```

Expected:

- Agent asks whether this is fake-node or real-node if unclear;
- Agent confirms chain, RPC mode, benchmark mode, QPS profile, observability,
  process names, disk metadata, and network metadata;
- fake-node may default RPC endpoint values, but it must still validate
  resource metadata;
- Agent does not submit benchmark execution without preflight, smoke, and user
  approval.
- any smoke artifacts are written into isolated job/runtime directories and do
  not overwrite a later real benchmark run.

### 4. Benchmark Mode And QPS Profile

Prompt:

```text
Use quick mode. What does quick mean, and what can I change?
```

Repeat for `standard` and `intensive`.

Expected:

- Agent explains the selected profile in user-facing language;
- Agent names the relevant tunable values, such as initial QPS, max QPS,
  QPS step, duration, and RPC mode, without forcing the user to edit config
  files directly;
- Agent asks whether to accept defaults or adjust a specific item;
- if the user changes one item, only that item changes and validators re-run.

### 5. RPC Mode, Weights, And Custom Methods

Prompt:

```text
Use mixed workload with getSlot 70% and getBlockHeight 30%.
```

Expected:

- weights are normalized or rejected with a clear reason if they do not sum to
  100%;
- per-method attribution is not lost for weighted methods;
- Agent asks whether to use default chain-template methods or add custom RPC
  methods.

### 6. Multi-Disk And Optional Accounts Disk

If the host has multiple disks, verify that the Agent shows numbered disk
candidates. If it does not, ask:

```text
If lsblk shows three disks, how will you ask me to choose LEDGER_DEVICE and ACCOUNTS_DEVICE?
```

Expected:

- numbered choices are shown;
- manual override is offered;
- `ACCOUNTS_DEVICE` is explicitly optional;
- disk baseline fields are requested when a device is selected.

### 7. Backtracking And Corrections

Prompt:

```text
I selected the wrong LEDGER_DEVICE. Go back and change it to <device>.
```

Expected:

- Agent acknowledges the correction;
- workflow state is updated or reverted;
- validators are re-run;
- next blocking question is asked.

### 8. Fake-Node To Real-Node Delta

Prompt:

```text
I tested fake-node. Now switch the same plan to a real node.
```

Expected:

- Agent reuses already confirmed resource metadata;
- Agent asks only for the delta: real `LOCAL_RPC_URL`, `MAINNET_RPC_URL` or
  sync-health decision, chain changes, RPC mode, method weights, and final
  approval;
- Agent does not force the user through the entire setup again.

If no real endpoint is provided, do not run the real-node benchmark. Verify the
planning conversation only.

### 9. Custom RPC Method

Prompt:

```text
Add a custom RPC method with three parameters to a mixed workload.
```

Expected:

- Agent asks for method name, parameter contract, parameter samples,
  request/response samples, mixed weight, and fake-node fixture plan;
- Agent does not claim production support before fixture recording and smoke
  validation.

### 10. Unsupported Chain

Prompt:

```text
I want to test a chain that is not in the current 36 templates.
```

Expected:

- Agent checks whether the chain belongs to an existing adapter family;
- if Gemini ADK `google_search` is available, the onboarding path may use it to
  find official RPC docs and examples;
- Agent asks for missing official docs, endpoints, request/response samples,
  and fixture evidence;
- Agent produces a coding handoff plan instead of pretending support exists.

Test both variants:

```text
The new chain is JSON-RPC and looks EVM-like.
The new chain does not fit any current adapter family.
```

Expected:

- same-family onboarding asks for chain-template evidence and fake-node
  fixtures;
- new-family onboarding asks for protocol docs, extractor design, adapter
  design, fixture recording, and smoke-test requirements;
- Agent does not generate unsupported production claims.

### 11. Knowledge Base Status

Prompt:

```text
Do you have an enterprise Knowledge Base connected?
```

Expected:

- Agent reports whether KB is disabled, noop, HTTP, or custom;
- if disabled, Agent explains it will answer from repository state and
  generated artifacts;
- if HTTP/custom is configured, Agent runs or suggests `knowledge-smoke`;
- KB errors do not block local fake-node benchmark validation.

### 12. Observability Choice

Prompt:

```text
Should I use the built-in Prometheus/Grafana or connect to an existing environment?
```

Expected:

- Agent explains disabled, local Prometheus/Grafana, and exporter-only modes;
- local ports are checked before local stack startup;
- for existing Prometheus/Grafana, Agent explains the exporter endpoint and
  that the external Prometheus must scrape it.
- local Prometheus/Grafana ports are not assumed available; conflicts must be
  surfaced before startup.

To force a port-conflict explanation without starting services, ask:

```text
What will you check before opening Prometheus on 9091 and Grafana on 3001?
```

Expected:

- Agent mentions port availability, exporter mode, auto-stop behavior, and
  user confirmation before local stack startup.

### 13. Detached Jobs, Logs, And Resume

After all gates pass, run only a short fake-node benchmark.

Expected:

- job runs detached by default;
- terminal shows where logs and job state are written;
- `jobs`, `status`, `logs <job_id>`, and `follow <job_id>` work through the
  Agent;
- `Ctrl+C` exits log-follow mode but does not kill the detached job;
- restarting `./bin/anychain-agent` from the same repository recovers the
  latest job state.

### 14. User Disorder And Contradictions

Test users who speak out of order:

```text
I want a benchmark.
Actually use real node.
No, switch back to fake-node.
Use mixed.
Wait, make it single.
The disk I gave before was wrong.
```

Expected:

- Agent updates intent and workflow state instead of appending contradictory
  values;
- previous answers are reused only when still valid;
- changed answers trigger re-validation;
- Agent asks the next blocking question instead of dumping a large checklist.

### 15. Out-Of-Scope Requests

Prompt:

```text
Write a trading bot for this chain.
```

Expected:

- Agent declines or redirects to benchmark-relevant capabilities;
- Agent does not execute unrelated shell commands;
- Agent can still answer benchmark-related follow-up questions afterward.

## Fixing Failures

If a scenario fails twice, inspect the redacted logs and fix the smallest
responsible code path:

- prompt or agent instruction issue: `agent/adk_app/instructions.py`;
- deterministic guard or tool issue: `agent/validators/` or
  `agent/adk_app/tools/`;
- terminal UX issue: `agent/terminal/`;
- workflow state issue: `agent/adk_app/workflow/`;
- live matrix gap: `tests/agent_live/*.json`;
- documentation drift: update the relevant README or docs page.

Do not fix business behavior by adding keyword lists, fuzzy matching, or regex
intent routing in terminal code. Intent understanding must remain model-driven
through ADK, with deterministic tools used as validation and execution gates.

## Evidence To Return

Return a concise report with:

- commands run;
- commit hash and branch tested;
- provider/model/auth mode used, with secrets redacted;
- whether Gemini ADK `google_search` was available;
- live matrix output paths;
- manual terminal cases tested;
- failures found and files changed;
- tests run after fixes;
- boundaries still untested, if any.

Use this report shape:

```text
Summary:
- ...

Environment:
- branch:
- commit:
- provider/model/auth:
- google_search:

Automated checks:
- command: result, evidence path

Manual terminal boundaries:
- boundary: pass/fail, notes

Fixes:
- file: reason

Remaining gaps:
- ...
```
