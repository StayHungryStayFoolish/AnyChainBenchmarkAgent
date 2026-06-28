# ADK Terminal Verification Plan

This plan is for an independent AI agent that has access to a terminal,
Gemini-compatible authentication, and Google ADK `google_search`. It must use
fake-node only. Do not use real blockchain nodes or private production data.

The goal is to verify AnyChain Agent behavior from the user terminal, not to
re-implement the Agent. If a problem is found, fix the smallest relevant code
path, run the listed checks, and record evidence.

## Required Reading

Before testing or changing code, read:

1. `CLAUDE.md`
2. `docs/zh/anychain-agent-ai-work-gate.md`
3. `agent/README.md`
4. `tests/agent_live/README.md`
5. `docs/zh/adk-agent-architecture.md`

## Environment Rules

- Use fake-node closed-loop tests only.
- Do not use personal, customer, or production endpoints.
- Do not commit API keys, ADC files, service account JSON, `.agent/`, live logs,
  generated benchmark archives, or terminal recordings containing secrets.
- Redact credentials and local hostnames from all shared logs.
- Run inside an isolated Python/ADK environment.
- If dependencies are missing, let `./bin/anychain-agent` ask for approval or
  run `bash scripts/install_agent_deps.sh --yes` explicitly before starting.

## Baseline Commands

```bash
bash scripts/install_agent_deps.sh --yes
./bin/anychain-agent
```

If the terminal has Gemini and Google authentication available, verify startup
prints a model configuration and whether web research is available. Web research
must be available only for Gemini/Google-authenticated modes that can import
ADK `google_search`.

## Mandatory Automated Checks

Run these before and after any code fix:

```bash
python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract
python3 tools/check_agent_boundaries.py --root .
python3 agent/cli.py adk-eval
git diff --check
```

Then run all live matrices with the configured model:

```bash
python3 tests/agent_live/run_live_matrix.py \
  --matrix tests/agent_live/agent_intent_smoke_scenarios.json \
  --require-live \
  --provider gemini \
  --model <gemini-model-name> \
  --timeout 240

python3 tests/agent_live/run_live_matrix.py \
  --matrix tests/agent_live/agent_product_acceptance_scenarios.json \
  --require-live \
  --provider gemini \
  --model <gemini-model-name> \
  --timeout 240

python3 tests/agent_live/run_live_matrix.py \
  --matrix tests/agent_live/agent_chaos_conversation_scenarios.json \
  --require-live \
  --provider gemini \
  --model <gemini-model-name> \
  --timeout 240

python3 tests/agent_live/run_live_matrix.py \
  --matrix tests/agent_live/agent_edge_acceptance_scenarios.json \
  --require-live \
  --provider gemini \
  --model <gemini-model-name> \
  --timeout 240
```

If the local provider name differs, use the repository-supported provider mode,
but keep the same matrices and acceptance criteria.

## Manual Terminal Boundary Tests

Run `./bin/anychain-agent` in a real terminal and verify the following behavior.

### Startup

Expected:

- startup diagnostics run automatically;
- framework context is loaded;
- model provider, model name, auth mode, and web research status are shown;
- dependency gaps are shown without blocking normal conversation;
- previous job status is shown if present;
- prompt is stable and supports Ctrl+C without corrupting input.

### Language And Input Editing

Test:

1. Type Chinese text with a typo, delete part of it, and retype.
2. Type English text with a typo, delete part of it, and retype.
3. Press Ctrl+C at an empty prompt.
4. Press Ctrl+C while entering text.

Expected:

- no ghost spaces;
- no duplicate `User>` prompts;
- Ctrl+C clears the current input or exits the current follow/log mode, but does
  not kill a detached benchmark job;
- responses match the language of the latest user intent except code, variable
  names, and commands.

### Benchmark Flow With Fake-Node

Use natural language, not direct low-level commands:

```text
I want to benchmark Solana using fake-node first.
```

Expected:

- Agent asks for target mode confirmation if unclear;
- Agent confirms chain, RPC mode, benchmark mode, QPS profile, observability,
  process names, disk and network metadata;
- fake-node mode may default RPC endpoint values, but it still requires resource
  metadata;
- Agent does not submit a real benchmark without preflight, smoke, and approval.

### Multi-Disk Interaction

If the host has multiple disks, verify the Agent shows numbered disk candidates.
If the host does not have multiple disks, ask a hypothetical question:

```text
If lsblk shows three disks, how will you ask me to choose LEDGER_DEVICE and ACCOUNTS_DEVICE?
```

Expected:

- numbered choices are shown;
- manual override is offered;
- `ACCOUNTS_DEVICE` is optional;
- disk baseline fields are requested when the device exists.

### Backtracking

Test:

```text
I selected the wrong LEDGER_DEVICE. Go back and change it to <device>.
```

Expected:

- Agent acknowledges the correction;
- workflow state is updated or reverted;
- validators are re-run;
- next blocking question is asked.

### Fake-Node To Real-Node Delta

Test:

```text
I tested fake-node. Now switch the same plan to a real node.
```

Expected:

- Agent reuses confirmed resource metadata;
- Agent asks for the delta: real `LOCAL_RPC_URL`, `MAINNET_RPC_URL` or
  sync-health decision, chain changes, RPC mode, method weights, and final
  approval;
- Agent does not ask the user to repeat every already-confirmed machine value.

### Custom RPC And Unsupported Chain

Test:

```text
Add a custom RPC method with three parameters to a mixed workload.
```

Expected:

- Agent asks for method name, parameter contract, parameter samples, expected
  request/response samples, mixed weight, and fake-node fixture plan;
- no production support claim is made before fixture and smoke validation.

Test:

```text
I want to test a chain that is not in the current 36 templates.
```

Expected:

- Agent checks whether it belongs to an existing adapter family;
- if web research is available, Agent may use `google_search` for official docs
  in onboarding/custom-RPC flows only;
- Agent asks for missing official docs, endpoint examples, request/response
  samples, and fixture evidence;
- Agent produces a coding handoff plan rather than pretending support exists.

### Observability

Test:

```text
Should I use the built-in Prometheus/Grafana or connect to an existing environment?
```

Expected:

- Agent explains disabled, local Prometheus/Grafana, and exporter-only modes;
- port conflicts are checked before local startup;
- for existing Prometheus/Grafana, Agent explains exporter endpoint and that
  the external system must scrape it.

### Detached Job And Logs

Test a short fake-node benchmark only after all gates pass.

Expected:

- job runs detached;
- `jobs`, `status`, `logs <job_id>`, and `follow <job_id>` work;
- `follow <job_id>` shows logs in the terminal;
- Ctrl+C exits log-follow mode but does not stop the detached job;
- pasted log snippets can be analyzed by the Agent.

## Failure Handling

If a live LLM call fails temporarily, retry once and keep both summaries. If the
same scenario fails twice, inspect the redacted log and fix the responsible
path:

- prompt/instruction problem: update `agent/adk_app/instructions.py`;
- missing or unsafe deterministic guard: update `agent/validators/` or
  `agent/adk_app/tools/`;
- terminal UX problem: update `agent/terminal/`;
- live matrix gap: update `tests/agent_live/*.json`;
- documentation drift: update the relevant README or docs page.

Do not fix business behavior by adding keyword lists, fuzzy matching, or regex
intent routing in terminal code.

## Evidence To Return

Return:

- commands run;
- matrix summary paths;
- failed scenario logs if any, redacted;
- code files changed;
- tests run after fixes;
- explicit list of boundaries still untested.
