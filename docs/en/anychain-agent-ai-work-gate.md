# AnyChain Agent AI Work Gate

This document is the project-specific gate for AI coding work on AnyChain
Agent. It complements the repository-level `AI_CODING_GUIDE.md` behavior contract.

Before changing Agent code, an AI coding agent must read:

1. `AI_CODING_GUIDE.md`.
2. This gate document.
3. `agent/README.md`.
4. The exact files it plans to edit.

If these rules conflict with an implementation shortcut, the rules win.

## Non-Negotiable Product Boundary

AnyChain Agent is an ADK-based domain agent for blockchain node benchmarking.
It must reduce user configuration burden and call deterministic benchmark
tools safely. It is not a shell script wizard, not a keyword router, and not a
collection of fallback demos.

The product loop is:

```text
Understand -> Plan -> Ask -> Configure -> Validate -> Execute -> Observe -> Analyze -> Iterate
```

ADK and the configured model own natural-language understanding, planning,
question selection, and iteration. Repository tools own deterministic checks,
configuration materialization, benchmark execution, evidence collection, and
artifact-backed analysis.

## Forbidden Patterns

Do not add or reintroduce:

- business intent routing in terminal code through keyword lists, fuzzy matches,
  regex guesses, or language-specific phrase tables;
- workflow shortcuts that bypass ADK sub-agents, typed tools, validators, user
  confirmation, preflight, or smoke testing;
- old non-ADK wizard/fallback logic for benchmark planning;
- phrase-patching that rewrites model style instead of fixing instructions or
  ADK workflow behavior;
- claims that an unsupported chain, RPC method, fixture, endpoint, or
  benchmark path works without evidence;
- changes to `config/agent_config.sh` unless the user explicitly asks;
- committed API keys, service account JSON, ADC files, generated runtime state,
  or live benchmark archives.

Stable terminal commands such as `help`, `doctor`, `jobs`, `status`, `logs`,
`follow`, and `exit` are allowed. Business requests must go through ADK.

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
was wrong, wants to go back, or changes the test target, ADK must update or
revert workflow state, re-run validators, and ask the next blocking question.

## Required Configuration Gates

Before smoke or real benchmark execution, validators must confirm:

- target mode: fake-node or real-node;
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

## Onboarding And Knowledge Boundary

For a chain outside the supported templates, ADK must first determine whether
it belongs to an existing adapter family. If framework knowledge is insufficient,
ask the user for official RPC documentation, endpoint information, method
examples, request/response samples, and fixture evidence.

When Gemini plus Google authentication is configured, ADK may use
`google_search` only in onboarding and custom-RPC research flows. Search results
are evidence, not authority to skip validation. Official documentation should
be preferred over blogs or forums.

When generating a secondary-development plan, the Agent must include:

- files to modify;
- chain template fields;
- adapter or new-family boundaries;
- fake-node fixture requirements;
- smoke and coverage checks;
- documentation updates required to keep the Agent knowledge current;
- PR and CI expectations.

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
python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract
python3 tools/check_agent_boundaries.py --root .
python3 agent/cli.py adk-eval
git diff --check
```

When live model behavior is affected and a safe key is available, run the
DeepSeek live acceptance matrices in an isolated environment. When benchmark
execution is affected, run fake-node smoke in Docker or an isolated Linux
environment.

If a boundary cannot be tested locally, report it as untested. Do not describe
untested behavior as complete.

## Review Checklist

Before finishing an Agent task, answer these internally:

- Did the change preserve ADK-owned intent recognition?
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
