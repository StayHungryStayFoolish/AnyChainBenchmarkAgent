"""Natural-language onboarding request answers for ADK and CLI paths."""

from __future__ import annotations

import re

from onboarding.chain_onboarding import generate_onboarding_package, format_onboarding_package


SUPPORTED_FAMILIES = {"jsonrpc", "rest", "bitcoin_jsonrpc", "substrate", "tendermint", "hedera_dual"}


def answer_onboarding_request(prompt: str) -> str:
    lowered = prompt.lower()
    if _looks_like_agent_platform(lowered):
        return _agent_platform_plan()
    if _looks_like_kb(lowered):
        return _kb_plan()
    if _looks_like_new_protocol(lowered):
        return _new_protocol_plan(prompt)
    chain = _extract_chain(prompt)
    methods = _extract_methods(prompt)
    family = _extract_family(lowered)
    if not chain:
        return _generic_onboarding_plan()
    package = generate_onboarding_package(
        chain,
        methods=methods,
        adapter_family=family,
    )
    lines = [
        format_onboarding_package(package),
        "",
        "Developer boundaries:",
        "- Keep chain-specific logic in config/chains, chain adapters, fake-node configs/fixtures, or sync-health metadata.",
        "- Do not mark the chain or RPC method as supported until validation and fake-node smoke pass.",
        "- Update docs/en and docs/zh for user-visible behavior.",
        "",
        "Detailed guide:",
        "- docs/en/secondary-development-guide.md",
        "- docs/zh/secondary-development-guide.md",
    ]
    return "\n".join(lines)


def _kb_plan() -> str:
    return "\n".join([
        "Enterprise Knowledge Base onboarding plan",
        "- Configure AGENT_KNOWLEDGE_PROVIDER in config/agent_config.sh.",
        "- Use agent/knowledge/http_provider.py for the generic HTTP adapter or AGENT_KNOWLEDGE_PROVIDER_MODULE for a custom provider.",
        "- Keep KB output as evidence only; local repository validation remains required before execution.",
        "- Required contract: POST /search, GET /chains/{chain}/rpc-methods, GET /chains/{chain}/rpc-samples, POST /workload/suggest.",
        "- Validate with: python3 agent/cli.py knowledge-smoke --query \"solana rpc methods\" --chain solana",
        "- Validate Agent contracts with: python3 -m unittest tests.test_agent_runtime_contract -v",
        "",
        "Detailed guide:",
        "- docs/en/secondary-development-guide.md",
        "- docs/zh/secondary-development-guide.md",
    ])


def _agent_platform_plan() -> str:
    return "\n".join([
        "Enterprise Agent platform integration plan",
        "- Use ./bin/anychain-agent for human terminal sessions.",
        "- Use python3 agent/cli.py for enterprise platform automation with JSON input/output.",
        "- Export the tool catalog with: python3 agent/cli.py tool-schema",
        "- Execute one stable tool call with: python3 agent/cli.py tool-call --name <tool> --arguments '<json>'",
        "- Configure LLM and optional KB defaults once in config/agent_config.sh or inject them through the platform runtime environment.",
        "- Keep secrets in the enterprise secret manager; do not write API keys, service-account JSON, private RPC URLs, or KB tokens to git.",
        "- Use tools such as doctor, load_capabilities, draft_request, generate_plan, run_preflight, submit_job, get_job_status, tail_job_log, analyze_artifacts, answer_artifact_question, diagnose_artifacts, draft_chain_template, gap_analysis, and knowledge_search.",
        "- For long-running benchmarks, submit detached/background jobs and use job_id plus artifact_index for resume, status, logs, and analysis.",
        "- Validate with: python3 agent/cli.py tool-schema",
        "- Validate with: python3 agent/cli.py tool-call --name load_capabilities",
        "- Validate Agent contracts with: python3 -m unittest tests.test_agent_runtime_contract -v",
        "",
        "Detailed guide:",
        "- README.md#enterprise-agent-platform-integration",
        "- agent/README.md#knowledge-base-and-enterprise-integration",
        "- docs/en/secondary-development-guide.md",
        "- docs/zh/secondary-development-guide.md",
    ])


def _new_protocol_plan(prompt: str) -> str:
    family = _extract_family(prompt.lower()) or "<new_family>"
    return "\n".join([
        f"New protocol family onboarding plan for {family}",
        "- Add tools/chain_adapters/<family>.py and register it in tools/chain_adapters/base.py.",
        "- Add tools/fake-node/handlers/<family>.go and tools/fake-node/configs/<family>.yaml.",
        "- Add a minimal config/chains/<chain>.json using the new adapter family.",
        "- Define proxy_extraction so proxy_method.csv can identify workload RPC methods.",
        "- Define sync-health parsing or explicitly mark it unsupported.",
        "- Record real fake-node fixtures for each workload method.",
        "- Validate with tests/test_chain_adapters.py, tests/test_param_spec.py, fake-node Go tests, runtime_probe.py, and full fake-node lifecycle smoke.",
        "- Explain in the PR why the existing six families cannot express this chain.",
        "",
        "Detailed guide:",
        "- docs/en/secondary-development-guide.md",
        "- docs/zh/secondary-development-guide.md",
    ])


def _generic_onboarding_plan() -> str:
    return "\n".join([
        "Secondary development plan",
        "- For KB integration, start with config/agent_config.sh and agent/knowledge/.",
        "- For a chain in an existing family, add config/chains/<chain>.json, params, proxy_extraction, sync-health metadata, and fake-node fixtures.",
        "- For a new protocol family, add chain adapter code, fake-node handler/config, template, fixtures, and lifecycle tests.",
        "- For a new RPC method, add rpc_methods.mixed_weighted, param_formats or param_spec, fake-node fixture mapping, and per-method report validation.",
        "- For workload changes, validate target generation, proxy attribution, per-method success/error counts, and latency percentiles.",
        "",
        "Required checks:",
        "- python3 tools/chain_adapters/cli.py validate-template --chain all",
        "- python3 tools/fake-node/check_fixture_coverage.py --json",
        "- python3 tools/fake-node/runtime_probe.py",
        "- python3 -m unittest tests.test_agent_runtime_contract -v",
        "- python3 tools/check_public_repo_markers.py --root .",
        "",
        "Detailed guide:",
        "- docs/en/secondary-development-guide.md",
        "- docs/zh/secondary-development-guide.md",
    ])


def _looks_like_kb(lowered: str) -> bool:
    return any(token in lowered for token in ("knowledge base", "kb", "rag", "知识库"))


def _looks_like_agent_platform(lowered: str) -> bool:
    return any(
        token in lowered
        for token in (
            "agent platform",
            "internal agent",
            "enterprise agent",
            "tool schema",
            "tool-call",
            "tool call",
            "openai-compatible tool",
            "企业 agent",
            "内部 agent",
            "agent 平台",
            "工具 schema",
        )
    )


def _looks_like_new_protocol(lowered: str) -> bool:
    return any(token in lowered for token in ("new protocol", "protocol family", "new family", "新增协议", "新的协议"))


def _extract_family(lowered: str) -> str | None:
    for family in SUPPORTED_FAMILIES:
        if family in lowered:
            return family
    return None


def _extract_chain(prompt: str) -> str:
    patterns = [
        r"(?:chain|node)\s+([A-Za-z0-9_-]+)",
        r"(?:add|onboard|support|draft)\s+([A-Za-z0-9_-]+)\s+(?:chain|node)",
        r"(?:新增|添加|支持)\s*([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip("-_ ")
            if value.lower() not in {"new", "a", "the", "chain", "node"}:
                return value.lower()
    return ""


def _extract_methods(prompt: str) -> list[str]:
    if not any(marker in prompt for marker in ("method", "methods", "rpc", "RPC", "方法")):
        return []
    methods: list[str] = []
    candidates = re.findall(r"\b[A-Za-z][A-Za-z0-9_./:-]{2,}\b", prompt)
    stop = {
        "add", "new", "chain", "node", "with", "method", "methods", "rpc", "support",
        "onboard", "family", "protocol", "custom", "existing", "integrate", "knowledge",
        "base", "template", "draft", "plan",
    }
    for item in candidates:
        lowered = item.lower()
        if lowered in stop or lowered in SUPPORTED_FAMILIES:
            continue
        if "_" in item or "." in item or item.startswith(("eth", "get", "chain", "system", "wallet")):
            if item not in methods:
                methods.append(item)
    return methods[:12]
