"""Compact framework context for Agent grounding.

This module deliberately avoids loading full README/docs content into every LLM
turn. The Agent needs a small, current map of what the framework is, where facts
come from, and which documents are authoritative for deeper answers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge.framework_capabilities import REPO_ROOT, load_framework_capabilities


DOC_INDEX = [
    {
        "topic": "quick_start_and_agent_usage",
        "en": "README.md",
        "zh": "README_ZH.md",
        "use_when": "User asks how to start AnyChain Agent or configure LLM/ADC/API-key auth.",
    },
    {
        "topic": "configuration_layer",
        "en": "config/README.md",
        "zh": "config/README.md",
        "use_when": "User asks which variables are required or how runtime.env/user_config.sh/internal_config.sh interact.",
    },
    {
        "topic": "framework_flow",
        "en": "docs/en/framework-flow.md",
        "zh": "docs/zh/framework-flow.md",
        "use_when": "User asks how entrypoint, monitoring, reports, archives, and observability connect.",
    },
    {
        "topic": "module_guide",
        "en": "docs/en/module-guide.md",
        "zh": "docs/zh/module-guide.md",
        "use_when": "User asks which folder/module owns a feature.",
    },
    {
        "topic": "add_chain_or_rpc",
        "en": "docs/en/how-to-add-chain.md",
        "zh": "docs/zh/how-to-add-chain.md",
        "use_when": "User asks to add a chain, adapter family, RPC method, param_spec, or fake-node fixture.",
    },
    {
        "topic": "closed_loop_fake_node",
        "en": "docs/en/local-closed-loop-testing.md",
        "zh": "docs/zh/local-closed-loop-testing.md",
        "use_when": "User asks how to validate without real blockchain nodes.",
    },
    {
        "topic": "secondary_development",
        "en": "docs/en/secondary-development-guide.md",
        "zh": "docs/zh/secondary-development-guide.md",
        "use_when": "User asks how to integrate KB, enterprise Agent platforms, or extend tools.",
    },
    {
        "topic": "pr_quality_gates",
        "en": "docs/en/github-pr-gates.md",
        "zh": "docs/zh/github-pr-gates.md",
        "use_when": "User asks contribution, PR, CI, or review requirements.",
    },
]


def load_framework_context(root: str | Path = REPO_ROOT, language: str = "en") -> dict[str, Any]:
    """Return a compact, current framework context for LLM grounding."""
    root = Path(root)
    capabilities = load_framework_capabilities(root)
    docs = _doc_index(root, language)
    return {
        "identity": {
            "name": "AnyChain Benchmark Agent",
            "purpose": "Help users configure, validate, run, resume, and analyze blockchain node benchmark jobs.",
            "entrypoint": "./bin/anychain-agent",
            "benchmark_engine_entrypoint": "./blockchain_node_benchmark.sh",
            "agent_runtime": "Google ADK-backed Agent runtime with deterministic AnyChain tools and gates.",
        },
        "operating_principles": [
            "Discover environment and dependency state before asking benchmark configuration questions.",
            "Use repository tools and current files as facts; do not rely on model memory for supported chains or RPC methods.",
            "Ask for user confirmation before dependency installation or real benchmark execution.",
            "Generate runtime.env as the per-job confirmed configuration artifact; users normally do not edit it manually.",
            "Use fake-node for local closed-loop validation when real endpoints are unavailable.",
            "For new chains or RPC methods, request official docs, KB evidence, or real request/response samples before claiming support.",
        ],
        "runtime_flow": [
            "startup doctor",
            "resume latest job context",
            "discover cloud/deployment/CPU/memory/disk/network",
            "load chain/RPC/fake-node capabilities",
            "confirm missing or ambiguous variables with the user",
            "generate plan and runtime.env",
            "run preflight",
            "run smoke",
            "submit detached benchmark job after approval",
            "analyze artifacts and cite file paths",
        ],
        "configuration_layers": [
            {"name": "config/agent_config.sh", "role": "Persistent Agent/LLM/KB provider configuration."},
            {"name": "config/user_config.sh", "role": "User-facing benchmark defaults and environment baselines."},
            {"name": "config/internal_config.sh", "role": "Advanced framework thresholds and internal behavior."},
            {"name": "config/chains/*.json", "role": "Per-chain RPC, param, proxy extraction, sync-health, and fake-node workload templates."},
            {"name": "runtime.env", "role": "Generated per-job confirmed configuration; highest priority for that job."},
        ],
        "capability_summary": {
            "chain_count": capabilities.get("chain_count"),
            "family_count": capabilities.get("family_count"),
            "families": capabilities.get("families"),
            "unique_rpc_method_count": capabilities.get("unique_rpc_method_count"),
            "configured_rpc_method_entries": capabilities.get("configured_rpc_method_entries"),
            "fake_node_fixture_file_count": capabilities.get("fake_node", {}).get("fixture_file_count"),
        },
        "extension_points": capabilities.get("extension_points", []),
        "authoritative_docs": docs,
        "context_policy": {
            "load_full_docs_by_default": False,
            "reason": "Full docs can be long and may include development history; load focused docs only when the user's question requires them.",
            "preferred_first_tools": [
                "run_doctor",
                "discover_environment",
                "load_framework_context",
                "load_framework_capabilities",
                "list_rpc_methods",
                "knowledge_search",
            ],
        },
    }


def render_framework_context_for_prompt(root: str | Path = REPO_ROOT, language: str = "en") -> str:
    """Render a small text context block suitable for one LLM turn."""
    context = load_framework_context(root, language)
    caps = context["capability_summary"]
    docs = context["authoritative_docs"][:8]
    doc_lines = "\n".join(
        f"- {doc['topic']}: {doc['path']} ({doc['use_when']})"
        for doc in docs
    )
    flow = " -> ".join(context["runtime_flow"])
    principles = "\n".join(f"- {item}" for item in context["operating_principles"])
    return (
        "AnyChain framework context (current repository facts):\n"
        f"- entrypoint: {context['identity']['entrypoint']}\n"
        f"- benchmark engine: {context['identity']['benchmark_engine_entrypoint']}\n"
        f"- chains/families/RPC: {caps['chain_count']} chains, {caps['family_count']} families, "
        f"{caps['unique_rpc_method_count']} unique RPC method names, "
        f"{caps['fake_node_fixture_file_count']} fake-node fixture files\n"
        f"- families: {caps['families']}\n"
        f"- runtime flow: {flow}\n"
        "Operating rules:\n"
        f"{principles}\n"
        "Authoritative docs to use when the user asks for details:\n"
        f"{doc_lines}\n"
    )


def _doc_index(root: Path, language: str) -> list[dict[str, str]]:
    key = "zh" if language == "zh" else "en"
    output = []
    for item in DOC_INDEX:
        path = item.get(key) or item["en"]
        output.append({
            "topic": item["topic"],
            "path": path,
            "exists": str((root / path).exists()).lower(),
            "use_when": item["use_when"],
        })
    return output
