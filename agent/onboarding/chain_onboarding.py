"""Generate plugin-style onboarding plans for chains and RPC workloads."""

from __future__ import annotations

from typing import Any

from knowledge.gap_analyzer import analyze_capability_gap
from llm.orchestrator import synthesize_with_fallback


SUPPORTED_FAMILIES = [
    "jsonrpc",
    "rest",
    "bitcoin_jsonrpc",
    "substrate",
    "tendermint",
    "hedera_dual",
]


def generate_onboarding_package(
    chain: str,
    methods: list[str] | None = None,
    adapter_family: str | None = None,
    rpc_mode: str = "mixed",
    llm_provider: Any | None = None,
) -> dict[str, Any]:
    methods = methods or []
    gap = analyze_capability_gap(chain, methods)
    family = adapter_family or gap.get("family") or "<choose-family>"
    method_entries = [{"method": method, "weight": _default_weight(methods)} for method in methods]
    if method_entries:
        remainder = 100 - sum(item["weight"] for item in method_entries)
        method_entries[-1]["weight"] += remainder
    package = {
        "chain": chain,
        "adapter_family": family,
        "supported_families": SUPPORTED_FAMILIES,
        "status": "ready" if gap["supported"] else "needs_onboarding",
        "gap_analysis": gap,
        "workload_plugin": {
            "rpc_mode": rpc_mode,
            "single": methods[0] if methods else "<method_name>",
            "mixed_weighted": method_entries or [{"method": "<method_name>", "weight": 100}],
            "param_contract": "Add param_formats or param_spec for every method that needs params.",
        },
        "chain_template_steps": [
            "Create or update config/chains/<chain>.json.",
            "Set _meta.adapter_family to one supported family.",
            "Configure rpc_methods.single and rpc_methods.mixed_weighted.",
            "Add param_formats or param_spec for custom method params.",
            "Add proxy_extraction so per-method attribution can identify method names.",
            "Add _meta.sync_health for node health monitoring.",
        ],
        "fake_node_steps": [
            "Record real request/response fixtures for every workload method.",
            "Run tools/fake-node/check_fixture_coverage.py --json.",
            "Run tools/fake-node/runtime_probe.py.",
        ],
        "validation_commands": [
            "python3 tools/chain_adapters/cli.py validate-template --chain <chain>",
            "python3 agent/cli.py gap-analysis --chain <chain> --method <method>",
            "python3 tools/fake-node/check_fixture_coverage.py --json",
            "python3 tools/fake-node/runtime_probe.py",
            "./bin/anychain-agent --prompt \"Create a <chain> fake-node smoke benchmark at 1 QPS\"",
        ],
        "developer_notes": [
            "If the chain fits an existing adapter family and param formats, no Python/Go code should be needed.",
            "If the request envelope, auth, routing, or response parsing is new, extend the adapter and fake-node mapping.",
            "Keep workload methods and fake-node fixtures aligned; per-method charts depend on method names in proxy_method.csv.",
        ],
    }
    package["llm_summary"] = synthesize_with_fallback(
        llm_provider,
        "onboarding",
        str(package),
        "",
        max_tokens=1400,
    )
    return package


def format_onboarding_package(package: dict[str, Any]) -> str:
    lines = [
        f"Onboarding package for {package['chain']}",
        f"- status: {package['status']}",
        f"- adapter_family: {package['adapter_family']}",
        f"- supported_families: {', '.join(package['supported_families'])}",
        "Workload plugin:",
        f"- rpc_mode: {package['workload_plugin']['rpc_mode']}",
        f"- single: {package['workload_plugin']['single']}",
        f"- mixed_weighted: {package['workload_plugin']['mixed_weighted']}",
        "Chain template steps:",
    ]
    lines.extend(f"- {step}" for step in package["chain_template_steps"])
    lines.append("fake-node steps:")
    lines.extend(f"- {step}" for step in package["fake_node_steps"])
    lines.append("Validation commands:")
    lines.extend(f"- {cmd}" for cmd in package["validation_commands"])
    return "\n".join(lines)


def _default_weight(methods: list[str]) -> int:
    if not methods:
        return 100
    return 100 // len(methods)
