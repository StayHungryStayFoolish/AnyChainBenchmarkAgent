"""Capability gap analysis for chains, RPC methods, and onboarding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .framework_capabilities import load_framework_capabilities
    from agent.onboarding.families import SUPPORTED_FAMILIES
except ImportError:  # script execution with agent/ on sys.path
    from knowledge.framework_capabilities import load_framework_capabilities
    from onboarding.families import SUPPORTED_FAMILIES


REPO_ROOT = Path(__file__).resolve().parents[2]


def analyze_capability_gap(
    chain: str,
    methods: list[str] | None = None,
    root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    inventory = load_framework_capabilities(root)
    chain = (chain or "").strip().lower()
    methods = [method.strip() for method in (methods or []) if method.strip()]
    chain_row = next((item for item in inventory["chains"] if item["chain"] == chain), None)
    gaps: list[dict[str, str]] = []
    supported_methods: list[str] = []
    missing_methods: list[str] = []

    if not chain_row:
        gaps.append({
            "type": "chain_template",
            "severity": "blocker",
            "message": f"Missing config/chains/{chain}.json chain template.",
        })
    else:
        if not chain_row.get("has_proxy_extraction"):
            gaps.append({"type": "proxy_extraction", "severity": "blocker", "message": "Missing proxy_extraction rules."})
        if not chain_row.get("mixed_weighted"):
            gaps.append({"type": "mixed_weighted", "severity": "warning", "message": "Missing rpc_methods.mixed_weighted workload entries."})
        if not chain_row.get("sync_health_mode"):
            gaps.append({"type": "sync_health", "severity": "warning", "message": "Missing _meta.sync_health configuration."})
        configured = set(chain_row.get("methods", []))
        for method in methods:
            if method in configured:
                supported_methods.append(method)
            else:
                missing_methods.append(method)
                gaps.append({
                    "type": "rpc_method",
                    "severity": "blocker",
                    "message": f"RPC method is not configured for {chain}: {method}",
                })

    fixture_root = Path(root) / "tools" / "fake-node" / "fixtures" / chain
    fixture_count = len(list(fixture_root.glob("*.json"))) if fixture_root.is_dir() else 0
    if chain and fixture_count == 0:
        gaps.append({
            "type": "fake_node_fixtures",
            "severity": "warning",
            "message": f"No fake-node fixtures found for {chain}.",
        })

    return {
        "chain": chain,
        "supported": bool(chain_row) and not missing_methods,
        "chain_exists": bool(chain_row),
        "family": chain_row.get("family", "") if chain_row else "",
        "supported_methods": supported_methods,
        "missing_methods": missing_methods,
        "fixture_count": fixture_count,
        "gaps": gaps,
        "onboarding_plan": onboarding_plan(chain, methods, gaps),
    }


def onboarding_plan(chain: str, methods: list[str], gaps: list[dict[str, str]]) -> list[str]:
    steps = []
    gap_types = {gap["type"] for gap in gaps}
    if "chain_template" in gap_types:
        steps.extend([
            f"Create config/chains/{chain}.json from config/chains/chain_template.json.bak.",
            f"Select _meta.adapter_family based on protocol: {', '.join(SUPPORTED_FAMILIES[:-1])}, or {SUPPORTED_FAMILIES[-1]}.",
            "Define rpc_methods.single, rpc_methods.mixed_weighted, param_formats, and proxy_extraction.",
        ])
    if "rpc_method" in gap_types:
        steps.extend([
            "Add missing RPC methods to rpc_methods.mixed_weighted with explicit weights.",
            "Add each method's param format or param_spec so target generation can build valid requests.",
            "Record request/response fixtures for fake-node before using the method in closed-loop tests.",
        ])
    if "fake_node_fixtures" in gap_types:
        steps.append("Run tools/fake-node/scripts/record_all_rpc_fixtures.py or the fixture recorder for the selected chain.")
    if "sync_health" in gap_types:
        steps.append("Add _meta.sync_health so block-height/sync-health monitoring can classify node health.")
    if not steps:
        steps.append("No blocking framework gaps detected. Run preflight and a fake-node smoke test next.")
    steps.append("Validate with python3 agent/cli.py capabilities and the fake-node local closed-loop guide.")
    return steps
