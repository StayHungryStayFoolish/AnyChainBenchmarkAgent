"""Capability gap analysis for chains, RPC methods, and onboarding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge.framework_capabilities import load_framework_capabilities


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
            "Select _meta.adapter_family based on protocol: jsonrpc, rest, bitcoin_jsonrpc, substrate, tendermint, or hedera_dual.",
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


def answer_gap_question(question: str) -> dict[str, Any] | None:
    lowered = question.lower()
    if not any(token in lowered for token in ("support", "missing", "gap", "add chain", "new chain", "新增", "缺", "支持")):
        return None
    inventory = load_framework_capabilities()
    chain = _find_chain(lowered, inventory)
    if not chain:
        chain = _find_unknown_chain(question)
    methods = _find_methods(question)
    if not chain and not methods:
        return None
    result = analyze_capability_gap(chain, methods)
    return {
        "intent": "framework_question",
        "answer": _format_gap_answer(result),
        "confidence": 0.9,
        "gap_analysis": result,
        "sources": [
            {"path": str(REPO_ROOT / "config" / "chains"), "line": 0, "text": "chain templates"},
            {"path": str(REPO_ROOT / "tools" / "fake-node" / "fixtures"), "line": 0, "text": "fake-node fixtures"},
        ],
    }


def _find_chain(text: str, inventory: dict[str, Any]) -> str:
    for item in sorted(inventory["chains"], key=lambda row: len(row["chain"]), reverse=True):
        if item["chain"] in text:
            return item["chain"]
    return ""


def _find_unknown_chain(question: str) -> str:
    tokens = [token.strip("`'\".,:;()[]{}?").lower() for token in question.split()]
    markers = {"chain", "chains", "node", "新增", "添加", "支持"}
    stop = markers | {"new", "add", "support", "supported", "does", "the", "a", "an", "and", "or", "rpc", "method", "methods", "如何", "怎么"}
    for idx, token in enumerate(tokens):
        if token in {"chain", "node"} and idx > 0:
            candidate = tokens[idx - 1]
            if candidate and candidate not in stop and len(candidate) >= 3:
                return candidate
        if token in markers and idx + 1 < len(tokens):
            candidate = tokens[idx + 1]
            if candidate and candidate not in stop and len(candidate) >= 3:
                return candidate
    return ""


def _find_methods(question: str) -> list[str]:
    methods = []
    stop = {
        "does", "support", "rpc", "method", "methods", "chain", "chains",
        "新增", "支持", "方法", "是否", "这个", "配置",
    }
    for token in question.replace(",", " ").split():
        cleaned = token.strip("`'\".,:;()[]{}?")
        lowered = cleaned.lower()
        if lowered in stop:
            continue
        if any(marker in cleaned for marker in ("_", ".", "GET", "POST")):
            methods.append(cleaned)
        elif any(ch.isupper() for ch in cleaned[1:]) and len(cleaned) >= 4:
            methods.append(cleaned)
    return methods


def _format_gap_answer(result: dict[str, Any]) -> str:
    status = "supported" if result["supported"] else "has gaps"
    lines = [f"{result['chain'] or '<unknown chain>'}: {status}."]
    if result.get("family"):
        lines.append(f"Adapter family: {result['family']}.")
    lines.append(f"fake-node fixtures: {result['fixture_count']}.")
    if result["gaps"]:
        lines.append("Detected gaps:")
        for gap in result["gaps"]:
            lines.append(f"- [{gap['severity']}] {gap['message']}")
    lines.append("Recommended plan:")
    for step in result["onboarding_plan"]:
        lines.append(f"- {step}")
    return "\n".join(lines)
