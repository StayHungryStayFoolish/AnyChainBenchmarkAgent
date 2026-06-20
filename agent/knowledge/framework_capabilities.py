"""Dynamic framework capability inventory from the local repository."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_framework_capabilities(root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(root)
    chains_dir = root / "config" / "chains"
    fixture_dir = root / "tools" / "fake-node" / "fixtures"
    chains = []
    family_counts: Counter[str] = Counter()
    all_methods: set[str] = set()

    for path in sorted(chains_dir.glob("*.json")):
        data = _read_json(path)
        name = path.stem
        family = data.get("_meta", {}).get("adapter_family", "unknown")
        methods = _extract_rpc_methods(data.get("rpc_methods", {}))
        family_counts[family] += 1
        all_methods.update(methods)
        chains.append({
            "chain": name,
            "family": family,
            "single": _single_method(data.get("rpc_methods", {})),
            "mixed_methods": _mixed_methods(data.get("rpc_methods", {})),
            "mixed_weighted": _mixed_weighted(data.get("rpc_methods", {})),
            "method_count": len(methods),
            "methods": sorted(methods),
            "sync_health_mode": data.get("_meta", {}).get("sync_health", {}).get("mode", ""),
            "has_health_probe": bool(data.get("_meta", {}).get("health_probe")),
            "has_proxy_extraction": bool(data.get("proxy_extraction")),
        })

    fixture_chains = _fixture_summary(fixture_dir)
    return {
        "chain_count": len(chains),
        "family_count": len(family_counts),
        "families": dict(sorted(family_counts.items())),
        "unique_rpc_method_count": len(all_methods),
        "configured_rpc_method_entries": sum(chain["method_count"] for chain in chains),
        "chains": chains,
        "fake_node": {
            "fixture_chain_count": len(fixture_chains),
            "fixture_file_count": sum(item["fixture_count"] for item in fixture_chains),
            "chains": fixture_chains,
        },
        "extension_points": [
            "config/chains/<chain>.json chain template",
            "rpc_methods.single and rpc_methods.mixed_weighted workload configuration",
            "param_formats and optional param_spec for method params",
            "_meta.adapter_family for protocol family routing",
            "proxy_extraction for per-method attribution",
            "tools/fake-node/fixtures/<chain>/ for local closed-loop responses",
        ],
    }


def answer_capability_question(question: str, root: str | Path = REPO_ROOT) -> dict[str, Any] | None:
    lowered = question.lower()
    if any(token in lowered for token in ("does ", "do ", "support ", "missing", "缺少", "是否支持")) and not any(
        token in lowered for token in ("how many", "which chains", "what chains", "支持多少", "支持哪些", "多少个")
    ):
        return None
    keywords = (
        "how many chains", "supported chains", "which chains", "what chains", "rpc method", "rpc methods",
        "adapter family", "fake-node fixture", "capability",
        "多少个链", "支持多少", "哪些链", "rpc method", "方法数量", "能力", "支持哪些",
    )
    if not any(token in lowered for token in keywords):
        return None
    inventory = load_framework_capabilities(root)
    if "rpc" in lowered or "method" in lowered or "方法" in lowered:
        body = _rpc_answer(inventory)
    elif "family" in lowered or "adapter" in lowered or "协议" in lowered:
        body = _family_answer(inventory)
    elif "fake" in lowered or "fixture" in lowered or "闭环" in lowered:
        body = _fixture_answer(inventory)
    else:
        body = _summary_answer(inventory)
    return {
        "intent": "framework_question",
        "answer": body,
        "confidence": 0.95,
        "sources": [
            {"path": str(Path(root) / "config" / "chains"), "line": 0, "text": "chain templates"},
            {"path": str(Path(root) / "tools" / "fake-node" / "fixtures"), "line": 0, "text": "fake-node fixtures"},
        ],
        "capabilities": {
            "chain_count": inventory["chain_count"],
            "families": inventory["families"],
            "unique_rpc_method_count": inventory["unique_rpc_method_count"],
            "configured_rpc_method_entries": inventory["configured_rpc_method_entries"],
            "fake_node_fixture_file_count": inventory["fake_node"]["fixture_file_count"],
        },
    }


def _summary_answer(inventory: dict[str, Any]) -> str:
    family_text = ", ".join(f"{name}={count}" for name, count in inventory["families"].items())
    return (
        f"The current framework supports {inventory['chain_count']} chain templates across "
        f"{inventory['family_count']} adapter families ({family_text}). It has "
        f"{inventory['unique_rpc_method_count']} unique configured RPC method names and "
        f"{inventory['fake_node']['fixture_file_count']} fake-node fixture files. "
        "Additional chains are supported by adding a chain template, RPC method definitions, "
        "param formats/specs, proxy extraction rules, and fake-node fixtures."
    )


def _family_answer(inventory: dict[str, Any]) -> str:
    lines = ["Current adapter families from config/chains:"]
    for name, count in inventory["families"].items():
        lines.append(f"- {name}: {count} chains")
    return "\n".join(lines)


def _rpc_answer(inventory: dict[str, Any]) -> str:
    lines = [
        f"Current chain templates define {inventory['unique_rpc_method_count']} unique RPC method names.",
        "Per-chain configured methods are derived from rpc_methods.single, rpc_methods.mixed, and rpc_methods.mixed_weighted.",
    ]
    for chain in inventory["chains"][:12]:
        sample = ", ".join(chain["methods"][:6])
        lines.append(f"- {chain['chain']} ({chain['family']}): {chain['method_count']} methods; {sample}")
    if len(inventory["chains"]) > 12:
        lines.append(f"- ... {len(inventory['chains']) - 12} more chains in config/chains")
    return "\n".join(lines)


def _fixture_answer(inventory: dict[str, Any]) -> str:
    lines = [
        f"fake-node currently has {inventory['fake_node']['fixture_file_count']} fixture files "
        f"across {inventory['fake_node']['fixture_chain_count']} fixture directories."
    ]
    for item in inventory["fake_node"]["chains"][:12]:
        lines.append(f"- {item['chain']}: {item['fixture_count']} fixtures")
    if len(inventory["fake_node"]["chains"]) > 12:
        lines.append(f"- ... {len(inventory['fake_node']['chains']) - 12} more fixture directories")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_rpc_methods(rpc_methods: dict[str, Any]) -> set[str]:
    methods = set()
    single = _single_method(rpc_methods)
    if single:
        methods.add(single)
    methods.update(_mixed_methods(rpc_methods))
    methods.update(item["method"] for item in _mixed_weighted(rpc_methods) if item.get("method"))
    return methods


def _single_method(rpc_methods: dict[str, Any]) -> str:
    value = rpc_methods.get("single", "")
    if isinstance(value, str):
        return value.strip()
    return ""


def _mixed_methods(rpc_methods: dict[str, Any]) -> list[str]:
    value = rpc_methods.get("mixed", [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _mixed_weighted(rpc_methods: dict[str, Any]) -> list[dict[str, Any]]:
    value = rpc_methods.get("mixed_weighted", [])
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        if not isinstance(item, dict):
            continue
        method = str(item.get("method", "")).strip()
        if not method:
            continue
        output.append({"method": method, "weight": item.get("weight", 0)})
    return output


def _fixture_summary(fixture_dir: Path) -> list[dict[str, Any]]:
    if not fixture_dir.is_dir():
        return []
    rows = []
    for child in sorted(fixture_dir.iterdir()):
        if child.is_dir():
            rows.append({"chain": child.name, "fixture_count": len(list(child.glob("*.json")))})
    rows.sort(key=lambda item: (-item["fixture_count"], item["chain"]))
    return rows
