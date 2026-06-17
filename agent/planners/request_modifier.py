"""Apply deterministic natural-language edits to benchmark requests."""

from __future__ import annotations

import copy
import re
from typing import Any

from qa.request_drafter import KNOWN_CHAINS


def looks_like_plan_modification(text: str) -> bool:
    lowered = text.lower()
    action_tokens = ("change ", "set ", "update ", "modify ", "switch ", "改", "设置", "调整", "切换", "改成")
    if any(token in lowered for token in action_tokens):
        return True
    if _extract_method_weights(text):
        return True
    return bool(re.search(r"(?:max(?:imum)?|initial|start|step|duration)\s*qps?[^\d]*(\d+)", lowered))


def apply_request_modification(request: dict[str, Any], text: str) -> tuple[dict[str, Any], list[str]]:
    """Return an updated request and human-readable change list."""
    updated = copy.deepcopy(request)
    changes: list[str] = []
    lowered = text.lower()

    chain = _extract_chain(lowered)
    if chain and chain != updated.get("chain"):
        updated["chain"] = chain
        changes.append(f"chain -> {chain}")

    if _mentions_mixed(lowered):
        if updated.get("rpc_mode") != "mixed":
            updated["rpc_mode"] = "mixed"
            changes.append("rpc_mode -> mixed")
    elif _mentions_single(lowered):
        if updated.get("rpc_mode") != "single":
            updated["rpc_mode"] = "single"
            changes.append("rpc_mode -> single")

    if any(token in lowered for token in ("fake-node", "fake node", "mock")):
        if not updated.get("use_fake_node"):
            updated["use_fake_node"] = True
            changes.append("use_fake_node -> true")
    if any(token in lowered for token in ("real node", "真实节点", "without fake", "no fake")):
        if updated.get("use_fake_node"):
            updated["use_fake_node"] = False
            changes.append("use_fake_node -> false")

    qps_changes = _apply_qps(updated, lowered)
    changes.extend(qps_changes)

    method_weights = _extract_method_weights(text)
    if method_weights:
        updated.setdefault("workload", {})["methods"] = method_weights
        updated["rpc_mode"] = "mixed"
        confirmations = set(updated.get("confirmations", []))
        confirmations.add("mixed_weights_confirmation")
        updated["confirmations"] = sorted(confirmations)
        changes.append("mixed workload methods -> " + ", ".join(f"{m['method']}:{m['weight']}" for m in method_weights))
        if "rpc_mode -> mixed" not in changes:
            changes.append("rpc_mode -> mixed")

    local_rpc = _extract_url(text)
    if local_rpc:
        updated["local_rpc_url"] = local_rpc
        changes.append(f"local_rpc_url -> {local_rpc}")

    return updated, changes


def _extract_chain(text: str) -> str | None:
    for chain in sorted(KNOWN_CHAINS, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9_-]){re.escape(chain)}(?![a-z0-9_-])", text):
            return chain
    return None


def _mentions_mixed(text: str) -> bool:
    return "mixed" in text or "权重" in text or "weight" in text


def _mentions_single(text: str) -> bool:
    return "single" in text or "单一" in text or "单个" in text


def _apply_qps(request: dict[str, Any], text: str) -> list[str]:
    qps = dict(request.get("qps", {}))
    changes: list[str] = []
    patterns = {
        "max": [
            r"(?:max(?:imum)?\s*qps|qps\s*max|最大\s*qps|最高\s*qps)[^\d]*(\d+)",
            r"(?:改成|设置为|set\s+to)\s*(\d+)\s*qps",
        ],
        "initial": [r"(?:initial|start|起始|初始)\s*qps[^\d]*(\d+)"],
        "step": [r"(?:step|步长)\s*(?:qps)?[^\d]*(\d+)"],
        "duration_seconds": [r"(?:duration|持续|时长)[^\d]*(\d+)\s*(?:s|sec|second|seconds|秒)?"],
    }
    for field, field_patterns in patterns.items():
        value = _first_int(text, field_patterns)
        if value is not None:
            qps[field] = value
            changes.append(f"qps.{field} -> {value}")
    if "qps" not in request and qps:
        request["qps"] = qps
    elif qps:
        request["qps"] = qps
    return changes


def _first_int(text: str, patterns: list[str]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_method_weights(text: str) -> list[dict[str, Any]]:
    matches = re.findall(r"([A-Za-z_][A-Za-z0-9_./-]*)\s*[:=]?\s*(\d{1,3})\s*%", text)
    if not matches:
        return []
    methods = [{"method": method, "weight": int(weight)} for method, weight in matches]
    total = sum(item["weight"] for item in methods)
    if total != 100:
        return []
    return methods


def _extract_url(text: str) -> str | None:
    match = re.search(r"https?://[^\s'\"<>]+", text)
    return match.group(0) if match else None
