"""Deterministic prompt-to-request drafting.

This module is intentionally conservative. It extracts obvious intent from a
prompt and leaves missing values for the QA layer instead of inventing details.
"""

from __future__ import annotations

import re
from typing import Any


KNOWN_CHAINS = {
    "algorand", "aptos", "arbitrum", "avalanche-c", "avalanche-x",
    "base", "bitcoin", "bitcoin-cash", "bnb", "cardano", "cosmos",
    "dogecoin", "ethereum", "fantom", "filecoin", "hedera", "litecoin",
    "near", "optimism", "polkadot", "polygon", "scroll", "solana",
    "starknet", "sui", "tezos", "ton", "tron", "xrpl", "zcash",
}


def draft_request(prompt: str) -> dict[str, Any]:
    text = prompt.strip()
    lowered = text.lower()

    chain = _extract_chain(lowered)
    goal = _extract_goal(lowered)
    deployment = _extract_deployment(lowered)
    use_fake_node = any(token in lowered for token in ("fake-node", "fake node", "mock"))
    rpc_mode = "mixed" if "mixed" in lowered else "single"

    request: dict[str, Any] = {
        "chain": chain or "",
        "goal": goal,
        "rpc_mode": rpc_mode,
        "use_fake_node": use_fake_node,
        "deployment": deployment,
        "observability": {
            "enabled": any(token in lowered for token in ("prometheus", "grafana", "observability")),
            "mode": "exporter" if "exporter" in lowered else "local",
        },
        "dependency_mode": "audit",
        "runner_mode": _extract_runner_mode(lowered),
        "bottleneck_focus": _extract_bottleneck_focus(lowered),
        "source_prompt": text,
    }
    if "smoke" in lowered or "closed loop" in lowered:
        request["recommended_initial_validation"] = "smoke"

    qps = _extract_qps(lowered)
    if qps:
        request["qps"] = qps

    return request


def _extract_chain(text: str) -> str | None:
    for chain in sorted(KNOWN_CHAINS, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9_-]){re.escape(chain)}(?![a-z0-9_-])", text):
            return chain
    if "bsc" in text or "bnb chain" in text:
        return "bnb"
    return None


def _extract_goal(text: str) -> str:
    if "regression" in text:
        return "regression"
    if "stress" in text or "极限" in text:
        return "stress"
    if "max" in text or "maximum" in text or "最大" in text:
        return "max_stable_qps"
    if "bottleneck" in text or "瓶颈" in text:
        return "bottleneck_confirmation"
    if "smoke" in text or "closed loop" in text:
        return "smoke"
    return "baseline"


def _extract_deployment(text: str) -> dict[str, str]:
    deployment_type = "unknown"
    provider = ""
    if any(token in text for token in ("gke", "eks", "k8s", "kubernetes", "pod", "namespace")):
        deployment_type = "kubernetes"
    elif any(token in text for token in ("vm", "gce", "ec2", "bare metal", "bare-metal")):
        deployment_type = "vm"

    if "gke" in text or "gcp" in text or "google" in text or "gce" in text:
        provider = "gcp"
    elif "eks" in text or "aws" in text or "ec2" in text:
        provider = "aws"

    return {"type": deployment_type, "provider": provider}


def _extract_bottleneck_focus(text: str) -> list[str]:
    focus = []
    mapping = {
        "cpu": ("cpu",),
        "memory": ("memory", "mem"),
        "disk": ("disk", "iops", "throughput", "磁盘"),
        "network": ("network", "nic", "bandwidth", "网络"),
        "sync_health": ("sync", "height", "slot", "区块高度"),
        "rpc_errors": ("error", "failure", "失败"),
    }
    for name, tokens in mapping.items():
        if any(token in text for token in tokens):
            focus.append(name)
    return focus or ["cpu", "memory", "disk", "network", "rpc_errors"]


def _extract_qps(text: str) -> dict[str, int]:
    matches = [int(value) for value in re.findall(r"(\d+)\s*qps", text)]
    if not matches:
        return {}
    max_qps = max(matches)
    return {
        "initial": min(100, max_qps),
        "max": max_qps,
        "step": max(1, max_qps // 10),
        "duration_seconds": 60,
    }


def _extract_runner_mode(text: str) -> str:
    if any(token in text for token in ("foreground", "前台", "keep terminal", "terminal")):
        return "foreground"
    if any(token in text for token in ("background", "detached", "后台", "断开", "resume")):
        return "detached"
    return "detached"
