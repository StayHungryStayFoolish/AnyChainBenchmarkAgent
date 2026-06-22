"""Inspect chain template values that require user review before execution."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:-]")


def inspect_chain_template(chain: str, root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Return workload and runtime variable requirements for one chain."""
    chain = (chain or "").strip().lower()
    path = Path(root) / "config" / "chains" / f"{chain}.json"
    if not chain or not path.is_file():
        return {
            "chain": chain,
            "exists": False,
            "path": str(path),
            "adapter_family": "",
            "single_method": "",
            "mixed_weighted": [],
            "param_formats": {},
            "param_spec_methods": [],
            "custom_rpc_extension_fields": [
                "rpc_methods.single",
                "rpc_methods.mixed",
                "rpc_methods.mixed_weighted",
                "param_formats",
                "param_spec",
                "_meta.rest_paths",
                "tools/fake-node/fixtures/<chain>/",
            ],
            "runtime_sample_variables": [],
            "runtime_endpoint_variables": [],
            "sync_health_mode": "",
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    rpc_methods = data.get("rpc_methods", {})
    params = data.get("params", {})
    return {
        "chain": chain,
        "exists": True,
        "path": str(path),
        "adapter_family": data.get("_meta", {}).get("adapter_family", ""),
        "single_method": _single_method(rpc_methods),
        "mixed_weighted": _mixed_weighted(rpc_methods),
        "param_formats": data.get("param_formats", {}),
        "param_spec_methods": sorted((data.get("param_spec") or {}).keys()),
        "custom_rpc_extension_fields": [
            "rpc_methods.single",
            "rpc_methods.mixed",
            "rpc_methods.mixed_weighted",
            "param_formats",
            "param_spec",
            "_meta.rest_paths",
            "tools/fake-node/fixtures/<chain>/",
        ],
        "runtime_sample_variables": _sample_variables(params),
        "runtime_endpoint_variables": _endpoint_variables(data),
        "sync_health_mode": data.get("_meta", {}).get("sync_health", {}).get("mode", ""),
    }


def _single_method(rpc_methods: dict[str, Any]) -> str:
    value = rpc_methods.get("single", "")
    return value.strip() if isinstance(value, str) else ""


def _mixed_weighted(rpc_methods: dict[str, Any]) -> list[dict[str, Any]]:
    value = rpc_methods.get("mixed_weighted", [])
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if not isinstance(item, dict):
            continue
        method = str(item.get("method", "")).strip()
        if method:
            rows.append({"method": method, "weight": int(item.get("weight", 0) or 0)})
    return rows


def _sample_variables(params: dict[str, Any]) -> list[str]:
    variables: set[str] = set()
    for value in params.values():
        if isinstance(value, str):
            variables.update(var for var in _ENV_REF_RE.findall(value) if var.startswith("TARGET_"))
    return sorted(variables)


def _endpoint_variables(data: dict[str, Any]) -> list[str]:
    variables = set()
    for key in ("rpc_url", "rest_url", "indexer_url", "sidecar_url", "evm_rpc_url", "json_rpc_url", "mirror_url"):
        value = data.get(key)
        if isinstance(value, str) and value.isupper() and value.endswith("_URL") and value != "LOCAL_RPC_URL":
            variables.add(value)
    text = json.dumps(data, sort_keys=True)
    for name in (
        "CHAIN_REST_URL",
        "CHAIN_INDEXER_URL",
        "CHAIN_SIDECAR_URL",
        "CHAIN_EVM_RPC_URL",
        "CHAIN_JSON_RPC_URL",
        "CHAIN_MIRROR_URL",
        "RPC_API_KEY",
    ):
        if name in text:
            variables.add(name)
    return sorted(variables)
