"""Draft chain templates for human-reviewed onboarding."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from onboarding.chain_onboarding import SUPPORTED_FAMILIES


def draft_chain_template(
    chain: str,
    adapter_family: str,
    methods: list[str],
    output: str | Path | None = None,
) -> dict[str, Any]:
    chain_id = _chain_id(chain)
    if adapter_family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported adapter family: {adapter_family}")
    methods = [_method(method) for method in methods if method.strip()]
    if not methods:
        methods = ["example_getBalance"]
    weights = _weights(methods)
    template = {
        "chain_type": chain_id,
        "rpc_url": "LOCAL_RPC_URL",
        "params": _default_params(),
        "system_addresses": [],
        "rpc_methods": {
            "single": methods[0],
            "mixed": ",".join(methods),
            "mixed_weighted": [
                {"method": method, "weight": weight}
                for method, weight in zip(methods, weights)
            ],
        },
        "param_formats": {method: "no_params" for method in methods},
        "param_spec": {
            method: {
                "status": "needs_review",
                "transport": _transport(adapter_family),
                "params": [],
                "notes": "Replace this with real parameter bindings when the method needs address, tx hash, block height, path, query, or body params.",
            }
            for method in methods
        },
        "_meta": {
            "adapter_family": adapter_family,
            "onboarding_status": "needs_review",
            "generated_by": "anychain-agent",
            "notes": [
                "This is a draft only. Do not treat the chain as supported until validation and fake-node fixture coverage pass.",
                "Replace param_formats/param_spec with real method parameter contracts.",
                "Record real request/response fixtures before using fake-node for production-like workloads.",
            ],
            "sync_health": {
                "mode": "needs_review",
                "threshold_env": "BLOCK_HEIGHT_DIFF_THRESHOLD",
                "time_threshold_env": "BLOCK_HEIGHT_TIME_THRESHOLD",
                "notes": "Fill with the local node sync-health method before real-node benchmark use.",
            },
        },
        "proxy_extraction": _proxy_extraction(adapter_family),
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "draft",
        "chain": chain_id,
        "adapter_family": adapter_family,
        "output": str(output) if output else "",
        "template": template,
        "validation_next_steps": [
            f"Review {chain_id} RPC methods and param formats.",
            "Record fake-node fixtures for every workload method.",
            "Run chain template validation and fake-node fixture coverage checks.",
            "Run an Agent fake-node smoke benchmark before real-node tests.",
        ],
    }


def _default_params() -> dict[str, str]:
    return {
        "account_count": "ACCOUNT_COUNT",
        "max_signatures": "ACCOUNT_MAX_SIGNATURES",
        "tx_batch_size": "ACCOUNT_TX_BATCH_SIZE",
        "semaphore_limit": "ACCOUNT_SEMAPHORE_LIMIT",
        "target_address": "${TARGET_ADDRESS:-REPLACE_WITH_REAL_ADDRESS}",
        "target_tx_hash": "${TARGET_TX_HASH:-REPLACE_WITH_REAL_TX_HASH}",
        "target_block_hash": "${TARGET_BLOCK_HASH:-REPLACE_WITH_REAL_BLOCK_HASH}",
        "target_height": "${TARGET_HEIGHT:-1}",
    }


def _proxy_extraction(adapter_family: str) -> dict[str, Any]:
    if adapter_family in {"jsonrpc", "bitcoin_jsonrpc", "substrate", "tendermint"}:
        return {
            "extractors": [
                {
                    "protocol": "json_rpc",
                    "method_source": "body.method",
                    "id_source": "body.id",
                    "params_source": "body.params",
                    "url_pattern": "^/$",
                    "batch_handling": "split",
                }
            ]
        }
    return {
        "extractors": [
            {
                "protocol": "rest",
                "method_source": "route_template",
                "url_pattern": "^/",
                "path_source": "url.path",
                "query_source": "url.query",
            }
        ]
    }


def _transport(adapter_family: str) -> str:
    if adapter_family in {"rest", "hedera_dual"}:
        return "rest"
    return "jsonrpc_list"


def _weights(methods: list[str]) -> list[int]:
    base = 100 // len(methods)
    weights = [base for _ in methods]
    weights[-1] += 100 - sum(weights)
    return weights


def _chain_id(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower()).strip("-")


def _method(value: str) -> str:
    return value.strip()
