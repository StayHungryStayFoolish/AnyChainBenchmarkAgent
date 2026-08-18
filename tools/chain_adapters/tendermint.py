"""TendermintAdapter — Tendermint/Cosmos SDK chains.

Covers 5 chains: Cosmos-Hub / Osmosis / Celestia / Injective / Sei.

Protocol: JSON-RPC-like but params are an OBJECT keyed by field name (not array).
Endpoint structure differs between:
    Tendermint RPC (port 26657): /status, /block, /abci_query
    Cosmos REST  (port 1317):    /cosmos/bank/v1beta1/balances/{address}

For benchmarking we use Tendermint RPC POST JSON-RPC 2.0 form:
    POST /
    Content-Type: application/json
    {"jsonrpc":"2.0","id":1,"method":"abci_query","params":{"path":"/cosmos.bank.v1beta1.Query/Balance","data":"...","prove":false}}

Health probe = `status` method → .result.sync_info.latest_block_height (string of int).
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from .base import ChainAdapter, register, _vegeta_get, _vegeta_post_json, _try_int, load_chain_template
from .jsonrpc import JsonRpcAdapter
from .param_spec import apply_rest_param_spec, build_jsonrpc_params, get_param_spec
from .rest import _is_fake_node_url
from .url_overrides import first_url, resolve_param

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CHAINS_DIR = _REPO_ROOT / "config" / "chains"


@register("tendermint")
class TendermintAdapter(ChainAdapter):

    def __init__(self):
        self._chain_cache: dict[str, dict] = {}
        self._jsonrpc = JsonRpcAdapter()

    def _load_chain(self, chain_name: str) -> dict:
        if chain_name not in self._chain_cache:
            self._chain_cache[chain_name] = load_chain_template(chain_name)
        return self._chain_cache[chain_name]

    @staticmethod
    def _pick_url(tpl: dict, kind: str, fallback: str) -> str:
        meta = tpl.get("_meta", {})
        if kind == "rest":
            override = first_url("CHAIN_REST_URL")
            if override:
                return override
        if kind == "evm":
            override = first_url("CHAIN_EVM_RPC_URL")
            if override:
                return override
        if kind == "rest" and meta.get("rest_url"):
            return meta["rest_url"]
        if kind == "evm" and meta.get("evm_rpc_url"):
            return meta["evm_rpc_url"]
        endpoints = meta.get("original_public_endpoints", [])
        for endpoint in endpoints:
            url = endpoint.get("url", "")
            notes = endpoint.get("notes", "").lower()
            if kind == "rest" and ("rest" in url or "lcd" in url or "rest" in notes or "lcd" in notes):
                return url
            if kind == "evm" and ("evm" in url or "evm" in notes or "drpc" in url):
                return url
        return fallback

    @staticmethod
    def _replace_path_samples(path: str, address: str, tpl: dict) -> str:
        params = tpl.get("params", {}) if isinstance(tpl.get("params"), dict) else {}
        samples = {
            "address": address,
            "addr": address,
            "height": str(resolve_param(params, "target_height", resolve_param(params, "height", "1"))),
            "hash": resolve_param(params, "target_tx_hash", resolve_param(params, "tx_hash", "")),
            "pool_id": str(resolve_param(params, "target_pool_id", resolve_param(params, "pool_id", "1"))),
        }
        for key, value in samples.items():
            path = path.replace("{" + key + "}", str(value))
        return path

    def build_vegeta_target(
        self, method: str, address: str, rpc_url: str, param_format: str = "",
    ) -> dict:
        """Tendermint RPC: POST JSON-RPC with object params.

        param_format conventions:
            no_params          → params: {}
            single_address     → params: {"address": "<addr>"} (caller-chain-specific)
            height_param       → params: {"height": "<addr>"} (block-height query, address holds height)
        """
        chain_name = os.environ.get("BLOCKCHAIN_NODE", "").lower()
        tpl = self._load_chain(chain_name) if chain_name else {}
        if method.startswith("eth_") and _is_fake_node_url(rpc_url):
            return self._jsonrpc.build_vegeta_target(method, address, rpc_url, param_format)
        if method.startswith("eth_"):
            evm_url = rpc_url if _is_fake_node_url(rpc_url) else self._pick_url(tpl, "evm", rpc_url)
            jsonrpc_format = "eth_call_object_latest" if method == "eth_call" else param_format
            return self._jsonrpc.build_vegeta_target(method, address, evm_url, jsonrpc_format)
        if method.startswith("/") or method.startswith("GET "):
            rest_paths = tpl.get("_meta", {}).get("rest_paths", {}) if isinstance(tpl, dict) else {}
            spec = rest_paths.get(method, {})
            path = spec.get("path", method)
            if path.startswith("GET "):
                path = path[4:]
            if method.startswith("GET ") and not spec:
                path = method[4:]
            base_url = rpc_url if _is_fake_node_url(rpc_url) else (
                spec.get("base_url") or (rpc_url if path in ("/status", "/block") else self._pick_url(tpl, "rest", rpc_url))
            )
            path = self._replace_path_samples(path, address, tpl)
            query_values = {}
            param_spec = get_param_spec(tpl, method)
            if param_spec:
                path, _, query_values = apply_rest_param_spec(param_spec, path, None, tpl, address)
            full_url = base_url.rstrip("/") + path
            if query_values:
                separator = "&" if "?" in full_url else "?"
                full_url += separator + urlencode(query_values, doseq=True)
            return _vegeta_get(full_url)
        param_spec = get_param_spec(tpl, method)
        if param_spec:
            params = build_jsonrpc_params(param_spec, tpl, address)
        else:
            params = self._build_params(param_format, address)
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        return _vegeta_post_json(rpc_url, body)

    @staticmethod
    def _build_params(param_format: str, address: str) -> dict:
        if param_format in ("no_params", ""):
            return {}
        if param_format == "single_address":
            return {"address": address}
        if param_format == "height_param":
            return {"height": address}
        if param_format == "abci_balance_query":
            # Cosmos bank balance query
            return {
                "path": "/cosmos.bank.v1beta1.Query/Balance",
                "data": address,
                "prove": False,
            }
        # default: single address as object
        return {"address": address}

    def health_check_request(self, rpc_url: str) -> dict:
        """Build a per-chain Tendermint/Cosmos height probe."""
        chain_name = os.environ.get("BLOCKCHAIN_NODE", "").lower()
        tpl = self._load_chain(chain_name) if chain_name else {}
        rpc_methods = tpl.get("rpc_methods", {}) if isinstance(tpl, dict) else {}
        mixed = str(rpc_methods.get("mixed", ""))
        if "eth_blockNumber" in mixed:
            return self._jsonrpc.health_check_request(rpc_url)
        if "/cosmos/base/tendermint/v1beta1/blocks/latest" in mixed:
            return {
                "method": "GET",
                "url": rpc_url.rstrip("/") + "/cosmos/base/tendermint/v1beta1/blocks/latest",
                "headers": {},
                "body": "",
                "parse_jq": ".block.header.height",
            }
        if "/status" in mixed:
            return {
                "method": "GET",
                "url": rpc_url.rstrip("/") + "/status",
                "headers": {},
                "body": "",
                "parse_jq": ".result.sync_info.latest_block_height // .sync_info.latest_block_height",
            }

        body = {"jsonrpc": "2.0", "id": 1, "method": "status", "params": {}}
        return {
            "method": "POST",
            "url": rpc_url,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, separators=(",", ":")),
            "parse_jq": ".result.sync_info.latest_block_height",
        }

    def parse_block_height(self, response_text: str) -> Optional[int]:
        if not response_text:
            return None
        try:
            obj = json.loads(response_text)
        except json.JSONDecodeError:
            return None
        try:
            return _try_int(obj["result"]["sync_info"]["latest_block_height"])
        except (KeyError, TypeError):
            pass
        try:
            return _try_int(obj["sync_info"]["latest_block_height"])
        except (KeyError, TypeError):
            pass
        try:
            return _try_int(obj["block"]["header"]["height"])
        except (KeyError, TypeError):
            pass
        return self._jsonrpc.parse_block_height(response_text)
