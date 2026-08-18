"""SubstrateAdapter — Substrate-based parachains.

Covers 5 chains: Polkadot / Kusama / Acala / Moonbeam / Astar.

Protocol: JSON-RPC 2.0 POST with positional array params.
Method namespace: state_*, chain_*, system_*, etc.

Examples:
    chain_getHeader            → params: [block_hash?]
    state_getStorage           → params: [storage_key, block_hash?]
    system_chain               → params: []

Health probe = `chain_getHeader` (no params) → .result.number is 0x-hex.
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


@register("substrate")
class SubstrateAdapter(ChainAdapter):

    def __init__(self):
        self._chain_cache: dict[str, dict] = {}
        self._jsonrpc = JsonRpcAdapter()

    def _load_chain(self, chain_name: str) -> dict:
        if chain_name not in self._chain_cache:
            self._chain_cache[chain_name] = load_chain_template(chain_name)
        return self._chain_cache[chain_name]

    def build_vegeta_target(
        self, method: str, address: str, rpc_url: str, param_format: str = "",
    ) -> dict:
        chain_name = os.environ.get("BLOCKCHAIN_NODE", "").lower()
        tpl = self._load_chain(chain_name) if chain_name else {}
        rest_paths = tpl.get("_meta", {}).get("rest_paths", {}) if isinstance(tpl, dict) else {}
        if method.startswith("GET ") or (method in rest_paths and not _is_fake_node_url(rpc_url)):
            if not chain_name:
                raise RuntimeError("SubstrateAdapter REST sidecar method requires BLOCKCHAIN_NODE env var")
            spec = rest_paths.get(method)
            if not spec:
                raise ValueError(f"substrate chain {chain_name}: method {method!r} not in _meta.rest_paths")
            base = (
                rpc_url if _is_fake_node_url(rpc_url)
                else first_url("CHAIN_SIDECAR_URL", tpl.get("_meta", {}).get("sidecar_url"), rpc_url)
            )
            path = spec["path"].replace("{address}", address).replace("{addr}", address)
            target_height = resolve_param(tpl.get("params", {}), "target_height", 1)
            path = path.replace("{height}", str(target_height))
            path = path.replace("{n}", str(target_height))
            query_values = {}
            param_spec = get_param_spec(tpl, method)
            if param_spec:
                path, _, query_values = apply_rest_param_spec(param_spec, path, None, tpl, address)
            full_url = base.rstrip("/") + path
            if query_values:
                separator = "&" if "?" in full_url else "?"
                full_url += separator + urlencode(query_values, doseq=True)
            return _vegeta_get(full_url)
        if method.startswith("eth_") and tpl.get("_meta", {}).get("evm_rpc_url"):
            return self._jsonrpc.build_vegeta_target(
                method,
                address,
                rpc_url if _is_fake_node_url(rpc_url)
                else first_url("CHAIN_EVM_RPC_URL", tpl["_meta"]["evm_rpc_url"]),
                param_format,
            )
        param_spec = get_param_spec(tpl, method)
        if param_spec:
            params = build_jsonrpc_params(param_spec, tpl, address)
        else:
            params = self._build_params(param_format, address, tpl)
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        return _vegeta_post_json(rpc_url, body)

    @staticmethod
    def _build_params(param_format: str, address: str, tpl: dict | None = None) -> list:
        cfg_params = tpl.get("params", {}) if isinstance(tpl, dict) else {}
        if param_format in ("no_params", ""):
            return []
        if param_format in ("[block_number]", "block_number"):
            return [int(resolve_param(cfg_params, "target_height", 1))]
        if param_format == "single_address":
            return [address]
        if param_format == "storage_key":
            # state_getStorage: [storage_key, block_hash?]
            return [address]
        if param_format == "block_hash":
            return [address]
        if param_format == "address_with_block":
            return [address, None]
        return [address]

    def health_check_request(self, rpc_url: str) -> dict:
        """Substrate health = chain header, or EVM block number for EVM-only templates."""
        chain_name = os.environ.get("BLOCKCHAIN_NODE", "").lower()
        tpl = self._load_chain(chain_name) if chain_name else {}
        mixed = str(tpl.get("rpc_methods", {}).get("mixed", ""))
        if "chain_getHeader" not in mixed and "eth_blockNumber" in mixed:
            return self._jsonrpc.health_check_request(rpc_url)
        body = {"jsonrpc": "2.0", "id": 1, "method": "chain_getHeader", "params": []}
        return {
            "method": "POST",
            "url": rpc_url,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, separators=(",", ":")),
            "parse_jq": ".result.number",
        }

    def parse_block_height(self, response_text: str) -> Optional[int]:
        if not response_text:
            return None
        try:
            obj = json.loads(response_text)
        except json.JSONDecodeError:
            return None
        try:
            number = obj["result"]["number"]
            return _try_int(number)
        except (KeyError, TypeError):
            pass
        return self._jsonrpc.parse_block_height(response_text)
