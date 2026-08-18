"""BitcoinJsonRpcAdapter — Bitcoin Core JSON-RPC family.

Covers 4 chains: Bitcoin / Litecoin / Dogecoin / Bitcoin-Cash.

Differences from standard JsonRpc:
    - Requires HTTP Basic Auth (rpcuser:rpcpassword)
    - Params are positional array (like JsonRpc) but no `jsonrpc` field is
      required by Bitcoin Core (it accepts both 1.0 and 2.0 form);
      we use 2.0 form for consistency
    - Health probe = `getblockcount` → .result is decimal int

Auth handling:
    BITCOIN_RPC_USER and BITCOIN_RPC_PASSWORD env vars (or read from
    _meta.basic_auth in chain template).
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from .base import ChainAdapter, register, _vegeta_get, _vegeta_post_json, _try_int, _b64, load_chain_template
from .param_spec import apply_rest_param_spec, build_jsonrpc_params, get_param_spec
from .rest import _is_fake_node_url
from .url_overrides import first_url, resolve_param

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CHAINS_DIR = _REPO_ROOT / "config" / "chains"


def _resolve_env_placeholder(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


@register("bitcoin_jsonrpc")
class BitcoinJsonRpcAdapter(ChainAdapter):

    def __init__(self):
        self._chain_cache: dict[str, dict] = {}

    def _load_chain(self, chain_name: str) -> dict:
        if chain_name not in self._chain_cache:
            self._chain_cache[chain_name] = load_chain_template(chain_name)
        return self._chain_cache[chain_name]

    def _auth_header(self) -> Optional[str]:
        user = os.environ.get("BITCOIN_RPC_USER", "")
        pwd = os.environ.get("BITCOIN_RPC_PASSWORD", "")
        if not user:
            return None
        creds = base64.b64encode(f"{user}:{pwd}".encode()).decode()
        return f"Basic {creds}"

    def _extra_headers(self) -> dict:
        """Return extra headers such as x-api-key."""
        headers = {}
        api_key = os.environ.get("RPC_API_KEY")
        if not api_key:
            chain_name = os.environ.get("BLOCKCHAIN_NODE", "").lower()
            if chain_name:
                try:
                    tpl = self._load_chain(chain_name)
                    for endpoint in tpl.get("_meta", {}).get("original_public_endpoints", []):
                        if isinstance(endpoint, dict) and endpoint.get("api_key"):
                            api_key = _resolve_env_placeholder(str(endpoint["api_key"]))
                            break
                except Exception:
                    pass
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    def build_vegeta_target(
        self, method: str, address: str, rpc_url: str, param_format: str = "",
    ) -> dict:
        chain_name = os.environ.get("BLOCKCHAIN_NODE", "").lower()
        tpl = self._load_chain(chain_name) if chain_name else {}
        rest_paths = tpl.get("_meta", {}).get("rest_paths", {}) if isinstance(tpl, dict) else {}
        if method.startswith("GET "):
            spec = rest_paths.get(method)
            if not spec:
                raise ValueError(f"bitcoin chain {chain_name}: method {method!r} not in _meta.rest_paths")
            params = tpl.get("params", {}) if isinstance(tpl.get("params"), dict) else {}
            samples = {
                "address": address,
                "blockhash": resolve_param(params, "target_block_hash", address),
                "txid": resolve_param(params, "target_txid", address),
            }
            path = spec["path"]
            for key, value in samples.items():
                path = path.replace("{" + key + "}", str(value))
            query_values = {}
            param_spec = get_param_spec(tpl, method)
            if param_spec:
                path, _, query_values = apply_rest_param_spec(param_spec, path, None, tpl, address)
            base_url = (
                rpc_url if _is_fake_node_url(rpc_url)
                else first_url("CHAIN_INDEXER_URL", "CHAIN_REST_URL", spec.get("base_url"), rpc_url)
            )
            full_url = base_url.rstrip("/") + path
            if query_values:
                separator = "&" if "?" in full_url else "?"
                full_url += separator + urlencode(query_values, doseq=True)
            return _vegeta_get(full_url, headers=self._extra_headers())
        param_spec = get_param_spec(tpl, method)
        if param_spec:
            params = build_jsonrpc_params(param_spec, tpl, address)
        else:
            params = self._build_params(param_format, address, method, tpl)
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        body_str = json.dumps(body, separators=(",", ":"))
        header = {"Content-Type": ["application/json"]}
        auth = self._auth_header()
        if auth:
            header["Authorization"] = [auth]
        # Add extra headers.
        extra_headers = self._extra_headers()
        for key, value in extra_headers.items():
            header[key] = [value]
        return {
            "method": "POST",
            "url": rpc_url,
            "header": header,
            "body": _b64(body_str),
        }

    @staticmethod
    def _build_params(param_format: str, address: str, method: str = "", tpl: dict | None = None) -> list:
        cfg_params = tpl.get("params", {}) if isinstance(tpl, dict) else {}
        block_hash = resolve_param(cfg_params, "target_block_hash", address)
        txid = resolve_param(cfg_params, "target_txid", address)
        if param_format == "no_params":
            return []
        if method == "estimatesmartfee" or param_format == "[conf_target]":
            return [6]
        if method == "getblock" or param_format in ("block_hash", "[blockhash]", "[blockhash,verbosity]"):
            return [block_hash, 2]
        if method == "getrawtransaction" or param_format in ("txid", "transaction_hash", "[txid,verbose]", "[txhash,verbose]"):
            return [txid, True]
        if param_format == "single_address":
            return [address]
        if param_format == "address_minconf_includewatchonly":
            # getreceivedbyaddress format: [address, minconf, include_watchonly]
            return [address, 1, False]
        return [address]

    def health_check_request(self, rpc_url: str) -> dict:
        chain_name = os.environ.get("BLOCKCHAIN_NODE", "").lower()
        tpl = self._load_chain(chain_name) if chain_name else {}
        probe = tpl.get("_meta", {}).get("health_probe", {}) if isinstance(tpl, dict) else {}
        method = probe.get("method", "getblockcount")
        params = probe.get("params", [])
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        headers = {"Content-Type": "application/json"}
        auth = self._auth_header()
        if auth:
            headers["Authorization"] = auth
        # Add extra headers.
        extra_headers = self._extra_headers()
        headers.update(extra_headers)
        return {
            "method": "POST",
            "url": rpc_url,
            "headers": headers,
            "body": json.dumps(body, separators=(",", ":")),
            "parse_jq": probe.get("parse_jq", ".result"),
        }

    def parse_block_height(self, response_text: str) -> Optional[int]:
        if not response_text:
            return None
        try:
            obj = json.loads(response_text)
        except json.JSONDecodeError:
            return None
        result = obj.get("result")
        parsed = _try_int(result)
        if parsed is not None:
            return parsed
        if isinstance(result, dict):
            for key in ("blocks", "headers"):
                parsed = _try_int(result.get(key))
                if parsed is not None:
                    return parsed
        return None
