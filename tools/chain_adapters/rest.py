"""RestAdapter — RESTful HTTP API (path templates, GET/POST mix).

Covers 5 chains: Aptos / Algorand / Hedera / TON / Tezos.

Method convention (chain template rpc_methods.single):
    For REST chains, `rpc_methods.single` holds a method NAME (not RPC),
    which the adapter maps to an HTTP path template containing {address}.

Examples:
    aptos:    "GET_ACCOUNT"          → GET  /v1/accounts/{address}
    algorand: "GET_ACCOUNT"          → GET  /v2/accounts/{address}
    hedera:   "GET_ACCOUNT"          → GET  /api/v1/accounts/{address}
    ton:      "GET_ACCOUNT"          → GET  /api/v2/getAddressInformation?address={address}
    tezos:    "GET_BLOCK_HEAD"       → GET  /chains/main/blocks/head

Health probe = standardized "get latest block" per chain.

Path templates are per-chain — adapter looks them up via _meta.rest_paths in
chain template:

    "_meta": {
        "adapter_family": "rest",
        "rest_paths": {
            "GET_ACCOUNT":     {"method": "GET", "path": "/v1/accounts/{address}"},
            "GET_BLOCK_HEIGHT": {"method": "GET", "path": "/v1"}
        }
    }
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlparse

from .base import ChainAdapter, register, _vegeta_get, _vegeta_post_json, _try_int, load_chain_template
from .param_spec import apply_rest_param_spec, get_param_spec
from .url_overrides import first_url, resolve_param

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CHAINS_DIR = _REPO_ROOT / "config" / "chains"


def _is_fake_node_url(rpc_url: str) -> bool:
    """Return true when rpc_url is a local/fake-node endpoint.

    Some real REST methods declare a public indexer/base_url because a
    production node may not expose that surface. During closed-loop fake-node
    testing, however, every generated target must stay on the supplied rpc_url.
    """
    host = (urlparse(rpc_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", "fake-node"}


@register("rest")
class RestAdapter(ChainAdapter):

    def __init__(self):
        self._chain_cache: dict[str, dict] = {}

    def _extra_headers(self) -> dict:
        """Return extra headers such as x-api-key."""
        headers = {}
        api_key = os.environ.get("RPC_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key
            headers["accept"] = "application/json"
            headers["content-type"] = "application/json"
        return headers

    def _load_chain(self, chain_name: str) -> dict:
        if chain_name not in self._chain_cache:
            self._chain_cache[chain_name] = load_chain_template(chain_name)
        return self._chain_cache[chain_name]

    @staticmethod
    def _sample_values(tpl: dict, address: str) -> dict[str, str]:
        params = tpl.get("params", {}) if isinstance(tpl.get("params"), dict) else {}
        samples = {
            "address": address,
            "addr": address,
            "hash": resolve_param(params, "target_tx_hash", resolve_param(params, "tx_hash", address)),
            "txid": resolve_param(params, "target_txid", resolve_param(params, "txid", address)),
            "tx_hash": resolve_param(params, "target_tx_hash", resolve_param(params, "tx_hash", address)),
            "block_hash": resolve_param(params, "target_block_hash", resolve_param(params, "block_hash", address)),
            "block": str(resolve_param(params, "target_block", resolve_param(params, "block", "head"))),
            "height": str(resolve_param(params, "target_height", resolve_param(params, "height", "1"))),
            "round": str(resolve_param(params, "target_round", resolve_param(params, "round", "1"))),
            "asset_id": str(resolve_param(params, "target_asset_id", resolve_param(params, "asset_id", "1"))),
            "asset": resolve_param(params, "target_asset", resolve_param(params, "asset", address)),
            "epoch": str(resolve_param(params, "target_epoch", resolve_param(params, "epoch", ""))),
            "vp": str(resolve_param(params, "target_vp", resolve_param(params, "vp", "0"))),
            "pool_id": str(resolve_param(params, "target_pool_id", resolve_param(params, "pool_id", "1"))),
        }
        return {k: v for k, v in samples.items() if v is not None}

    @classmethod
    def _substitute_placeholders(cls, value, samples: dict[str, str]):
        if isinstance(value, str):
            for key, sample in samples.items():
                token = "{" + key + "}"
                if value == token:
                    return sample
                value = value.replace(token, str(sample))
            return value
        if isinstance(value, list):
            return [cls._substitute_placeholders(item, samples) for item in value]
        if isinstance(value, dict):
            return {k: cls._substitute_placeholders(v, samples) for k, v in value.items()}
        return value

    def _resolve_path(self, chain_name: str, method: str, address: str) -> tuple[str, str, dict | None, str | None, dict]:
        """Return (http_method, path, body_dict_or_None, base_url, query_values).

        body comes from optional `_meta.rest_paths[method].body` template.
        Body may contain placeholders ({address}, {addresses_array}) which are
        substituted; pass `None` for GET methods.
        """
        tpl = self._load_chain(chain_name)
        rest_paths = tpl.get("_meta", {}).get("rest_paths", {})
        if method not in rest_paths:
            raise ValueError(
                f"REST chain {chain_name}: method {method!r} not in _meta.rest_paths. "
                f"Available: {list(rest_paths)}"
            )
        spec = rest_paths[method]
        samples = self._sample_values(tpl, address)
        path = self._substitute_placeholders(spec["path"], samples)
        body = None
        if "body" in spec:
            body = self._substitute_placeholders(spec["body"], samples)
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    pass
        param_spec = get_param_spec(tpl, method)
        query_values = {}
        if param_spec:
            path, body, query_values = apply_rest_param_spec(param_spec, path, body, tpl, address)
        return spec.get("method", "GET"), path, body, spec.get("base_url"), query_values

    def build_vegeta_target(
        self, method: str, address: str, rpc_url: str, param_format: str = "",
    ) -> dict:
        """For REST, `method` is a logical method NAME mapped via _meta.rest_paths.
        rpc_url base is the chain's base URL (LOCAL_RPC_URL). The full URL is
        rpc_url + path. The chain name is taken from BLOCKCHAIN_NODE env var
        (master_qps_executor sets this).
        """
        chain_name = os.environ.get("BLOCKCHAIN_NODE", "").lower()
        if not chain_name:
            raise RuntimeError("RestAdapter requires BLOCKCHAIN_NODE env var")
        http_method, path, body, method_base_url, query_values = self._resolve_path(chain_name, method, address)
        # Strip trailing slash from rpc_url, ensure path starts with /
        base = (
            rpc_url if _is_fake_node_url(rpc_url)
            else first_url("CHAIN_INDEXER_URL", "CHAIN_REST_URL", method_base_url, rpc_url)
        ).rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        full_url = base + path
        if query_values:
            separator = "&" if "?" in full_url else "?"
            full_url += separator + urlencode(query_values, doseq=True)
        extra_headers = self._extra_headers()
        if http_method.upper() == "GET":
            return _vegeta_get(full_url, headers=extra_headers)
        else:
            # fix: read body from _meta.rest_paths[method].body template
            # (falls back to empty dict to preserve old behavior for chains that
            # haven't declared a body template yet).
            return _vegeta_post_json(full_url, body if body is not None else {}, extra_headers)

    def health_check_request(self, rpc_url: str) -> dict:
        """REST health probe varies per chain. Default: GET / (root).
        Per-chain override via _meta.health_probe in chain template:
            {"method": "GET", "path": "/v1", "parse_jq": ".block_height"}
        """
        chain_name = os.environ.get("BLOCKCHAIN_NODE", "").lower()
        probe = {}
        if chain_name:
            tpl = self._load_chain(chain_name)
            probe = tpl.get("_meta", {}).get("health_probe", {})
            if chain_name == "ton" and not probe:
                probe = {
                    "method": "GET",
                    "path": "/lookupBlock?workchain=-1&shard=-9223372036854775808&seqno=72033975",
                    "parse_jq": ".result.seqno",
                }
        method = probe.get("method", "GET")
        path = probe.get("path", "/")
        parse_jq = probe.get("parse_jq", ".block_height // .height // .level")
        base = rpc_url.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        headers = self._extra_headers()
        return {
            "method": method,
            "url": base + path,
            "headers": headers,
            "body": "",
            "parse_jq": parse_jq,
        }

    def parse_block_height(self, response_text: str) -> Optional[int]:
        """Try multiple common JSON paths."""
        if not response_text:
            return None
        try:
            obj = json.loads(response_text)
        except json.JSONDecodeError:
            return None
        for key in ("block_height", "height", "level", "ledger_index",
                    "last-round", "blockHeight", "block_no"):
            v = obj.get(key) if isinstance(obj, dict) else None
            if v is not None:
                parsed = _try_int(v)
                if parsed is not None:
                    return parsed
        # Tezos returns the full block header — try .header.level
        if isinstance(obj, dict):
            header = obj.get("header")
            if isinstance(header, dict):
                return _try_int(header.get("level"))
            result = obj.get("result")
            if isinstance(result, dict):
                for key in ("seqno", "block_height", "height"):
                    parsed = _try_int(result.get(key))
                    if parsed is not None:
                        return parsed
                block_id = result.get("block_id")
                if isinstance(block_id, dict):
                    parsed = _try_int(block_id.get("seqno"))
                    if parsed is not None:
                        return parsed
            block = obj.get("block")
            if isinstance(block, dict):
                header = block.get("header")
                if isinstance(header, dict):
                    parsed = _try_int(header.get("height"))
                    if parsed is not None:
                        return parsed
        # Cardano (Koios /tip) returns a JSON array — try first element's
        # block_no / block_height / abs_slot fallback.
        if isinstance(obj, list) and obj:
            first = obj[0]
            if isinstance(first, dict):
                for key in ("block_no", "block_height", "height", "abs_slot"):
                    v = first.get(key)
                    if v is not None:
                        parsed = _try_int(v)
                        if parsed is not None:
                            return parsed
        return None
