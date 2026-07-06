"""HederaDualAdapter — Hedera dual-protocol adapter (Mirror REST + JSON-RPC Relay).

Hedera is a dual-protocol chain:
  - Mirror REST  https://mainnet-public.mirrornode.hedera.com  (account/balance/tx)
  - JSON-RPC Relay https://mainnet.hashio.io/api               (EVM-compat eth_*)

Per-request routing:
  - methods starting with eth_ or known JSON-RPC namespaces use JSON-RPC POST
    against _meta.json_rpc_url from the chain template
  - all other methods use REST paths from _meta.rest_paths and the rpc_url
    passed to build_vegeta_target (LOCAL_RPC_URL = Mirror REST)

Why not simply compose RestAdapter + JsonRpcAdapter externally?
  ChainAdapter and its factory map chain -> family 1:1. Hedera needs
  per-request protocol routing inside one chain, so it is modeled as a family.

Test invariant (L1-CLI): address must appear in either generated URL or body.
  REST side -> address in path -> URL
  JSON-RPC side -> address in params[0] -> body
"""
from __future__ import annotations
import json
import os
import re
from typing import Optional

from .base import ChainAdapter, register, _vegeta_get, _vegeta_post_json, _try_int, load_chain_template
from .rest import RestAdapter, _is_fake_node_url
from .jsonrpc import JsonRpcAdapter
from .url_overrides import first_url, resolve_param


_JSONRPC_METHOD_RE = re.compile(r"^(eth_|net_|web3_|debug_|trace_)")


def _is_jsonrpc_method(method: str) -> bool:
    """Decide if a method name should be routed to the JSON-RPC Relay.

    True for eth_* / net_* / web3_* / debug_* / trace_* (standard EVM
    JSON-RPC namespaces). False for REST path-style keys like
    'GET /api/v1/accounts/{addr}'.
    """
    return bool(_JSONRPC_METHOD_RE.match(method))


@register("hedera_dual")
class HederaDualAdapter(ChainAdapter):
    """Per-request protocol routing for Hedera Mirror REST + JSON-RPC Relay."""

    def __init__(self):
        # Delegate the two protocol concerns to the existing single-protocol
        # adapters — no logic duplication, only routing.
        self._rest = RestAdapter()
        self._jsonrpc = JsonRpcAdapter()
        self._chain_cache: dict[str, dict] = {}

    def _load_chain(self, chain_name: str) -> dict:
        if chain_name not in self._chain_cache:
            self._chain_cache[chain_name] = load_chain_template(chain_name)
        return self._chain_cache[chain_name]

    def _get_chain_name(self) -> str:
        chain_name = os.environ.get("BLOCKCHAIN_NODE", "").lower()
        if not chain_name:
            raise RuntimeError("HederaDualAdapter requires BLOCKCHAIN_NODE env var")
        return chain_name

    def _get_jsonrpc_url(self, chain_name: str) -> str:
        """JSON-RPC url comes from chain template _meta.json_rpc_url.

        Required because LOCAL_RPC_URL (the single env var used as rpc_url
        for this run) points at Mirror REST; JSON-RPC traffic needs a
        separate endpoint (Hashio relay for hedera).
        """
        tpl = self._load_chain(chain_name)
        url = first_url("CHAIN_JSON_RPC_URL", tpl.get("_meta", {}).get("json_rpc_url"))
        if not url:
            raise ValueError(
                f"hedera_dual chain {chain_name}: _meta.json_rpc_url not set; "
                f"required for routing eth_*/net_*/web3_* methods."
            )
        return url

    # ─── ABC contract ──────────────────────────────────────────────────────

    def build_vegeta_target(
        self, method: str, address: str, rpc_url: str, param_format: str = "",
    ) -> dict:
        chain_name = self._get_chain_name()
        if _is_jsonrpc_method(method):
            jsonrpc_url = rpc_url if _is_fake_node_url(rpc_url) else self._get_jsonrpc_url(chain_name)
            tpl = self._load_chain(chain_name)
            address = resolve_param(tpl.get("params", {}), "target_evm_address", address)
            return self._jsonrpc.build_vegeta_target(
                method=method, address=address,
                rpc_url=jsonrpc_url, param_format=param_format,
            )
        # REST path-style: delegate to RestAdapter (uses BLOCKCHAIN_NODE +
        # _meta.rest_paths from chain template, same as before)
        tpl = self._load_chain(chain_name)
        rest_url = (
            rpc_url if _is_fake_node_url(rpc_url)
            else first_url("CHAIN_MIRROR_URL", "CHAIN_REST_URL", tpl.get("_meta", {}).get("mirror_url"), rpc_url)
        )
        return self._rest.build_vegeta_target(
            method=method, address=address,
            rpc_url=rest_url, param_format=param_format,
        )

    def health_check_request(self, rpc_url: str) -> dict:
        """Health probe uses REST side (_meta.health_probe).

        Mirror REST is the primary observability surface; JSON-RPC liveness
        is an orthogonal concern. If both must be probed, run two probes
        externally (eth_blockNumber against json_rpc_url).
        """
        return self._rest.health_check_request(rpc_url)

    def parse_block_height(self, response_text: str) -> Optional[int]:
        """Best-effort: try REST shape first (.block_height / .blocks[0].number),
        then JSON-RPC shape (.result as hex or int).
        """
        h = self._rest.parse_block_height(response_text)
        if h is not None:
            return h
        h = self._jsonrpc.parse_block_height(response_text)
        if h is not None:
            return h
        try:
            obj = json.loads(response_text)
        except json.JSONDecodeError:
            return None
        # Hedera Mirror REST has no conventional block height. Use the latest
        # consensus timestamp as the monotonic health cursor when available.
        if isinstance(obj, dict):
            balance = obj.get("balance")
            if isinstance(balance, dict):
                ts = balance.get("timestamp")
                if isinstance(ts, str) and ts:
                    return _try_int(ts.split(".", 1)[0])
        return None
