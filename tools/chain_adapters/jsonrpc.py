"""JsonRpcAdapter — standard JSON-RPC 2.0 POST.

Covers 16 chains: Ethereum / BSC / Polygon / Base / Scroll / Arbitrum /
Optimism / zkSync-Era / Linea / Solana / Sui / StarkNet / NEAR /
Avalanche-C / Avalanche-X / Tron.

Param formats (from baseline target_generator.sh L77-109):
    no_params              → []
    single_address         → ["<addr>"]
    address_latest         → ["<addr>", "latest"]      (EVM)
    latest_address         → ["latest", "<addr>"]      (StarkNet)
    address_storage_latest → ["<addr>", "0x0", "latest"]   (eth_getStorageAt)
    address_key_latest     → ["<addr>", "0x1", "latest"]   (starknet_getStorageAt)
    address_with_options   → ["<addr>", {showType:true, showContent:true, showDisplay:false}]  (sui)

Health probe per-subchain method (from common_functions.sh L194-275):
    solana → getBlockHeight    → decimal int
    EVM    → eth_blockNumber   → 0x-hex
    starknet → starknet_blockNumber → decimal int
    sui    → sui_getTotalTransactionBlocks → decimal int

Chains beyond these baseline-5 (NEAR, tron, arbitrum, optimism, zksync-era,
linea, avalanche-c, avalanche-x) — health uses chain-template-specific method
read from rpc_methods.single field. Default health probe = chain's single method
with empty params, expecting either decimal int or 0x-hex result.
"""
from __future__ import annotations
import json
import re
import os
from pathlib import Path
from typing import Optional

from .base import ChainAdapter, register, _vegeta_post_json, _try_int, load_chain_template
from .param_spec import apply_rest_param_spec, build_jsonrpc_params, get_param_spec
from .url_overrides import resolve_param

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CHAINS_DIR = _REPO_ROOT / "config" / "chains"


@register("jsonrpc")
class JsonRpcAdapter(ChainAdapter):

    def __init__(self):
        self._chain_cache: dict[str, dict] = {}

    def _load_chain(self, chain_name: str) -> dict:
        if chain_name not in self._chain_cache:
            self._chain_cache[chain_name] = load_chain_template(chain_name)
        return self._chain_cache[chain_name]

    def build_vegeta_target(
        self, method: str, address: str, rpc_url: str, param_format: str = "",
    ) -> dict:
        chain_name = os.environ.get("BLOCKCHAIN_NODE", "").lower()
        tpl = self._load_chain(chain_name) if chain_name else {}
        if method.startswith("/"):
            param_spec = get_param_spec(tpl, method)
            if param_spec and param_spec.get("transport") == "rest_body":
                _, body, _ = apply_rest_param_spec(param_spec, method, {}, tpl, address)
            else:
                body = self._build_path_body(method, param_format, address)
            return _vegeta_post_json(rpc_url.rstrip("/") + method, body)
        if method.startswith("eth_") and rpc_url.rstrip("/") == "https://api.trongrid.io":
            rpc_url = rpc_url.rstrip("/") + "/jsonrpc"
        if tpl.get("chain_type") == "near":
            params = self._build_near_params(method, address, tpl)
            body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
            return _vegeta_post_json(rpc_url, body)
        if (
            tpl.get("chain_type") == "solana"
            and method == "getTokenAccountBalance"
            and "localhost" not in rpc_url
            and "127.0.0.1" not in rpc_url
        ):
            cfg_params = tpl.get("params", {}) if isinstance(tpl.get("params"), dict) else {}
            params = [resolve_param(cfg_params, "target_token_account", address)]
            body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
            return _vegeta_post_json(rpc_url, body)
        param_spec = get_param_spec(tpl, method)
        if param_spec:
            params = build_jsonrpc_params(param_spec, tpl, address)
        else:
            params = self._build_params(param_format, address, tpl)
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        return _vegeta_post_json(rpc_url, body)

    @staticmethod
    def _build_near_params(method: str, address: str, tpl: dict):
        cfg_params = tpl.get("params", {}) if isinstance(tpl, dict) else {}
        if method == "query":
            return {"request_type": "view_account", "finality": "final", "account_id": address}
        if method == "block":
            return {"finality": "final"}
        if method == "gas_price":
            return [None]
        if method == "validators":
            return [None]
        if method == "tx":
            return [
                resolve_param(cfg_params, "target_tx_hash", address),
                resolve_param(cfg_params, "target_signer_id", address),
            ]
        return []

    def _build_path_body(self, method: str, param_format: str, address: str) -> dict:
        chain_name = os.environ.get("BLOCKCHAIN_NODE", "").lower()
        tpl = self._load_chain(chain_name) if chain_name else {}
        params = tpl.get("params", {}) if isinstance(tpl.get("params"), dict) else {}
        txid = resolve_param(params, "target_txid", resolve_param(params, "txid", address))
        contract = resolve_param(params, "target_contract_address", address)
        if param_format == "no_params" or method == "/wallet/getnowblock":
            return {}
        if param_format == "body_value_txid_nopfx" or method == "/wallet/gettransactionbyid":
            return {"value": txid}
        if param_format == "body_owner_contract_selector_parameter" or method == "/wallet/triggerconstantcontract":
            return {
                "owner_address": address,
                "contract_address": contract,
                "function_selector": "balanceOf(address)",
                "parameter": "0000000000000000000000000000000000000000000000000000000000000000",
                "visible": True,
            }
        return {"address": address, "visible": True}

    @staticmethod
    def _build_params(param_format: str, address: str, tpl: dict | None = None):
        cfg_params = tpl.get("params", {}) if isinstance(tpl, dict) else {}
        # Baseline parameter formats
        if param_format == "no_params":
            return []
        if param_format == "height_encoding":
            return {"height": int(resolve_param(cfg_params, "target_height", 1)), "encoding": "json"}
        if param_format == "txid_encoding":
            return {"txID": resolve_param(cfg_params, "target_txid", address), "encoding": "json"}
        if param_format == "addresses_limit_encoding":
            return {"addresses": [address], "limit": 1, "encoding": "hex"}
        if param_format == "single_address" and tpl and tpl.get("chain_type") == "avalanche-x":
            return {"address": address}
        if param_format == "single_address":
            return [address]
        if param_format == "address_latest":
            return [address, "latest"]
        if param_format == "latest_address":
            return ["latest", address]
        if param_format == "address_storage_latest":
            return [address, "0x0", "latest"]
        if param_format == "address_key_latest":
            return [address, "0x1", "latest"]
        if param_format == "address_with_options":
            # Sui-style: [addr, {showType, showContent, showDisplay}]
            return [
                address,
                {"showType": True, "showContent": True, "showDisplay": False},
            ]
        # chain-specific: EVM standard formats (arbitrum/optimism/zksync-era/linea/avalanche-c)
        if param_format == "block_number":
            # eth_getBlockByNumber → ["latest", false]   (don't return full txs)
            return ["latest", False]
        if param_format == "block_number_int":
            # zks_getBlockDetails / similar → [<int>]
            # Use a recent block number placeholder; production callers should
            # override via injected `address` (parseable as int) for real load
            try:
                bn = int(address)
            except (TypeError, ValueError):
                bn = 1  # safe fallback — node returns null but request is valid
            return [bn]
        if param_format == "transaction_hash":
            # eth_getTransactionByHash/Receipt → [<tx_hash>]
            # `address` arg is repurposed as tx_hash by the caller; if not given,
            # use a 32-byte placeholder (vegeta still gets a syntactically valid
            # request and node returns null result, which counts as success)
            tx_hash = address if address.startswith("0x") and len(address) == 66 else "0x" + "0" * 64
            return [tx_hash]
        if param_format == "eth_call_object_latest":
            # eth_call → [{to, data}, "latest"]
            # data = balanceOf(0x0)  → 0x70a08231 + 32-byte padded zero
            return [
                {"to": address, "data": "0x70a08231" + "0" * 64},
                "latest",
            ]
        if param_format == "object_single":
            # linea_estimateGas / similar → [{from, to, value, ...}] (no latest)
            return [{"from": address, "to": address, "value": "0x1"}]
        # default fallback (matches old target_generator.sh L107)
        return [address]

    def health_check_request(self, rpc_url: str) -> dict:
        """Build the per-chain block-height probe.

        The jsonrpc family covers several protocols whose height method is not
        always `eth_blockNumber`; keep this mapping close to the adapter that
        already knows each chain's request shape.
        """
        chain_name = os.environ.get("BLOCKCHAIN_NODE", "").lower()
        tpl = self._load_chain(chain_name) if chain_name else {}
        chain_type = tpl.get("chain_type", chain_name)
        probe = tpl.get("_meta", {}).get("health_probe", {}) if isinstance(tpl, dict) else {}

        if chain_type == "tron" and rpc_url.rstrip("/") == "https://api.trongrid.io":
            rpc_url = rpc_url.rstrip("/") + "/jsonrpc"

        if probe:
            rpc_method = probe.get("rpc_method") or probe.get("method")
            params = probe.get("params", [])
            if not isinstance(params, list):
                params = []
            body = {"jsonrpc": "2.0", "id": 1, "method": rpc_method, "params": params}
            parse_jq = probe.get("parse_jq", ".result")
        elif chain_type == "solana":
            body = {"jsonrpc": "2.0", "id": 1, "method": "getBlockHeight", "params": []}
            parse_jq = ".result"
        elif chain_type == "starknet":
            body = {"jsonrpc": "2.0", "id": 1, "method": "starknet_blockNumber", "params": []}
            parse_jq = ".result"
        elif chain_type == "sui":
            body = {"jsonrpc": "2.0", "id": 1, "method": "sui_getTotalTransactionBlocks", "params": []}
            parse_jq = ".result"
        elif chain_type == "near":
            body = {"jsonrpc": "2.0", "id": 1, "method": "block", "params": {"finality": "final"}}
            parse_jq = ".result.header.height"
        elif chain_type == "avalanche-x":
            body = {"jsonrpc": "2.0", "id": 1, "method": "avm.getHeight", "params": {}}
            parse_jq = ".result.height"
        else:
            body = {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}
            parse_jq = ".result"
        return {
            "method": "POST",
            "url": rpc_url,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, separators=(",", ":")),
            "parse_jq": parse_jq,
        }

    def parse_block_height(self, response_text: str) -> Optional[int]:
        """Try .result as a height cursor or reported-lag signal."""
        if not response_text:
            return None
        try:
            obj = json.loads(response_text)
        except json.JSONDecodeError:
            return None
        error = obj.get("error")
        if isinstance(error, dict):
            data = error.get("data")
            if isinstance(data, dict):
                parsed = _try_int(data.get("numSlotsBehind"))
                if parsed is not None:
                    return parsed
        result = obj.get("result")
        if result is False:
            return 0
        if isinstance(result, str) and result.lower() == "ok":
            return 0
        parsed = _try_int(result)
        if parsed is not None:
            return parsed
        if isinstance(result, dict):
            current_block = _try_int(result.get("currentBlock"))
            highest_block = _try_int(result.get("highestBlock"))
            if current_block is not None and highest_block is not None:
                return max(highest_block - current_block, 0)
            for path in (
                ("header", "height"),        # NEAR block
                ("height",),                 # Avalanche-X avm.getHeight
                ("block_id", "seqno"),       # TON-style fallback if routed here
            ):
                cur = result
                for key in path:
                    if not isinstance(cur, dict):
                        cur = None
                        break
                    cur = cur.get(key)
                parsed = _try_int(cur)
                if parsed is not None:
                    return parsed
        return None
