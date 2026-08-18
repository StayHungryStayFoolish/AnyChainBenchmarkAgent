"""CLI bridge — called from bash (target_generator.sh, common_functions.sh)
to access ChainAdapter functionality.

Usage:
    python3 cli.py build-target --chain ethereum --method eth_getBalance \\
        --address 0xabc... --rpc-url http://localhost:8545 \\
        [--param-format address_latest]

    python3 cli.py health-probe --chain ethereum --rpc-url http://localhost:8545

    python3 cli.py family --chain ethereum
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# cli.py lives in tools/chain_adapters/. Add tools/ to sys.path so
# `from chain_adapters import …` resolves as a top-level package.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))           # tools/chain_adapters
_TOOLS_DIR = os.path.dirname(_THIS_DIR)                          # tools
sys.path.insert(0, _TOOLS_DIR)

from chain_adapters import get_adapter  # noqa: E402
from chain_adapters.base import load_chain_template  # noqa: E402
from chain_adapters.param_spec import get_param_spec  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CHAINS_DIR = _REPO_ROOT / "config" / "chains"


def _get_param_format(chain: str, method: str) -> str:
    """Read param_format from chain template `param_formats.<method>`.

    Mirrors config_loader.sh:600 `get_param_format_from_json()` (bash path):
      - reads from `param_formats` (method→format map), NOT `params`
        (which holds fetcher config like account_count/output_file).
      - default fallback is "single_address" (matches bash line 105 case).

    Historical bug note:
      An earlier implementation accidentally read tpl["params"] (fetcher
      config dict whose values are bash env var names like "ACCOUNT_COUNT")
      and fell back to "". The JsonRpcAdapter's own default fallback is also
      `[address]`, so byte-equality test_3 happened to pass via symmetric
      fallback — but real production calls had wrong params:
        * eth_getBalance(addr) → [addr]   (should be [addr, "latest"])
        * eth_blockNumber()    → [addr]   (should be [])
      Discovered when hedera_dual mixed C1 live-curl returned HTTP 400.
      See KNOWN_BROKEN_MIXED in tests/test_chain_adapters.py.
    """
    tpl = load_chain_template(chain)
    param_formats = tpl.get("param_formats", {})
    if isinstance(param_formats, dict):
        return param_formats.get(method, "single_address")
    return "single_address"


def _load_chain_template(chain: str) -> dict:
    return load_chain_template(chain)


def _chain_methods(tpl: dict) -> list[str]:
    methods: list[str] = []
    rpc_methods = tpl.get("rpc_methods", {})
    single = rpc_methods.get("single")
    if isinstance(single, str) and single:
        methods.append(single)
    mixed = rpc_methods.get("mixed", [])
    if isinstance(mixed, list):
        methods.extend(m for m in mixed if isinstance(m, str) and m)
    mixed_weighted = rpc_methods.get("mixed_weighted", [])
    if isinstance(mixed_weighted, list):
        for item in mixed_weighted:
            if isinstance(item, dict) and isinstance(item.get("method"), str):
                methods.append(item["method"])
            elif isinstance(item, str):
                methods.append(item)
    return sorted(set(methods))


def _sample_address(tpl: dict) -> str:
    params = tpl.get("params", {}) if isinstance(tpl.get("params"), dict) else {}
    value = params.get("target_address") or params.get("address")
    if isinstance(value, str) and value:
        return value
    chain_type = str(tpl.get("chain_type", ""))
    if chain_type in {"solana"}:
        return "11111111111111111111111111111111"
    if chain_type in {"near"}:
        return "example.near"
    return "0x0000000000000000000000000000000000000000"


def _method_has_declared_builder(tpl: dict, method: str) -> bool:
    param_formats = tpl.get("param_formats", {})
    rest_paths = tpl.get("_meta", {}).get("rest_paths", {})
    param_spec = tpl.get("param_spec", {})
    return (
        method.startswith("/")
        or (isinstance(param_formats, dict) and method in param_formats)
        or (isinstance(rest_paths, dict) and method in rest_paths)
        or (isinstance(param_spec, dict) and method in param_spec)
    )


def cmd_validate_template(args):
    chains = [p.stem for p in sorted(_CHAINS_DIR.glob("*.json"))] if args.chain == "all" else [args.chain]
    failures: list[str] = []
    for chain in chains:
        try:
            tpl = _load_chain_template(chain)
            adapter = get_adapter(chain)
            os.environ["BLOCKCHAIN_NODE"] = chain
            methods = _chain_methods(tpl)
            if not methods:
                raise ValueError("no rpc_methods.single/mixed/mixed_weighted methods found")
            used_methods = set(methods)
            specs = tpl.get("param_spec", {})
            if specs is not None and not isinstance(specs, dict):
                raise ValueError("param_spec must be an object when configured")
            if isinstance(specs, dict):
                for method in specs:
                    get_param_spec(tpl, method)
                    if method not in used_methods:
                        print(f"WARN {chain}/{method}: param_spec is not referenced by rpc_methods")
            address = _sample_address(tpl)
            for method in methods:
                if not _method_has_declared_builder(tpl, method):
                    raise ValueError(
                        f"{method}: missing param_formats, _meta.rest_paths, or param_spec entry"
                    )
                param_format = _get_param_format(chain, method)
                adapter.build_vegeta_target(
                    method=method,
                    address=address,
                    rpc_url=args.rpc_url,
                    param_format=param_format,
                )
            print(f"OK {chain}: {len(methods)} methods")
        except Exception as exc:
            failures.append(f"{chain}: {exc}")
            print(f"ERROR {chain}: {exc}", file=sys.stderr)
    if failures:
        print("\nTemplate validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        sys.exit(1)


def cmd_build_target(args):
    adapter = get_adapter(args.chain)
    # Honor BLOCKCHAIN_NODE so RestAdapter can resolve _meta.rest_paths
    # --chain is the explicit user intent; ALWAYS override any inherited
    # BLOCKCHAIN_NODE env var (setdefault is wrong here: it lets a stale
    # parent-process value silently hijack the call and route RestAdapter
    # to the wrong chain template — caught in when test_7 leaked
    # BLOCKCHAIN_NODE=aptos into the test_10 subprocess and all algorand
    # CLI calls then queried aptos's rest_paths).
    os.environ["BLOCKCHAIN_NODE"] = args.chain
    param_format = args.param_format or _get_param_format(args.chain, args.method)
    target = adapter.build_vegeta_target(
        method=args.method,
        address=args.address,
        rpc_url=args.rpc_url,
        param_format=param_format,
    )
    # Vegeta accepts compact JSON, one target per line
    print(json.dumps(target, separators=(",", ":")))


def cmd_build_targets_batch(args):
    """Read tab-separated (method\\taddress) pairs from stdin,
    one per line. Emit one vegeta target JSON per line, in order.

    Hugely faster than calling cli.py per target (Python startup cost
    amortizes across all targets in one process).
    """
    adapter = get_adapter(args.chain)
    os.environ["BLOCKCHAIN_NODE"] = args.chain  # always override; see cmd_build_target
    # Pre-cache param_format per method
    pf_cache: dict[str, str] = {}
    out = sys.stdout
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        if "\t" in line:
            method, address = line.split("\t", 1)
        else:
            # Allow space separator as fallback
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            method, address = parts
        if method not in pf_cache:
            pf_cache[method] = _get_param_format(args.chain, method)
        target = adapter.build_vegeta_target(
            method=method, address=address,
            rpc_url=args.rpc_url, param_format=pf_cache[method],
        )
        out.write(json.dumps(target, separators=(",", ":")) + "\n")
    out.flush()


def cmd_health_probe(args):
    adapter = get_adapter(args.chain)
    os.environ["BLOCKCHAIN_NODE"] = args.chain  # always override; see cmd_build_target
    req = adapter.health_check_request(args.rpc_url)
    print(json.dumps(req, separators=(",", ":")))


def cmd_family(args):
    adapter = get_adapter(args.chain)
    print(adapter.protocol_family)


def cmd_parse_height(args):
    adapter = get_adapter(args.chain)
    response = sys.stdin.read()
    height = adapter.parse_block_height(response)
    if height is None:
        print("N/A")
        sys.exit(1)
    print(height)


def main():
    ap = argparse.ArgumentParser(prog="chain_adapters.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-target")
    b.add_argument("--chain", required=True)
    b.add_argument("--method", required=True)
    b.add_argument("--address", required=True)
    b.add_argument("--rpc-url", required=True)
    b.add_argument("--param-format", default="")
    b.set_defaults(func=cmd_build_target)

    bb = sub.add_parser("build-targets-batch",
        help="Read TSV (method\\taddress) from stdin, emit vegeta JSON per line")
    bb.add_argument("--chain", required=True)
    bb.add_argument("--rpc-url", required=True)
    bb.set_defaults(func=cmd_build_targets_batch)

    h = sub.add_parser("health-probe")
    h.add_argument("--chain", required=True)
    h.add_argument("--rpc-url", required=True)
    h.set_defaults(func=cmd_health_probe)

    f = sub.add_parser("family")
    f.add_argument("--chain", required=True)
    f.set_defaults(func=cmd_family)

    ph = sub.add_parser("parse-height", help="Read response JSON from stdin, print height")
    ph.add_argument("--chain", required=True)
    ph.set_defaults(func=cmd_parse_height)

    vt = sub.add_parser("validate-template", help="Validate chain template target construction")
    vt.add_argument("--chain", required=True, help="Chain name or 'all'")
    vt.add_argument("--rpc-url", default="http://localhost:19000")
    vt.set_defaults(func=cmd_validate_template)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
