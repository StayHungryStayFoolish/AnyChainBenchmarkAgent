#!/usr/bin/env python3
"""Generate deterministic target seed files from the active chain template.

The benchmark supports 36 chains through `config/chains/*.json`. Runtime target
generation is driven by common template fields instead of per-chain alias maps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ENV_DEFAULT_RE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*):-([^}]*)\}$")


def convert_env_value(value: Any) -> Any:
    """Convert placeholder values to JSON, bool, or number when possible."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped.startswith(("[", "{")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return value


def replace_env_vars(obj: Any) -> Any:
    """Resolve `${ENV:-fallback}` and direct ENV-name placeholders."""
    if isinstance(obj, dict):
        return {key: replace_env_vars(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [replace_env_vars(value) for value in obj]
    if not isinstance(obj, str):
        return obj

    default_match = ENV_DEFAULT_RE.match(obj)
    if default_match:
        env_name, fallback = default_match.groups()
        return convert_env_value(os.environ.get(env_name) or fallback)

    if obj in os.environ:
        return convert_env_value(os.environ[obj])

    return obj


def load_chain_config() -> dict[str, Any]:
    raw_config = os.environ.get("CHAIN_CONFIG")
    if not raw_config:
        raise ValueError("CHAIN_CONFIG environment variable is required")

    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in CHAIN_CONFIG: {exc}") from exc

    config = replace_env_vars(config)
    if not config.get("chain_type"):
        raise ValueError("chain_type is required in CHAIN_CONFIG")

    params = config.setdefault("params", {})
    params.setdefault("account_count", 1000)
    params.setdefault("output_file", "active_accounts.txt")
    params.setdefault("target_address", "")
    return config


def int_param(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_args(config: dict[str, Any]) -> argparse.Namespace:
    params = config.get("params", {})
    parser = argparse.ArgumentParser(
        description=f"Generate target seeds for {config.get('chain_type', 'blockchain')}"
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=int_param(params.get("account_count"), 1000),
        help="Number of target seed entries to write.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(params.get("output_file", "active_accounts.txt")),
        help="Output file consumed by tools/target_generator.sh.",
    )
    parser.add_argument(
        "-u",
        "--rpc-url",
        type=str,
        default=str(config.get("rpc_url", "")),
        help="Accepted for CLI compatibility. Target seed generation does not query RPC.",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default=str(params.get("target_address", "")),
        help="Target address override. Defaults to params.target_address.",
    )
    parser.add_argument(
        "-m",
        "--max-signatures",
        type=int,
        default=int_param(params.get("max_signatures"), 50000),
        help="Accepted for CLI compatibility. No live signature scan is performed.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    return parser.parse_args()


def flatten_system_addresses(system_addresses: Any) -> list[str]:
    if isinstance(system_addresses, dict):
        values = system_addresses.values()
    elif isinstance(system_addresses, list):
        values = system_addresses
    else:
        return []

    addresses: list[str] = []
    for value in values:
        if isinstance(value, str) and value:
            addresses.append(value)
        elif isinstance(value, list):
            addresses.extend(str(item) for item in value if item)
    return addresses


def selected_rpc_methods(config: dict[str, Any]) -> list[str]:
    rpc_methods = config.get("rpc_methods")
    if not isinstance(rpc_methods, dict):
        return []

    mode = str(os.environ.get("RPC_MODE", "single")).strip().lower()
    if mode == "mixed":
        weighted = rpc_methods.get("mixed_weighted")
        if isinstance(weighted, list):
            methods = [
                str(item.get("method", "")).strip()
                for item in weighted
                if isinstance(item, dict) and str(item.get("method", "")).strip()
            ]
            if methods:
                return methods

    raw_methods = rpc_methods.get(mode)
    if isinstance(raw_methods, str):
        return [method.strip() for method in raw_methods.split(",") if method.strip()]
    if isinstance(raw_methods, list):
        return [str(method).strip() for method in raw_methods if str(method).strip()]
    return []


def method_requires_target_seed(config: dict[str, Any], method: str) -> bool:
    param_spec = config.get("param_spec")
    if isinstance(param_spec, dict):
        spec = param_spec.get(method)
        if isinstance(spec, dict) and isinstance(spec.get("params"), list):
            return len(spec["params"]) > 0

    param_formats = config.get("param_formats")
    if isinstance(param_formats, dict) and param_formats.get(method) == "no_params":
        return False
    return True


def workload_requires_target_seed(config: dict[str, Any]) -> bool:
    methods = selected_rpc_methods(config)
    if not methods:
        return True
    return any(method_requires_target_seed(config, method) for method in methods)


def select_target_seed(config: dict[str, Any], requested_target: str) -> str:
    if requested_target:
        return requested_target

    params = config.get("params", {})
    target_address = params.get("target_address")
    if isinstance(target_address, str) and target_address:
        return target_address

    system_addresses = flatten_system_addresses(config.get("system_addresses"))
    if system_addresses:
        return system_addresses[0]

    if not workload_requires_target_seed(config):
        return "__ANYCHAIN_NO_TARGET__"

    raise ValueError(
        "No target seed is available. Set TARGET_ADDRESS in config/user_config.sh "
        "or define params.target_address in the chain template."
    )


def write_target_seed_file(output_file: str, target_seed: str, count: int) -> None:
    if count <= 0:
        raise ValueError("--count must be greater than zero")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join([target_seed] * count) + "\n"
    output_path.write_text(lines, encoding="utf-8")


def main() -> int:
    try:
        config = load_chain_config()
        args = parse_args(config)
        target_seed = select_target_seed(config, args.target)
        write_target_seed_file(args.output, target_seed, args.count)

        if args.verbose:
            chain = config.get("chain_type", "unknown")
            print(f"Generated {args.count} target seed entries for {chain}: {args.output}")
            print(f"Target seed: {target_seed}")
            if args.rpc_url:
                print("RPC URL was provided but not queried during target seed generation.")

        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
