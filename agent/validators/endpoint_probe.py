"""Live RPC endpoint and selected-method probes for Agent gates."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from utils.redaction import redact
except ModuleNotFoundError:
    from agent.utils.redaction import redact


REPO_ROOT = Path(__file__).resolve().parents[2]
CHAIN_ADAPTER_CLI = REPO_ROOT / "tools" / "chain_adapters" / "cli.py"
CHAINS_DIR = REPO_ROOT / "config" / "chains"
EVIDENCE_DIR = REPO_ROOT / ".agent" / "evidence" / "endpoint-probes"
DEFAULT_ADDRESS = "0x0000000000000000000000000000000000000000"

# The generic (single-POST-JSON-RPC-call) probe applies to adapter families
# whose RPC transport is a plain `{"jsonrpc": "2.0", "method": ..., "params":
# ...}` POST body — `jsonrpc` (EVM/non-EVM), `substrate` (Substrate's own
# JSON-RPC, e.g. `system_chain`), and `bitcoin_jsonrpc` (Bitcoin-Core-style
# RPC, e.g. `getblockchaininfo`). `rest`/`tendermint`/`hedera_dual` are
# GET-path/REST-shaped transports the generic probe does not build requests
# for; they are excluded here deliberately, not by oversight (see
# `NEW_CHAIN_SAFE_HEALTH_METHOD`'s docstring and known-issues.md).
GENERIC_JSONRPC_PROBE_FAMILIES = {"jsonrpc", "substrate", "bitcoin_jsonrpc"}

# A safe, parameter-less method usable to sanity-check a brand-new chain's
# endpoint before any chain template exists for it (Case 2: "new chain in an
# existing adapter family"). Populated only for families whose generic probe
# is a plain POST JSON-RPC call (see `GENERIC_JSONRPC_PROBE_FAMILIES`) and
# that have one universally-supported such method across real chains in the
# family. `bitcoin_jsonrpc`'s `getblockchaininfo` matches the `health_probe`
# every existing bitcoin_jsonrpc template already declares in `_meta`.
NEW_CHAIN_SAFE_HEALTH_METHOD = {
    "jsonrpc": "eth_chainId",
    "substrate": "system_chain",
    "bitcoin_jsonrpc": "getblockchaininfo",
}

# Families whose RPC transport is GET-path/REST-shaped rather than a plain
# POST JSON-RPC call. The generic JSON-RPC probe above cannot serve these; a
# separate generic REST probe (below) builds GET/POST HTTP requests instead.
REST_SHAPED_FAMILIES = {"rest", "tendermint", "hedera_dual"}

# Like `NEW_CHAIN_SAFE_HEALTH_METHOD` but for REST-shaped families: a
# parameter-less GET path usable to sanity-check a template-less chain's
# endpoint. Only `tendermint` has one that is genuinely universal across real
# chains in the family -- essentially every Cosmos-SDK chain exposes the
# SDK's own `/cosmos/base/tendermint/...` gRPC-gateway REST module regardless
# of the chain's own app-specific API surface (confirmed live on cosmos-hub
# and Juno). `rest`/`hedera_dual` chains are too heterogeneous (Algorand,
# Aptos, Cardano, Tezos, TON, Hedera's mirror node... each has a completely
# different REST API) for any single path to be safe to assume; those two
# fall back to `_probe_bare_reachability` instead (see `validate_rpc_endpoint`).
NEW_CHAIN_SAFE_REST_HEALTH_METHOD = {
    "tendermint": "GET /cosmos/base/tendermint/v1beta1/blocks/latest",
}


def validate_rpc_endpoint(
    chain: str,
    endpoint: str,
    methods: list[str] | None = None,
    address: str = "",
    timeout: float = 3.0,
    adapter_family: str = "",
    method_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Probe a user-provided RPC endpoint with production adapter requests."""
    chain = (chain or "").strip().lower()
    endpoint = (endpoint or "").strip()
    adapter_family = (adapter_family or "").strip().lower()
    method_params = method_params or {}
    result: dict[str, Any] = {
        "chain": chain,
        "endpoint": endpoint,
        "transport": adapter_family or _adapter_family(chain),
        "safe_method": "",
        "selected_method": "",
        "selected_methods": [],
        "params": {},
        "status": "pending",
        "http_status": None,
        "response_shape_hash": "",
        "error": "",
        "evidence_file": "",
        "session": _runtime_session_metadata(),
        "ready": False,
        "checks": [],
        "warnings": [],
        "blockers": [],
    }
    if not chain:
        result["blockers"].append("chain is required")
        return _finalize_result(result)
    if not _valid_endpoint(endpoint):
        result["blockers"].append("endpoint must be an http(s), ws, or wss URL")
        return _finalize_result(result)
    if endpoint.startswith(("ws://", "wss://")):
        result["warnings"].append("websocket endpoints are shape-checked only by this HTTP probe")
        result["ready"] = False
        result["blockers"].append("provide an HTTP endpoint for live RPC method validation")
        return _finalize_result(result)

    selected_methods = methods or _default_methods(chain)
    if _should_use_generic_jsonrpc_probe(chain, result["transport"], selected_methods, method_params):
        return _validate_generic_jsonrpc_endpoint(
            result,
            endpoint=endpoint,
            methods=selected_methods,
            method_params=method_params,
            timeout=timeout,
        )
    if _should_use_generic_rest_probe(chain, result["transport"], selected_methods):
        return _validate_generic_rest_endpoint(
            result,
            endpoint=endpoint,
            methods=selected_methods,
            method_params=method_params,
            timeout=timeout,
        )
    if not _chain_template_exists(chain) and not selected_methods and result["transport"] in REST_SHAPED_FAMILIES:
        safe_rest_method = NEW_CHAIN_SAFE_REST_HEALTH_METHOD.get(result["transport"])
        if safe_rest_method:
            return _validate_generic_rest_endpoint(
                result,
                endpoint=endpoint,
                methods=[safe_rest_method],
                method_params={},
                timeout=timeout,
            )
        # No family-wide safe path exists (`rest`/`hedera_dual` are too
        # heterogeneous). Fall back to a bare reachability check rather than
        # the template-requiring adapter path below, which would crash
        # (`FileNotFoundError`) for a chain with no `config/chains/<chain>.json`
        # -- Case 2's defining condition. A real method is validated next, once
        # the user supplies one.
        return _validate_bare_reachability(result, endpoint=endpoint, timeout=timeout)

    health = _probe_health(chain, endpoint, timeout)
    result["safe_method"] = str(health.get("rpc_method") or health.get("name") or "endpoint_health_probe")
    result["checks"].append(health)
    result["selected_methods"] = selected_methods[:5]
    result["selected_method"] = selected_methods[0] if selected_methods else ""
    sample_address = address or _sample_address(chain)
    for method in selected_methods[:5]:
        result["checks"].append(_probe_method(chain, endpoint, method, sample_address, timeout))

    failed = [item for item in result["checks"] if not item.get("passed")]
    result["ready"] = not failed
    result["blockers"].extend(f"{item.get('name')}: {item.get('detail')}" for item in failed)
    return _finalize_result(result)


def _should_use_generic_jsonrpc_probe(
    chain: str,
    adapter_family: str,
    methods: list[str],
    method_params: dict[str, Any],
) -> bool:
    if _chain_template_exists(chain):
        if adapter_family in GENERIC_JSONRPC_PROBE_FAMILIES and _has_jsonrpc_custom_probe(chain, methods, method_params):
            return True
        return False
    if adapter_family in GENERIC_JSONRPC_PROBE_FAMILIES:
        return True
    return bool(methods) and all(_looks_like_jsonrpc_method(method) for method in methods)


def _has_jsonrpc_custom_probe(chain: str, methods: list[str], method_params: dict[str, Any]) -> bool:
    if not methods:
        return False
    default_methods = set(_default_methods(chain))
    if any(method not in default_methods for method in methods):
        return True
    return any(method in method_params for method in methods)


def _validate_generic_jsonrpc_endpoint(
    result: dict[str, Any],
    *,
    endpoint: str,
    methods: list[str],
    method_params: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    selected_methods = list(dict.fromkeys([method.strip() for method in methods if method.strip()]))[:5]
    result["transport"] = "jsonrpc"
    result["selected_methods"] = selected_methods
    result["selected_method"] = selected_methods[0] if selected_methods else ""
    if not selected_methods:
        result["blockers"].append("generic JSON-RPC validation requires at least one method or request sample")
        return _finalize_result(result)
    # Use the first selected method as the liveness/health indicator instead of
    # assuming eth_chainId (EVM-only): the caller passes the chain's own health
    # method first, so a non-EVM jsonrpc chain (solana getHealth, etc.) is not
    # rejected by an EVM method it does not implement.
    health_method = selected_methods[0]
    result["safe_method"] = health_method
    result["checks"].append(_probe_generic_jsonrpc_method(endpoint, health_method, method_params, timeout, name="endpoint_health_probe"))
    for method in selected_methods:
        result["checks"].append(_probe_generic_jsonrpc_method(endpoint, method, method_params, timeout))
    failed = [item for item in result["checks"] if not item.get("passed")]
    result["ready"] = not failed
    result["blockers"].extend(f"{item.get('name')}: {item.get('detail')}" for item in failed)
    if result["ready"]:
        if _chain_template_exists(str(result.get("chain") or "")):
            result["warnings"].append(
                "generic JSON-RPC probe passed for a custom method or explicit method sample; keep changes job-local until workload and fixture gates pass"
            )
        else:
            result["warnings"].append(
                "generic JSON-RPC probe passed for an unsupported chain; create a reviewed job-local chain override or chain template before benchmark execution"
            )
    return _finalize_result(result)


def _probe_generic_jsonrpc_method(
    endpoint: str,
    method: str,
    method_params: dict[str, Any],
    timeout: float,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    params = method_params.get(method, [])
    if not isinstance(params, (list, dict)):
        return _check(
            name or f"method_probe:{method}",
            False,
            "generic JSON-RPC params must be a list or object; provide a request sample",
            rpc_method=method,
            params={"params": "<invalid>"},
        )
    request_data = {
        "method": "POST",
        "url": endpoint,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, separators=(",", ":")),
    }
    status, sample = _call_request(request_data, timeout)
    return _check(
        name or f"method_probe:{method}",
        _acceptable_status(status, sample),
        f"status={status}, sample={sample[:160]}",
        rpc_method=method,
        params=redact(params),
        http_status=status,
        response_shape_hash=_response_shape_hash(sample),
        response_sample=sample[:512],
    )


def _should_use_generic_rest_probe(chain: str, adapter_family: str, methods: list[str]) -> bool:
    """Mirror of `_should_use_generic_jsonrpc_probe` for GET-path/REST methods.

    A templated chain already goes through the correct per-chain adapter
    (`_probe_health`/`_probe_method`, which knows the chain's real REST param
    formats); this only applies to a template-less chain (Case 2) whose
    user-supplied method(s) are themselves `GET /path`-shaped.
    """

    if _chain_template_exists(chain):
        return False
    if adapter_family not in REST_SHAPED_FAMILIES:
        return False
    return bool(methods) and all(_looks_like_rest_method(method) for method in methods)


def _looks_like_rest_method(value: str) -> bool:
    return bool(re.match(r"^(GET|POST|PUT|PATCH|DELETE)\s+/", (value or "").strip(), re.IGNORECASE))


def _split_rest_method(method: str) -> tuple[str, str]:
    match = re.match(r"^(GET|POST|PUT|PATCH|DELETE)\s+(/\S*)", method.strip(), re.IGNORECASE)
    if not match:
        return "GET", method.strip()
    return match.group(1).upper(), match.group(2)


def _fill_rest_path_placeholders(path: str, params: Any) -> str:
    """Substitute `{placeholder}` path segments with the schema-evidence value.

    A genuinely new chain has no known address/id format, so this trusts
    whatever the user supplied as schema evidence (the same trust model
    `_probe_generic_jsonrpc_method` already applies to JSON-RPC `params`)
    rather than guessing a chain-specific sample value. A single scalar
    param fills the first placeholder; a list fills placeholders in order.
    Query-string placeholders (e.g. `?account.id={addr}`) are filled the
    same way as path placeholders.
    """

    values: list[str] = []
    if isinstance(params, list):
        values = [str(item) for item in params]
    elif isinstance(params, (str, int, float)) and str(params):
        values = [str(params)]
    if not values:
        return path

    def _replace(_match: re.Match) -> str:
        return values.pop(0) if values else _match.group(0)

    return re.sub(r"\{[^{}]+\}", _replace, path)


def _validate_generic_rest_endpoint(
    result: dict[str, Any],
    *,
    endpoint: str,
    methods: list[str],
    method_params: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    selected_methods = list(dict.fromkeys([method.strip() for method in methods if method.strip()]))[:5]
    result["transport"] = result.get("transport") or "rest"
    result["selected_methods"] = selected_methods
    result["selected_method"] = selected_methods[0] if selected_methods else ""
    if not selected_methods:
        result["blockers"].append("generic REST validation requires at least one GET/POST path")
        return _finalize_result(result)
    health_method = selected_methods[0]
    result["safe_method"] = health_method
    result["checks"].append(_probe_generic_rest_method(endpoint, health_method, method_params, timeout, name="endpoint_health_probe"))
    for method in selected_methods:
        result["checks"].append(_probe_generic_rest_method(endpoint, method, method_params, timeout))
    failed = [item for item in result["checks"] if not item.get("passed")]
    result["ready"] = not failed
    result["blockers"].extend(f"{item.get('name')}: {item.get('detail')}" for item in failed)
    if result["ready"]:
        result["warnings"].append(
            "generic REST probe passed for an unsupported chain; create a reviewed job-local chain override or chain template before benchmark execution"
        )
    return _finalize_result(result)


def _probe_generic_rest_method(
    endpoint: str,
    method: str,
    method_params: dict[str, Any],
    timeout: float,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    http_method, path = _split_rest_method(method)
    params = method_params.get(method, [])
    filled_path = _fill_rest_path_placeholders(path, params)
    url = endpoint.rstrip("/") + filled_path
    request_data = {"method": http_method, "url": url, "headers": {}, "body": ""}
    status, sample = _call_request(request_data, timeout)
    return _check(
        name or f"method_probe:{method}",
        _acceptable_status(status, sample),
        f"status={status}, sample={sample[:160]}",
        rpc_method=method,
        params=redact(params),
        http_status=status,
        response_shape_hash=_response_shape_hash(sample),
        response_sample=sample[:512],
    )


def _validate_bare_reachability(result: dict[str, Any], *, endpoint: str, timeout: float) -> dict[str, Any]:
    """Minimal sanity check for a template-less chain in a REST-shaped family

    with no single safe path known across the whole family (`rest`/
    `hedera_dual` -- Algorand, Aptos, Cardano, Tezos, TON, Hedera's mirror
    node all expose completely different REST APIs). Accepts *any* completed
    HTTP response (2xx through 5xx) as "the server is alive"; a genuine
    protocol/method-level check happens once the user supplies a real method,
    via `_validate_generic_rest_endpoint` above. This only needs to catch a
    dead host, a typo'd domain, or the wrong protocol -- not validate the
    chain's actual API, which the untemplated chain's protocol is not yet
    known well enough to do.
    """

    result["transport"] = result.get("transport") or "rest"
    result["safe_method"] = "endpoint_health_probe"
    status, sample = _call_request({"method": "GET", "url": endpoint, "headers": {}, "body": ""}, timeout)
    reachable = isinstance(status, int)
    check = _check(
        "endpoint_health_probe",
        reachable,
        f"status={status}, sample={sample[:160]}",
        http_status=status,
        response_shape_hash=_response_shape_hash(sample),
        response_sample=sample[:512],
    )
    result["checks"].append(check)
    result["ready"] = reachable
    if not reachable:
        result["blockers"].append(f"endpoint_health_probe: {check['detail']}")
    else:
        result["warnings"].append(
            "bare reachability check only (no chain-specific method validated yet); provide a real method next to validate the actual API"
        )
    return _finalize_result(result)


def _probe_health(chain: str, endpoint: str, timeout: float) -> dict[str, Any]:
    try:
        target = _run_adapter_json(["health-probe", "--chain", chain, "--rpc-url", endpoint])
        status, sample = _call_request(_request_from_probe(target), timeout)
        return _check(
            "endpoint_health_probe",
            _acceptable_status(status, sample),
            f"status={status}, sample={sample[:160]}",
            rpc_method=str(target.get("rpc_method") or target.get("body", ""))[:120],
            http_status=status,
            response_shape_hash=_response_shape_hash(sample),
            response_sample=sample[:512],
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic gate must report exact blocker
        return _check("endpoint_health_probe", False, f"{type(exc).__name__}: {exc}")


def _probe_method(chain: str, endpoint: str, method: str, address: str, timeout: float) -> dict[str, Any]:
    try:
        target = _run_adapter_json([
            "build-target",
            "--chain",
            chain,
            "--method",
            method,
            "--address",
            address,
            "--rpc-url",
            endpoint,
        ])
        status, sample = _call_request(_request_from_target(target), timeout)
        request_data = _request_from_target(target)
        return _check(
            f"method_probe:{method}",
            _acceptable_status(status, sample),
            f"status={status}, sample={sample[:160]}",
            rpc_method=method,
            params=_extract_params(request_data),
            http_status=status,
            response_shape_hash=_response_shape_hash(sample),
            response_sample=sample[:512],
        )
    except Exception as exc:  # noqa: BLE001
        return _check(f"method_probe:{method}", False, f"{type(exc).__name__}: {exc}")


def _run_adapter_json(args: list[str]) -> dict[str, Any]:
    output = subprocess.check_output(
        ["python3", str(CHAIN_ADAPTER_CLI), *args],
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )
    return json.loads(output)


def _call_request(request_data: dict[str, Any], timeout: float) -> tuple[int | str, str]:
    body = request_data.get("body")
    # An empty string is falsy but is still a `str`, so the old
    # `isinstance(body, str) and body` guard fell through to `else body` for
    # GET-shaped requests (whose body is always "") and passed the literal
    # empty string as `data` to `urllib.request.Request` — urllib requires
    # `data` to be `None`/bytes/an iterable of bytes, so every GET-method probe
    # (the default method for bitcoin_jsonrpc/rest/hedera_dual/tendermint
    # families and some substrate mixed methods) raised
    # "TypeError: POST data should be bytes..." regardless of endpoint health.
    data = body.encode("utf-8") if isinstance(body, str) and body else None
    headers = dict(request_data.get("headers") or {})
    headers.setdefault("User-Agent", "AnyChain-Benchmark-Agent/1.0")
    req = urllib.request.Request(
        request_data["url"],
        data=data,
        method=request_data.get("method", "GET"),
        headers=headers,
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read(4096).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(4096).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return "ERR", f"{type(exc).__name__}: {exc}; elapsed={time.time() - started:.2f}s"


def _request_from_probe(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": probe.get("method", "GET"),
        "url": probe["url"],
        "headers": probe.get("headers", {}),
        "body": probe.get("body", ""),
    }


def _request_from_target(target: dict[str, Any]) -> dict[str, Any]:
    headers = {}
    for key, value in (target.get("header") or {}).items():
        headers[key] = value[0] if isinstance(value, list) and value else value
    body = base64.b64decode(target["body"]).decode("utf-8") if target.get("body") else ""
    return {
        "method": target.get("method", "GET"),
        "url": target["url"],
        "headers": headers,
        "body": body,
    }


def _acceptable_status(status: int | str, sample: str) -> bool:
    if not isinstance(status, int) or status < 200 or status >= 300:
        return False
    try:
        payload = json.loads(sample)
    except Exception:
        return True
    if isinstance(payload, dict) and payload.get("error"):
        return False
    return True


def _valid_endpoint(value: str) -> bool:
    parsed = urlparse(value or "")
    return parsed.scheme in {"http", "https", "ws", "wss"} and bool(parsed.netloc)


def _chain_template_exists(chain: str) -> bool:
    return bool(chain) and (CHAINS_DIR / f"{chain}.json").is_file()


def _looks_like_jsonrpc_method(method: str) -> bool:
    value = (method or "").strip()
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+$", value))


def _default_methods(chain: str) -> list[str]:
    try:
        template = json.loads((CHAINS_DIR / f"{chain}.json").read_text(encoding="utf-8"))
    except Exception:
        return []
    rpc = template.get("rpc_methods", {}) if isinstance(template.get("rpc_methods"), dict) else {}
    methods: list[str] = []
    single = rpc.get("single")
    if isinstance(single, str) and single:
        methods.append(single)
    weighted = rpc.get("mixed_weighted") or []
    if isinstance(weighted, list):
        for item in weighted:
            if isinstance(item, dict) and item.get("method"):
                methods.append(str(item["method"]))
    return list(dict.fromkeys(methods))


_ENV_PLACEHOLDER_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-(.*))?\}$", re.DOTALL)


def _resolve_env_placeholder(value: str) -> str:
    """Resolve a `${VAR:-default}` / `${VAR-default}` / `${VAR}` shell-style
    placeholder the way chain templates store sample values.

    Chain templates keep addresses as `${TARGET_ADDRESS:-0x...}`. Passing that
    raw string to a probe sends the literal `${...}` as the RPC argument, which a
    real node rejects ("hex string without 0x prefix"), so a healthy endpoint is
    wrongly failed. Expand to the env value if set, otherwise the default.
    """

    text = str(value or "").strip()
    match = _ENV_PLACEHOLDER_RE.match(text)
    if not match:
        return text
    name, default = match.group(1), match.group(2)
    env_value = os.environ.get(name, "")
    if env_value:
        return env_value
    return default if default is not None else ""


def _sample_address(chain: str) -> str:
    try:
        template = json.loads((CHAINS_DIR / f"{chain}.json").read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_ADDRESS
    params = template.get("params", {}) if isinstance(template.get("params"), dict) else {}
    value = params.get("target_address") or params.get("address")
    if isinstance(value, str) and value:
        resolved = _resolve_env_placeholder(value)
        if resolved:
            return resolved
    if chain == "solana":
        return "11111111111111111111111111111111"
    if chain == "near":
        return "example.near"
    return DEFAULT_ADDRESS


def _check(name: str, passed: bool, detail: str = "", **extra: Any) -> dict[str, Any]:
    payload = {"name": name, "passed": bool(passed), "detail": detail}
    payload.update(extra)
    return payload


def _finalize_result(result: dict[str, Any]) -> dict[str, Any]:
    checks = result.get("checks") or []
    method_checks = [item for item in checks if str(item.get("name", "")).startswith("method_probe:")]
    selected = method_checks[0] if method_checks else (checks[0] if checks else {})
    if selected:
        result["http_status"] = selected.get("http_status")
        result["response_shape_hash"] = selected.get("response_shape_hash", "")
        result["params"] = selected.get("params", {})
    result["status"] = _derive_status(result)
    if result.get("blockers"):
        result["error"] = str(result["blockers"][0])
    evidence_file = _write_evidence(result)
    result["evidence_file"] = str(evidence_file.relative_to(REPO_ROOT))
    return result


def _derive_status(result: dict[str, Any]) -> str:
    if result.get("ready"):
        return "ok"
    blockers = " | ".join(str(item) for item in result.get("blockers", []))
    if "chain is required" in blockers:
        return "needs_chain"
    if "endpoint must be" in blockers or "provide an HTTP endpoint" in blockers:
        return "needs_endpoint"
    if "endpoint_health_probe" in blockers:
        return "needs_valid_endpoint"
    if "method_probe:" in blockers:
        return "needs_valid_params"
    return "blocked"


def _write_evidence(result: dict[str, Any]) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    chain = _safe_name(str(result.get("chain") or "unknown"))
    digest_input = json.dumps(
        {
            "chain": result.get("chain"),
            "endpoint": result.get("endpoint"),
            "selected_methods": result.get("selected_methods"),
            "status": result.get("status"),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:12]
    path = EVIDENCE_DIR / f"{chain}-{digest}.json"
    path.write_text(json.dumps(redact(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _runtime_session_metadata() -> dict[str, Any]:
    return {
        "id": os.environ.get("ANYCHAIN_AGENT_SESSION_ID") or "default",
        "purpose": os.environ.get("ANYCHAIN_AGENT_SESSION_PURPOSE") or "user",
        "checkpoint_path": os.environ.get("ANYCHAIN_AGENT_CHECKPOINT_PATH") or "",
    }


def _chain_meta(chain: str) -> dict[str, Any]:
    try:
        template = json.loads((CHAINS_DIR / f"{chain}.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    meta = template.get("_meta")
    return meta if isinstance(meta, dict) else {}


def _adapter_family(chain: str) -> str:
    return str(_chain_meta(chain).get("adapter_family") or "")


def health_probe_methods(chain: str, adapter_family: str) -> tuple[list[str] | None, dict[str, Any]]:
    """Return (methods, method_params) for a cheap liveness probe of a chain's endpoint.

    Uses the chain template's declared param-less health method — EVM chains use
    `eth_syncing`, solana uses `getHealth`, etc. — instead of assuming every
    `jsonrpc`-transport chain is EVM (`eth_chainId`), which wrongly rejects
    non-EVM jsonrpc chains (solana/sui/near/starknet/tron/avalanche-x). Returns
    (None, {}) so the probe falls back to the adapter's own per-chain health check
    when no param-less method is declared. For a brand-new chain with no
    template yet (Case 2), falls back to `NEW_CHAIN_SAFE_HEALTH_METHOD`'s
    per-family default when one exists.

    Restricting this to `GENERIC_JSONRPC_PROBE_FAMILIES` families is load-bearing,
    not incidental: for any other family, a template-less chain has no methods
    here, `_should_use_generic_jsonrpc_probe` cannot pick the generic path, and
    `validate_rpc_endpoint` falls through to the template-requiring
    `_probe_health`/`_probe_method` (`tools/chain_adapters/cli.py`), which raises
    `FileNotFoundError` for a chain with no `config/chains/<chain>.json` -- i.e.
    Case 2 endpoint validation was structurally unable to ever pass for any
    family without an entry here (found live testing a genuinely new `substrate`
    chain, Moonriver, against `substrate`'s DEFAULT_GROUP_ORDER path).
    """

    chain = (chain or "").strip().lower()
    adapter_family = (adapter_family or "").strip().lower()
    if adapter_family not in GENERIC_JSONRPC_PROBE_FAMILIES:
        return None, {}
    hp = _chain_meta(chain).get("health_probe")
    hp = hp if isinstance(hp, dict) else {}
    method = str(hp.get("method") or "").strip()
    if method and " " not in method and "/" not in method:
        return [method], {method: list(hp.get("params") or [])}
    if _chain_template_exists(chain):
        return None, {}
    safe_method = NEW_CHAIN_SAFE_HEALTH_METHOD.get(adapter_family)
    if not safe_method:
        return None, {}
    return [safe_method], {safe_method: []}


def _response_shape_hash(sample: str) -> str:
    try:
        parsed = json.loads(sample)
    except Exception:
        parsed = {"_text": type(sample).__name__}
    shape = _shape(parsed)
    encoded = json.dumps(shape, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _shape(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_shape(item) for item in value[:3]]
    return type(value).__name__


def _extract_params(request_data: dict[str, Any]) -> Any:
    body = request_data.get("body")
    if isinstance(body, str) and body:
        try:
            parsed = json.loads(body)
        except Exception:
            return {"body": "<non-json>"}
        if isinstance(parsed, dict):
            return redact(parsed.get("params", {}))
    return {}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"
