"""Pure normalization and extraction helpers for Harness input values."""

from __future__ import annotations

import json
import re
from typing import Any

import yaml


def normalize_scalar(value: Any) -> str:
    """Trim one scalar using the Harness answer contract."""

    text = str(value or "").strip()
    if len(text) >= 2 and text[0] in {"`", "'", '"'} and text[-1] in {"`", "'", '"'}:
        text = text[1:-1].strip()
    while text and text[-1] in {",", "，", ";", "；", "、"}:
        text = text[:-1].strip()
    return text


def normalize_target_mode(value: Any) -> str:
    """Return a canonical target mode, or an empty string when unsupported."""

    text = normalize_scalar(value).casefold().replace("_", "-")
    return {
        "fake": "fake-node",
        "fake-node": "fake-node",
        "fakenode": "fake-node",
        "real": "real-node",
        "real-node": "real-node",
        "realnode": "real-node",
        "sync": "sync-observe",
        "sync-observe": "sync-observe",
        "syncobserve": "sync-observe",
    }.get(text, "")


def normalize_observability_mode(value: Any) -> str:
    """Return the canonical observability product mode."""

    text = normalize_scalar(value).casefold().replace("_", "-")
    return {
        "disabled": "disabled",
        "disable": "disabled",
        "off": "disabled",
        "local": "local",
        "prometheus-grafana": "local",
        "local-prometheus-grafana": "local",
        "exporter": "exporter",
        "exporter-only": "exporter",
    }.get(text, "")


_RPC_METHOD_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[._:-][A-Za-z0-9]+)*$")
_REST_METHOD_IDENTITY_RE = re.compile(
    r"^(?:GET|POST|PUT|PATCH|DELETE|HEAD)\s+/\S*$",
    re.IGNORECASE,
)


def looks_like_rpc_method_token(value: Any) -> bool:
    """Return whether a scalar is an exact non-REST wire method token."""

    text = normalize_scalar(value)
    return bool(
        text
        and len(text) <= 128
        and not any(ord(character) < 32 for character in text)
        and _RPC_METHOD_TOKEN_RE.fullmatch(text)
    )


def looks_like_rest_method_identity(value: Any) -> bool:
    """Return whether a scalar is an exact HTTP-verb plus path identity."""

    text = normalize_scalar(value)
    return bool(
        text
        and len(text) <= 256
        and not any(ord(character) < 32 for character in text)
        and _REST_METHOD_IDENTITY_RE.fullmatch(text)
    )


def looks_like_wire_method_identity(value: Any) -> bool:
    """Reject prose placeholders before a method mutation enters the queue."""

    return looks_like_rpc_method_token(value) or looks_like_rest_method_identity(value)


def target_mode_evidence_matches(mode: Any, evidence: Any, origin_text: Any) -> bool:
    """Verify that cited user text explicitly expresses one product mode.

    The evidence must be an exact excerpt from the current user turn. Product
    identifiers are accepted directly; sync-observe also accepts the complete
    semantic concept (sync/catch-up plus observation) in either word order.
    """

    canonical = normalize_target_mode(mode)
    source = normalize_scalar(evidence)
    origin = str(origin_text or "")
    if not canonical or not source or source not in origin:
        return False
    normalized = source.casefold().replace("_", "-")
    normalized_origin = origin.casefold().replace("_", "-")
    identifiers = {
        "fake-node": ("fake-node", "fake node", "fakenode", "模拟节点"),
        "real-node": ("real-node", "real node", "realnode", "真实节点"),
        "sync-observe": ("sync-observe", "sync observe", "syncobserve"),
    }
    mentioned_modes = {
        product_mode
        for product_mode, product_identifiers in identifiers.items()
        if any(identifier in normalized_origin for identifier in product_identifiers)
    }
    if len(mentioned_modes) > 1:
        return False
    if any(identifier in normalized for identifier in identifiers[canonical]):
        return True
    if canonical != "sync-observe":
        return False
    english_tokens = set(re.findall(r"[a-z]+", normalized))
    english_concept = bool(english_tokens & {"sync", "synchronization", "catchup"}) and bool(
        english_tokens & {"observe", "observing", "observation", "monitor", "monitoring"}
    )
    chinese_concept = (
        any(concept in source for concept in ("同步", "追块", "追赶"))
        and any(concept in source for concept in ("观察", "监控", "查看"))
    )
    return english_concept or chinese_concept


def looks_like_url_value(value: Any) -> bool:
    """Return whether a complete scalar is an HTTP, WS, host, or localhost URL."""

    text = str(value or "").strip()
    return bool(
        re.match(r"^(https?|wss?)://\S+$", text, re.IGNORECASE)
        or re.match(
            r"^(localhost|127\.0\.0\.1|\[[0-9a-fA-F:]+\]|[A-Za-z0-9.-]+):[0-9]{2,5}(?:/.*)?$",
            text,
        )
    )


def extract_url_candidate(value: Any) -> str:
    """Extract the first HTTP, WebSocket, host, or localhost endpoint."""

    text = str(value or "").strip()
    if not text:
        return ""
    scalar = text.strip("`'\"").rstrip(".,;，。；")
    if looks_like_url_value(scalar):
        return scalar
    match = re.search(r"\b(?:https?|wss?)://[^\s'\"`，。；;]+", text, flags=re.IGNORECASE)
    if match:
        return match.group(0).rstrip(".,;，。；")
    match = re.search(
        r"\b(?:localhost|127\.0\.0\.1|\[[0-9a-fA-F:]+\]|[A-Za-z0-9.-]+):[0-9]{2,5}(?:/[^\s'\"`，。；;]*)?",
        text,
    )
    return match.group(0).rstrip(".,;，。；") if match else ""


def extract_url_candidates(value: Any) -> tuple[str, ...]:
    """Return every distinct endpoint token found in source order."""

    text = str(value or "")
    matches = re.findall(
        r"\b(?:https?|wss?)://[^\s'\"`，。；;]+"
        r"|\b(?:localhost|127\.0\.0\.1|\[[0-9a-fA-F:]+\]|[A-Za-z0-9.-]+):"
        r"[0-9]{2,5}(?:/[^\s'\"`，。；;]*)?",
        text,
        flags=re.IGNORECASE,
    )
    output: list[str] = []
    for match in matches:
        candidate = match.rstrip(".,;，。；")
        if looks_like_url_value(candidate) and candidate not in output:
            output.append(candidate)
    return tuple(output)


def extract_json_values(value: Any) -> list[Any]:
    """Decode every valid JSON object or array embedded in a value."""

    decoder = json.JSONDecoder()
    output: list[Any] = []
    text = str(value or "")
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            output.append(parsed)
    return output


_RPC_WIRE_KEYS = frozenset({"request", "response", "method", "jsonrpc", "params", "result", "error"})


def extract_rpc_wire_values(value: Any) -> list[Any]:
    """Decode source-exact JSON/YAML wire documents embedded in one turn.

    This parser establishes syntax ownership only. It does not decide whether
    the user wants to configure a method, and it never mutates workflow state.
    """

    text = str(value or "").strip()
    if not text:
        return []
    output = list(extract_json_values(text))
    yaml_candidates: list[str] = [text]
    yaml_candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:ya?ml)?\s*\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if match.group(1).strip()
    )
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line != line.lstrip() or ":" not in line:
            continue
        key = line.split(":", 1)[0].strip().strip("\"'").casefold()
        if key in _RPC_WIRE_KEYS:
            yaml_candidates.append("\n".join(lines[index:]).strip())
            break
    for candidate in yaml_candidates:
        if not candidate or candidate[0] in "[{":
            continue
        try:
            parsed = yaml.safe_load(candidate)
        except yaml.YAMLError:
            continue
        if isinstance(parsed, (dict, list)) and parsed not in output:
            output.append(parsed)
    return output


def _rpc_wire_payloads(value: Any) -> list[Any]:
    payloads: list[Any] = []
    for document in extract_rpc_wire_values(value):
        payloads.append(document)
        if isinstance(document, dict):
            for key in ("request", "response"):
                nested = document.get(key)
                if isinstance(nested, (dict, list)):
                    payloads.append(nested)
    return payloads


def extract_rpc_method_identities(value: Any) -> list[str]:
    """Return source-grounded wire method identities from RPC documents.

    This is syntax extraction only.  It deliberately does not decide whether
    an example should be selected, rejected, or treated as documentation; the
    active Harness contract and semantic admission layer own that decision.
    """

    methods: list[str] = []
    for payload in _rpc_wire_payloads(value):
        if not isinstance(payload, dict):
            continue
        method = next(
            (
                item
                for key, item in payload.items()
                if str(key).casefold() == "method"
            ),
            None,
        )
        normalized = normalize_scalar(method)
        if looks_like_wire_method_identity(normalized) and normalized not in methods:
            methods.append(normalized)
    return methods


def extract_rpc_method_token_candidates(value: Any) -> list[str]:
    """Return syntax-only method candidates embedded in prose or wire data.

    Exact JSON-RPC documents remain the strongest source. For prose, expose
    only tokens with wire-like separators (or an HTTP verb/path identity), so
    ordinary words are not promoted to method identities. Semantic admission
    still decides whether the user selected any candidate.
    """

    methods = extract_rpc_method_identities(value)
    text = str(value or "")
    for verb, path in re.findall(
        r"(?i)\b(GET|POST|PUT|PATCH|DELETE|HEAD)\s+(/\S+)",
        text,
    ):
        candidate = normalize_scalar(f"{verb.upper()} {path}")
        if looks_like_rest_method_identity(candidate) and candidate not in methods:
            methods.append(candidate)
    separated = re.findall(
        r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]*(?:[._:][A-Za-z0-9]+)+)(?![A-Za-z0-9])",
        text,
    )
    camel_case = [
        token
        for token in re.findall(r"(?<![A-Za-z0-9])([a-z][A-Za-z0-9]{2,127})(?![A-Za-z0-9])", text)
        if re.search(r"[a-z][A-Z]", token)
    ]
    for token in (*separated, *camel_case):
        candidate = normalize_scalar(token)
        if looks_like_rpc_method_token(candidate) and candidate not in methods:
            methods.append(candidate)
    return methods


def has_rpc_wire_evidence(value: Any, *, allow_params_only: bool = False) -> bool:
    """Return whether one turn contains an attributable RPC wire fact."""

    for payload in _rpc_wire_payloads(value):
        if isinstance(payload, list):
            if allow_params_only:
                return True
            continue
        if not isinstance(payload, dict):
            continue
        keys = {str(key).casefold() for key in payload}
        if {"method", "params"}.issubset(keys) or "result" in keys or "error" in keys:
            return True
        if allow_params_only and keys and keys.issubset({"params", "arguments", "args"}):
            params = next(iter(payload.values()))
            if isinstance(params, (list, dict)):
                return True
    return False


def has_rpc_response_evidence(value: Any) -> bool:
    """Return whether one turn contains a response object, not only a request."""

    return any(
        isinstance(payload, dict)
        and any(str(key).casefold() in {"result", "error"} for key in payload)
        and not any(str(key).casefold() == "method" for key in payload)
        for payload in _rpc_wire_payloads(value)
    )


def extract_json_object_or_array(value: Any) -> str:
    """Return the first outer JSON object/array substring in user input."""

    text = str(value or "").strip()
    if not text:
        return ""
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start < 0 or end <= start:
            continue
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return candidate
    return ""


def declares_no_rpc_params(value: Any) -> bool:
    text = str(value or "").strip().casefold()
    return bool(text) and any(
        marker in text
        for marker in (
            "no parameters",
            "no params",
            "without parameters",
            "without params",
            "params: none",
            "params none",
            "没有参数",
            "无参数",
            "不需要参数",
            "参数为空",
        )
    )


def extract_rpc_params_or_request(value: Any) -> tuple[str, Any | None]:
    """Extract a JSON-RPC method and params from JSON/YAML request text."""

    text = str(value or "").strip()
    if not text:
        return "", None
    if declares_no_rpc_params(text):
        return "", []
    for parsed in _rpc_wire_payloads(text):
        if isinstance(parsed, dict) and "params" in parsed and (
            "method" in parsed or "jsonrpc" in parsed
        ):
            params = parsed.get("params")
            return normalize_scalar(parsed.get("method")), params if isinstance(params, (list, dict)) else None
        if isinstance(parsed, list):
            return "", parsed
        if isinstance(parsed, dict) and set(parsed).issubset({"params", "arguments", "args"}):
            params = parsed.get("params", parsed.get("arguments", parsed.get("args")))
            return "", params if isinstance(params, (list, dict)) else None
    return "", None


def is_direct_json_rpc_input(value: Any) -> bool:
    """Return whether the complete answer is a JSON object or array."""

    text = str(value or "").strip()
    if not text or text[0] not in "[{":
        return False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, (dict, list))


def schema_evidence_from_turn_text(value: Any, *, method_hint: str = "") -> str:
    """Recover direct request/schema evidence from the original action turn."""

    text = str(value or "").strip()
    if not text:
        return ""
    parsed_method, parsed = extract_rpc_params_or_request(text)
    if parsed is not None:
        if parsed_method or not method_hint:
            return text
        return json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method_hint, "params": parsed},
            ensure_ascii=False,
        )
    if declares_no_rpc_params(text) and method_hint:
        return json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method_hint, "params": []},
            ensure_ascii=False,
        )
    return ""


def parse_weight_spec(value: Any) -> dict[str, int]:
    """Parse the supported structured RPC-weight forms without policy checks.

    This is the shared syntax authority for flat JSON, a JSON ``weights``
    wrapper, YAML-style mappings, and ``method=weight`` assignments. Method
    membership and total-weight rules remain with the workload domain.
    """

    direct_value = _weight_mapping(value)
    if direct_value is not None:
        return direct_value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    direct = _weight_mapping(parsed)
    if direct is not None:
        return direct

    embedded: list[dict[str, int]] = []
    for candidate in extract_json_values(text):
        weights = _weight_mapping(candidate)
        if weights is not None and weights not in embedded:
            embedded.append(weights)
    if len(embedded) == 1:
        return embedded[0]
    if len(embedded) > 1:
        return {}
    output = {}
    pairs = re.findall(
        r"(?<![A-Za-z0-9_./:-])([A-Za-z][A-Za-z0-9_./:-]*)\s*(?:=|:)\s*([+-]?[0-9]+)\s*(?=$|[,，\n\r])",
        text,
    )
    if pairs:
        for method, weight in pairs:
            output[method.strip()] = int(weight)
        return output
    for chunk in re.split(r"[,，]\s*", text):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            return {}
        method, weight = chunk.split("=", 1)
        method = method.strip()
        if not method:
            return {}
        try:
            output[method] = int(weight.strip())
        except ValueError:
            return {}
    return output


def _weight_mapping(value: Any) -> dict[str, int] | None:
    """Normalize one complete JSON weight mapping without applying policy."""

    if not isinstance(value, dict):
        return None
    if set(value) == {"weights"} and isinstance(value.get("weights"), dict):
        value = value["weights"]
    if not value:
        return None
    output: dict[str, int] = {}
    for key, item in value.items():
        method = str(key).strip()
        if not method or isinstance(item, bool):
            return None
        if isinstance(item, int):
            output[method] = item
            continue
        item_text = str(item).strip()
        if not re.fullmatch(r"[+-]?[0-9]+", item_text):
            return None
        output[method] = int(item_text)
    return output


def single_method_weight_number_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
        return text
    match = re.search(
        r"(?:weight|权重)\s*(?:is|为|=|:|：)?\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    return match.group(1) if match else ""


def parse_weight_spec_for_methods(value: Any, methods: list[str]) -> dict[str, int]:
    """Also accept an inline numeric weight when exactly one method is available."""

    weights = parse_weight_spec(value)
    if weights:
        return weights
    unique = [method for method in dict.fromkeys(normalize_scalar(item) for item in methods) if method]
    if len(unique) != 1:
        return {}
    number = single_method_weight_number_text(value)
    if not number:
        return {}
    try:
        return {unique[0]: int(float(number))}
    except ValueError:
        return {}


_ADAPTER_FAMILY_NEGATION_RE = re.compile(
    r"(没有|没|无|不是|不用|非|not|no|without|non[- ]?)"
    r"(?:[\s\-_,，、/|]|也|且|and|or|json[-_ ]?rpc|jsonrpc|evm|rest|substrate|polkadot|"
    r"tendermint|cosmos(?:[ -]?sdk)?|cometbft|bitcoin(?:[_ ]?jsonrpc)?|hedera|"
    r"ethereum[-_ ]?compatible|api|http)*$"
)


def adapter_family_hint(value: Any) -> str:
    """Normalize an explicit adapter-family statement without ignoring negation."""

    text = normalize_scalar(value).casefold()
    if not text:
        return ""

    def mentioned(pattern: str) -> bool:
        match = re.search(pattern, text)
        return bool(match) and not _ADAPTER_FAMILY_NEGATION_RE.search(text[: match.start()])

    if mentioned(r"\b(evm|json[-_ ]?rpc|ethereum[-_ ]?compatible|eth_[a-z0-9_]+)\b"):
        return "jsonrpc"
    if mentioned(r"substrate|polkadot"):
        return "substrate"
    if mentioned(r"\b(rest|http api|rest api)\b"):
        return "rest"
    if mentioned(r"tendermint|cosmos sdk|cometbft"):
        return "tendermint"
    if mentioned(r"bitcoin[_ ]jsonrpc|bitcoin json-rpc"):
        return "bitcoin_jsonrpc"
    if mentioned(r"hedera"):
        return "hedera_dual"
    return ""
