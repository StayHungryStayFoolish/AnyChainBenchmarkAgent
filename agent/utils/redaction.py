"""Secret redaction helpers for Agent output."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SECRET_KEYS = {
    "rpc_api_key",
    "api_key",
    "authorization",
    "bearer",
    "password",
    "token",
    "access_token",
    "refresh_token",
}


URL_CREDENTIAL_RE = re.compile(r"((?:https?|wss?)://)([^/@:\s]+):([^/@\s]+)@")
INLINE_URL_RE = re.compile(r"(?:https?|wss?)://[^\s'\"<>]+")
BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
AUTHORIZATION_RE = re.compile(
    r"(?i)(authorization\s*:\s*['\"]?(?:basic|bearer)\s+)"
    r"([A-Za-z0-9._~+/=-]+)"
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Za-z0-9_-]*(?:API[_-]?KEY|PASSWORD|TOKEN|SECRET|AUTHORIZATION)[A-Za-z0-9_-]*\s*[:=]\s*)(['\"]?)[^'\"\s,}]+(\2)"
)
QUOTED_SECRET_ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    (?P<key_quote>["'])
    (?P<key>[A-Za-z0-9_-]*(?:API[_-]?KEY|PASSWORD|TOKEN|SECRET|AUTHORIZATION)[A-Za-z0-9_-]*)
    (?P=key_quote)
    \s*:\s*
    (?P<value_quote>["'])
    (?P<value>[^"']+)
    (?P=value_quote)
    """
)
TOKENISH_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_-]{24,}$")
URL_TRAILING_PUNCTUATION = ".,;:!?)]}，。；：！？）】"


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "***REDACTED***" if _is_secret_key(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, str):
        redacted = URL_CREDENTIAL_RE.sub(r"\1***:***@", value)
        redacted = INLINE_URL_RE.sub(
            lambda match: _redact_url_path_tokens(match.group(0)),
            redacted,
        )
        redacted = BEARER_RE.sub(r"\1***REDACTED***", redacted)
        redacted = AUTHORIZATION_RE.sub(r"\1***REDACTED***", redacted)
        redacted = QUOTED_SECRET_ASSIGNMENT_RE.sub(
            _redact_quoted_secret_assignment,
            redacted,
        )
        redacted = SECRET_ASSIGNMENT_RE.sub(_redact_secret_assignment, redacted)
        return redacted
    return value


def secret_values(value: str) -> tuple[str, ...]:
    """Return exact secret substrings recognized by the redaction policy."""

    text = str(value)
    found: list[str] = []
    try:
        structured = json.loads(text)
    except (TypeError, ValueError):
        structured = None

    def collect_structured(item: Any, *, secret_context: bool = False) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                collect_structured(
                    child,
                    secret_context=_is_secret_key(str(key)),
                )
        elif isinstance(item, list):
            for child in item:
                collect_structured(child, secret_context=secret_context)
        elif secret_context and isinstance(item, (str, int, float)):
            secret = str(item)
            if secret:
                found.append(secret)

    if structured is not None:
        collect_structured(structured)
    for match in URL_CREDENTIAL_RE.finditer(text):
        found.extend((match.group(2), match.group(3)))
    for match in INLINE_URL_RE.finditer(text):
        raw_url = match.group(0).rstrip(URL_TRAILING_PUNCTUATION)
        try:
            parsed = urlsplit(raw_url)
        except ValueError:
            continue
        found.extend(
            part
            for part in parsed.path.split("/")
            if TOKENISH_PATH_SEGMENT_RE.match(part)
        )
        found.extend(
            item
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if _is_secret_key(key) or TOKENISH_PATH_SEGMENT_RE.match(item)
        )
        if TOKENISH_PATH_SEGMENT_RE.match(parsed.fragment):
            found.append(parsed.fragment)
    for match in BEARER_RE.finditer(text):
        token = match.group(0)[len(match.group(1)):]
        if token:
            found.append(token)
    for match in AUTHORIZATION_RE.finditer(text):
        token = match.group(2)
        if token:
            found.append(token)
    for match in QUOTED_SECRET_ASSIGNMENT_RE.finditer(text):
        token = match.group("value")
        if token and _is_secret_key(match.group("key")):
            found.append(token)
    for match in SECRET_ASSIGNMENT_RE.finditer(text):
        raw = match.group(0)[len(match.group(1)):].strip()
        quote = match.group(2)
        if quote and raw.startswith(quote) and raw.endswith(quote):
            raw = raw[1:-1]
        key = match.group(1).split(":", 1)[0].split("=", 1)[0].strip()
        if not _looks_like_secret_assignment_key(key):
            continue
        if _is_secret_key(key) and raw.casefold() in {"basic", "bearer"}:
            continue
        if raw:
            found.append(raw)
    return tuple(dict.fromkeys(item for item in found if item))


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(
        normalized == secret
        or normalized.endswith(f"_{secret}")
        or normalized.startswith(f"{secret}_")
        for secret in SECRET_KEYS
    )


def _redact_url_path_tokens(value: str) -> str:
    suffix = ""
    while value and value[-1] in URL_TRAILING_PUNCTUATION:
        suffix = value[-1] + suffix
        value = value[:-1]
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.netloc:
        return value
    path_parts = [
        "***REDACTED***" if TOKENISH_PATH_SEGMENT_RE.match(part) else part
        for part in parsed.path.split("/")
    ]
    query = urlencode([
        (
            key,
            "***REDACTED***"
            if _is_secret_key(key) or TOKENISH_PATH_SEGMENT_RE.match(item)
            else item,
        )
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
    ])
    fragment = (
        "***REDACTED***"
        if TOKENISH_PATH_SEGMENT_RE.match(parsed.fragment)
        else parsed.fragment
    )
    return urlunsplit(
        (parsed.scheme, parsed.netloc, "/".join(path_parts), query, fragment)
    ) + suffix


def _redact_secret_assignment(match: re.Match[str]) -> str:
    prefix = match.group(1)
    quote = match.group(2)
    key = prefix.split(":", 1)[0].split("=", 1)[0].strip()
    if not _looks_like_secret_assignment_key(key):
        return match.group(0)
    return f"{prefix}{quote}***REDACTED***{quote}"


def _redact_quoted_secret_assignment(match: re.Match[str]) -> str:
    if not _is_secret_key(match.group("key")):
        return match.group(0)
    return (
        f"{match.group('key_quote')}{match.group('key')}"
        f"{match.group('key_quote')}: "
        f"{match.group('value_quote')}***REDACTED***"
        f"{match.group('value_quote')}"
    )


def _looks_like_secret_assignment_key(key: str) -> bool:
    normalized = key.strip()
    if not normalized:
        return False
    upper = normalized.upper()
    if normalized != upper and "_" not in normalized and "-" not in normalized:
        return False
    return _is_secret_key(upper)
