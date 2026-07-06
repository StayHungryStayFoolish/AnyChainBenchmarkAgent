"""Secret redaction helpers for Agent output."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


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


URL_CREDENTIAL_RE = re.compile(r"(https?://)([^/@:\s]+):([^/@\s]+)@")
BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Za-z0-9_-]*(?:API[_-]?KEY|PASSWORD|TOKEN|SECRET|AUTHORIZATION)[A-Za-z0-9_-]*\s*[:=]\s*)(['\"]?)[^'\"\s,}]+(\2)"
)
TOKENISH_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_-]{24,}$")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "***REDACTED***" if _is_secret_key(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        redacted = URL_CREDENTIAL_RE.sub(r"\1***:***@", value)
        redacted = _redact_url_path_tokens(redacted)
        redacted = BEARER_RE.sub(r"\1***REDACTED***", redacted)
        redacted = SECRET_ASSIGNMENT_RE.sub(_redact_secret_assignment, redacted)
        return redacted
    return value


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(secret in normalized for secret in SECRET_KEYS)


def _redact_url_path_tokens(value: str) -> str:
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
    return urlunsplit((parsed.scheme, parsed.netloc, "/".join(path_parts), parsed.query, parsed.fragment))


def _redact_secret_assignment(match: re.Match[str]) -> str:
    prefix = match.group(1)
    quote = match.group(2)
    key = prefix.split(":", 1)[0].split("=", 1)[0].strip()
    if not _looks_like_secret_assignment_key(key):
        return match.group(0)
    return f"{prefix}{quote}***REDACTED***{quote}"


def _looks_like_secret_assignment_key(key: str) -> bool:
    normalized = key.strip()
    if not normalized:
        return False
    upper = normalized.upper()
    if normalized != upper and "_" not in normalized and "-" not in normalized:
        return False
    return _is_secret_key(upper)
