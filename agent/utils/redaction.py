"""Secret redaction helpers for Agent output."""

from __future__ import annotations

import re
from typing import Any


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
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|PASSWORD|TOKEN|SECRET|AUTHORIZATION)[A-Z0-9_]*\s*[:=]\s*)(['\"]?)[^'\"\s,}]+(\2)"
)


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
        redacted = BEARER_RE.sub(r"\1***REDACTED***", redacted)
        redacted = SECRET_ASSIGNMENT_RE.sub(r"\1\2***REDACTED***\3", redacted)
        return redacted
    return value


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(secret in normalized for secret in SECRET_KEYS)
