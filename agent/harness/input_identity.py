"""Canonical user-input identity shared across product and evidence boundaries."""

from __future__ import annotations

import hashlib


def canonical_user_input(value: str) -> str:
    return str(value).strip()


def user_input_hash(value: str) -> str:
    return hashlib.sha256(canonical_user_input(value).encode("utf-8")).hexdigest()
