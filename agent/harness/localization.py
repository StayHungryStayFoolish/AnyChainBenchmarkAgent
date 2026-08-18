"""Minimal locale selection for model-independent product text."""

from __future__ import annotations


def localized(language: str, zh: str, en: str) -> str:
    return zh if str(language or "").startswith("zh") else en
