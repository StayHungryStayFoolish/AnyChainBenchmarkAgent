"""Shared chain identity helpers for terminal and workflow gates.

These helpers only normalize exact, low-risk chain identifiers. They must not
turn a brand prefix into a supported chain when the user typed a longer chain
name such as ``bnb greenfield`` or ``ethereum classic``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


_CANONICAL_ALIASES = {
    "eth": "ethereum",
    "ethereum": "ethereum",
    "btc": "bitcoin",
    "bitcoin": "bitcoin",
    "bnb": "bsc",
    "bsc": "bsc",
    "bnb-chain": "bsc",
    "bnb chain": "bsc",
    "bnb-smart-chain": "bsc",
    "bnb smart chain": "bsc",
    "binance-smart-chain": "bsc",
    "binance smart chain": "bsc",
}

def canonical_chain_aliases(extra: dict[str, Any] | None = None) -> dict[str, str]:
    aliases = dict(_CANONICAL_ALIASES)
    if isinstance(extra, dict):
        for key, value in extra.items():
            key_text = _normalize_scalar(key)
            value_text = _normalize_scalar(value)
            if key_text and value_text:
                aliases[key_text] = value_text
    return aliases


def canonicalize_chain_scalar(
    value: Any,
    *,
    known_chains: set[str] | list[str] | None = None,
    aliases: dict[str, Any] | None = None,
) -> str:
    """Return a supported canonical chain for an exact scalar answer.

    ``bnb`` may become ``bsc`` and ``btc`` may become ``bitcoin``. Longer input
    such as ``bnb greenfield`` is not a scalar alias and returns ``""`` unless
    it is itself a known chain template.
    """

    text = _normalize_scalar(value)
    if not text:
        return ""
    known = _normalize_known_chains(known_chains)
    chain_aliases = canonical_chain_aliases(aliases)
    canonical = chain_aliases.get(text, text)
    if not known or canonical in known:
        return canonical
    return ""


def repo_chain_names(repo_root: str | Path | None = None) -> list[str]:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    chains_dir = root / "config" / "chains"
    if not chains_dir.is_dir():
        return []
    return sorted(path.stem.lower() for path in chains_dir.glob("*.json"))


def _normalize_known_chains(values: set[str] | list[str] | None) -> set[str]:
    return {_normalize_scalar(item) for item in (values or []) if _normalize_scalar(item)}


def _normalize_scalar(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    while text and text[-1] in {",", "，", ";", "；"}:
        text = text[:-1].strip()
    return " ".join(text.split())
