"""Shared chain identity helpers for terminal and workflow gates.

These helpers only normalize exact, low-risk chain identifiers. They must not
turn a brand prefix into a supported chain when the user typed a longer chain
name such as ``bnb greenfield`` or ``ethereum classic``.
"""

from __future__ import annotations

import re
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

FRAMEWORK_CONTEXT_TOKENS = frozenset(
    {
        "benchmark",
        "test",
        "quick",
        "standard",
        "intensive",
        "single",
        "mixed",
        "rpc",
        "method",
        "methods",
        "weight",
        "weights",
        "workload",
        "fake",
        "fake-node",
        "fakenode",
        "mock",
        "real",
        "real-node",
        "realnode",
    }
)


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


def extract_supported_chain_from_text(
    text: str,
    *,
    known_chains: set[str] | list[str],
    aliases: dict[str, Any] | None = None,
) -> str:
    """Extract only exact supported chain scalar text.

    Natural-language phrases such as ``test bnb fake-node`` or
    ``bnb greenfield`` are not parsed here. They must go through ADK/LLM and
    typed workflow tools so unsupported-chain onboarding, clarification, and
    endpoint gates remain in control.
    """

    lowered = str(text or "").strip().lower()
    if not lowered:
        return ""
    known = _normalize_known_chains(known_chains)
    if not known:
        return ""

    return canonicalize_chain_scalar(lowered, known_chains=known, aliases=aliases)


def repo_chain_names(repo_root: str | Path | None = None) -> list[str]:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    chains_dir = root / "config" / "chains"
    if not chains_dir.is_dir():
        return []
    return sorted(path.stem.lower() for path in chains_dir.glob("*.json"))


def framework_chain_names(framework_summary: dict[str, Any]) -> list[str]:
    names = []
    for item in list((framework_summary or {}).get("chains") or []):
        if isinstance(item, dict):
            name = _normalize_scalar(item.get("chain"))
            if name:
                names.append(name)
    return sorted(set(names))


def is_full_new_chain_name_candidate(value: str) -> bool:
    """Return true when text is plausible as a full unsupported chain name.

    Short partial tokens such as ``sola`` should stay in chain selection for
    clarification. Multi-token names such as ``bnb greenfield`` are specific
    enough to enter unsupported-chain onboarding, where endpoint/RPC evidence
    is required before any fixture or smoke claim.
    """
    text = str(value or "").strip()
    if not text or text.isdigit():
        return False
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if len(tokens) >= 2:
        return True
    if re.search(r"[\u3400-\u9fff]", text) and len(text) >= 3:
        return True
    return False


def _normalize_known_chains(values: set[str] | list[str] | None) -> set[str]:
    return {_normalize_scalar(item) for item in (values or []) if _normalize_scalar(item)}


def _normalize_scalar(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    while text and text[-1] in {",", "，", ";", "；"}:
        text = text[:-1].strip()
    return " ".join(text.split())
