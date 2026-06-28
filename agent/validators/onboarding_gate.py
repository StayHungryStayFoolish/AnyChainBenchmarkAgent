"""Onboarding validation and handoff helpers."""

from __future__ import annotations

from typing import Any

from onboarding.chain_onboarding import generate_onboarding_package


def build_onboarding_handoff(
    chain: str,
    family: str,
    methods: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an evidence-aware onboarding handoff for ADK agents."""
    package = generate_onboarding_package(chain, methods=methods or [], adapter_family=family)
    evidence = evidence or {}
    missing_evidence = []
    for key in ("docs_url", "request_samples", "response_samples", "sync_health_method"):
        if not evidence.get(key):
            missing_evidence.append(key)
    return {
        "package": package,
        "evidence": evidence,
        "missing_evidence": missing_evidence,
        "ready_for_coding": not missing_evidence and bool(methods),
        "docs_must_update": [
            "docs/en/how-to-add-chain.md",
            "docs/zh/how-to-add-chain.md",
            "docs/en/secondary-development-guide.md",
            "docs/zh/secondary-development-guide.md",
        ],
    }
