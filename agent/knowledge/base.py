"""Knowledge provider contract reserved for Phase 3 integrations.

The Agent must be able to run without a knowledge base. When an enterprise
knowledge base is available, providers implement this contract to reduce manual
QA loops without changing the benchmark execution engine.
"""

from __future__ import annotations

from typing import Any, Protocol


class KnowledgeProvider(Protocol):
    def capabilities(self) -> dict[str, bool]:
        """Return which optional knowledge features this provider supports."""

    def search(self, query: str) -> list[dict]:
        """Return knowledge snippets relevant to a user request."""

    def identify_chain(self, chain_hint: str, endpoint: str | None = None) -> dict:
        """Return known chain metadata and likely adapter family.

        Expected keys may include chain, adapter_family, protocol, confidence,
        docs_url, and notes.
        """

    def get_rpc_methods(self, chain: str) -> list[dict]:
        """Return known RPC methods and workload hints for a chain."""

    def get_rpc_samples(self, chain: str) -> list[dict]:
        """Return real request samples that can inform workloads or fixtures."""

    def get_method_param_samples(self, chain: str, method: str) -> list[dict]:
        """Return safe sample params for one RPC method."""

    def get_response_fixtures(self, chain: str, method: str) -> list[dict]:
        """Return known request/response fixtures for fake-node onboarding."""

    def suggest_workload(self, chain: str, goal: str, rpc_mode: str = "mixed") -> dict:
        """Return suggested single/mixed workload methods and weights."""

    def suggest_chain_template(self, chain: str, adapter_family: str | None = None) -> dict:
        """Return a chain template fragment for unsupported-chain onboarding."""

    def get_sync_health_methods(self, chain: str) -> dict:
        """Return local height, target height, lag, or health RPC hints."""

    def get_known_bottlenecks(self, chain: str) -> list[dict]:
        """Return prior bottleneck findings for a chain or deployment profile."""

    def get_chain_runtime_notes(self, chain: str) -> dict:
        """Return chain-specific deployment and monitoring notes."""


class NoopKnowledgeProvider:
    def capabilities(self) -> dict[str, bool]:
        return {
            "search": False,
            "chain_identification": False,
            "rpc_methods": False,
            "rpc_samples": False,
            "fixtures": False,
            "workload_suggestions": False,
            "chain_template_suggestions": False,
            "sync_health_methods": False,
            "known_bottlenecks": False,
            "runtime_notes": False,
        }

    def search(self, query: str) -> list[dict]:
        return []

    def identify_chain(self, chain_hint: str, endpoint: str | None = None) -> dict:
        return {"chain": chain_hint, "confidence": 0.0, "source": "noop"}

    def get_rpc_methods(self, chain: str) -> list[dict]:
        return []

    def get_rpc_samples(self, chain: str) -> list[dict]:
        return []

    def get_method_param_samples(self, chain: str, method: str) -> list[dict]:
        return []

    def get_response_fixtures(self, chain: str, method: str) -> list[dict]:
        return []

    def suggest_workload(self, chain: str, goal: str, rpc_mode: str = "mixed") -> dict:
        return {"rpc_mode": rpc_mode, "methods": [], "source": "noop"}

    def suggest_chain_template(self, chain: str, adapter_family: str | None = None) -> dict:
        return {"chain": chain, "adapter_family": adapter_family, "source": "noop", "template": {}}

    def get_sync_health_methods(self, chain: str) -> dict:
        return {}

    def get_known_bottlenecks(self, chain: str) -> list[dict]:
        return []

    def get_chain_runtime_notes(self, chain: str) -> dict:
        return {}
