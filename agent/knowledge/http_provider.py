"""Generic HTTP knowledge-base adapter for enterprise integrations."""

from __future__ import annotations

import json
from typing import Any
from urllib import parse as urlparse
from urllib import request as urlrequest

from knowledge.base import NoopKnowledgeProvider


class HTTPKnowledgeProvider(NoopKnowledgeProvider):
    """Minimal HTTP adapter with a stable AnyChain KB contract."""

    def __init__(self, base_url: str, auth_ref: str = "", timeout_seconds: int = 30):
        self.base_url = base_url.rstrip("/")
        self.auth_ref = auth_ref
        self.timeout_seconds = timeout_seconds

    def capabilities(self) -> dict[str, bool]:
        return {
            "search": True,
            "chain_identification": True,
            "rpc_methods": True,
            "rpc_samples": True,
            "fixtures": True,
            "workload_suggestions": True,
            "chain_template_suggestions": True,
            "sync_health_methods": True,
            "known_bottlenecks": True,
            "runtime_notes": True,
        }

    def search(self, query: str) -> list[dict]:
        payload = self._post("/search", {"query": query})
        return _list(payload, "results")

    def identify_chain(self, chain_hint: str, endpoint: str | None = None) -> dict:
        return self._post("/chains/identify", {"chain_hint": chain_hint, "endpoint": endpoint})

    def get_rpc_methods(self, chain: str) -> list[dict]:
        return _list(self._get(f"/chains/{_quote(chain)}/rpc-methods"), "methods")

    def get_rpc_samples(self, chain: str) -> list[dict]:
        return _list(self._get(f"/chains/{_quote(chain)}/rpc-samples"), "samples")

    def get_method_param_samples(self, chain: str, method: str) -> list[dict]:
        return _list(self._get(f"/chains/{_quote(chain)}/rpc-methods/{_quote(method)}/param-samples"), "samples")

    def get_response_fixtures(self, chain: str, method: str) -> list[dict]:
        return _list(self._get(f"/chains/{_quote(chain)}/rpc-methods/{_quote(method)}/fixtures"), "fixtures")

    def suggest_workload(self, chain: str, goal: str, rpc_mode: str = "mixed") -> dict:
        return self._post("/workload/suggest", {"chain": chain, "goal": goal, "rpc_mode": rpc_mode})

    def suggest_chain_template(self, chain: str, adapter_family: str | None = None) -> dict:
        return self._post("/chain-template/suggest", {"chain": chain, "adapter_family": adapter_family})

    def get_sync_health_methods(self, chain: str) -> dict:
        return self._get(f"/chains/{_quote(chain)}/sync-health")

    def get_known_bottlenecks(self, chain: str) -> list[dict]:
        return _list(self._get(f"/chains/{_quote(chain)}/bottlenecks"), "bottlenecks")

    def get_chain_runtime_notes(self, chain: str) -> dict:
        return self._get(f"/chains/{_quote(chain)}/runtime-notes")

    def _get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, payload)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urlrequest.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers=self._headers(),
        )
        with urlrequest.urlopen(req, timeout=self.timeout_seconds) as response:  # nosec B310 - user-configured enterprise KB endpoint
            text = response.read().decode("utf-8")
        return json.loads(text) if text else {}

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.auth_ref:
            headers["Authorization"] = f"Bearer {self.auth_ref}"
        return headers


def _quote(value: str) -> str:
    return urlparse.quote(value, safe="")


def _list(payload: dict[str, Any], key: str) -> list[dict]:
    value = payload.get(key, [])
    return value if isinstance(value, list) else []
