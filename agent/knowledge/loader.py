"""Knowledge provider loader."""

from __future__ import annotations

import importlib
from typing import Any, Mapping

from llm.config import load_agent_environment
from knowledge.base import KnowledgeProvider, NoopKnowledgeProvider
from knowledge.http_provider import HTTPKnowledgeProvider


def load_knowledge_provider(env: Mapping[str, str] | None = None) -> KnowledgeProvider:
    source = env or load_agent_environment()
    provider = source.get("AGENT_KNOWLEDGE_PROVIDER", "disabled").strip().lower()
    if provider in {"", "disabled", "noop"}:
        return NoopKnowledgeProvider()
    if provider == "http":
        base_url = source.get("AGENT_KNOWLEDGE_BASE_URL", "").strip()
        if not base_url:
            raise ValueError("AGENT_KNOWLEDGE_BASE_URL is required when AGENT_KNOWLEDGE_PROVIDER=http")
        return HTTPKnowledgeProvider(base_url, auth_ref=source.get("AGENT_KNOWLEDGE_AUTH_REF", "").strip())
    if provider == "custom":
        return _load_custom_provider(source.get("AGENT_KNOWLEDGE_PROVIDER_MODULE", "").strip())
    raise ValueError(f"unsupported AGENT_KNOWLEDGE_PROVIDER: {provider}")


def provider_status(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    source = env or load_agent_environment()
    provider = source.get("AGENT_KNOWLEDGE_PROVIDER", "disabled").strip().lower()
    try:
        loaded = load_knowledge_provider(source)
        return {
            "provider": provider,
            "enabled": provider not in {"", "disabled", "noop"},
            "capabilities": loaded.capabilities(),
            "error": "",
        }
    except Exception as exc:
        return {
            "provider": provider,
            "enabled": False,
            "capabilities": NoopKnowledgeProvider().capabilities(),
            "error": str(exc),
        }


def _load_custom_provider(module_path: str) -> KnowledgeProvider:
    if ":" not in module_path:
        raise ValueError("AGENT_KNOWLEDGE_PROVIDER_MODULE must be module.path:Factory")
    module_name, factory_name = module_path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    provider = factory()
    return provider
