"""Google ADK feature compatibility detection.

This module is intentionally import-safe in environments without google-adk or
model credentials. It checks which ADK runtime capabilities are importable so
the Agent can decide whether to use ADK workflow agents, session services, and
evaluation APIs instead of guessing from docs.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureProbe:
    name: str
    import_path: str
    available: bool
    symbol: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "import_path": self.import_path,
            "symbol": self.symbol,
            "available": self.available,
            "error": self.error,
        }


FEATURE_CANDIDATES = {
    "llm_agent": [
        ("google.adk.agents", "LlmAgent"),
        ("google.adk.agents", "Agent"),
    ],
    "base_agent": [
        ("google.adk.agents", "BaseAgent"),
    ],
    "sequential_agent": [
        ("google.adk.agents", "SequentialAgent"),
    ],
    "parallel_agent": [
        ("google.adk.agents", "ParallelAgent"),
    ],
    "loop_agent": [
        ("google.adk.agents", "LoopAgent"),
    ],
    "workflow": [
        ("google.adk.workflow", "Workflow"),
        ("google.adk.workflow", "FunctionNode"),
        ("google.adk.workflow", "JoinNode"),
    ],
    "runner": [
        ("google.adk.runners", "Runner"),
    ],
    "in_memory_session_service": [
        ("google.adk.sessions", "InMemorySessionService"),
    ],
    "database_session_service": [
        ("google.adk.sessions", "DatabaseSessionService"),
    ],
    "evaluator": [
        ("google.adk.evaluation", "AgentEvaluator"),
        ("google.adk.evaluation.agent_evaluator", "AgentEvaluator"),
    ],
    "function_tool": [
        ("google.adk.tools", "FunctionTool"),
    ],
}


def adk_feature_report() -> dict[str, Any]:
    """Return an offline-safe ADK capability report."""
    package_available, package_error = _can_import("google.adk")
    version = _version()
    probes = {
        name: _probe_any(name, candidates).as_dict()
        for name, candidates in FEATURE_CANDIDATES.items()
    }
    return {
        "python": {
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "supported_for_adk": sys.version_info >= (3, 10),
        },
        "package": {
            "import_name": "google.adk",
            "available": package_available,
            "version": version,
            "error": package_error,
        },
        "features": probes,
        "implementation_recommendation": _recommendation(package_available, probes),
    }


def _probe_any(name: str, candidates: list[tuple[str, str]]) -> FeatureProbe:
    errors = []
    for module_name, symbol in candidates:
        available, error = _has_symbol(module_name, symbol)
        if available:
            return FeatureProbe(name=name, import_path=module_name, symbol=symbol, available=True)
        errors.append(f"{module_name}.{symbol}: {error}")
    first_module, first_symbol = candidates[0]
    return FeatureProbe(
        name=name,
        import_path=first_module,
        symbol=first_symbol,
        available=False,
        error="; ".join(errors),
    )


def _has_symbol(module_name: str, symbol: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return False, str(exc)
    try:
        getattr(module, symbol)
        return True, ""
    except AttributeError:
        return False, "symbol not found"
    except Exception as exc:
        return False, str(exc)


def _can_import(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _version() -> str:
    for package in ("google-adk", "google_adk"):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return ""


def _recommendation(package_available: bool, probes: dict[str, dict[str, Any]]) -> list[str]:
    if not package_available:
        return [
            "Install google-adk in an isolated Python 3.10+ environment before enabling ADK-native workflows.",
            "Continue running offline deterministic Agent tests until ADK is available.",
        ]
    actions = []
    if probes.get("workflow", {}).get("available"):
        actions.append("Use ADK-native workflow agents for the benchmark orchestration layer.")
    else:
        actions.append("ADK Workflow API was not detected; use Sequential/Loop agents temporarily and keep deterministic nodes isolated.")
    if probes.get("database_session_service", {}).get("available"):
        actions.append("Use DatabaseSessionService for resumable Agent sessions.")
    else:
        actions.append("DatabaseSessionService was not detected; keep AnyChain file-backed session/job state as the durable source.")
    if probes.get("evaluator", {}).get("available"):
        actions.append("Use ADK AgentEvaluator for trajectory and multi-turn behavior tests.")
    else:
        actions.append("ADK evaluator was not detected; keep offline unittest/tool-contract tests and add evaluator support later.")
    return actions
