"""Enterprise integration manifest for ADK platform owners."""

from __future__ import annotations

from .actions import get_action_tools
from .planning import get_planning_tools
from .read_only import get_read_only_tools


def enterprise_integration_manifest() -> dict:
    """Return the ADK integration boundary for enterprise Agent platforms."""
    read_only = [tool.__name__ for tool in get_read_only_tools()]
    planning = [tool.__name__ for tool in get_planning_tools()]
    actions = [tool.__name__ for tool in get_action_tools()]
    return {
        "status": "ok",
        "agent": "AnyChain Benchmark Agent",
        "integration_mode": "ADK tool orchestration over deterministic AnyChain benchmark tools",
        "read_only_tools": read_only,
        "planning_tools": planning,
        "confirmation_gated_tools": actions,
        "requirements": [
            "read-only tools may run without user approval",
            "planning tools may generate artifacts but must not launch benchmarks",
            "confirmation-gated tools require explicit user approval",
            "real benchmark launch must follow preflight and smoke",
            "artifact paths must be preserved in platform logs",
            "credential values and model raw responses must not be logged",
            "KB adapter failures must be reported; local repo capabilities remain available as separate evidence",
        ],
        "knowledge_base": {
            "default": "local repo capability provider",
            "optional_http_contract": [
                "POST /search",
                "GET /chains/{chain}/rpc-methods",
                "GET /chains/{chain}/rpc-samples",
                "POST /workload/suggest",
            ],
        },
    }


def get_enterprise_tools() -> list:
    """Return enterprise metadata tool callables."""
    return [enterprise_integration_manifest]
