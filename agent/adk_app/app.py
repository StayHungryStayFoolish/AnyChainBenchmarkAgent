"""ADK app diagnostics used by CLI and tests."""

from __future__ import annotations

from .instructions import ADK_MIGRATION_BOUNDARY, ROOT_INSTRUCTION
from .compat import adk_feature_report
from .models import adk_status
from .runner_bridge import runner_bridge_status


def status_payload() -> dict:
    """Return ADK readiness without requiring model credentials."""
    status = adk_status().as_dict()
    runner_status = runner_bridge_status().as_dict()
    return {
        "status": "ready" if status["available"] else "not_installed",
        "adk": status,
        "features": adk_feature_report().get("features", {}),
        "runner": runner_status,
        "root_instruction_present": bool(ROOT_INSTRUCTION),
        "migration_boundary": ADK_MIGRATION_BOUNDARY,
        "next_actions": [
            "install google-adk in an isolated environment",
            "configure a real model provider in config/agent_config.sh",
            "run python3 agent/cli.py adk-eval for no-key contract checks",
            "run python3 agent/cli.py llm-smoke only after credentials are configured",
        ],
    }
