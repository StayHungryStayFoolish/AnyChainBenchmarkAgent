"""ADK tool registry."""

from __future__ import annotations

from .actions import get_action_tools
from .auth import get_auth_tools
from .enterprise import get_enterprise_tools
from .planning import get_planning_tools
from .read_only import get_read_only_tools
from .validators import get_validator_tools
from .workflow_state import get_workflow_state_tools


def get_adk_tools(include_actions: bool = False) -> list:
    """Return ADK tool callables.

    Action tools are available to the ADK root agent only when explicitly
    requested. They must be paired with ADK confirmation callbacks and runner
    guardrails.
    """
    tools = [
        *get_workflow_state_tools(),
        *get_read_only_tools(),
        *get_auth_tools(),
        *get_planning_tools(),
        *get_validator_tools(),
        *get_enterprise_tools(),
    ]
    if include_actions:
        return [*tools, *get_action_tools()]
    return tools
