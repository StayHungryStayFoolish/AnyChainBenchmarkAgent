"""Official ADK discovery module.

Google ADK expects an agent package to expose ``root_agent`` from a module that
the ADK CLI can discover. Keep this file intentionally thin; the actual agent
definition lives in ``root_agent.py``.
"""

from .root_agent import root_agent

__all__ = ["root_agent"]
