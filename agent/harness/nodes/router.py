"""Router node for the AnyChain Agent Harness."""

from __future__ import annotations

from ..state import AgentGraphState


def route_after_user_turn(state: AgentGraphState) -> str:
    _ = state
    return "turn"
