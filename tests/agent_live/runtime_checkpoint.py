"""Shared reviewed-checkpoint setup for Linux product-CLI evidence runners."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from agent.harness.checkpoints import create_sqlite_checkpointer
from agent.harness.graph import build_graph
from agent.harness.invariants import validate_state
from agent.harness.state import project_checkpoint_state
from tests.agent_live.harness_contract_scenarios import question_scenarios


def reviewed_scenario_state(scenario_id: str) -> Mapping[str, Any]:
    """Return a copy of one executable state from the authoritative registry."""

    scenarios = {item.scenario_id: item for item in question_scenarios("en")}
    scenario = scenarios.get(str(scenario_id))
    if scenario is None or scenario.seed_state is None:
        raise ValueError(f"dynamic target requires an executable scenario: {scenario_id}")
    return deepcopy(dict(scenario.seed_state))


def seed_runtime_checkpoint(
    seed_state: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    session_id: str,
    session_purpose: str,
) -> None:
    """Persist one reviewed state through the product checkpointer boundary."""

    state = deepcopy(dict(seed_state))
    state["last_user_input"] = ""
    state["thread_id"] = session_id
    state["session"] = {
        "id": session_id,
        "purpose": session_purpose,
        "created_at": "1970-01-01T00:00:00Z",
        "updated_at": "1970-01-01T00:00:00Z",
    }
    validate_state(state)
    checkpointer = create_sqlite_checkpointer(checkpoint_path)
    try:
        graph = build_graph(checkpointer)
        graph.update_state(
            {"configurable": {"thread_id": session_id}},
            project_checkpoint_state(state),
        )
    finally:
        manager = getattr(checkpointer, "_anychain_context_manager", None)
        if manager is not None:
            manager.__exit__(None, None, None)
