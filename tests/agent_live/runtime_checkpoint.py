"""Shared reviewed-checkpoint setup for Linux product-CLI evidence runners."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from agent.harness.checkpoints import create_sqlite_checkpointer
from agent.harness.graph import build_graph
from agent.harness.invariants import validate_state
from agent.harness.state import project_checkpoint_state
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.harness_contract_scenarios import (
    ActionTransitionScenario,
    QuestionScenario,
    action_transition_scenarios,
    canonical_question_contract,
    canonical_scenario_state,
    question_scenarios,
)

ReviewedScenario = QuestionScenario | ActionTransitionScenario


def reviewed_scenario_state(scenario_id: str) -> Mapping[str, Any]:
    """Return a copy of one executable state from the authoritative registry."""

    scenarios = {
        item.scenario_id: item
        for item in [*question_scenarios("en"), *action_transition_scenarios("en")]
    }
    scenario = scenarios.get(str(scenario_id))
    if scenario is None or not scenario.seed_state:
        raise ValueError(f"dynamic target requires an executable scenario: {scenario_id}")
    return deepcopy(dict(scenario.seed_state))


@lru_cache(maxsize=None)
def reviewed_scenario(scenario_id: str) -> ReviewedScenario:
    """Return one authoritative executable scenario for evidence setup."""

    scenarios = {
        item.scenario_id: item
        for item in [*question_scenarios("en"), *action_transition_scenarios("en")]
    }
    scenario = scenarios.get(str(scenario_id))
    if scenario is None or not scenario.seed_state:
        raise ValueError(f"evidence target requires an executable scenario: {scenario_id}")
    return scenario


def reviewed_pending_contract(
    scenario: ReviewedScenario,
) -> dict[str, Any]:
    """Return the reviewed pending contract for a typed start scenario."""

    if isinstance(scenario, QuestionScenario):
        return deepcopy(dict(scenario.question))
    if isinstance(scenario, ActionTransitionScenario):
        return {}
    raise TypeError(f"unsupported reviewed scenario: {type(scenario).__name__}")


@dataclass(frozen=True)
class SeedReceipt:
    scenario_id: str
    scenario_state_fingerprint: str
    seed_state_hash: str
    projected_state_hash: str
    checkpoint_sha256: str
    checkpoint_path: str
    session_id: str
    session_purpose: str
    pending_question_id: str
    pending_contract_hash: str

    @property
    def payload(self) -> dict[str, str]:
        return asdict(self)

    @property
    def receipt_hash(self) -> str:
        return content_hash(self.payload)


def seed_runtime_checkpoint(
    seed_state: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    session_id: str,
    session_purpose: str,
    scenario_id: str,
    scenario_state_fingerprint: str,
) -> SeedReceipt:
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
    checkpoint = Path(checkpoint_path).resolve()
    projected = project_checkpoint_state(state)
    checkpointer = create_sqlite_checkpointer(checkpoint)
    try:
        graph = build_graph(checkpointer)
        graph.update_state(
            {"configurable": {"thread_id": session_id}},
            projected,
        )
    finally:
        manager = getattr(checkpointer, "_anychain_context_manager", None)
        if manager is not None:
            manager.__exit__(None, None, None)
    question = dict(state.get("pending_question") or {})
    return SeedReceipt(
        scenario_id=str(scenario_id),
        scenario_state_fingerprint=str(scenario_state_fingerprint),
        seed_state_hash=content_hash(canonical_scenario_state(seed_state)),
        projected_state_hash=content_hash(projected),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_path=str(checkpoint),
        session_id=session_id,
        session_purpose=session_purpose,
        pending_question_id=str(question.get("id") or ""),
        pending_contract_hash=content_hash(canonical_question_contract(question)),
    )
