"""Revision-bound scheduler artifacts for response-driven Chaos.

This module schedules coverage work only. It does not call Codex, generate user
turns, execute the CLI, or claim that a scheduled row was observed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from tests.agent_live.coverage_evidence import content_hash


CHAOS_SCHEDULE_SCHEMA_VERSION = 4
JOURNEY_SCHEDULE_SCHEMA_VERSION = 1
DEFAULT_DEFERRED_CONTINUATION_TURN_BUDGET = 12


@dataclass(frozen=True)
class ScheduledCoverageTarget:
    target_id: str
    edge_key: str
    persona: str
    goal: str
    sequence_id: str = ""
    tuple_ids: tuple[str, ...] = ()
    scenario_id: str = ""
    continuation_turn_budget: int = 0


@dataclass(frozen=True)
class ChaosSchedule:
    schedule_id: str
    seed: int
    revision: Mapping[str, str]
    ledger_hash: str
    targets: tuple[ScheduledCoverageTarget, ...]
    schema_version: int = CHAOS_SCHEDULE_SCHEMA_VERSION


@dataclass(frozen=True)
class JourneyOutcomeContract:
    outcome_id: str
    required_postcondition_ids: tuple[str, ...]


@dataclass(frozen=True)
class JourneySchedule:
    schedule_id: str
    journey_id: str
    seed: int
    revision: Mapping[str, str]
    start_scenario: str
    persona: str
    mission: str
    allowed_risk_factors: tuple[str, ...]
    max_turns: int
    terminal_outcome: JourneyOutcomeContract
    forbidden_outcomes: tuple[JourneyOutcomeContract, ...]
    schema_version: int = JOURNEY_SCHEDULE_SCHEMA_VERSION


def build_chaos_schedule(
    ledger: Mapping[str, Any],
    *,
    revision: Mapping[str, str],
    seed: int,
    targets: Sequence[Mapping[str, Any]],
) -> ChaosSchedule:
    """Bind requested targets to applicable edges in one immutable ledger."""

    if dict(ledger.get("revision") or {}) != dict(revision):
        raise ValueError("coverage ledger revision does not match the schedule revision")
    edge_index = {
        str(edge.get("edge_key") or ""): edge
        for edge in ledger.get("edges") or ()
        if str(edge.get("edge_key") or "")
    }
    scheduled: list[ScheduledCoverageTarget] = []
    seen_target_ids: set[str] = set()
    for index, raw in enumerate(targets):
        edge_key = str(raw.get("edge_key") or "").strip()
        edge = edge_index.get(edge_key)
        if edge is None:
            raise ValueError(f"unknown authoritative ledger edge: {edge_key}")
        lane = (edge.get("evidence") or {}).get("dynamic_dual_ai") or {}
        if not bool(lane.get("required")):
            raise ValueError(
                "cannot schedule a ledger edge outside the dynamic_dual_ai lane: "
                f"{edge_key} ({lane.get('applicability_reason') or 'unspecified'})"
            )
        target_id = str(raw.get("target_id") or f"target-{index + 1}").strip()
        if not target_id or target_id in seen_target_ids:
            raise ValueError(f"invalid or duplicate schedule target id: {target_id!r}")
        seen_target_ids.add(target_id)
        persona = str(raw.get("persona") or "").strip()
        goal = str(raw.get("goal") or "").strip()
        if not persona or not goal:
            raise ValueError(f"schedule target requires persona and goal: {target_id}")
        scenario_id = str(raw.get("scenario_id") or "").strip()
        executable_scenarios = {
            str(item) for item in edge.get("executable_scenario_ids") or ()
        }
        if scenario_id and scenario_id not in executable_scenarios:
            raise ValueError(
                f"schedule scenario is not authoritative for edge: {scenario_id} -> {edge_key}"
            )
        deferred_contract = dict(edge.get("deferred_transition_contract") or {})
        requires_continuation = deferred_contract.get("requires_linked_journey") is True
        declared_budget = deferred_contract.get(
            "max_continuation_turns",
            DEFAULT_DEFERRED_CONTINUATION_TURN_BUDGET if requires_continuation else 0,
        )
        if isinstance(declared_budget, bool) or not isinstance(declared_budget, int):
            raise ValueError(f"invalid deferred continuation budget: {edge_key}")
        continuation_turn_budget = int(declared_budget)
        if requires_continuation and continuation_turn_budget <= 0:
            raise ValueError(f"deferred edge has no executable continuation budget: {edge_key}")
        if not requires_continuation and continuation_turn_budget != 0:
            raise ValueError(f"immediate edge declares a continuation budget: {edge_key}")
        requested_budget = raw.get("continuation_turn_budget")
        if requested_budget is not None and int(requested_budget) != continuation_turn_budget:
            raise ValueError(
                f"schedule cannot override ledger continuation budget: {edge_key}"
            )
        scheduled.append(ScheduledCoverageTarget(
            target_id=target_id,
            edge_key=edge_key,
            persona=persona,
            goal=goal,
            sequence_id=str(raw.get("sequence_id") or ""),
            tuple_ids=tuple(str(item) for item in raw.get("tuple_ids") or ()),
            scenario_id=scenario_id,
            continuation_turn_budget=continuation_turn_budget,
        ))
    if not scheduled:
        raise ValueError("Chaos schedule has no coverage targets")

    ledger_hash = content_hash(ledger)
    identity = {
        "seed": int(seed),
        "revision": dict(revision),
        "ledger_hash": ledger_hash,
        "targets": [asdict(target) for target in scheduled],
    }
    return ChaosSchedule(
        schedule_id=content_hash(identity),
        seed=int(seed),
        revision=dict(revision),
        ledger_hash=ledger_hash,
        targets=tuple(scheduled),
    )


def schedule_payload(schedule: ChaosSchedule) -> dict[str, Any]:
    payload = asdict(schedule)
    payload["revision"] = dict(schedule.revision)
    payload["targets"] = [
        {**asdict(target), "tuple_ids": list(target.tuple_ids)}
        for target in schedule.targets
    ]
    return payload


def validate_chaos_schedule(
    schedule: ChaosSchedule,
    *,
    ledger: Mapping[str, Any],
    revision: Mapping[str, str],
) -> None:
    rebuilt = build_chaos_schedule(
        ledger,
        revision=revision,
        seed=schedule.seed,
        targets=[asdict(target) for target in schedule.targets],
    )
    if schedule.schema_version != CHAOS_SCHEDULE_SCHEMA_VERSION:
        raise ValueError("unsupported Chaos schedule schema")
    if schedule.schedule_id != rebuilt.schedule_id:
        raise ValueError("Chaos schedule identity is stale or was modified")
    if schedule.ledger_hash != rebuilt.ledger_hash:
        raise ValueError("Chaos schedule ledger hash is stale")


def write_chaos_schedule(schedule: ChaosSchedule, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(schedule_payload(schedule), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def build_journey_schedule(
    *,
    revision: Mapping[str, str],
    seed: int,
    journey: Mapping[str, Any],
) -> JourneySchedule:
    """Build one immutable open-journey contract without prescribing user turns."""

    allowed_fields = {
        "journey_id",
        "start_scenario",
        "persona",
        "mission",
        "allowed_risk_factors",
        "max_turns",
        "terminal_outcome",
        "forbidden_outcomes",
    }
    unknown_fields = sorted(set(journey) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            "journey schedule contains unsupported fields: " + ", ".join(unknown_fields)
        )

    journey_id = _required_journey_text(journey, "journey_id")
    start_scenario = _required_journey_text(journey, "start_scenario")
    persona = _required_journey_text(journey, "persona")
    mission = _required_journey_text(journey, "mission")
    allowed_risk_factors = _unique_journey_ids(
        journey.get("allowed_risk_factors") or (),
        field="allowed_risk_factors",
    )
    max_turns = journey.get("max_turns")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
        raise ValueError("journey schedule max_turns must be a positive integer")

    terminal_outcome = _journey_outcome_contract(
        journey.get("terminal_outcome"),
        field="terminal_outcome",
    )
    raw_forbidden = journey.get("forbidden_outcomes")
    if not isinstance(raw_forbidden, Sequence) or isinstance(raw_forbidden, (str, bytes)):
        raise ValueError("journey schedule forbidden_outcomes must be a sequence")
    forbidden_outcomes = tuple(
        _journey_outcome_contract(item, field=f"forbidden_outcomes[{index}]")
        for index, item in enumerate(raw_forbidden)
    )
    forbidden_ids = [item.outcome_id for item in forbidden_outcomes]
    if len(forbidden_ids) != len(set(forbidden_ids)):
        raise ValueError("journey schedule forbidden outcome ids must be unique")
    if terminal_outcome.outcome_id in forbidden_ids:
        raise ValueError("journey terminal outcome cannot also be forbidden")

    identity = {
        "journey_id": journey_id,
        "seed": int(seed),
        "revision": dict(revision),
        "start_scenario": start_scenario,
        "persona": persona,
        "mission": mission,
        "allowed_risk_factors": list(allowed_risk_factors),
        "max_turns": max_turns,
        "terminal_outcome": _journey_outcome_payload(terminal_outcome),
        "forbidden_outcomes": [
            _journey_outcome_payload(item) for item in forbidden_outcomes
        ],
    }
    return JourneySchedule(
        schedule_id=content_hash(identity),
        journey_id=journey_id,
        seed=int(seed),
        revision=MappingProxyType(dict(revision)),
        start_scenario=start_scenario,
        persona=persona,
        mission=mission,
        allowed_risk_factors=allowed_risk_factors,
        max_turns=max_turns,
        terminal_outcome=terminal_outcome,
        forbidden_outcomes=forbidden_outcomes,
    )


def journey_schedule_payload(schedule: JourneySchedule) -> dict[str, Any]:
    return {
        "schema_version": schedule.schema_version,
        "schedule_id": schedule.schedule_id,
        "journey_id": schedule.journey_id,
        "seed": schedule.seed,
        "revision": dict(schedule.revision),
        "start_scenario": schedule.start_scenario,
        "persona": schedule.persona,
        "mission": schedule.mission,
        "allowed_risk_factors": list(schedule.allowed_risk_factors),
        "max_turns": schedule.max_turns,
        "terminal_outcome": _journey_outcome_payload(schedule.terminal_outcome),
        "forbidden_outcomes": [
            _journey_outcome_payload(item) for item in schedule.forbidden_outcomes
        ],
    }


def validate_journey_schedule(
    schedule: JourneySchedule,
    *,
    revision: Mapping[str, str],
) -> None:
    if schedule.schema_version != JOURNEY_SCHEDULE_SCHEMA_VERSION:
        raise ValueError("unsupported Journey schedule schema")
    if dict(schedule.revision) != dict(revision):
        raise ValueError("Journey schedule revision does not match the active revision")
    rebuilt = build_journey_schedule(
        revision=revision,
        seed=schedule.seed,
        journey={
            "journey_id": schedule.journey_id,
            "start_scenario": schedule.start_scenario,
            "persona": schedule.persona,
            "mission": schedule.mission,
            "allowed_risk_factors": schedule.allowed_risk_factors,
            "max_turns": schedule.max_turns,
            "terminal_outcome": schedule.terminal_outcome,
            "forbidden_outcomes": schedule.forbidden_outcomes,
        },
    )
    if schedule.schedule_id != rebuilt.schedule_id:
        raise ValueError("Journey schedule identity is stale or was modified")


def write_journey_schedule(schedule: JourneySchedule, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            journey_schedule_payload(schedule),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return target


def _required_journey_text(journey: Mapping[str, Any], field: str) -> str:
    value = str(journey.get(field) or "").strip()
    if not value:
        raise ValueError(f"journey schedule requires {field}")
    return value


def _unique_journey_ids(values: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"journey schedule {field} must be a sequence")
    normalized = tuple(str(item).strip() for item in values)
    if any(not item for item in normalized):
        raise ValueError(f"journey schedule {field} cannot contain empty ids")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"journey schedule {field} must contain unique ids")
    return normalized


def _journey_outcome_contract(value: Any, *, field: str) -> JourneyOutcomeContract:
    if isinstance(value, JourneyOutcomeContract):
        contract = value
    elif isinstance(value, Mapping):
        allowed_fields = {"outcome_id", "required_postcondition_ids"}
        unknown_fields = sorted(set(value) - allowed_fields)
        if unknown_fields:
            raise ValueError(
                f"journey schedule {field} contains unsupported fields: "
                + ", ".join(unknown_fields)
            )
        outcome_id = str(value.get("outcome_id") or "").strip()
        if not outcome_id:
            raise ValueError(f"journey schedule {field} requires outcome_id")
        contract = JourneyOutcomeContract(
            outcome_id=outcome_id,
            required_postcondition_ids=_unique_journey_ids(
                value.get("required_postcondition_ids") or (),
                field=f"{field}.required_postcondition_ids",
            ),
        )
    else:
        raise ValueError(f"journey schedule {field} must be an outcome contract")
    if not contract.outcome_id.strip():
        raise ValueError(f"journey schedule {field} requires outcome_id")
    if not contract.required_postcondition_ids:
        raise ValueError(
            f"journey schedule {field} requires at least one postcondition id"
        )
    return JourneyOutcomeContract(
        outcome_id=contract.outcome_id.strip(),
        required_postcondition_ids=_unique_journey_ids(
            contract.required_postcondition_ids,
            field=f"{field}.required_postcondition_ids",
        ),
    )


def _journey_outcome_payload(contract: JourneyOutcomeContract) -> dict[str, Any]:
    return {
        "outcome_id": contract.outcome_id,
        "required_postcondition_ids": list(contract.required_postcondition_ids),
    }
