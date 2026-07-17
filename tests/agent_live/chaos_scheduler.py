"""Revision-bound scheduler artifacts for response-driven Chaos.

This module schedules coverage work only. It does not call Codex, generate user
turns, execute the CLI, or claim that a scheduled row was observed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.agent_live.coverage_evidence import content_hash


CHAOS_SCHEDULE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ScheduledCoverageTarget:
    target_id: str
    edge_key: str
    persona: str
    goal: str
    sequence_id: str = ""
    tuple_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChaosSchedule:
    schedule_id: str
    seed: int
    revision: Mapping[str, str]
    ledger_hash: str
    targets: tuple[ScheduledCoverageTarget, ...]
    schema_version: int = CHAOS_SCHEDULE_SCHEMA_VERSION


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
        scheduled.append(ScheduledCoverageTarget(
            target_id=target_id,
            edge_key=edge_key,
            persona=persona,
            goal=goal,
            sequence_id=str(raw.get("sequence_id") or ""),
            tuple_ids=tuple(str(item) for item in raw.get("tuple_ids") or ()),
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
