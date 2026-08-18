"""Versioned critical-sequence contracts and denominator reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CRITICAL_SEQUENCE_SCHEMA_VERSION = 2
DEFAULT_REGISTRY_PATH = Path(__file__).with_name("critical_sequences.v2.json")


@dataclass(frozen=True)
class CriticalSequenceStep:
    step_id: str
    coverage_ids: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class CriticalSequence:
    sequence_id: str
    title: str
    steps: tuple[CriticalSequenceStep, ...]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CriticalSequenceRegistry:
    registry_id: str
    version: int
    sequences: tuple[CriticalSequence, ...]


@dataclass(frozen=True)
class CriticalSequenceObservation:
    sequence_id: str
    registry_version: int
    round_id: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ObservedCoverageTurn:
    evidence_id: str
    round_id: str
    turn_index: int
    revision: Mapping[str, str]
    coverage_ids: tuple[str, ...]


def load_critical_sequence_registry(
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> CriticalSequenceRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return critical_sequence_registry_from_dict(payload)


def critical_sequence_registry_from_dict(
    payload: Mapping[str, Any],
) -> CriticalSequenceRegistry:
    version = int(payload.get("version") or 0)
    if version != CRITICAL_SEQUENCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported critical-sequence registry version: {version}")
    registry_id = str(payload.get("registry_id") or "").strip()
    if not registry_id:
        raise ValueError("critical-sequence registry_id is required")
    sequences: list[CriticalSequence] = []
    sequence_ids: set[str] = set()
    for raw_sequence in payload.get("sequences") or []:
        sequence_id = str(raw_sequence.get("sequence_id") or "").strip()
        if not sequence_id or sequence_id in sequence_ids:
            raise ValueError(f"invalid or duplicate critical sequence id: {sequence_id!r}")
        sequence_ids.add(sequence_id)
        steps: list[CriticalSequenceStep] = []
        step_ids: set[str] = set()
        for raw_step in raw_sequence.get("steps") or []:
            step_id = str(raw_step.get("step_id") or "").strip()
            coverage_ids = tuple(
                str(item).strip() for item in raw_step.get("coverage_ids") or ()
                if str(item).strip()
            )
            if not step_id or step_id in step_ids or not coverage_ids:
                raise ValueError(
                    f"invalid or duplicate step in critical sequence {sequence_id}: {step_id!r}"
                )
            step_ids.add(step_id)
            steps.append(CriticalSequenceStep(
                step_id=step_id,
                coverage_ids=coverage_ids,
                description=str(raw_step.get("description") or ""),
            ))
        if not steps:
            raise ValueError(f"critical sequence has no steps: {sequence_id}")
        sequences.append(CriticalSequence(
            sequence_id=sequence_id,
            title=str(raw_sequence.get("title") or sequence_id),
            steps=tuple(steps),
            tags=tuple(str(item) for item in raw_sequence.get("tags") or ()),
        ))
    if not sequences:
        raise ValueError("critical-sequence registry has no sequences")
    return CriticalSequenceRegistry(registry_id, version, tuple(sequences))


def match_critical_sequence(
    sequence: CriticalSequence,
    observed_turn_coverage_ids: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    """Return the ordered prefix of steps matched by observed CLI turns."""

    matched: list[str] = []
    next_turn = 0
    normalized_turns = [set(str(item) for item in turn) for turn in observed_turn_coverage_ids]
    for step in sequence.steps:
        required = set(step.coverage_ids)
        found = next(
            (
                index for index in range(next_turn, len(normalized_turns))
                if required.issubset(normalized_turns[index])
            ),
            None,
        )
        if found is None:
            break
        matched.append(step.step_id)
        next_turn = found + 1
    return tuple(matched)


def critical_sequence_denominator_report(
    registry: CriticalSequenceRegistry,
    observations: Iterable[CriticalSequenceObservation],
    *,
    evidence_index: Mapping[str, ObservedCoverageTurn] | None = None,
    revision: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Report explicit sequence and step denominators without vague pass claims."""

    available_evidence = dict(evidence_index or {})
    active_revision = dict(revision or {})
    observations_by_sequence: dict[
        str, list[tuple[CriticalSequenceObservation, tuple[tuple[str, ...], ...]]]
    ] = {}
    invalid_observations: list[dict[str, str]] = []
    for observation in observations:
        if observation.registry_version != registry.version:
            invalid_observations.append({
                "sequence_id": observation.sequence_id,
                "round_id": observation.round_id,
                "reason": "registry version mismatch",
            })
            continue
        turns: list[ObservedCoverageTurn] = []
        reason = ""
        if not observation.evidence_ids:
            reason = "critical sequence observation has no evidence ids"
        for evidence_id in observation.evidence_ids:
            turn = available_evidence.get(evidence_id)
            if turn is None:
                reason = f"unknown evidence id: {evidence_id}"
                break
            if turn.evidence_id != evidence_id:
                reason = f"evidence identity mismatch: {evidence_id}"
                break
            if turn.round_id != observation.round_id:
                reason = f"evidence round mismatch: {evidence_id}"
                break
            if active_revision and dict(turn.revision) != active_revision:
                reason = f"evidence revision mismatch: {evidence_id}"
                break
            if not turn.coverage_ids:
                reason = f"evidence has no observed coverage ids: {evidence_id}"
                break
            turns.append(turn)
        if reason:
            invalid_observations.append({
                "sequence_id": observation.sequence_id,
                "round_id": observation.round_id,
                "reason": reason,
            })
            continue
        turns.sort(key=lambda item: item.turn_index)
        if len({turn.turn_index for turn in turns}) != len(turns):
            invalid_observations.append({
                "sequence_id": observation.sequence_id,
                "round_id": observation.round_id,
                "reason": "evidence repeats a turn index",
            })
            continue
        observations_by_sequence.setdefault(observation.sequence_id, []).append((
            observation,
            tuple(turn.coverage_ids for turn in turns),
        ))

    rows: list[dict[str, Any]] = []
    covered_steps = 0
    passed_sequences = 0
    for sequence in registry.sequences:
        best_matched: tuple[str, ...] = ()
        best_observation: CriticalSequenceObservation | None = None
        for observation, observed_turns in observations_by_sequence.get(sequence.sequence_id, []):
            matched = match_critical_sequence(sequence, observed_turns)
            if len(matched) > len(best_matched):
                best_matched = matched
                best_observation = observation
        expected_step_ids = tuple(step.step_id for step in sequence.steps)
        passed = best_matched == expected_step_ids
        covered_steps += len(best_matched)
        passed_sequences += int(passed)
        rows.append({
            "sequence_id": sequence.sequence_id,
            "expected_steps": list(expected_step_ids),
            "matched_steps": list(best_matched),
            "uncovered_steps": list(expected_step_ids[len(best_matched):]),
            "passed": passed,
            "round_id": best_observation.round_id if best_observation else "",
            "evidence_ids": list(best_observation.evidence_ids) if best_observation else [],
        })

    total_sequences = len(registry.sequences)
    total_steps = sum(len(sequence.steps) for sequence in registry.sequences)
    return {
        "registry_id": registry.registry_id,
        "registry_version": registry.version,
        "sequence_denominator": total_sequences,
        "passed_sequences": passed_sequences,
        "uncovered_sequences": total_sequences - passed_sequences,
        "step_denominator": total_steps,
        "covered_steps": covered_steps,
        "uncovered_steps": total_steps - covered_steps,
        "invalid_observations": invalid_observations,
        "sequences": rows,
    }


def observed_coverage_turn_from_artifact(
    *,
    bundle_path: str | Path,
    artifact_id: str,
    edge: Mapping[str, Any],
    revision: Mapping[str, str],
    trusted_public_key_b64: str,
) -> ObservedCoverageTurn:
    """Create sequence input only from a validated shard-bundle member."""

    from tests.agent_live.coverage_evidence import (
        load_validated_pty_artifact_bundle,
        validate_pty_cli_evidence_artifact,
    )

    try:
        items, _digest = load_validated_pty_artifact_bundle(
            bundle_path,
            trusted_public_key_b64=trusted_public_key_b64,
        )
    except ValueError as exc:
        reason = str(exc)
        raise ValueError(f"invalid PTY shard bundle: {reason}")
    matches = [
        item
        for item in items
        if str(item["artifact"].get("evidence_id") or "") == artifact_id
    ]
    if len(matches) != 1:
        raise ValueError("PTY observation is not a unique shard-bundle member")
    item = matches[0]
    artifact = item["artifact"]
    valid, reason = validate_pty_cli_evidence_artifact(
        artifact,
        edge=edge,
        revision=revision,
        authority=item["authority"],
        trusted_public_key_b64=trusted_public_key_b64,
    )
    if not valid:
        raise ValueError(f"invalid PTY observation artifact: {reason}")
    observation = dict(artifact.get("turn_observation") or {})
    postcondition = dict(observation.get("verified_postcondition") or {})
    coverage_ids = tuple(str(item) for item in postcondition.get("observed_coverage_ids") or ())
    turn = dict(artifact.get("turn") or {})
    return ObservedCoverageTurn(
        evidence_id=str(artifact.get("evidence_id") or ""),
        round_id=str(turn.get("session_id") or ""),
        turn_index=int(turn.get("turn_index") or 0),
        revision=dict(revision),
        coverage_ids=coverage_ids,
    )
