"""Append-only, cross-revision discovery history for Agent live testing.

This ledger records what was attempted and how the attempt ended. It never
qualifies a release and deliberately has no dependency on the revision-bound
coverage ledger or its evidence ingestion APIs.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


DISCOVERY_LEDGER_SCHEMA_VERSION = 1
RESULT_CLASSIFICATIONS = frozenset({
    "passed",
    "product_failed",
    "simulator_invalid",
    "infrastructure_interrupted",
    "externally_blocked",
})
REPAIR_STAGES = frozenset({
    "none",
    "source_open",
    "source_fixed",
    "targeted_verified",
    "chaos_verified",
    "formally_closed",
})
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class DiscoveryAttempt:
    """One immutable terminal shard result retained across source revisions."""

    attempt_id: str
    batch_id: str
    shard_id: str
    stable_coverage_ids: tuple[str, ...]
    journey_ids: tuple[str, ...]
    revision: Mapping[str, str]
    schedule_hash: str
    response_hashes: tuple[str, ...]
    decision_hashes: tuple[str, ...]
    transcript_hash: str
    persona: str
    input_classes: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    started_at: str
    finished_at: str
    classification: str
    diagnostic_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    defect_id: str
    repair_stage: str
    supersedes_attempt_id: str = ""
    schema_version: int = DISCOVERY_LEDGER_SCHEMA_VERSION


def build_discovery_attempt(
    *,
    batch_id: str,
    shard_id: str,
    stable_coverage_ids: Iterable[str] = (),
    journey_ids: Iterable[str] = (),
    revision: Mapping[str, str],
    schedule_hash: str,
    response_hashes: Iterable[str],
    decision_hashes: Iterable[str],
    transcript_hash: str,
    persona: str,
    input_classes: Iterable[str],
    sequence_ids: Iterable[str],
    factor_ids: Iterable[str],
    started_at: str,
    finished_at: str,
    classification: str,
    diagnostic_ids: Iterable[str] = (),
    evidence_ids: Iterable[str] = (),
    defect_id: str = "",
    repair_stage: str = "none",
    supersedes_attempt_id: str = "",
) -> DiscoveryAttempt:
    """Build and validate a content-addressed discovery attempt."""

    attempt = DiscoveryAttempt(
        attempt_id="",
        batch_id=str(batch_id).strip(),
        shard_id=str(shard_id).strip(),
        stable_coverage_ids=_normalized_ids(stable_coverage_ids),
        journey_ids=_normalized_ids(journey_ids),
        revision={str(key): str(value) for key, value in revision.items()},
        schedule_hash=str(schedule_hash).strip(),
        response_hashes=_normalized_ids(response_hashes),
        decision_hashes=_normalized_ids(decision_hashes),
        transcript_hash=str(transcript_hash).strip(),
        persona=str(persona).strip(),
        input_classes=_normalized_ids(input_classes),
        sequence_ids=_normalized_ids(sequence_ids),
        factor_ids=_normalized_ids(factor_ids),
        started_at=str(started_at).strip(),
        finished_at=str(finished_at).strip(),
        classification=str(classification).strip(),
        diagnostic_ids=_normalized_ids(diagnostic_ids),
        evidence_ids=_normalized_ids(evidence_ids),
        defect_id=str(defect_id).strip(),
        repair_stage=str(repair_stage).strip(),
        supersedes_attempt_id=str(supersedes_attempt_id).strip(),
    )
    attempt = replace(attempt, attempt_id=_attempt_id(attempt))
    validate_discovery_attempt(attempt)
    return attempt


def append_discovery_attempt(path: str | Path, attempt: DiscoveryAttempt) -> str:
    """Validate and atomically append one attempt without rewriting history."""

    validate_discovery_attempt(attempt)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    with os.fdopen(descriptor, "r+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        existing = _load_lines(stream.read().splitlines(), source=str(target))
        _validate_append(existing, attempt)
        stream.seek(0, os.SEEK_END)
        stream.write(_canonical_json(attempt_payload(attempt)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return attempt.attempt_id


def load_discovery_attempts(path: str | Path) -> tuple[DiscoveryAttempt, ...]:
    """Load and fully validate an append-only discovery log."""

    target = Path(path)
    if not target.exists():
        return ()
    return _load_lines(target.read_text(encoding="utf-8").splitlines(), source=str(target))


def aggregate_discovery_attempts(
    attempts: Iterable[DiscoveryAttempt],
) -> dict[str, Any]:
    """Derive a reproducible snapshot without changing the authoritative log."""

    rows = tuple(sorted(attempts, key=lambda attempt: attempt.attempt_id))
    _validate_record_set(rows)
    superseded_ids = {
        attempt.supersedes_attempt_id
        for attempt in rows
        if attempt.supersedes_attempt_id
    }
    active = tuple(row for row in rows if row.attempt_id not in superseded_ids)
    snapshot = {
        "schema_version": 1,
        "attempt_count": len(rows),
        "active_attempt_count": len(active),
        "scheduled": len(rows),
        "started": len(rows),
        "completed": len(rows),
        "classification_counts": _counts(rows, "classification", RESULT_CLASSIFICATIONS),
        "active_classification_counts": _counts(
            active,
            "classification",
            RESULT_CLASSIFICATIONS,
        ),
        "repair_stage_counts": _counts(rows, "repair_stage", REPAIR_STAGES),
        "active_repair_stage_counts": _counts(active, "repair_stage", REPAIR_STAGES),
        "coverage": _subject_index(rows, active, "stable_coverage_ids"),
        "journeys": _subject_index(rows, active, "journey_ids"),
        "attempt_ids": [row.attempt_id for row in rows],
        "active_attempt_ids": [row.attempt_id for row in active],
        "attempts": {
            row.attempt_id: attempt_payload(row)
            for row in rows
        },
    }
    snapshot["snapshot_id"] = _content_hash(snapshot)
    return snapshot


def attempt_payload(attempt: DiscoveryAttempt) -> dict[str, Any]:
    payload = asdict(attempt)
    for name in (
        "stable_coverage_ids",
        "journey_ids",
        "response_hashes",
        "decision_hashes",
        "input_classes",
        "sequence_ids",
        "factor_ids",
        "diagnostic_ids",
        "evidence_ids",
    ):
        payload[name] = list(payload[name])
    payload["revision"] = dict(payload["revision"])
    return payload


def validate_discovery_attempt(attempt: DiscoveryAttempt) -> None:
    """Reject incomplete, malformed, or non-content-addressed attempts."""

    if attempt.schema_version != DISCOVERY_LEDGER_SCHEMA_VERSION:
        raise ValueError("unsupported discovery attempt schema")
    required_text = {
        "attempt_id": attempt.attempt_id,
        "batch_id": attempt.batch_id,
        "shard_id": attempt.shard_id,
        "schedule_hash": attempt.schedule_hash,
        "transcript_hash": attempt.transcript_hash,
        "persona": attempt.persona,
        "started_at": attempt.started_at,
        "finished_at": attempt.finished_at,
        "classification": attempt.classification,
        "repair_stage": attempt.repair_stage,
        "revision.commit": attempt.revision.get("commit"),
        "revision.worktree_hash": attempt.revision.get("worktree_hash"),
    }
    missing = sorted(name for name, value in required_text.items() if not str(value or "").strip())
    if missing:
        raise ValueError(f"discovery attempt is missing: {', '.join(missing)}")
    if not attempt.stable_coverage_ids and not attempt.journey_ids:
        raise ValueError("discovery attempt requires a stable coverage or journey id")
    if attempt.classification not in RESULT_CLASSIFICATIONS:
        raise ValueError(f"invalid discovery classification: {attempt.classification}")
    if attempt.repair_stage not in REPAIR_STAGES:
        raise ValueError(f"invalid discovery repair stage: {attempt.repair_stage}")
    if not attempt.input_classes or not attempt.sequence_ids or not attempt.factor_ids:
        raise ValueError(
            "discovery attempt requires input class, sequence, and factor identities"
        )
    if not attempt.diagnostic_ids and not attempt.evidence_ids:
        raise ValueError("discovery attempt requires a diagnostic or evidence id")
    if attempt.classification == "passed" and (
        not attempt.response_hashes
        or not attempt.decision_hashes
        or not attempt.evidence_ids
    ):
        raise ValueError(
            "passed discovery attempt requires response hashes, decision hashes, and evidence ids"
        )
    if attempt.classification == "product_failed" and (
        not attempt.defect_id or not attempt.diagnostic_ids
    ):
        raise ValueError(
            "product_failed discovery attempt requires a defect id and diagnostic ids"
        )
    if attempt.repair_stage != "none" and not attempt.defect_id:
        raise ValueError("non-default repair stage requires a defect id")
    for name, values in (
        ("stable_coverage_ids", attempt.stable_coverage_ids),
        ("journey_ids", attempt.journey_ids),
        ("input_classes", attempt.input_classes),
        ("sequence_ids", attempt.sequence_ids),
        ("factor_ids", attempt.factor_ids),
        ("diagnostic_ids", attempt.diagnostic_ids),
        ("evidence_ids", attempt.evidence_ids),
    ):
        _validate_ids(name, values)
    for name, values in (
        ("response_hashes", attempt.response_hashes),
        ("decision_hashes", attempt.decision_hashes),
    ):
        _validate_hashes(name, values)
    for name, value in (
        ("schedule_hash", attempt.schedule_hash),
        ("transcript_hash", attempt.transcript_hash),
        ("revision.worktree_hash", str(attempt.revision.get("worktree_hash") or "")),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} is not a SHA-256 hash")
    started = _timestamp(attempt.started_at, "started_at")
    finished = _timestamp(attempt.finished_at, "finished_at")
    if finished < started:
        raise ValueError("finished_at precedes started_at")
    if attempt.supersedes_attempt_id and not _is_sha256(attempt.supersedes_attempt_id):
        raise ValueError("supersedes_attempt_id is not a content address")
    if attempt.attempt_id != _attempt_id(attempt):
        raise ValueError("discovery attempt content address mismatch")


def _validate_append(
    existing: tuple[DiscoveryAttempt, ...],
    attempt: DiscoveryAttempt,
) -> None:
    _validate_history(existing)
    by_id = {row.attempt_id: row for row in existing}
    if attempt.attempt_id in by_id:
        raise ValueError(f"duplicate discovery attempt: {attempt.attempt_id}")
    if not attempt.supersedes_attempt_id:
        return
    previous = by_id.get(attempt.supersedes_attempt_id)
    if previous is None:
        raise ValueError("superseded discovery attempt is not in the retained history")
    if _subject_scope(previous) != _subject_scope(attempt):
        raise ValueError("a superseding attempt must retain the same discovery subject scope")
    if any(row.supersedes_attempt_id == previous.attempt_id for row in existing):
        raise ValueError("a discovery attempt already has a superseding successor")


def _validate_history(attempts: tuple[DiscoveryAttempt, ...]) -> None:
    seen: dict[str, DiscoveryAttempt] = {}
    successors: set[str] = set()
    for attempt in attempts:
        validate_discovery_attempt(attempt)
        if attempt.attempt_id in seen:
            raise ValueError(f"duplicate discovery attempt: {attempt.attempt_id}")
        if attempt.supersedes_attempt_id:
            previous = seen.get(attempt.supersedes_attempt_id)
            if previous is None:
                raise ValueError("supersession must reference an earlier retained attempt")
            if attempt.supersedes_attempt_id in successors:
                raise ValueError("discovery history contains a supersession fork")
            if _subject_scope(previous) != _subject_scope(attempt):
                raise ValueError("supersession changed discovery subject scope")
            successors.add(attempt.supersedes_attempt_id)
        seen[attempt.attempt_id] = attempt


def _validate_record_set(attempts: tuple[DiscoveryAttempt, ...]) -> None:
    """Validate retained records without relying on shard collection order."""

    by_id: dict[str, DiscoveryAttempt] = {}
    successors: set[str] = set()
    for attempt in attempts:
        validate_discovery_attempt(attempt)
        if attempt.attempt_id in by_id:
            raise ValueError(f"duplicate discovery attempt: {attempt.attempt_id}")
        by_id[attempt.attempt_id] = attempt
    for attempt in attempts:
        previous_id = attempt.supersedes_attempt_id
        if not previous_id:
            continue
        previous = by_id.get(previous_id)
        if previous is None:
            raise ValueError("superseded discovery attempt is not retained")
        if previous_id in successors:
            raise ValueError("discovery records contain a supersession fork")
        if _subject_scope(previous) != _subject_scope(attempt):
            raise ValueError("supersession changed discovery subject scope")
        successors.add(previous_id)
    for attempt in attempts:
        visited: set[str] = set()
        current = attempt
        while current.supersedes_attempt_id:
            if current.attempt_id in visited:
                raise ValueError("discovery records contain a supersession cycle")
            visited.add(current.attempt_id)
            current = by_id[current.supersedes_attempt_id]


def _load_lines(lines: Iterable[str], *, source: str) -> tuple[DiscoveryAttempt, ...]:
    attempts: list[DiscoveryAttempt] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            raise ValueError(f"blank discovery ledger row: {source}:{line_number}")
        try:
            payload = json.loads(raw_line)
            if not isinstance(payload, Mapping):
                raise TypeError("row is not an object")
            values = dict(payload)
            for name in (
                "stable_coverage_ids",
                "journey_ids",
                "response_hashes",
                "decision_hashes",
                "input_classes",
                "sequence_ids",
                "factor_ids",
                "diagnostic_ids",
                "evidence_ids",
            ):
                values[name] = tuple(values.get(name) or ())
            attempt = DiscoveryAttempt(**values)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid discovery ledger row: {source}:{line_number}: {exc}") from exc
        attempts.append(attempt)
    result = tuple(attempts)
    _validate_history(result)
    return result


def _subject_scope(attempt: DiscoveryAttempt) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return attempt.stable_coverage_ids, attempt.journey_ids


def _subject_index(
    rows: tuple[DiscoveryAttempt, ...],
    active: tuple[DiscoveryAttempt, ...],
    field_name: str,
) -> dict[str, Any]:
    all_ids = sorted({item for row in rows for item in getattr(row, field_name)})
    active_ids = {row.attempt_id for row in active}
    result: dict[str, Any] = {}
    for subject_id in all_ids:
        subject_rows = tuple(
            row for row in rows if subject_id in getattr(row, field_name)
        )
        active_subject_rows = tuple(
            row for row in subject_rows if row.attempt_id in active_ids
        )
        latest = max(
            active_subject_rows or subject_rows,
            key=lambda row: (
                _timestamp(row.finished_at, "finished_at"),
                _timestamp(row.started_at, "started_at"),
                row.attempt_id,
            ),
        )
        result[subject_id] = {
            "attempt_ids": [row.attempt_id for row in subject_rows],
            "active_attempt_ids": [
                row.attempt_id
                for row in subject_rows
                if row.attempt_id in active_ids
            ],
            "latest_attempt_id": latest.attempt_id,
            "latest_revision": dict(latest.revision),
            "latest_classification": latest.classification,
            "latest_defect_id": latest.defect_id,
            "latest_repair_stage": latest.repair_stage,
            "personas": sorted({row.persona for row in subject_rows}),
            "input_classes": sorted({item for row in subject_rows for item in row.input_classes}),
            "sequence_ids": sorted({item for row in subject_rows for item in row.sequence_ids}),
            "factor_ids": sorted({item for row in subject_rows for item in row.factor_ids}),
            "rerun_count": max(0, len(subject_rows) - 1),
        }
    return result


def _counts(
    rows: tuple[DiscoveryAttempt, ...],
    field_name: str,
    allowed: frozenset[str],
) -> dict[str, int]:
    return {
        value: sum(getattr(row, field_name) == value for row in rows)
        for value in sorted(allowed)
    }


def _attempt_id(attempt: DiscoveryAttempt) -> str:
    payload = attempt_payload(replace(attempt, attempt_id=""))
    payload.pop("attempt_id", None)
    return _content_hash(payload)


def _normalized_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values)


def _validate_ids(name: str, values: tuple[str, ...]) -> None:
    if any(not value for value in values):
        raise ValueError(f"{name} contains an empty value")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate values")


def _validate_hashes(name: str, values: tuple[str, ...]) -> None:
    if any(not _is_sha256(value) for value in values):
        raise ValueError(f"{name} contains a non-SHA-256 value")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate values")


def _timestamp(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(character in "0123456789abcdef" for character in value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
