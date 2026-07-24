"""Bridge frozen G4 obligations to the response-driven Journey runtime.

This module is an adapter, not an executor.  It creates immutable Journey
definitions without future user turns and converts artifacts from an already
completed real-PTY Journey into the Phase 8 product-obligation evidence
contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.agent_live.chaos_scheduler import (
    build_journey_schedule,
    journey_schedule_payload,
    validate_journey_schedule,
)
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.dynamic_dual_ai_chaos import JOURNEY_EVIDENCE_SCHEMA_VERSION
from tests.agent_live.product_chaos_obligations import (
    validate_product_chaos_obligations,
)
from tests.agent_live.product_obligation_evidence import (
    PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION,
)


PRODUCT_CHAOS_JOURNEY_MANIFEST_SCHEMA_VERSION = 1
CHECKPOINT_DIFF_SCHEMA_VERSION = 1
DEFINITION_MANIFEST_TYPE = "product_chaos_journey_definition_manifest"
CHECKPOINT_DIFF_TYPE = "product_chaos_checkpoint_diffs"
JOURNEY_RUNNER = "dynamic_dual_ai_journey"
REAL_PTY_TRANSPORT = "real_pty"
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"

_SOURCE_CLASSIFICATION_TO_OUTCOME = {
    "passed": "passed",
    "product_failed": "failed",
    "simulator_invalid": "failed",
    "infrastructure_interrupted": "failed",
    "externally_blocked": "externally_blocked",
}


def load_frozen_product_chaos_catalog(
    path: str | Path,
) -> tuple[tuple[dict[str, Any], ...], dict[str, str]]:
    """Load and validate one complete, revision-bound G4 obligation catalog."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_rows = payload.get("obligations")
        declared_revision = payload.get("revision_binding")
    else:
        raw_rows = payload
        declared_revision = None
    if (
        not isinstance(raw_rows, Sequence)
        or isinstance(raw_rows, (str, bytes))
        or not raw_rows
        or any(not isinstance(row, Mapping) for row in raw_rows)
    ):
        raise ValueError("frozen G4 catalog must contain an obligations sequence")
    rows = tuple(dict(row) for row in raw_rows)
    revision = _obligation_revision(rows[0])
    if declared_revision is not None and dict(declared_revision) != revision:
        raise ValueError("frozen G4 catalog revision binding is inconsistent")
    validate_product_chaos_obligations(rows, revision=revision)
    return rows, revision


def build_product_chaos_journey_definition(
    obligation: Mapping[str, Any],
    *,
    revision: Mapping[str, str],
) -> dict[str, Any]:
    """Create one batch-compatible ``lane=journey`` definition."""

    row = _validated_obligation(obligation, revision=revision)
    simulator = dict(row["simulator_contract"])
    verifier = dict(row["verifier_contract"])
    factors = dict(row["factors"])
    journey = {
        "journey_id": row["obligation_id"],
        "start_scenario": dict(row["start_contract"])["scenario_id"],
        "persona": simulator["persona"],
        "mission": simulator["mission"],
        "allowed_risk_factors": [
            f"{name}:{value}" for name, value in sorted(factors.items())
        ],
        "max_turns": simulator["max_turns"],
        "terminal_outcome": {
            "outcome_id": f"{row['obligation_id']}-complete",
            "required_postcondition_ids": list(
                verifier["required_postcondition_ids"]
            ),
        },
        "forbidden_outcomes": [
            {
                "outcome_id": f"forbidden-{postcondition_id}",
                "required_postcondition_ids": [postcondition_id],
            }
            for postcondition_id in verifier["forbidden_postcondition_ids"]
        ],
    }
    definition = {
        "lane": "journey",
        "verifier_registry": verifier["registry_import"],
        "journey": journey,
    }
    schedule = build_journey_schedule(
        revision=revision,
        seed=int(row["seed"]),
        journey=journey,
    )
    validate_journey_schedule(schedule, revision=revision)
    _reject_future_turns(definition)
    return definition


def build_product_chaos_journey_manifest(
    obligations: Sequence[Mapping[str, Any]],
    *,
    revision: Mapping[str, str],
    obligation_id: str | None = None,
) -> dict[str, Any]:
    """Build a single/all definition manifest bound to the frozen contracts."""

    rows = tuple(dict(row) for row in obligations)
    validate_product_chaos_obligations(rows, revision=revision)
    selected = (
        tuple(row for row in rows if row["obligation_id"] == obligation_id)
        if obligation_id
        else rows
    )
    if obligation_id and not selected:
        raise ValueError(f"unknown G4 obligation: {obligation_id}")
    definitions: list[dict[str, Any]] = []
    for row in selected:
        definition = build_product_chaos_journey_definition(
            row,
            revision=revision,
        )
        schedule = build_journey_schedule(
            revision=revision,
            seed=int(row["seed"]),
            journey=definition["journey"],
        )
        definitions.append({
            "obligation_id": row["obligation_id"],
            "obligation_contract_hash": row["contract_hash"],
            "revision_binding": dict(revision),
            "definition": definition,
            "definition_hash": content_hash(definition),
            "schedule": journey_schedule_payload(schedule),
            "schedule_hash": content_hash(journey_schedule_payload(schedule)),
        })
    unsigned = {
        "schema_version": PRODUCT_CHAOS_JOURNEY_MANIFEST_SCHEMA_VERSION,
        "artifact_type": DEFINITION_MANIFEST_TYPE,
        "revision_binding": dict(revision),
        "catalog_hash": content_hash(rows),
        "selection": "single" if obligation_id else "all",
        "definition_count": len(definitions),
        "generation_is_execution": False,
        "prewritten_future_turns": False,
        "definitions": definitions,
    }
    manifest = {**unsigned, "manifest_hash": content_hash(unsigned)}
    _reject_future_turns(manifest)
    return manifest


def write_product_chaos_journey_manifest(
    manifest: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Write a manifest once; an existing output is never overwritten."""

    return _write_json_once(Path(path).resolve(), dict(manifest))


def convert_completed_journey_to_product_evidence(
    obligation: Mapping[str, Any],
    *,
    revision: Mapping[str, str],
    runtime_root: str | Path,
    evidence_path: str | Path,
    checkpoint_diff_path: str | Path | None = None,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> Path:
    """Convert one completed Journey run into product-obligation evidence.

    The converter does not execute the Journey and does not upgrade an
    incomplete run.  It validates and binds the existing real-PTY artifacts.
    """

    row = _validated_obligation(obligation, revision=revision)
    if provider != DEFAULT_PROVIDER or not str(model).strip():
        raise ValueError("G4 product evidence requires an explicit DeepSeek provider/model")
    root = Path(runtime_root).resolve()
    output = Path(evidence_path).resolve()
    diff_output = Path(
        checkpoint_diff_path
        or output.with_name(f"{output.stem}-checkpoint-diffs.json")
    ).resolve()
    source_paths = {
        "journey_result": root / "journey-result.json",
        "journey_schedule": root / "journey-schedule.json",
        "transcript": root / "transcript.txt",
        "runtime_events": root / "turn-events.jsonl",
        "checkpoint_store": root / "checkpoints.sqlite",
    }
    for role, source_path in source_paths.items():
        _require_nonempty_file(source_path, role=role)

    result = _load_mapping(source_paths["journey_result"], "journey result")
    source_evidence_path = _resolve_source_evidence_path(
        root,
        str(result.get("evidence_path") or ""),
    )
    _require_nonempty_file(source_evidence_path, role="journey_evidence")
    source_evidence = _load_mapping(source_evidence_path, "Journey evidence")
    source_schedule = _load_mapping(
        source_paths["journey_schedule"],
        "Journey schedule",
    )

    definition = build_product_chaos_journey_definition(row, revision=revision)
    expected_schedule = build_journey_schedule(
        revision=revision,
        seed=int(row["seed"]),
        journey=definition["journey"],
    )
    expected_schedule_payload = journey_schedule_payload(expected_schedule)
    if source_schedule != expected_schedule_payload:
        raise ValueError("completed Journey schedule does not match the G4 obligation")
    _validate_source_journey(
        result=result,
        source_evidence=source_evidence,
        schedule_payload=expected_schedule_payload,
        obligation=row,
        revision=revision,
        provider=provider,
        model=model,
    )

    events = _load_runtime_events(source_paths["runtime_events"])
    checkpoint_diff = _build_checkpoint_diff_artifact(
        obligation=row,
        revision=revision,
        result=result,
        source_evidence=source_evidence,
        events=events,
        source_paths=source_paths,
    )
    artifact_sources = {
        **source_paths,
        "journey_evidence": source_evidence_path,
    }
    artifact_descriptors = [
        _artifact_descriptor(role, path)
        for role, path in artifact_sources.items()
    ]

    if output.exists() or diff_output.exists():
        raise FileExistsError("product evidence outputs are immutable")
    artifact_descriptors.append({
        "role": "checkpoint_diff",
        "path": str(diff_output),
        "sha256": hashlib.sha256(_json_bytes(checkpoint_diff)).hexdigest(),
    })
    artifact_hashes = [item["sha256"] for item in artifact_descriptors]

    classification = str(result.get("terminal_classification") or "")
    outcome = _SOURCE_CLASSIFICATION_TO_OUTCOME[classification]
    verifier_results = _product_verifier_results(
        obligation=row,
        source_evidence=source_evidence,
        classification=classification,
        outcome=outcome,
        artifact_hashes=artifact_hashes,
    )
    started_at, finished_at = _execution_interval(
        source_evidence,
        tuple(artifact_sources.values()),
    )
    execution = {
        "execution_id": content_hash({
            "journey_evidence_id": source_evidence["evidence_id"],
            "session_id": source_evidence["session_id"],
            "obligation_id": row["obligation_id"],
        }),
        "runner": JOURNEY_RUNNER,
        "transport": REAL_PTY_TRANSPORT,
        "provider": provider,
        "model": model,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    identity = {
        "obligation_id": row["obligation_id"],
        "obligation_contract_hash": row["contract_hash"],
        "revision_binding": dict(revision),
        "execution_id": execution["execution_id"],
        "artifact_sha256s": sorted(artifact_hashes),
    }
    unsigned = {
        "schema_version": PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION,
        "evidence_id": content_hash(identity),
        "obligation_id": row["obligation_id"],
        "obligation_contract_hash": row["contract_hash"],
        "revision_binding": dict(revision),
        "outcome": outcome,
        "execution": execution,
        "artifacts": artifact_descriptors,
        "verifier_results": verifier_results,
    }
    payload = {**unsigned, "evidence_hash": content_hash(unsigned)}
    _write_json_once(diff_output, checkpoint_diff)
    try:
        return _write_json_once(output, payload)
    except Exception:
        diff_output.unlink(missing_ok=True)
        raise


def _validated_obligation(
    obligation: Mapping[str, Any],
    *,
    revision: Mapping[str, str],
) -> dict[str, Any]:
    row = dict(obligation)
    if _obligation_revision(row) != dict(revision):
        raise ValueError("G4 obligation revision does not match the active revision")
    unsigned = dict(row)
    recorded_hash = str(unsigned.pop("contract_hash", "") or "")
    if not recorded_hash or recorded_hash != content_hash(unsigned):
        raise ValueError("G4 obligation contract hash is invalid")
    if row.get("status") != "not_run":
        raise ValueError("G4 obligation catalog cannot claim execution")
    for field in (
        "obligation_id",
        "seed",
        "factors",
        "start_contract",
        "simulator_contract",
        "verifier_contract",
    ):
        if field not in row:
            raise ValueError(f"G4 obligation is missing {field}")
    simulator = dict(row["simulator_contract"])
    if (
        simulator.get("selection_mode") != "response_driven"
        or simulator.get("actor") != "codex_as_user"
        or simulator.get("prewritten_future_turns_forbidden") is not True
        or simulator.get("read_complete_agent_response_before_each_turn") is not True
    ):
        raise ValueError("G4 obligation is not response-driven")
    return row


def _validate_source_journey(
    *,
    result: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    schedule_payload: Mapping[str, Any],
    obligation: Mapping[str, Any],
    revision: Mapping[str, str],
    provider: str,
    model: str,
) -> None:
    evidence_unsigned = dict(source_evidence)
    artifact_hash = str(evidence_unsigned.pop("artifact_hash", "") or "")
    if not artifact_hash or artifact_hash != content_hash(evidence_unsigned):
        raise ValueError("source Journey evidence artifact hash is invalid")
    evidence_id = str(evidence_unsigned.pop("evidence_id", "") or "")
    if not evidence_id or evidence_id != content_hash(evidence_unsigned):
        raise ValueError("source Journey evidence identity is invalid")
    if source_evidence.get("artifact_type") != "dynamic_dual_ai_journey_evidence":
        raise ValueError("source artifact is not Journey runtime evidence")
    if source_evidence.get("schema_version") != JOURNEY_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("source Journey evidence schema is unsupported")
    if source_evidence.get("runner_type") != JOURNEY_RUNNER:
        raise ValueError("source Journey did not use the response-driven runner")
    if source_evidence.get("content_redacted") is not True:
        raise ValueError("source Journey transcript was not redacted")
    if dict(source_evidence.get("revision") or {}) != dict(revision):
        raise ValueError("source Journey revision is stale")
    if source_evidence.get("journey_id") != obligation["obligation_id"]:
        raise ValueError("source Journey targets a different obligation")
    if source_evidence.get("schedule_id") != schedule_payload["schedule_id"]:
        raise ValueError("source Journey schedule id is stale")
    if source_evidence.get("schedule_hash") != content_hash(schedule_payload):
        raise ValueError("source Journey schedule hash is stale")
    verifier = dict(obligation["verifier_contract"])
    registry = dict(source_evidence.get("verifier_registry") or {})
    if registry.get("registry_id") != verifier["registry_id"]:
        raise ValueError("source Journey verifier registry is stale")
    if source_evidence.get("provider") != provider or source_evidence.get("model") != model:
        raise ValueError("source Journey provider/model does not match DeepSeek")

    classification = str(source_evidence.get("terminal_classification") or "")
    if classification not in _SOURCE_CLASSIFICATION_TO_OUTCOME:
        raise ValueError(f"unsupported Journey terminal classification: {classification}")
    if result.get("terminal_classification") != classification:
        raise ValueError("Journey result and evidence classifications differ")
    if result.get("schema_version") != 1:
        raise ValueError("Journey result schema is unsupported")
    if result.get("execution_status") != classification:
        raise ValueError("Journey result execution status is inconsistent")
    if result.get("journey_id") != obligation["obligation_id"]:
        raise ValueError("Journey result targets a different obligation")
    if dict(result.get("revision") or {}) != dict(revision):
        raise ValueError("Journey result revision is stale")
    if result.get("schedule_id") != schedule_payload["schedule_id"]:
        raise ValueError("Journey result schedule id is stale")
    if result.get("evidence_id") != source_evidence["evidence_id"]:
        raise ValueError("Journey result points to different evidence")
    if result.get("verifier_registry_id") != verifier["registry_id"]:
        raise ValueError("Journey result verifier registry is stale")
    if result.get("turns") != source_evidence.get("turns"):
        raise ValueError("Journey result and evidence turn records differ")
    _validate_response_bound_turns(source_evidence.get("turns") or ())
    completed = int(source_evidence.get("completed_turn_count", -1))
    if completed != len(source_evidence.get("turns") or ()):
        raise ValueError("Journey completed turn count is inconsistent")
    if int(result.get("completed_turn_count", -1)) != completed:
        raise ValueError("Journey result completed turn count is inconsistent")
    qualifies = classification == "passed"
    if source_evidence.get("qualifying_evidence") is not qualifies:
        raise ValueError("Journey qualifying-evidence flag contradicts its classification")
    if result.get("qualifying_evidence") is not qualifies:
        raise ValueError("Journey result qualifying-evidence flag is inconsistent")


def _validate_response_bound_turns(turns: Sequence[Any]) -> None:
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise ValueError("Journey turn evidence is invalid")
        provenance = turn.get("decision_provenance")
        identity = turn.get("turn_identity")
        if not isinstance(provenance, Mapping) or not isinstance(identity, Mapping):
            raise ValueError("Journey turn lacks response-bound decision provenance")
        for field in ("previous_response_hash", "user_message_hash"):
            value = str(provenance.get(field) or "")
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("Journey decision provenance hash is invalid")
        selected_at_ns = provenance.get("selected_at_ns")
        submitted_at_ns = provenance.get("submitted_at_ns")
        if (
            isinstance(selected_at_ns, bool)
            or not isinstance(selected_at_ns, int)
            or isinstance(submitted_at_ns, bool)
            or not isinstance(submitted_at_ns, int)
            or selected_at_ns <= 0
            or submitted_at_ns < selected_at_ns
            or selected_at_ns != turn.get("selected_at_ns")
            or submitted_at_ns != identity.get("user_message_submitted_at_ns")
        ):
            raise ValueError("Journey decision provenance timing is invalid")


def _build_checkpoint_diff_artifact(
    *,
    obligation: Mapping[str, Any],
    revision: Mapping[str, str],
    result: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    session_id = str(source_evidence.get("session_id") or "")
    if not session_id:
        raise ValueError("source Journey has no session identity")
    indexed: dict[int, Mapping[str, Any]] = {}
    normalized_events: list[dict[str, Any]] = []
    for event in events:
        if dict(event.get("revision") or {}) != dict(revision):
            raise ValueError("runtime event revision is stale")
        if event.get("thread_id") != session_id:
            raise ValueError("runtime event belongs to another Journey session")
        turn_index = int(event.get("turn_index", -1))
        if turn_index in indexed:
            raise ValueError("runtime event stream contains duplicate turn indexes")
        indexed[turn_index] = event
        normalized_events.append({
            "turn_index": turn_index,
            "event_type": str(event.get("event_type") or ""),
            "before_fingerprint": str(event.get("before_fingerprint") or ""),
            "after_fingerprint": str(event.get("after_fingerprint") or ""),
            "active_group": str(event.get("active_group") or ""),
            "pending_question_id": str(event.get("pending_question_id") or ""),
            "admitted_action_types": list(event.get("admitted_action_types") or ()),
            "state_diff_hashes": dict(event.get("state_diff_hashes") or {}),
            "after_value_hashes": dict(event.get("after_value_hashes") or {}),
        })
    for turn in result.get("turns") or ():
        turn_index = int(turn.get("turn_index", -1))
        event = indexed.get(turn_index)
        identity = dict(turn.get("turn_identity") or {})
        if event is None:
            raise ValueError("Journey result turn has no matching runtime event")
        if (
            identity.get("before_fingerprint") != event.get("before_fingerprint")
            or identity.get("after_fingerprint") != event.get("after_fingerprint")
        ):
            raise ValueError("Journey result and runtime event fingerprints differ")
    unsigned = {
        "schema_version": CHECKPOINT_DIFF_SCHEMA_VERSION,
        "artifact_type": CHECKPOINT_DIFF_TYPE,
        "obligation_id": obligation["obligation_id"],
        "obligation_contract_hash": obligation["contract_hash"],
        "revision_binding": dict(revision),
        "session_id": session_id,
        "schedule_id": result["schedule_id"],
        "checkpoint_store_sha256": _sha256_file(source_paths["checkpoint_store"]),
        "runtime_events_sha256": _sha256_file(source_paths["runtime_events"]),
        "events": normalized_events,
    }
    return {**unsigned, "artifact_hash": content_hash(unsigned)}


def _product_verifier_results(
    *,
    obligation: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    classification: str,
    outcome: str,
    artifact_hashes: Sequence[str],
) -> list[dict[str, Any]]:
    verifier = dict(obligation["verifier_contract"])
    required = tuple(verifier["required_postcondition_ids"])
    forbidden = tuple(verifier["forbidden_postcondition_ids"])
    observations = _latest_postcondition_observations(source_evidence)
    results: list[dict[str, Any]] = []
    for postcondition_id in (*required, *forbidden):
        observed = observations.get(postcondition_id)
        if outcome == "externally_blocked":
            status = "externally_blocked"
        elif observed is None:
            status = "failed"
        elif postcondition_id in forbidden:
            status = "failed" if observed["satisfied"] else "passed"
        else:
            status = "passed" if observed["satisfied"] else "failed"
        details = json.dumps({
            "source_classification": classification,
            "postcondition_id": postcondition_id,
            "observation": observed,
        }, ensure_ascii=False, sort_keys=True)
        results.append({
            "verifier_id": postcondition_id,
            "status": status,
            "details": details,
            "evidence_sha256s": list(artifact_hashes),
        })
    if outcome == "failed" and all(item["status"] == "passed" for item in results):
        results[0]["status"] = "failed"
        results[0]["details"] = json.dumps({
            "source_classification": classification,
            "reason": "Journey failed outside a satisfied terminal verifier",
        }, sort_keys=True)
    return results


def _latest_postcondition_observations(
    source_evidence: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    frames = [source_evidence.get("initial_verification") or {}]
    frames.extend(source_evidence.get("turns") or ())
    observed: dict[str, dict[str, Any]] = {}
    for frame in frames:
        outcomes = [frame.get("terminal_outcome")]
        outcomes.extend(frame.get("forbidden_outcomes") or ())
        for outcome in outcomes:
            if not isinstance(outcome, Mapping):
                continue
            for item in outcome.get("postconditions") or ():
                if not isinstance(item, Mapping):
                    continue
                postcondition_id = str(item.get("postcondition_id") or "")
                if postcondition_id:
                    observed[postcondition_id] = {
                        "satisfied": item.get("satisfied") is True,
                        "details": item.get("details"),
                        "verifier": item.get("verifier"),
                    }
    return observed


def _execution_interval(
    source_evidence: Mapping[str, Any],
    source_paths: Sequence[Path],
) -> tuple[str, str]:
    timestamps: list[int] = []
    for turn in source_evidence.get("turns") or ():
        identity = dict(turn.get("turn_identity") or {})
        for field in (
            "previous_response_received_at_ns",
            "user_message_submitted_at_ns",
            "agent_response_received_at_ns",
        ):
            value = identity.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                timestamps.append(value)
    if not timestamps:
        timestamps = [path.stat().st_mtime_ns for path in source_paths]
    started_ns = min(timestamps)
    finished_ns = max(timestamps)
    if started_ns >= finished_ns:
        raise ValueError("Journey artifacts do not prove a non-empty execution interval")
    return _iso_from_ns(started_ns), _iso_from_ns(finished_ns)


def _load_runtime_events(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"runtime event line {line_number} is not an object")
        required = {
            "event_type",
            "thread_id",
            "before_fingerprint",
            "after_fingerprint",
            "turn_index",
            "revision",
            "state_diff_hashes",
        }
        if not required.issubset(payload):
            raise ValueError(f"runtime event line {line_number} is incomplete")
        rows.append(dict(payload))
    if not rows:
        raise ValueError("runtime event stream is empty")
    return tuple(rows)


def _artifact_descriptor(role: str, path: Path) -> dict[str, str]:
    _require_nonempty_file(path, role=role)
    return {"role": role, "path": str(path.resolve()), "sha256": _sha256_file(path)}


def _resolve_source_evidence_path(root: Path, raw_path: str) -> Path:
    if not raw_path:
        raise ValueError("Journey result has no evidence path")
    declared = Path(raw_path).expanduser()
    candidates = (
        declared,
        root / "evidence" / declared.name,
    )
    matches = tuple(path.resolve() for path in candidates if path.is_file())
    if not matches:
        raise ValueError("Journey result evidence path cannot be resolved")
    if len(set(matches)) != 1:
        raise ValueError("Journey result evidence path is ambiguous")
    return matches[0]


def _reject_future_turns(value: Any) -> None:
    forbidden_keys = {"turns", "user_message", "future_turns", "messages"}
    if isinstance(value, Mapping):
        leaked = forbidden_keys & set(value)
        if leaked:
            raise ValueError(
                "Journey definition contains prewritten future dialogue: "
                + ", ".join(sorted(leaked))
            )
        for nested in value.values():
            _reject_future_turns(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _reject_future_turns(nested)


def _obligation_revision(obligation: Mapping[str, Any]) -> dict[str, str]:
    binding = obligation.get("revision_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("G4 obligation has no revision binding")
    revision = {
        "commit": str(binding.get("commit") or "").strip(),
        "worktree_hash": str(binding.get("worktree_hash") or "").strip(),
    }
    if not all(revision.values()):
        raise ValueError("G4 obligation revision binding is incomplete")
    return revision


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(payload)


def _require_nonempty_file(path: Path, *, role: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"required {role} artifact is missing or empty: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).isoformat()


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adapt frozen G4 obligations to response-driven Journey artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    definitions = subparsers.add_parser(
        "definitions",
        help="Generate a single/all immutable Journey definition manifest.",
    )
    definitions.add_argument("--catalog", required=True, type=Path)
    definitions.add_argument("--output", required=True, type=Path)
    definitions.add_argument("--obligation-id")

    evidence = subparsers.add_parser(
        "evidence",
        help="Convert an already completed Journey runtime into G4 evidence.",
    )
    evidence.add_argument("--catalog", required=True, type=Path)
    evidence.add_argument("--obligation-id", required=True)
    evidence.add_argument("--runtime-root", required=True, type=Path)
    evidence.add_argument("--output", required=True, type=Path)
    evidence.add_argument("--checkpoint-diff", type=Path)
    evidence.add_argument("--provider", default=DEFAULT_PROVIDER)
    evidence.add_argument("--model", default=DEFAULT_MODEL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows, revision = load_frozen_product_chaos_catalog(args.catalog)
    if args.command == "definitions":
        manifest = build_product_chaos_journey_manifest(
            rows,
            revision=revision,
            obligation_id=args.obligation_id,
        )
        write_product_chaos_journey_manifest(manifest, args.output)
        return 0

    matches = tuple(
        row for row in rows if row["obligation_id"] == args.obligation_id
    )
    if len(matches) != 1:
        raise ValueError(f"unknown G4 obligation: {args.obligation_id}")
    convert_completed_journey_to_product_evidence(
        matches[0],
        revision=revision,
        runtime_root=args.runtime_root,
        evidence_path=args.output,
        checkpoint_diff_path=args.checkpoint_diff,
        provider=args.provider,
        model=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
