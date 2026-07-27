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
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.agent_live.chaos_scheduler import (
    build_journey_schedule,
    journey_schedule_payload,
    validate_journey_schedule,
)
from tests.agent_live.batch_orchestrator import (
    TimeoutPolicy,
    freeze_batch_manifest,
)
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.coverage_evidence import PtyCliTurnRecord, RuntimeTurnEvent
from tests.agent_live.codex_simulator_bridge import validate_simulator_attestation
from tests.agent_live.completed_journey_batch import (
    G4_ARTIFACT_TYPE,
    convert_completed_journey_batch,
)
from tests.agent_live.container_process_guard import (
    validate_cleanup_receipt_artifact,
)
from tests.agent_live.dynamic_dual_ai_chaos import (
    JOURNEY_EVIDENCE_SCHEMA_VERSION,
    JourneyDecisionProvenance,
    JourneyVerifierContext,
    journey_outcome_verifier_registry_payload,
)
from tests.agent_live.formal_journey_catalog import (
    FORMAL_JOURNEY_VERIFIER_REGISTRY,
)
from tests.agent_live.product_chaos_obligations import (
    build_product_chaos_obligations,
    validate_product_chaos_obligations,
)
from tests.agent_live.product_obligation_evidence import (
    PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION,
)
from tests.agent_live.runtime_checkpoint import (
    reviewed_scenario,
    seed_runtime_checkpoint,
)


PRODUCT_CHAOS_JOURNEY_MANIFEST_SCHEMA_VERSION = 1
PRODUCT_CHAOS_TARGET_SET_SCHEMA_VERSION = 1
CHECKPOINT_DIFF_SCHEMA_VERSION = 1
DEFINITION_MANIFEST_TYPE = "product_chaos_journey_definition_manifest"
CHECKPOINT_DIFF_TYPE = "product_chaos_checkpoint_diffs"
JOURNEY_RUNNER = "dynamic_dual_ai_journey"
REAL_PTY_TRANSPORT = "real_pty"
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_REQUIRED_ENV_NAMES = ("DEEPSEEK_API_KEY",)

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
        "subject_group": factors["subject_group"],
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


def build_product_chaos_target_payloads(
    definition_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Translate frozen definitions without changing obligation schedules."""

    definitions = definition_manifest.get("definitions")
    revision = definition_manifest.get("revision_binding")
    if (
        definition_manifest.get("artifact_type") != DEFINITION_MANIFEST_TYPE
        or not isinstance(definitions, Sequence)
        or isinstance(definitions, (str, bytes))
        or not definitions
        or not isinstance(revision, Mapping)
    ):
        raise ValueError("G4 definition manifest is incomplete")
    unsigned = {
        key: value for key, value in definition_manifest.items()
        if key != "manifest_hash"
    }
    if definition_manifest.get("manifest_hash") != content_hash(unsigned):
        raise ValueError("G4 definition manifest identity is stale")
    payloads: list[dict[str, Any]] = []
    for row in definitions:
        if not isinstance(row, Mapping):
            raise ValueError("G4 definition row is invalid")
        definition = row.get("definition")
        schedule = row.get("schedule")
        if not isinstance(definition, Mapping) or not isinstance(schedule, Mapping):
            raise ValueError("G4 definition row lacks definition or schedule")
        if row.get("definition_hash") != content_hash(definition):
            raise ValueError("G4 definition hash is stale")
        if row.get("schedule_hash") != content_hash(schedule):
            raise ValueError("G4 schedule hash is stale")
        payload = {
            **dict(definition),
            "frozen_execution": {
                "obligation_id": str(row.get("obligation_id") or ""),
                "seed": int(schedule["seed"]),
                "schedule_id": str(schedule["schedule_id"]),
                "schedule_hash": str(row["schedule_hash"]),
                "subject_group": str(schedule["subject_group"]),
                "revision_binding": dict(revision),
            },
            "simulator_attestation_contract": {
                "required": True,
                "identity_strength": "auditable_declaration_only",
                "scripted_actor_qualifies": False,
                "cryptographic_identity_claimed": False,
            },
        }
        _reject_future_turns(payload)
        payloads.append(payload)
    return tuple(payloads)


def write_product_chaos_target_set(
    definition_manifest: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write immutable batch targets and their own revision-bound manifest."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"G4 target directory is immutable: {destination}")
    payloads = build_product_chaos_target_payloads(definition_manifest)
    revision = dict(definition_manifest["revision_binding"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-staging-",
        dir=destination.parent,
    ) as temporary:
        staging = Path(temporary)
        preflight_dir = staging / "preflight"
        preflight_dir.mkdir()
        scenario_preflights: dict[str, dict[str, Any]] = {}
        scenario_ids = sorted({
            str(payload["journey"]["start_scenario"])
            for payload in payloads
        })
        for index, scenario_id in enumerate(scenario_ids, start=1):
            scenario = reviewed_scenario(scenario_id)
            relative_checkpoint = (
                Path("preflight")
                / f"{index:02d}-{content_hash(scenario_id)[:16]}.sqlite"
            )
            receipt = seed_runtime_checkpoint(
                scenario.seed_state,
                checkpoint_path=staging / relative_checkpoint,
                session_id=f"g4-preflight-{content_hash(scenario_id)[:16]}",
                session_purpose="product-chaos-reachability-preflight",
                scenario_id=scenario_id,
                scenario_state_fingerprint=scenario.state_fingerprint,
            )
            normalized_receipt = {
                **receipt.payload,
                "checkpoint_path": str(relative_checkpoint),
            }
            scenario_preflights[scenario_id] = {
                "scenario_id": scenario_id,
                "scenario_state_fingerprint": scenario.state_fingerprint,
                "checkpoint_path": str(relative_checkpoint),
                "checkpoint_sha256": receipt.checkpoint_sha256,
                "seed_receipt": normalized_receipt,
                "seed_receipt_hash": content_hash(normalized_receipt),
            }
        targets: list[dict[str, Any]] = []
        for index, payload in enumerate(payloads, start=1):
            name = f"{index:02d}.json"
            encoded = _json_bytes(payload)
            (staging / name).write_bytes(encoded)
            frozen = payload["frozen_execution"]
            scenario_id = str(payload["journey"]["start_scenario"])
            scenario_preflight = scenario_preflights[scenario_id]
            preflight_id = content_hash({
                "obligation_id": frozen["obligation_id"],
                "schedule_id": frozen["schedule_id"],
                "subject_group": frozen["subject_group"],
                "scenario_id": scenario_id,
                "seed_receipt_hash": scenario_preflight["seed_receipt_hash"],
            })
            targets.append({
                "index": index,
                "obligation_id": frozen["obligation_id"],
                "seed": frozen["seed"],
                "schedule_id": frozen["schedule_id"],
                "subject_group": frozen["subject_group"],
                "scenario_id": scenario_id,
                "preflight_id": preflight_id,
                "path": name,
                "sha256": hashlib.sha256(encoded).hexdigest(),
            })
        obligation_set = [
            {
                "obligation_id": row["obligation_id"],
                "schedule_id": row["schedule_id"],
                "seed": row["seed"],
                "subject_group": row["subject_group"],
            }
            for row in sorted(
                targets,
                key=lambda item: (
                    item["obligation_id"],
                    item["schedule_id"],
                ),
            )
        ]
        unsigned = {
            "schema_version": PRODUCT_CHAOS_TARGET_SET_SCHEMA_VERSION,
            "artifact_type": "product_chaos_target_set",
            "revision_binding": revision,
            "definition_manifest_hash": definition_manifest["manifest_hash"],
            "definition_selection": definition_manifest["selection"],
            "expected_obligation_set_hash": content_hash(
                obligation_set
            ),
            "coverage_denominators": {
                "generated": len(targets),
                "executable": len(targets),
                "qualifying": 0,
            },
            "reachability_preflight_complete": True,
            "scenario_preflight_count": len(scenario_preflights),
            "scenario_preflights": [
                scenario_preflights[key]
                for key in sorted(scenario_preflights)
            ],
            "target_count": len(targets),
            "targets": targets,
        }
        manifest = {**unsigned, "manifest_hash": content_hash(unsigned)}
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(staging, destination)
    return destination / "manifest.json"


def freeze_product_chaos_batch(
    *,
    repo_root: str | Path,
    targets_dir: str | Path,
    manifest_path: str | Path,
    runtime_base: str | Path,
    worker_runtime: str = "linux",
    max_concurrency: int | None = None,
    required_env_names: Sequence[str] = DEFAULT_REQUIRED_ENV_NAMES,
    timeout_policy: TimeoutPolicy = TimeoutPolicy(),
    environment: Mapping[str, str] | None = None,
) -> Any:
    """Freeze one G4 batch while preserving every target's own schedule seed."""

    normalized_required_env_names = _require_execution_environment(
        required_env_names,
        environment=environment,
    )
    target_root = Path(targets_dir).resolve()
    target_manifest = _load_mapping(
        target_root / "manifest.json",
        "G4 target-set manifest",
    )
    if target_manifest.get("artifact_type") != "product_chaos_target_set":
        raise ValueError("target directory is not a G4 product target set")
    if target_manifest.get("definition_selection") != "all":
        raise ValueError(
            "G4 batch requires the complete obligation target set"
        )
    unsigned = {
        key: value for key, value in target_manifest.items()
        if key != "manifest_hash"
    }
    if target_manifest.get("manifest_hash") != content_hash(unsigned):
        raise ValueError("G4 target-set manifest identity is stale")
    targets = target_manifest.get("targets")
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise ValueError("G4 target-set manifest has no targets")
    if target_manifest.get("target_count") != len(targets):
        raise ValueError("G4 target-set manifest count is stale")
    preflight_rows = target_manifest.get("scenario_preflights")
    if (
        target_manifest.get("reachability_preflight_complete") is not True
        or not isinstance(preflight_rows, list)
        or target_manifest.get("scenario_preflight_count")
        != len(preflight_rows)
    ):
        raise ValueError("G4 target-set reachability preflight is incomplete")
    scenario_preflights: dict[str, dict[str, Any]] = {}
    for row in preflight_rows:
        if not isinstance(row, Mapping):
            raise ValueError("G4 scenario preflight row is invalid")
        scenario_id = str(row.get("scenario_id") or "")
        receipt = row.get("seed_receipt")
        checkpoint = target_root / str(row.get("checkpoint_path") or "")
        if (
            not scenario_id
            or scenario_id in scenario_preflights
            or not isinstance(receipt, Mapping)
            or content_hash(receipt) != row.get("seed_receipt_hash")
            or receipt.get("scenario_id") != scenario_id
            or receipt.get("checkpoint_path") != row.get("checkpoint_path")
            or receipt.get("checkpoint_sha256") != row.get("checkpoint_sha256")
            or _sha256_file(checkpoint) != row.get("checkpoint_sha256")
        ):
            raise ValueError("G4 scenario reachability receipt is invalid")
        scenario = reviewed_scenario(scenario_id)
        if (
            row.get("scenario_state_fingerprint")
            != scenario.state_fingerprint
            or receipt.get("scenario_state_fingerprint")
            != scenario.state_fingerprint
        ):
            raise ValueError("G4 scenario reachability fingerprint is stale")
        scenario_preflights[scenario_id] = dict(row)
    for row in targets:
        if not isinstance(row, Mapping):
            raise ValueError("G4 target-set row is invalid")
        path = target_root / str(row.get("path") or "")
        if _sha256_file(path) != str(row.get("sha256") or ""):
            raise ValueError("G4 target file hash is stale")
        payload = _load_mapping(path, "G4 target")
        frozen = dict(payload.get("frozen_execution") or {})
        scenario_id = str((payload.get("journey") or {}).get("start_scenario") or "")
        preflight = scenario_preflights.get(scenario_id)
        if {
            "obligation_id": frozen.get("obligation_id"),
            "schedule_id": frozen.get("schedule_id"),
            "seed": frozen.get("seed"),
            "subject_group": frozen.get("subject_group"),
        } != {
            "obligation_id": row.get("obligation_id"),
            "schedule_id": row.get("schedule_id"),
            "seed": row.get("seed"),
            "subject_group": row.get("subject_group"),
        }:
            raise ValueError("G4 target row differs from its frozen execution")
        expected_preflight_id = content_hash({
            "obligation_id": frozen.get("obligation_id"),
            "schedule_id": frozen.get("schedule_id"),
            "subject_group": frozen.get("subject_group"),
            "scenario_id": scenario_id,
            "seed_receipt_hash": (
                preflight.get("seed_receipt_hash") if preflight else ""
            ),
        })
        if (
            preflight is None
            or row.get("scenario_id") != scenario_id
            or row.get("preflight_id") != expected_preflight_id
        ):
            raise ValueError("G4 target has no matching reachability preflight")
    obligation_set = [
        {
            "obligation_id": str(row.get("obligation_id") or ""),
            "schedule_id": str(row.get("schedule_id") or ""),
            "seed": int(row.get("seed") or 0),
            "subject_group": str(row.get("subject_group") or ""),
        }
        for row in sorted(
            targets,
            key=lambda item: (
                str(item.get("obligation_id") or ""),
                str(item.get("schedule_id") or ""),
            ),
        )
    ]
    expected_set_hash = str(
        target_manifest.get("expected_obligation_set_hash") or ""
    )
    if (
        not expected_set_hash
        or content_hash(obligation_set) != expected_set_hash
        or len({
            (
                row["obligation_id"],
                row["schedule_id"],
                row["seed"],
                row["subject_group"],
            )
            for row in obligation_set
        })
        != len(obligation_set)
    ):
        raise ValueError("G4 target obligation set is incomplete or duplicated")
    revision = dict(target_manifest["revision_binding"])
    authoritative_manifest = build_product_chaos_journey_manifest(
        build_product_chaos_obligations(revision=revision),
        revision=revision,
    )
    authoritative_payloads = {
        payload["frozen_execution"]["obligation_id"]: payload
        for payload in build_product_chaos_target_payloads(
            authoritative_manifest
        )
    }
    authoritative_set = [
        {
            "obligation_id": row["obligation_id"],
            "schedule_id": row["schedule"]["schedule_id"],
            "seed": row["schedule"]["seed"],
            "subject_group": row["schedule"]["subject_group"],
        }
        for row in sorted(
            authoritative_manifest["definitions"],
            key=lambda item: (
                item["obligation_id"],
                item["schedule"]["schedule_id"],
            ),
        )
    ]
    if (
        target_manifest.get("definition_manifest_hash")
        != authoritative_manifest["manifest_hash"]
        or obligation_set != authoritative_set
        or expected_set_hash != content_hash(authoritative_set)
    ):
        raise ValueError(
            "G4 target obligation set differs from the authoritative catalog"
        )
    for row in targets:
        payload = _load_mapping(
            target_root / str(row["path"]),
            "G4 target",
        )
        expected_payload = authoritative_payloads.get(
            str(row["obligation_id"])
        )
        if expected_payload is None or payload != expected_payload:
            raise ValueError(
                "G4 target payload differs from the authoritative catalog"
            )
    if target_manifest.get("coverage_denominators") != {
        "generated": len(authoritative_set),
        "executable": len(authoritative_set),
        "qualifying": 0,
    }:
        raise ValueError("G4 target coverage denominators are inconsistent")
    return freeze_batch_manifest(
        repo_root=repo_root,
        targets_dir=target_root,
        manifest_path=manifest_path,
        runtime_base=runtime_base,
        shard_count=len(targets),
        max_concurrency=max_concurrency,
        expected_revision=revision,
        required_env_names=normalized_required_env_names,
        timeout_policy=timeout_policy,
        worker_runtime=worker_runtime,
        expected_obligation_set_hash=expected_set_hash,
    )


def _require_execution_environment(
    required_env_names: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Fail before freezing when a required provider credential is unavailable."""

    names = tuple(sorted({
        str(name).strip()
        for name in required_env_names
        if str(name).strip()
    }))
    if not names:
        raise ValueError("G4 batch requires at least one required environment name")
    active_environment = os.environ if environment is None else environment
    missing = tuple(
        name
        for name in names
        if not str(active_environment.get(name) or "").strip()
    )
    if missing:
        raise RuntimeError(
            "G4 batch required environment is missing or empty: "
            + ", ".join(missing)
        )
    return names


def convert_completed_journey_to_product_evidence(
    obligation: Mapping[str, Any],
    *,
    revision: Mapping[str, str],
    round_id: str,
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
    normalized_round_id = str(round_id).strip()
    if not normalized_round_id:
        raise ValueError("G4 product evidence requires an explicit round identity")
    if provider != DEFAULT_PROVIDER or not str(model).strip():
        raise ValueError("G4 product evidence requires an explicit DeepSeek provider/model")
    root = Path(runtime_root).resolve()
    output = Path(evidence_path).resolve()
    diff_output = Path(
        checkpoint_diff_path
        or (
            output.parent.parent
            / f"{output.parent.name}-checkpoint-diffs"
            / f"{output.stem}-checkpoint-diffs.json"
        )
    ).resolve()
    if output.parent == diff_output.parent:
        raise ValueError(
            "product evidence and checkpoint-diff artifacts require separate directories"
        )
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
    execution_proof = _validate_source_journey(
        result=result,
        source_evidence=source_evidence,
        schedule_payload=expected_schedule_payload,
        obligation=row,
        revision=revision,
        provider=provider,
        model=model,
        runtime_root=root,
    )
    actor_declarations = _validate_response_bound_attestations(
        source_evidence=source_evidence,
        obligation=row,
        schedule_payload=expected_schedule_payload,
    )
    session_id = str(source_evidence.get("session_id") or "").strip()
    request_ids = [
        str(
            dict(turn.get("decision_provenance") or {}).get(
                "broker_request_id"
            )
            or ""
        ).strip()
        for turn in tuple(source_evidence.get("turns") or ())
        if isinstance(turn, Mapping)
    ]
    if (
        not session_id
        or not request_ids
        or any(not request_id for request_id in request_ids)
        or len(request_ids) != len(set(request_ids))
    ):
        raise ValueError("G4 Journey has incomplete session/request identities")

    events = _load_runtime_events(source_paths["runtime_events"])
    rerun_observations = _rerun_journey_verifiers(
        obligation=row,
        source_evidence=source_evidence,
        schedule=expected_schedule,
        events=events,
        transcript_path=source_paths["transcript"],
    )
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
    if execution_proof:
        artifact_sources["process_guard_receipt"] = Path(
            str(execution_proof["path"])
        )
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
        rerun_observations=rerun_observations,
        actor_declarations=actor_declarations,
        classification=classification,
        outcome=outcome,
        artifact_hashes=artifact_hashes,
    )
    started_at, finished_at = _execution_interval(
        source_evidence,
        tuple(artifact_sources.values()),
    )
    execution = {
        "execution_id": source_evidence["execution_id"],
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
        "round_id": normalized_round_id,
        "session_id": session_id,
        "request_ids": request_ids,
        "execution_id": execution["execution_id"],
        "artifact_sha256s": sorted(artifact_hashes),
    }
    unsigned = {
        "schema_version": PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION,
        "evidence_id": content_hash(identity),
        "obligation_id": row["obligation_id"],
        "obligation_contract_hash": row["contract_hash"],
        "revision_binding": dict(revision),
        "round_id": normalized_round_id,
        "session_id": session_id,
        "request_ids": request_ids,
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
    canonical = _canonical_obligation_index(
        str(revision.get("commit") or ""),
        str(revision.get("worktree_hash") or ""),
    ).get(str(row["obligation_id"]))
    if canonical is None or row != canonical:
        raise ValueError("G4 obligation differs from the authoritative catalog")
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
    runtime_root: Path,
) -> dict[str, Any]:
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
    expected_registry = journey_outcome_verifier_registry_payload(
        FORMAL_JOURNEY_VERIFIER_REGISTRY
    )
    if (
        registry.get("registry_id") != verifier["registry_id"]
        or registry != expected_registry
    ):
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
    if qualifies:
        return _validate_source_execution_proof(
            source_evidence,
            runtime_root=runtime_root,
        )
    return {}


def _validate_source_execution_proof(
    source_evidence: Mapping[str, Any],
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    execution_id = str(source_evidence.get("execution_id") or "")
    proof = source_evidence.get("execution_proof")
    if (
        not execution_id
        or source_evidence.get("qualification_reason")
        != "trusted_container_pty_execution"
        or not isinstance(proof, Mapping)
        or proof.get("proof_type") != "container_pty_process_guard"
        or proof.get("transport_kind") != "container_pty_bridge"
        or proof.get("execution_id") != execution_id
    ):
        raise ValueError("source Journey lacks a trusted execution-bound PTY proof")
    proof_path = Path(str(proof.get("path") or ""))
    expected_receipt_root = (
        runtime_root.resolve() / "container-cleanup-receipts"
    )
    if proof_path.parent.resolve() != expected_receipt_root:
        raise ValueError("source Journey PTY proof is outside its runtime root")
    validated = validate_cleanup_receipt_artifact(
        proof_path,
        execution_id=execution_id,
        required_roles=(
            "container_bridge",
            "agent_process_group_leader",
        ),
        allowed_roots=(expected_receipt_root,),
    )
    expected = {
        "proof_type": "container_pty_process_guard",
        "transport_kind": "container_pty_bridge",
        **validated,
    }
    if dict(proof) != expected:
        raise ValueError("source Journey PTY proof differs from its receipt")
    return expected


@lru_cache(maxsize=None)
def _canonical_obligation_index(
    commit: str,
    worktree_hash: str,
) -> dict[str, dict[str, Any]]:
    revision = {"commit": commit, "worktree_hash": worktree_hash}
    return {
        str(row["obligation_id"]): dict(row)
        for row in build_product_chaos_obligations(revision=revision)
    }


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


def _validate_response_bound_attestations(
    *,
    source_evidence: Mapping[str, Any],
    obligation: Mapping[str, Any],
    schedule_payload: Mapping[str, Any],
) -> list[dict[str, str]]:
    turns = tuple(source_evidence.get("turns") or ())
    actors: dict[tuple[str, str, str], dict[str, str]] = {}
    seen_request_ids: set[str] = set()
    execution_id = str(source_evidence.get("execution_id") or "")
    session_id = str(source_evidence.get("session_id") or "")
    if not execution_id or not session_id:
        raise ValueError("Journey evidence has no execution/session identity")
    for turn in turns:
        turn_index = int(turn.get("turn_index") or 0)
        provenance = dict(turn.get("decision_provenance") or {})
        decision = dict(turn.get("decision") or {})
        attestation = provenance.get("simulator_attestation")
        context_binding = provenance.get("simulator_context_binding")
        request_id = str(provenance.get("broker_request_id") or "")
        if not isinstance(attestation, Mapping):
            raise ValueError(
                "Journey decision provenance has no simulator attestation"
            )
        if not isinstance(context_binding, Mapping):
            raise ValueError(
                "Journey decision provenance has no simulator context binding"
            )
        if not request_id or request_id in seen_request_ids:
            raise ValueError(
                "Journey decision provenance has a missing or replayed request id"
            )
        seen_request_ids.add(request_id)
        control_identity = dict(
            context_binding.get("control_identity") or {}
        )
        if (
            provenance.get("execution_id") != execution_id
            or provenance.get("obligation_id") != obligation["obligation_id"]
            or context_binding.get("session_id") != session_id
            or control_identity.get("lane") != "journey"
            or control_identity.get("schedule_id")
            != schedule_payload["schedule_id"]
        ):
            raise ValueError(
                "Journey decision provenance control identity is stale"
            )
        reconstructed = {
            "previous_response_hash": str(
                provenance.get("previous_response_hash") or ""
            ),
            "user_message": str(decision.get("user_message") or ""),
            "persona": str(decision.get("persona") or ""),
            "mission": str(decision.get("mission") or ""),
            "rationale": str(decision.get("rationale") or ""),
            "risk_factor_ids": list(decision.get("risk_factor_ids") or ()),
            "broker_request_id": request_id,
        }
        validate_simulator_attestation(
            attestation,
            previous_response_hash=reconstructed["previous_response_hash"],
            decision_hash=content_hash(reconstructed),
            request_id=request_id,
            context_hash=content_hash(context_binding),
            user_message_hash=str(
                provenance.get("user_message_hash") or ""
            ),
            turn_index=turn_index,
            submitted_at_ns=int(provenance.get("submitted_at_ns") or 0),
        )
        if str(attestation.get("user_message_hash") or "") != str(
            provenance.get("user_message_hash") or ""
        ):
            raise ValueError("simulator attestation user-message binding is stale")
        if _sha256_text(reconstructed["user_message"]) != str(
            provenance.get("user_message_hash") or ""
        ):
            raise ValueError("Journey decision text does not match its provenance hash")
        actor = dict(attestation["actor"])
        key = (
            str(actor["actor_kind"]),
            str(actor["task_id"]),
            str(actor["model"]),
        )
        actors[key] = {
            "actor_kind": key[0],
            "task_id": key[1],
            "model": key[2],
        }
    return [actors[key] for key in sorted(actors)]


def _sha256_text(value: str) -> str:
    return content_hash(value)


def _runtime_event(payload: Mapping[str, Any]) -> RuntimeTurnEvent:
    return RuntimeTurnEvent(
        schema_version=int(payload["schema_version"]),
        event_type=str(payload["event_type"]),
        observation=str(payload.get("observation") or ""),
        thread_id=str(payload["thread_id"]),
        session_purpose=str(payload.get("session_purpose") or ""),
        before_fingerprint=str(payload["before_fingerprint"]),
        after_fingerprint=str(payload["after_fingerprint"]),
        turn_index=int(payload["turn_index"]),
        active_group=str(payload.get("active_group") or ""),
        pending_question_id=str(payload.get("pending_question_id") or ""),
        action_queue_types=tuple(payload.get("action_queue_types") or ()),
        pending_contract=dict(payload.get("pending_contract") or {}),
        revision=dict(payload.get("revision") or {}),
        admitted_action_types=tuple(payload.get("admitted_action_types") or ()),
        admitted_action_targets=tuple(
            dict(item) for item in payload.get("admitted_action_targets") or ()
        ),
        admitted_action_provenance=tuple(
            dict(item) for item in payload.get("admitted_action_provenance") or ()
        ),
        turn_receipt_summary=dict(payload.get("turn_receipt_summary") or {}),
        pending_transition=dict(payload.get("pending_transition") or {}),
        render_manifest=dict(payload.get("render_manifest") or {}),
        execution_receipt_summary=dict(
            payload.get("execution_receipt_summary") or {}
        ),
        control_receipts=tuple(
            dict(item) for item in payload.get("control_receipts") or ()
        ),
        state_diff_hashes=dict(payload.get("state_diff_hashes") or {}),
        material_state_diff_hashes=dict(
            payload.get("material_state_diff_hashes") or {}
        ),
        after_value_hashes=dict(payload.get("after_value_hashes") or {}),
        next_result=dict(payload.get("next_result") or {}),
        runtime_event_id=str(payload.get("runtime_event_id") or ""),
        runtime_event_sequence=payload.get("runtime_event_sequence"),
        runtime_event_payload_hash=str(
            payload.get("runtime_event_payload_hash") or ""
        ),
        terminal_event_id=str(payload.get("terminal_event_id") or ""),
        transaction_id=str(payload.get("transaction_id") or ""),
        terminal_outcome=str(payload.get("terminal_outcome") or ""),
        render_hash=str(payload.get("render_hash") or ""),
        base_revision=payload.get("base_revision"),
        base_checkpoint_thread_id=str(
            payload.get("base_checkpoint_thread_id") or ""
        ),
        base_checkpoint_id=str(payload.get("base_checkpoint_id") or ""),
        product_revision=payload.get("product_revision"),
        product_checkpoint_thread_id=str(
            payload.get("product_checkpoint_thread_id") or ""
        ),
        product_checkpoint_id=str(
            payload.get("product_checkpoint_id") or ""
        ),
        product_authority_id=str(
            payload.get("product_authority_id") or ""
        ),
        physical_thread_id=str(payload.get("physical_thread_id") or ""),
        attempt_checkpoint_id=str(
            payload.get("attempt_checkpoint_id") or ""
        ),
    )


def _transcript_responses(
    transcript_path: Path,
    decisions: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, ...]]:
    text = transcript_path.read_text(encoding="utf-8")
    cursor = 0
    previous_response = ""
    responses: list[str] = []
    for index, decision in enumerate(decisions):
        marker = "User> " + str(decision.get("user_message") or "")
        position = text.find(marker, cursor)
        if position < 0:
            raise ValueError("transcript is not bound to the recorded Journey decisions")
        if index == 0:
            previous_response = text[:position].rstrip()
        response_start = position + len(marker)
        next_marker = (
            "User> " + str(decisions[index + 1].get("user_message") or "")
            if index + 1 < len(decisions)
            else ""
        )
        if next_marker:
            response_end = text.find(next_marker, response_start)
            if response_end < 0:
                raise ValueError("transcript Journey turn order is incomplete")
        else:
            response_end = len(text)
        responses.append(text[response_start:response_end].strip())
        cursor = response_end
    return previous_response, tuple(responses)


def _rerun_journey_verifiers(
    *,
    obligation: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    schedule: Any,
    events: Sequence[Mapping[str, Any]],
    transcript_path: Path,
) -> dict[str, dict[str, Any]]:
    runtime_events = tuple(_runtime_event(event) for event in events)
    if len(runtime_events) < 2:
        raise ValueError("independent verifier rerun requires baseline and committed events")
    initial = runtime_events[0]
    committed = runtime_events[1:]
    source_turns = tuple(source_evidence.get("turns") or ())
    if len(committed) != len(source_turns):
        raise ValueError("runtime events and Journey turns have different cardinality")
    decisions = tuple(dict(turn.get("decision") or {}) for turn in source_turns)
    previous_response, responses = _transcript_responses(
        transcript_path, decisions
    )
    pty_turns: list[PtyCliTurnRecord] = []
    provenance_rows: list[JourneyDecisionProvenance] = []
    transcript: list[tuple[str, str]] = []
    for index, (source_turn, event, decision, response) in enumerate(
        zip(source_turns, committed, decisions, responses, strict=True)
    ):
        identity = dict(source_turn.get("turn_identity") or {})
        provenance = dict(source_turn.get("decision_provenance") or {})
        user_message = str(decision.get("user_message") or "")
        turn = PtyCliTurnRecord(
            session_id=str(source_evidence["session_id"]),
            turn_index=int(source_turn["turn_index"]),
            previous_agent_response=previous_response,
            user_message=user_message,
            agent_response=response,
            provider=str(source_evidence["provider"]),
            model=str(source_evidence["model"]),
            before_fingerprint=str(identity["before_fingerprint"]),
            after_fingerprint=str(identity["after_fingerprint"]),
            transcript_hash=str(identity["transcript_hash"]),
            previous_response_received_at_ns=int(
                identity["previous_response_received_at_ns"]
            ),
            user_message_submitted_at_ns=int(
                identity["user_message_submitted_at_ns"]
            ),
            agent_response_received_at_ns=int(
                identity["agent_response_received_at_ns"]
            ),
        )
        pty_turns.append(turn)
        provenance_rows.append(JourneyDecisionProvenance(
            turn_index=turn.turn_index,
            previous_response_hash=str(provenance["previous_response_hash"]),
            selected_at_ns=int(provenance["selected_at_ns"]),
            submitted_at_ns=int(provenance["submitted_at_ns"]),
            user_message_hash=str(provenance["user_message_hash"]),
            persona=str(decision["persona"]),
            mission=str(decision["mission"]),
            rationale=str(decision["rationale"]),
            risk_factor_ids=tuple(decision.get("risk_factor_ids") or ()),
        ))
        transcript.append((user_message, response))
        previous_response = response
    context = JourneyVerifierContext(
        schedule=schedule,
        initial_event=initial,
        current_event=committed[-1],
        completed_turns=tuple(pty_turns),
        transcript=tuple(transcript),
        observed_edge_keys=tuple(source_evidence.get("observed_edge_keys") or ()),
        latest_turn=pty_turns[-1],
        completed_events=committed,
        completed_decisions=tuple(provenance_rows),
    )
    verifier = dict(obligation["verifier_contract"])
    observations: dict[str, dict[str, Any]] = {}
    for postcondition_id in (
        *verifier["required_postcondition_ids"],
        *verifier["forbidden_postcondition_ids"],
    ):
        definition = FORMAL_JOURNEY_VERIFIER_REGISTRY.definitions[postcondition_id]
        result = definition.verifier(
            replace(context, evaluating_postcondition_id=postcondition_id)
        )
        observations[postcondition_id] = {
            "satisfied": result.satisfied,
            "details": dict(result.details),
            "verifier": {
                "verifier_id": definition.verifier_id,
                "verifier_version": definition.verifier_version,
                "implementation_hash": definition.implementation_hash,
            },
        }
    return observations


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
            "admitted_action_provenance": list(
                event.get("admitted_action_provenance") or ()
            ),
            "turn_receipt_summary": dict(
                event.get("turn_receipt_summary") or {}
            ),
            "pending_transition": dict(event.get("pending_transition") or {}),
            "render_manifest": dict(event.get("render_manifest") or {}),
            "execution_receipt_summary": dict(
                event.get("execution_receipt_summary") or {}
            ),
            "control_receipts": list(event.get("control_receipts") or ()),
            "state_diff_hashes": dict(event.get("state_diff_hashes") or {}),
            "material_state_diff_hashes": dict(
                event.get("material_state_diff_hashes") or {}
            ),
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
    rerun_observations: Mapping[str, Mapping[str, Any]],
    actor_declarations: Sequence[Mapping[str, str]],
    classification: str,
    outcome: str,
    artifact_hashes: Sequence[str],
) -> list[dict[str, Any]]:
    verifier = dict(obligation["verifier_contract"])
    required = tuple(verifier["required_postcondition_ids"])
    forbidden = tuple(verifier["forbidden_postcondition_ids"])
    verifier_bindings = {
        str(item["postcondition_id"]): dict(item)
        for item in (
            *(verifier.get("required_bindings") or ()),
            *(verifier.get("forbidden_bindings") or ()),
        )
    }
    recorded_observations = _latest_postcondition_observations(source_evidence)
    results: list[dict[str, Any]] = []
    for postcondition_id in (*required, *forbidden):
        observed = rerun_observations.get(postcondition_id)
        recorded = recorded_observations.get(postcondition_id)
        if (
            classification == "passed"
            and observed is not None
            and recorded is not None
            and observed.get("satisfied") is not recorded.get("satisfied")
        ):
            missing = list(
                dict(observed.get("details") or {}).get(
                    "missing_product_receipts"
                ) or ()
            )
            raise ValueError(
                "independent verifier rerun contradicts recorded Journey verification: "
                + postcondition_id
                + (f"; missing product receipts: {missing}" if missing else "")
            )
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
            "recorded_observation": recorded,
            "verification_mode": "independent_rerun",
            "simulator_actor_declarations": list(actor_declarations),
        }, ensure_ascii=False, sort_keys=True)
        results.append({
            "verifier_id": postcondition_id,
            "verifier_version": verifier_bindings[postcondition_id][
                "verifier_version"
            ],
            "implementation_hash": verifier_bindings[postcondition_id][
                "implementation_hash"
            ],
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

    targets = subparsers.add_parser(
        "targets",
        help="Translate a frozen definition manifest into immutable batch targets.",
    )
    targets.add_argument("--definitions", required=True, type=Path)
    targets.add_argument("--output-dir", required=True, type=Path)

    batch = subparsers.add_parser(
        "batch",
        help="Freeze a G4 batch from immutable targets without changing row seeds.",
    )
    batch.add_argument("--repo-root", required=True, type=Path)
    batch.add_argument("--targets-dir", required=True, type=Path)
    batch.add_argument("--output", required=True, type=Path)
    batch.add_argument("--runtime-base", required=True, type=Path)
    batch.add_argument("--worker-runtime", choices=("linux", "docker"), default="linux")
    batch.add_argument("--max-concurrency", type=_positive_int)
    batch.add_argument(
        "--shard-timeout-seconds",
        type=_positive_float,
        default=TimeoutPolicy().shard_seconds,
    )
    batch.add_argument(
        "--decision-timeout-seconds",
        type=_positive_float,
        default=TimeoutPolicy().decision_seconds,
    )
    batch.add_argument(
        "--cleanup-timeout-seconds",
        type=_positive_float,
        default=TimeoutPolicy().cleanup_seconds,
    )

    evidence = subparsers.add_parser(
        "evidence",
        help="Convert an already completed Journey runtime into G4 evidence.",
    )
    evidence.add_argument("--catalog", required=True, type=Path)
    evidence.add_argument("--obligation-id", required=True)
    evidence.add_argument("--round-id", required=True)
    evidence.add_argument("--runtime-root", required=True, type=Path)
    evidence.add_argument("--output", required=True, type=Path)
    evidence.add_argument("--checkpoint-diff", type=Path)
    evidence.add_argument("--provider", default=DEFAULT_PROVIDER)
    evidence.add_argument("--model", default=DEFAULT_MODEL)

    evidence_batch = subparsers.add_parser(
        "evidence-batch",
        help="Convert one complete response-driven G4 batch into declared evidence.",
    )
    evidence_batch.add_argument("--catalog", required=True, type=Path)
    evidence_batch.add_argument("--manifest", required=True, type=Path)
    evidence_batch.add_argument("--result-index", required=True, type=Path)
    evidence_batch.add_argument("--round-id", required=True)
    evidence_batch.add_argument("--output-dir", required=True, type=Path)
    evidence_batch.add_argument("--provider", default=DEFAULT_PROVIDER)
    evidence_batch.add_argument("--model", default=DEFAULT_MODEL)
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "targets":
        definition_manifest = _load_mapping(
            args.definitions.resolve(),
            "G4 definition manifest",
        )
        write_product_chaos_target_set(definition_manifest, args.output_dir)
        return 0
    if args.command == "batch":
        freeze_product_chaos_batch(
            repo_root=args.repo_root,
            targets_dir=args.targets_dir,
            manifest_path=args.output,
            runtime_base=args.runtime_base,
            worker_runtime=args.worker_runtime,
            max_concurrency=args.max_concurrency,
            timeout_policy=TimeoutPolicy(
                shard_seconds=args.shard_timeout_seconds,
                decision_seconds=args.decision_timeout_seconds,
                cleanup_seconds=args.cleanup_timeout_seconds,
            ),
        )
        return 0

    rows, revision = load_frozen_product_chaos_catalog(args.catalog)
    if args.command == "definitions":
        manifest = build_product_chaos_journey_manifest(
            rows,
            revision=revision,
            obligation_id=args.obligation_id,
        )
        write_product_chaos_journey_manifest(manifest, args.output)
        return 0
    if args.command == "evidence-batch":
        def convert_one(
            obligation: Mapping[str, Any],
            runtime_root: Path,
            evidence_path: Path,
            checkpoint_diff: Path | None,
        ) -> Path:
            return convert_completed_journey_to_product_evidence(
                obligation,
                revision=revision,
                round_id=args.round_id,
                runtime_root=runtime_root,
                evidence_path=evidence_path,
                checkpoint_diff_path=checkpoint_diff,
                provider=args.provider,
                model=args.model,
            )

        convert_completed_journey_batch(
            manifest_path=args.manifest,
            result_index_path=args.result_index,
            output_dir=args.output_dir,
            obligations=rows,
            revision=revision,
            artifact_type=G4_ARTIFACT_TYPE,
            round_id=args.round_id,
            convert_one=convert_one,
        )
        return 0

    matches = tuple(
        row for row in rows if row["obligation_id"] == args.obligation_id
    )
    if len(matches) != 1:
        raise ValueError(f"unknown G4 obligation: {args.obligation_id}")
    convert_completed_journey_to_product_evidence(
        matches[0],
        revision=revision,
        round_id=args.round_id,
        runtime_root=args.runtime_root,
        evidence_path=args.output,
        checkpoint_diff_path=args.checkpoint_diff,
        provider=args.provider,
        model=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
