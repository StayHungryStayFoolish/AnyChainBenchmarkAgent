"""Formal conversion and admission index for completed Journey batches."""

from __future__ import annotations

import hashlib
import json
import os
import ctypes
import errno
import fcntl
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tests.agent_live.batch_orchestrator import (
    BATCH_RESULT_SCHEMA_VERSION,
    JourneyControllerAuthoritySnapshot,
    load_frozen_manifest,
    load_validated_journey_controller_authority,
    validate_completed_shard_result,
)
from tests.agent_live.coverage_evidence import (
    content_hash,
    verify_controller_payload_signature,
)
from tests.agent_live.product_obligation_evidence import (
    admit_product_obligation_evidence,
)


SCHEMA_VERSION = 1
G3_ARTIFACT_TYPE = "g3_open_completed_batch_evidence"
G4_ARTIFACT_TYPE = "g4_completed_batch_evidence"
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
_COMPLETED_JOURNEY_SOURCE_AUTHORITY = object()


@dataclass(frozen=True)
class CompletedJourneyRuntimeArtifact:
    relative_path: str
    content: bytes
    mtime_ns: int


@dataclass(frozen=True)
class CompletedJourneySource:
    """Controller-admitted source passed only by completed-batch conversion."""

    runtime_root: Path
    obligation_id: str
    shard_id: str
    execution_id: str
    controller_snapshot: JourneyControllerAuthoritySnapshot
    runtime_artifacts: tuple[CompletedJourneyRuntimeArtifact, ...]
    _authority: object = field(repr=False, compare=False)

    def require_authority(self, *, obligation_id: str) -> None:
        if (
            self._authority is not _COMPLETED_JOURNEY_SOURCE_AUTHORITY
            or self.obligation_id != obligation_id
            or not self.shard_id
            or not self.execution_id
        ):
            raise ValueError(
                "Journey conversion requires completed-batch authority"
            )

    def materialize_runtime(self, destination: Path) -> Path:
        self.require_authority(obligation_id=self.obligation_id)
        root = destination.resolve()
        root.mkdir(mode=0o700, parents=True)
        for artifact in self.runtime_artifacts:
            path = root / artifact.relative_path
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.write_bytes(artifact.content)
            os.utime(
                path,
                ns=(artifact.mtime_ns, artifact.mtime_ns),
            )
            path.chmod(0o400)
        candidate = root / "journey-controller-admission" / "candidate.json"
        candidate.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        candidate.write_bytes(self.controller_snapshot.candidate_bytes)
        candidate.chmod(0o400)
        candidate.parent.chmod(0o500)
        return root


def convert_completed_journey_batch(
    *,
    manifest_path: str | Path,
    result_index_path: str | Path,
    output_dir: str | Path,
    obligations: Sequence[Mapping[str, Any]],
    revision: Mapping[str, str],
    artifact_type: str,
    round_id: str,
    expected_authority_trust_root_id: str,
    convert_one: Callable[
        [Mapping[str, Any], CompletedJourneySource, Path, Path | None],
        Path,
    ],
) -> Path:
    """Convert and publish a completed batch under one lifecycle lock."""

    destination = Path(output_dir).expanduser().resolve()
    with _completed_batch_lock(destination):
        for staging in _completed_batch_staging_paths(destination):
            _remove_completed_batch_path(staging)
        orphan_marker = _reconciliation_marker(destination)
        if not destination.exists() and orphan_marker.exists():
            _clear_reconciliation_marker(orphan_marker)
        if destination.exists():
            try:
                _validate_recovery_source_binding(
                    destination / "manifest.json",
                    manifest_path=manifest_path,
                    result_index_path=result_index_path,
                    obligations=obligations,
                    revision=revision,
                    expected_authority_trust_root_id=(
                        expected_authority_trust_root_id
                    ),
                )
                load_completed_journey_batch_evidence(
                    destination / "manifest.json",
                    obligations=obligations,
                    revision=revision,
                    artifact_type=artifact_type,
                    round_id=round_id,
                    expected_authority_trust_root_id=(
                        expected_authority_trust_root_id
                    ),
                    _allow_durability_uncertain=True,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise FileExistsError(
                    "completed-batch destination is not the same "
                    "recoverable publication"
                ) from exc
            _reconcile_completed_batch_publication(destination)
            return destination / "manifest.json"
        try:
            return _convert_completed_journey_batch_locked(
                manifest_path=manifest_path,
                result_index_path=result_index_path,
                output_dir=destination,
                obligations=obligations,
                revision=revision,
                artifact_type=artifact_type,
                round_id=round_id,
                expected_authority_trust_root_id=(
                    expected_authority_trust_root_id
                ),
                convert_one=convert_one,
            )
        finally:
            for staging in _completed_batch_staging_paths(destination):
                _remove_completed_batch_path(staging)


def _convert_completed_journey_batch_locked(
    *,
    manifest_path: str | Path,
    result_index_path: str | Path,
    output_dir: str | Path,
    obligations: Sequence[Mapping[str, Any]],
    revision: Mapping[str, str],
    artifact_type: str,
    round_id: str,
    expected_authority_trust_root_id: str,
    convert_one: Callable[
        [Mapping[str, Any], CompletedJourneySource, Path, Path | None],
        Path,
    ],
) -> Path:
    """Convert every passed Journey shard and publish one declared evidence index."""

    expected = {
        str(row.get("obligation_id") or ""): dict(row)
        for row in obligations
        if str(row.get("obligation_id") or "")
    }
    if not expected or len(expected) != len(obligations):
        raise ValueError("completed-batch obligations are incomplete or duplicated")
    manifest, sources, execution_ids, result_index = _validated_completed_batch(
        manifest_path=manifest_path,
        result_index_path=result_index_path,
        expected_obligation_ids=set(expected),
        revision=revision,
        expected_authority_trust_root_id=expected_authority_trust_root_id,
    )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"completed-batch evidence destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f".{destination.name}.staging-{uuid.uuid4().hex}"
    )
    staging.mkdir(mode=0o700)
    evidence_dir = staging / "evidence"
    checkpoint_diff_dir = staging / "checkpoint-diffs"
    evidence_dir.mkdir()
    if artifact_type == G4_ARTIFACT_TYPE:
        checkpoint_diff_dir.mkdir()

    evidence_paths: list[Path] = []
    evidence_identities: dict[str, dict[str, str]] = {}
    for obligation_id in sorted(expected):
        filename = hashlib.sha256(obligation_id.encode("utf-8")).hexdigest()
        evidence_path = evidence_dir / f"evidence-{filename}.json"
        checkpoint_diff = (
            checkpoint_diff_dir / f"checkpoint-diff-{filename}.json"
            if artifact_type == G4_ARTIFACT_TYPE
            else None
        )
        converted = convert_one(
            expected[obligation_id],
            sources[obligation_id],
            evidence_path,
            checkpoint_diff,
        )
        if converted.resolve() != evidence_path.resolve():
            raise ValueError("completed-batch converter wrote outside its declared path")
        evidence_document = _load_mapping(evidence_path)
        identity = {
            "evidence_id": str(evidence_document.get("evidence_id") or ""),
            "execution_id": str(
                (evidence_document.get("execution") or {}).get("execution_id")
                if isinstance(evidence_document.get("execution"), Mapping)
                else ""
            ),
            "obligation_contract_hash": str(
                evidence_document.get("obligation_contract_hash") or ""
            ),
        }
        if (
            not all(len(value) == 64 for value in identity.values())
            or identity["obligation_contract_hash"]
            != str(expected[obligation_id].get("contract_hash") or "")
            or identity["execution_id"] != execution_ids[obligation_id]
        ):
            raise ValueError("converted evidence identity is incomplete or stale")
        evidence_identities[obligation_id] = identity
        evidence_paths.append(evidence_path)

    admitted = admit_product_obligation_evidence(
        obligations=tuple(expected.values()),
        evidence_paths=evidence_paths,
        revision=revision,
    )
    if (
        admitted.get("complete") is not True
        or admitted.get("passed") != len(expected)
        or admitted.get("failed") != 0
        or admitted.get("external") != 0
        or admitted.get("not_run") != 0
    ):
        raise ValueError("completed-batch evidence did not produce all qualifying passes")

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "revision_binding": dict(revision),
        "round_id": str(round_id),
        "authority_trust_root_id": expected_authority_trust_root_id,
        "batch_manifest": _source_reference(
            Path(manifest_path), "manifest_id", manifest.manifest_id
        ),
        "batch_result": _source_reference(
            Path(result_index_path), "index_id", str(result_index["index_id"])
        ),
        "obligation_count": len(expected),
        "evidence": [
            {
                "obligation_id": obligation_id,
                **evidence_identities[obligation_id],
                "path": str(
                    (destination / path.relative_to(staging)).resolve()
                ),
                "sha256": _sha256(path),
            }
            for obligation_id, path in zip(
                sorted(expected),
                evidence_paths,
                strict=True,
            )
        ],
    }
    index = {**unsigned, "index_hash": content_hash(unsigned)}
    index_path = staging / "manifest.json"
    _write_once(index_path, index)
    _fsync_completed_batch_tree(staging)
    for path in staging.rglob("*"):
        path.chmod(0o500 if path.is_dir() else 0o400)
    staging.chmod(0o500)
    _fsync_completed_batch_tree(staging)
    marker = _write_reconciliation_marker(
        destination,
        index_hash=str(index["index_hash"]),
    )
    try:
        _rename_directory_noreplace(staging, destination)
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
        _clear_reconciliation_marker(marker)
    except OSError as exc:
        if destination.exists():
            _ensure_reconciliation_marker(
                destination,
                index_hash=str(index["index_hash"]),
            )
            raise RuntimeError(
                "completed-batch publication durability is uncertain; "
                "retry with the same sources to reconcile"
            ) from exc
        marker.unlink(missing_ok=True)
        raise
    return destination / "manifest.json"


@contextmanager
def _completed_batch_lock(destination: Path):
    lock_path = destination.with_name(f".{destination.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _completed_batch_staging_paths(
    destination: Path,
) -> tuple[Path, ...]:
    return tuple(
        destination.parent.glob(f".{destination.name}.staging-*")
    )


def _remove_completed_batch_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        path.chmod(0o700)
        for child in path.rglob("*"):
            if not child.is_symlink():
                child.chmod(0o700 if child.is_dir() else 0o600)
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_completed_batch_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(root)


def _reconciliation_marker(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.name}.durability-uncertain.json"
    )


def _write_reconciliation_marker(
    destination: Path,
    *,
    index_hash: str,
) -> Path:
    marker = _reconciliation_marker(destination)
    payload = {
        "schema_version": 1,
        "destination": str(destination),
        "index_hash": index_hash,
        "state": "durability_uncertain",
    }
    if marker.exists():
        existing = _load_mapping(marker)
        if existing != payload:
            raise RuntimeError(
                "completed-batch reconciliation marker conflicts with "
                "the publication"
            )
        return marker
    _write_once(marker, payload)
    marker.chmod(0o400)
    with marker.open("rb") as handle:
        os.fsync(handle.fileno())
    _fsync_directory(marker.parent)
    return marker


def _ensure_reconciliation_marker(
    destination: Path,
    *,
    index_hash: str,
) -> None:
    try:
        _write_reconciliation_marker(
            destination,
            index_hash=index_hash,
        )
    except OSError:
        # The visible marker still protects the live filesystem namespace.
        # A retry must reconcile the destination before qualification.
        pass


def _clear_reconciliation_marker(marker: Path) -> None:
    marker.unlink(missing_ok=True)
    _fsync_directory(marker.parent)


def _validate_reconciliation_marker(
    destination: Path,
    *,
    index_hash: str,
) -> Path | None:
    marker = _reconciliation_marker(destination)
    if not marker.exists():
        return None
    _require_immutable_regular(
        marker,
        description="completed-batch reconciliation marker",
    )
    payload = _load_mapping(marker)
    if payload != {
        "schema_version": 1,
        "destination": str(destination),
        "index_hash": index_hash,
        "state": "durability_uncertain",
    }:
        raise ValueError(
            "completed-batch reconciliation marker is invalid"
        )
    return marker


def _reconcile_completed_batch_publication(destination: Path) -> None:
    index_payload = _load_mapping(destination / "manifest.json")
    index_hash = str(index_payload.get("index_hash") or "")
    marker = _validate_reconciliation_marker(
        destination,
        index_hash=index_hash,
    )
    try:
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
    except OSError as exc:
        _ensure_reconciliation_marker(
            destination,
            index_hash=index_hash,
        )
        raise RuntimeError(
            "completed-batch publication durability remains uncertain; "
            "reconciliation did not complete"
        ) from exc
    if marker is not None:
        try:
            _clear_reconciliation_marker(marker)
        except OSError as exc:
            _ensure_reconciliation_marker(
                destination,
                index_hash=index_hash,
            )
            raise RuntimeError(
                "completed-batch reconciliation marker could not be "
                "durably cleared"
            ) from exc


def _validate_recovery_source_binding(
    index_path: Path,
    *,
    manifest_path: str | Path,
    result_index_path: str | Path,
    obligations: Sequence[Mapping[str, Any]],
    revision: Mapping[str, str],
    expected_authority_trust_root_id: str,
) -> None:
    payload = _load_mapping(index_path)
    manifest, _, _, result = _validated_completed_batch(
        manifest_path=manifest_path,
        result_index_path=result_index_path,
        expected_obligation_ids={
            str(row.get("obligation_id") or "") for row in obligations
        },
        revision=revision,
        expected_authority_trust_root_id=expected_authority_trust_root_id,
    )
    expected_manifest = _source_reference(
        Path(manifest_path),
        "manifest_id",
        str(manifest.manifest_id),
    )
    expected_result = _source_reference(
        Path(result_index_path),
        "index_id",
        str(result.get("index_id") or ""),
    )
    if (
        payload.get("batch_manifest") != expected_manifest
        or payload.get("batch_result") != expected_result
    ):
        raise ValueError(
            "completed-batch retry sources do not match the published "
            "manifest/result path, digest, and identity"
        )


def load_completed_journey_batch_evidence(
    index_path: str | Path,
    *,
    obligations: Sequence[Mapping[str, Any]],
    revision: Mapping[str, str],
    artifact_type: str,
    round_id: str,
    expected_authority_trust_root_id: str,
    _allow_durability_uncertain: bool = False,
) -> tuple[Path, ...]:
    raw_path = Path(index_path).expanduser()
    if raw_path.is_symlink():
        raise ValueError("completed-batch evidence index cannot be a symlink")
    path = raw_path.resolve()
    if path.name != "manifest.json" or not path.is_file():
        raise ValueError("completed-batch evidence index is missing")
    if path.stat().st_mode & 0o222 or path.parent.stat().st_mode & 0o222:
        raise ValueError("completed-batch evidence index must be immutable")
    payload = _load_mapping(path)
    marker = _validate_reconciliation_marker(
        path.parent,
        index_hash=str(payload.get("index_hash") or ""),
    )
    if marker is not None and not _allow_durability_uncertain:
        raise ValueError(
            "completed-batch publication durability is uncertain; "
            "reconciliation is required"
        )
    unsigned = {key: value for key, value in payload.items() if key != "index_hash"}
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("artifact_type") != artifact_type
        or dict(payload.get("revision_binding") or {}) != dict(revision)
        or str(payload.get("round_id") or "") != str(round_id)
        or payload.get("authority_trust_root_id")
        != expected_authority_trust_root_id
        or payload.get("index_hash") != content_hash(unsigned)
    ):
        raise ValueError("completed-batch evidence index contract is invalid")
    manifest_ref = payload.get("batch_manifest")
    result_ref = payload.get("batch_result")
    if not isinstance(manifest_ref, Mapping) or not isinstance(result_ref, Mapping):
        raise ValueError("completed-batch source references are invalid")
    manifest_source = Path(str(manifest_ref.get("path") or ""))
    result_source = Path(str(result_ref.get("path") or ""))
    manifest, sources, execution_ids, result_index = (
        _validated_completed_batch(
        manifest_path=manifest_source,
        result_index_path=result_source,
        expected_obligation_ids={
            str(row.get("obligation_id") or "") for row in obligations
        },
        revision=revision,
        expected_authority_trust_root_id=expected_authority_trust_root_id,
        )
    )
    if (
        _sha256(manifest_source) != str(manifest_ref.get("sha256") or "")
        or manifest.manifest_id != str(manifest_ref.get("manifest_id") or "")
        or _sha256(result_source) != str(result_ref.get("sha256") or "")
        or str(result_index.get("index_id") or "")
        != str(result_ref.get("index_id") or "")
    ):
        raise ValueError("completed-batch source reference binding is invalid")
    expected_ids = {
        str(row.get("obligation_id") or "") for row in obligations
    }
    expected_contracts = {
        str(row.get("obligation_id") or ""): str(row.get("contract_hash") or "")
        for row in obligations
    }
    rows = payload.get("evidence")
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError("completed-batch evidence index cardinality is invalid")
    indexed: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("completed-batch evidence row is invalid")
        obligation_id = str(row.get("obligation_id") or "")
        raw_evidence_path = Path(str(row.get("path") or ""))
        if raw_evidence_path.is_symlink():
            raise ValueError("completed-batch evidence row cannot be a symlink")
        evidence_path = raw_evidence_path.resolve()
        if (
            set(row) != {
                "obligation_id",
                "obligation_contract_hash",
                "evidence_id",
                "execution_id",
                "path",
                "sha256",
            }
            or obligation_id not in expected_ids
            or obligation_id in indexed
            or evidence_path.parent != path.parent / "evidence"
            or evidence_path.parent.stat().st_mode & 0o222
            or not evidence_path.is_file()
            or evidence_path.stat().st_mode & 0o222
            or _sha256(evidence_path) != str(row.get("sha256") or "")
            or str(row.get("obligation_contract_hash") or "")
            != expected_contracts[obligation_id]
        ):
            raise ValueError("completed-batch evidence row binding is invalid")
        evidence_document = _load_mapping(evidence_path)
        execution = evidence_document.get("execution")
        source = sources.get(obligation_id)
        artifact_rows = evidence_document.get("artifacts")
        if (
            evidence_document.get("obligation_id") != obligation_id
            or evidence_document.get("obligation_contract_hash")
            != row.get("obligation_contract_hash")
            or evidence_document.get("evidence_id") != row.get("evidence_id")
            or not isinstance(execution, Mapping)
            or execution.get("execution_id") != row.get("execution_id")
            or execution.get("execution_id") != execution_ids.get(obligation_id)
            or source is None
            or not isinstance(artifact_rows, list)
        ):
            raise ValueError("completed-batch evidence row binding is invalid")
        resolved_runtime = source.runtime_root.resolve()
        for artifact_row in artifact_rows:
            artifact_sha = str(
                artifact_row.get("sha256") or ""
            ) if isinstance(artifact_row, Mapping) else ""
            if (
                not isinstance(artifact_row, Mapping)
                or set(artifact_row) != {"role", "path", "sha256"}
                or not str(artifact_row.get("role") or "")
                or len(artifact_sha) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in artifact_sha
                )
            ):
                raise ValueError(
                    "completed-batch runtime artifact binding is invalid"
                )
            artifact_path = Path(str(artifact_row.get("path") or ""))
            resolved_artifact = artifact_path.resolve()
            if not resolved_artifact.is_relative_to(resolved_runtime):
                raise ValueError(
                    "completed-batch runtime artifact binding is invalid"
                )
        indexed[obligation_id] = evidence_path
    if set(indexed) != expected_ids:
        raise ValueError("completed-batch evidence obligation set is incomplete")
    paths = tuple(indexed[key] for key in sorted(indexed))
    admitted = admit_product_obligation_evidence(
        obligations=obligations,
        evidence_paths=paths,
        revision=revision,
    )
    if admitted.get("complete") is not True:
        raise ValueError("completed-batch evidence admission is incomplete")
    return paths


def _validated_completed_batch(
    *,
    manifest_path: str | Path,
    result_index_path: str | Path,
    expected_obligation_ids: set[str],
    revision: Mapping[str, str],
    expected_authority_trust_root_id: str,
) -> tuple[
    Any,
    dict[str, CompletedJourneySource],
    dict[str, str],
    dict[str, Any],
]:
    _require_immutable_regular(
        Path(manifest_path),
        description="completed-batch manifest",
    )
    _require_immutable_regular(
        Path(result_index_path),
        description="completed-batch result index",
    )
    manifest = load_frozen_manifest(manifest_path)
    if not manifest.controller_owned_execution:
        raise ValueError(
            "completed-batch evidence is not controller-owned"
        )
    if (
        not expected_authority_trust_root_id
        or manifest.pty_authority_trust_root_id
        != expected_authority_trust_root_id
    ):
        raise ValueError("completed-batch manifest trust root is not externally trusted")
    if dict(manifest.revision) != dict(revision):
        raise ValueError("completed-batch manifest revision mismatch")
    shards = tuple(manifest.shards)
    if (
        len(shards) != len(expected_obligation_ids)
        or any(shard.lane != "journey" for shard in shards)
        or {shard.obligation_id for shard in shards} != expected_obligation_ids
    ):
        raise ValueError("completed-batch manifest obligation set mismatch")
    result = _load_mapping(Path(result_index_path))
    signed_payload = {
        key: value
        for key, value in result.items()
        if key not in {"index_id", "controller_signature_b64"}
    }
    signed_index = {
        **signed_payload,
        "controller_signature_b64": str(
            result.get("controller_signature_b64") or ""
        ),
    }
    if (
        result.get("schema_version") != BATCH_RESULT_SCHEMA_VERSION
        or result.get("index_id") != content_hash(signed_index)
        or result.get("batch_id") != manifest.batch_id
        or result.get("manifest_id") != manifest.manifest_id
        or result.get("execution_authority_mode") != "controller_owned_v1"
        or result.get("pty_authority_trust_root_id")
        != manifest.pty_authority_trust_root_id
        or result.get("pty_authority_public_key_b64")
        != manifest.pty_authority_public_key_b64
        or not verify_controller_payload_signature(
            signed_payload,
            signature_b64=str(
                result.get("controller_signature_b64") or ""
            ),
            trusted_public_key_b64=manifest.pty_authority_public_key_b64,
        )
        or dict(result.get("revision") or {}) != dict(revision)
        or result.get("execution_status") != "discovery_complete"
        or result.get("scheduled") != len(shards)
        or result.get("started") != len(shards)
        or result.get("completed") != len(shards)
        or (result.get("batch_survivor_proof") or {}).get("cleaned") is not True
    ):
        raise ValueError("completed-batch result index is not a complete clean execution")
    result_rows = result.get("shards")
    if not isinstance(result_rows, list) or len(result_rows) != len(shards):
        raise ValueError("completed-batch result shard set is invalid")
    by_shard = {
        str(row.get("shard_id") or ""): dict(row)
        for row in result_rows
        if isinstance(row, Mapping)
    }
    if len(by_shard) != len(shards):
        raise ValueError("completed-batch result shard identities are invalid")
    sources: dict[str, CompletedJourneySource] = {}
    execution_ids: dict[str, str] = {}
    for shard in shards:
        row = by_shard.get(shard.shard_id)
        response_hashes = row.get("response_hashes") if row else None
        decision_hashes = row.get("decision_hashes") if row else None
        if (
            row is None
            or row.get("classification") != "passed"
            or row.get("target_hash") != shard.target_hash
            or int(row.get("finished_at_ns") or 0)
            <= int(row.get("started_at_ns") or 0)
            or not isinstance(response_hashes, list)
            or not isinstance(decision_hashes, list)
            or len(response_hashes) != len(decision_hashes)
            or not row.get("evidence_hashes")
        ):
            raise ValueError(
                f"completed-batch shard is not qualifying: {shard.shard_id}"
            )
        validate_completed_shard_result(shard, row)
        runtime_root = Path(shard.runtime_root).resolve()
        journey_result_path = runtime_root / "journey-result.json"
        transcript_path = runtime_root / "transcript.txt"
        authority_path = runtime_root / "journey-controller-admission"
        journey_result = _load_mapping(journey_result_path)
        controller_snapshot = load_validated_journey_controller_authority(
            manifest=manifest,
            shard=shard,
            bundle_path=authority_path,
        )
        authority_candidate = controller_snapshot.candidate_payload()
        if (
            row.get("evidence_hashes") != [
                controller_snapshot.bundle_digest
            ]
            or str(
                authority_candidate.get("evidence_id")
                or ""
            )
            != str(journey_result.get("evidence_id") or "")
            or row.get("schedule_result_hash") != _sha256(journey_result_path)
            or row.get("transcript_hash") != _sha256(transcript_path)
        ):
            raise ValueError(
                f"completed-batch shard evidence digest is stale: {shard.shard_id}"
            )
        sources[shard.obligation_id] = CompletedJourneySource(
            runtime_root=runtime_root,
            obligation_id=shard.obligation_id,
            shard_id=shard.shard_id,
            execution_id=str(shard.execution_id),
            controller_snapshot=controller_snapshot,
            runtime_artifacts=_capture_completed_runtime(
                runtime_root,
                authority_candidate,
            ),
            _authority=_COMPLETED_JOURNEY_SOURCE_AUTHORITY,
        )
        execution_ids[shard.obligation_id] = str(shard.execution_id)
    return manifest, sources, execution_ids, result


def _capture_completed_runtime(
    runtime_root: Path,
    controller_candidate: Mapping[str, Any],
) -> tuple[CompletedJourneyRuntimeArtifact, ...]:
    paths = {
        runtime_root / name
        for name in (
            "journey-result.json",
            "journey-schedule.json",
            "transcript.txt",
            "turn-events.jsonl",
            "checkpoints.sqlite",
        )
    }
    proof = controller_candidate.get("execution_proof")
    if isinstance(proof, Mapping) and str(proof.get("path") or ""):
        paths.add(Path(str(proof["path"])).expanduser())
    retained = controller_candidate.get("retained_artifacts")
    if isinstance(retained, Mapping):
        for reference in retained.values():
            if isinstance(reference, Mapping) and str(
                reference.get("path") or ""
            ):
                paths.add(Path(str(reference["path"])).expanduser())

    captured: list[CompletedJourneyRuntimeArtifact] = []
    root = runtime_root.resolve()
    for declared in sorted(paths, key=str):
        path = declared if declared.is_absolute() else root / declared
        if path.is_symlink():
            raise ValueError("completed Journey runtime contains a symlink")
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                "completed Journey runtime artifact is outside its root"
            )
        if not resolved.exists():
            continue
        if not resolved.is_file():
            raise ValueError(
                "completed Journey runtime artifact is not a regular file"
            )
        before = resolved.stat()
        content = resolved.read_bytes()
        after = resolved.stat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after or len(content) != after.st_size:
            raise ValueError(
                "completed Journey runtime changed during snapshot capture"
            )
        captured.append(CompletedJourneyRuntimeArtifact(
            relative_path=str(resolved.relative_to(root)),
            content=content,
            mtime_ns=after.st_mtime_ns,
        ))
    return tuple(captured)


def _source_reference(path: Path, identity_field: str, identity: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    _require_immutable_regular(resolved, description="completed-batch source")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        identity_field: identity,
    }


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"completed-batch JSON is invalid: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"completed-batch JSON must be an object: {path}")
    return dict(payload)


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_immutable_regular(path: Path, *, description: str) -> None:
    resolved = path.expanduser()
    if (
        resolved.is_symlink()
        or not resolved.is_file()
        or resolved.stat().st_mode & 0o222
    ):
        raise ValueError(f"{description} must be an immutable regular file")


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication requires renameat2")
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(
            f"completed-batch evidence destination exists: {destination}"
        )
    raise OSError(error, os.strerror(error), str(destination))
