"""Formal conversion and admission index for completed Journey batches."""

from __future__ import annotations

import hashlib
import json
import os
import ctypes
import errno
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tests.agent_live.batch_orchestrator import (
    BATCH_RESULT_SCHEMA_VERSION,
    load_frozen_manifest,
    validate_completed_shard_result,
)
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.product_obligation_evidence import (
    admit_product_obligation_evidence,
)


SCHEMA_VERSION = 1
G3_ARTIFACT_TYPE = "g3_open_completed_batch_evidence"
G4_ARTIFACT_TYPE = "g4_completed_batch_evidence"
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


def convert_completed_journey_batch(
    *,
    manifest_path: str | Path,
    result_index_path: str | Path,
    output_dir: str | Path,
    obligations: Sequence[Mapping[str, Any]],
    revision: Mapping[str, str],
    artifact_type: str,
    round_id: str,
    convert_one: Callable[[Mapping[str, Any], Path, Path, Path | None], Path],
) -> Path:
    """Convert every passed Journey shard and publish one declared evidence index."""

    expected = {
        str(row.get("obligation_id") or ""): dict(row)
        for row in obligations
        if str(row.get("obligation_id") or "")
    }
    if not expected or len(expected) != len(obligations):
        raise ValueError("completed-batch obligations are incomplete or duplicated")
    manifest, runtime_roots, execution_ids, result_index = _validated_completed_batch(
        manifest_path=manifest_path,
        result_index_path=result_index_path,
        expected_obligation_ids=set(expected),
        revision=revision,
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
            runtime_roots[obligation_id],
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
    for path in staging.rglob("*"):
        path.chmod(0o500 if path.is_dir() else 0o400)
    staging.chmod(0o500)
    _rename_directory_noreplace(staging, destination)
    return destination / "manifest.json"


def load_completed_journey_batch_evidence(
    index_path: str | Path,
    *,
    obligations: Sequence[Mapping[str, Any]],
    revision: Mapping[str, str],
    artifact_type: str,
    round_id: str,
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
    unsigned = {key: value for key, value in payload.items() if key != "index_hash"}
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("artifact_type") != artifact_type
        or dict(payload.get("revision_binding") or {}) != dict(revision)
        or str(payload.get("round_id") or "") != str(round_id)
        or payload.get("index_hash") != content_hash(unsigned)
    ):
        raise ValueError("completed-batch evidence index contract is invalid")
    manifest_ref = payload.get("batch_manifest")
    result_ref = payload.get("batch_result")
    if not isinstance(manifest_ref, Mapping) or not isinstance(result_ref, Mapping):
        raise ValueError("completed-batch source references are invalid")
    manifest_source = Path(str(manifest_ref.get("path") or ""))
    result_source = Path(str(result_ref.get("path") or ""))
    manifest, runtime_roots, execution_ids, result_index = (
        _validated_completed_batch(
        manifest_path=manifest_source,
        result_index_path=result_source,
        expected_obligation_ids={
            str(row.get("obligation_id") or "") for row in obligations
        },
        revision=revision,
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
        runtime_root = runtime_roots.get(obligation_id)
        artifact_rows = evidence_document.get("artifacts")
        if (
            evidence_document.get("obligation_id") != obligation_id
            or evidence_document.get("obligation_contract_hash")
            != row.get("obligation_contract_hash")
            or evidence_document.get("evidence_id") != row.get("evidence_id")
            or not isinstance(execution, Mapping)
            or execution.get("execution_id") != row.get("execution_id")
            or execution.get("execution_id") != execution_ids.get(obligation_id)
            or runtime_root is None
            or not isinstance(artifact_rows, list)
        ):
            raise ValueError("completed-batch evidence row binding is invalid")
        resolved_runtime = runtime_root.resolve()
        for artifact_row in artifact_rows:
            if not isinstance(artifact_row, Mapping):
                raise ValueError(
                    "completed-batch runtime artifact binding is invalid"
                )
            artifact_path = Path(str(artifact_row.get("path") or ""))
            if artifact_path.is_symlink():
                raise ValueError(
                    "completed-batch runtime artifact binding is invalid"
                )
            resolved_artifact = artifact_path.resolve()
            if (
                not resolved_artifact.is_relative_to(resolved_runtime)
                or not resolved_artifact.is_file()
            ):
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
) -> tuple[Any, dict[str, Path], dict[str, str], dict[str, Any]]:
    _require_immutable_regular(
        Path(manifest_path),
        description="completed-batch manifest",
    )
    _require_immutable_regular(
        Path(result_index_path),
        description="completed-batch result index",
    )
    manifest = load_frozen_manifest(manifest_path)
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
    unsigned = {key: value for key, value in result.items() if key != "index_id"}
    if (
        result.get("schema_version") != BATCH_RESULT_SCHEMA_VERSION
        or result.get("index_id") != content_hash(unsigned)
        or result.get("batch_id") != manifest.batch_id
        or result.get("manifest_id") != manifest.manifest_id
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
    roots: dict[str, Path] = {}
    execution_ids: dict[str, str] = {}
    for shard in shards:
        row = by_shard.get(shard.shard_id)
        if (
            row is None
            or row.get("classification") != "passed"
            or row.get("target_hash") != shard.target_hash
            or int(row.get("finished_at_ns") or 0)
            <= int(row.get("started_at_ns") or 0)
            or not row.get("response_hashes")
            or len(row.get("response_hashes") or ())
            != len(row.get("decision_hashes") or ())
            or not row.get("evidence_hashes")
        ):
            raise ValueError(
                f"completed-batch shard is not qualifying: {shard.shard_id}"
            )
        validate_completed_shard_result(shard, row)
        roots[shard.obligation_id] = Path(shard.runtime_root).resolve()
        execution_ids[shard.obligation_id] = str(shard.execution_id)
    return manifest, roots, execution_ids, result


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
