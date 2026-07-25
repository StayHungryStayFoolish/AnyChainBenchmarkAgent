"""Immutable collection authority for one G5 real-execution attempt."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.agent_live.coverage_evidence import G5_SCENARIO_ADMISSION


MANIFEST_SCHEMA_VERSION = 1
POINTER_SCHEMA_VERSION = 1
MANIFEST_ARTIFACT_TYPE = "g5_real_execution_collection"
POINTER_ARTIFACT_TYPE = "g5_real_execution_collection_pointer"
SCENARIOS = tuple(
    scenario_id
    for scenario_id, _admission in sorted(
        G5_SCENARIO_ADMISSION.items(),
        key=lambda item: item[1].sequence_index,
    )
)


def create_g5_attempt(evidence_root: Path) -> tuple[str, Path]:
    attempt_id = uuid.uuid4().hex
    attempt_dir = evidence_root.resolve() / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return attempt_id, attempt_dir


def publish_g5_collection(
    *,
    g5_root: Path,
    attempt_id: str,
    revision: Mapping[str, str],
    evidence_paths: Sequence[Path],
    status: str,
    failure_path: Path | None = None,
) -> Path:
    if status not in {"complete", "failed"}:
        raise ValueError("G5 collection status must be complete or failed")
    root = g5_root.resolve()
    attempt_root = (root / "evidence" / _require_attempt_id(attempt_id)).resolve()
    evidence = [
        _evidence_binding(path, attempt_root=attempt_root)
        for path in evidence_paths
    ]
    observed = tuple(str(item["scenario_id"]) for item in evidence)
    if observed != SCENARIOS[: len(observed)]:
        raise ValueError("G5 evidence must form the ordered scenario prefix")
    if status == "complete" and observed != SCENARIOS:
        raise ValueError("complete G5 collection requires all four scenarios")

    payload: dict[str, Any] = {
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "attempt_id": _require_attempt_id(attempt_id),
        "evidence_root": str(attempt_root),
        "repository_revision": dict(revision),
        "status": status,
        "scenario_order": list(SCENARIOS),
        "evidence": evidence,
        "failure_evidence": (
            _file_binding(failure_path, attempt_root=attempt_root)
            if failure_path is not None
            else None
        ),
    }
    payload["collection_id"] = _canonical_hash(payload)
    serialized = _serialized(payload)
    digest = hashlib.sha256(serialized).hexdigest()
    manifest = root / "collections" / f"g5-collection-{digest}.json"
    _write_immutable(manifest, serialized)

    pointer = {
        "artifact_type": POINTER_ARTIFACT_TYPE,
        "schema_version": POINTER_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "repository_revision": dict(revision),
        "manifest_path": str(manifest),
        "manifest_sha256": digest,
    }
    _replace_json(root / "active-collection.json", pointer)
    return manifest


def load_active_g5_collection(
    g5_root: Path,
    *,
    revision: Mapping[str, str],
) -> tuple[dict[str, Any] | None, str]:
    root = g5_root.resolve()
    pointer_path = root / "active-collection.json"
    try:
        pointer = _read_object(pointer_path, "G5 collection pointer")
    except (OSError, RuntimeError, ValueError) as exc:
        return None, str(exc)
    if (
        pointer.get("artifact_type") != POINTER_ARTIFACT_TYPE
        or pointer.get("schema_version") != POINTER_SCHEMA_VERSION
    ):
        return None, "G5 collection pointer schema is invalid"
    if dict(pointer.get("repository_revision") or {}) != dict(revision):
        return None, "G5 collection pointer revision mismatch"
    try:
        attempt_id = _require_attempt_id(str(pointer.get("attempt_id") or ""))
        manifest = Path(str(pointer.get("manifest_path") or "")).resolve()
        collections_root = (root / "collections").resolve()
        if manifest.parent != collections_root:
            raise ValueError("G5 manifest is outside the collection authority")
        digest = str(pointer.get("manifest_sha256") or "")
        if (
            manifest.is_symlink()
            or not manifest.is_file()
            or manifest.stat().st_mode & 0o222
            or _sha256(manifest) != digest
            or manifest.name != f"g5-collection-{digest}.json"
        ):
            raise ValueError("G5 manifest binding is invalid")
        payload = _read_object(manifest, "G5 collection manifest")
        _validate_manifest(
            payload,
            attempt_id=attempt_id,
            revision=revision,
            attempt_root=(root / "evidence" / attempt_id).resolve(),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return None, str(exc)
    return payload, ""


def _validate_manifest(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
    revision: Mapping[str, str],
    attempt_root: Path,
) -> None:
    if (
        payload.get("artifact_type") != MANIFEST_ARTIFACT_TYPE
        or payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("G5 collection manifest schema is invalid")
    if payload.get("attempt_id") != attempt_id:
        raise ValueError("G5 collection attempt identity mismatch")
    expected_attempt_root = (
        Path(str(payload.get("evidence_root") or "")).resolve()
    )
    if expected_attempt_root != attempt_root:
        raise ValueError("G5 collection evidence root is invalid")
    if dict(payload.get("repository_revision") or {}) != dict(revision):
        raise ValueError("G5 collection manifest revision mismatch")
    if tuple(payload.get("scenario_order") or ()) != SCENARIOS:
        raise ValueError("G5 collection scenario registry mismatch")
    status = str(payload.get("status") or "")
    if status not in {"complete", "failed"}:
        raise ValueError("G5 collection status is invalid")
    evidence = list(payload.get("evidence") or ())
    observed: list[str] = []
    for sequence, item in enumerate(evidence, start=1):
        if not isinstance(item, Mapping):
            raise ValueError("G5 evidence binding shape is invalid")
        if int(item.get("sequence") or 0) != sequence:
            raise ValueError("G5 evidence sequence is invalid")
        _validate_file_binding(item, attempt_root=expected_attempt_root)
        observed.append(str(item.get("scenario_id") or ""))
    if tuple(observed) != SCENARIOS[: len(observed)]:
        raise ValueError("G5 evidence is not the legal scenario prefix")
    if status == "complete" and tuple(observed) != SCENARIOS:
        raise ValueError("complete G5 collection is missing scenarios")
    if status == "complete" and payload.get("failure_evidence") is not None:
        raise ValueError("complete G5 collection cannot bind failure evidence")
    failure = payload.get("failure_evidence")
    if failure is not None:
        if not isinstance(failure, Mapping):
            raise ValueError("G5 failure binding shape is invalid")
        _validate_file_binding(failure, attempt_root=expected_attempt_root)
    unsigned = dict(payload)
    collection_id = str(unsigned.pop("collection_id", ""))
    if _canonical_hash(unsigned) != collection_id:
        raise ValueError("G5 collection identity mismatch")


def _evidence_binding(
    path: Path,
    *,
    attempt_root: Path,
) -> dict[str, Any]:
    payload = _read_object(path, "G5 real-execution evidence")
    scenario_id = str(payload.get("scenario_id") or "")
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown G5 scenario evidence: {scenario_id}")
    binding = _file_binding(path, attempt_root=attempt_root)
    binding.update(
        sequence=SCENARIOS.index(scenario_id) + 1,
        scenario_id=scenario_id,
        evidence_id=str(payload.get("evidence_id") or ""),
    )
    return binding


def _file_binding(
    path: Path,
    *,
    attempt_root: Path,
) -> dict[str, Any]:
    resolved = path.resolve()
    if (
        path.is_symlink()
        or not resolved.is_file()
        or resolved.parent != attempt_root
    ):
        raise ValueError(f"G5 artifact is missing or symlinked: {path}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
    }


def _validate_file_binding(
    item: Mapping[str, Any],
    *,
    attempt_root: Path,
) -> None:
    path = Path(str(item.get("path") or ""))
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve().parent != attempt_root
        or path.stat().st_mode & 0o222
        or _sha256(path) != str(item.get("sha256") or "")
    ):
        raise ValueError("G5 artifact binding is invalid")


def _require_attempt_id(value: str) -> str:
    if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("G5 attempt identity is invalid")
    return value


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or symlinked")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _serialized(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    content = _serialized(payload)
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
