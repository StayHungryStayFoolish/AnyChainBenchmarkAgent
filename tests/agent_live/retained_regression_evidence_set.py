"""Immutable aggregation for the complete Phase 8 G3 evidence set.

The exact and response-driven producers retain their own runtime artifacts.
This module is the sole collection publisher: it validates both producer
indexes and all 60 product-evidence documents before atomically publishing one
manifest that binds the complete set.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.agent_live.coverage_evidence import (
    content_hash,
    verify_controller_payload_signature,
)
from tests.agent_live.batch_orchestrator import (
    BATCH_MANIFEST_SCHEMA_VERSION,
    BATCH_RESULT_SCHEMA_VERSION,
)
from tests.agent_live.product_obligation_evidence import (
    admit_product_obligation_evidence,
)
from tests.agent_live.completed_journey_batch import (
    G3_ARTIFACT_TYPE,
    load_completed_journey_batch_evidence,
)
from tests.agent_live.retained_regression_runner import (
    RETAINED_REGRESSION_EXACT_COUNT,
    RETAINED_REGRESSION_OPEN_COUNT,
    load_retained_regression_provider,
    validate_retained_regression_runner_provider,
)


RETAINED_REGRESSION_EVIDENCE_SET_SCHEMA_VERSION = 1
RETAINED_REGRESSION_EVIDENCE_SET_ARTIFACT_TYPE = (
    "retained_regression_evidence_set"
)
EXACT_RUNNER = "retained-regression-real-cli-v1"
OPEN_RUNNER = "retained-regression-response-driven-journey-v1"
TOTAL_EVIDENCE_COUNT = (
    RETAINED_REGRESSION_EXACT_COUNT + RETAINED_REGRESSION_OPEN_COUNT
)
_MANIFEST_NAME = "manifest.json"
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


def publish_retained_regression_evidence_set(
    *,
    output_dir: str | Path,
    provider: Mapping[str, Any],
    obligations: Sequence[Mapping[str, Any]],
    revision: Mapping[str, str],
    exact_execution_index_path: str | Path,
    open_batch_manifest_path: str | Path,
    open_batch_result_path: str | Path,
    evidence_paths: Sequence[str | Path],
    expected_authority_trust_root_id: str,
) -> Path:
    """Validate and atomically publish one complete immutable G3 collection."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"G3 evidence-set destination is immutable: {destination}"
        )
    manifest = _build_validated_manifest(
        provider=provider,
        obligations=obligations,
        revision=revision,
        exact_execution_index_path=exact_execution_index_path,
        open_batch_manifest_path=open_batch_manifest_path,
        open_batch_result_path=open_batch_result_path,
        evidence_paths=evidence_paths,
        expected_authority_trust_root_id=expected_authority_trust_root_id,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f".{destination.name}.staging-{uuid.uuid4().hex}"
    )
    staging.mkdir(mode=0o700)
    manifest_path = staging / _MANIFEST_NAME
    _write_fsynced_json(manifest_path, manifest)
    _fsync_directory(staging)
    _make_tree_read_only(staging)
    _fsync_directory(staging)
    _rename_directory_noreplace(staging, destination)
    _fsync_directory(destination.parent)
    _fsync_directory(destination)
    _fsync_directory(destination.parent)
    return destination / _MANIFEST_NAME


def load_retained_regression_evidence_set(
    manifest_path: str | Path,
    *,
    provider: Mapping[str, Any],
    obligations: Sequence[Mapping[str, Any]],
    revision: Mapping[str, str],
    expected_authority_trust_root_id: str,
) -> dict[str, Any]:
    """Load only the collection declared by one immutable G3 manifest."""

    path = Path(manifest_path).expanduser().resolve()
    if path.name != _MANIFEST_NAME or not path.is_file():
        raise ValueError("G3 evidence-set manifest is missing")
    root = path.parent
    if path.stat().st_mode & 0o222 or root.stat().st_mode & 0o222:
        raise ValueError("G3 evidence-set must be immutable")
    declared_names = {_MANIFEST_NAME}
    observed_names = {item.name for item in root.iterdir()}
    if observed_names != declared_names:
        raise ValueError("G3 evidence-set directory contains undeclared files")

    manifest = _load_mapping(path, "G3 evidence-set manifest")
    recorded_hash = str(manifest.get("manifest_hash") or "")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if not _is_sha256(recorded_hash) or content_hash(unsigned) != recorded_hash:
        raise ValueError("G3 evidence-set manifest hash drifted")
    rebuilt = _build_validated_manifest(
        provider=provider,
        obligations=obligations,
        revision=revision,
        exact_execution_index_path=_source_path(
            manifest, "exact_execution_index"
        ),
        open_batch_manifest_path=_source_path(
            manifest, "open_batch_manifest"
        ),
        open_batch_result_path=_source_path(manifest, "open_batch_result"),
        evidence_paths=[
            str(dict(row)["path"])
            for row in _required_rows(manifest, "evidence")
        ],
        expected_authority_trust_root_id=expected_authority_trust_root_id,
    )
    if rebuilt != manifest:
        raise ValueError("G3 evidence-set manifest differs from validated sources")
    return manifest


def _build_validated_manifest(
    *,
    provider: Mapping[str, Any],
    obligations: Sequence[Mapping[str, Any]],
    revision: Mapping[str, str],
    exact_execution_index_path: str | Path,
    open_batch_manifest_path: str | Path,
    open_batch_result_path: str | Path,
    evidence_paths: Sequence[str | Path],
    expected_authority_trust_root_id: str,
) -> dict[str, Any]:
    active_revision = _validated_revision(revision)
    validate_retained_regression_runner_provider(
        provider,
        obligations=obligations,
        revision=active_revision,
    )
    provider_hash = _required_sha256(provider, "provider_hash")
    obligation_index, obligation_set_hash = _obligation_contract(
        obligations, active_revision
    )

    exact_path = _resolved_immutable_file(
        exact_execution_index_path, "G3 exact execution index"
    )
    open_manifest_path = _resolved_immutable_file(
        open_batch_manifest_path, "G3 open batch manifest"
    )
    open_result_path = _resolved_immutable_file(
        open_batch_result_path, "G3 open batch result"
    )
    exact_index = _load_mapping(exact_path, "G3 exact execution index")
    open_manifest = _load_mapping(open_manifest_path, "G3 open batch manifest")
    open_result = _load_mapping(open_result_path, "G3 open batch result")

    exact_sources = _validate_exact_index(
        exact_index,
        provider_hash=provider_hash,
        revision=active_revision,
        obligations=obligation_index,
    )
    open_sources = _validate_open_batch(
        manifest=open_manifest,
        result=open_result,
        manifest_path=open_manifest_path.resolve(),
        revision=active_revision,
        obligations=obligation_index,
        provider_targets={
            str(row.get("obligation_id") or ""): dict(row)
            for row in provider.get("targets") or ()
            if isinstance(row, Mapping)
        },
        expected_authority_trust_root_id=expected_authority_trust_root_id,
    )

    resolved_evidence = tuple(
        _resolved_file(value, "G3 product evidence") for value in evidence_paths
    )
    if len(resolved_evidence) != TOTAL_EVIDENCE_COUNT:
        raise ValueError("G3 evidence-set requires exactly 60 evidence documents")
    if len(set(resolved_evidence)) != len(resolved_evidence):
        raise ValueError("G3 evidence-set contains duplicate evidence paths")
    admitted = admit_product_obligation_evidence(
        obligations=obligations,
        evidence_paths=resolved_evidence,
        revision=active_revision,
    )
    if (
        admitted.get("complete") is not True
        or admitted.get("passed") != TOTAL_EVIDENCE_COUNT
        or admitted.get("failed") != 0
        or admitted.get("external") != 0
        or admitted.get("not_run") != 0
    ):
        raise ValueError("G3 evidence-set does not contain 60 qualifying passes")

    evidence_rows = _validated_evidence_rows(
        resolved_evidence,
        obligations=obligation_index,
        exact_sources=exact_sources,
        open_sources=open_sources,
    )
    unsigned = {
        "schema_version": RETAINED_REGRESSION_EVIDENCE_SET_SCHEMA_VERSION,
        "artifact_type": RETAINED_REGRESSION_EVIDENCE_SET_ARTIFACT_TYPE,
        "revision_binding": active_revision,
        "provider_id": str(provider.get("provider_id") or ""),
        "provider_hash": provider_hash,
        "obligation_set_hash": obligation_set_hash,
        "obligation_count": TOTAL_EVIDENCE_COUNT,
        "exact_count": RETAINED_REGRESSION_EXACT_COUNT,
        "open_count": RETAINED_REGRESSION_OPEN_COUNT,
        "authority_trust_root_id": expected_authority_trust_root_id,
        "exact_execution_index": _source_reference(
            exact_path,
            identity_field="index_hash",
            identity_value=str(exact_index["index_hash"]),
        ),
        "open_batch_manifest": _source_reference(
            open_manifest_path,
            identity_field="manifest_id",
            identity_value=str(open_manifest["manifest_id"]),
        ),
        "open_batch_result": _source_reference(
            open_result_path,
            identity_field="index_id",
            identity_value=str(open_result["index_id"]),
        ),
        "evidence": evidence_rows,
    }
    return {**unsigned, "manifest_hash": content_hash(unsigned)}


def _obligation_contract(
    obligations: Sequence[Mapping[str, Any]],
    revision: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], str]:
    rows = [dict(row) for row in obligations]
    if len(rows) != TOTAL_EVIDENCE_COUNT:
        raise ValueError("G3 obligation set must contain exactly 60 rows")
    index: dict[str, dict[str, Any]] = {}
    identity_rows: list[dict[str, str]] = []
    for row in rows:
        obligation_id = _required_text(row, "obligation_id")
        variant = _required_text(row, "variant")
        contract_hash = _required_sha256(row, "contract_hash")
        if obligation_id in index:
            raise ValueError("G3 obligation set contains duplicate ids")
        row_revision = dict(row.get("revision_binding") or {}).get("revision")
        if row_revision != dict(revision):
            raise ValueError(f"G3 obligation revision drifted: {obligation_id}")
        index[obligation_id] = row
        identity_rows.append({
            "obligation_id": obligation_id,
            "variant": variant,
            "contract_hash": contract_hash,
        })
    exact_count = sum(row["variant"] == "exact" for row in rows)
    if exact_count != RETAINED_REGRESSION_EXACT_COUNT:
        raise ValueError("G3 obligation set does not contain 15 exact variants")
    if len(rows) - exact_count != RETAINED_REGRESSION_OPEN_COUNT:
        raise ValueError("G3 obligation set does not contain 45 open variants")
    return index, content_hash(sorted(
        identity_rows, key=lambda row: row["obligation_id"]
    ))


def _validate_exact_index(
    index: Mapping[str, Any],
    *,
    provider_hash: str,
    revision: Mapping[str, str],
    obligations: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[Path, str]]:
    unsigned = {key: value for key, value in index.items() if key != "index_hash"}
    if (
        index.get("schema_version") != 1
        or index.get("artifact_type")
        != "retained_regression_exact_execution_index"
        or index.get("provider_hash") != provider_hash
        or index.get("revision_binding") != revision
        or index.get("selection") != "all"
        or index.get("scheduled") != RETAINED_REGRESSION_EXACT_COUNT
        or index.get("completed") != RETAINED_REGRESSION_EXACT_COUNT
        or index.get("index_hash") != content_hash(unsigned)
    ):
        raise ValueError("G3 exact execution index is invalid")
    rows = _required_rows(index, "evidence")
    exact_ids = {
        obligation_id
        for obligation_id, row in obligations.items()
        if row.get("variant") == "exact"
    }
    if len(rows) != RETAINED_REGRESSION_EXACT_COUNT:
        raise ValueError("G3 exact execution index must contain 15 rows")
    sources: dict[str, tuple[Path, str]] = {}
    for raw in rows:
        row = dict(raw)
        if set(row) != {"obligation_id", "path", "sha256"}:
            raise ValueError("G3 exact execution row is invalid")
        obligation_id = _required_text(row, "obligation_id")
        path = _resolved_file(row["path"], "G3 exact evidence")
        sha256 = _required_sha256(row, "sha256")
        if (
            obligation_id not in exact_ids
            or obligation_id in sources
            or _sha256_file(path) != sha256
        ):
            raise ValueError("G3 exact execution evidence drifted")
        sources[obligation_id] = (path, sha256)
    if set(sources) != exact_ids:
        raise ValueError("G3 exact execution index obligation set drifted")
    return sources


def _validate_open_batch(
    *,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    manifest_path: Path,
    revision: Mapping[str, str],
    obligations: Mapping[str, Mapping[str, Any]],
    provider_targets: Mapping[str, Mapping[str, Any]],
    expected_authority_trust_root_id: str,
) -> dict[str, str]:
    manifest_unsigned = {
        key: value
        for key, value in manifest.items()
        if key not in {"batch_id", "manifest_id"}
    }
    batch_id = content_hash(manifest_unsigned)
    manifest_id = content_hash({**manifest_unsigned, "batch_id": batch_id})
    shards = _required_rows(manifest, "shards")
    if (
        not expected_authority_trust_root_id
        or manifest.get("pty_authority_trust_root_id")
        != expected_authority_trust_root_id
    ):
        raise ValueError("G3 open batch trust root is not externally trusted")
    open_ids = {
        obligation_id
        for obligation_id, row in obligations.items()
        if row.get("variant") != "exact"
    }
    if (
        manifest.get("schema_version") != BATCH_MANIFEST_SCHEMA_VERSION
        or manifest.get("batch_id") != batch_id
        or manifest.get("manifest_id") != manifest_id
        or manifest.get("revision") != revision
        or manifest.get("manifest_path") != str(manifest_path)
        or manifest.get("controller_owned_execution") is not True
        or manifest.get("shard_count") != RETAINED_REGRESSION_OPEN_COUNT
        or len(shards) != RETAINED_REGRESSION_OPEN_COUNT
        or manifest.get("worker_runtime") not in {"docker", "linux"}
    ):
        raise ValueError("G3 open batch manifest is invalid")

    shard_ids: set[str] = set()
    execution_by_obligation: dict[str, str] = {}
    for raw in shards:
        row = dict(raw)
        obligation_id = _required_text(row, "obligation_id")
        shard_id = _required_text(row, "shard_id")
        execution_id = _required_text(row, "execution_id")
        target = provider_targets.get(obligation_id)
        journey_definition = (
            target.get("journey_definition")
            if isinstance(target, Mapping)
            else None
        )
        frozen_execution = (
            journey_definition.get("frozen_execution")
            if isinstance(journey_definition, Mapping)
            else None
        )
        if (
            row.get("lane") != "journey"
            or obligation_id not in open_ids
            or obligation_id in execution_by_obligation
            or shard_id in shard_ids
            or not isinstance(frozen_execution, Mapping)
            or {
                "schedule_id": row.get("schedule_id"),
                "seed": row.get("seed"),
                "subject_group": row.get("subject_group"),
            }
            != {
                "schedule_id": frozen_execution.get("schedule_id"),
                "seed": frozen_execution.get("seed"),
                "subject_group": frozen_execution.get("subject_group"),
            }
        ):
            raise ValueError("G3 open batch shard contract drifted")
        shard_ids.add(shard_id)
        execution_by_obligation[obligation_id] = execution_id
    if set(execution_by_obligation) != open_ids:
        raise ValueError("G3 open batch obligation set drifted")
    observed_set_hash = content_hash([
        {
            "obligation_id": str(row["obligation_id"]),
            "schedule_id": str(row["schedule_id"]),
            "seed": int(row["seed"]),
            "subject_group": str(row["subject_group"]),
        }
        for row in sorted(
            (dict(item) for item in shards),
            key=lambda item: (
                str(item["obligation_id"]),
                str(item["schedule_id"]),
            ),
        )
    ])
    if manifest.get("expected_obligation_set_hash") != observed_set_hash:
        raise ValueError("G3 open batch obligation-set hash drifted")

    result_signed_payload = {
        key: value
        for key, value in result.items()
        if key not in {"index_id", "controller_signature_b64"}
    }
    result_signed_index = {
        **result_signed_payload,
        "controller_signature_b64": str(
            result.get("controller_signature_b64") or ""
        ),
    }
    result_rows = _required_rows(result, "shards")
    result_shard_ids = {
        _required_text(dict(row), "shard_id") for row in result_rows
    }
    if (
        result.get("schema_version") != BATCH_RESULT_SCHEMA_VERSION
        or result.get("index_id") != content_hash(result_signed_index)
        or result.get("batch_id") != batch_id
        or result.get("manifest_id") != manifest_id
        or result.get("execution_authority_mode") != "controller_owned_v1"
        or result.get("revision") != revision
        or result.get("pty_authority_trust_root_id")
        != manifest.get("pty_authority_trust_root_id")
        or result.get("pty_authority_public_key_b64")
        != manifest.get("pty_authority_public_key_b64")
        or not verify_controller_payload_signature(
            result_signed_payload,
            signature_b64=str(
                result.get("controller_signature_b64") or ""
            ),
            trusted_public_key_b64=str(
                manifest.get("pty_authority_public_key_b64") or ""
            ),
        )
        or result.get("execution_status") != "discovery_complete"
        or result.get("scheduled") != RETAINED_REGRESSION_OPEN_COUNT
        or result.get("started") != RETAINED_REGRESSION_OPEN_COUNT
        or result.get("completed") != RETAINED_REGRESSION_OPEN_COUNT
        or len(result_rows) != RETAINED_REGRESSION_OPEN_COUNT
        or result_shard_ids != shard_ids
        or len(result_shard_ids) != RETAINED_REGRESSION_OPEN_COUNT
        or any(dict(row).get("classification") != "passed" for row in result_rows)
    ):
        raise ValueError("G3 open batch result is invalid")
    return execution_by_obligation


def _validated_evidence_rows(
    paths: Sequence[Path],
    *,
    obligations: Mapping[str, Mapping[str, Any]],
    exact_sources: Mapping[str, tuple[Path, str]],
    open_sources: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        payload = _load_mapping(path, "G3 product evidence")
        obligation_id = _required_text(payload, "obligation_id")
        obligation = obligations.get(obligation_id)
        if obligation is None or obligation_id in seen:
            raise ValueError("G3 product evidence obligation set drifted")
        seen.add(obligation_id)
        variant = _required_text(obligation, "variant")
        execution = payload.get("execution")
        if not isinstance(execution, Mapping):
            raise ValueError("G3 product evidence execution is missing")
        runner = _required_text(execution, "runner")
        execution_id = _required_text(execution, "execution_id")
        expected_runner = EXACT_RUNNER if variant == "exact" else OPEN_RUNNER
        if runner != expected_runner:
            raise ValueError("G3 evidence variant/runner substitution detected")
        source_sha256 = _sha256_file(path)
        if variant == "exact":
            source = exact_sources.get(obligation_id)
            if source != (path, source_sha256):
                raise ValueError("G3 exact evidence differs from its execution index")
        elif open_sources.get(obligation_id) != execution_id:
            raise ValueError("G3 open evidence differs from its batch execution")
        rows.append({
            "obligation_id": obligation_id,
            "variant": variant,
            "runner": runner,
            "execution_id": execution_id,
            "evidence_id": _required_sha256(payload, "evidence_id"),
            "evidence_hash": _required_sha256(payload, "evidence_hash"),
            "path": str(path),
            "sha256": source_sha256,
        })
    if seen != set(obligations):
        raise ValueError("G3 evidence-set is missing obligations")
    return sorted(rows, key=lambda row: row["obligation_id"])


def _source_reference(
    path: Path,
    *,
    identity_field: str,
    identity_value: str,
) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        identity_field: identity_value,
    }


def _source_path(manifest: Mapping[str, Any], field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"G3 evidence-set source is missing: {field}")
    return _required_text(value, "path")


def _required_rows(value: Mapping[str, Any], field: str) -> list[Mapping[str, Any]]:
    rows = value.get(field)
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or not all(isinstance(row, Mapping) for row in rows)
    ):
        raise ValueError(f"G3 collection field is invalid: {field}")
    return list(rows)


def _validated_revision(value: Mapping[str, str]) -> dict[str, str]:
    revision = {
        "commit": str(value.get("commit") or "").strip(),
        "worktree_hash": str(value.get("worktree_hash") or "").strip(),
    }
    if not all(revision.values()):
        raise ValueError("G3 evidence-set requires a revision binding")
    return revision


def _resolved_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _resolved_immutable_file(value: str | Path, label: str) -> Path:
    path = _resolved_file(value, label)
    if path.stat().st_mode & 0o222:
        raise ValueError(f"{label} must be immutable: {path}")
    return path


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one object: {path}")
    return dict(payload)


def _required_text(value: Mapping[str, Any], field: str) -> str:
    text = str(value.get(field) or "").strip()
    if not text:
        raise ValueError(f"G3 collection field is missing: {field}")
    return text


def _required_sha256(value: Mapping[str, Any], field: str) -> str:
    digest = _required_text(value, field)
    if not _is_sha256(digest):
        raise ValueError(f"G3 collection hash is invalid: {field}")
    return digest


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 64
        and text == text.lower()
        and all(character in "0123456789abcdef" for character in text)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_fsynced_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o400,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(
            "G3 evidence-set publication requires Linux renameat2"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
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
        raise FileExistsError(error, os.strerror(error), str(destination))
    raise OSError(error, os.strerror(error), str(destination))


def _make_tree_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
        elif path.is_dir():
            path.chmod(0o500)
    root.chmod(0o500)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish one immutable complete G3 evidence set.",
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--provider", required=True, type=Path)
    parser.add_argument("--exact-index", required=True, type=Path)
    parser.add_argument("--open-batch-manifest", required=True, type=Path)
    parser.add_argument("--open-batch-result", required=True, type=Path)
    parser.add_argument("--open-evidence-index", required=True, type=Path)
    parser.add_argument("--authority-trust-root-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    provider, obligations, revision = load_retained_regression_provider(
        repo_root=args.repo_root,
        provider_path=args.provider,
    )
    exact_index = _load_mapping(
        args.exact_index.resolve(),
        "G3 exact execution index",
    )
    exact_rows = _required_rows(exact_index, "evidence")
    if len(exact_rows) != RETAINED_REGRESSION_EXACT_COUNT:
        raise ValueError("G3 exact execution index does not declare 15 rows")
    exact_paths: list[Path] = []
    for row in exact_rows:
        path = _resolved_file(str(row.get("path") or ""), "G3 exact evidence")
        if _sha256_file(path) != str(row.get("sha256") or ""):
            raise ValueError("G3 exact evidence hash differs from its index")
        exact_paths.append(path)
    open_obligations = tuple(
        row for row in obligations if row.get("variant") != "exact"
    )
    open_paths = load_completed_journey_batch_evidence(
        args.open_evidence_index,
        obligations=open_obligations,
        revision=revision,
        artifact_type=G3_ARTIFACT_TYPE,
        round_id="",
        expected_authority_trust_root_id=args.authority_trust_root_id,
    )
    publish_retained_regression_evidence_set(
        output_dir=args.output_dir,
        provider=provider,
        obligations=obligations,
        revision=revision,
        exact_execution_index_path=args.exact_index,
        open_batch_manifest_path=args.open_batch_manifest,
        open_batch_result_path=args.open_batch_result,
        evidence_paths=(*exact_paths, *open_paths),
        expected_authority_trust_root_id=args.authority_trust_root_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
