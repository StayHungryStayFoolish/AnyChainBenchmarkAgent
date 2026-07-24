"""Shared validation for host-observed Docker runtime attestations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.agent_live.real_execution_host_supervisor import (
    ATTESTATION_SCHEMA_VERSION,
    BENCH_SERVICE,
    LINUX_WORKER_PREFIX,
    TARGET_SERVICE,
)


def validate_host_attestation_file(
    path: str | Path,
    *,
    repo_root: str | Path,
    revision: Mapping[str, str],
    worker_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    raw_path = Path(path)
    if raw_path.is_symlink():
        raise ValueError("G5 host attestation cannot be a symlink")
    attestation_path = raw_path.resolve(strict=True)
    repository = Path(repo_root).resolve(strict=True)
    try:
        attestation_path.relative_to(repository)
    except ValueError as exc:
        raise ValueError("G5 host attestation is outside the repository") from exc
    if not attestation_path.is_file():
        raise ValueError("G5 host attestation is not a regular file")

    serialized = attestation_path.read_bytes()
    observed_hash = hashlib.sha256(serialized).hexdigest()
    expected_name = f"real-execution-host-attestation-{observed_hash}.json"
    if attestation_path.name != expected_name:
        raise ValueError("G5 host attestation filename is not content-addressed")
    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise ValueError("G5 host attestation is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("G5 host attestation must be a JSON object")
    attestation = dict(payload)
    if (
        attestation.get("schema_version") != ATTESTATION_SCHEMA_VERSION
        or attestation.get("artifact_type") != "real_execution_host_attestation"
        or dict(attestation.get("repository_revision") or {}) != dict(revision)
    ):
        raise ValueError("G5 host attestation identity does not match this execution")

    assurance = dict(attestation.get("assurance") or {})
    if (
        assurance.get("kind") != "docker_runtime_observation"
        or assurance.get("cryptographic_identity") is not False
    ):
        raise ValueError("G5 host attestation assurance contract is invalid")
    compose = dict(attestation.get("compose") or {})
    worker_container = dict(attestation.get("worker_container") or {})
    target_container = dict(attestation.get("target_container") or {})
    project = str(compose.get("project") or "")
    if (
        not project
        or compose.get("worker_service") != BENCH_SERVICE
        or compose.get("target_service") != TARGET_SERVICE
        or worker_container.get("compose_project") != project
        or worker_container.get("compose_service") != BENCH_SERVICE
        or target_container.get("compose_project") != project
        or target_container.get("compose_service") != TARGET_SERVICE
    ):
        raise ValueError("G5 host attestation compose identity is invalid")
    _validate_container_contract(worker_container, label="worker")
    _validate_container_contract(target_container, label="target")

    worker = dict(attestation.get("worker") or {})
    observed_argv = [str(item) for item in worker.get("argv") or ()]
    if tuple(observed_argv[: len(LINUX_WORKER_PREFIX)]) != LINUX_WORKER_PREFIX:
        raise ValueError("G5 host attestation worker command is invalid")
    if worker_argv is not None:
        expected_argv = [*LINUX_WORKER_PREFIX, *map(str, worker_argv)]
        if observed_argv != expected_argv:
            raise ValueError(
                "G5 host attestation worker argv does not match this process"
            )
    expected_argv_hash = _content_hash({"argv": observed_argv}, newline=True)
    if (
        worker.get("argv_sha256") != expected_argv_hash
        or worker.get("execution_boundary")
        != "docker_compose_exec_linux_worker"
    ):
        raise ValueError("G5 host attestation worker contract is invalid")
    inspect_hashes = dict(attestation.get("docker_inspect_sha256") or {})
    if set(inspect_hashes) != {BENCH_SERVICE, TARGET_SERVICE} or not all(
        _is_sha256(str(value or "")) for value in inspect_hashes.values()
    ):
        raise ValueError("G5 host attestation Docker inspect hashes are invalid")
    return {
        **attestation,
        "attestation_file": str(attestation_path),
        "attestation_file_sha256": observed_hash,
    }


def _validate_container_contract(
    container: Mapping[str, Any],
    *,
    label: str,
) -> None:
    container_id = str(container.get("container_id") or "")
    if (
        len(container_id) < 12
        or any(character not in "0123456789abcdef" for character in container_id.lower())
        or not str(container.get("image_digest") or "")
        or not str(container.get("image_reference") or "")
        or not str(container.get("container_name") or "")
    ):
        raise ValueError(f"G5 host attestation {label} container identity is invalid")
    networks = dict(container.get("networks") or {})
    if not networks:
        raise ValueError(f"G5 host attestation {label} container has no network")
    for network_name, raw in networks.items():
        network = dict(raw or {})
        if (
            not str(network_name)
            or not str(network.get("network_id") or "")
            or not str(network.get("endpoint_id") or "")
            or not isinstance(network.get("aliases"), list)
        ):
            raise ValueError(
                f"G5 host attestation {label} container network is invalid"
            )


def _content_hash(value: Any, *, newline: bool = False) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if newline:
        serialized += "\n"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )
