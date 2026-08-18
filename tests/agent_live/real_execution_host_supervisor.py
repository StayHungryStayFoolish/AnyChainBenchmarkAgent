#!/usr/bin/env python3
"""Host-side Docker boundary for the Linux G5 real-execution worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from agent.harness.runtime_identity import repository_revision


ATTESTATION_ENV = "ANYCHAIN_G5_HOST_ATTESTATION"
ATTESTATION_SCHEMA_VERSION = 1
BENCH_SERVICE = "bench"
TARGET_SERVICE = "geth-dev"
LINUX_WORKER_PREFIX = (
    "/workspace/.venv-adk/bin/python",
    "-m",
    "tests.agent_live.execute_real_execution_ledger",
)

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
RevisionResolver = Callable[[str | Path], Mapping[str, str]]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class HostSupervisorResult:
    """Result of one attested Linux-worker launch."""

    attestation_path: Path
    attestation_sha256: str
    worker_command: tuple[str, ...]
    docker_exec_command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_timestamp(clock: Clock) -> str:
    observed = clock()
    if observed.tzinfo is None:
        raise ValueError("host supervisor clock must return a timezone-aware value")
    return (
        observed.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )


def _compose_prefix(compose_project: str | None) -> tuple[str, ...]:
    prefix = ("docker", "compose")
    project = str(compose_project or "").strip()
    if project:
        return (*prefix, "--project-name", project)
    return prefix


def build_worker_command(worker_args: Sequence[str]) -> tuple[str, ...]:
    """Build the only worker command the host supervisor may launch."""

    normalized: list[str] = []
    for raw in worker_args:
        value = str(raw)
        if "\x00" in value:
            raise ValueError("worker arguments cannot contain NUL bytes")
        normalized.append(value)
    return (*LINUX_WORKER_PREFIX, *normalized)


def build_docker_exec_command(
    *,
    compose_project: str | None,
    attestation_container_path: str,
    worker_command: Sequence[str],
) -> tuple[str, ...]:
    """Build a shell-free Docker exec command for the fixed bench service."""

    attestation_path = str(attestation_container_path).strip()
    if not attestation_path.startswith("/workspace/"):
        raise ValueError("worker attestation path must be inside /workspace")
    command = tuple(str(item) for item in worker_command)
    if command[: len(LINUX_WORKER_PREFIX)] != LINUX_WORKER_PREFIX:
        raise ValueError("host supervisor may launch only the G5 Linux worker")
    return (
        *_compose_prefix(compose_project),
        "exec",
        "-T",
        "-e",
        f"{ATTESTATION_ENV}={attestation_path}",
        BENCH_SERVICE,
        *command,
    )


def _checked_command(
    command_runner: CommandRunner,
    command: Sequence[str],
    *,
    repo_root: Path,
    timeout_seconds: float,
    purpose: str,
) -> subprocess.CompletedProcess[str]:
    result = command_runner(
        tuple(command),
        cwd=repo_root,
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        stderr = str(result.stderr or "").strip()
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"{purpose} failed with exit {result.returncode}{detail}")
    return result


def _inspect_container(
    command_runner: CommandRunner,
    *,
    repo_root: Path,
    compose_project: str | None,
    service: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    compose_result = _checked_command(
        command_runner,
        (*_compose_prefix(compose_project), "ps", "-q", service),
        repo_root=repo_root,
        timeout_seconds=timeout_seconds,
        purpose=f"docker compose {service} resolution",
    )
    container_refs = [
        line.strip()
        for line in str(compose_result.stdout or "").splitlines()
        if line.strip()
    ]
    if len(container_refs) != 1:
        raise RuntimeError(f"docker compose must resolve exactly one {service} container")
    container_ref = container_refs[0]
    inspect_result = _checked_command(
        command_runner,
        ("docker", "inspect", container_ref),
        repo_root=repo_root,
        timeout_seconds=timeout_seconds,
        purpose=f"docker inspect {service}",
    )
    try:
        inspected = json.loads(str(inspect_result.stdout or ""))
    except json.JSONDecodeError as exc:
        raise RuntimeError("docker inspect returned invalid JSON") from exc
    if (
        not isinstance(inspected, list)
        or len(inspected) != 1
        or not isinstance(inspected[0], Mapping)
    ):
        raise RuntimeError("docker inspect must return exactly one container object")
    return dict(inspected[0]), str(inspect_result.stdout or "")


def _container_contract(
    inspected: Mapping[str, Any],
    *,
    expected_project: str | None,
    expected_service: str,
) -> dict[str, Any]:
    config = dict(inspected.get("Config") or {})
    labels = dict(config.get("Labels") or {})
    state = dict(inspected.get("State") or {})
    network_settings = dict(inspected.get("NetworkSettings") or {})
    networks = dict(network_settings.get("Networks") or {})
    service = str(labels.get("com.docker.compose.service") or "")
    project = str(labels.get("com.docker.compose.project") or "")
    if service != expected_service:
        raise RuntimeError(
            f"inspected container is not the compose {expected_service} service"
        )
    if expected_project and project != expected_project:
        raise RuntimeError("inspected container compose project does not match request")
    if not project:
        raise RuntimeError("inspected container has no compose project label")
    if state.get("Running") is not True:
        raise RuntimeError("inspected container is not running")

    container_id = str(inspected.get("Id") or "")
    image_digest = str(inspected.get("Image") or "")
    image_reference = str(config.get("Image") or "")
    if not container_id or not image_digest or not image_reference:
        raise RuntimeError("inspected container identity is incomplete")
    if not networks:
        raise RuntimeError("inspected container has no attached network")

    network_contract: dict[str, Any] = {}
    for name, raw in sorted(networks.items()):
        details = dict(raw or {})
        network_id = str(details.get("NetworkID") or "")
        endpoint_id = str(details.get("EndpointID") or "")
        if not name or not network_id or not endpoint_id:
            raise RuntimeError("inspected bench network identity is incomplete")
        network_contract[str(name)] = {
            "network_id": network_id,
            "endpoint_id": endpoint_id,
            "gateway": str(details.get("Gateway") or ""),
            "ip_address": str(details.get("IPAddress") or ""),
            "global_ipv6_address": str(details.get("GlobalIPv6Address") or ""),
            "aliases": sorted(
                str(alias)
                for alias in (details.get("Aliases") or ())
                if str(alias)
            ),
        }

    return {
        "compose_project": project,
        "compose_service": service,
        "container_id": container_id,
        "container_name": str(inspected.get("Name") or "").lstrip("/"),
        "image_digest": image_digest,
        "image_reference": image_reference,
        "networks": network_contract,
    }


def _write_content_addressed_attestation(
    attestation_dir: Path,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    attestation_dir.mkdir(parents=True, exist_ok=True)
    serialized = _canonical_bytes(payload)
    digest = _sha256_bytes(serialized)
    destination = (
        attestation_dir / f"real-execution-host-attestation-{digest}.json"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".real-execution-host-attestation-",
        suffix=".tmp",
        dir=attestation_dir,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file():
                raise RuntimeError(
                    "content-addressed attestation path is not a regular file"
                )
            if destination.read_bytes() != serialized:
                raise RuntimeError(
                    "content-addressed attestation path contains different bytes"
                )
        directory_fd = os.open(attestation_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    if _sha256_bytes(destination.read_bytes()) != digest:
        raise RuntimeError("persisted host attestation content hash mismatch")
    return destination, digest


def supervise_real_execution(
    *,
    repo_root: str | Path,
    attestation_dir: str | Path,
    worker_args: Sequence[str],
    compose_project: str | None = None,
    timeout_seconds: float = 3600.0,
    command_runner: CommandRunner = _run_command,
    revision_resolver: RevisionResolver = repository_revision,
    clock: Clock = lambda: datetime.now(timezone.utc),
) -> HostSupervisorResult:
    """Attest Docker runtime identity, then launch the fixed Linux worker."""

    root = Path(repo_root).resolve(strict=True)
    output_dir = Path(attestation_dir).resolve()
    try:
        output_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("host attestation directory must be inside the repository") from exc

    bench_inspected, bench_raw_inspect = _inspect_container(
        command_runner,
        repo_root=root,
        compose_project=compose_project,
        service=BENCH_SERVICE,
        timeout_seconds=timeout_seconds,
    )
    bench_container = _container_contract(
        bench_inspected,
        expected_project=str(compose_project or "").strip() or None,
        expected_service=BENCH_SERVICE,
    )
    target_inspected, target_raw_inspect = _inspect_container(
        command_runner,
        repo_root=root,
        compose_project=bench_container["compose_project"],
        service=TARGET_SERVICE,
        timeout_seconds=timeout_seconds,
    )
    target_container = _container_contract(
        target_inspected,
        expected_project=bench_container["compose_project"],
        expected_service=TARGET_SERVICE,
    )
    worker_command = build_worker_command(worker_args)
    worker_contract = {
        "argv": list(worker_command),
        "argv_sha256": _sha256_bytes(_canonical_bytes({"argv": list(worker_command)})),
        "execution_boundary": "docker_compose_exec_linux_worker",
    }
    attestation = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "artifact_type": "real_execution_host_attestation",
        "assurance": {
            "kind": "docker_runtime_observation",
            "cryptographic_identity": False,
            "claim": "No cryptographic actor or host identity is asserted.",
        },
        "repository_revision": dict(revision_resolver(root)),
        "compose": {
            "project": bench_container["compose_project"],
            "worker_service": bench_container["compose_service"],
            "target_service": target_container["compose_service"],
        },
        "worker_container": bench_container,
        "target_container": target_container,
        "docker_inspect_sha256": {
            BENCH_SERVICE: _sha256_bytes(bench_raw_inspect.encode("utf-8")),
            TARGET_SERVICE: _sha256_bytes(target_raw_inspect.encode("utf-8")),
        },
        "worker": worker_contract,
        "attested_at": _utc_timestamp(clock),
    }
    attestation_path, attestation_sha256 = _write_content_addressed_attestation(
        output_dir,
        attestation,
    )
    relative_attestation = attestation_path.relative_to(root)
    container_attestation_path = f"/workspace/{relative_attestation.as_posix()}"
    docker_exec_command = build_docker_exec_command(
        compose_project=bench_container["compose_project"],
        attestation_container_path=container_attestation_path,
        worker_command=worker_command,
    )
    worker_result = command_runner(
        docker_exec_command,
        cwd=root,
        timeout_seconds=timeout_seconds,
    )
    return HostSupervisorResult(
        attestation_path=attestation_path,
        attestation_sha256=attestation_sha256,
        worker_command=worker_command,
        docker_exec_command=docker_exec_command,
        returncode=int(worker_result.returncode),
        stdout=str(worker_result.stdout or ""),
        stderr=str(worker_result.stderr or ""),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attest the Docker bench runtime on the host and launch the fixed "
            "G5 Linux worker through docker compose exec -T."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--attestation-dir",
        type=Path,
        default=Path(".agent/evidence/g5-host-attestations"),
    )
    parser.add_argument("--compose-project", default="")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument(
        "worker_args",
        nargs=argparse.REMAINDER,
        help="arguments passed to the fixed Linux real-execution worker",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    worker_args = list(args.worker_args)
    if worker_args[:1] == ["--"]:
        worker_args = worker_args[1:]
    attestation_dir = args.attestation_dir
    if not attestation_dir.is_absolute():
        attestation_dir = args.repo_root / attestation_dir
    result = supervise_real_execution(
        repo_root=args.repo_root,
        attestation_dir=attestation_dir,
        worker_args=worker_args,
        compose_project=args.compose_project or None,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({
        "attestation_path": str(result.attestation_path),
        "attestation_sha256": result.attestation_sha256,
        "worker_returncode": result.returncode,
    }, sort_keys=True))
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.stdout:
        print(result.stdout, end="")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
