from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from tests.agent_live.real_execution_host_attestation import (
    validate_host_attestation_file,
)
from tests.agent_live.real_execution_host_supervisor import (
    ATTESTATION_ENV,
    BENCH_SERVICE,
    LINUX_WORKER_PREFIX,
    TARGET_SERVICE,
    build_docker_exec_command,
    build_worker_command,
    supervise_real_execution,
)


class RecordingRunner:
    def __init__(
        self,
        *,
        inspect_payload: dict | None = None,
        target_inspect_payload: dict | None = None,
        worker_returncode: int = 0,
    ) -> None:
        self.inspect_payload = inspect_payload or _inspect_payload()
        self.target_inspect_payload = target_inspect_payload or _inspect_payload(
            service=TARGET_SERVICE,
            container_id="b" * 64,
            image_digest="sha256:target-image-digest",
            image_reference="ethereum/client-go:test",
            container_name="blockchain-node-benchmark-geth-dev-1",
            aliases=(TARGET_SERVICE, "b" * 64),
        )
        self.worker_returncode = worker_returncode
        self.calls: list[tuple[tuple[str, ...], Path, float]] = []

    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        normalized = tuple(command)
        self.calls.append((normalized, cwd, timeout_seconds))
        if normalized[-3:] == ("ps", "-q", BENCH_SERVICE):
            return subprocess.CompletedProcess(normalized, 0, "container-123\n", "")
        if normalized[-3:] == ("ps", "-q", TARGET_SERVICE):
            return subprocess.CompletedProcess(
                normalized, 0, "target-container-123\n", ""
            )
        if normalized[:2] == ("docker", "inspect"):
            payload = (
                self.target_inspect_payload
                if normalized[-1] == "target-container-123"
                else self.inspect_payload
            )
            return subprocess.CompletedProcess(
                normalized,
                0,
                json.dumps([payload]),
                "",
            )
        if "exec" in normalized:
            return subprocess.CompletedProcess(
                normalized,
                self.worker_returncode,
                "worker stdout\n",
                "worker stderr\n" if self.worker_returncode else "",
            )
        raise AssertionError(f"unexpected command: {normalized}")


def _inspect_payload(
    *,
    project: str = "blockchain-node-benchmark",
    service: str = BENCH_SERVICE,
    running: bool = True,
    container_id: str = "a" * 64,
    image_digest: str = "sha256:image-digest",
    image_reference: str = "blockchain-node-benchmark:test-ubuntu",
    container_name: str = "blockchain-node-benchmark-bench-1",
    aliases: tuple[str, ...] = ("bench", "a" * 64),
) -> dict:
    return {
        "Id": container_id,
        "Name": f"/{container_name}",
        "Image": image_digest,
        "Config": {
            "Image": image_reference,
            "Labels": {
                "com.docker.compose.project": project,
                "com.docker.compose.service": service,
            },
        },
        "State": {"Running": running},
        "NetworkSettings": {
            "Networks": {
                "blockchain-node-benchmark_default": {
                    "NetworkID": "network-123",
                    "EndpointID": "endpoint-123",
                    "Gateway": "172.20.0.1",
                    "IPAddress": "172.20.0.2",
                    "GlobalIPv6Address": "",
                    "Aliases": list(aliases),
                },
            },
        },
    }


class RealExecutionHostSupervisorTest(unittest.TestCase):
    def test_builds_fixed_shell_free_worker_and_exec_commands(self) -> None:
        worker = build_worker_command(("--fake-plan", "/workspace/fake.json"))
        self.assertEqual(worker[:3], LINUX_WORKER_PREFIX)
        self.assertEqual(
            worker[3:],
            ("--fake-plan", "/workspace/fake.json"),
        )
        command = build_docker_exec_command(
            compose_project="project-a",
            attestation_container_path="/workspace/.agent/evidence/attestation.json",
            worker_command=worker,
        )
        self.assertEqual(
            command[:4],
            ("docker", "compose", "--project-name", "project-a"),
        )
        self.assertEqual(
            command[4:9],
            (
                "exec",
                "-T",
                "-e",
                f"{ATTESTATION_ENV}=/workspace/.agent/evidence/attestation.json",
                BENCH_SERVICE,
            ),
        )
        self.assertEqual(command[9:], worker)
        self.assertNotIn("bash", command)
        self.assertNotIn("sh", command)

    def test_rejects_non_worker_commands_and_non_workspace_attestations(self) -> None:
        with self.assertRaisesRegex(ValueError, "only the G5 Linux worker"):
            build_docker_exec_command(
                compose_project=None,
                attestation_container_path="/workspace/evidence.json",
                worker_command=("/workspace/bin/anychain", "benchmark"),
            )
        with self.assertRaisesRegex(ValueError, "inside /workspace"):
            build_docker_exec_command(
                compose_project=None,
                attestation_container_path="/tmp/evidence.json",
                worker_command=LINUX_WORKER_PREFIX,
            )

    def test_attests_compose_runtime_atomically_before_worker_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            runner = RecordingRunner()
            observed_time = datetime(
                2026, 7, 24, 1, 2, 3, 456789, tzinfo=timezone.utc
            )
            result = supervise_real_execution(
                repo_root=root,
                attestation_dir=root / ".agent/evidence/g5-host",
                worker_args=("--fake-plan", "/workspace/plans/fake.json"),
                compose_project="blockchain-node-benchmark",
                timeout_seconds=37.0,
                command_runner=runner,
                revision_resolver=lambda _: {
                    "commit": "abc123",
                    "worktree_hash": "f" * 64,
                },
                clock=lambda: observed_time,
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(len(runner.calls), 5)
            self.assertEqual(
                runner.calls[0][0],
                (
                    "docker",
                    "compose",
                    "--project-name",
                    "blockchain-node-benchmark",
                    "ps",
                    "-q",
                    BENCH_SERVICE,
                ),
            )
            self.assertEqual(
                runner.calls[1][0],
                ("docker", "inspect", "container-123"),
            )
            self.assertEqual(
                runner.calls[2][0][-3:],
                ("ps", "-q", TARGET_SERVICE),
            )
            self.assertEqual(
                runner.calls[3][0],
                ("docker", "inspect", "target-container-123"),
            )
            self.assertEqual(runner.calls[4][0], result.docker_exec_command)
            self.assertIn("exec", result.docker_exec_command)
            self.assertIn("-T", result.docker_exec_command)
            self.assertTrue(result.attestation_path.is_file())

            raw = result.attestation_path.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), result.attestation_sha256)
            self.assertEqual(
                result.attestation_path.name,
                f"real-execution-host-attestation-{result.attestation_sha256}.json",
            )
            payload = json.loads(raw)
            self.assertEqual(payload["repository_revision"]["commit"], "abc123")
            self.assertEqual(
                payload["compose"],
                {
                    "project": "blockchain-node-benchmark",
                    "worker_service": BENCH_SERVICE,
                    "target_service": TARGET_SERVICE,
                },
            )
            self.assertEqual(
                payload["worker_container"]["container_id"],
                "a" * 64,
            )
            self.assertEqual(
                payload["worker_container"]["image_digest"],
                "sha256:image-digest",
            )
            self.assertEqual(
                payload["worker_container"]["networks"][
                    "blockchain-node-benchmark_default"
                ]["network_id"],
                "network-123",
            )
            self.assertEqual(
                payload["target_container"]["container_id"],
                "b" * 64,
            )
            self.assertEqual(payload["worker"]["argv"], list(result.worker_command))
            self.assertEqual(
                payload["worker"]["argv_sha256"],
                hashlib.sha256(
                    (
                        json.dumps(
                            {"argv": list(result.worker_command)},
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(payload["attested_at"], "2026-07-24T01:02:03.456789Z")
            self.assertFalse(payload["assurance"]["cryptographic_identity"])
            self.assertIn(
                "No cryptographic",
                payload["assurance"]["claim"],
            )

            injected = next(
                item
                for item in result.docker_exec_command
                if item.startswith(f"{ATTESTATION_ENV}=")
            )
            self.assertTrue(injected.startswith(f"{ATTESTATION_ENV}=/workspace/"))
            validated = validate_host_attestation_file(
                result.attestation_path,
                repo_root=root,
                revision={
                    "commit": "abc123",
                    "worktree_hash": "f" * 64,
                },
                worker_argv=("--fake-plan", "/workspace/plans/fake.json"),
            )
            self.assertEqual(
                validated["attestation_file_sha256"],
                result.attestation_sha256,
            )

    def test_consumer_rejects_tampered_or_cross_revision_host_attestation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            revision = {
                "commit": "abc123",
                "worktree_hash": "f" * 64,
            }
            result = supervise_real_execution(
                repo_root=root,
                attestation_dir=root / "evidence",
                worker_args=("--help",),
                compose_project="blockchain-node-benchmark",
                command_runner=RecordingRunner(),
                revision_resolver=lambda _: revision,
                clock=lambda: datetime(2026, 7, 24, tzinfo=timezone.utc),
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                validate_host_attestation_file(
                    result.attestation_path,
                    repo_root=root,
                    revision={
                        "commit": "different",
                        "worktree_hash": "f" * 64,
                    },
                    worker_argv=("--help",),
                )
            raw = result.attestation_path.read_bytes()
            result.attestation_path.write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "content-addressed"):
                validate_host_attestation_file(
                    result.attestation_path,
                    repo_root=root,
                    revision=revision,
                    worker_argv=("--help",),
                )

    def test_content_addressed_write_is_idempotent_for_same_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            kwargs = {
                "repo_root": root,
                "attestation_dir": root / "evidence",
                "worker_args": ("--help",),
                "compose_project": "blockchain-node-benchmark",
                "revision_resolver": lambda _: {
                    "commit": "abc123",
                    "worktree_hash": "0" * 64,
                },
                "clock": lambda: datetime(2026, 7, 24, tzinfo=timezone.utc),
            }
            first = supervise_real_execution(
                **kwargs,
                command_runner=RecordingRunner(),
            )
            second = supervise_real_execution(
                **kwargs,
                command_runner=RecordingRunner(),
            )
            self.assertEqual(first.attestation_path, second.attestation_path)
            self.assertEqual(first.attestation_sha256, second.attestation_sha256)
            self.assertEqual(len(list((root / "evidence").glob("*.json"))), 1)

    def test_rejects_preexisting_symlink_at_content_address(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            kwargs = {
                "repo_root": root,
                "attestation_dir": root / "evidence",
                "worker_args": ("--help",),
                "compose_project": "blockchain-node-benchmark",
                "revision_resolver": lambda _: {
                    "commit": "abc123",
                    "worktree_hash": "0" * 64,
                },
                "clock": lambda: datetime(2026, 7, 24, tzinfo=timezone.utc),
            }
            first = supervise_real_execution(
                **kwargs,
                command_runner=RecordingRunner(),
            )
            saved = root / "saved-attestation.json"
            first.attestation_path.rename(saved)
            first.attestation_path.symlink_to(saved)
            with self.assertRaisesRegex(RuntimeError, "not a regular file"):
                supervise_real_execution(
                    **kwargs,
                    command_runner=RecordingRunner(),
                )

    def test_rejects_wrong_compose_identity_before_worker_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            runner = RecordingRunner(
                inspect_payload=_inspect_payload(project="different-project")
            )
            with self.assertRaisesRegex(RuntimeError, "project does not match"):
                supervise_real_execution(
                    repo_root=root,
                    attestation_dir=root / "evidence",
                    worker_args=(),
                    compose_project="expected-project",
                    command_runner=runner,
                    revision_resolver=lambda _: {
                        "commit": "abc",
                        "worktree_hash": "1" * 64,
                    },
                )
            self.assertEqual(len(runner.calls), 2)

    def test_rejects_attestation_directory_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as repo_tmp, tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(ValueError, "inside the repository"):
                supervise_real_execution(
                    repo_root=repo_tmp,
                    attestation_dir=out,
                    worker_args=(),
                    command_runner=RecordingRunner(),
                    revision_resolver=lambda _: {
                        "commit": "abc",
                        "worktree_hash": "2" * 64,
                    },
                )

    def test_worker_exit_is_propagated_without_host_side_business_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            runner = RecordingRunner(worker_returncode=23)
            result = supervise_real_execution(
                repo_root=root,
                attestation_dir=root / "evidence",
                worker_args=("--help",),
                command_runner=runner,
                revision_resolver=lambda _: {
                    "commit": "abc",
                    "worktree_hash": "3" * 64,
                },
            )
            self.assertEqual(result.returncode, 23)
            self.assertEqual(result.stdout, "worker stdout\n")
            self.assertEqual(result.stderr, "worker stderr\n")
            self.assertEqual(
                {call[0][0] for call in runner.calls},
                {"docker"},
            )


if __name__ == "__main__":
    unittest.main()
