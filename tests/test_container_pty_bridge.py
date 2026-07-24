"""Lifecycle ownership tests for the container-local PTY bridge."""

from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.agent_live import container_pty_bridge
from tests.agent_live.container_process_guard import CleanupReceiptArtifact
from tests.agent_live.dynamic_dual_ai_chaos import ContainerPtyBridgeTransport


class _FakeProcess:
    pid = 321

    def poll(self) -> int:
        return 0


class _FakeTransport:
    instances: list["_FakeTransport"] = []
    start_error: BaseException | None = None

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._process = _FakeProcess()
        self.started = 0
        self.closed = 0
        self.__class__.instances.append(self)

    def start(self, *, env: object) -> None:
        self.started += 1
        if self.start_error is not None:
            raise self.start_error

    def close(self) -> None:
        self.closed += 1

    def read_complete_agent_response(self, *, timeout_seconds: float) -> str:
        return "Agent response"

    def submit_bracketed_paste(self, message: str) -> None:
        return None

    def send_interrupt(self) -> None:
        return None


class _FakeGuard:
    instance: "_FakeGuard | None" = None
    cleaned = True

    def __init__(self, artifact: CleanupReceiptArtifact) -> None:
        self.artifact = artifact
        self.registered: list[tuple[int, str]] = []
        self.cleanup_calls = 0

    @classmethod
    def from_environment(cls) -> "_FakeGuard":
        assert cls.instance is not None
        return cls.instance

    def register_pid(self, pid: int, *, role: str) -> None:
        self.registered.append((pid, role))

    def cleanup(self, *, reapers: object) -> CleanupReceiptArtifact:
        self.cleanup_calls += 1
        return self.artifact


class _SignalInput:
    def __init__(self, handlers: dict[int, object]) -> None:
        self.handlers = handlers

    def __iter__(self) -> "_SignalInput":
        return self

    def __next__(self) -> str:
        handler = self.handlers[signal.SIGTERM]
        handler(signal.SIGTERM, None)
        raise StopIteration


class ContainerPtyBridgeLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        receipt_path = Path(self.temp_dir.name) / "receipt.json"
        receipt_path.write_text("{}\n", encoding="utf-8")
        payload = {"receipt_id": "receipt-id", "cleaned": True}
        _FakeGuard.instance = _FakeGuard(
            CleanupReceiptArtifact(path=receipt_path, payload=payload)
        )
        _FakeTransport.instances = []
        _FakeTransport.start_error = None

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run(self, stdin: object) -> tuple[int, str, _FakeTransport, _FakeGuard]:
        output = io.StringIO()
        with (
            mock.patch.object(container_pty_bridge, "ContainerProcessGuard", _FakeGuard),
            mock.patch.object(container_pty_bridge, "SubprocessPtyTransport", _FakeTransport),
            mock.patch.object(container_pty_bridge.sys, "stdin", stdin),
            mock.patch.object(container_pty_bridge.sys, "stdout", output),
            mock.patch.object(container_pty_bridge.signal, "signal"),
        ):
            result = container_pty_bridge.main(["--cwd", "/tmp", "--", "product"])
        transport = _FakeTransport.instances[-1]
        guard = _FakeGuard.instance
        assert guard is not None
        return result, output.getvalue(), transport, guard

    def test_close_waits_for_the_finally_owned_cleanup_receipt(self) -> None:
        result, output, transport, guard = self._run(io.StringIO('{"op":"close"}\n'))

        response = json.loads(output)
        self.assertEqual(result, 0)
        self.assertTrue(response["ok"])
        self.assertEqual(response["cleanup_receipt"]["receipt_id"], "receipt-id")
        self.assertEqual(
            response["cleanup_receipt"]["sha256"],
            hashlib.sha256(b"{}\n").hexdigest(),
        )
        self.assertEqual(guard.cleanup_calls, 1)
        self.assertEqual(transport.closed, 1)
        self.assertEqual(guard.registered, [(321, "agent_process_group_leader")])

    def test_eof_and_operation_exception_use_the_same_finally_owner(self) -> None:
        eof_result, eof_output, eof_transport, eof_guard = self._run(io.StringIO(""))
        self.assertEqual(eof_result, 0)
        self.assertEqual(eof_output, "")
        self.assertEqual(eof_guard.cleanup_calls, 1)
        self.assertEqual(eof_transport.closed, 1)

        result, output, transport, guard = self._run(io.StringIO('{"op":"unknown"}\n'))
        self.assertEqual(result, 1)
        error_response = json.loads(output)
        self.assertFalse(error_response["ok"])
        self.assertEqual(
            error_response["cleanup_receipt"]["receipt_id"], "receipt-id"
        )
        self.assertEqual(guard.cleanup_calls, 2)
        self.assertEqual(transport.closed, 1)

    def test_signal_and_start_exception_use_the_same_finally_owner(self) -> None:
        handlers: dict[int, object] = {}

        def install(signum: int, handler: object) -> None:
            handlers[signum] = handler

        output = io.StringIO()
        with (
            mock.patch.object(container_pty_bridge, "ContainerProcessGuard", _FakeGuard),
            mock.patch.object(container_pty_bridge, "SubprocessPtyTransport", _FakeTransport),
            mock.patch.object(container_pty_bridge.sys, "stdin", _SignalInput(handlers)),
            mock.patch.object(container_pty_bridge.sys, "stdout", output),
            mock.patch.object(container_pty_bridge.signal, "signal", side_effect=install),
        ):
            with self.assertRaisesRegex(SystemExit, "143"):
                container_pty_bridge.main(["--cwd", "/tmp", "--", "product"])
        signal_transport = _FakeTransport.instances[-1]
        signal_guard = _FakeGuard.instance
        assert signal_guard is not None
        self.assertEqual(signal_guard.cleanup_calls, 1)
        self.assertEqual(signal_transport.closed, 1)

        _FakeTransport.instances = []
        _FakeTransport.start_error = RuntimeError("start failed")
        with (
            mock.patch.object(container_pty_bridge, "ContainerProcessGuard", _FakeGuard),
            mock.patch.object(container_pty_bridge, "SubprocessPtyTransport", _FakeTransport),
            mock.patch.object(container_pty_bridge.signal, "signal"),
        ):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                container_pty_bridge.main(["--cwd", "/tmp", "--", "product"])
        start_transport = _FakeTransport.instances[-1]
        self.assertEqual(signal_guard.cleanup_calls, 2)
        self.assertEqual(start_transport.closed, 1)

    def test_incomplete_cleanup_fails_closed(self) -> None:
        assert _FakeGuard.instance is not None
        payload = {"receipt_id": "failed-receipt", "cleaned": False}
        _FakeGuard.instance.artifact = CleanupReceiptArtifact(
            path=_FakeGuard.instance.artifact.path,
            payload=payload,
        )

        result, output, _transport, guard = self._run(io.StringIO('{"op":"close"}\n'))

        response = json.loads(output)
        self.assertEqual(result, 1)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error_type"], "ContainerCleanupError")
        self.assertFalse(response["cleanup_receipt"]["cleaned"])
        self.assertEqual(guard.cleanup_calls, 1)


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux procfs")
class ContainerPtyBridgeExecutionProofIntegrationTest(unittest.TestCase):
    def test_real_bridge_produces_a_validated_process_cleanup_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_dir = Path(temporary) / "receipts"
            execution_id = "container-pty-integration"
            repl = (
                "import sys\n"
                "print('Agent> ready\\nUser> ', end='', flush=True)\n"
                "for _line in sys.stdin:\n"
                "    print('Agent> received\\nUser> ', end='', flush=True)\n"
            )
            transport = ContainerPtyBridgeTransport(
                (
                    sys.executable,
                    "-m",
                    "tests.agent_live.container_pty_bridge",
                    "--cwd",
                    str(Path.cwd()),
                    "--",
                    sys.executable,
                    "-u",
                    "-c",
                    repl,
                ),
                cwd=Path.cwd(),
                execution_id=execution_id,
                cleanup_receipt_dir=receipt_dir,
            )

            transport.start(env=os.environ)
            self.assertEqual(
                transport.read_complete_agent_response(timeout_seconds=5),
                "Agent> ready",
            )
            transport.submit_bracketed_paste("hello")
            self.assertEqual(
                transport.read_complete_agent_response(timeout_seconds=5),
                "Agent> received",
            )
            transport.close()

            proof = transport.validated_execution_proof()
            self.assertEqual(proof["execution_id"], execution_id)
            self.assertEqual(proof["proof_type"], "container_pty_process_guard")
            self.assertEqual(proof["transport_kind"], "container_pty_bridge")
            self.assertTrue(proof["cleaned"])
            roles = {
                role
                for row in proof["registered_processes"]
                for role in row["roles"]
            }
            self.assertTrue(
                {"container_bridge", "agent_process_group_leader"}.issubset(roles)
            )
            self.assertEqual(len(proof["zero_survivor_scans"]), 2)


if __name__ == "__main__":
    unittest.main()
