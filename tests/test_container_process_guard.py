"""Focused Linux contracts for container-side Chaos process cleanup."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.agent_live.container_process_guard import (
    EXECUTION_ID_ENV,
    ContainerProcessGuard,
    ProcessIdentity,
)


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux procfs")
class ContainerProcessGuardTest(unittest.TestCase):
    def _spawn(self, execution_id: str, code: str) -> subprocess.Popen[bytes]:
        env = os.environ.copy()
        env[EXECUTION_ID_ENV] = execution_id
        return subprocess.Popen(
            [sys.executable, "-c", code],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

    def _stop(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)

    def test_exact_environment_match_excludes_near_values(self) -> None:
        execution_id = f"exact-{os.getpid()}-{time.time_ns()}"
        exact = self._spawn(execution_id, "import time; time.sleep(60)")
        near = self._spawn(execution_id + "-other", "import time; time.sleep(60)")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                guard = ContainerProcessGuard(
                    execution_id,
                    receipt_dir=temp_dir,
                    term_grace_seconds=0.1,
                    kill_grace_seconds=0.1,
                    scan_interval_seconds=0.01,
                )
                guard.register_pid(exact.pid, role="agent_process_group_leader")
                artifact = guard.cleanup(reapers={exact.pid: exact.poll})

            self.assertTrue(artifact.cleaned)
            self.assertIsNotNone(exact.poll())
            self.assertIsNone(near.poll())
            registered_pids = {
                row["pid"] for row in artifact.payload["registered_processes"]
            }
            self.assertIn(exact.pid, registered_pids)
            self.assertNotIn(near.pid, registered_pids)
        finally:
            self._stop(exact)
            self._stop(near)

    def test_term_then_kill_reaps_escaped_stubborn_match_and_hashes_receipt(self) -> None:
        execution_id = f"kill-{os.getpid()}-{time.time_ns()}"
        direct = self._spawn(execution_id, "import time; time.sleep(60)")
        stubborn = self._spawn(
            execution_id,
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)",
        )
        self.assertEqual(stubborn.stdout.readline().strip(), b"ready")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                guard = ContainerProcessGuard(
                    execution_id,
                    receipt_dir=temp_dir,
                    term_grace_seconds=0.1,
                    kill_grace_seconds=0.2,
                    scan_interval_seconds=0.01,
                )
                agent_identity = guard.register_pid(
                    direct.pid, role="agent_process_group_leader"
                )
                artifact = guard.cleanup(
                    reapers={direct.pid: direct.poll, stubborn.pid: stubborn.poll}
                )

                unsigned = {
                    key: value
                    for key, value in artifact.payload.items()
                    if key != "receipt_id"
                }
                self.assertEqual(
                    artifact.receipt_id,
                    hashlib.sha256(_canonical(unsigned)).hexdigest(),
                )
                self.assertEqual(
                    artifact.path.name,
                    f"container-cleanup-receipt-{artifact.receipt_id}.json",
                )
                self.assertEqual(
                    json.loads(artifact.path.read_text(encoding="utf-8")),
                    artifact.payload,
                )

            self.assertTrue(artifact.cleaned)
            self.assertEqual(len(artifact.payload["zero_survivor_scans"]), 2)
            self.assertTrue(all(
                not scan["survivors"]
                for scan in artifact.payload["zero_survivor_scans"]
            ))
            self.assertEqual(agent_identity.pid, direct.pid)
            killed_pids = {
                row["identity"]["pid"]
                for row in artifact.payload["kill_decisions"]
                if row["sent"]
            }
            self.assertIn(stubborn.pid, killed_pids)
            self.assertIsNotNone(direct.poll())
            self.assertIsNotNone(stubborn.poll())
        finally:
            self._stop(direct)
            self._stop(stubborn)

    def test_signal_is_refused_after_registered_identity_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            guard = ContainerProcessGuard("identity-change", receipt_dir=temp_dir)
            registered = ProcessIdentity(123, 123, 10, 99)
            changed = ProcessIdentity(123, 123, 11, 99)
            sent: list[tuple[int, int]] = []
            guard._send_signal = lambda pid, signum: sent.append((pid, signum))
            with mock.patch.object(guard, "_read_identity", return_value=changed):
                decision = guard._signal_decision(registered, signal.SIGTERM)

        self.assertEqual(decision["outcome"], "identity_changed")
        self.assertFalse(decision["sent"])
        self.assertEqual(sent, [])

    def test_proc_scan_uncertainty_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proc_root = Path(temp_dir) / "proc"
            receipt_dir = Path(temp_dir) / "receipts"
            self._write_fake_process(proc_root, os.getpid(), start_ticks=1, environ=b"")
            malformed = proc_root / "4242"
            malformed.mkdir(parents=True)
            (malformed / "stat").write_text("malformed\n", encoding="utf-8")
            (malformed / "environ").write_bytes(
                b"ANYCHAIN_CHAOS_EXECUTION_ID=uncertain\0"
            )
            (malformed / "ns").mkdir()
            (malformed / "ns" / "pid").write_text("", encoding="utf-8")

            guard = ContainerProcessGuard(
                "uncertain",
                receipt_dir=receipt_dir,
                proc_root=proc_root,
                term_grace_seconds=0,
                kill_grace_seconds=0,
                scan_interval_seconds=0.001,
                container_id="test-container",
            )
            artifact = guard.cleanup()

        self.assertFalse(artifact.cleaned)
        self.assertTrue(artifact.payload["errors"])
        self.assertEqual(len(artifact.payload["zero_survivor_scans"]), 2)

    def test_bridge_identity_with_exact_execution_id_is_never_a_survivor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proc_root = Path(temp_dir) / "proc"
            receipt_dir = Path(temp_dir) / "receipts"
            self._write_fake_process(
                proc_root,
                os.getpid(),
                start_ticks=1,
                environ=b"ANYCHAIN_CHAOS_EXECUTION_ID=self-match\0",
            )
            signals: list[tuple[int, int]] = []
            guard = ContainerProcessGuard(
                "self-match",
                receipt_dir=receipt_dir,
                proc_root=proc_root,
                term_grace_seconds=0,
                kill_grace_seconds=0,
                scan_interval_seconds=0.001,
                send_signal=lambda pid, signum: signals.append((pid, signum)),
                container_id="test-container",
            )

            artifact = guard.cleanup()

        self.assertTrue(artifact.cleaned)
        self.assertEqual(signals, [])
        self.assertTrue(all(
            not scan["survivors"] for scan in artifact.payload["survivor_scans"]
        ))

    def test_execution_id_scan_never_claims_a_bridge_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proc_root = Path(temp_dir) / "proc"
            self._write_fake_process(proc_root, os.getpid(), start_ticks=1, environ=b"")
            self._write_fake_process(
                proc_root,
                4242,
                start_ticks=2,
                environ=b"ANYCHAIN_CHAOS_EXECUTION_ID=shared-supervisor\0",
            )
            guard = ContainerProcessGuard(
                "shared-supervisor",
                receipt_dir=Path(temp_dir) / "receipts",
                proc_root=proc_root,
                container_id="test-container",
            )
            guard._ancestor_pids = frozenset({4242})

            matches, errors = guard._matching_execution_identities()

        self.assertEqual(matches, ())
        self.assertEqual(errors, ())

    @staticmethod
    def _write_fake_process(
        proc_root: Path,
        pid: int,
        *,
        start_ticks: int,
        environ: bytes,
    ) -> None:
        process_root = proc_root / str(pid)
        (process_root / "ns").mkdir(parents=True)
        fields = ["S", "0", str(pid), *("0" for _ in range(16)), str(start_ticks)]
        (process_root / "stat").write_text(
            f"{pid} (test process) {' '.join(fields)}\n",
            encoding="utf-8",
        )
        (process_root / "environ").write_bytes(environ)
        (process_root / "ns" / "pid").write_text("", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
