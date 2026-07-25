from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.run_offline_python_suite import (
    _compile_process_tree_guard,
    _install_process_tree_guard,
    _remove_provider_credentials,
)


class OfflinePythonSuiteIsolationTest(unittest.TestCase):
    def test_provider_credentials_are_removed_by_contract(self) -> None:
        environment = {
            "DEEPSEEK_API_KEY": "secret",
            "VENDOR_ACCESS_TOKEN": "secret",
            "PATH": "/bin",
        }

        removed = _remove_provider_credentials(environment)

        self.assertEqual(
            removed,
            ("DEEPSEEK_API_KEY", "VENDOR_ACCESS_TOKEN"),
        )
        self.assertEqual(environment, {"PATH": "/bin"})

    def test_process_tree_guard_blocks_external_and_allows_loopback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            library = _compile_process_tree_guard(Path(tmpdir))
            environment = os.environ.copy()
            environment["LD_PRELOAD"] = str(library)
            external = subprocess.run(
                (
                    sys.executable,
                    "-c",
                    "import socket; socket.create_connection(('1.1.1.1', 53), .1)",
                ),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            loopback = subprocess.run(
                (
                    sys.executable,
                    "-c",
                    "import socket; s=socket.socket(); "
                    "print(s.connect_ex(('127.0.0.1', 9)))",
                ),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(external.returncode, 0)
        self.assertIn("Network is unreachable", external.stderr)
        self.assertEqual(loopback.returncode, 0, loopback.stderr)

    def test_installation_reports_the_actual_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            library = Path(tmpdir) / "guard.so"
            previous = os.environ.copy()
            try:
                os.environ["DEEPSEEK_API_KEY"] = "secret"
                removed = _install_process_tree_guard(library)
                self.assertIn("DEEPSEEK_API_KEY", removed)
                self.assertEqual(
                    os.environ["ANYCHAIN_OFFLINE_NETWORK_GUARD"],
                    "ld_preload_v1",
                )
                self.assertTrue(os.environ["LD_PRELOAD"].startswith(str(library)))
            finally:
                os.environ.clear()
                os.environ.update(previous)


if __name__ == "__main__":
    unittest.main()
