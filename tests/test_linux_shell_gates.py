from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.agent_live.linux_shell_gates import (
    execute_required_linux_shell_gates,
    load_linux_shell_gate_manifest,
)


class LinuxShellGateManifestTest(unittest.TestCase):
    def test_repository_manifest_classifies_every_tracked_shell_test(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = load_linux_shell_gate_manifest(root)

        self.assertEqual(manifest["denominator"], 51)
        self.assertEqual(
            {entry["classification"] for entry in manifest["entries"]},
            {"required", "diagnostic"},
        )

    def test_unclassified_or_missing_shell_test_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tests = root / "tests"
            tests.mkdir()
            (tests / "one.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "scope": "tracked_tests_recursive_shell",
                    "entries": [{
                        "path": "tests/missing.sh",
                        "classification": "required",
                    }],
                }),
                encoding="utf-8",
            )

            tracked = subprocess_result(0, b"tests/one.sh\0", b"")
            with (
                patch(
                    "tests.agent_live.linux_shell_gates.subprocess.run",
                    return_value=tracked,
                ),
                self.assertRaisesRegex(ValueError, "classification drift"),
            ):
                load_linux_shell_gate_manifest(root, manifest_path)

    def test_non_required_entry_requires_reason_and_valid_coverage_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "scope": "tracked_tests_recursive_shell",
                    "entries": [{
                        "path": "tests/helper.sh",
                        "classification": "equivalently_covered",
                        "covered_by": "tests/missing.sh",
                        "reason": "covered elsewhere",
                    }],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid coverage owner"):
                load_linux_shell_gate_manifest(root, manifest_path)

    def test_required_gate_failure_is_preserved(self) -> None:
        manifest = {
            "manifest": "manifest.json",
            "denominator": 2,
            "entries": [
                {"path": "tests/pass.sh", "classification": "required"},
                {"path": "tests/fail.sh", "classification": "required"},
            ],
        }
        completed = (
            subprocess_result(0, "ok", ""),
            subprocess_result(7, "", "failed"),
        )
        with patch(
            "tests.agent_live.linux_shell_gates.subprocess.run",
            side_effect=completed,
        ):
            result = execute_required_linux_shell_gates(Path("/repo"), manifest)

        self.assertEqual(result["required_denominator"], 2)
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["status"], "failed")


def subprocess_result(returncode: int, stdout: str, stderr: str):
    from subprocess import CompletedProcess

    return CompletedProcess(("bash",), returncode, stdout, stderr)


if __name__ == "__main__":
    unittest.main()
