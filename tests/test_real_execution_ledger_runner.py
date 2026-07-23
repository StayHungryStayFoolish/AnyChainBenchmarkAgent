from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.agent_live.execute_real_execution_ledger import (
    _edge_by_action,
    _job_evidence_files,
)


class RealExecutionLedgerRunnerTest(unittest.TestCase):
    def test_direct_script_entrypoint_loads_repository_modules(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "tests/agent_live/execute_real_execution_ledger.py"),
                "--help",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--rpc-plan", result.stdout)
        self.assertIn("--sync-plan", result.stdout)

    def test_edge_lookup_requires_one_applicable_execution_edge(self) -> None:
        edge = {
            "action_type": "approve_preflight_smoke",
            "evidence": {"real_execution": {"required": True}},
        }
        self.assertIs(_edge_by_action({"edges": [edge]}, "approve_preflight_smoke"), edge)
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            _edge_by_action({"edges": []}, "approve_preflight_smoke")

    def test_job_evidence_files_are_hashed_and_job_owned(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            job_id = "job_contract"
            run_dir = Path(tmpdir) / job_id
            run_dir.mkdir()
            for name, content in (
                ("job.json", json.dumps({"job_id": job_id})),
                ("plan.json", json.dumps({"plan_id": "plan"})),
                ("runtime.env", "BLOCKCHAIN_NODE=ethereum\n"),
                ("artifact-index.json", json.dumps({"job_id": job_id})),
                ("benchmark.log", "completed\n"),
            ):
                (run_dir / name).write_text(content, encoding="utf-8")
            job_artifacts, log_artifacts = _job_evidence_files({
                "job_id": job_id,
                "run_dir": str(run_dir),
                "runtime_env_file": str(run_dir / "runtime.env"),
                "artifact_index": str(run_dir / "artifact-index.json"),
            })
            self.assertEqual(len(job_artifacts), 4)
            self.assertEqual(len(log_artifacts), 1)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in (*job_artifacts, *log_artifacts)))


if __name__ == "__main__":
    unittest.main()
