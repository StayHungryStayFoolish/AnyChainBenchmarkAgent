"""Regression tests for Phase 5: fixing the Harness -> ADK dependency

inversion (architecture audit Finding E). `agent.runners.benchmark_pipeline`
now owns the prepare/smoke/submit business logic that both
`agent.harness.nodes.execution` and `agent.adk_app.tools.{actions,planning}`
call into, instead of the Harness reaching backwards into the ADK bridge.
These functions previously had zero direct test coverage.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class BenchmarkPipelineTest(unittest.TestCase):
    def test_prepare_benchmark_run_returns_tool_result_envelope(self) -> None:
        from agent.runners.benchmark_pipeline import prepare_benchmark_run

        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_benchmark_run(
                chain="bsc",
                goal="smoke",
                use_fake_node=True,
                output_dir=tmpdir,
            )
            for key in ("status", "data", "evidence_paths", "warnings", "next_actions"):
                self.assertIn(key, result)
            self.assertIn("plan", result["data"])
            self.assertTrue(Path(result["data"]["plan_file"]).is_file())

    def test_adk_prepare_benchmark_run_is_thin_wrapper_over_core(self) -> None:
        from agent.adk_app.tools.planning import prepare_benchmark_run as adk_prepare_benchmark_run
        from agent.runners.benchmark_pipeline import prepare_benchmark_run as core_prepare_benchmark_run

        with tempfile.TemporaryDirectory() as tmpdir_a, tempfile.TemporaryDirectory() as tmpdir_b:
            adk_result = adk_prepare_benchmark_run(
                chain="bsc", goal="smoke", use_fake_node=True, output_dir=tmpdir_a
            )
            core_result = core_prepare_benchmark_run(
                chain="bsc", goal="smoke", use_fake_node=True, output_dir=tmpdir_b
            )
            self.assertEqual(adk_result["status"], core_result["status"])
            self.assertEqual(
                adk_result["data"]["inferred_values"],
                core_result["data"]["inferred_values"],
            )

    def test_run_fake_node_smoke_benchmark_blocks_when_plan_file_missing(self) -> None:
        from agent.runners.benchmark_pipeline import run_fake_node_smoke_benchmark

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_fake_node_smoke_benchmark(
                str(Path(tmpdir) / "does_not_exist.json"), jobs_dir=tmpdir
            )
            self.assertEqual(result["status"], "blocked")

    def test_submit_benchmark_job_blocks_when_plan_file_missing(self) -> None:
        from agent.runners.benchmark_pipeline import submit_benchmark_job

        with tempfile.TemporaryDirectory() as tmpdir:
            result = submit_benchmark_job(str(Path(tmpdir) / "does_not_exist.json"), jobs_dir=tmpdir)
            self.assertEqual(result["status"], "blocked")

    def test_adk_action_wrappers_still_gate_on_approved(self) -> None:
        from agent.adk_app.tools.actions import run_fake_node_smoke_benchmark, submit_benchmark_job

        smoke = run_fake_node_smoke_benchmark("unused.json", approved=False)
        submit = submit_benchmark_job("unused.json", approved=False)
        self.assertEqual(smoke["status"], "needs_confirmation")
        self.assertEqual(submit["status"], "needs_confirmation")
        self.assertTrue(smoke["requires_user_confirmation"])
        self.assertTrue(submit["requires_user_confirmation"])

    def test_smoke_message_renders_actual_terminal_commands_not_dict_keys(self) -> None:
        """Regression test for a code-review finding: `_smoke_message` used to

        iterate the `terminal_commands` dict directly (yielding its keys, e.g.
        the literal word "status") instead of its values (the actual runnable
        command text), because `_job_terminal_commands` always returns a dict,
        never the list `_smoke_message` defaulted to.
        """

        from agent.harness.nodes.execution import _smoke_message

        smoke = {
            "status": "ok",
            "data": {
                "job": {"job_id": "job-123"},
                "terminal_commands": {
                    "status": "status job-123",
                    "logs": "logs job-123",
                    "follow": "follow job-123",
                    "analyze": "analyze latest job",
                },
            },
        }
        message = _smoke_message(smoke)
        self.assertIn("status job-123", message)
        self.assertIn("logs job-123", message)
        self.assertNotIn("Use: status; logs; follow; analyze", message)

    def test_read_only_job_terminal_commands_is_the_shared_pipeline_helper(self) -> None:
        """Regression test: `read_only.py` used to keep its own byte-for-byte

        copy of `_job_terminal_commands`/`_job_user_next_actions` instead of
        importing the canonical versions from `benchmark_pipeline`, defeating
        the single-source-of-truth goal of this phase.
        """

        from agent.adk_app.tools import read_only
        from agent.runners import benchmark_pipeline

        self.assertIs(read_only._job_terminal_commands, benchmark_pipeline._job_terminal_commands)
        self.assertIs(read_only._job_user_next_actions, benchmark_pipeline._job_user_next_actions)


if __name__ == "__main__":
    unittest.main()
