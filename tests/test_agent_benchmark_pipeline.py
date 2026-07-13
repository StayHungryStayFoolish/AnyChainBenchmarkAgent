"""Regression tests for Phase 5: fixing the Harness -> ADK dependency

inversion (architecture audit Finding E). `agent.runners.benchmark_pipeline`
owns the prepare/smoke/submit business logic that both
`agent.harness.nodes.execution` and `agent.tools.executor` (the CLI
tool-call/tool-schema surface) call into directly, instead of either caller
reaching through a duplicate ADK wrapper layer (retired; see
`agent/README.md`'s "Retired files must not return").
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

    def test_executor_prepare_benchmark_run_is_the_shared_pipeline_function(self) -> None:
        """`agent.tools.executor` must call the pipeline function directly,

        not a duplicate wrapper — this is an identity check, not a
        behavior-equivalence check, so it fails loudly if a second
        implementation is ever reintroduced.
        """
        from agent.runners.benchmark_pipeline import prepare_benchmark_run as core_prepare_benchmark_run
        from agent.tools.executor import prepare_benchmark_run as executor_prepare_benchmark_run

        self.assertIs(executor_prepare_benchmark_run, core_prepare_benchmark_run)

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

    def test_executor_gates_run_fake_node_smoke_benchmark_and_install_dependencies_on_approved(self) -> None:
        from agent.tools.executor import execute_tool

        smoke = execute_tool("run_fake_node_smoke_benchmark", {"plan_file": "unused.json"})
        install = execute_tool("install_dependencies", {})
        self.assertEqual(smoke["status"], "needs_confirmation")
        self.assertEqual(install["status"], "needs_confirmation")
        self.assertTrue(smoke["requires_user_confirmation"])
        self.assertTrue(install["requires_user_confirmation"])

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


class BenchmarkSubprocessEnvTest(unittest.TestCase):
    """C.4 (known-issues.md validation gap): real benchmark execution never

    completed in this environment. Root cause once `pandas`/`matplotlib`/etc.
    and a Go toolchain were installed: the benchmark pipeline's own analysis/
    report scripts call bare `python3`, not the Agent's own venv -- and
    `scripts/install_deps.sh` documents a separate repo-root `.venv` as the
    framework's own Python environment, which the Agent process does not
    necessarily have on `PATH` when it spawns the benchmark subprocess. The
    subprocess then failed partway through (well after Vegeta itself had
    already run) with confusing per-script failure messages, not a clear
    "missing Python package" error. Verified live end to end: a real
    fake-node run with `.venv` on `PATH` completed fully (real Vegeta load,
    real analysis scripts, real bilingual HTML report, real archiving) where
    it previously failed at the report-generation stage.
    """

    def test_prepends_project_venv_bin_to_path_when_it_exists(self) -> None:
        import os
        import tempfile
        from unittest.mock import patch

        from agent.runners.materialize import benchmark_subprocess_env

        with tempfile.TemporaryDirectory() as tmp_repo:
            venv_bin = Path(tmp_repo) / ".venv" / "bin"
            venv_bin.mkdir(parents=True)
            with patch("agent.runners.materialize._REPO_ROOT", Path(tmp_repo)):
                env = benchmark_subprocess_env({"FOO": "bar"})
            self.assertEqual(env["FOO"], "bar")
            self.assertTrue(env["PATH"].startswith(str(venv_bin) + os.pathsep))
            # The rest of the original PATH must still be present, not replaced.
            self.assertIn(os.environ.get("PATH", ""), env["PATH"])

    def test_leaves_path_unchanged_when_no_project_venv_exists(self) -> None:
        import os
        import tempfile
        from unittest.mock import patch

        from agent.runners.materialize import benchmark_subprocess_env

        with tempfile.TemporaryDirectory() as tmp_repo:
            with patch("agent.runners.materialize._REPO_ROOT", Path(tmp_repo)):
                env = benchmark_subprocess_env({})
            self.assertEqual(env.get("PATH", ""), os.environ.get("PATH", ""))


if __name__ == "__main__":
    unittest.main()
