from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.runners.application_service import (
    BenchmarkExecutionService,
    ExecutionFailureCode,
    ExecutionFailure,
    ExecutionOperation,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)


class BenchmarkExecutionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = BenchmarkExecutionService()

    def test_operation_policy_registry_covers_every_execution_operation(self) -> None:
        from agent.runners.application_service import EXECUTION_OPERATION_SPECS

        self.assertEqual(set(EXECUTION_OPERATION_SPECS), set(ExecutionOperation))
        for operation, spec in EXECUTION_OPERATION_SPECS.items():
            self.assertIs(spec.operation, operation)
            self.assertTrue(spec.runner_kind)
        for operation in (
            ExecutionOperation.FAKE_NODE_SMOKE,
            ExecutionOperation.REAL_NODE_SMOKE,
            ExecutionOperation.FINAL_BENCHMARK,
            ExecutionOperation.SYNC_OBSERVE,
        ):
            self.assertTrue(EXECUTION_OPERATION_SPECS[operation].requires_approval)
            self.assertTrue(EXECUTION_OPERATION_SPECS[operation].requires_plan_file)

    def test_every_submission_without_approval_is_blocked_before_backend_work(self) -> None:
        with (
            patch("agent.runners.application_service.run_fake_node_smoke_benchmark") as fake_smoke,
            patch("agent.runners.application_service.run_real_node_smoke_benchmark") as real_smoke,
            patch("agent.runners.application_service.submit_benchmark_job") as submit,
            patch("agent.runners.application_service.submit_job") as mock_submit,
        ):
            for operation in (
                ExecutionOperation.FAKE_NODE_SMOKE,
                ExecutionOperation.REAL_NODE_SMOKE,
                ExecutionOperation.FINAL_BENCHMARK,
                ExecutionOperation.SYNC_OBSERVE,
            ):
                with self.subTest(operation=operation):
                    result = self.service.execute(
                        ExecutionRequest(operation=operation, plan_file="not-read.json")
                    )
                    self.assertEqual(result.status, ExecutionStatus.BLOCKED)
                    self.assertEqual(result.failure.code, ExecutionFailureCode.APPROVAL_REQUIRED)
                    self.assertTrue(result.requires_user_confirmation)

        fake_smoke.assert_not_called()
        real_smoke.assert_not_called()
        submit.assert_not_called()
        mock_submit.assert_not_called()

    def test_prepare_uses_typed_result_envelope(self) -> None:
        payload = {
            "status": "ok",
            "data": {"plan_file": "prepared.json"},
            "evidence_paths": ["prepared.json"],
            "warnings": [],
            "next_actions": ["preflight"],
        }
        with patch("agent.runners.application_service.prepare_benchmark_run", return_value=payload) as backend:
            result = self.service.execute(
                ExecutionRequest(
                    operation=ExecutionOperation.PREPARE,
                    prepare_kwargs={"chain": "solana"},
                )
            )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.data["plan_file"], "prepared.json")
        backend.assert_called_once_with(chain="solana")

    def test_preflight_blocker_is_a_typed_retryable_failure(self) -> None:
        preflight = {"passed": False, "blockers": ["endpoint unreachable"], "warnings": []}
        with patch("agent.runners.application_service.run_preflight", return_value=preflight):
            result = self.service.execute(
                ExecutionRequest(operation=ExecutionOperation.PREFLIGHT, plan={"plan_id": "p1"})
            )

        self.assertEqual(result.status, ExecutionStatus.BLOCKED)
        self.assertEqual(result.failure.code, ExecutionFailureCode.PREFLIGHT_BLOCKED)
        self.assertTrue(result.failure.retryable)
        self.assertEqual(result.data["preflight"], preflight)

    def test_repeated_approved_request_reuses_one_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_file = root / "plan.json"
            plan_file.write_text(
                json.dumps(
                    {
                        "plan_id": "typed-service-idempotency",
                        "chain": "solana",
                        "strategy": "smoke",
                        "execution": {
                            "command": ["true"],
                            "environment": {},
                            "runner_mode": "foreground",
                        },
                    }
                ),
                encoding="utf-8",
            )
            request = ExecutionRequest(
                operation=ExecutionOperation.FINAL_BENCHMARK,
                plan_file=plan_file,
                jobs_dir=root / "jobs",
                approved=True,
                mock=True,
            )

            first = self.service.execute(request)
            second = self.service.execute(request)

            self.assertTrue(first.succeeded)
            self.assertTrue(second.succeeded)
            self.assertFalse(first.reused)
            self.assertTrue(second.reused)
            self.assertEqual(first.data["job"]["job_id"], second.data["job"]["job_id"])
            self.assertTrue(first.idempotency_key.startswith("final_benchmark:"))
            self.assertEqual(len(list((root / "jobs").glob("job_*/job.json"))), 1)
            self.assertEqual(len(list((root / "jobs").glob("job_*/plan.json"))), 1)
            self.assertFalse((root / "jobs" / "execution_copies").exists())

    def test_final_request_for_sync_plan_is_typed_as_sync_observe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file = Path(tmpdir) / "sync.json"
            plan_file.write_text(
                json.dumps({"plan_id": "sync", "workflow_type": "sync_observe", "execution": {}}),
                encoding="utf-8",
            )
            payload = {
                "status": "ok",
                "data": {"job": {"job_id": "job_sync", "status": "running"}},
                "evidence_paths": [],
                "warnings": [],
                "next_actions": [],
            }
            with patch("agent.runners.application_service.submit_benchmark_job", return_value=payload):
                result = self.service.execute(
                    ExecutionRequest(
                        operation=ExecutionOperation.FINAL_BENCHMARK,
                        plan_file=plan_file,
                        approved=True,
                    )
                )

        self.assertEqual(result.operation, ExecutionOperation.SYNC_OBSERVE)
        self.assertTrue(result.idempotency_key.startswith("sync_observe:"))

    def test_backend_exception_is_returned_as_typed_failure(self) -> None:
        with patch("agent.runners.application_service.prepare_benchmark_run", side_effect=RuntimeError("boom")):
            result = self.service.execute(
                ExecutionRequest(operation=ExecutionOperation.PREPARE, prepare_kwargs={})
            )

        self.assertEqual(result.status, ExecutionStatus.FAILED)
        self.assertEqual(result.failure.code, ExecutionFailureCode.INTERNAL_ERROR)
        self.assertTrue(result.failure.retryable)

    def test_harness_routes_pre_job_typed_failure_into_recovery(self) -> None:
        from agent.harness.domains.execution_runtime import execute_approved_preflight_and_smoke
        from agent.harness.coordinator import _apply_handler_result
        from agent.harness.state import new_state

        state = new_state("typed-submit-failure")
        state.update(
            {
                "target_mode": "fake-node",
                "workflow_mode": "rpc_benchmark",
                "preflight": {"approved": True, "execution_request_id": "request-1"},
            }
        )
        prepared = ExecutionResult(
            operation=ExecutionOperation.PREPARE,
            status=ExecutionStatus.OK,
            data={
                "plan": {"plan_id": "p1", "execution": {}},
                "plan_file": "unused.json",
                "preflight": {"passed": True, "blockers": []},
            },
        )
        failed = ExecutionResult(
            operation=ExecutionOperation.FAKE_NODE_SMOKE,
            status=ExecutionStatus.FAILED,
            warnings=("runner unavailable",),
            failure=ExecutionFailure(
                code=ExecutionFailureCode.INTERNAL_ERROR,
                message="runner unavailable",
                retryable=True,
            ),
        )
        with patch(
            "agent.harness.domains.execution_runtime.execution_service.execute",
            side_effect=[prepared, failed],
        ):
            result = execute_approved_preflight_and_smoke(state)

        updated = _apply_handler_result(state, result, owner="execution")
        self.assertEqual(result.recovery_command.operation, "activate")
        self.assertEqual(updated["job"]["status"], "failed")
        self.assertEqual(updated["job"]["error"], "runner unavailable")
        self.assertEqual(updated["active_group"], "failure_recovery")
        self.assertEqual(updated["pending_question"]["id"], "failure_recovery_action")


class ExecutionEntryPointStructureTests(unittest.TestCase):
    def test_transport_and_harness_adapters_do_not_import_parallel_submit_paths(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        forbidden = {
            "agent/cli.py": ("import run_preflight", "import submit_job"),
            "agent/tools/executor.py": (
                "import prepare_benchmark_run",
                "import run_preflight",
                "import run_fake_node_smoke_benchmark",
                "import submit_job",
            ),
            "agent/harness/domains/execution_runtime.py": (
                "import prepare_benchmark_run",
                "import run_preflight",
                "import run_fake_node_smoke_benchmark",
                "import run_real_node_smoke_benchmark",
                "import submit_benchmark_job",
            ),
        }
        for relative, fragments in forbidden.items():
            source = (repo / relative).read_text(encoding="utf-8")
            for fragment in fragments:
                self.assertNotIn(fragment, source, f"{relative} bypasses the execution service")
            self.assertIn("application_service import", source)


if __name__ == "__main__":
    unittest.main()
