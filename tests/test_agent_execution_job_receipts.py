"""Owner-issued evidence for execution submission and job status reads."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class JobManagerReceiptTest(unittest.TestCase):
    @staticmethod
    def _write_plan(
        root: Path,
        *,
        workflow_type: str = "rpc_benchmark",
        operation: str = "final_benchmark",
        execution_key: str = "execution:test",
    ) -> Path:
        plan_file = root / "plan.json"
        command = ["./blockchain_node_benchmark.sh", "--sync-observe", "--duration", "15"]
        if workflow_type == "rpc_benchmark":
            command = ["./blockchain_node_benchmark.sh", "--quick", "--single"]
        plan_file.write_text(
            json.dumps(
                {
                    "plan_id": "receipt-plan",
                    "workflow_type": workflow_type,
                    "execution": {
                        "idempotency_key": execution_key,
                        "runner_mode": "foreground",
                        "command": command,
                        "environment": {
                            "LOCAL_RPC_URL": "https://user:secret-token@example.invalid"
                        },
                    },
                    "execution_provenance": {"operation": operation},
                }
            ),
            encoding="utf-8",
        )
        return plan_file

    def test_idempotent_submission_receipts_prove_one_physical_job(self) -> None:
        from agent.runners.job_manager import submit_job, verify_job_receipt

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_file = self._write_plan(root)
            jobs_dir = root / "jobs"
            first = submit_job(plan_file, jobs_dir=jobs_dir, mock=True, approved=True)
            second = submit_job(plan_file, jobs_dir=jobs_dir, mock=True, approved=True)
            job_files = list(jobs_dir.glob("job_*/job.json"))

        created = first["execution_receipts"]["submission"]
        reused = second["execution_receipts"]["submission_attempt"]
        self.assertTrue(verify_job_receipt(created))
        self.assertTrue(verify_job_receipt(reused))
        self.assertEqual(created["disposition"], "created")
        self.assertEqual(reused["disposition"], "reused")
        self.assertEqual(created["job_id"], reused["job_id"])
        self.assertEqual(reused["matching_job_count"], 1)
        self.assertEqual(len(job_files), 1)
        self.assertNotIn("execution:test", json.dumps(created, sort_keys=True))
        self.assertNotIn("secret-token", json.dumps(created, sort_keys=True))

        extra_field = dict(created)
        extra_field["untrusted"] = True
        self.assertFalse(verify_job_receipt(extra_field))

        boolean_count = dict(created)
        boolean_count["matching_job_count"] = True
        boolean_count["receipt_id"] = "0" * 64
        self.assertFalse(verify_job_receipt(boolean_count))

        contradictory = dict(created)
        contradictory["workflow_type"] = "sync_observe"
        contradictory["receipt_id"] = "0" * 64
        self.assertFalse(verify_job_receipt(contradictory))

    def test_job_read_receipt_binds_status_to_job_file(self) -> None:
        from agent.runners.job_manager import get_job, submit_job, verify_job_receipt

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_file = self._write_plan(root)
            jobs_dir = root / "jobs"
            submitted = submit_job(plan_file, jobs_dir=jobs_dir, mock=True, approved=True)
            observed = get_job(submitted["job_id"], jobs_dir=jobs_dir)

        receipt = observed["execution_receipts"]["last_read"]
        self.assertTrue(verify_job_receipt(receipt))
        self.assertEqual(receipt["job_id"], submitted["job_id"])
        self.assertEqual(receipt["persisted_status"], "completed")
        self.assertEqual(receipt["observed_status"], "completed")
        self.assertEqual(receipt["status_source"], "persisted_job")
        self.assertEqual(receipt["artifact_keys"], sorted(submitted["artifacts"]))

        tampered = dict(receipt)
        tampered["observed_status"] = "failed"
        self.assertFalse(verify_job_receipt(tampered))

    def test_sync_observe_receipts_prove_no_vegeta_contract_or_artifact(self) -> None:
        from agent.runners.job_manager import get_job, submit_job, verify_job_receipt

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_file = self._write_plan(
                root,
                workflow_type="sync_observe",
                operation="sync_observe",
                execution_key="execution:sync",
            )
            jobs_dir = root / "jobs"
            submitted = submit_job(plan_file, jobs_dir=jobs_dir, mock=True, approved=True)
            observed = get_job(submitted["job_id"], jobs_dir=jobs_dir)

        submission = submitted["execution_receipts"]["submission"]
        read = observed["execution_receipts"]["last_read"]
        self.assertTrue(verify_job_receipt(submission))
        self.assertTrue(verify_job_receipt(read))
        self.assertEqual(submission["workflow_type"], "sync_observe")
        self.assertEqual(submission["load_generator"], "none")
        self.assertFalse(submission["vegeta_allowed"])
        self.assertFalse(read["vegeta_artifact_present"])
        self.assertNotIn("vegeta_json", read["artifact_keys"])

    def test_sync_observe_read_receipt_reports_unexpected_vegeta_artifact(self) -> None:
        from agent.runners.job_manager import get_job, submit_job, verify_job_receipt

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_file = self._write_plan(
                root,
                workflow_type="sync_observe",
                operation="sync_observe",
                execution_key="execution:sync-artifact",
            )
            jobs_dir = root / "jobs"
            submitted = submit_job(plan_file, jobs_dir=jobs_dir, mock=True, approved=True)
            job_file = jobs_dir / submitted["job_id"] / "job.json"
            persisted = json.loads(job_file.read_text(encoding="utf-8"))
            persisted["artifacts"]["vegeta_json"] = "/redacted/vegeta.json"
            job_file.write_text(json.dumps(persisted), encoding="utf-8")
            observed = get_job(submitted["job_id"], jobs_dir=jobs_dir)

        receipt = observed["execution_receipts"]["last_read"]
        self.assertTrue(verify_job_receipt(receipt))
        self.assertTrue(receipt["vegeta_artifact_present"])
        self.assertIn("vegeta_json", receipt["artifact_keys"])

    def test_all_job_read_surfaces_carry_the_manager_receipt(self) -> None:
        from agent.runners.job_manager import (
            list_jobs,
            resume_job,
            submit_job,
            tail_job_log,
            verify_job_receipt,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_file = self._write_plan(root)
            jobs_dir = root / "jobs"
            submitted = submit_job(plan_file, jobs_dir=jobs_dir, mock=True, approved=True)
            log_file = Path(submitted["run_dir"]) / "benchmark.log"
            log_file.write_text("completed\n", encoding="utf-8")
            listed = list_jobs(jobs_dir=jobs_dir, limit=1)[0]
            resumed = resume_job(submitted["job_id"], jobs_dir=jobs_dir)
            tailed = tail_job_log(submitted["job_id"], jobs_dir=jobs_dir)

        receipts = (
            listed["execution_receipts"]["last_read"],
            resumed["job_read_receipt"],
            tailed["job_read_receipt"],
        )
        self.assertTrue(all(verify_job_receipt(dict(receipt)) for receipt in receipts))
        self.assertEqual({receipt["job_id"] for receipt in receipts}, {submitted["job_id"]})


class ExecutionReceiptPropagationTest(unittest.TestCase):
    def test_runtime_summary_projects_only_valid_manager_receipts(self) -> None:
        from agent.harness.graph import _execution_receipt_summary
        from agent.runners.job_manager import submit_job

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_file = JobManagerReceiptTest._write_plan(root)
            submitted = submit_job(
                plan_file,
                jobs_dir=root / "jobs",
                mock=True,
                approved=True,
            )

        summary = _execution_receipt_summary({"job": submitted})
        self.assertEqual(
            summary["manager_submission_receipt_id"],
            submitted["execution_receipts"]["submission"]["receipt_id"],
        )
        self.assertEqual(summary["manager_submission_disposition"], "created")

        tampered = json.loads(json.dumps(submitted))
        tampered["execution_receipts"]["submission"]["matching_job_count"] = 99
        summary = _execution_receipt_summary({"job": tampered})
        self.assertEqual(summary["manager_submission_receipt_id"], "")
        self.assertEqual(summary["manager_matching_job_count"], 0)

        other_job = json.loads(json.dumps(submitted))
        other_job["job_id"] = "job_other"
        summary = _execution_receipt_summary({"job": other_job})
        self.assertEqual(summary["manager_submission_receipt_id"], "")

    def test_application_service_returns_manager_submission_receipt(self) -> None:
        from agent.runners.application_service import (
            BenchmarkExecutionService,
            ExecutionOperation,
            ExecutionRequest,
        )
        from agent.runners.job_manager import verify_job_receipt

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_file = JobManagerReceiptTest._write_plan(root)
            result = BenchmarkExecutionService().execute(
                ExecutionRequest(
                    operation=ExecutionOperation.FINAL_BENCHMARK,
                    approved=True,
                    plan_file=plan_file,
                    jobs_dir=root / "jobs",
                    mock=True,
                )
            )

        self.assertEqual(len(result.receipts), 1)
        self.assertTrue(verify_job_receipt(dict(result.receipts[0])))
        self.assertEqual(result.receipts[0]["receipt_type"], "job_submission")

    def test_reconcile_retains_valid_manager_read_receipt_in_execution_state(self) -> None:
        from agent.harness.domains.execution import reconcile_execution_state
        from agent.harness.invariants import apply_state_delta
        from agent.harness.state import new_state
        from agent.runners.job_manager import get_job, submit_job

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_file = JobManagerReceiptTest._write_plan(root)
            jobs_dir = root / "jobs"
            submitted = submit_job(plan_file, jobs_dir=jobs_dir, mock=True, approved=True)
            persisted = get_job(submitted["job_id"], jobs_dir=jobs_dir)

        state = new_state("receipt-reconcile")
        state["job"] = {"job_id": submitted["job_id"], "status": "running"}
        with patch("agent.harness.domains.execution.get_job", return_value=persisted):
            result = reconcile_execution_state(state)
        updated = apply_state_delta(state, result.delta, owner="execution")
        self.assertEqual(updated["job"]["status"], "completed")
        self.assertEqual(
            updated["job"]["execution_receipts"]["last_read"]["observed_status"],
            "completed",
        )

    def test_oracle_uses_only_a_valid_manager_read_receipt_for_live_status(self) -> None:
        from agent.harness.oracle import compute_next_action
        from agent.runners.job_manager import get_job, submit_job

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_file = JobManagerReceiptTest._write_plan(root)
            jobs_dir = root / "jobs"
            submitted = submit_job(plan_file, jobs_dir=jobs_dir, mock=True, approved=True)
            persisted = get_job(submitted["job_id"], jobs_dir=jobs_dir)

        state = {"job": {"job_id": submitted["job_id"], "status": "running"}}
        with patch("agent.harness.oracle.get_job", return_value=persisted):
            action = compute_next_action(state)
        self.assertEqual(action.execution_status, "job_completed")
        self.assertEqual(action.job_read_receipt["observed_status"], "completed")

        tampered = dict(persisted)
        tampered["status"] = "failed"
        with patch("agent.harness.oracle.get_job", return_value=tampered):
            action = compute_next_action(state)
        self.assertEqual(action.execution_status, "job_unverified")
        self.assertEqual(action.job_read_receipt, {})

    def test_job_consultation_rejects_tampered_live_status(self) -> None:
        from agent.harness.domains.orientation import consultation_fragment
        from agent.harness.response_catalog import render_fragment
        from agent.runners.job_manager import get_job, submit_job

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_file = JobManagerReceiptTest._write_plan(root)
            jobs_dir = root / "jobs"
            submitted = submit_job(plan_file, jobs_dir=jobs_dir, mock=True, approved=True)
            persisted = get_job(submitted["job_id"], jobs_dir=jobs_dir)

        state = {"job": {"job_id": submitted["job_id"], "status": "running"}}
        with patch("agent.harness.domains.orientation.get_job", return_value=persisted):
            fragment = consultation_fragment(state, {"topic": "job_status"})
            message = render_fragment(fragment, "en").text
        self.assertIn("verified status: `completed`", message)

        tampered = dict(persisted)
        tampered["status"] = "failed"
        with patch("agent.harness.domains.orientation.get_job", return_value=tampered):
            fragment = consultation_fragment(state, {"topic": "job_status"})
            message = render_fragment(fragment, "en").text
        self.assertIn("cannot be verified", message)
        self.assertIn("last-known status is `running`", message)


if __name__ == "__main__":
    unittest.main()
