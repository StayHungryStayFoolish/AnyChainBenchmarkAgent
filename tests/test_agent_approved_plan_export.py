from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from agent.harness.contracts import HandlerResult, StateDelta
from agent.harness.domains.execution import question_for_execution
from agent.harness.graph import AnyChainGraphRuntime
from agent.harness.graph import _commit_action_step
from agent.harness.invariants import StateInvariantError
from agent.harness.questions import choice_question, question_text
from agent.harness.state import new_state
from agent.runners.execution_scenarios import (
    RPC_BENCHMARK_WORKFLOW,
    scenario_by_id,
)
from agent.runners.application_service import (
    BenchmarkExecutionService,
    ExecutionOperation,
    ExecutionRequest,
)
from agent.runners.job_manager import _submission_receipt
from tests.agent_live.export_approved_plan import (
    _approval_chain,
    cleanup_approved_plan_evidence,
    export_approved_plan,
    validate_approved_plan_artifact,
    validate_completed_cleanup_receipt,
)
from tests.agent_live.coverage_evidence import _real_execution_scenario_error
from tests.agent_live import export_approved_plan as approved_plan_module


EMPTY_WORKTREE_HASH = hashlib.sha256(b"").hexdigest()


class ApprovedPlanExportTests(unittest.TestCase):
    def _cleanup_contract(
        self,
        artifact: Path,
        plan: Path,
    ) -> dict[str, str]:
        return {
            "artifact_file": str(artifact),
            "plan_file": str(plan),
            "approval_artifact_sha256": approved_plan_module._sha256(artifact),
            "plan_sha256": approved_plan_module._sha256(plan),
            "workflow": RPC_BENCHMARK_WORKFLOW,
            "target_mode": "real-node",
            "approval_action": "approve_preflight_smoke",
        }

    def _approved_runtime(
        self,
        root: Path,
    ) -> tuple[Path, dict[str, str], dict, dict]:
        prepared = root / ".agent" / "prepared"
        prepared.mkdir(parents=True)
        source_plan = prepared / "plan.json"
        plan = {
            "plan_id": "approved-plan-export",
            "chain": "bsc",
            "strategy": "smoke",
            "workflow_type": RPC_BENCHMARK_WORKFLOW,
            "use_fake_node": False,
            "execution": {
                "command": ["true"],
                "environment": {
                    "LOCAL_RPC_URL": "http://geth-dev:8545",
                },
                "runner_mode": "foreground",
            },
        }
        source_plan.write_text(json.dumps(plan), encoding="utf-8")
        checkpoint = root / "checkpoint.sqlite"
        runtime = AnyChainGraphRuntime(
            thread_id="thread-1",
            checkpoint_path=checkpoint,
            session_purpose="user",
        )
        state = new_state("thread-1", session_purpose="user")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = RPC_BENCHMARK_WORKFLOW
        state["chain_identity"] = {
            "raw": "bsc",
            "canonical": "bsc",
            "status": "confirmed",
            "case": "known",
        }
        state["active_group"] = "preflight_smoke_execution"
        state["plan"] = plan
        state["plan_file"] = str(source_plan)
        state["pending_question"] = choice_question(
            "preflight_smoke_execution",
            "preflight_smoke_confirm",
            question_text("question.execution.preflight_smoke.prompt"),
            owner="execution",
            field="preflight_smoke_confirmed",
            kind="yes_no",
            options=[
                {
                    "label": question_text("question.control.option.yes"),
                    "value": True,
                    "action": {"type": "approve_preflight_smoke"},
                },
                {
                    "label": question_text("question.control.option.no"),
                    "value": False,
                    "action": {"type": "reject_preflight_smoke"},
                },
            ],
            queue_barrier=True,
        )
        state["pending_question"]["execution_request_id"] = "request-1"
        runtime._persist_state(state)

        def execute_approved(input_state: dict) -> HandlerResult:
            output = deepcopy(input_state)
            output["plan"] = deepcopy(plan)
            output["plan_file"] = str(source_plan)
            execution_key = str(
                (input_state.get("side_effect_intent") or {}).get(
                    "idempotency_key"
                )
                or ""
            )
            submission = _submission_receipt(
                plan=deepcopy(plan),
                job_id="job-1",
                execution_key=execution_key,
                disposition="created",
                matching_job_count=1,
            )
            output["job"] = {
                "job_id": "job-1",
                "status": "queued",
                "execution_receipts": {
                    "submission": submission,
                    "submission_attempt": submission,
                },
            }
            return HandlerResult(
                delta=StateDelta.between(input_state, output),
                clear_pending=True,
                completion="completed",
            )

        revision = {
            "commit": "a" * 40,
            "worktree_hash": EMPTY_WORKTREE_HASH,
        }
        with (
            patch(
                "agent.harness.domains.execution.execute_approved_preflight_and_smoke",
                side_effect=execute_approved,
            ),
            patch(
                "agent.harness.graph.repository_revision",
                return_value=revision,
            ),
        ):
            approved = runtime.invoke("y", language="en")
        runtime.close()
        approval_receipts = [
            item
            for item in (approved.get("turn_context") or {}).get("control_receipts") or []
            if item.get("receipt_type") == "execution_approval"
        ]
        self.assertEqual(len(approval_receipts), 1)
        self.assertEqual(approval_receipts[0]["repository_revision"], revision)
        return checkpoint, revision, plan, approved

    def test_export_is_immutable_and_binds_real_graph_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint, revision, _plan, _approved = self._approved_runtime(root)
            output = root / "evidence"
            with patch(
                "tests.agent_live.export_approved_plan.repository_revision",
                return_value=revision,
            ):
                exported_plan, artifact = export_approved_plan(
                    repo_root=root,
                    checkpoint_path=checkpoint,
                    thread_id="thread-1",
                    session_purpose="user",
                    output_dir=output,
                )
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            exported_checkpoint = Path(payload["checkpoint_file"]["path"])
            self.assertEqual(
                stat.S_IMODE(exported_checkpoint.stat().st_mode),
                0o400,
            )
            self.assertEqual(
                stat.S_IMODE(exported_plan.stat().st_mode),
                0o400,
            )
            valid, reason = validate_approved_plan_artifact(
                artifact,
                expected_revision=revision,
                expected_plan_file=exported_plan,
                expected_workflow=RPC_BENCHMARK_WORKFLOW,
                expected_target_mode="real-node",
            )
            self.assertTrue(valid, reason)

            checkpoint.write_bytes(b"source changed after export")
            valid, reason = validate_approved_plan_artifact(
                artifact,
                expected_revision=revision,
                expected_plan_file=exported_plan,
                expected_workflow=RPC_BENCHMARK_WORKFLOW,
                expected_target_mode="real-node",
            )
            self.assertTrue(valid, reason)

            os.chmod(exported_checkpoint, 0o600)
            exported_checkpoint.write_bytes(b"tampered")
            valid, reason = validate_approved_plan_artifact(
                artifact,
                expected_revision=revision,
                expected_plan_file=exported_plan,
                expected_workflow=RPC_BENCHMARK_WORKFLOW,
                expected_target_mode="real-node",
            )
            self.assertFalse(valid)
            self.assertIn("checkpoint evidence", reason)

    def test_final_benchmark_requires_its_distinct_approval_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint, revision, plan, approved = self._approved_runtime(root)
            runtime = AnyChainGraphRuntime(
                thread_id="thread-1",
                checkpoint_path=checkpoint,
                session_purpose="user",
            )
            resumed = deepcopy(approved)
            resumed["job"]["status"] = "completed"
            resumed["smoke"] = {
                "purpose": "real_node_isolated_smoke",
                "status": "completed",
                "job_id": resumed["job"]["job_id"],
            }
            resumed["final_benchmark"] = {}
            resumed["active_group"] = "job_monitoring"
            resumed["pending_question"] = question_for_execution(
                resumed,
                "job_monitoring",
            )
            runtime._persist_state(resumed)
            source_plan = root / ".agent" / "prepared" / "plan.json"

            def execute_final(input_state: dict) -> HandlerResult:
                output = deepcopy(input_state)
                execution_key = str(
                    (input_state.get("side_effect_intent") or {}).get(
                        "idempotency_key"
                    )
                    or ""
                )
                result = BenchmarkExecutionService().execute(
                    ExecutionRequest(
                        operation=ExecutionOperation.FINAL_BENCHMARK,
                        plan_file=source_plan,
                        jobs_dir=root / "jobs",
                        approved=True,
                        mock=True,
                        idempotency_key=execution_key,
                    )
                )
                self.assertTrue(result.succeeded, result.to_dict())
                output["job"] = deepcopy(result.data["job"])
                job_id = str(output["job"]["job_id"])
                output.setdefault("final_benchmark", {}).update(
                    approved=True,
                    job_id=job_id,
                    status="queued",
                )
                return HandlerResult(
                    delta=StateDelta.between(input_state, output),
                    clear_pending=True,
                    completion="completed",
                )

            with (
                patch(
                    "agent.harness.domains.execution.execute_approved_final_benchmark",
                    side_effect=execute_final,
                ),
                patch(
                    "agent.harness.graph.repository_revision",
                    return_value=revision,
                ),
            ):
                final_state = runtime.invoke("y", language="en")
            runtime.close()
            approval = next(
                item
                for item in final_state["turn_context"]["control_receipts"]
                if item.get("receipt_type") == "execution_approval"
            )
            self.assertEqual(
                approval["approval_action_type"],
                "approve_final_benchmark",
            )
            with patch(
                "tests.agent_live.export_approved_plan.repository_revision",
                return_value=revision,
            ):
                exported_plan, artifact = export_approved_plan(
                    repo_root=root,
                    checkpoint_path=checkpoint,
                    thread_id="thread-1",
                    session_purpose="user",
                    output_dir=root / "final-evidence",
                )
            valid, reason = validate_approved_plan_artifact(
                artifact,
                expected_revision=revision,
                expected_plan_file=exported_plan,
                expected_workflow=RPC_BENCHMARK_WORKFLOW,
                expected_target_mode="real-node",
                expected_approval_action="approve_final_benchmark",
            )
            self.assertTrue(valid, reason)
            valid, reason = validate_approved_plan_artifact(
                artifact,
                expected_revision=revision,
                expected_plan_file=exported_plan,
                expected_workflow=RPC_BENCHMARK_WORKFLOW,
                expected_target_mode="real-node",
                expected_approval_action="approve_preflight_smoke",
            )
            self.assertFalse(valid)
            self.assertIn("wrong approval stage", reason)
            composed_reason = _real_execution_scenario_error(
                {
                    "revision": revision,
                    "request": {
                        "scenario_id": "rpc_real_node_final",
                        "service_operation": "final_benchmark",
                        "approved_plan_file": str(exported_plan),
                        "approved_plan_sha256": hashlib.sha256(
                            exported_plan.read_bytes()
                        ).hexdigest(),
                        "approved_plan_revision": revision,
                        "approved_plan_provenance_file": str(artifact),
                        "approved_plan_provenance_sha256": hashlib.sha256(
                            artifact.read_bytes()
                        ).hexdigest(),
                        "admission_envelope_file": str(
                            root / "missing-envelope.json"
                        ),
                        "admission_envelope_sha256": "",
                    },
                },
                scenario_by_id("rpc_real_node_final"),
            )
            self.assertIn("G5 admission envelope", composed_reason)
            self.assertNotIn("approved-plan provenance", composed_reason)

    def test_export_rejects_a_dirty_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with (
                patch(
                    "tests.agent_live.export_approved_plan.repository_revision",
                    return_value={"commit": "a" * 40, "worktree_hash": "b" * 64},
                ),
                self.assertRaisesRegex(ValueError, "clean tracked worktree"),
            ):
                export_approved_plan(
                    repo_root=root,
                    checkpoint_path=root / "checkpoint.sqlite",
                    thread_id="thread-1",
                    session_purpose="user",
                    output_dir=root / "evidence",
                )

    def test_validator_returns_failure_for_malformed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "approval.json"
            artifact.write_text('{"checkpoint_file": []}', encoding="utf-8")
            valid, reason = validate_approved_plan_artifact(
                artifact,
                expected_revision={
                    "commit": "a" * 40,
                    "worktree_hash": EMPTY_WORKTREE_HASH,
                },
                expected_plan_file=root / "missing.json",
                expected_workflow=RPC_BENCHMARK_WORKFLOW,
                expected_target_mode="real-node",
            )
            self.assertFalse(valid)
            self.assertIn("approved-plan artifact is invalid", reason)

    def test_approval_chain_rejects_negative_or_side_effect_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _checkpoint, _revision, _plan, approved = self._approved_runtime(
                Path(tmpdir)
            )
            negative = deepcopy(approved)
            answer = next(
                item
                for item in negative["completed_actions"]
                if item.get("type") == "answer_pending"
            )
            answer["selected_value"] = False
            with self.assertRaisesRegex(ValueError, "affirmative"):
                _approval_chain(negative)

            substituted = deepcopy(approved)
            substituted["side_effect_receipt"]["job_id"] = "job-substituted"
            with self.assertRaisesRegex(ValueError, "submitted side effect"):
                _approval_chain(substituted)

    def test_approval_chain_rejects_a_valid_submission_for_another_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _checkpoint, _revision, _plan, approved = self._approved_runtime(
                Path(tmpdir)
            )
            substituted = deepcopy(approved)
            intent = substituted["side_effect_intent"]
            foreign_plan = {
                "chain": "ethereum",
                "workflow_type": RPC_BENCHMARK_WORKFLOW,
                "use_fake_node": False,
            }
            foreign_submission = _submission_receipt(
                plan=foreign_plan,
                job_id=substituted["job"]["job_id"],
                execution_key=intent["idempotency_key"],
                disposition="created",
                matching_job_count=1,
            )
            substituted["job"]["execution_receipts"] = {
                "submission": foreign_submission,
                "submission_attempt": foreign_submission,
            }
            approval = next(
                item
                for item in substituted["turn_context"]["control_receipts"]
                if item.get("receipt_type") == "execution_approval"
            )
            approval["job_submission_receipt_id"] = foreign_submission["receipt_id"]
            approval["job_submission_receipt_hash"] = hashlib.sha256(
                json.dumps(
                    foreign_submission,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            unsigned = {
                key: value
                for key, value in approval.items()
                if key != "receipt_id"
            }
            approval["receipt_id"] = hashlib.sha256(
                json.dumps(
                    unsigned,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "submitted side effect"):
                _approval_chain(substituted)

    def test_validator_requires_content_addressed_approval_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint, revision, _plan, _approved = self._approved_runtime(
                root
            )
            with patch(
                "tests.agent_live.export_approved_plan.repository_revision",
                return_value=revision,
            ):
                exported_plan, artifact = export_approved_plan(
                    repo_root=root,
                    checkpoint_path=checkpoint,
                    thread_id="thread-1",
                    session_purpose="user",
                    output_dir=root / "evidence",
                )
            renamed = artifact.with_name("approval-renamed.json")
            artifact.rename(renamed)
            valid, reason = validate_approved_plan_artifact(
                renamed,
                expected_revision=revision,
                expected_plan_file=exported_plan,
                expected_workflow=RPC_BENCHMARK_WORKFLOW,
                expected_target_mode="real-node",
                expected_approval_action="approve_preflight_smoke",
            )
            self.assertFalse(valid)
            self.assertIn("identity", reason)

    def test_normal_turn_does_not_probe_repository_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(
                thread_id="normal-turn",
                checkpoint_path=Path(tmpdir) / "checkpoint.sqlite",
            )
            state = new_state("normal-turn")
            with (
                patch.object(runtime, "_load_state", return_value=state),
                patch.object(runtime.graph, "invoke", return_value=state),
                patch("agent.harness.graph.repository_revision") as revision,
            ):
                runtime.invoke("What can you do?", language="en")
            runtime.close()
            revision.assert_not_called()

    def test_execution_commit_fails_if_revision_changes_after_owner_step(
        self,
    ) -> None:
        expected = {
            "commit": "a" * 40,
            "worktree_hash": EMPTY_WORKTREE_HASH,
        }
        changed = {
            "commit": "b" * 40,
            "worktree_hash": EMPTY_WORKTREE_HASH,
        }
        state = new_state("revision-race")
        state["selected_action"] = {
            "type": "approve_preflight_smoke",
            "action_type": "approve_preflight_smoke",
        }
        state["turn_context"]["repository_revision"] = expected
        with (
            patch(
                "agent.harness.graph.repository_revision",
                return_value=changed,
            ),
            self.assertRaisesRegex(
                StateInvariantError,
                "changed during execution approval",
            ),
        ):
            _commit_action_step(state, None)

    def test_export_reads_one_committed_snapshot_while_wal_writer_is_active(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint, revision, _plan, _approved = self._approved_runtime(
                root
            )
            writer = sqlite3.connect(checkpoint)
            try:
                writer.execute("PRAGMA journal_mode=WAL")
                writer.execute("BEGIN IMMEDIATE")
                writer.execute(
                    "UPDATE checkpoints SET metadata = ? WHERE thread_id = ?",
                    (b"uncommitted-substitution", "thread-1"),
                )
                with patch(
                    "tests.agent_live.export_approved_plan.repository_revision",
                    return_value=revision,
                ):
                    exported_plan, artifact = export_approved_plan(
                        repo_root=root,
                        checkpoint_path=checkpoint,
                        thread_id="thread-1",
                        session_purpose="user",
                        output_dir=root / "wal-evidence",
                    )
                valid, reason = validate_approved_plan_artifact(
                    artifact,
                    expected_revision=revision,
                    expected_plan_file=exported_plan,
                    expected_workflow=RPC_BENCHMARK_WORKFLOW,
                    expected_target_mode="real-node",
                    expected_approval_action="approve_preflight_smoke",
                )
                self.assertTrue(valid, reason)
            finally:
                writer.rollback()
                writer.close()

    def test_terminal_cleanup_deletes_confidential_sources_and_keeps_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint, revision, _plan, _approved = self._approved_runtime(
                root
            )
            with patch(
                "tests.agent_live.export_approved_plan.repository_revision",
                return_value=revision,
            ):
                exported_plan, artifact = export_approved_plan(
                    repo_root=root,
                    checkpoint_path=checkpoint,
                    thread_id="thread-1",
                    session_purpose="user",
                    output_dir=root / "evidence",
                )
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            exported_checkpoint = Path(payload["checkpoint_file"]["path"])
            receipt = cleanup_approved_plan_evidence(
                [
                    self._cleanup_contract(artifact, exported_plan),
                ],
                expected_revision=revision,
                output_dir=root / "cleanup",
            )
            self.assertFalse(artifact.exists())
            self.assertFalse(exported_plan.exists())
            self.assertFalse(exported_checkpoint.exists())
            self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o400)
            cleanup = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(cleanup["deleted_file_count"], 3)

    def test_cleanup_rename_failure_keeps_all_confidential_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint, revision, _plan, _approved = self._approved_runtime(
                root
            )
            with patch(
                "tests.agent_live.export_approved_plan.repository_revision",
                return_value=revision,
            ):
                exported_plan, artifact = export_approved_plan(
                    repo_root=root,
                    checkpoint_path=checkpoint,
                    thread_id="thread-1",
                    session_purpose="user",
                    output_dir=root / "evidence",
                )
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            exported_checkpoint = Path(payload["checkpoint_file"]["path"])
            cleanup_dir = root / "cleanup"
            with (
                patch(
                    "tests.agent_live.export_approved_plan.os.rename",
                    side_effect=OSError("rename interrupted"),
                ),
                self.assertRaises(OSError),
            ):
                cleanup_approved_plan_evidence(
                    [
                        self._cleanup_contract(artifact, exported_plan),
                    ],
                    expected_revision=revision,
                    output_dir=cleanup_dir,
                )
            self.assertTrue(artifact.is_file())
            self.assertTrue(exported_plan.is_file())
            self.assertTrue(exported_checkpoint.is_file())
            prepared = list(
                cleanup_dir.glob("approved-plan-cleanup-prepared-*.json")
            )
            self.assertEqual(len(prepared), 1)
            self.assertEqual(
                json.loads(prepared[0].read_text(encoding="utf-8"))["status"],
                "prepared",
            )

    def test_cleanup_rejects_exact_plan_digest_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint, revision, _plan, _approved = self._approved_runtime(
                root
            )
            with patch(
                "tests.agent_live.export_approved_plan.repository_revision",
                return_value=revision,
            ):
                exported_plan, artifact = export_approved_plan(
                    repo_root=root,
                    checkpoint_path=checkpoint,
                    thread_id="thread-1",
                    session_purpose="user",
                    output_dir=root / "evidence",
                )
            contract = self._cleanup_contract(artifact, exported_plan)
            contract["plan_sha256"] = "f" * 64
            with self.assertRaisesRegex(
                ValueError,
                "contract digest differs",
            ):
                cleanup_approved_plan_evidence(
                    [contract],
                    expected_revision=revision,
                    output_dir=root / "cleanup",
                )
            self.assertTrue(artifact.is_file())
            self.assertTrue(exported_plan.is_file())

    def test_formal_path_is_readonly_before_followup_seal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint, revision, _plan, _approved = self._approved_runtime(
                root
            )
            evidence = root / "evidence"
            with (
                patch(
                    "tests.agent_live.export_approved_plan.repository_revision",
                    return_value=revision,
                ),
                patch(
                    "tests.agent_live.export_approved_plan._seal",
                    side_effect=SystemExit("crash after atomic publication"),
                ),
                self.assertRaises(SystemExit),
            ):
                export_approved_plan(
                    repo_root=root,
                    checkpoint_path=checkpoint,
                    thread_id="thread-1",
                    session_purpose="user",
                    output_dir=evidence,
                )
            published = list(evidence.iterdir())
            self.assertEqual(len(published), 1)
            self.assertFalse(published[0].name.startswith("."))
            self.assertEqual(stat.S_IMODE(published[0].stat().st_mode), 0o400)

    def test_final_receipt_failure_fails_closed_and_retry_recovers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint, revision, _plan, _approved = self._approved_runtime(
                root
            )
            with patch(
                "tests.agent_live.export_approved_plan.repository_revision",
                return_value=revision,
            ):
                exported_plan, artifact = export_approved_plan(
                    repo_root=root,
                    checkpoint_path=checkpoint,
                    thread_id="thread-1",
                    session_purpose="user",
                    output_dir=root / "evidence",
                )
            contract = self._cleanup_contract(artifact, exported_plan)
            original_write = approved_plan_module._write_exclusive

            def fail_final_receipt(path: Path, payload: bytes) -> None:
                if path.name.startswith("approved-plan-cleanup-") and not (
                    "prepared-" in path.name or "committed-" in path.name
                ):
                    raise OSError("final receipt storage unavailable")
                original_write(path, payload)

            with patch(
                "tests.agent_live.export_approved_plan._write_exclusive",
                side_effect=fail_final_receipt,
            ), self.assertRaisesRegex(
                OSError,
                "final receipt storage unavailable",
            ):
                cleanup_approved_plan_evidence(
                    [
                        contract,
                    ],
                    expected_revision=revision,
                    output_dir=root / "cleanup",
                )
            self.assertFalse((root / "evidence").exists())
            committed = list(
                (root / "cleanup").glob(
                    "approved-plan-cleanup-committed-*.json"
                )
            )
            self.assertEqual(len(committed), 1)
            committed_payload = json.loads(
                committed[0].read_text(encoding="utf-8")
            )
            self.assertFalse(
                Path(committed_payload["quarantine_path"]).exists()
            )

            receipt = cleanup_approved_plan_evidence(
                [
                    contract,
                ],
                expected_revision=revision,
                output_dir=root / "cleanup",
            )
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")

    def test_rename_then_crash_recovers_from_prepared_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint, revision, _plan, _approved = self._approved_runtime(
                root
            )
            with patch(
                "tests.agent_live.export_approved_plan.repository_revision",
                return_value=revision,
            ):
                exported_plan, artifact = export_approved_plan(
                    repo_root=root,
                    checkpoint_path=checkpoint,
                    thread_id="thread-1",
                    session_purpose="user",
                    output_dir=root / "evidence",
                )
            contract = self._cleanup_contract(artifact, exported_plan)
            original_commit = approved_plan_module._write_cleanup_committed
            with (
                patch(
                    "tests.agent_live.export_approved_plan._write_cleanup_committed",
                    side_effect=SystemExit("simulated process crash"),
                ),
                self.assertRaises(SystemExit),
            ):
                cleanup_approved_plan_evidence(
                    [
                        contract,
                    ],
                    expected_revision=revision,
                    output_dir=root / "cleanup",
                )
            self.assertFalse((root / "evidence").exists())

            with patch(
                "tests.agent_live.export_approved_plan._write_cleanup_committed",
                wraps=original_commit,
            ):
                receipt = cleanup_approved_plan_evidence(
                    [
                        contract,
                    ],
                    expected_revision=revision,
                    output_dir=root / "cleanup",
                )
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertFalse(Path(payload["quarantine_path"]).exists())

    def test_partial_quarantine_delete_resumes_to_completed_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint, revision, _plan, _approved = self._approved_runtime(
                root
            )
            with patch(
                "tests.agent_live.export_approved_plan.repository_revision",
                return_value=revision,
            ):
                exported_plan, artifact = export_approved_plan(
                    repo_root=root,
                    checkpoint_path=checkpoint,
                    thread_id="thread-1",
                    session_purpose="user",
                    output_dir=root / "evidence",
                )
            contract = self._cleanup_contract(artifact, exported_plan)

            def partial_delete(path: Path) -> None:
                next(iter(path.iterdir())).unlink()
                raise OSError("simulated partial delete")

            with (
                patch(
                    "tests.agent_live.export_approved_plan.shutil.rmtree",
                    side_effect=partial_delete,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "remains quarantined",
                ),
            ):
                cleanup_approved_plan_evidence(
                    [
                        contract,
                    ],
                    expected_revision=revision,
                    output_dir=root / "cleanup",
                )

            receipt = cleanup_approved_plan_evidence(
                [
                    contract,
                ],
                expected_revision=revision,
                output_dir=root / "cleanup",
            )
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertFalse(Path(payload["quarantine_path"]).exists())

    def test_cleanup_rejects_malformed_journal_in_transaction_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint, revision, _plan, _approved = self._approved_runtime(
                root
            )
            with patch(
                "tests.agent_live.export_approved_plan.repository_revision",
                return_value=revision,
            ):
                exported_plan, artifact = export_approved_plan(
                    repo_root=root,
                    checkpoint_path=checkpoint,
                    thread_id="thread-1",
                    session_purpose="user",
                    output_dir=root / "evidence",
                )
            cleanup_dir = root / "cleanup"
            cleanup_dir.mkdir(mode=0o700)
            malformed = cleanup_dir / (
                "approved-plan-cleanup-committed-invalid.json"
            )
            malformed.write_text("{", encoding="utf-8")
            malformed.chmod(0o400)
            with self.assertRaises(json.JSONDecodeError):
                cleanup_approved_plan_evidence(
                    [
                        self._cleanup_contract(artifact, exported_plan),
                    ],
                    expected_revision=revision,
                    output_dir=cleanup_dir,
                )

    def test_completed_receipt_requires_full_prepared_and_committed_chain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cleanup_dir = root / "cleanup"
            cleanup_dir.mkdir(mode=0o700)
            revision = {
                "commit": "a" * 40,
                "worktree_hash": EMPTY_WORKTREE_HASH,
            }
            receipt = {
                "schema_version": 1,
                "artifact_type": "approved_plan_cleanup_receipt",
                "status": "completed",
                "security_contract": approved_plan_module.SECURITY_CONTRACT,
                "revision": revision,
                "prepared_journal": str(cleanup_dir / "missing-prepared.json"),
                "prepared_journal_sha256": "b" * 64,
                "committed_journal": str(cleanup_dir / "missing-committed.json"),
                "committed_journal_sha256": "c" * 64,
                "evidence_root": str(root / "evidence"),
                "quarantine_path": str(root / ".evidence.cleanup-invalid"),
                "validated_contracts": [],
                "deleted_files": [],
                "deleted_file_count": 0,
                "completed_at": "2026-07-25T00:00:00Z",
            }
            receipt["receipt_hash"] = approved_plan_module._content_hash(
                receipt
            )
            path = cleanup_dir / (
                f"approved-plan-cleanup-{receipt['receipt_hash']}.json"
            )
            path.write_bytes(approved_plan_module._canonical_bytes(receipt))
            path.chmod(0o400)
            with self.assertRaises(OSError):
                validate_completed_cleanup_receipt(
                    path,
                    expected_revision=revision,
                    expected_evidence_root=root / "evidence",
                    expected_contracts=[
                        {
                            "approval_action": "approve_preflight_smoke",
                            "workflow": RPC_BENCHMARK_WORKFLOW,
                            "target_mode": "real-node",
                        },
                    ],
                    output_dir=cleanup_dir,
                )


if __name__ == "__main__":
    unittest.main()
