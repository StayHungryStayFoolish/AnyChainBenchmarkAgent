"""Tests for tamper-evident Harness coverage artifacts."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from agent.harness.coverage_events import (
    capture_coverage_events,
    emit_coverage_event,
    observe_compiled_graph_turn,
)
from agent.harness.domains.environment import question_for_environment
from agent.harness.state import new_state
from tests.agent_live.coverage_evidence import (
    COMPILED_GRAPH_RUNNER,
    build_evidence_artifact,
    build_real_execution_evidence_artifact,
    validate_evidence_artifact,
    validate_real_execution_evidence_artifact,
)
from tests.agent_live.graph_turn import invoke_product_graph_turn


class CoverageEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.edge = {
            "edge_key": "provider_deployment::MACHINE_TYPE::variant::valid_literal::manual",
            "contract_hash": "contract",
            "contract_variant_hash": "variant",
            "edge_type": "manual_input",
            "question_id": "MACHINE_TYPE",
            "expected_admitted": True,
            "evidence": {
                "deterministic": {
                    "required": True,
                    "applicability_reason": "unit-test deterministic contract",
                }
            },
        }
        self.revision = {"commit": "commit", "worktree_hash": "a" * 64}

    def _observed_turn(self) -> tuple[dict, dict, object, list]:
        seed = new_state("coverage-evidence", language="en", session_purpose="coverage")
        seed["confirmed_config"] = {"CLOUD_REGION": "test-region", "CLOUD_ZONE": "test-zone"}
        question = question_for_environment(seed, "provider_deployment")
        assert question is not None
        before = {
            **seed,
            "active_group": "provider_deployment",
            "pending_question": question,
            "last_user_input": "n2-standard-16",
        }
        with capture_coverage_events() as events:
            observation = observe_compiled_graph_turn(
                invoke_product_graph_turn,
                before,
                edge_key=self.edge["edge_key"],
                input_value="n2-standard-16",
            )
        return seed, before, observation, [event.as_dict() for event in events]

    def _artifact(self) -> dict:
        seed, before, observation, events = self._observed_turn()
        return build_evidence_artifact(
            edge=self.edge,
            evidence_class="deterministic",
            scenario_id="scenario",
            runner_type=COMPILED_GRAPH_RUNNER,
            revision=self.revision,
            input_value="n2-standard-16",
            seed_state=seed,
            before_state=before,
            after_state=observation.after,
            events=events,
            exit_status=0,
            outcome="passed",
        )

    def test_valid_artifact_requires_revision_exit_status_and_graph_observation(self) -> None:
        valid, reason = validate_evidence_artifact(self._artifact(), edge=self.edge, revision=self.revision)
        self.assertTrue(valid, reason)

    def test_tampering_or_revision_change_invalidates_artifact(self) -> None:
        artifact = self._artifact()
        artifact["after_state_hash"] = "tampered"
        valid, _ = validate_evidence_artifact(artifact, edge=self.edge, revision=self.revision)
        self.assertFalse(valid)

        artifact = self._artifact()
        valid, _ = validate_evidence_artifact(
            artifact,
            edge=self.edge,
            revision={"commit": "other", "worktree_hash": "a" * 64},
        )
        self.assertFalse(valid)

    def test_empty_identity_or_malformed_exit_status_is_rejected(self) -> None:
        artifact = self._artifact()
        artifact["revision"] = {"commit": "", "worktree_hash": "a" * 64}
        valid, _ = validate_evidence_artifact(artifact, edge=self.edge, revision=artifact["revision"])
        self.assertFalse(valid)

        artifact = self._artifact()
        artifact["exit_status"] = "not-an-integer"
        valid, _ = validate_evidence_artifact(artifact, edge=self.edge, revision=self.revision)
        self.assertFalse(valid)

    def test_failed_outcome_requires_nonzero_exit_and_failure_event(self) -> None:
        seed, before, observation, events = self._observed_turn()
        artifact = build_evidence_artifact(
            edge=self.edge,
            evidence_class="deterministic",
            scenario_id="failed-scenario",
            runner_type=COMPILED_GRAPH_RUNNER,
            revision=self.revision,
            input_value="n2-standard-16",
            seed_state=seed,
            before_state=before,
            after_state=observation.after,
            events=events,
            exit_status=1,
            outcome="failed",
            error="AssertionError: reviewed postcondition failed",
        )
        valid, reason = validate_evidence_artifact(artifact, edge=self.edge, revision=self.revision)
        self.assertTrue(valid, reason)

    def test_observer_rejects_a_helper_that_only_claims_to_be_the_product_graph(self) -> None:
        with self.assertRaisesRegex(ValueError, "helper mismatch"):
            observe_compiled_graph_turn(
                lambda state: state,
                {"last_user_input": "x"},
                edge_key=self.edge["edge_key"],
                input_value="x",
            )

    def test_event_capture_is_scoped_and_does_not_mutate_runtime_state(self) -> None:
        state = {"confirmed_config": {"BLOCKCHAIN_NODE": "solana"}}
        with capture_coverage_events() as events:
            emit_coverage_event(
                "sample_observation",
                edge_key=self.edge["edge_key"],
                component="test",
                details={"state": state},
            )
        emit_coverage_event("outside_capture", edge_key="ignored", component="test")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "sample_observation")
        self.assertEqual(state, {"confirmed_config": {"BLOCKCHAIN_NODE": "solana"}})

    def test_real_execution_artifact_is_independent_hashed_job_evidence(self) -> None:
        edge = {
            "edge_key": "@action_only/execution::::variant::action_only_transition::approve",
            "contract_hash": "execution-contract",
            "contract_variant_hash": "execution-variant",
            "evidence": {
                "real_execution": {
                    "required": True,
                    "applicability_reason": "side-effect obligation",
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir) / "job-1"
            job_dir.mkdir()
            job_file = job_dir / "job.json"
            log_file = job_dir / "benchmark.log"
            job_file.write_text('{"job_id":"job-1","status":"completed"}\n', encoding="utf-8")
            log_file.write_text("benchmark completed\n", encoding="utf-8")
            hashed = {"path": str(job_file), "sha256": hashlib.sha256(job_file.read_bytes()).hexdigest()}
            log = {"path": str(log_file), "sha256": hashlib.sha256(log_file.read_bytes()).hexdigest()}
            artifact = build_real_execution_evidence_artifact(
                edge=edge,
                revision=self.revision,
                operation_kind="preflight_smoke",
                request={"approved_plan_id": "plan-1"},
                result={"status": "completed"},
                job_id="job-1",
                job_artifacts=(hashed,),
                log_artifacts=(log,),
            )
            valid, reason = validate_real_execution_evidence_artifact(
                artifact,
                edge=edge,
                revision=self.revision,
            )
            self.assertTrue(valid, reason)
            artifact["result"]["status"] = "failed"
            valid, reason = validate_real_execution_evidence_artifact(
                artifact,
                edge=edge,
                revision=self.revision,
            )
            self.assertFalse(valid)
            self.assertIn("result hash mismatch", reason)

    def test_real_execution_builder_rejects_missing_or_forged_files(self) -> None:
        edge = {
            "edge_key": "@action_only/execution::::variant::action_only_transition::approve",
            "contract_hash": "execution-contract",
            "contract_variant_hash": "execution-variant",
            "evidence": {"real_execution": {"required": True, "applicability_reason": "side effect"}},
        }
        missing = {"path": "/tmp/job-forged/job.json", "sha256": "b" * 64}
        with self.assertRaisesRegex(ValueError, "does not exist"):
            build_real_execution_evidence_artifact(
                edge=edge,
                revision=self.revision,
                operation_kind="preflight_smoke",
                request={"approved_plan_id": "plan-1"},
                result={"status": "completed"},
                job_id="job-forged",
                job_artifacts=(missing,),
                log_artifacts=(missing,),
            )

    def test_artifact_builder_rejects_non_required_lane(self) -> None:
        edge = {
            **self.edge,
            "evidence": {
                "deterministic": {
                    "required": False,
                    "applicability_reason": "semantic-only row",
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "not required for deterministic"):
            build_evidence_artifact(
                edge=edge,
                evidence_class="deterministic",
                scenario_id="scenario",
                runner_type=COMPILED_GRAPH_RUNNER,
                revision=self.revision,
                input_value="x",
                seed_state={},
                before_state={"last_user_input": "x"},
                after_state={},
                events=(),
                exit_status=0,
                outcome="passed",
            )


if __name__ == "__main__":
    unittest.main()
