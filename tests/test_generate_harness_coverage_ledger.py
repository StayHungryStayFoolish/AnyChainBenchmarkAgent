"""Focused contract tests for the generated Harness coverage ledger."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.agent_live.generate_harness_coverage_ledger import (
    EVIDENCE_CLASSES,
    MANUAL_INPUT_CLASSES,
    build_ledger,
    contract_variant_hash,
    derive_overall_status,
    execution_exit_code,
)
from tests.agent_live.coverage_evidence import build_evidence_artifact, write_evidence_artifact
from tests.agent_live.coverage_evidence import COMPILED_GRAPH_RUNNER
from agent.harness.coverage_events import capture_coverage_events, observe_compiled_graph_turn
from tests.agent_live.graph_turn import invoke_product_graph_turn
from tests.agent_live.harness_contract_scenarios import question_scenarios


class HarnessCoverageLedgerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.revision = {"commit": "test-commit", "worktree_hash": "a" * 64}
        cls.ledger = build_ledger(revision=cls.revision)

    def test_live_registry_counts_and_runtime_only_questions_are_represented(self) -> None:
        from agent.harness.action_registry import ACTION_SPECS
        from agent.workflows.group_registry import GROUPS

        summary = self.ledger["summary"]
        self.assertEqual(summary["groups"], len(GROUPS))
        self.assertEqual(summary["registered_questions"], sum(len(group.questions) for group in GROUPS))
        self.assertEqual(summary["actions"], len(ACTION_SPECS))
        self.assertEqual(summary["action_only_transitions"], len(ACTION_SPECS))
        self.assertIn("job_monitoring:real_node_smoke_confirm", self.ledger["runtime_only_questions"])
        self.assertIn("job_monitoring:real_node_final_benchmark_confirm", self.ledger["runtime_only_questions"])

    def test_variant_identity_is_unique_and_ignores_localized_presentation(self) -> None:
        edge_keys = [edge["edge_key"] for edge in self.ledger["edges"]]
        self.assertEqual(len(edge_keys), len(set(edge_keys)))
        base = {
            "contract_version": 1,
            "id": "q",
            "group": "opening",
            "kind": "yes_no",
            "field": "answer",
            "prompt": "English prompt",
            "options": [{
                "id": "yes",
                "label": "Yes",
                "value": True,
                "action": {"type": "answer_pending"},
                "expected_patch": {"answer": True},
            }],
        }
        localized = {**base, "prompt": "中文提示"}
        localized["options"] = [{**base["options"][0], "label": "是"}]
        self.assertEqual(contract_variant_hash(base), contract_variant_hash(localized))

        resume_variants = {
            edge["contract_variant_hash"]
            for edge in self.ledger["edges"]
            if edge["group"] == "opening" and edge["question_id"] == "resume_harness_session"
        }
        self.assertGreaterEqual(len(resume_variants), 2)

    def test_edge_identity_ignores_volatile_scenario_session_timestamps(self) -> None:
        with patch("agent.harness.state._utc_timestamp", return_value="2026-01-01T00:00:00Z"):
            first = build_ledger(revision=self.revision)
        with patch("agent.harness.state._utc_timestamp", return_value="2026-12-31T23:59:59Z"):
            second = build_ledger(revision=self.revision)
        self.assertEqual(
            [edge["edge_key"] for edge in first["edges"]],
            [edge["edge_key"] for edge in second["edges"]],
        )

    def test_manual_contracts_enumerate_every_equivalence_class(self) -> None:
        manual_variants = {
            (edge["group"], edge["question_id"], edge["contract_variant_hash"])
            for edge in self.ledger["edges"]
            if edge["edge_type"] == "manual_input"
        }
        self.assertTrue(manual_variants)
        for variant in manual_variants:
            classes = {
                edge["input_class"]
                for edge in self.ledger["edges"]
                if (edge["group"], edge["question_id"], edge["contract_variant_hash"]) == variant
                and edge["edge_type"] == "manual_input"
            }
            self.assertEqual(classes, set(MANUAL_INPUT_CLASSES), variant)

        scalar_url_row = next(
            edge for edge in self.ledger["edges"]
            if edge["question_id"] == "MACHINE_TYPE" and edge["input_class"] == "reachable_url"
        )
        self.assertFalse(scalar_url_row["applicable"])
        self.assertEqual(scalar_url_row["overall_status"], "not_applicable")

    def test_catalog_alone_never_derives_passed_overall_status(self) -> None:
        applicable = [edge for edge in self.ledger["edges"] if edge["applicable"]]
        self.assertTrue(applicable)
        for edge in applicable:
            self.assertEqual(edge["evidence"]["catalog"]["status"], "passed")
            self.assertNotEqual(edge["overall_status"], "passed")
            self.assertEqual(set(edge["evidence"]), set(EVIDENCE_CLASSES))
        self.assertNotEqual(self.ledger["summary"]["overall_status"], "passed")

        catalog_only = json.loads(json.dumps(applicable[0]))
        for evidence_class in EVIDENCE_CLASSES:
            if evidence_class != "catalog":
                catalog_only["evidence"][evidence_class]["status"] = "not_applicable"
        self.assertEqual(derive_overall_status(catalog_only), "not_run")

    def test_existing_execution_evidence_requires_a_valid_artifact(self) -> None:
        first = build_ledger(revision=self.revision)
        edge = next(
            item for item in first["edges"]
            if item["question_id"] == "MACHINE_TYPE"
            and item["input_class"] == "valid_literal"
        )
        lane_reason = edge["evidence"]["deterministic"]["applicability_reason"]
        edge["evidence"]["deterministic"] = {
            "required": True,
            "applicability_reason": lane_reason,
            "status": "passed",
            "evidence_ids": ["missing.json"],
        }
        edge["deterministic_test_id"] = "stale-test-id"
        rejected = build_ledger({"schema_version": 4, "edges": [edge]}, revision=self.revision)
        restored = next(item for item in rejected["edges"] if item["edge_key"] == edge["edge_key"])
        self.assertEqual(restored["evidence"]["deterministic"]["status"], "not_run")
        self.assertEqual(restored["deterministic_test_id"], "")

        edge["evidence"]["deterministic"] = {
            "required": True,
            "applicability_reason": lane_reason,
            "status": "failed",
            "evidence_ids": ["missing.json"],
        }
        rejected_failure = build_ledger({"schema_version": 4, "edges": [edge]}, revision=self.revision)
        restored_failure = next(
            item for item in rejected_failure["edges"] if item["edge_key"] == edge["edge_key"]
        )
        self.assertEqual(restored_failure["evidence"]["deterministic"]["status"], "not_run")

        with TemporaryDirectory() as tmpdir:
            scenario = next(
                item for item in question_scenarios("en")
                if item.scenario_id == "provider_machine"
            )
            before = dict(scenario.seed_state or {})
            before["pending_question"] = dict(scenario.question)
            before["active_group"] = "provider_deployment"
            before["last_user_input"] = "n2-standard-16"
            with capture_coverage_events() as events:
                observation = observe_compiled_graph_turn(
                    invoke_product_graph_turn,
                    before,
                    edge_key=edge["edge_key"],
                    input_value="n2-standard-16",
                )
            artifact = build_evidence_artifact(
                edge=edge,
                evidence_class="deterministic",
                scenario_id="scenario",
                runner_type=COMPILED_GRAPH_RUNNER,
                revision=self.revision,
                input_value="n2-standard-16",
                seed_state=dict(scenario.seed_state or {}),
                before_state=before,
                after_state=observation.after,
                events=[event.as_dict() for event in events],
                exit_status=0,
                outcome="passed",
            )
            artifact_path = write_evidence_artifact(artifact, tmpdir).resolve()
            edge["evidence"]["deterministic"] = {
                "required": True,
                "applicability_reason": lane_reason,
                "status": "passed",
                "evidence_ids": [str(artifact_path)],
            }
            merged = build_ledger({"schema_version": 4, "edges": [edge]}, revision=self.revision)
            restored = next(item for item in merged["edges"] if item["edge_key"] == edge["edge_key"])
            self.assertEqual(restored["evidence"]["deterministic"]["status"], "passed")
            self.assertEqual(restored["evidence"]["deterministic"]["evidence_ids"], [str(artifact_path)])

            changed_revision = {"commit": "other", "worktree_hash": "b" * 64}
            invalidated = build_ledger(
                {"schema_version": 4, "edges": [edge]},
                revision=changed_revision,
            )
            restored = next(item for item in invalidated["edges"] if item["edge_key"] == edge["edge_key"])
            self.assertEqual(restored["evidence"]["deterministic"]["status"], "not_run")

    def test_catalog_build_never_populates_execution_evidence(self) -> None:
        for edge in self.ledger["edges"]:
            for evidence_class in ("deterministic", "real_cli", "dynamic_dual_ai", "real_execution"):
                evidence = edge["evidence"][evidence_class]
                self.assertNotEqual(evidence["status"], "passed")
                self.assertEqual(evidence["evidence_ids"], [])

        closure = self.ledger["summary"]["execution_closure"]["deterministic"]
        self.assertEqual(closure["status"], "incomplete")
        self.assertGreater(closure["not_run"], 0)
        self.assertEqual(execution_exit_code(self.ledger, "deterministic"), 2)

    def test_catalog_generator_runs_as_a_script_outside_repository_cwd(self) -> None:
        script = Path(__file__).resolve().parent / "agent_live" / "generate_harness_coverage_ledger.py"
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "ledger.json"
            subprocess.run(
                [sys.executable, str(script), "--output", str(output)],
                cwd=tmpdir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            ledger = json.loads(output.read_text(encoding="utf-8"))
        execution_passes = [
            evidence
            for edge in ledger["edges"]
            for name, evidence in edge["evidence"].items()
            if name != "catalog" and evidence["status"] == "passed"
        ]
        self.assertEqual(execution_passes, [])

    def test_old_ledger_schema_cannot_restore_execution_evidence(self) -> None:
        edge = next(
            item for item in self.ledger["edges"]
            if item["evidence"]["deterministic"]["required"]
        )
        stale = json.loads(json.dumps(edge))
        stale["evidence"]["deterministic"].update({
            "status": "passed",
            "evidence_ids": ["stale.json"],
        })
        rebuilt = build_ledger(
            {"schema_version": 3, "edges": [stale]},
            revision=self.revision,
        )
        restored = next(item for item in rebuilt["edges"] if item["edge_key"] == edge["edge_key"])
        self.assertEqual(restored["evidence"]["deterministic"]["status"], "not_run")
        self.assertEqual(restored["evidence"]["deterministic"]["evidence_ids"], [])


if __name__ == "__main__":
    unittest.main()
