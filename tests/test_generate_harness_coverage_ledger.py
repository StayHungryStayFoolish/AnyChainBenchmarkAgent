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
    RUNNER_CONTRACTS,
    build_ledger,
    contract_variant_hash,
    derive_overall_status,
    execution_exit_code,
)
from tests.agent_live.chaos_scheduler import build_chaos_schedule
from tests.agent_live.coverage_evidence import build_evidence_artifact, write_evidence_artifact
from tests.agent_live.coverage_evidence import COMPILED_GRAPH_RUNNER
from agent.harness.coverage_events import capture_coverage_events, observe_compiled_graph_turn
from tests.agent_live.graph_turn import invoke_product_graph_turn
from tests.agent_live.harness_contract_scenarios import question_scenarios
from tests.agent_live.harness_contract_scenarios import QuestionScenario


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

    def test_custom_rpc_manual_edges_use_canonical_owner_and_declared_successors(self) -> None:
        expected = {
            "custom_rpc_method": (
                "custom_rpc.catalog.draft.method",
                {
                    "custom_rpc_parameter_confirm",
                    "custom_rpc_schema_evidence",
                    "custom_rpc_schema_confirm",
                    "custom_rpc_response_confirm",
                },
            ),
            "custom_rpc_schema_evidence": (
                "custom_rpc.catalog.last_transition.command",
                {
                    "custom_rpc_parameter_confirm",
                    "custom_rpc_schema_confirm",
                    "custom_rpc_response_confirm",
                },
            ),
            "new_chain_method": (
                "custom_rpc.catalog.draft.method",
                {
                    "new_chain_parameter_confirm",
                    "new_chain_schema_evidence",
                    "new_chain_schema_confirm",
                    "new_chain_response_confirm",
                },
            ),
            "new_chain_schema_evidence": (
                "custom_rpc.catalog.last_transition.command",
                {
                    "new_chain_parameter_confirm",
                    "new_chain_schema_confirm",
                    "new_chain_response_confirm",
                },
            ),
        }
        for question_id, (path, next_ids) in expected.items():
            edges = [
                edge
                for edge in self.ledger["edges"]
                if edge["edge_type"] == "manual_input"
                and edge["question_id"] == question_id
            ]
            self.assertTrue(edges, question_id)
            for edge in edges:
                postcondition = edge["expected_postcondition"]
                self.assertEqual(postcondition["path"], path)
                self.assertEqual(set(postcondition["next_question_ids"]), next_ids)
                self.assertTrue(edge["executable_scenario_ids"])

    def test_structured_environment_paste_enters_review_gate(self) -> None:
        for question_id, field in (
            ("CLOUD_REGION", "CLOUD_REGION"),
            ("network_interface", "NETWORK_INTERFACE"),
        ):
            edges = [
                edge
                for edge in self.ledger["edges"]
                if edge["edge_type"] == "manual_input"
                and edge["question_id"] == question_id
                and edge["input_class"] == "structured_json_yaml_env_curl"
            ]
            self.assertTrue(edges, question_id)
            for edge in edges:
                self.assertEqual(edge["action_type"], "propose_config_values")
                self.assertTrue(edge["interrupts_pending_contract"])
                self.assertEqual(
                    edge["expected_postcondition"]["path"],
                    f"inferred_config.pending_review.config_values.{field}",
                )
                self.assertEqual(
                    edge["expected_postcondition"]["next_question_ids"],
                    ["inferred_config_review"],
                )

    def test_semantic_chain_input_uses_declared_domain_action(self) -> None:
        semantic_classes = {
            "natural_language_answer",
            "multiline_prose",
            "structured_json_yaml_env_curl",
        }
        edges = [
            edge
            for edge in self.ledger["edges"]
            if edge["edge_type"] == "manual_input"
            and edge["question_id"] == "chain"
            and edge["input_class"] in semantic_classes
        ]
        self.assertEqual({edge["input_class"] for edge in edges}, semantic_classes)
        for edge in edges:
            self.assertEqual(edge["action_type"], "choose_chain")
            self.assertEqual(
                edge["expected_postcondition"]["path"],
                "chain_identity.canonical",
            )

        scalar = next(
            edge
            for edge in self.ledger["edges"]
            if edge["edge_type"] == "manual_input"
            and edge["question_id"] == "MACHINE_TYPE"
            and edge["input_class"] == "natural_language_answer"
        )
        self.assertEqual(scalar["action_type"], "answer_pending")

    def test_manual_action_override_must_be_accepted_by_question(self) -> None:
        scenario = QuestionScenario(
            scenario_id="invalid-manual-owner",
            question={
                "id": "chain",
                "group": "chain_identity",
                "kind": "chain",
                "field": "chain",
                "prompt": "chain",
                "manual_input_allowed": True,
                "accepted_action_types": ["choose_chain"],
                "options": [],
            },
            seed_state={"active_group": "chain_identity"},
            manual_action_overrides={"natural_language_answer": "answer_pending"},
        )
        with patch(
            "tests.agent_live.generate_harness_coverage_ledger.question_scenarios",
            return_value=[scenario],
        ):
            with self.assertRaisesRegex(
                ValueError,
                "manual actions outside the question contract",
            ):
                build_ledger(revision=self.revision)

    def test_real_execution_contract_names_the_linux_producer(self) -> None:
        contract = RUNNER_CONTRACTS["real_execution"]
        self.assertEqual(contract["status"], "implemented")
        self.assertEqual(
            contract["producer"],
            "tests/agent_live/execute_real_execution_ledger.py",
        )
        self.assertEqual(contract["artifact_schema"], "real_execution_evidence.v1")
        self.assertEqual(contract["gap"], "")
        self.assertEqual(self.ledger["runner_contracts"]["real_execution"], contract)

    def test_semantic_coordinator_actions_have_only_matching_runtime_seeds(self) -> None:
        expected = {
            "change_group": "action_change_group",
            "go_back": "action_go_back",
            "queue_workflow_goal": "action_queue_workflow_goal",
            "activate_next_workflow_goal": "action_activate_next_workflow_goal",
            "discard_next_workflow_goal": "action_discard_next_workflow_goal",
        }
        action_edges = {
            edge["action_type"]: edge
            for edge in self.ledger["edges"]
            if edge["edge_type"] == "action_transition"
        }
        for action_type, scenario_id in expected.items():
            edge = action_edges[action_type]
            self.assertEqual(edge["question_id"], "")
            self.assertEqual(edge["catalog_scenario_ids"], [scenario_id])
            self.assertEqual(edge["executable_scenario_ids"], [scenario_id])
        for action_type, edge in action_edges.items():
            if action_type not in expected:
                self.assertEqual(edge["executable_scenario_ids"], [])

        schedule = build_chaos_schedule(
            self.ledger,
            revision=self.revision,
            seed=177,
            targets=[
                {
                    "target_id": f"semantic-{action_type}",
                    "edge_key": action_edges[action_type]["edge_key"],
                    "persona": "response-driven operator",
                    "goal": f"exercise {action_type} from its reviewed state",
                    "scenario_id": scenario_id,
                }
                for action_type, scenario_id in expected.items()
            ],
        )
        self.assertEqual(len(schedule.targets), len(expected))

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

    def test_resume_variants_use_saved_context_relations_not_literal_groups(self) -> None:
        resume_edges = [
            edge for edge in self.ledger["edges"]
            if edge["group"] == "opening"
            and edge["question_id"] == "resume_harness_session"
            and edge["option_id"] == "1"
            and edge["input_class"] == "exact_option"
            and edge["expected_state_relations"]
        ]
        self.assertEqual(len(resume_edges), 2)
        self.assertEqual(
            {tuple(edge["catalog_scenario_ids"]) for edge in resume_edges},
            {("resume",), ("resume_qps",)},
        )
        for edge in resume_edges:
            self.assertEqual(edge["expected_postcondition"], {"resume_context": {}})
            self.assertNotIn("chain_identity", json.dumps(edge["expected_state_relations"]))
            self.assertNotIn("qps_profile", json.dumps(edge["expected_state_relations"]))
            self.assertIn(
                {
                    "kind": "path_equals_before_path",
                    "after_path": "action_queue",
                    "before_path": "action_queue",
                },
                edge["expected_state_relations"],
            )

    def test_relation_only_postcondition_requires_both_paths_to_exist(self) -> None:
        from tests.agent_live.execute_harness_contract_ledger import _verify_postcondition

        edge = {
            "edge_type": "question_option",
            "expected_postcondition": {},
            "expected_state_relations": [{
                "kind": "path_equals_before_path",
                "after_path": "active_group",
                "before_path": "resume_context.active_group",
            }],
            "return_policy": "fallback",
        }
        before = {"resume_context": {"active_group": "qps_profile"}, "turn_index": 1}
        after = {"active_group": "qps_profile", "turn_index": 2}

        _verify_postcondition(edge, "1", before, after, expected_admitted=True)
        with self.assertRaisesRegex(AssertionError, "state relation path is absent"):
            _verify_postcondition(edge, "1", {}, after, expected_admitted=True)

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
            merged = build_ledger({"schema_version": 5, "edges": [edge]}, revision=self.revision)
            restored = next(item for item in merged["edges"] if item["edge_key"] == edge["edge_key"])
            self.assertEqual(restored["evidence"]["deterministic"]["status"], "passed")
            self.assertEqual(restored["evidence"]["deterministic"]["evidence_ids"], [str(artifact_path)])

            changed_revision = {"commit": "other", "worktree_hash": "b" * 64}
            invalidated = build_ledger(
                {"schema_version": 5, "edges": [edge]},
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
