"""Focused contract tests for the generated Harness coverage ledger."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.agent_live.generate_harness_coverage_ledger import (
    EVIDENCE_CLASSES,
    LEDGER_SCHEMA_VERSION,
    MANUAL_INPUT_CLASSES,
    RUNNER_CONTRACTS,
    build_ledger,
    contract_variant_hash,
    derive_overall_status,
    execution_exit_code,
    ingest_evidence_artifacts,
)
from tests.agent_live.chaos_scheduler import build_chaos_schedule
from tests.agent_live.coverage_evidence import build_evidence_artifact, write_evidence_artifact
from tests.agent_live.coverage_evidence import COMPILED_GRAPH_RUNNER
from tests.agent_live.coverage_events import capture_coverage_events, observe_compiled_graph_turn
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
        from agent.workflows.group_registry import USER_NAVIGABLE_GROUPS

        self.assertEqual(
            summary["action_only_transitions"],
            len(ACTION_SPECS) - 1 + len(USER_NAVIGABLE_GROUPS),
        )
        self.assertIn("job_monitoring:real_node_smoke_confirm", self.ledger["runtime_only_questions"])
        self.assertIn("job_monitoring:real_node_final_benchmark_confirm", self.ledger["runtime_only_questions"])

    def test_dynamic_action_edges_expose_authoritative_simulator_contract(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE

        for action_type in ("queue_workflow_goal", "discard_next_workflow_goal", "change_group"):
            with self.subTest(action_type=action_type):
                edge = next(
                    item
                    for item in self.ledger["edges"]
                    if item["edge_type"] == "action_transition"
                    and item["action_type"] == action_type
                )
                spec = ACTION_BY_TYPE[action_type]
                self.assertEqual(edge["simulator_action_contract"], {
                    "purpose": spec.purpose,
                    "effect": spec.effect,
                    "allowed_arguments": list(spec.allowed_arguments),
                    "required_arguments": list(spec.required_arguments),
                    "constraints": list(spec.constraints),
                })

    def test_navigation_contract_declares_immediate_and_prerequisite_deferred_routes(self) -> None:
        edge = next(
            item
            for item in self.ledger["edges"]
            if item["edge_type"] == "action_transition"
            and item["action_type"] == "change_group"
            and item["expected_postcondition"].get("target_group") == "sync_observe"
        )

        self.assertEqual(
            edge["expected_postcondition"]["navigation_transition"],
            {
                "immediate_group": "sync_observe",
                "prerequisite_deferred_groups": [
                    "target_mode",
                    "chain_identity",
                    "endpoint_process",
                ],
            },
        )
        self.assertEqual(
            edge["deferred_transition_contract"],
            {
                "requires_linked_journey": True,
                "max_continuation_turns": 12,
                "entry_prerequisite_groups": [
                    "target_mode",
                    "chain_identity",
                    "endpoint_process",
                ],
                "terminal_group": "sync_observe",
                "terminal_postcondition": {"target_group": "sync_observe"},
                "linkage": [
                    "same_thread",
                    "same_revision",
                    "contiguous_turn_index",
                    "contiguous_fingerprint_chain",
                ],
            },
        )

    def test_natural_language_option_declares_contract_owner_equivalence(self) -> None:
        natural = next(
            item
            for item in self.ledger["edges"]
            if item["question_id"] == "observability_mode"
            and item["option_value"] == "exporter"
            and item["input_class"] == "natural_language_option"
        )
        exact = next(
            item
            for item in self.ledger["edges"]
            if item["question_id"] == "observability_mode"
            and item["option_value"] == "exporter"
            and item["input_class"] == "exact_option"
        )

        self.assertEqual(
            natural["expected_admitted_action_types"],
            ["answer_pending", "set_observability"],
        )
        self.assertEqual(
            natural["prerequisite_deferred_actions"],
            {"set_observability": ["target_mode"]},
        )
        self.assertTrue(
            natural["deferred_transition_contract"]["requires_linked_journey"]
        )
        self.assertEqual(
            natural["deferred_transition_contract"]["terminal_postcondition"],
            {"observability.mode": "exporter"},
        )
        self.assertEqual(exact["expected_admitted_action_types"], ["answer_pending"])
        self.assertEqual(exact["prerequisite_deferred_actions"], {})
        self.assertEqual(exact["deferred_transition_contract"], {})

    def test_source_grounded_navigation_actions_declare_non_mutation_scope(self) -> None:
        from agent.harness.action_registry import ACTION_SPECS

        navigation = [
            spec
            for spec in ACTION_SPECS
            if spec.effect == "workflow_navigation"
            and "source_evidence" in spec.allowed_arguments
        ]
        self.assertTrue(navigation)
        self.assertEqual(
            [
                spec.action_type
                for spec in navigation
                if "non_mutation_scope" not in spec.semantic_support_relations
            ],
            [],
        )

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

    def test_domain_manual_evidence_paths_come_from_question_contracts(self) -> None:
        expected = {
            "custom_rpc_endpoint": "custom_rpc.endpoint",
            "new_chain_endpoint": "endpoint_evidence.candidate_endpoint",
            "case3_protocol_evidence": "secondary_handoff.evidence",
            "chain": "chain_identity.canonical",
            "chain_change_input": "chain_identity.change_candidate.canonical",
            "SYNC_OBSERVE_RPC_URL": "endpoint_evidence.sync_rpc_url_ready",
        }
        for question_id, path in expected.items():
            edges = [
                edge
                for edge in self.ledger["edges"]
                if edge["edge_type"] == "manual_input"
                and edge["question_id"] == question_id
            ]
            self.assertTrue(edges, question_id)
            self.assertTrue(all(
                edge["expected_postcondition"]["path"] == path
                for edge in edges
            ), question_id)

    def test_endpoint_rejection_evidence_is_declared_by_question_contract(self) -> None:
        for question_id in ("LOCAL_RPC_URL", "SYNC_OBSERVE_RPC_URL"):
            edge = next(
                item for item in self.ledger["edges"]
                if item["question_id"] == question_id
                and item["input_class"] == "unreachable_or_mismatched_url"
            )
            self.assertFalse(edge["expected_admitted"])
            self.assertIs(edge["expected_postcondition"]["rejection_value"], False)

        ordinary = next(
            item for item in self.ledger["edges"]
            if item["question_id"] == "ACCOUNTS_VOL_MAX_IOPS"
            and item["expected_admitted"] is False
        )
        self.assertNotIn("rejection_value", ordinary["expected_postcondition"])

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
            expected_action = (
                "choose_chain"
                if "chain_manual" in edge["catalog_scenario_ids"]
                else "answer_pending"
            )
            self.assertEqual(edge["action_type"], expected_action)
            self.assertEqual(
                edge["expected_postcondition"]["path"],
                "chain_identity.canonical",
            )

        replacement_edges = [
            edge
            for edge in self.ledger["edges"]
            if edge["edge_type"] == "manual_input"
            and edge["question_id"] == "chain_change_input"
            and edge["input_class"] in semantic_classes
        ]
        self.assertEqual(
            {edge["input_class"] for edge in replacement_edges},
            semantic_classes,
        )
        for edge in replacement_edges:
            self.assertEqual(edge["action_type"], "change_chain")
            self.assertEqual(
                edge["expected_postcondition"]["path"],
                "chain_identity.change_candidate.canonical",
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
        self.assertEqual(contract["artifact_schema"], "real_execution_evidence.v3")
        self.assertEqual(contract["gap"], "")
        self.assertEqual(self.ledger["runner_contracts"]["real_execution"], contract)

    def test_real_execution_closure_is_counted_by_scenario(self) -> None:
        closure = self.ledger["summary"]["execution_closure"]["real_execution"]
        self.assertEqual(closure["required_denominator"], 4)
        self.assertEqual(closure["not_run"], 4)
        lanes = {
            edge["action_type"]: edge["evidence"]["real_execution"]
            for edge in self.ledger["edges"]
            if edge["evidence"]["real_execution"]["required"]
        }
        self.assertEqual(
            lanes["approve_preflight_smoke"]["required_scenario_ids"],
            ["rpc_fake_node_smoke", "rpc_real_node_smoke", "sync_observe_bounded"],
        )
        self.assertEqual(
            lanes["approve_final_benchmark"]["required_scenario_ids"],
            ["rpc_real_node_final"],
        )

    def test_partial_real_execution_scenario_does_not_complete_action_lane(self) -> None:
        ledger = build_ledger(revision=self.revision)
        edge = next(
            item
            for item in ledger["edges"]
            if item["action_type"] == "approve_preflight_smoke"
            and item["evidence"]["real_execution"]["required"]
        )
        with TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "rpc-smoke.json"
            artifact = {
                "edge_key": edge["edge_key"],
                "evidence_class": "real_execution",
                "scenario_id": "rpc_real_node_smoke",
                "outcome": "passed",
            }
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            with patch(
                "tests.agent_live.generate_harness_coverage_ledger.load_valid_evidence_reference",
                return_value=(artifact, ""),
            ):
                updated = ingest_evidence_artifacts(ledger, [artifact_path])

        updated_edge = next(
            item for item in updated["edges"]
            if item["edge_key"] == edge["edge_key"]
        )
        lane = updated_edge["evidence"]["real_execution"]
        self.assertEqual(lane["status"], "not_run")
        self.assertEqual(
            lane["scenario_evidence"]["rpc_real_node_smoke"]["status"],
            "passed",
        )
        self.assertNotIn("sync_observe_bounded", lane["scenario_evidence"])
        closure = updated["summary"]["execution_closure"]["real_execution"]
        self.assertEqual(closure["observed_pass"], 1)
        self.assertEqual(closure["not_run"], 3)

    def test_rebuild_preserves_each_valid_real_execution_scenario(self) -> None:
        ledger = build_ledger(revision=self.revision)
        edge = next(
            item
            for item in ledger["edges"]
            if item["action_type"] == "approve_preflight_smoke"
            and item["evidence"]["real_execution"]["required"]
        )
        references = {
            "rpc_fake_node_smoke": "/tmp/rpc-fake-node-smoke.json",
            "rpc_real_node_smoke": "/tmp/rpc-real-node-smoke.json",
            "sync_observe_bounded": "/tmp/sync-observe-bounded.json",
        }
        old_edge = deepcopy(edge)
        old_edge["evidence"]["real_execution"].update({
            "status": "passed",
            "evidence_ids": list(references.values()),
        })

        def load_reference(reference: str, **_kwargs: object) -> tuple[dict[str, str], str]:
            scenario_id = next(
                scenario
                for scenario, expected_reference in references.items()
                if reference == expected_reference
            )
            return {
                "evidence_class": "real_execution",
                "scenario_id": scenario_id,
                "outcome": "passed",
            }, ""

        with patch(
            "tests.agent_live.generate_harness_coverage_ledger.load_valid_evidence_reference",
            side_effect=load_reference,
        ):
            rebuilt = build_ledger(
                {"schema_version": LEDGER_SCHEMA_VERSION, "edges": [old_edge]},
                revision=self.revision,
            )

        restored = next(
            item for item in rebuilt["edges"]
            if item["edge_key"] == edge["edge_key"]
        )
        lane = restored["evidence"]["real_execution"]
        self.assertEqual(lane["status"], "passed")
        self.assertEqual(
            {
                scenario: evidence["status"]
                for scenario, evidence in lane["scenario_evidence"].items()
            },
            {
                "rpc_fake_node_smoke": "passed",
                "rpc_real_node_smoke": "passed",
                "sync_observe_bounded": "passed",
            },
        )

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

        structured_owner = {**base, "structured_input_owner": True}
        self.assertNotEqual(
            contract_variant_hash(base),
            contract_variant_hash(structured_owner),
        )

        described = deepcopy(base)
        described["options"][0]["description"] = "Validates with recorded fixtures."
        changed_description = deepcopy(described)
        changed_description["options"][0]["description"] = "Measures real-node performance."
        self.assertNotEqual(
            contract_variant_hash(described),
            contract_variant_hash(changed_description),
        )

        resume_variants = {
            edge["contract_variant_hash"]
            for edge in self.ledger["edges"]
            if edge["group"] == "opening" and edge["question_id"] == "resume_harness_session"
        }
        self.assertGreaterEqual(len(resume_variants), 2)

    def test_action_identity_includes_semantic_support_contract(self) -> None:
        from agent.harness.action_registry import ACTION_SPECS, ActionSpec

        go_back = next(spec for spec in ACTION_SPECS if spec.action_type == "go_back")
        self.assertEqual(
            ActionSpec("future_action", "test", "future action").semantic_support_relations,
            (),
        )

        def edge_for(specs: tuple[object, ...]) -> dict[str, object]:
            with patch("agent.harness.action_registry.ACTION_SPECS", specs):
                ledger = build_ledger(revision=self.revision)
            return next(
                edge
                for edge in ledger["edges"]
                if edge["edge_type"] == "action_transition"
                and edge["action_type"] == "go_back"
            )

        reordered = replace(
            go_back,
            semantic_support_relations=tuple(
                reversed(go_back.semantic_support_relations)
            ),
        )
        expanded = replace(
            go_back,
            semantic_support_relations=tuple(
                relation
                for relation in go_back.semantic_support_relations
                if relation != "non_mutation_scope"
            ),
        )
        reordered_specs = tuple(
            reordered if spec.action_type == "go_back" else spec
            for spec in ACTION_SPECS
        )
        expanded_specs = tuple(
            expanded if spec.action_type == "go_back" else spec
            for spec in ACTION_SPECS
        )
        changed_effect = replace(go_back, effect="read_only")
        changed_effect_specs = tuple(
            changed_effect if spec.action_type == "go_back" else spec
            for spec in ACTION_SPECS
        )

        baseline_edge = next(
            edge
            for edge in self.ledger["edges"]
            if edge["edge_type"] == "action_transition"
            and edge["action_type"] == "go_back"
        )
        reordered_edge = edge_for(reordered_specs)
        expanded_edge = edge_for(expanded_specs)
        changed_effect_edge = edge_for(changed_effect_specs)

        self.assertEqual(
            baseline_edge["semantic_recovery_source_argument"],
            go_back.semantic_recovery_source_argument,
        )
        self.assertEqual(
            baseline_edge["semantic_support_relations"],
            sorted(go_back.semantic_support_relations),
        )
        for identity_field in ("contract_hash", "contract_variant_hash", "edge_key"):
            self.assertEqual(baseline_edge[identity_field], reordered_edge[identity_field])
            self.assertNotEqual(baseline_edge[identity_field], expanded_edge[identity_field])
            self.assertNotEqual(
                baseline_edge[identity_field],
                changed_effect_edge[identity_field],
            )

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
        rejected_failure = build_ledger(
            {"schema_version": LEDGER_SCHEMA_VERSION - 1, "edges": [edge]},
            revision=self.revision,
        )
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
            from tests.agent_live.graph_turn import reviewed_execution_planner

            with capture_coverage_events() as events, reviewed_execution_planner(
                expected_input="n2-standard-16",
                expected_admitted=True,
            ):
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
            merged = build_ledger(
                {"schema_version": LEDGER_SCHEMA_VERSION, "edges": [edge]},
                revision=self.revision,
            )
            restored = next(item for item in merged["edges"] if item["edge_key"] == edge["edge_key"])
            self.assertEqual(restored["evidence"]["deterministic"]["status"], "passed")
            self.assertEqual(restored["evidence"]["deterministic"]["evidence_ids"], [str(artifact_path)])

            changed_revision = {"commit": "other", "worktree_hash": "b" * 64}
            invalidated = build_ledger(
                {"schema_version": LEDGER_SCHEMA_VERSION, "edges": [edge]},
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
