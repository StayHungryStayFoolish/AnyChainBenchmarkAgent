"""Lane-specific proof obligations for the Harness coverage ledger."""

from __future__ import annotations

import unittest

from tests.agent_live.chaos_scheduler import build_chaos_schedule
from tests.agent_live.generate_harness_coverage_ledger import (
    EVIDENCE_CLASSES,
    build_ledger,
    evidence_applicability,
)


class CoverageLaneApplicabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.revision = {"commit": "test", "worktree_hash": "a" * 64}
        cls.ledger = build_ledger(revision=cls.revision)

    def test_every_lane_declares_required_and_a_reason(self) -> None:
        for edge in self.ledger["edges"]:
            self.assertEqual(set(edge["evidence"]), set(EVIDENCE_CLASSES))
            for lane_name, lane in edge["evidence"].items():
                self.assertIsInstance(lane["required"], bool)
                self.assertTrue(lane["applicability_reason"], (edge["edge_key"], lane_name))
                self.assertEqual(
                    (lane["required"], lane["applicability_reason"]),
                    evidence_applicability(edge, lane_name),
                )
                self.assertEqual(
                    lane["status"] == "not_applicable",
                    not lane["required"],
                )

    def test_lane_denominators_are_derived_and_distinct(self) -> None:
        summary = self.ledger["summary"]
        self.assertEqual(summary["inventory_denominator"], len(self.ledger["edges"]))
        self.assertEqual(
            summary["behavior_applicable"],
            sum(bool(edge["applicable"]) for edge in self.ledger["edges"]),
        )
        denominators = {
            lane: summary["execution_closure"][lane]["required_denominator"]
            for lane in ("deterministic", "real_cli", "dynamic_dual_ai", "real_execution")
        }
        self.assertGreater(denominators["real_cli"], denominators["dynamic_dual_ai"])
        self.assertGreater(denominators["deterministic"], denominators["real_execution"])
        self.assertGreater(denominators["dynamic_dual_ai"], denominators["real_execution"])
        self.assertGreater(denominators["real_execution"], 0)
        self.assertGreater(len(set(denominators.values())), 2)

    def test_runner_contracts_disclose_implemented_runners_and_gaps(self) -> None:
        contracts = self.ledger["runner_contracts"]
        self.assertEqual(contracts["deterministic"]["status"], "implemented")
        self.assertTrue(contracts["deterministic"]["producer"])
        self.assertEqual(contracts["dynamic_dual_ai"]["status"], "implemented")
        self.assertTrue(contracts["dynamic_dual_ai"]["producer"])
        for lane in ("real_cli", "real_execution"):
            self.assertEqual(contracts[lane]["status"], "artifact_contract_only")
            self.assertFalse(contracts[lane]["producer"])
            self.assertTrue(contracts[lane]["gap"])

    def test_matrix_separates_syntax_semantics_actions_and_side_effects(self) -> None:
        exact = next(
            edge for edge in self.ledger["edges"]
            if edge["input_class"] == "exact_option"
            and edge["evidence"]["deterministic"]["required"]
        )
        self.assertTrue(exact["evidence"]["real_cli"]["required"])
        self.assertFalse(exact["evidence"]["dynamic_dual_ai"]["required"])

        semantic = next(
            edge for edge in self.ledger["edges"]
            if edge["input_class"] == "natural_language_option"
            and edge["evidence"]["dynamic_dual_ai"]["required"]
        )
        self.assertFalse(semantic["evidence"]["deterministic"]["required"])
        self.assertTrue(semantic["evidence"]["real_cli"]["required"])

        semantic_action = next(
            edge for edge in self.ledger["edges"]
            if edge.get("action_type") == "change_group"
        )
        self.assertFalse(semantic_action["evidence"]["deterministic"]["required"])
        self.assertFalse(semantic_action["evidence"]["real_cli"]["required"])
        self.assertTrue(semantic_action["evidence"]["dynamic_dual_ai"]["required"])

        registry_action = next(
            edge for edge in self.ledger["edges"]
            if edge["edge_type"] == "action_transition"
            and edge.get("action_type") == "answer_pending"
        )
        self.assertFalse(registry_action["evidence"]["dynamic_dual_ai"]["required"])

        execution = next(
            edge for edge in self.ledger["edges"]
            if edge["evidence"]["real_execution"]["required"]
        )
        self.assertEqual(execution["edge_type"], "action_transition")
        self.assertFalse(execution["evidence"]["deterministic"]["required"])

    def test_dynamic_scheduler_rejects_exact_and_nonsemantic_action_rows(self) -> None:
        rejected = [
            next(edge for edge in self.ledger["edges"] if edge["input_class"] == "exact_option"),
            next(
                edge for edge in self.ledger["edges"]
                if edge["edge_type"] == "action_transition"
                and edge.get("action_type") == "answer_pending"
            ),
        ]
        for edge in rejected:
            with self.assertRaisesRegex(ValueError, "outside the dynamic_dual_ai lane"):
                build_chaos_schedule(
                    self.ledger,
                    revision=self.revision,
                    seed=1,
                    targets=[{
                        "edge_key": edge["edge_key"],
                        "persona": "operator",
                        "goal": "exercise semantic behavior",
                    }],
                )

        dynamic = next(
            edge for edge in self.ledger["edges"]
            if edge["evidence"]["dynamic_dual_ai"]["required"]
        )
        schedule = build_chaos_schedule(
            self.ledger,
            revision=self.revision,
            seed=2,
            targets=[{
                "edge_key": dynamic["edge_key"],
                "persona": "operator",
                "goal": "answer naturally after reading the response",
            }],
        )
        self.assertEqual(schedule.targets[0].edge_key, dynamic["edge_key"])

        control = next(
            edge for edge in self.ledger["edges"]
            if edge.get("action_type") == "change_group"
        )
        control_schedule = build_chaos_schedule(
            self.ledger,
            revision=self.revision,
            seed=3,
            targets=[{
                "edge_key": control["edge_key"],
                "persona": "operator",
                "goal": "interrupt the current group using natural language",
            }],
        )
        self.assertEqual(control_schedule.targets[0].edge_key, control["edge_key"])


if __name__ == "__main__":
    unittest.main()
