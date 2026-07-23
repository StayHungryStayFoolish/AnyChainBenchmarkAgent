from __future__ import annotations

import unittest
from unittest.mock import patch


class AgentContractProjectionTest(unittest.TestCase):
    def test_action_registry_projects_by_owner_group_and_lifetime(self) -> None:
        from agent.harness.action_registry import project_action_specs

        orientation = project_action_specs(owners=frozenset({"orientation"}))
        self.assertTrue(orientation)
        self.assertTrue(all(spec.owner == "orientation" for spec in orientation))

        qps = project_action_specs(groups=frozenset({"qps_profile"}))
        self.assertTrue(qps)
        self.assertTrue(all(spec.target_group == "qps_profile" for spec in qps))

        local = project_action_specs(lifetimes=frozenset({"turn_local"}))
        self.assertTrue(local)
        self.assertTrue(all(spec.lifetime == "turn_local" for spec in local))

    def test_action_schema_projection_does_not_leak_other_owners(self) -> None:
        from agent.harness.context import action_schema

        schema = action_schema(owners=frozenset({"performance"}))
        self.assertTrue(schema)
        self.assertEqual({row["owner"] for row in schema}, {"performance"})
        self.assertLess(len(schema), len(action_schema()))

    def test_current_turn_rejects_retired_arguments_envelope(self) -> None:
        from agent.harness.action_registry import validate_action_contract

        with self.assertRaisesRegex(ValueError, "arguments.v1 is retired"):
            validate_action_contract({
                "type": "change_target_mode",
                "arguments": {"target_mode": "fake-node"},
            })

    def test_current_turn_does_not_compile_retired_custom_rpc_action(self) -> None:
        from agent.harness.coordinator import _normalized_action_queue

        actions = _normalized_action_queue({
            "actions": [{
                "type": "start_custom_rpc",
                "rpc_endpoint": "http://example.invalid",
                "source_evidence": "http://example.invalid",
            }]
        })
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["type"], "unknown")
        self.assertIn("undeclared action type", actions[0]["reason"])

    def test_checkpoint_migrator_can_still_compile_retired_custom_rpc(self) -> None:
        from agent.harness.action_registry import compile_legacy_custom_rpc_action

        actions = compile_legacy_custom_rpc_action({
            "type": "start_custom_rpc",
            "rpc_endpoint": "http://example.invalid",
            "source_evidence": "http://example.invalid",
        })
        self.assertEqual([action["type"] for action in actions], ["rpc_catalog_command"])

    def test_every_cataloged_option_uses_a_current_action(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE
        from tests.agent_live.generate_harness_coverage_ledger import build_ledger

        ledger = build_ledger(None)
        option_edges = [
            edge
            for edge in ledger["edges"]
            if edge["edge_type"] == "question_option"
        ]
        self.assertTrue(option_edges)
        for edge in option_edges:
            self.assertIn(edge["action_type"], ACTION_BY_TYPE, edge["edge_key"])

    def test_acceptance_controller_is_the_only_phase_authority(self) -> None:
        from tests.agent_live.run_product_acceptance import build_report

        with patch(
            "tests.agent_live.run_product_acceptance._g0_source",
            return_value={"gate": "G0", "status": "passed"},
        ):
            phase_one = build_report(2)
            future = build_report(3)
        self.assertEqual(
            phase_one["authority"],
            "tests/agent_live/run_product_acceptance.py",
        )
        self.assertEqual(phase_one["status"], "passed")
        self.assertEqual(future["status"], "incomplete")
        self.assertEqual(future["gates"]["G1"]["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
