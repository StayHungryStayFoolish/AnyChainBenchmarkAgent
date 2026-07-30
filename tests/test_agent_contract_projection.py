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
        from agent.harness.admission import _normalized_action_queue

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

        passed_phase = {"status": "passed", "checks": []}
        with (
            patch(
                "tests.agent_live.run_product_acceptance._g0_source",
                return_value={"gate": "G0", "status": "passed"},
            ),
            patch(
                "tests.agent_live.run_product_acceptance._phase1_source",
                return_value=passed_phase,
            ),
            patch(
                "tests.agent_live.run_product_acceptance._phase2_source",
                return_value=passed_phase,
            ),
            patch(
                "tests.agent_live.run_product_acceptance._phase3_source",
                return_value=passed_phase,
            ),
            patch(
                "tests.agent_live.run_product_acceptance._phase4_source",
                return_value=passed_phase,
            ),
            patch(
                "tests.agent_live.run_product_acceptance._phase5_source",
                return_value=passed_phase,
            ),
            patch(
                "tests.agent_live.run_product_acceptance._phase6_source",
                return_value=passed_phase,
            ),
            patch(
                "tests.agent_live.run_product_acceptance._phase7_source",
                return_value=passed_phase,
            ),
            patch(
                "tests.agent_live.run_product_acceptance._phase8_source",
                return_value={
                    "status": "incomplete",
                    "gates": {
                        gate: {"status": "incomplete"}
                        for gate in ("G3", "G4", "G5")
                    } | {"G6": {"status": "not_run"}},
                },
            ),
        ):
            phase_two = build_report(2)
            phase_five = build_report(5)
            phase_six = build_report(6)
            phase_seven = build_report(7)
            future = build_report(8)
        self.assertEqual(
            phase_two["authority"],
            "tests/agent_live/run_product_acceptance.py",
        )
        self.assertEqual(phase_two["status"], "passed")
        self.assertEqual(phase_five["status"], "passed")
        self.assertEqual(phase_five["gates"]["G1"]["status"], "passed")
        self.assertEqual(phase_six["status"], "passed")
        self.assertEqual(phase_six["gates"]["G2"]["status"], "passed")
        self.assertEqual(phase_six["gates"]["G3"]["status"], "not_run")
        self.assertEqual(phase_six["gates"]["G3"]["owning_phase"], 8)
        self.assertEqual(phase_seven["status"], "passed")
        self.assertEqual(phase_seven["gates"]["G2"]["status"], "passed")
        self.assertEqual(phase_seven["gates"]["G3"]["status"], "not_run")
        self.assertEqual(future["status"], "incomplete")
        self.assertEqual(future["gates"]["G2"]["status"], "passed")
        self.assertEqual(future["implemented_through_phase"], 8)
        self.assertEqual(future["gates"]["G3"]["status"], "incomplete")

    def test_phase_six_completion_preserves_zero_open_edges(self) -> None:
        from tests.agent_live.run_product_acceptance import _phase6_complete

        arguments = {
            "ledger": {
                "summary": {"groups": 20},
                "uncataloged_questions": [],
            },
            "closure": {
                "status": "complete",
                "required_denominator": 221,
                "open_required": 0,
            },
            "observed_domain_owners": {"one", "two"},
            "expected_domain_owners": {"one", "two"},
            "regressions": {"status": "passed"},
            "shell_gates": {"status": "passed"},
            "checks": [{"passed": True}],
        }
        self.assertTrue(_phase6_complete(**arguments))
        arguments["closure"]["open_required"] = 1
        self.assertFalse(_phase6_complete(**arguments))

    def test_product_acceptance_uses_offline_isolation_runner(self) -> None:
        from tests.agent_live.run_product_acceptance import FULL_PYTHON_SUITE_COMMAND

        self.assertEqual(
            FULL_PYTHON_SUITE_COMMAND[1:],
            ("tests/run_offline_python_suite.py",),
        )

    def test_phase_one_runs_real_contract_checks(self) -> None:
        from tests.agent_live.run_product_acceptance import _phase1_source

        with patch(
            "tests.agent_live.run_product_acceptance._checked_phase",
            return_value={"phase": 1, "status": "passed", "checks": ["real"]},
        ) as checked:
            result = _phase1_source()

        self.assertEqual(result["checks"], ["real"])
        checked.assert_called_once_with(
            1,
            "tests.test_agent_contract_projection",
            "tests.test_agent_question_prompts",
        )

    def test_revision_identity_distinguishes_dirty_diagnostic_reports(self) -> None:
        from tests.agent_live.run_product_acceptance import (
            EMPTY_WORKTREE_HASH,
            _revision_id,
        )

        self.assertEqual(
            _revision_id({"commit": "abc", "worktree_hash": EMPTY_WORKTREE_HASH}),
            "abc",
        )
        self.assertEqual(
            _revision_id({"commit": "abc", "worktree_hash": "f" * 64}),
            "abc-ffffffffffffffff",
        )


if __name__ == "__main__":
    unittest.main()
