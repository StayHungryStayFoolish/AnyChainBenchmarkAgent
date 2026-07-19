"""Tests for catalog-only versus executable Harness scenarios."""

from __future__ import annotations

import unittest

from agent.harness.invariants import validate_state
from tests.agent_live.harness_contract_scenarios import (
    action_transition_scenarios,
    manual_input_case,
    question_scenarios,
)
from tests.agent_live.runtime_checkpoint import reviewed_scenario_state


class HarnessContractScenarioTest(unittest.TestCase):
    def test_every_contract_scenario_has_one_reviewed_seed_authority(self) -> None:
        scenarios = question_scenarios("en")
        self.assertGreaterEqual(len(scenarios), 50)
        self.assertTrue(all(item.executable for item in scenarios))
        self.assertTrue(all(item.state_fingerprint for item in scenarios))
        self.assertEqual(len({item.scenario_id for item in scenarios}), len(scenarios))
        self.assertEqual(
            len({(item.scenario_id, item.state_fingerprint) for item in scenarios}),
            len(scenarios),
        )
        executable_ids = {item.scenario_id for item in scenarios if item.executable}
        self.assertTrue({
            "ledger_ledger_device",
            "network_bandwidth",
            "custom_schema",
            "new_chain_schema",
            "workload_mixed",
            "qps_adjust_value",
            "sync_duration",
            "failure_recovery",
        }.issubset(executable_ids))

    def test_manual_case_generator_does_not_invent_natural_language_evidence(self) -> None:
        scenario = next(
            item for item in question_scenarios("en")
            if item.scenario_id == "provider_machine"
        )
        self.assertIsNotNone(manual_input_case(scenario.question, "valid_literal"))
        self.assertIsNone(manual_input_case(scenario.question, "natural_language_answer"))
        self.assertIsNone(manual_input_case(scenario.question, "multiline_prose"))

    def test_workload_scenarios_include_reachable_upstream_prerequisites(self) -> None:
        scenarios = {item.scenario_id: item for item in question_scenarios("en")}

        for scenario_id, rpc_mode in (("workload_single", "single"), ("workload_mixed", "mixed")):
            state = dict(scenarios[scenario_id].seed_state or {})
            self.assertEqual(state.get("target_mode"), "fake-node")
            self.assertEqual(state.get("workflow_mode"), "rpc_benchmark")
            self.assertEqual(state.get("rpc_mode"), rpc_mode)
            self.assertEqual((state.get("chain_identity") or {}).get("status"), "confirmed")
            self.assertEqual((state.get("pending_question") or {}).get("id"), "workload_confirm")

    def test_semantic_coordinator_actions_have_independent_reviewed_seeds(self) -> None:
        scenarios = action_transition_scenarios("en")
        self.assertEqual(len(scenarios), 5)
        self.assertEqual(len({item.scenario_id for item in scenarios}), len(scenarios))
        self.assertEqual(len({item.action_type for item in scenarios}), len(scenarios))
        self.assertEqual(
            {item.action_type for item in scenarios},
            {
                "change_group",
                "go_back",
                "queue_workflow_goal",
                "activate_next_workflow_goal",
                "discard_next_workflow_goal",
            },
        )
        for scenario in scenarios:
            state = dict(scenario.seed_state)
            self.assertFalse(state.get("pending_question"))
            validate_state(state)  # type: ignore[arg-type]
            resolved = dict(reviewed_scenario_state(scenario.scenario_id))
            self.assertFalse(resolved.get("pending_question"))
            self.assertEqual(resolved.get("target_mode"), state.get("target_mode"))
            self.assertEqual(resolved.get("workflow_mode"), state.get("workflow_mode"))
            self.assertEqual(resolved.get("active_group"), state.get("active_group"))
            self.assertEqual(resolved.get("group_history"), state.get("group_history"))
            self.assertEqual(resolved.get("workflow_goals"), state.get("workflow_goals"))
            validate_state(resolved)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
