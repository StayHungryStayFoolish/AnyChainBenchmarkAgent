"""Tests for catalog-only versus executable Harness scenarios."""

from __future__ import annotations

import unittest

from tests.agent_live.harness_contract_scenarios import manual_input_case, question_scenarios


class HarnessContractScenarioTest(unittest.TestCase):
    def test_only_reviewed_scenarios_have_seed_state(self) -> None:
        scenarios = question_scenarios("en")
        self.assertGreaterEqual(sum(item.executable for item in scenarios), 50)
        self.assertTrue(any(not item.executable for item in scenarios))
        self.assertTrue(all(item.state_fingerprint for item in scenarios))
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


if __name__ == "__main__":
    unittest.main()
