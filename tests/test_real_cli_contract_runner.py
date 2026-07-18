"""Focused contracts for the deterministic real-CLI evidence producer."""

from __future__ import annotations

import unittest

from tests.agent_live.execute_real_cli_contract_ledger import (
    _require_target_contract,
    eligible_edges,
)
from tests.agent_live.generate_harness_coverage_ledger import (
    build_ledger,
    contract_variant_hash,
)
from tests.agent_live.harness_contract_scenarios import question_scenarios


class RealCliContractRunnerTest(unittest.TestCase):
    def test_eligible_rows_require_real_cli_without_dynamic_and_have_a_seed_artifact(self) -> None:
        ledger = {
            "edges": [
                {
                    "edge_key": "eligible",
                    "deterministic_test_id": "/tmp/evidence.json",
                    "evidence": {
                        "real_cli": {"required": True},
                        "dynamic_dual_ai": {"required": False},
                    },
                },
                {
                    "edge_key": "dynamic",
                    "deterministic_test_id": "/tmp/evidence.json",
                    "evidence": {
                        "real_cli": {"required": True},
                        "dynamic_dual_ai": {"required": True},
                    },
                },
                {
                    "edge_key": "unseeded",
                    "deterministic_test_id": "",
                    "evidence": {
                        "real_cli": {"required": True},
                        "dynamic_dual_ai": {"required": False},
                    },
                },
            ]
        }
        self.assertEqual([edge["edge_key"] for edge in eligible_edges(ledger)], ["eligible"])

    def test_target_contract_must_match_question_and_contract_hash(self) -> None:
        scenario = next(item for item in question_scenarios("en") if item.scenario_id == "opening")
        edge = {
            "question_id": scenario.question["id"],
            "contract_hash": contract_variant_hash(scenario.question),
        }
        _require_target_contract(edge, scenario.question)
        with self.assertRaisesRegex(RuntimeError, "question"):
            _require_target_contract({**edge, "question_id": "other"}, scenario.question)
        with self.assertRaisesRegex(RuntimeError, "contract hash"):
            _require_target_contract({**edge, "contract_hash": "wrong"}, scenario.question)

    def test_authoritative_ledger_exposes_partial_runner_and_excludes_whitespace(self) -> None:
        ledger = build_ledger(revision={"commit": "test", "worktree_hash": "a" * 64})
        self.assertEqual(ledger["runner_contracts"]["real_cli"]["status"], "partial")
        self.assertTrue(ledger["runner_contracts"]["real_cli"]["producer"])
        self.assertTrue(all(
            not edge["evidence"]["real_cli"]["required"]
            for edge in ledger["edges"]
            if edge["input_class"] == "empty_whitespace"
        ))


if __name__ == "__main__":
    unittest.main()
