"""Focused contracts for the finite Phase 8C G4 obligation catalog."""

from __future__ import annotations

import copy
import unittest

from tests.agent_live.formal_journey_catalog import (
    FORMAL_JOURNEY_VERIFIER_REGISTRY,
)
from tests.agent_live.product_chaos_obligations import (
    PRODUCT_CHAOS_SEED,
    build_product_chaos_obligations,
    product_chaos_obligation_report,
    validate_product_chaos_obligations,
)
from tests.agent_live.runtime_checkpoint import reviewed_scenario


REVISION = {"commit": "a" * 40, "worktree_hash": "b" * 64}


class ProductChaosObligationCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = build_product_chaos_obligations(revision=REVISION)

    def test_catalog_has_one_finite_denominator_from_both_models(self) -> None:
        report = product_chaos_obligation_report(self.rows, revision=REVISION)
        self.assertEqual(report["seed"], PRODUCT_CHAOS_SEED)
        self.assertEqual(report["required_denominator"], 135)
        self.assertEqual(report["generated_count"], 135)
        self.assertEqual(report["not_run_count"], 135)
        self.assertEqual(report["observed_pass_count"], 0)
        self.assertEqual(report["observed_fail_count"], 0)
        self.assertFalse(report["generation_is_execution"])
        self.assertEqual(
            sorted(report["by_model"].values()),
            [63, 72],
        )

    def test_obligation_ids_and_source_rows_are_stable_and_unique(self) -> None:
        rebuilt = build_product_chaos_obligations(revision=REVISION)
        self.assertEqual(self.rows, rebuilt)
        self.assertEqual(
            len({row["obligation_id"] for row in self.rows}),
            len(self.rows),
        )
        self.assertEqual(
            len({
                (row["model"]["model_id"], row["source_row_id"])
                for row in self.rows
            }),
            len(self.rows),
        )
        self.assertEqual({row["status"] for row in self.rows}, {"not_run"})

    def test_every_row_binds_an_executable_reviewed_start_and_real_verifiers(self) -> None:
        verifier_ids = set(FORMAL_JOURNEY_VERIFIER_REGISTRY.definitions)
        for row in self.rows:
            scenario = reviewed_scenario(row["start_contract"]["scenario_id"])
            self.assertTrue(scenario.seed_state)
            self.assertEqual(
                row["start_contract"]["scenario_state_fingerprint"],
                scenario.state_fingerprint,
            )
            verifier = row["verifier_contract"]
            self.assertTrue(set(verifier["required_postcondition_ids"]) <= verifier_ids)
            self.assertTrue(set(verifier["forbidden_postcondition_ids"]) <= verifier_ids)
            self.assertTrue(verifier["required_bindings"])

    def test_pending_and_session_factors_cannot_be_hidden_by_opening(self) -> None:
        for row in self.rows:
            factors = row["factors"]
            scenario_id = row["start_contract"]["scenario_id"]
            scenario = reviewed_scenario(scenario_id)
            question = dict(scenario.seed_state.get("pending_question") or {})
            pending = factors["pending_state"]
            if pending == "none" and scenario_id != "resume_quarantine":
                self.assertFalse(question, row["obligation_id"])
            elif pending == "manual":
                self.assertTrue(question.get("manual_input_allowed"), row["obligation_id"])
            elif pending == "choice":
                self.assertTrue(question.get("options"), row["obligation_id"])
            if factors.get("session_state") not in {None, "fresh"}:
                self.assertNotEqual(scenario_id, "opening", row["obligation_id"])
            if factors.get("session_state") == "quarantine":
                self.assertEqual(scenario_id, "resume_quarantine")

    def test_validation_fails_closed_on_opening_fallback_for_nonfresh_state(self) -> None:
        rows = copy.deepcopy(list(self.rows))
        index = next(
            i for i, row in enumerate(rows)
            if row["factors"].get("session_state") == "partial"
        )
        opening = reviewed_scenario("opening")
        rows[index]["start_contract"]["scenario_id"] = "opening"
        rows[index]["start_contract"]["scenario_state_fingerprint"] = (
            opening.state_fingerprint
        )
        with self.assertRaisesRegex(ValueError, "start mapping drifted"):
            validate_product_chaos_obligations(rows, revision=REVISION)

    def test_validation_fails_closed_on_execution_claim_or_unknown_verifier(self) -> None:
        executed = copy.deepcopy(list(self.rows))
        executed[0]["status"] = "passed"
        with self.assertRaisesRegex(ValueError, "claimed execution"):
            validate_product_chaos_obligations(executed, revision=REVISION)

        unknown = copy.deepcopy(list(self.rows))
        unknown[0]["verifier_contract"]["required_postcondition_ids"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "unknown product Chaos verifiers"):
            validate_product_chaos_obligations(unknown, revision=REVISION)

    def test_validation_fails_closed_on_stale_revision(self) -> None:
        with self.assertRaisesRegex(ValueError, "stale product Chaos revision"):
            validate_product_chaos_obligations(
                copy.deepcopy(list(self.rows)),
                revision={"commit": "c" * 40, "worktree_hash": "d" * 64},
            )

    def test_model_strength_seed_and_factors_are_preserved_per_row(self) -> None:
        for row in self.rows:
            self.assertEqual(row["seed"], PRODUCT_CHAOS_SEED)
            self.assertEqual(row["revision_binding"], REVISION)
            self.assertEqual(row["model"]["strength"], 2 if row["model"]["lane"] == "pairwise" else 3)
            self.assertTrue(row["source_row_id"])
            self.assertTrue(row["factors"])
            simulator = row["simulator_contract"]
            self.assertEqual(simulator["selection_mode"], "response_driven")
            self.assertTrue(simulator["prewritten_future_turns_forbidden"])
            self.assertTrue(simulator["read_complete_agent_response_before_each_turn"])


if __name__ == "__main__":
    unittest.main()
