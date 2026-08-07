"""Product factor-model coverage tests; these do not claim CLI execution."""

from __future__ import annotations

import unittest

from tests.agent_live.product_chaos_factors import (
    STATE_CONTROL_FACTOR_NAMES,
    build_product_factor_model,
    build_state_control_factor_model,
    generate_product_covering_array,
)


class ProductChaosFactorCoverageTest(unittest.TestCase):
    def test_group_mode_feasibility_is_derived_from_product_registry(self) -> None:
        model = build_product_factor_model()
        forbidden = {
            tuple(sorted(constraint.values))
            for constraint in model.forbidden
        }

        self.assertIn(
            tuple(sorted({
                "workflow_mode": "fake",
                "subject_group": "chain_auxiliary_endpoints",
            }.items())),
            forbidden,
        )
        self.assertIn(
            tuple(sorted({
                "workflow_mode": "sync",
                "subject_group": "qps_profile",
            }.items())),
            forbidden,
        )
        self.assertNotIn(
            tuple(sorted({
                "workflow_mode": "fake",
                "chain_case": "case1",
                "subject_group": "endpoint_process",
            }.items())),
            forbidden,
        )

    def test_model_contains_every_required_product_dimension(self) -> None:
        model = build_product_factor_model()
        domains = {factor.name: set(factor.values) for factor in model.factors}
        self.assertEqual(domains["workflow_mode"], {"fake", "real", "sync"})
        self.assertEqual(domains["language"], {"en", "zh"})
        self.assertEqual(
            domains["session_state"], {"fresh", "partial", "complete", "quarantine"}
        )
        self.assertEqual(domains["pending_state"], {"none", "manual", "choice"})
        self.assertEqual(domains["group_state"], {"partial", "completed", "invalidated"})
        self.assertEqual(domains["interruption_depth"], {"0", "1", "2+"})
        self.assertTrue(
            {"exact", "natural", "multiline", "structured", "contradictory"}
            <= domains["input_shape"]
        )
        self.assertEqual(domains["chain_case"], {"known", "case1", "case2", "case3"})
        self.assertTrue(
            {"default_single", "default_mixed", "custom_single", "custom_mixed"}
            <= domains["workload"]
        )
        self.assertEqual(
            domains["evidence_shape"], {"none", "request", "response", "split", "docs"}
        )
        self.assertTrue({"none", "back", "jump", "correct", "retry"} <= domains["recovery"])

    def test_pairwise_product_model_has_explicit_complete_denominator(self) -> None:
        result = generate_product_covering_array(
            build_product_factor_model(), strength=2, seed=20260716
        )
        report = result.report
        self.assertGreater(report["feasible_tuple_denominator"], 0)
        self.assertGreater(report["infeasible_tuple_count"], 0)
        self.assertEqual(
            report["theoretical_tuple_count"],
            report["feasible_tuple_denominator"] + report["infeasible_tuple_count"],
        )
        self.assertEqual(report["generated_tuple_count"], report["feasible_tuple_denominator"])
        self.assertEqual(report["generated_uncovered_tuple_count"], 0)
        self.assertEqual(report["generated_uncovered_tuples"], [])
        self.assertEqual(report["execution_status"], "not_run")
        self.assertEqual(report["observed_tuple_count"], 0)
        self.assertEqual(report["seed"], 20260716)
        self.assertEqual(set(report["factors"]), {
            factor.name for factor in build_product_factor_model().factors
        })
        self.assertGreater(len(report["constraints"]), 0)
        self.assertLess(report["generated_row_count"], report["feasible_tuple_denominator"])

    def test_state_control_projection_has_complete_three_way_coverage(self) -> None:
        model = build_state_control_factor_model()
        self.assertEqual({factor.name for factor in model.factors}, set(STATE_CONTROL_FACTOR_NAMES))
        result = generate_product_covering_array(model, strength=3, seed=20260716)
        report = result.report
        self.assertGreater(report["feasible_tuple_denominator"], 0)
        self.assertGreaterEqual(report["infeasible_tuple_count"], 0)
        self.assertEqual(
            report["theoretical_tuple_count"],
            report["feasible_tuple_denominator"] + report["infeasible_tuple_count"],
        )
        self.assertEqual(report["generated_uncovered_tuple_count"], 0)
        self.assertEqual(report["generated_tuple_count"], report["feasible_tuple_denominator"])
        self.assertEqual(report["execution_status"], "not_run")
        self.assertEqual(report["observed_tuple_count"], 0)

    def test_constraints_exclude_impossible_product_rows(self) -> None:
        model = build_product_factor_model()
        pairwise = generate_product_covering_array(model, strength=2, seed=7)
        rows = pairwise.rows
        self.assertFalse(any(
            row["workflow_mode"] == "sync" and row["workload"] != "not_applicable"
            for row in rows
        ))
        self.assertFalse(any(
            row["chain_case"] == "case3" and row["workload"] != "not_applicable"
            for row in rows
        ))
        self.assertFalse(any(
            row["session_state"] == "quarantine" and row["recovery"] != "reset"
            for row in rows
        ))
        self.assertFalse(any(
            row["recovery"] == "back" and row["interruption_depth"] == "0"
            for row in rows
        ))

    def test_constraints_preserve_valid_sync_and_evidence_crosses(self) -> None:
        rows = generate_product_covering_array(
            build_product_factor_model(), strength=2, seed=29
        ).rows
        self.assertTrue(any(
            row["workflow_mode"] == "sync"
            and row["chain_case"] == "known"
            and row["workload"] == "not_applicable"
            for row in rows
        ))
        self.assertTrue(any(
            row["workflow_mode"] == "sync"
            and row["chain_case"] == "case2"
            and row["workload"] == "not_applicable"
            for row in rows
        ))
        self.assertTrue(any(
            row["chain_case"] == "known"
            and row["workload"] == "default_single"
            and row["evidence_shape"] == "none"
            for row in rows
        ))
        self.assertTrue(any(
            row["chain_case"] == "case2"
            and row["workload"] == "custom_single"
            and row["evidence_shape"] == "split"
            for row in rows
        ))

    def test_generation_is_reproducible(self) -> None:
        model = build_state_control_factor_model()
        first = generate_product_covering_array(model, strength=3, seed=41)
        second = generate_product_covering_array(model, strength=3, seed=41)
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(first.report, second.report)


if __name__ == "__main__":
    unittest.main()
