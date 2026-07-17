"""Tests for constrained pairwise and three-way covering arrays."""

from __future__ import annotations

import unittest

from tests.agent_live.covering_arrays import (
    Factor,
    FactorModel,
    ForbiddenCombination,
    CoveringRowObservation,
    covering_array_execution_report,
    covering_array_report,
    generate_covering_array,
)


class CoveringArrayCoverageTest(unittest.TestCase):
    def _model(self) -> FactorModel:
        return FactorModel(
            model_id="agent-chaos-factors",
            version=1,
            factors=(
                Factor("language", ("en", "zh")),
                Factor("mode", ("fake", "real")),
                Factor("input", ("choice", "natural", "multiline")),
                Factor("recovery", ("none", "jump")),
            ),
            forbidden=(
                ForbiddenCombination.from_mapping(
                    "fake-multiline-is-out-of-scope",
                    {"mode": "fake", "input": "multiline"},
                ),
            ),
        )

    def test_pairwise_generation_covers_every_feasible_tuple(self) -> None:
        result = generate_covering_array(self._model(), strength=2, seed=17)
        self.assertEqual(result.report["generated_uncovered_tuple_count"], 0)
        self.assertEqual(result.report["generation_coverage_ratio"], 1.0)
        self.assertEqual(result.report["execution_status"], "not_run")
        self.assertEqual(result.report["observed_tuple_count"], 0)
        self.assertGreater(result.report["infeasible_tuple_count"], 0)
        self.assertEqual(
            result.report["theoretical_tuple_count"],
            result.report["feasible_tuple_denominator"]
            + result.report["infeasible_tuple_count"],
        )

    def test_generation_is_reproducible_for_a_seed(self) -> None:
        first = generate_covering_array(self._model(), strength=2, seed=23)
        second = generate_covering_array(self._model(), strength=2, seed=23)
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(first.report, second.report)

    def test_three_way_strength_is_configurable_and_complete(self) -> None:
        result = generate_covering_array(self._model(), strength=3, seed=5)
        self.assertEqual(result.strength, 3)
        self.assertEqual(result.report["generated_uncovered_tuple_count"], 0)
        self.assertEqual(result.report["generation_coverage_ratio"], 1.0)

    def test_bounded_generation_reports_uncovered_feasible_tuples(self) -> None:
        result = generate_covering_array(
            self._model(), strength=3, seed=5, max_rows=1
        )
        self.assertEqual(len(result.rows), 1)
        self.assertGreater(result.report["generated_uncovered_tuple_count"], 0)
        self.assertLess(result.report["generation_coverage_ratio"], 1.0)

    def test_invalid_and_infeasible_rows_do_not_count_as_coverage(self) -> None:
        report = covering_array_report(
            self._model(),
            (
                {"language": "en", "mode": "fake", "input": "multiline", "recovery": "none"},
                {"language": "en", "mode": "unknown", "input": "choice", "recovery": "none"},
            ),
            strength=2,
        )
        self.assertEqual(report["generated_row_count"], 0)
        self.assertEqual(len(report["invalid_rows"]), 2)
        self.assertEqual(report["generated_tuple_count"], 0)

    def test_generated_rows_do_not_count_until_revision_bound_evidence_observes_them(self) -> None:
        generated = generate_covering_array(self._model(), strength=2, seed=31)
        revision = {"commit": "test", "worktree_hash": "a" * 64}
        one = CoveringRowObservation(
            row_id=generated.row_ids[0],
            revision=revision,
            outcome="passed",
            evidence_ids=("evidence-1",),
            observed_row=generated.rows[0],
        )
        partial = covering_array_execution_report(
            self._model(),
            generated,
            (one,),
            revision=revision,
            valid_evidence_ids={"evidence-1"},
        )
        self.assertEqual(partial["execution_status"], "incomplete")
        self.assertEqual(partial["observed_row_count"], 1)
        self.assertGreater(partial["observed_tuple_count"], 0)
        self.assertLess(
            partial["observed_tuple_count"],
            partial["feasible_tuple_denominator"],
        )

        invented = CoveringRowObservation(
            row_id=generated.row_ids[0],
            revision=revision,
            outcome="passed",
            evidence_ids=("invented",),
            observed_row=generated.rows[0],
        )
        rejected = covering_array_execution_report(
            self._model(),
            generated,
            (invented,),
            revision=revision,
            valid_evidence_ids={"evidence-1"},
        )
        self.assertEqual(rejected["observed_row_count"], 0)
        self.assertEqual(rejected["execution_status"], "failed")
        self.assertIn("unknown evidence", rejected["invalid_observations"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
