"""Tests for versioned critical-sequence coverage denominators."""

from __future__ import annotations

import unittest

from tests.agent_live.critical_sequences import (
    CriticalSequenceObservation,
    ObservedCoverageTurn,
    critical_sequence_denominator_report,
    critical_sequence_registry_from_dict,
    load_critical_sequence_registry,
    match_critical_sequence,
)


class CriticalSequenceCoverageTest(unittest.TestCase):
    revision = {"commit": "test", "worktree_hash": "a" * 64}

    def test_default_registry_is_versioned_and_has_explicit_denominator(self) -> None:
        registry = load_critical_sequence_registry()
        report = critical_sequence_denominator_report(registry, ())
        self.assertEqual(registry.version, 2)
        self.assertEqual(report["sequence_denominator"], len(registry.sequences))
        self.assertEqual(report["passed_sequences"], 0)
        self.assertGreater(report["step_denominator"], report["sequence_denominator"])

    def test_steps_must_be_observed_in_order(self) -> None:
        registry = load_critical_sequence_registry()
        sequence = registry.sequences[0]
        ordered = tuple(step.coverage_ids for step in sequence.steps)
        reversed_turns = tuple(reversed(ordered))
        self.assertEqual(match_critical_sequence(sequence, ordered), tuple(
            step.step_id for step in sequence.steps
        ))
        self.assertNotEqual(match_critical_sequence(sequence, reversed_turns), tuple(
            step.step_id for step in sequence.steps
        ))

    def test_report_exposes_partial_uncovered_steps_and_evidence(self) -> None:
        registry = load_critical_sequence_registry()
        sequence = registry.sequences[0]
        first_two = tuple(step.coverage_ids for step in sequence.steps[:2])
        observation = CriticalSequenceObservation(
            sequence_id=sequence.sequence_id,
            registry_version=registry.version,
            round_id="round-1",
            evidence_ids=("evidence-1", "evidence-2"),
        )
        evidence_index = {
            f"evidence-{index}": ObservedCoverageTurn(
                evidence_id=f"evidence-{index}",
                round_id="round-1",
                turn_index=index,
                revision=self.revision,
                coverage_ids=coverage_ids,
            )
            for index, coverage_ids in enumerate(first_two, start=1)
        }
        report = critical_sequence_denominator_report(
            registry,
            (observation,),
            evidence_index=evidence_index,
            revision=self.revision,
        )
        row = report["sequences"][0]
        self.assertEqual(row["matched_steps"], [step.step_id for step in sequence.steps[:2]])
        self.assertEqual(row["uncovered_steps"], [step.step_id for step in sequence.steps[2:]])
        self.assertEqual(row["evidence_ids"], ["evidence-1", "evidence-2"])
        self.assertFalse(row["passed"])

    def test_wrong_registry_version_does_not_count(self) -> None:
        registry = load_critical_sequence_registry()
        sequence = registry.sequences[0]
        observation = CriticalSequenceObservation(
            sequence_id=sequence.sequence_id,
            registry_version=999,
            round_id="stale-round",
            evidence_ids=("stale",),
        )
        report = critical_sequence_denominator_report(registry, (observation,))
        self.assertEqual(report["passed_sequences"], 0)

    def test_invented_or_wrong_revision_evidence_ids_cannot_complete_sequence(self) -> None:
        registry = load_critical_sequence_registry()
        sequence = registry.sequences[0]
        invented = CriticalSequenceObservation(
            sequence_id=sequence.sequence_id,
            registry_version=registry.version,
            round_id="round-invented",
            evidence_ids=("invented-evidence",),
        )
        report = critical_sequence_denominator_report(
            registry,
            (invented,),
            evidence_index={},
            revision=self.revision,
        )
        self.assertEqual(report["passed_sequences"], 0)
        self.assertIn("unknown evidence id", report["invalid_observations"][0]["reason"])

        real_id = "stale-evidence"
        stale = CriticalSequenceObservation(
            sequence_id=sequence.sequence_id,
            registry_version=registry.version,
            round_id="round-stale",
            evidence_ids=(real_id,),
        )
        report = critical_sequence_denominator_report(
            registry,
            (stale,),
            evidence_index={
                real_id: ObservedCoverageTurn(
                    evidence_id=real_id,
                    round_id="round-stale",
                    turn_index=1,
                    revision={"commit": "old", "worktree_hash": "b" * 64},
                    coverage_ids=sequence.steps[0].coverage_ids,
                )
            },
            revision=self.revision,
        )
        self.assertEqual(report["passed_sequences"], 0)
        self.assertIn("revision mismatch", report["invalid_observations"][0]["reason"])

    def test_duplicate_sequence_and_step_ids_are_rejected(self) -> None:
        base = {
            "registry_id": "registry",
            "version": 2,
            "sequences": [{
                "sequence_id": "duplicate",
                "steps": [
                    {"step_id": "same", "coverage_ids": ["edge:a"]},
                    {"step_id": "same", "coverage_ids": ["edge:b"]},
                ],
            }],
        }
        with self.assertRaisesRegex(ValueError, "invalid or duplicate step"):
            critical_sequence_registry_from_dict(base)

    def test_registry_covers_required_product_risk_sequences(self) -> None:
        registry = load_critical_sequence_registry()
        ids = {sequence.sequence_id for sequence in registry.sequences}
        required = {
            "repeated-back-across-partial-groups",
            "compound-mode-chain-endpoint-turn",
            "case1-custom-rpc-no-params",
            "case1-custom-rpc-multi-params-weights",
            "case2-custom-rpc-no-params",
            "case2-custom-rpc-multi-params-weights",
            "case3-handoff-return-case2",
            "partial-group-three-level-jumps",
            "execution-failure-correct-retry",
            "session-partial-resume-modify-reset",
            "sync-observe-never-enters-rpc-load",
            "correction-after-rejection",
        }
        self.assertEqual(required - ids, set())

    def test_custom_rpc_sequences_require_schema_response_and_weight_evidence(self) -> None:
        registry = load_critical_sequence_registry()
        required_ids = {
            "case1-custom-rpc-no-params",
            "case1-custom-rpc-multi-params-weights",
            "case2-custom-rpc-no-params",
            "case2-custom-rpc-multi-params-weights",
        }
        sequences = {
            sequence.sequence_id: sequence for sequence in registry.sequences
            if sequence.sequence_id in required_ids
        }
        self.assertEqual(len(sequences), 4)
        for sequence in sequences.values():
            coverage = {
                coverage_id
                for step in sequence.steps
                for coverage_id in step.coverage_ids
            }
            self.assertIn("probe:rpc-schema", coverage)
            self.assertIn("weight-total:100", coverage)
            self.assertTrue(
                "rpc-params:empty-array" in coverage or "rpc-params:multiple" in coverage
            )
        for sequence_id in (
            "case1-custom-rpc-multi-params-weights",
            "case2-custom-rpc-multi-params-weights",
        ):
            coverage = {
                coverage_id
                for step in sequences[sequence_id].steps
                for coverage_id in step.coverage_ids
            }
            self.assertIn("rpc-param:index-name-type-semantics", coverage)
            self.assertIn("rpc-response:shape-confirmed", coverage)

    def test_sync_sequence_declares_negative_execution_invariant(self) -> None:
        registry = load_critical_sequence_registry()
        sequence = next(
            item for item in registry.sequences
            if item.sequence_id == "sync-observe-never-enters-rpc-load"
        )
        coverage = {
            coverage_id for step in sequence.steps for coverage_id in step.coverage_ids
        }
        self.assertIn("invariant:sync-no-rpc-workload-qps-fixture-vegeta", coverage)
        self.assertIn("artifact:sync-report", coverage)

    def test_correction_after_rejection_is_a_multi_turn_sequence(self) -> None:
        registry = load_critical_sequence_registry()
        sequence = next(
            item for item in registry.sequences
            if item.sequence_id == "correction-after-rejection"
        )
        self.assertEqual(
            [step.step_id for step in sequence.steps],
            ["invalid-input", "question-preserved", "corrected-input", "workflow-continues"],
        )
        self.assertIn("multi-turn", sequence.tags)


if __name__ == "__main__":
    unittest.main()
