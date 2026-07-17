"""Atomic ingestion tests for revision-bound Harness execution evidence."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.agent_live.coverage_evidence import (
    COMPILED_GRAPH_RUNNER,
    build_evidence_artifact,
    write_evidence_artifact,
)
from tests.agent_live.execute_harness_contract_ledger import execute_ledger
from tests.agent_live.generate_harness_coverage_ledger import (
    build_ledger,
    ingest_evidence_artifacts,
)


class HarnessEvidenceIngestionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.revision = {"commit": "ingestion-test", "worktree_hash": "a" * 64}
        cls._temporary_directory = TemporaryDirectory()
        cls.artifact_dir = Path(cls._temporary_directory.name)
        executed = execute_ledger(
            artifact_dir=cls.artifact_dir,
            revision=cls.revision,
        )
        cls.references = [
            Path(edge["evidence"]["deterministic"]["evidence_ids"][0])
            for edge in executed["edges"]
            if edge["evidence"]["deterministic"]["status"] == "passed"
        ]
        if len(cls.references) < 2:
            raise AssertionError("ingestion tests require two deterministic pass artifacts")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def _clean_ledger(self) -> dict[str, object]:
        return build_ledger(revision=self.revision)

    def test_multiple_valid_artifacts_are_ingested_and_duplicates_are_idempotent(self) -> None:
        ledger = self._clean_ledger()
        imported = ingest_evidence_artifacts(
            ledger,
            [self.references[0], self.references[1], self.references[0]],
        )
        passed = [
            edge for edge in imported["edges"]
            if edge["evidence"]["deterministic"]["status"] == "passed"
        ]
        self.assertEqual(len(passed), 2)
        self.assertTrue(all(
            len(edge["evidence"]["deterministic"]["evidence_ids"]) == 1
            for edge in passed
        ))
        self.assertEqual(
            imported["summary"]["execution_closure"]["deterministic"]["observed_pass"],
            2,
        )
        self.assertEqual(
            ledger["summary"]["execution_closure"]["deterministic"]["observed_pass"],
            0,
            "ingestion must not mutate the caller's ledger",
        )

    def test_stale_revision_unknown_edge_wrong_lane_and_tampering_fail_atomically(self) -> None:
        base = self._clean_ledger()
        valid_reference = self.references[0]
        stale = build_ledger(
            revision={"commit": "other", "worktree_hash": "b" * 64},
        )
        with self.assertRaisesRegex(ValueError, "repository revision mismatch"):
            ingest_evidence_artifacts(stale, [valid_reference])

        artifact = json.loads(valid_reference.read_text(encoding="utf-8"))
        cases = {
            "unknown edge": {**artifact, "edge_key": "missing::edge"},
            "not required": {**artifact, "evidence_class": "dynamic_dual_ai"},
            "hash mismatch": deepcopy(artifact),
        }
        cases["hash mismatch"]["turn_evidence"]["input"] = "tampered"
        for name, invalid in cases.items():
            with self.subTest(name=name), TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "invalid.json"
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    ingest_evidence_artifacts(base, [valid_reference, path])
                self.assertEqual(
                    base["summary"]["execution_closure"]["deterministic"]["observed_pass"],
                    0,
                )

    def test_conflicting_valid_outcomes_fail_without_partial_mutation(self) -> None:
        ledger = self._clean_ledger()
        passed_path = self.references[0]
        passed = json.loads(passed_path.read_text(encoding="utf-8"))
        edge = next(item for item in ledger["edges"] if item["edge_key"] == passed["edge_key"])
        turn = passed["turn_evidence"]
        failed = build_evidence_artifact(
            edge=edge,
            evidence_class="deterministic",
            scenario_id="conflicting-observation",
            runner_type=COMPILED_GRAPH_RUNNER,
            revision=self.revision,
            input_value=turn["input"],
            seed_state=turn["seed"],
            before_state=turn["before"],
            after_state=turn["after"],
            events=passed["event_trace"],
            exit_status=1,
            outcome="failed",
            error="independent verifier rejected the postcondition",
            admitted=bool(turn["admitted_action"]["admitted"]),
        )
        failed_path = write_evidence_artifact(failed, self.artifact_dir)

        with self.assertRaisesRegex(ValueError, "conflicting evidence outcomes"):
            ingest_evidence_artifacts(ledger, [passed_path, failed_path])
        self.assertEqual(
            ledger["summary"]["execution_closure"]["deterministic"]["observed_pass"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
