from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from tests.agent_live.discovery_ledger import (
    aggregate_discovery_attempts,
    append_discovery_attempt,
    attempt_payload,
    build_discovery_attempt,
    load_discovery_attempts,
    validate_discovery_attempt,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
REVISION_A = {"commit": "revision-a", "worktree_hash": HASH_A}
REVISION_B = {"commit": "revision-b", "worktree_hash": HASH_B}


class DiscoveryLedgerTest(unittest.TestCase):
    def _attempt(self, **changes):
        values = {
            "batch_id": "batch-001",
            "shard_id": "shard-04",
            "stable_coverage_ids": ("coverage:resume:modify",),
            "journey_ids": (),
            "revision": REVISION_A,
            "schedule_hash": HASH_A,
            "response_hashes": (HASH_A,),
            "decision_hashes": (HASH_B,),
            "transcript_hash": HASH_C,
            "persona": "uncertain first-time operator",
            "input_classes": ("natural_language_option",),
            "sequence_ids": ("resume-modify",),
            "factor_ids": ("language:en", "session:resume"),
            "started_at": "2026-07-22T12:00:00Z",
            "finished_at": "2026-07-22T12:01:00Z",
            "classification": "product_failed",
            "diagnostic_ids": ("diagnostic-001",),
            "evidence_ids": (),
            "defect_id": "R272",
            "repair_stage": "source_open",
        }
        values.update(changes)
        return build_discovery_attempt(**values)

    def test_attempt_is_content_addressed_and_mapping_order_independent(self) -> None:
        first = self._attempt(revision={"commit": "revision-a", "worktree_hash": HASH_A})
        second = self._attempt(revision={"worktree_hash": HASH_A, "commit": "revision-a"})

        self.assertEqual(first.attempt_id, second.attempt_id)
        validate_discovery_attempt(first)
        with self.assertRaisesRegex(ValueError, "content address mismatch"):
            validate_discovery_attempt(replace(first, persona="changed after hashing"))

    def test_append_preserves_history_and_requires_explicit_supersession(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "attempts.jsonl"
            failed = self._attempt()
            append_discovery_attempt(ledger_path, failed)
            original_prefix = ledger_path.read_bytes()

            verified = self._attempt(
                batch_id="batch-002",
                shard_id="shard-09",
                revision=REVISION_B,
                schedule_hash=HASH_B,
                classification="passed",
                diagnostic_ids=(),
                evidence_ids=("evidence-009",),
                repair_stage="targeted_verified",
                supersedes_attempt_id=failed.attempt_id,
            )
            append_discovery_attempt(ledger_path, verified)

            self.assertTrue(ledger_path.read_bytes().startswith(original_prefix))
            self.assertEqual(load_discovery_attempts(ledger_path), (failed, verified))
            with self.assertRaisesRegex(ValueError, "duplicate discovery attempt"):
                append_discovery_attempt(ledger_path, verified)

    def test_supersession_must_reference_retained_same_scope_without_fork(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "attempts.jsonl"
            first = self._attempt()
            append_discovery_attempt(ledger_path, first)

            missing = self._attempt(
                classification="passed",
                repair_stage="targeted_verified",
                diagnostic_ids=(),
                evidence_ids=("evidence-missing-link",),
                supersedes_attempt_id="d" * 64,
            )
            with self.assertRaisesRegex(ValueError, "not in the retained history"):
                append_discovery_attempt(ledger_path, missing)

            wrong_scope = self._attempt(
                stable_coverage_ids=("coverage:other",),
                classification="passed",
                repair_stage="targeted_verified",
                diagnostic_ids=(),
                evidence_ids=("evidence-wrong-scope",),
                supersedes_attempt_id=first.attempt_id,
            )
            with self.assertRaisesRegex(ValueError, "same discovery subject scope"):
                append_discovery_attempt(ledger_path, wrong_scope)

            successor = self._attempt(
                batch_id="batch-002",
                classification="passed",
                repair_stage="targeted_verified",
                diagnostic_ids=(),
                evidence_ids=("evidence-successor",),
                supersedes_attempt_id=first.attempt_id,
            )
            append_discovery_attempt(ledger_path, successor)
            fork = self._attempt(
                batch_id="batch-003",
                classification="passed",
                repair_stage="chaos_verified",
                diagnostic_ids=(),
                evidence_ids=("evidence-fork",),
                supersedes_attempt_id=first.attempt_id,
            )
            with self.assertRaisesRegex(ValueError, "superseding successor"):
                append_discovery_attempt(ledger_path, fork)

    def test_invalid_append_does_not_change_existing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "attempts.jsonl"
            first = self._attempt()
            append_discovery_attempt(ledger_path, first)
            before = ledger_path.read_bytes()

            invalid = replace(first, attempt_id="0" * 64)
            with self.assertRaisesRegex(ValueError, "content address mismatch"):
                append_discovery_attempt(ledger_path, invalid)
            self.assertEqual(ledger_path.read_bytes(), before)

    def test_load_rejects_tampered_or_truncated_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "attempts.jsonl"
            attempt = self._attempt()
            payload = attempt_payload(attempt)
            payload["classification"] = "externally_blocked"
            ledger_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "content address mismatch"):
                load_discovery_attempts(ledger_path)

            ledger_path.write_text("{\"attempt_id\":", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid discovery ledger row"):
                load_discovery_attempts(ledger_path)

    def test_product_failure_and_repair_stage_require_defect_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "product_failed.*defect id"):
            self._attempt(defect_id="", repair_stage="none")
        with self.assertRaisesRegex(ValueError, "repair stage requires a defect id"):
            self._attempt(
                classification="simulator_invalid",
                defect_id="",
                repair_stage="targeted_verified",
            )

    def test_classification_specific_evidence_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "passed.*evidence ids"):
            self._attempt(
                classification="passed",
                diagnostic_ids=("diagnostic-only",),
                evidence_ids=(),
                defect_id="",
                repair_stage="none",
            )
        interrupted = self._attempt(
            classification="infrastructure_interrupted",
            response_hashes=(),
            decision_hashes=(),
            diagnostic_ids=("pty-timeout",),
            defect_id="",
            repair_stage="none",
        )
        validate_discovery_attempt(interrupted)

        with self.assertRaisesRegex(ValueError, "input class, sequence, and factor"):
            self._attempt(input_classes=())

    def test_aggregation_is_reproducible_and_does_not_erase_parallel_history(self) -> None:
        failed = self._attempt()
        verified = self._attempt(
            batch_id="batch-002",
            revision=REVISION_B,
            schedule_hash=HASH_B,
            classification="passed",
            diagnostic_ids=(),
            evidence_ids=("evidence-002",),
            repair_stage="chaos_verified",
            supersedes_attempt_id=failed.attempt_id,
        )
        independent = self._attempt(
            batch_id="batch-003",
            shard_id="shard-12",
            stable_coverage_ids=(),
            journey_ids=("journey:case2-return",),
            classification="externally_blocked",
            diagnostic_ids=("diagnostic-012",),
            defect_id="",
            repair_stage="none",
        )

        first = aggregate_discovery_attempts((failed, verified, independent))
        second = aggregate_discovery_attempts((independent, verified, failed))

        self.assertEqual(first, second)
        self.assertEqual(first["attempt_count"], 3)
        self.assertEqual(first["active_attempt_count"], 2)
        self.assertEqual(first["classification_counts"]["product_failed"], 1)
        self.assertEqual(first["active_classification_counts"]["product_failed"], 0)
        self.assertEqual(first["active_classification_counts"]["passed"], 1)
        self.assertEqual(first["active_classification_counts"]["externally_blocked"], 1)
        self.assertEqual(
            first["coverage"]["coverage:resume:modify"]["attempt_ids"],
            sorted((failed.attempt_id, verified.attempt_id)),
        )
        self.assertEqual(
            first["coverage"]["coverage:resume:modify"]["active_attempt_ids"],
            [verified.attempt_id],
        )
        coverage = first["coverage"]["coverage:resume:modify"]
        self.assertEqual(coverage["latest_attempt_id"], verified.attempt_id)
        self.assertEqual(coverage["latest_revision"], REVISION_B)
        self.assertEqual(coverage["latest_classification"], "passed")
        self.assertEqual(coverage["latest_repair_stage"], "chaos_verified")
        self.assertEqual(coverage["rerun_count"], 1)
        self.assertEqual(
            first["attempts"][failed.attempt_id],
            attempt_payload(failed),
        )

    def test_concurrent_shards_append_without_overwriting_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "attempts.jsonl"
            attempts = tuple(
                self._attempt(
                    batch_id="batch-concurrent",
                    shard_id=f"shard-{index:02d}",
                    stable_coverage_ids=(f"coverage:concurrent:{index:02d}",),
                    classification="passed",
                    diagnostic_ids=(),
                    evidence_ids=(f"evidence-{index:02d}",),
                    defect_id="",
                    repair_stage="none",
                )
                for index in range(12)
            )

            with ThreadPoolExecutor(max_workers=6) as pool:
                appended_ids = tuple(
                    pool.map(
                        lambda attempt: append_discovery_attempt(ledger_path, attempt),
                        attempts,
                    )
                )

            retained = load_discovery_attempts(ledger_path)
            self.assertEqual(len(retained), len(attempts))
            self.assertEqual(
                {attempt.attempt_id for attempt in retained},
                set(appended_ids),
            )


if __name__ == "__main__":
    unittest.main()
