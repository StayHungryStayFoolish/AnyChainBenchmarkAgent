"""Contracts for the G4 obligation-to-Journey provider and evidence adapter."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.agent_live.chaos_scheduler import (
    build_journey_schedule,
    journey_schedule_payload,
)
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.product_chaos_journey_provider import (
    CHECKPOINT_DIFF_TYPE,
    build_product_chaos_journey_definition,
    build_product_chaos_journey_manifest,
    convert_completed_journey_to_product_evidence,
    load_frozen_product_chaos_catalog,
    main,
)
from tests.agent_live.product_chaos_obligations import (
    build_product_chaos_obligations,
)
from tests.agent_live.product_obligation_evidence import (
    admit_product_obligation_evidence,
)


REVISION = {"commit": "abc123", "worktree_hash": "frozen-tree"}


class ProductChaosJourneyProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.obligations = build_product_chaos_obligations(revision=REVISION)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.obligation = self.obligations[0]

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_catalog(self) -> Path:
        path = self.root / "catalog.json"
        path.write_text(
            json.dumps({
                "revision_binding": REVISION,
                "obligations": self.obligations,
            }),
            encoding="utf-8",
        )
        return path

    def _runtime(
        self,
        *,
        classification: str = "passed",
        provider: str = "deepseek",
        model: str = "deepseek-chat",
    ) -> Path:
        runtime = self.root / f"runtime-{classification}"
        runtime.mkdir()
        definition = build_product_chaos_journey_definition(
            self.obligation,
            revision=REVISION,
        )
        schedule = build_journey_schedule(
            revision=REVISION,
            seed=self.obligation["seed"],
            journey=definition["journey"],
        )
        schedule_payload = journey_schedule_payload(schedule)
        (runtime / "journey-schedule.json").write_text(
            json.dumps(schedule_payload),
            encoding="utf-8",
        )
        (runtime / "transcript.txt").write_text(
            "Agent> start\nUser> continue\nAgent> done\n",
            encoding="utf-8",
        )
        (runtime / "checkpoints.sqlite").write_bytes(b"sqlite-checkpoint-data")
        event = {
            "schema_version": 2,
            "event_type": "turn_committed",
            "thread_id": "g4-session",
            "session_purpose": "dynamic-dual-ai-chaos",
            "before_fingerprint": "a" * 64,
            "after_fingerprint": "b" * 64,
            "turn_index": 1,
            "active_group": "opening",
            "pending_question_id": "opening_next_action",
            "pending_contract": {"id": "opening_next_action"},
            "revision": REVISION,
            "action_queue_types": [],
            "admitted_action_types": ["answer_pending"],
            "admitted_action_targets": [{"type": "answer_pending", "group": "opening"}],
            "state_diff_hashes": {
                "target_mode": {"before": "c" * 64, "after": "d" * 64},
            },
            "after_value_hashes": {},
            "next_result": {"kind": "question", "question_id": "opening_next_action"},
        }
        (runtime / "turn-events.jsonl").write_text(
            json.dumps(event) + "\n",
            encoding="utf-8",
        )

        required = self.obligation["verifier_contract"]["required_postcondition_ids"]
        forbidden = self.obligation["verifier_contract"]["forbidden_postcondition_ids"]
        terminal_postconditions = [
            {
                "postcondition_id": item,
                "satisfied": classification == "passed",
                "details": {"checked": True},
                "verifier": {"verifier_id": item, "verifier_version": 2},
            }
            for item in required
        ]
        forbidden_outcomes = [
            {
                "outcome_id": f"forbidden-{item}",
                "satisfied": False,
                "postconditions": [{
                    "postcondition_id": item,
                    "satisfied": False,
                    "details": {"checked": True},
                    "verifier": {"verifier_id": item, "verifier_version": 2},
                }],
            }
            for item in forbidden
        ]
        turn = {
            "turn_index": 1,
            "selected_at_ns": 2_000_000_000,
            "turn_identity": {
                "transcript_hash": "e" * 64,
                "before_fingerprint": event["before_fingerprint"],
                "after_fingerprint": event["after_fingerprint"],
                "previous_response_received_at_ns": 1_000_000_000,
                "user_message_submitted_at_ns": 2_000_000_000,
                "agent_response_received_at_ns": 3_000_000_000,
            },
            "decision": {
                "user_message": "continue",
                "persona": definition["journey"]["persona"],
                "mission": definition["journey"]["mission"],
                "rationale": "response driven",
                "risk_factor_ids": [],
            },
            "decision_provenance": {
                "previous_response_hash": "2" * 64,
                "user_message_hash": "3" * 64,
                "selected_at_ns": 2_000_000_000,
                "submitted_at_ns": 2_000_000_000,
            },
            "observed_edges": [],
            "terminal_outcome": {
                "outcome_id": definition["journey"]["terminal_outcome"]["outcome_id"],
                "satisfied": classification == "passed",
                "postconditions": terminal_postconditions,
            },
            "forbidden_outcomes": forbidden_outcomes,
        }
        qualifying = classification == "passed"
        journey_payload = {
            "schema_version": 2,
            "artifact_type": "dynamic_dual_ai_journey_evidence",
            "runner_type": "dynamic_dual_ai_journey",
            "schedule_id": schedule.schedule_id,
            "schedule_hash": content_hash(schedule_payload),
            "journey_id": self.obligation["obligation_id"],
            "revision": REVISION,
            "verifier_registry": {
                "registry_id": self.obligation["verifier_contract"]["registry_id"],
            },
            "session_id": "g4-session",
            "provider": provider,
            "model": model,
            "terminal_classification": classification,
            "qualifying_evidence": qualifying,
            "terminal_outcome_id": (
                definition["journey"]["terminal_outcome"]["outcome_id"]
                if qualifying else ""
            ),
            "max_turns": definition["journey"]["max_turns"],
            "completed_turn_count": 1,
            "observed_edge_keys": [],
            "failure_reason": "" if qualifying else classification,
            "transcript_hash": "f" * 64,
            "source_transcript_hash": "1" * 64,
            "content_redacted": True,
            "initial_verification": {
                "terminal_outcome": {
                    "outcome_id": definition["journey"]["terminal_outcome"]["outcome_id"],
                    "satisfied": False,
                    "postconditions": [
                        {
                            "postcondition_id": item,
                            "satisfied": False,
                            "details": {},
                            "verifier": {"verifier_id": item},
                        }
                        for item in required
                    ],
                },
                "forbidden_outcomes": forbidden_outcomes,
            },
            "turns": [turn],
        }
        journey_evidence_id = content_hash(journey_payload)
        journey_evidence = {**journey_payload, "evidence_id": journey_evidence_id}
        journey_evidence["artifact_hash"] = content_hash(journey_evidence)
        evidence_dir = runtime / "evidence"
        evidence_dir.mkdir()
        source_evidence_path = evidence_dir / f"journey-{journey_evidence_id}.json"
        source_evidence_path.write_text(
            json.dumps(journey_evidence),
            encoding="utf-8",
        )
        result = {
            "schema_version": 1,
            "schedule_id": schedule.schedule_id,
            "journey_id": self.obligation["obligation_id"],
            "revision": REVISION,
            "execution_status": classification,
            "terminal_classification": classification,
            "terminal_outcome_id": journey_payload["terminal_outcome_id"],
            "max_turns": definition["journey"]["max_turns"],
            "completed_turn_count": 1,
            "observed_edge_keys": [],
            "failure_reason": journey_payload["failure_reason"],
            "evidence_id": journey_evidence_id,
            "evidence_path": str(source_evidence_path),
            "qualifying_evidence": qualifying,
            "verifier_registry_id": self.obligation["verifier_contract"]["registry_id"],
            "initial_verification": journey_payload["initial_verification"],
            "turns": [turn],
        }
        (runtime / "journey-result.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return runtime

    def test_all_definitions_are_batch_compatible_and_contain_no_future_turns(self) -> None:
        manifest = build_product_chaos_journey_manifest(
            self.obligations,
            revision=REVISION,
        )
        self.assertEqual(manifest["definition_count"], 135)
        self.assertFalse(manifest["generation_is_execution"])
        self.assertFalse(manifest["prewritten_future_turns"])
        for row in manifest["definitions"]:
            with self.subTest(obligation_id=row["obligation_id"]):
                definition = row["definition"]
                self.assertEqual(set(definition), {"lane", "verifier_registry", "journey"})
                self.assertEqual(definition["lane"], "journey")
                self.assertEqual(
                    definition["journey"]["journey_id"],
                    row["obligation_id"],
                )
                schedule = build_journey_schedule(
                    revision=REVISION,
                    seed=row["schedule"]["seed"],
                    journey=definition["journey"],
                )
                self.assertEqual(journey_schedule_payload(schedule), row["schedule"])
                serialized = json.dumps(definition)
                self.assertNotIn('"turns"', serialized)
                self.assertNotIn('"user_message"', serialized)

    def test_single_manifest_and_cli_are_revision_and_contract_bound(self) -> None:
        catalog = self._write_catalog()
        output = self.root / "single-manifest.json"
        self.assertEqual(main([
            "definitions",
            "--catalog", str(catalog),
            "--output", str(output),
            "--obligation-id", self.obligation["obligation_id"],
        ]), 0)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["definition_count"], 1)
        row = payload["definitions"][0]
        self.assertEqual(row["obligation_contract_hash"], self.obligation["contract_hash"])
        self.assertEqual(row["revision_binding"], REVISION)
        self.assertEqual(
            payload["manifest_hash"],
            content_hash({key: value for key, value in payload.items() if key != "manifest_hash"}),
        )
        with self.assertRaises(FileExistsError):
            main([
                "definitions",
                "--catalog", str(catalog),
                "--output", str(output),
            ])

    def test_catalog_rejects_tampering_and_stale_revision(self) -> None:
        catalog = self._write_catalog()
        rows, revision = load_frozen_product_chaos_catalog(catalog)
        self.assertEqual((len(rows), revision), (135, REVISION))
        payload = json.loads(catalog.read_text(encoding="utf-8"))
        payload["obligations"][0]["contract_hash"] = "0" * 64
        catalog.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "contract hash"):
            load_frozen_product_chaos_catalog(catalog)

    def test_passed_journey_converts_and_is_admitted_with_checkpoint_diffs(self) -> None:
        runtime = self._runtime()
        output = self.root / "product-evidence.json"
        converted = convert_completed_journey_to_product_evidence(
            self.obligation,
            revision=REVISION,
            runtime_root=runtime,
            evidence_path=output,
        )
        payload = json.loads(converted.read_text(encoding="utf-8"))
        self.assertEqual(payload["outcome"], "passed")
        self.assertEqual(payload["execution"]["transport"], "real_pty")
        self.assertEqual(payload["execution"]["provider"], "deepseek")
        diff = next(item for item in payload["artifacts"] if item["role"] == "checkpoint_diff")
        diff_payload = json.loads(Path(diff["path"]).read_text(encoding="utf-8"))
        self.assertEqual(diff_payload["artifact_type"], CHECKPOINT_DIFF_TYPE)
        self.assertEqual(diff_payload["events"][0]["state_diff_hashes"]["target_mode"]["after"], "d" * 64)
        summary = admit_product_obligation_evidence(
            obligations=[self.obligation],
            evidence_paths=[converted],
            revision=REVISION,
        )
        self.assertEqual((summary["passed"], summary["failed"]), (1, 0))
        self.assertTrue(summary["complete"])

    def test_failed_and_blocked_journeys_are_mapped_honestly(self) -> None:
        cases = (
            ("product_failed", "failed"),
            ("simulator_invalid", "failed"),
            ("infrastructure_interrupted", "failed"),
            ("externally_blocked", "externally_blocked"),
        )
        for index, (classification, expected) in enumerate(cases):
            with self.subTest(classification=classification):
                runtime = self._runtime(classification=classification)
                output = self.root / f"evidence-{index}.json"
                convert_completed_journey_to_product_evidence(
                    self.obligation,
                    revision=REVISION,
                    runtime_root=runtime,
                    evidence_path=output,
                )
                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(payload["outcome"], expected)
                statuses = {row["status"] for row in payload["verifier_results"]}
                self.assertIn(expected, statuses)
                summary = admit_product_obligation_evidence(
                    obligations=[self.obligation],
                    evidence_paths=[output],
                    revision=REVISION,
                )
                self.assertFalse(summary["complete"])

    def test_conversion_fails_closed_on_provider_or_runtime_tampering(self) -> None:
        runtime = self._runtime(
            classification="infrastructure_interrupted",
            provider="openai",
        )
        with self.assertRaisesRegex(ValueError, "provider/model"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=self.root / "wrong-provider.json",
            )
        with self.assertRaisesRegex(ValueError, "explicit DeepSeek"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=self.root / "forged-provider-boundary.json",
                provider="openai",
                model="gpt",
            )

        runtime = self._runtime()
        events = runtime / "turn-events.jsonl"
        payload = json.loads(events.read_text(encoding="utf-8"))
        payload["after_fingerprint"] = "9" * 64
        events.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprints differ"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=runtime,
                evidence_path=self.root / "tampered-runtime.json",
            )

    def test_conversion_does_not_accept_catalog_generation_as_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            convert_completed_journey_to_product_evidence(
                self.obligation,
                revision=REVISION,
                runtime_root=self.root / "not-run",
                evidence_path=self.root / "not-run.json",
            )

        forged = copy.deepcopy(self.obligation)
        forged["status"] = "passed"
        unsigned = dict(forged)
        unsigned.pop("contract_hash")
        forged["contract_hash"] = content_hash(unsigned)
        with self.assertRaisesRegex(ValueError, "cannot claim execution"):
            build_product_chaos_journey_definition(forged, revision=REVISION)


if __name__ == "__main__":
    unittest.main()
