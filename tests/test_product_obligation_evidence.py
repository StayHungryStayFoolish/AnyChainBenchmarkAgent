from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.product_obligation_evidence import (
    admit_product_obligation_evidence,
)


REVISION = {"commit": "abc123", "worktree_hash": "frozen-tree"}


def _obligation(name: str, *, nested_revision: bool = False) -> dict[str, Any]:
    unsigned = {
        "obligation_id": name,
        "revision_binding": (
            {"revision": dict(REVISION), "source_fixture": {"sanitized": True}}
            if nested_revision
            else dict(REVISION)
        ),
        "execution_status" if nested_revision else "status": "not_run",
        "verifier_contract": {
            "required_postcondition_ids": ["required_result"],
            "forbidden_postcondition_ids": ["forbidden_absent"],
        },
    }
    return {**unsigned, "contract_hash": content_hash(unsigned)}


class ProductObligationEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.obligations = [
            _obligation("g3-one", nested_revision=True),
            _obligation("g4-one"),
        ]

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _artifact(self, role: str, suffix: str) -> dict[str, str]:
        path = self.root / f"{role}-{suffix}.json"
        path.write_text(f'{{"role":"{role}","suffix":"{suffix}"}}', encoding="utf-8")
        return {
            "role": role,
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def _evidence(
        self,
        obligation: dict[str, Any],
        *,
        outcome: str = "passed",
        suffix: str = "one",
    ) -> tuple[Path, dict[str, Any]]:
        artifacts = [
            self._artifact("transcript", suffix),
            self._artifact("runtime_events", suffix),
            self._artifact("checkpoint_diff", suffix),
        ]
        artifact_hashes = [item["sha256"] for item in artifacts]
        statuses = {
            "passed": ("passed", "passed"),
            "failed": ("failed", "passed"),
            "externally_blocked": ("externally_blocked", "passed"),
        }[outcome]
        execution = {
            "execution_id": f"execution-{suffix}",
            "runner": "phase8-response-driven-runner",
            "transport": "real_pty",
            "provider": "deepseek",
            "model": "deepseek-chat",
            "started_at": "2026-07-24T00:00:00Z",
            "finished_at": "2026-07-24T00:01:00Z",
        }
        identity = {
            "obligation_id": obligation["obligation_id"],
            "obligation_contract_hash": obligation["contract_hash"],
            "revision_binding": dict(REVISION),
            "execution_id": execution["execution_id"],
            "artifact_sha256s": sorted(artifact_hashes),
        }
        unsigned = {
            "schema_version": 1,
            "evidence_id": content_hash(identity),
            "obligation_id": obligation["obligation_id"],
            "obligation_contract_hash": obligation["contract_hash"],
            "revision_binding": dict(REVISION),
            "outcome": outcome,
            "execution": execution,
            "artifacts": artifacts,
            "verifier_results": [
                {
                    "verifier_id": "required_result",
                    "status": statuses[0],
                    "details": "checked against runtime events and state diff",
                    "evidence_sha256s": artifact_hashes,
                },
                {
                    "verifier_id": "forbidden_absent",
                    "status": statuses[1],
                    "details": "forbidden behavior was independently checked",
                    "evidence_sha256s": artifact_hashes,
                },
            ],
        }
        payload = {**unsigned, "evidence_hash": content_hash(unsigned)}
        path = self.root / f"evidence-{suffix}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path, payload

    def _rewrite(self, path: Path, payload: dict[str, Any], *, rehash: bool = True) -> None:
        if rehash:
            unsigned = dict(payload)
            unsigned.pop("evidence_hash", None)
            payload["evidence_hash"] = content_hash(unsigned)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_empty_evidence_is_not_run_and_catalog_generation_is_not_execution(self) -> None:
        summary = admit_product_obligation_evidence(
            obligations=self.obligations,
            evidence_paths=[],
            revision=REVISION,
        )
        self.assertEqual(summary["denominator"], 2)
        self.assertEqual(summary["not_run"], 2)
        self.assertEqual(summary["passed"], 0)
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["generation_is_execution"])

    def test_all_obligations_must_pass_before_summary_is_complete(self) -> None:
        first, _ = self._evidence(self.obligations[0], suffix="g3")
        partial = admit_product_obligation_evidence(
            obligations=self.obligations,
            evidence_paths=[first],
            revision=REVISION,
        )
        self.assertEqual((partial["passed"], partial["not_run"]), (1, 1))
        self.assertFalse(partial["complete"])

        second, _ = self._evidence(self.obligations[1], suffix="g4")
        complete = admit_product_obligation_evidence(
            obligations=self.obligations,
            evidence_paths=[first, second],
            revision=REVISION,
        )
        self.assertEqual((complete["passed"], complete["not_run"]), (2, 0))
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["status"], "complete")

    def test_failed_and_external_outcomes_are_counted_but_never_complete(self) -> None:
        failed, _ = self._evidence(self.obligations[0], outcome="failed", suffix="fail")
        blocked, _ = self._evidence(
            self.obligations[1],
            outcome="externally_blocked",
            suffix="blocked",
        )
        summary = admit_product_obligation_evidence(
            obligations=self.obligations,
            evidence_paths=[failed, blocked],
            revision=REVISION,
        )
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["external"], 1)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["status"], "failed")

    def test_missing_duplicate_and_unknown_evidence_fail_closed(self) -> None:
        path, payload = self._evidence(self.obligations[0])
        with self.assertRaisesRegex(ValueError, "duplicate evidence path"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[path, path],
                revision=REVISION,
            )
        with self.assertRaisesRegex(ValueError, "does not exist"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[self.root / "missing.json"],
                revision=REVISION,
            )
        payload["obligation_id"] = "unknown"
        self._rewrite(path, payload)
        with self.assertRaisesRegex(ValueError, "unknown obligation"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[path],
                revision=REVISION,
            )

    def test_duplicate_or_conflicting_obligation_evidence_fails_atomically(self) -> None:
        first, _ = self._evidence(self.obligations[0], suffix="first")
        second, _ = self._evidence(self.obligations[0], outcome="failed", suffix="second")
        with self.assertRaisesRegex(ValueError, "duplicate or conflicting"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[first, second],
                revision=REVISION,
            )

    def test_stale_revision_and_contract_hash_fail_closed(self) -> None:
        path, payload = self._evidence(self.obligations[0])
        payload["revision_binding"]["commit"] = "stale"
        self._rewrite(path, payload)
        with self.assertRaisesRegex(ValueError, "stale evidence revision"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[path],
                revision=REVISION,
            )

        path, payload = self._evidence(self.obligations[0], suffix="contract")
        payload["obligation_contract_hash"] = "0" * 64
        self._rewrite(path, payload)
        with self.assertRaisesRegex(ValueError, "contract hash mismatch"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[path],
                revision=REVISION,
            )

    def test_forged_obligation_and_evidence_hashes_fail_closed(self) -> None:
        forged = copy.deepcopy(self.obligations)
        forged[0]["verifier_contract"]["required_postcondition_ids"].append("invented")
        with self.assertRaisesRegex(ValueError, "forged frozen obligation"):
            admit_product_obligation_evidence(
                obligations=forged,
                evidence_paths=[],
                revision=REVISION,
            )

        path, payload = self._evidence(self.obligations[0])
        payload["outcome"] = "failed"
        self._rewrite(path, payload, rehash=False)
        with self.assertRaisesRegex(ValueError, "forged evidence document"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[path],
                revision=REVISION,
            )

        path, payload = self._evidence(self.obligations[0], suffix="identity")
        payload["evidence_id"] = "f" * 64
        self._rewrite(path, payload)
        with self.assertRaisesRegex(ValueError, "unstable evidence identity"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[path],
                revision=REVISION,
            )

    def test_passed_declaration_without_real_artifacts_fails_closed(self) -> None:
        path, payload = self._evidence(self.obligations[0])
        payload["artifacts"] = []
        self._rewrite(path, payload)
        with self.assertRaisesRegex(ValueError, "runtime artifacts are missing"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[path],
                revision=REVISION,
            )

    def test_g4_pass_cannot_bypass_journey_and_process_provenance(self) -> None:
        obligation = _obligation("g4-strict")
        unsigned = {
            key: value
            for key, value in obligation.items()
            if key != "contract_hash"
        }
        unsigned.update({
            "model": {"model_id": "product-chaos"},
            "factors": {"subject_group": "chain_identity"},
            "start_contract": {"scenario_id": "opening"},
        })
        obligation = {**unsigned, "contract_hash": content_hash(unsigned)}
        path, _payload = self._evidence(obligation, suffix="g4-strict")
        with self.assertRaisesRegex(ValueError, "G4 runtime provenance"):
            admit_product_obligation_evidence(
                obligations=[obligation],
                evidence_paths=[path],
                revision=REVISION,
            )

    def test_missing_required_artifact_or_hash_mismatch_fails_closed(self) -> None:
        path, payload = self._evidence(self.obligations[0])
        payload["artifacts"].pop()
        self._rewrite(path, payload)
        with self.assertRaisesRegex(ValueError, "required runtime artifacts"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[path],
                revision=REVISION,
            )

        path, payload = self._evidence(self.obligations[0], suffix="tampered-file")
        artifact_path = self.root / payload["artifacts"][0]["path"]
        artifact_path.write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[path],
                revision=REVISION,
            )

    def test_non_pty_or_incomplete_execution_identity_fails_closed(self) -> None:
        path, payload = self._evidence(self.obligations[0])
        payload["execution"]["transport"] = "scripted"
        self._rewrite(path, payload)
        with self.assertRaisesRegex(ValueError, "real PTY"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[path],
                revision=REVISION,
            )

    def test_verifier_set_status_and_artifact_references_fail_closed(self) -> None:
        path, payload = self._evidence(self.obligations[0])
        payload["verifier_results"].pop()
        self._rewrite(path, payload)
        with self.assertRaisesRegex(ValueError, "result set is incomplete"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[path],
                revision=REVISION,
            )

        path, payload = self._evidence(self.obligations[0], suffix="status")
        payload["verifier_results"][0]["status"] = "failed"
        self._rewrite(path, payload)
        with self.assertRaisesRegex(ValueError, "non-passing verifier"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[path],
                revision=REVISION,
            )

        path, payload = self._evidence(self.obligations[0], suffix="reference")
        payload["verifier_results"][0]["evidence_sha256s"] = ["f" * 64]
        self._rewrite(path, payload)
        with self.assertRaisesRegex(ValueError, "invalid artifact references"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[path],
                revision=REVISION,
            )

    def test_catalog_status_and_revision_are_frozen(self) -> None:
        executed = copy.deepcopy(self.obligations)
        executed[0]["execution_status"] = "passed"
        unsigned = dict(executed[0])
        unsigned.pop("contract_hash")
        executed[0]["contract_hash"] = content_hash(unsigned)
        with self.assertRaisesRegex(ValueError, "catalog generation claimed execution"):
            admit_product_obligation_evidence(
                obligations=executed,
                evidence_paths=[],
                revision=REVISION,
            )

        with self.assertRaisesRegex(ValueError, "stale frozen obligation revision"):
            admit_product_obligation_evidence(
                obligations=self.obligations,
                evidence_paths=[],
                revision={"commit": "different", "worktree_hash": "frozen-tree"},
            )


if __name__ == "__main__":
    unittest.main()
