from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.dynamic_dual_ai_chaos import ContainerPtyBridgeTransport
from tests.agent_live.product_obligation_evidence import (
    PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION,
    admit_product_chaos_rounds,
    admit_product_obligation_evidence,
)


REVISION = {"commit": "abc123", "worktree_hash": "frozen-tree"}
REQUIRED_IMPLEMENTATION_HASH = "1" * 64
FORBIDDEN_IMPLEMENTATION_HASH = "2" * 64


def _obligation(
    name: str,
    *,
    nested_revision: bool = False,
    exact: bool = False,
) -> dict[str, Any]:
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
            "required_bindings": [{
                "postcondition_id": "required_result",
                "verifier_version": 1,
                "implementation_hash": REQUIRED_IMPLEMENTATION_HASH,
            }],
            "forbidden_bindings": [{
                "postcondition_id": "forbidden_absent",
                "verifier_version": 1,
                "implementation_hash": FORBIDDEN_IMPLEMENTATION_HASH,
            }],
        },
    }
    if exact:
        unsigned.update({
            "variant": "exact",
            "seed_contract": {"scenario_id": "opening"},
            "stimulus_contract": {"turns": ["Hi"]},
        })
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
        round_id: str = "round-1",
        execution_id: str | None = None,
        runner: str = "phase8-response-driven-runner",
        process_guard_path: Path | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        artifacts = [
            self._artifact("transcript", suffix),
            self._artifact("runtime_events", suffix),
            self._artifact("checkpoint_diff", suffix),
        ]
        if process_guard_path is not None:
            artifacts.append({
                "role": "process_guard_receipt",
                "path": str(process_guard_path.resolve()),
                "sha256": hashlib.sha256(
                    process_guard_path.read_bytes()
                ).hexdigest(),
            })
        artifact_hashes = [item["sha256"] for item in artifacts]
        statuses = {
            "passed": ("passed", "passed"),
            "failed": ("failed", "passed"),
            "externally_blocked": ("externally_blocked", "passed"),
        }[outcome]
        execution = {
            "execution_id": execution_id or f"execution-{suffix}",
            "runner": runner,
            "transport": "real_pty",
            "provider": "deepseek",
            "model": "deepseek-chat",
            "started_at": "2026-07-24T00:00:00Z",
            "finished_at": "2026-07-24T00:01:00Z",
        }
        session_id = f"session-{round_id}-{suffix}"
        request_ids = [f"request-{round_id}-{suffix}"]
        identity = {
            "obligation_id": obligation["obligation_id"],
            "obligation_contract_hash": obligation["contract_hash"],
            "revision_binding": dict(REVISION),
            "round_id": round_id,
            "session_id": session_id,
            "request_ids": request_ids,
            "execution_id": execution["execution_id"],
            "artifact_sha256s": sorted(artifact_hashes),
        }
        unsigned = {
            "schema_version": PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION,
            "evidence_id": content_hash(identity),
            "obligation_id": obligation["obligation_id"],
            "obligation_contract_hash": obligation["contract_hash"],
            "revision_binding": dict(REVISION),
            "round_id": round_id,
            "session_id": session_id,
            "request_ids": request_ids,
            "outcome": outcome,
            "execution": execution,
            "artifacts": artifacts,
            "verifier_results": [
                {
                    "verifier_id": "required_result",
                    "verifier_version": 1,
                    "implementation_hash": REQUIRED_IMPLEMENTATION_HASH,
                    "status": statuses[0],
                    "details": "checked against runtime events and state diff",
                    "evidence_sha256s": artifact_hashes,
                },
                {
                    "verifier_id": "forbidden_absent",
                    "verifier_version": 1,
                    "implementation_hash": FORBIDDEN_IMPLEMENTATION_HASH,
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

    def _genuine_process_proof(
        self,
        *,
        execution_id: str,
    ) -> dict[str, Any]:
        receipt_dir = self.root / "container-cleanup-receipts"
        repl = (
            "import sys\n"
            "print('Agent> ready\\nUser> ', end='', flush=True)\n"
            "for _line in sys.stdin:\n"
            "    print('Agent> received\\nUser> ', end='', flush=True)\n"
        )
        transport = ContainerPtyBridgeTransport(
            (
                sys.executable,
                "-m",
                "tests.agent_live.container_pty_bridge",
                "--cwd",
                str(Path.cwd()),
                "--",
                sys.executable,
                "-u",
                "-c",
                repl,
            ),
            cwd=Path.cwd(),
            execution_id=execution_id,
            cleanup_receipt_dir=receipt_dir,
        )
        transport.start(env=os.environ)
        self.assertEqual(
            transport.read_complete_agent_response(timeout_seconds=5),
            "Agent> ready",
        )
        transport.close()
        return dict(transport.validated_execution_proof())

    def test_two_round_admission_requires_complete_unique_identity_sets(self) -> None:
        evidence_by_round: dict[str, list[Path]] = {}
        for round_id in ("round-1", "round-2"):
            evidence_by_round[round_id] = [
                self._evidence(
                    obligation,
                    suffix=f"{round_id}-{index}",
                    round_id=round_id,
                )[0]
                for index, obligation in enumerate(self.obligations)
            ]
        summary = admit_product_chaos_rounds(
            obligations=self.obligations,
            evidence_by_round=evidence_by_round,
            revision=REVISION,
        )
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["total_denominator"], 4)
        self.assertEqual(
            summary["new_s1_s2_root_classes"],
            {"round-1": [], "round-2": []},
        )

    def test_two_round_admission_rejects_wrong_round_and_cross_round_reuse(self) -> None:
        first = self._evidence(
            self.obligations[0],
            suffix="first",
            round_id="round-1",
        )[0]
        second = self._evidence(
            self.obligations[0],
            suffix="second",
            round_id="round-2",
        )[0]
        with self.assertRaisesRegex(ValueError, "round mismatch"):
            admit_product_chaos_rounds(
                obligations=[self.obligations[0]],
                evidence_by_round={
                    "round-1": [first],
                    "round-2": [first],
                },
                revision=REVISION,
            )

        first_payload = json.loads(first.read_text(encoding="utf-8"))
        second_payload = json.loads(second.read_text(encoding="utf-8"))
        second_payload["session_id"] = first_payload["session_id"]
        identity = {
            "obligation_id": second_payload["obligation_id"],
            "obligation_contract_hash": second_payload["obligation_contract_hash"],
            "revision_binding": second_payload["revision_binding"],
            "round_id": second_payload["round_id"],
            "session_id": second_payload["session_id"],
            "request_ids": second_payload["request_ids"],
            "execution_id": second_payload["execution"]["execution_id"],
            "artifact_sha256s": sorted(
                item["sha256"] for item in second_payload["artifacts"]
            ),
        }
        second_payload["evidence_id"] = content_hash(identity)
        self._rewrite(second, second_payload)
        with self.assertRaisesRegex(ValueError, "cross-round identity reuse"):
            admit_product_chaos_rounds(
                obligations=[self.obligations[0]],
                evidence_by_round={
                    "round-1": [first],
                    "round-2": [second],
                },
                revision=REVISION,
            )

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

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux procfs")
    def test_g3_exact_admits_genuine_process_cleanup_proof(self) -> None:
        execution_id = "g3-exact-genuine-cleanup"
        proof = self._genuine_process_proof(execution_id=execution_id)
        obligation = _obligation(
            "g3-exact-genuine",
            nested_revision=True,
            exact=True,
        )
        evidence_path, _payload = self._evidence(
            obligation,
            suffix="g3-exact-genuine",
            round_id="g3-retained-regression",
            execution_id=execution_id,
            runner="retained-regression-real-cli-v1",
            process_guard_path=Path(proof["path"]),
        )

        summary = admit_product_obligation_evidence(
            obligations=[obligation],
            evidence_paths=[evidence_path],
            revision=REVISION,
        )

        self.assertTrue(summary["complete"])
        admitted = summary["admitted_evidence"][obligation["obligation_id"]]
        self.assertEqual(
            admitted["evidence_sha256"],
            hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        )

    def test_g3_exact_rejects_missing_process_cleanup_proof(self) -> None:
        obligation = _obligation(
            "g3-exact-missing-cleanup",
            nested_revision=True,
            exact=True,
        )
        evidence_path, _payload = self._evidence(
            obligation,
            suffix="g3-exact-missing-cleanup",
            round_id="g3-retained-regression",
            execution_id="g3-exact-missing-cleanup",
            runner="retained-regression-real-cli-v1",
        )

        with self.assertRaisesRegex(
            ValueError,
            "required G3 exact process proof is missing",
        ):
            admit_product_obligation_evidence(
                obligations=[obligation],
                evidence_paths=[evidence_path],
                revision=REVISION,
            )

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux procfs")
    def test_g3_exact_rejects_process_runtime_identity_and_role_tampering(
        self,
    ) -> None:
        execution_id = "g3-exact-tamper"
        proof = self._genuine_process_proof(execution_id=execution_id)
        receipt_path = Path(proof["path"])
        obligation = _obligation(
            "g3-exact-tamper",
            nested_revision=True,
            exact=True,
        )
        evidence_path, _payload = self._evidence(
            obligation,
            suffix="g3-exact-identity-tamper",
            round_id="g3-retained-regression",
            execution_id="different-execution",
            runner="retained-regression-real-cli-v1",
            process_guard_path=receipt_path,
        )
        with self.assertRaisesRegex(ValueError, "execution id mismatch"):
            admit_product_obligation_evidence(
                obligations=[obligation],
                evidence_paths=[evidence_path],
                revision=REVISION,
            )

        foreign_root = self.root / "foreign-cleanup-receipts"
        foreign_root.mkdir()
        foreign_receipt = foreign_root / receipt_path.name
        foreign_receipt.write_bytes(receipt_path.read_bytes())
        evidence_path, _payload = self._evidence(
            obligation,
            suffix="g3-exact-runtime-tamper",
            round_id="g3-retained-regression",
            execution_id=execution_id,
            runner="retained-regression-real-cli-v1",
            process_guard_path=foreign_receipt,
        )
        with self.assertRaisesRegex(ValueError, "outside its runtime"):
            admit_product_obligation_evidence(
                obligations=[obligation],
                evidence_paths=[evidence_path],
                revision=REVISION,
            )

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for row in receipt["registered_processes"]:
            row["roles"] = [
                role
                for role in row["roles"]
                if role != "agent_process_group_leader"
            ]
        unsigned_receipt = {
            key: value for key, value in receipt.items()
            if key != "receipt_id"
        }
        receipt["receipt_id"] = hashlib.sha256(
            json.dumps(
                unsigned_receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        tampered_receipt = receipt_path.with_name(
            f"container-cleanup-receipt-{receipt['receipt_id']}.json"
        )
        tampered_receipt.write_text(
            json.dumps(receipt, sort_keys=True),
            encoding="utf-8",
        )
        evidence_path, _payload = self._evidence(
            obligation,
            suffix="g3-exact-role-tamper",
            round_id="g3-retained-regression",
            execution_id=execution_id,
            runner="retained-regression-real-cli-v1",
            process_guard_path=tampered_receipt,
        )
        with self.assertRaisesRegex(ValueError, "missing required registered roles"):
            admit_product_obligation_evidence(
                obligations=[obligation],
                evidence_paths=[evidence_path],
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

    def test_rehashed_pass_with_stale_verifier_implementation_fails_closed(self) -> None:
        path, payload = self._evidence(
            self.obligations[0],
            suffix="stale-verifier",
        )
        payload["verifier_results"][0]["implementation_hash"] = "f" * 64
        self._rewrite(path, payload)

        with self.assertRaisesRegex(ValueError, "implementation binding is stale"):
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
