from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.agent_live.completed_journey_batch import (
    G4_ARTIFACT_TYPE,
    _validated_completed_batch,
    convert_completed_journey_batch,
    load_completed_journey_batch_evidence,
)
from tests.agent_live.batch_orchestrator import BATCH_RESULT_SCHEMA_VERSION
from tests.agent_live.coverage_evidence import content_hash


REVISION = {"commit": "a" * 40, "worktree_hash": "b" * 64}


class CompletedJourneyBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        for path in sorted(self.root.rglob("*"), reverse=True):
            path.chmod(0o700 if path.is_dir() else 0o600)
        self.temp.cleanup()

    def test_completed_batch_rejects_a_nonpassing_shard(self) -> None:
        runtime = self.root / "runtime"
        runtime.mkdir()
        shard = SimpleNamespace(
            shard_id="shard-1",
            lane="journey",
            obligation_id="obligation-1",
            target_hash="c" * 64,
            runtime_root=str(runtime),
            execution_id="e" * 64,
        )
        manifest = SimpleNamespace(
            revision=REVISION,
            shards=(shard,),
            batch_id="batch-1",
            manifest_id="manifest-1",
        )
        result = {
            "batch_id": "batch-1",
            "manifest_id": "manifest-1",
            "revision": REVISION,
            "execution_status": "discovery_complete",
            "release_status": "not_evaluated",
            "scheduled": 1,
            "started": 1,
            "completed": 1,
            "discovery_attempt_ids": ["attempt-1"],
            "classification_counts": {"passed": 1},
            "batch_survivor_proof": {"cleaned": True},
            "shards": [{
                "shard_id": "shard-1",
                "classification": "passed",
                "target_hash": "c" * 64,
                "started_at_ns": 1,
                "finished_at_ns": 2,
                "response_hashes": ["d" * 64],
                "decision_hashes": ["e" * 64],
                "evidence_hashes": ["f" * 64],
            }],
            "schema_version": BATCH_RESULT_SCHEMA_VERSION,
        }
        result["index_id"] = content_hash(result)
        result_path = self.root / "result.json"
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        manifest_path.chmod(0o400)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        result_path.chmod(0o400)
        with (
            patch(
                "tests.agent_live.completed_journey_batch.load_frozen_manifest",
                return_value=manifest,
            ),
            patch(
                "tests.agent_live.completed_journey_batch.validate_completed_shard_result",
            ),
        ):
            _, roots, execution_ids, _ = _validated_completed_batch(
                manifest_path=manifest_path,
                result_index_path=result_path,
                expected_obligation_ids={"obligation-1"},
                revision=REVISION,
            )
            self.assertEqual(roots["obligation-1"], runtime)
            self.assertEqual(execution_ids["obligation-1"], shard.execution_id)

            result_path.chmod(0o600)
            result["shards"][0]["classification"] = "product_failed"
            unsigned = {key: value for key, value in result.items() if key != "index_id"}
            result["index_id"] = content_hash(unsigned)
            result_path.write_text(json.dumps(result), encoding="utf-8")
            result_path.chmod(0o400)
            with self.assertRaisesRegex(ValueError, "not qualifying"):
                _validated_completed_batch(
                    manifest_path=manifest_path,
                    result_index_path=result_path,
                    expected_obligation_ids={"obligation-1"},
                    revision=REVISION,
                )

    def test_conversion_publishes_a_declared_read_only_index(self) -> None:
        runtime = self.root / "runtime"
        runtime.mkdir()
        manifest_source = self.root / "batch-manifest.json"
        result_source = self.root / "batch-result.json"
        manifest_source.write_text("{}", encoding="utf-8")
        result_source.write_text("{}", encoding="utf-8")
        manifest_source.chmod(0o400)
        result_source.chmod(0o400)
        manifest = SimpleNamespace(
            manifest_id="manifest-1",
            batch_id="batch-1",
        )
        obligation = {
            "obligation_id": "obligation-1",
            "contract_hash": "c" * 64,
        }

        def convert_one(_obligation, _runtime, evidence_path, checkpoint_diff):
            runtime_artifact = _runtime / "runtime-artifact.txt"
            runtime_artifact.write_text("runtime\n", encoding="utf-8")
            evidence_path.write_text(json.dumps({
                "obligation_id": "obligation-1",
                "obligation_contract_hash": "c" * 64,
                "evidence_id": "d" * 64,
                "execution": {"execution_id": "e" * 64},
                "artifacts": [{
                    "role": "runtime",
                    "path": str(runtime_artifact),
                    "sha256": __import__("hashlib").sha256(
                        runtime_artifact.read_bytes()
                    ).hexdigest(),
                }],
            }), encoding="utf-8")
            self.assertIsNotNone(checkpoint_diff)
            checkpoint_diff.write_text('{"diff": true}\n', encoding="utf-8")
            return evidence_path

        with (
            patch(
                "tests.agent_live.completed_journey_batch._validated_completed_batch",
                return_value=(
                    manifest,
                    {"obligation-1": runtime},
                    {"obligation-1": "e" * 64},
                    {"index_id": "index-1"},
                ),
            ),
            patch(
                "tests.agent_live.completed_journey_batch.admit_product_obligation_evidence",
                return_value={
                    "complete": True,
                    "passed": 1,
                    "failed": 0,
                    "external": 0,
                    "not_run": 0,
                },
            ),
        ):
            index = convert_completed_journey_batch(
                manifest_path=manifest_source,
                result_index_path=result_source,
                output_dir=self.root / "published",
                obligations=(obligation,),
                revision=REVISION,
                artifact_type=G4_ARTIFACT_TYPE,
                round_id="round-1",
                convert_one=convert_one,
            )
            paths = load_completed_journey_batch_evidence(
                index,
                obligations=(obligation,),
                revision=REVISION,
                artifact_type=G4_ARTIFACT_TYPE,
                round_id="round-1",
            )
        self.assertEqual(len(paths), 1)
        self.assertFalse(index.stat().st_mode & 0o222)
        self.assertFalse(index.parent.stat().st_mode & 0o222)

    def test_declared_index_rejects_evidence_tampering(self) -> None:
        evidence_root = self.root / "evidence-set"
        evidence_dir = evidence_root / "evidence"
        evidence_dir.mkdir(parents=True)
        runtime_artifact = self.root / "runtime-artifact.txt"
        runtime_artifact.write_text("runtime\n", encoding="utf-8")
        evidence = evidence_dir / "evidence.json"
        evidence.write_text(json.dumps({
            "obligation_id": "obligation-1",
            "obligation_contract_hash": "c" * 64,
            "evidence_id": "d" * 64,
            "execution": {"execution_id": "e" * 64},
            "artifacts": [{
                "role": "runtime",
                "path": str(runtime_artifact),
                "sha256": __import__("hashlib").sha256(
                    runtime_artifact.read_bytes()
                ).hexdigest(),
            }],
        }), encoding="utf-8")
        obligation = {
            "obligation_id": "obligation-1",
            "contract_hash": "c" * 64,
        }
        batch_manifest = self.root / "batch-manifest.json"
        batch_result = self.root / "batch-result.json"
        batch_manifest.write_text("{}", encoding="utf-8")
        batch_result.write_text("{}", encoding="utf-8")
        batch_manifest.chmod(0o400)
        batch_result.chmod(0o400)
        unsigned = {
            "schema_version": 1,
            "artifact_type": G4_ARTIFACT_TYPE,
            "revision_binding": REVISION,
            "round_id": "round-1",
            "batch_manifest": {
                "path": str(batch_manifest),
                "sha256": __import__("hashlib").sha256(
                    batch_manifest.read_bytes()
                ).hexdigest(),
                "manifest_id": "manifest-1",
            },
            "batch_result": {
                "path": str(batch_result),
                "sha256": __import__("hashlib").sha256(
                    batch_result.read_bytes()
                ).hexdigest(),
                "index_id": "index-1",
            },
            "obligation_count": 1,
            "evidence": [{
                "obligation_id": "obligation-1",
                "obligation_contract_hash": "c" * 64,
                "evidence_id": "d" * 64,
                "execution_id": "e" * 64,
                "path": str(evidence.resolve()),
                "sha256": __import__("hashlib").sha256(evidence.read_bytes()).hexdigest(),
            }],
        }
        index = evidence_root / "manifest.json"
        index.write_text(
            json.dumps({**unsigned, "index_hash": content_hash(unsigned)}),
            encoding="utf-8",
        )
        evidence.chmod(0o400)
        evidence_dir.chmod(0o500)
        index.chmod(0o400)
        evidence_root.chmod(0o500)
        manifest = SimpleNamespace(manifest_id="manifest-1")
        with (
            patch(
                "tests.agent_live.completed_journey_batch._validated_completed_batch",
                return_value=(
                    manifest,
                    {"obligation-1": self.root},
                    {"obligation-1": "e" * 64},
                    {"index_id": "index-1"},
                ),
            ),
            patch(
                "tests.agent_live.completed_journey_batch.admit_product_obligation_evidence",
                return_value={"complete": True},
            ),
        ):
            self.assertEqual(
                len(load_completed_journey_batch_evidence(
                    index,
                    obligations=(obligation,),
                    revision=REVISION,
                    artifact_type=G4_ARTIFACT_TYPE,
                    round_id="round-1",
                )),
                1,
            )
            with patch(
                "tests.agent_live.completed_journey_batch._validated_completed_batch",
                return_value=(
                    manifest,
                    {"obligation-1": self.root},
                    {"obligation-1": "f" * 64},
                    {"index_id": "index-1"},
                ),
            ), self.assertRaisesRegex(ValueError, "row binding"):
                load_completed_journey_batch_evidence(
                    index,
                    obligations=(obligation,),
                    revision=REVISION,
                    artifact_type=G4_ARTIFACT_TYPE,
                    round_id="round-1",
                )
            evidence.chmod(0o600)
            evidence.write_text('{"tampered": true}\n', encoding="utf-8")
            evidence.chmod(0o400)
            with self.assertRaisesRegex(ValueError, "row binding"):
                load_completed_journey_batch_evidence(
                    index,
                    obligations=(obligation,),
                    revision=REVISION,
                    artifact_type=G4_ARTIFACT_TYPE,
                    round_id="round-1",
                )

    def test_conversion_rejects_execution_identity_substitution_atomically(self) -> None:
        runtime = self.root / "runtime"
        runtime.mkdir()
        manifest_source = self.root / "batch-manifest.json"
        result_source = self.root / "batch-result.json"
        manifest_source.write_text("{}", encoding="utf-8")
        result_source.write_text("{}", encoding="utf-8")
        manifest_source.chmod(0o400)
        result_source.chmod(0o400)
        manifest = SimpleNamespace(
            manifest_id="manifest-1",
            batch_id="batch-1",
        )
        obligation = {
            "obligation_id": "obligation-1",
            "contract_hash": "c" * 64,
        }

        def convert_one(_obligation, _runtime, evidence_path, _checkpoint_diff):
            evidence_path.write_text(json.dumps({
                "obligation_id": "obligation-1",
                "obligation_contract_hash": "c" * 64,
                "evidence_id": "d" * 64,
                "execution": {"execution_id": "e" * 64},
            }), encoding="utf-8")
            return evidence_path

        destination = self.root / "published"
        with (
            patch(
                "tests.agent_live.completed_journey_batch._validated_completed_batch",
                return_value=(
                    manifest,
                    {"obligation-1": runtime},
                    {"obligation-1": "f" * 64},
                    {"index_id": "index-1"},
                ),
            ),
            self.assertRaisesRegex(ValueError, "identity"),
        ):
            convert_completed_journey_batch(
                manifest_path=manifest_source,
                result_index_path=result_source,
                output_dir=destination,
                obligations=(obligation,),
                revision=REVISION,
                artifact_type=G4_ARTIFACT_TYPE,
                round_id="round-1",
                convert_one=convert_one,
            )
        self.assertFalse(destination.exists())

    def test_loader_rejects_symlinked_index(self) -> None:
        real = self.root / "manifest.json"
        real.write_text("{}", encoding="utf-8")
        real.chmod(0o400)
        link = self.root / "linked-manifest.json"
        link.symlink_to(real)
        with self.assertRaisesRegex(ValueError, "symlink"):
            load_completed_journey_batch_evidence(
                link,
                obligations=({"obligation_id": "obligation-1"},),
                revision=REVISION,
                artifact_type=G4_ARTIFACT_TYPE,
                round_id="round-1",
            )


if __name__ == "__main__":
    unittest.main()
