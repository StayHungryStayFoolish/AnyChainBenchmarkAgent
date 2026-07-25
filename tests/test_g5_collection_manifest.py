from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.agent_live.g5_collection_manifest import (
    SCENARIOS,
    create_g5_attempt,
    load_active_g5_collection,
    publish_g5_collection,
)


REVISION = {"commit": "a" * 40, "worktree_hash": "b" * 64}


class G5CollectionManifestTest(unittest.TestCase):
    def _evidence(self, root: Path, scenario_id: str, sequence: int) -> Path:
        path = root / f"{sequence:02d}-{scenario_id}.json"
        path.write_text(
            json.dumps({
                "evidence_class": "real_execution",
                "scenario_id": scenario_id,
                "evidence_id": f"{sequence}" * 64,
            }),
            encoding="utf-8",
        )
        path.chmod(0o400)
        return path

    def test_complete_collection_uses_only_declared_attempt_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "g5"
            attempt_id, attempt_dir = create_g5_attempt(root / "evidence")
            evidence = [
                self._evidence(attempt_dir, scenario_id, sequence)
                for sequence, scenario_id in enumerate(SCENARIOS, start=1)
            ]
            stale = root / "evidence" / "stale.json"
            stale.write_text('{"scenario_id":"not-authoritative"}', encoding="utf-8")

            manifest = publish_g5_collection(
                g5_root=root,
                attempt_id=attempt_id,
                revision=REVISION,
                evidence_paths=evidence,
                status="complete",
            )
            payload, reason = load_active_g5_collection(
                root,
                revision=REVISION,
            )
            manifest_text = manifest.read_text(encoding="utf-8")

        self.assertEqual(reason, "")
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(
            [item["scenario_id"] for item in payload["evidence"]],
            list(SCENARIOS),
        )
        self.assertNotIn(str(stale), manifest_text)

    def test_failed_collection_accepts_only_an_ordered_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "g5"
            attempt_id, attempt_dir = create_g5_attempt(root / "evidence")
            first = self._evidence(attempt_dir, SCENARIOS[0], 1)
            failure = attempt_dir / "failure.json"
            failure.write_text('{"artifact_type":"failure"}', encoding="utf-8")
            failure.chmod(0o400)

            publish_g5_collection(
                g5_root=root,
                attempt_id=attempt_id,
                revision=REVISION,
                evidence_paths=[first],
                status="failed",
                failure_path=failure,
            )
            payload, reason = load_active_g5_collection(
                root,
                revision=REVISION,
            )

        self.assertEqual(reason, "")
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["evidence"][0]["scenario_id"], SCENARIOS[0])

    def test_collection_rejects_skipped_scenario_and_tampered_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "g5"
            attempt_id, attempt_dir = create_g5_attempt(root / "evidence")
            second = self._evidence(attempt_dir, SCENARIOS[1], 2)
            with self.assertRaisesRegex(ValueError, "ordered scenario prefix"):
                publish_g5_collection(
                    g5_root=root,
                    attempt_id=attempt_id,
                    revision=REVISION,
                    evidence_paths=[second],
                    status="failed",
                )

            first = self._evidence(attempt_dir, SCENARIOS[0], 1)
            publish_g5_collection(
                g5_root=root,
                attempt_id=attempt_id,
                revision=REVISION,
                evidence_paths=[first],
                status="failed",
            )
            pointer = root / "active-collection.json"
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            payload["manifest_sha256"] = "0" * 64
            pointer.write_text(json.dumps(payload), encoding="utf-8")
            collection, reason = load_active_g5_collection(
                root,
                revision=REVISION,
            )

        self.assertIsNone(collection)
        self.assertIn("binding is invalid", reason)

    def test_collection_rejects_evidence_from_another_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "g5"
            attempt_id, _attempt_dir = create_g5_attempt(root / "evidence")
            _other_id, other_dir = create_g5_attempt(root / "evidence")
            foreign = self._evidence(other_dir, SCENARIOS[0], 1)

            with self.assertRaisesRegex(ValueError, "missing or symlinked"):
                publish_g5_collection(
                    g5_root=root,
                    attempt_id=attempt_id,
                    revision=REVISION,
                    evidence_paths=[foreign],
                    status="failed",
                )

    def test_collection_is_revision_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "g5"
            attempt_id, attempt_dir = create_g5_attempt(root / "evidence")
            first = self._evidence(attempt_dir, SCENARIOS[0], 1)
            publish_g5_collection(
                g5_root=root,
                attempt_id=attempt_id,
                revision=REVISION,
                evidence_paths=[first],
                status="failed",
            )
            payload, reason = load_active_g5_collection(
                root,
                revision={"commit": "c" * 40, "worktree_hash": "d" * 64},
            )

        self.assertIsNone(payload)
        self.assertIn("revision mismatch", reason)


if __name__ == "__main__":
    unittest.main()
