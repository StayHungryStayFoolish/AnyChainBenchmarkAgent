from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from tests.agent_live.batch_orchestrator import (
    BATCH_MANIFEST_SCHEMA_VERSION,
    BATCH_RESULT_SCHEMA_VERSION,
)
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.retained_regression_evidence_set import (
    EXACT_RUNNER,
    OPEN_RUNNER,
    load_retained_regression_evidence_set,
    publish_retained_regression_evidence_set,
)
from tests.agent_live.retained_regression_obligations import (
    build_retained_regression_obligations,
)
from tests.agent_live.retained_regression_runner import (
    build_retained_regression_runner_provider,
)


REVISION = {
    "commit": "a" * 40,
    "worktree_hash": "b" * 64,
}
REPO_ROOT = Path(__file__).resolve().parents[1]


class RetainedRegressionEvidenceSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.obligations = build_retained_regression_obligations(
            repo_root=REPO_ROOT,
            revision=REVISION
        )
        cls.provider = build_retained_regression_runner_provider(
            obligations=cls.obligations,
            revision=REVISION,
        )

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.evidence_paths = self._write_evidence()
        self.exact_index = self._write_exact_index()
        self.open_manifest, self.open_result = self._write_open_batch()

    def tearDown(self) -> None:
        for path in sorted(self.root.rglob("*"), reverse=True):
            try:
                path.chmod(0o700 if path.is_dir() else 0o600)
            except FileNotFoundError:
                pass
        self.temp.cleanup()

    def test_publish_and_load_complete_immutable_collection(self) -> None:
        destination = self.root / "published"
        with self._admission_passes():
            manifest_path = self._publish(destination)
            manifest = load_retained_regression_evidence_set(
                manifest_path,
                provider=self.provider,
                obligations=self.obligations,
                revision=REVISION,
            )
        self.assertEqual(manifest["obligation_count"], 60)
        self.assertEqual(manifest["exact_count"], 15)
        self.assertEqual(manifest["open_count"], 45)
        self.assertEqual(len(manifest["evidence"]), 60)
        self.assertFalse(manifest_path.stat().st_mode & 0o222)
        self.assertFalse(destination.stat().st_mode & 0o222)

    def test_existing_destination_and_concurrent_publish_cannot_overwrite(self) -> None:
        destination = self.root / "published"
        with self._admission_passes():
            self._publish(destination)
            with self.assertRaises(FileExistsError):
                self._publish(destination)

        concurrent_destination = self.root / "concurrent"
        barrier = threading.Barrier(2)

        def publish() -> str:
            barrier.wait()
            try:
                with self._admission_passes():
                    self._publish(concurrent_destination)
                return "published"
            except FileExistsError:
                return "rejected"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _index: publish(), range(2)))
        self.assertEqual(sorted(outcomes), ["published", "rejected"])
        self.assertTrue((concurrent_destination / "manifest.json").is_file())
        staging = tuple(self.root.glob(".concurrent.staging-*"))
        self.assertEqual(len(staging), 1)
        self.assertFalse(staging[0].stat().st_mode & 0o222)

    def test_missing_extra_and_wrong_variant_cardinality_fail_closed(self) -> None:
        with self._admission_passes():
            with self.assertRaisesRegex(ValueError, "exactly 60"):
                self._publish(
                    self.root / "missing",
                    evidence_paths=self.evidence_paths[:-1],
                )
            with self.assertRaisesRegex(ValueError, "exactly 60"):
                self._publish(
                    self.root / "extra",
                    evidence_paths=[
                        *self.evidence_paths,
                        self.evidence_paths[0],
                    ],
                )

        exact = self._read(self.exact_index)
        exact["scheduled"] = 14
        exact["completed"] = 14
        exact["evidence"] = exact["evidence"][:-1]
        self._rehash(exact, "index_hash")
        self._write(self.exact_index, exact, mode=0o400)
        with self._admission_passes():
            with self.assertRaisesRegex(ValueError, "exact execution index"):
                self._publish(self.root / "wrong-exact")

        self._write_exact_index()
        manifest = self._read(self.open_manifest)
        manifest["shards"] = manifest["shards"][:-1]
        manifest["shard_count"] = 44
        self._rehash_batch_manifest(manifest)
        self._write(self.open_manifest, manifest, mode=0o400)
        with self._admission_passes():
            with self.assertRaisesRegex(ValueError, "open batch manifest"):
                self._publish(self.root / "wrong-open")

    def test_variant_runner_substitution_fails_closed(self) -> None:
        exact_obligation = next(
            row for row in self.obligations if row["variant"] == "exact"
        )
        path = self._evidence_by_id(exact_obligation["obligation_id"])
        payload = self._read(path)
        payload["execution"]["runner"] = OPEN_RUNNER
        self._write(path, payload)
        self._refresh_exact_index_hash(path)
        with self._admission_passes():
            with self.assertRaisesRegex(ValueError, "variant/runner"):
                self._publish(self.root / "wrong-runner")

    def test_open_batch_must_match_provider_frozen_schedule(self) -> None:
        manifest = self._read(self.open_manifest)
        manifest["shards"][0]["schedule_id"] = "c" * 64
        manifest["expected_obligation_set_hash"] = content_hash([
            {
                "obligation_id": row["obligation_id"],
                "schedule_id": row["schedule_id"],
                "seed": row["seed"],
                "subject_group": row["subject_group"],
            }
            for row in sorted(
                manifest["shards"],
                key=lambda row: (row["obligation_id"], row["schedule_id"]),
            )
        ])
        self._rehash_batch_manifest(manifest)
        self._write(self.open_manifest, manifest, mode=0o400)
        result = self._read(self.open_result)
        result["batch_id"] = manifest["batch_id"]
        result["manifest_id"] = manifest["manifest_id"]
        self._rehash(result, "index_id")
        self._write(self.open_result, result, mode=0o400)
        with self._admission_passes():
            with self.assertRaisesRegex(ValueError, "shard contract"):
                self._publish(self.root / "wrong-provider-batch")

    def test_load_rejects_source_hash_revision_and_manifest_drift(self) -> None:
        destination = self.root / "published"
        with self._admission_passes():
            manifest_path = self._publish(destination)

        evidence = self.evidence_paths[0]
        evidence.write_text(
            evidence.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        with self._admission_passes():
            with self.assertRaises(ValueError):
                load_retained_regression_evidence_set(
                    manifest_path,
                    provider=self.provider,
                    obligations=self.obligations,
                    revision=REVISION,
                )

        evidence.write_text(
            json.dumps(self._evidence_payload_for_path(evidence)),
            encoding="utf-8",
        )
        self._refresh_exact_index_hash(evidence)
        wrong_revision = {**REVISION, "worktree_hash": "c" * 64}
        with self._admission_passes():
            with self.assertRaises(ValueError):
                load_retained_regression_evidence_set(
                    manifest_path,
                    provider=self.provider,
                    obligations=self.obligations,
                    revision=wrong_revision,
                )

        manifest_path.chmod(0o600)
        manifest = self._read(manifest_path)
        manifest["manifest_hash"] = "0" * 64
        self._write(manifest_path, manifest, mode=0o400)
        with self._admission_passes():
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                load_retained_regression_evidence_set(
                    manifest_path,
                    provider=self.provider,
                    obligations=self.obligations,
                    revision=REVISION,
                )

    def test_provider_and_open_result_hash_drift_fail_closed(self) -> None:
        provider = copy.deepcopy(self.provider)
        provider["provider_hash"] = "0" * 64
        with self._admission_passes():
            with self.assertRaises(ValueError):
                publish_retained_regression_evidence_set(
                    output_dir=self.root / "wrong-provider",
                    provider=provider,
                    obligations=self.obligations,
                    revision=REVISION,
                    exact_execution_index_path=self.exact_index,
                    open_batch_manifest_path=self.open_manifest,
                    open_batch_result_path=self.open_result,
                    evidence_paths=self.evidence_paths,
                )

        result = self._read(self.open_result)
        result["completed"] = 44
        self._write(self.open_result, result, mode=0o400)
        with self._admission_passes():
            with self.assertRaisesRegex(ValueError, "open batch result"):
                self._publish(self.root / "wrong-result")

    def test_load_rejects_undeclared_files_and_obligation_set_drift(self) -> None:
        destination = self.root / "published"
        with self._admission_passes():
            manifest_path = self._publish(destination)
        destination.chmod(0o700)
        extra = destination / "extra.json"
        extra.write_text("{}\n", encoding="utf-8")
        destination.chmod(0o500)
        with self._admission_passes():
            with self.assertRaisesRegex(ValueError, "undeclared"):
                load_retained_regression_evidence_set(
                    manifest_path,
                    provider=self.provider,
                    obligations=self.obligations,
                    revision=REVISION,
                )

        extra.unlink()
        drifted = copy.deepcopy(self.obligations)
        drifted[0]["variant"] = "isomorphic"
        with patch(
            "tests.agent_live.retained_regression_evidence_set."
            "validate_retained_regression_runner_provider"
        ), self._admission_passes():
            with self.assertRaises(ValueError):
                load_retained_regression_evidence_set(
                    manifest_path,
                    provider=self.provider,
                    obligations=drifted,
                    revision=REVISION,
                )

    def test_interrupted_publish_never_exposes_partial_destination(self) -> None:
        destination = self.root / "published"
        with patch(
            "tests.agent_live.retained_regression_evidence_set."
            "_rename_directory_noreplace",
            side_effect=OSError("interrupted"),
        ), self._admission_passes():
            with self.assertRaisesRegex(OSError, "interrupted"):
                self._publish(destination)
        self.assertFalse(destination.exists())
        staging = tuple(self.root.glob(".published.staging-*"))
        self.assertEqual(len(staging), 1)
        self.assertTrue((staging[0] / "manifest.json").is_file())

    def _publish(
        self,
        destination: Path,
        *,
        evidence_paths: list[Path] | None = None,
    ) -> Path:
        return publish_retained_regression_evidence_set(
            output_dir=destination,
            provider=self.provider,
            obligations=self.obligations,
            revision=REVISION,
            exact_execution_index_path=self.exact_index,
            open_batch_manifest_path=self.open_manifest,
            open_batch_result_path=self.open_result,
            evidence_paths=evidence_paths or self.evidence_paths,
        )

    def _admission_passes(self):
        return patch(
            "tests.agent_live.retained_regression_evidence_set."
            "admit_product_obligation_evidence",
            return_value={
                "complete": True,
                "passed": 60,
                "failed": 0,
                "external": 0,
                "not_run": 0,
            },
        )

    def _write_evidence(self) -> list[Path]:
        evidence_root = self.root / "source-evidence"
        evidence_root.mkdir()
        paths: list[Path] = []
        for index, obligation in enumerate(self.obligations, start=1):
            runner = (
                EXACT_RUNNER
                if obligation["variant"] == "exact"
                else OPEN_RUNNER
            )
            payload = {
                "obligation_id": obligation["obligation_id"],
                "evidence_id": hashlib.sha256(
                    f"evidence:{obligation['obligation_id']}".encode()
                ).hexdigest(),
                "evidence_hash": hashlib.sha256(
                    f"document:{obligation['obligation_id']}".encode()
                ).hexdigest(),
                "execution": {
                    "execution_id": f"execution-{index:02d}",
                    "runner": runner,
                },
            }
            path = evidence_root / f"{index:02d}.json"
            self._write(path, payload)
            paths.append(path.resolve())
        return paths

    def _write_exact_index(self) -> Path:
        exact_paths = [
            path
            for obligation, path in zip(
                self.obligations, self.evidence_paths, strict=True
            )
            if obligation["variant"] == "exact"
        ]
        exact_obligations = [
            obligation
            for obligation in self.obligations
            if obligation["variant"] == "exact"
        ]
        unsigned = {
            "schema_version": 1,
            "artifact_type": "retained_regression_exact_execution_index",
            "provider_hash": self.provider["provider_hash"],
            "revision_binding": REVISION,
            "selection": "all",
            "scheduled": 15,
            "completed": 15,
            "evidence": [
                {
                    "obligation_id": obligation["obligation_id"],
                    "path": str(path),
                    "sha256": self._sha256(path),
                }
                for obligation, path in zip(
                    exact_obligations, exact_paths, strict=True
                )
            ],
        }
        path = self.root / "exact-execution-index.json"
        self._write(
            path,
            {**unsigned, "index_hash": content_hash(unsigned)},
            mode=0o400,
        )
        return path.resolve()

    def _write_open_batch(self) -> tuple[Path, Path]:
        manifest_path = (self.root / "open-batch-manifest.json").resolve()
        open_rows = [
            (obligation, path)
            for obligation, path in zip(
                self.obligations, self.evidence_paths, strict=True
            )
            if obligation["variant"] != "exact"
        ]
        shards = []
        for index, (obligation, path) in enumerate(open_rows, start=1):
            payload = self._read(path)
            target = next(
                row
                for row in self.provider["targets"]
                if row["obligation_id"] == obligation["obligation_id"]
            )
            frozen = target["journey_definition"]["frozen_execution"]
            shards.append({
                "shard_id": f"shard-{index:02d}",
                "index": index,
                "obligation_id": obligation["obligation_id"],
                "execution_id": payload["execution"]["execution_id"],
                "lane": "journey",
                "schedule_id": frozen["schedule_id"],
                "seed": frozen["seed"],
                "subject_group": frozen["subject_group"],
            })
        obligation_set_hash = content_hash([
            {
                "obligation_id": row["obligation_id"],
                "schedule_id": row["schedule_id"],
                "seed": row["seed"],
                "subject_group": row["subject_group"],
            }
            for row in sorted(
                shards,
                key=lambda row: (row["obligation_id"], row["schedule_id"]),
            )
        ])
        unsigned = {
            "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
            "manifest_path": str(manifest_path),
            "revision": REVISION,
            "shard_count": 45,
            "worker_runtime": "linux",
            "expected_obligation_set_hash": obligation_set_hash,
            "shards": shards,
        }
        batch_id = content_hash(unsigned)
        manifest = {
            **unsigned,
            "batch_id": batch_id,
            "manifest_id": content_hash({**unsigned, "batch_id": batch_id}),
        }
        self._write(manifest_path, manifest, mode=0o400)

        result_unsigned = {
            "schema_version": BATCH_RESULT_SCHEMA_VERSION,
            "batch_id": manifest["batch_id"],
            "manifest_id": manifest["manifest_id"],
            "revision": REVISION,
            "execution_status": "discovery_complete",
            "release_status": "not_evaluated",
            "scheduled": 45,
            "started": 45,
            "completed": 45,
            "shards": [
                {
                    "shard_id": row["shard_id"],
                    "classification": "passed",
                }
                for row in shards
            ],
        }
        result_path = (self.root / "open-batch-result.json").resolve()
        self._write(
            result_path,
            {
                **result_unsigned,
                "index_id": content_hash(result_unsigned),
            },
            mode=0o400,
        )
        return manifest_path, result_path

    def _refresh_exact_index_hash(self, evidence_path: Path) -> None:
        index = self._read(self.exact_index)
        for row in index["evidence"]:
            if Path(row["path"]).resolve() == evidence_path.resolve():
                row["sha256"] = self._sha256(evidence_path)
        self._rehash(index, "index_hash")
        self._write(self.exact_index, index, mode=0o400)

    def _evidence_by_id(self, obligation_id: str) -> Path:
        for path in self.evidence_paths:
            if self._read(path)["obligation_id"] == obligation_id:
                return path
        raise AssertionError(obligation_id)

    def _evidence_payload_for_path(self, path: Path) -> dict:
        obligation_id = self._read(path)["obligation_id"]
        index = self.evidence_paths.index(path)
        obligation = self.obligations[index]
        return {
            "obligation_id": obligation_id,
            "evidence_id": hashlib.sha256(
                f"evidence:{obligation_id}".encode()
            ).hexdigest(),
            "evidence_hash": hashlib.sha256(
                f"document:{obligation_id}".encode()
            ).hexdigest(),
            "execution": {
                "execution_id": f"execution-{index + 1:02d}",
                "runner": (
                    EXACT_RUNNER
                    if obligation["variant"] == "exact"
                    else OPEN_RUNNER
                ),
            },
        }

    @staticmethod
    def _read(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write(path: Path, payload: dict, *, mode: int = 0o600) -> None:
        if path.exists():
            path.chmod(0o600)
        path.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        path.chmod(mode)

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _rehash(payload: dict, field: str) -> None:
        unsigned = {key: value for key, value in payload.items() if key != field}
        payload[field] = content_hash(unsigned)

    @staticmethod
    def _rehash_batch_manifest(payload: dict) -> None:
        unsigned = {
            key: value
            for key, value in payload.items()
            if key not in {"batch_id", "manifest_id"}
        }
        payload["batch_id"] = content_hash(unsigned)
        payload["manifest_id"] = content_hash({
            **unsigned,
            "batch_id": payload["batch_id"],
        })


if __name__ == "__main__":
    unittest.main()
