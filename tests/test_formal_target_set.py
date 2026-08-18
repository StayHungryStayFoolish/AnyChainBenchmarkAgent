from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.formal_target_set import (
    EMPTY_WORKTREE_HASH,
    FORMAL_EDGE_COUNT,
    FORMAL_JOURNEY_COUNT,
    build_formal_target_payloads,
    write_formal_target_set,
)
from tests.agent_live.batch_orchestrator import freeze_batch_manifest


def _revision(*, dirty: bool = False) -> dict[str, str]:
    return {
        "commit": "a" * 40,
        "worktree_hash": "b" * 64 if dirty else EMPTY_WORKTREE_HASH,
    }


def _edge(index: int, *, passed: bool = False) -> dict:
    group = f"group-{index % 12}"
    input_class = ("natural_language_option", "valid_literal", "action_only_transition")[index % 3]
    edge_type = "action_transition" if input_class == "action_only_transition" else "question_option"
    return {
        "edge_key": f"{group}::question-{index}::contract::{input_class}::option-{index}",
        "edge_type": edge_type,
        "group": group,
        "question_id": f"question-{index}",
        "input_class": input_class,
        "option_or_action": f"option-{index}",
        "expected_postcondition": {"value": index},
        "applicable": True,
        "executable_scenario_ids": [f"scenario-{index}"],
        "evidence": {
            "dynamic_dual_ai": {
                "required": True,
                "status": "passed" if passed else "not_run",
            }
        },
    }


def _ledger(count: int = 30, *, revision: dict[str, str] | None = None) -> dict:
    return {
        "revision": revision or _revision(),
        "edges": [_edge(index) for index in range(count)],
    }


class FormalTargetSetTest(unittest.TestCase):
    def test_build_is_deterministic_and_emits_exact_mixed_profile(self) -> None:
        revision = _revision()
        ledger = _ledger(revision=revision)
        first = build_formal_target_payloads(ledger, revision=revision, seed=71)
        second = build_formal_target_payloads(ledger, revision=revision, seed=71)
        self.assertEqual(first, second)
        self.assertEqual(len(first), FORMAL_EDGE_COUNT + FORMAL_JOURNEY_COUNT)
        self.assertTrue(all("targets" in row for row in first[:FORMAL_EDGE_COUNT]))
        self.assertTrue(all(row.get("lane") == "journey" for row in first[FORMAL_EDGE_COUNT:]))
        edge_keys = [row["targets"][0]["edge_key"] for row in first[:FORMAL_EDGE_COUNT]]
        self.assertEqual(len(edge_keys), len(set(edge_keys)))

    def test_build_rejects_mismatched_revision_and_insufficient_edges(self) -> None:
        with self.assertRaisesRegex(ValueError, "revision"):
            build_formal_target_payloads(
                _ledger(revision=_revision()),
                revision={"commit": "c" * 40, "worktree_hash": EMPTY_WORKTREE_HASH},
                seed=1,
            )
        with self.assertRaisesRegex(ValueError, "found 23"):
            build_formal_target_payloads(_ledger(23), revision=_revision(), seed=1)

    def test_passed_and_duplicate_edges_cannot_fill_the_quota(self) -> None:
        ledger = _ledger(24)
        ledger["edges"][0]["evidence"]["dynamic_dual_ai"]["status"] = "passed"
        ledger["edges"].append(dict(ledger["edges"][1]))
        with self.assertRaisesRegex(ValueError, "found 23"):
            build_formal_target_payloads(ledger, revision=_revision(), seed=2)

    def test_writer_rejects_dirty_and_stale_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "targets"
            with patch(
                "tests.agent_live.formal_target_set.repository_revision",
                return_value=_revision(dirty=True),
            ):
                with self.assertRaisesRegex(ValueError, "clean worktree"):
                    write_formal_target_set(
                        _ledger(), repo_root=tmpdir, output_dir=output, seed=4
                    )
            self.assertFalse(output.exists())
            stale = _ledger(revision=_revision())
            active = {"commit": "d" * 40, "worktree_hash": EMPTY_WORKTREE_HASH}
            with patch(
                "tests.agent_live.formal_target_set.repository_revision",
                return_value=active,
            ):
                with self.assertRaisesRegex(ValueError, "not bound"):
                    write_formal_target_set(
                        stale, repo_root=tmpdir, output_dir=output, seed=4
                    )
            self.assertFalse(output.exists())

    def test_writer_records_hashes_and_is_immutable(self) -> None:
        revision = _revision()
        ledger = _ledger(revision=revision)
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "tests.agent_live.formal_target_set.repository_revision",
            return_value=revision,
        ):
            output = Path(tmpdir) / "targets"
            manifest_path = write_formal_target_set(
                ledger, repo_root=tmpdir, output_dir=output, seed=9
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_ledger_hash"], content_hash(ledger))
            self.assertEqual(manifest["lane_counts"], {"edge": 24, "journey": 8})
            self.assertEqual(len(manifest["targets"]), 32)
            for row in manifest["targets"]:
                payload = (output / row["path"]).read_bytes()
                import hashlib
                self.assertEqual(row["sha256"], hashlib.sha256(payload).hexdigest())
            with self.assertRaises(FileExistsError):
                write_formal_target_set(
                    ledger, repo_root=tmpdir, output_dir=output, seed=9
                )

    def test_generated_set_freezes_as_formal_linux_manifest(self) -> None:
        revision = _revision()
        ledger = _ledger(revision=revision)
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "tests.agent_live.formal_target_set.repository_revision",
            return_value=revision,
        ):
            root = Path(tmpdir)
            targets = root / "targets"
            write_formal_target_set(
                ledger, repo_root=root, output_dir=targets, seed=17
            )
            with (
                patch(
                    "tests.agent_live.batch_orchestrator.repository_revision",
                    return_value=revision,
                ),
                patch(
                    "tests.agent_live.batch_orchestrator.build_ledger",
                    return_value=ledger,
                ),
            ):
                manifest = freeze_batch_manifest(
                    repo_root=root,
                    targets_dir=targets,
                    manifest_path=root / "manifest.json",
                    runtime_base=root / "runtime",
                    expected_revision=revision,
                    worker_runtime="linux",
                    formal_profile=True,
                )
            self.assertEqual(manifest.shard_count, 32)
            self.assertEqual(sum(row.lane == "edge" for row in manifest.shards), 24)
            self.assertEqual(sum(row.lane == "journey" for row in manifest.shards), 8)


if __name__ == "__main__":
    unittest.main()
